"""
TurboQuant score module — attention computation over compressed + exact segments.

Handles the read path:
  - Compute attention scores over compressed historical KV (via Triton or PyTorch fallback)
  - Compute attention scores over exact recent buffer (via standard matmul / SDPA)
  - Merge logits and weighted values from both segments

Design rule: compressed path is only invoked when history is large enough
to justify it (>= 16 tokens).
"""

from __future__ import annotations

import math
import logging
import torch
import torch.nn.functional as F

from turboquant.store import FlatCache, CompressedKVStore
from turboquant.kv_cache import dequantize_values
from turboquant.quantizer import TurboQuantProd, HybridQuantized

logger = logging.getLogger("turboquant.score")

MIN_HISTORY_FOR_TQ = 16


def compute_hybrid_attention(
    query: torch.Tensor,
    store: CompressedKVStore,
    recent_k: Optional[torch.Tensor],
    recent_v: Optional[torch.Tensor],
    num_query_heads: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute attention output combining compressed history and exact recent buffer.

    Args:
        query: (num_tokens, num_query_heads, head_dim) — typically num_tokens=1 for decode
        store: compressed KV store with historical tokens
        recent_k: (recent_len, num_kv_heads, head_dim) or None
        recent_v: (recent_len, num_kv_heads, head_dim) or None
        num_query_heads: total query heads (for GQA expansion)
        scale: attention scale factor (default: 1/sqrt(head_dim))

    Returns:
        output: (num_tokens, num_query_heads, head_dim)
    """
    head_dim = store.head_dim
    num_kv_heads = store.num_kv_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    flat = store.get_flat_cache()
    has_history = flat is not None and flat.num_tokens >= MIN_HISTORY_FOR_TQ
    has_recent = recent_k is not None and recent_k.shape[0] > 0

    if not has_history and not has_recent:
        return torch.zeros(
            query.shape[0], num_query_heads, head_dim,
            device=query.device, dtype=query.dtype,
        )

    gqa_ratio = num_query_heads // num_kv_heads

    if has_history and not has_recent:
        return _attend_compressed_only(
            query, flat, store.quantizer, gqa_ratio, num_kv_heads, scale
        )

    if not has_history and has_recent:
        return _attend_exact_only(
            query, recent_k, recent_v, gqa_ratio, num_kv_heads, scale
        )

    # Both segments present — merge via log-sum-exp trick
    return _attend_hybrid(
        query, flat, store.quantizer, recent_k, recent_v,
        gqa_ratio, num_kv_heads, head_dim, scale,
    )


def _attend_compressed_only(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer,  # Can be TurboQuantProd or TurboQuantHybrid
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Attention over compressed history only (PyTorch path)."""
    from turboquant.quantizer import HybridQuantized
    
    if isinstance(flat.prod_q, HybridQuantized):
        # Hybrid path: use quantizer's attention_score method which handles the split
        # attention_score returns (T, Q, N) but we need to apply softmax per-query-head
        scores = quantizer.attention_score(query, flat.prod_q)  # (T, Q, N_hist)
        v_dequant = dequantize_values(flat.value_q, 32)  # (H_kv, N_hist, D)
        
        # Apply softmax and compute output with GQA
        # scores: (T, Q, N_hist), v_dequant: (H_kv, N_hist, D)
        # Need to expand v_dequant to match Q heads
        T, Q, N_hist = scores.shape
        D = v_dequant.shape[-1]
        
        # Reshape scores to (T, H_kv, G, N_hist)
        scores_gqa = scores.view(T, num_kv_heads, gqa_ratio, N_hist)
        weights_gqa = F.softmax(scores_gqa * scale, dim=-1)  # (T, H_kv, G, N_hist)
        
        # Compute output: sum over N_hist
        # out: (T, H_kv, G, D) = weights[T,H_kv,G,N] @ v[H_kv,N,D]
        out = torch.einsum("thgn,hnd->thgd", weights_gqa, v_dequant.float())
        return out.reshape(T, Q, D).to(query.dtype)
    else:
        # Original path unchanged
        k_dequant = quantizer.dequantize(flat.prod_q)
        v_dequant = dequantize_values(flat.value_q, 32)
        return _matmul_attend(query, k_dequant, v_dequant, gqa_ratio, num_kv_heads, scale)


def _attend_exact_only(
    query: torch.Tensor,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Attention over exact recent buffer only."""
    return _matmul_attend(
        query, recent_k.transpose(0, 1), recent_v.transpose(0, 1),
        gqa_ratio, num_kv_heads, scale,
    )


def _attend_hybrid(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer,  # Can be TurboQuantProd or TurboQuantHybrid
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
) -> torch.Tensor:
    """Merge compressed history + exact recent via concatenated attention."""
    from turboquant.quantizer import HybridQuantized
    
    if isinstance(flat.prod_q, HybridQuantized):
        # Hybrid path: use quantizer's attention_score for history
        # Reshape query to (T, H_kv, G, D) for proper GQA handling
        T = query.shape[0]
        query_4d = query.float().view(T, num_kv_heads, gqa_ratio, head_dim)
        
        scores_hist = quantizer.attention_score(query_4d, flat.prod_q)  # (T, H_kv, N_hist)
        v_hist = dequantize_values(flat.value_q, 32)  # (H_kv, N_hist, D)
        
        # Recent keys/values are already in exact form
        k_recent = recent_k.transpose(0, 1)   # (H_kv, N_recent, D)
        v_recent = recent_v.transpose(0, 1)
        
        # Compute scores for recent part
        # query_4d: (T, H_kv, G, D)
        # k_recent_float: (H_kv, N_recent, D)
        k_recent_float = k_recent.float()
        # scores: (T, H_kv, G, N_recent) = <q[T,H_kv,G,D], k[H_kv,N_recent,D]>
        scores_recent = torch.einsum("thgd,hnd->thgn", query_4d, k_recent_float) * scale
        
        # scores_hist: (T, H_kv, N_hist) -> expand to (T, H_kv, G, N_hist)
        scores_hist_expanded = scores_hist.unsqueeze(2).expand(-1, -1, gqa_ratio, -1)
        
        # Concatenate scores and apply softmax
        scores_all = torch.cat([scores_hist_expanded, scores_recent], dim=-1)
        weights_all = F.softmax(scores_all, dim=-1)
        
        # Split weights and compute output
        n_hist = scores_hist.shape[-1]
        weights_hist = weights_all[:, :, :, :n_hist].reshape(T, num_kv_heads * gqa_ratio, -1)
        weights_recent = weights_all[:, :, :, n_hist:].reshape(T, num_kv_heads * gqa_ratio, -1)
        
        out_hist = torch.matmul(weights_hist, v_hist.float())
        out_recent = torch.matmul(weights_recent, v_recent.float())
        
        return (out_hist + out_recent).to(query.dtype)
    else:
        # Original path unchanged
        k_hist = quantizer.dequantize(flat.prod_q)  # (H_kv, N_hist, D)
        v_hist = dequantize_values(flat.value_q, 32)

        k_recent = recent_k.transpose(0, 1)   # (H_kv, N_recent, D)
        v_recent = recent_v.transpose(0, 1)

        k_all = torch.cat([k_hist.float(), k_recent.float()], dim=1)
        v_all = torch.cat([v_hist.float(), v_recent.float()], dim=1)

        return _matmul_attend(query, k_all, v_all, gqa_ratio, num_kv_heads, scale)


def _matmul_attend(
    query: torch.Tensor,
    kv_keys: torch.Tensor,
    kv_values: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Standard matmul attention with GQA support.

    query: (T, Q_heads, D)
    kv_keys: (H_kv, N, D)
    kv_values: (H_kv, N, D)

    Returns: (T, Q_heads, D)
    """
    T, Q, D = query.shape
    H_kv = num_kv_heads
    if Q != H_kv * gqa_ratio:
        raise ValueError(
            f"Incompatible GQA shapes: Q={Q}, H_kv={H_kv}, gqa_ratio={gqa_ratio}"
        )

    # Avoid repeat_interleave(Q/H) on KV tensors to keep memory bounded at long context.
    # q: (T, Q, D) -> (H_kv, G, T, D)
    q = query.float().view(T, H_kv, gqa_ratio, D).permute(1, 2, 0, 3)
    k = kv_keys.float().unsqueeze(1)   # (H_kv, 1, N, D) broadcast over G
    v = kv_values.float().unsqueeze(1) # (H_kv, 1, N, D) broadcast over G

    # scores: (H_kv, G, T, N)
    scores = torch.einsum("hgtd,hgnd->hgtn", q, k) * scale
    weights = F.softmax(scores, dim=-1)
    out = torch.einsum("hgtn,hgnd->hgtd", weights, v)

    # Back to (T, Q, D)
    return out.permute(2, 0, 1, 3).reshape(T, Q, D).to(query.dtype)
