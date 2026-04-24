"""
TurboQuant quantizers — Algorithm 1 (MSE) and Algorithm 2 (inner product).

These operate on tensors of shape (..., d) where d is the embedding dimension
(typically head_dim = 128 for modern LLMs).
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, NamedTuple

from turboquant.codebook import get_codebook_tensors
from turboquant.rotation import (
    generate_rotation_matrix,
    generate_qjl_matrix,
    rotate_forward,
    rotate_backward,
)


class MSEQuantized(NamedTuple):
    """Output of TurboQuant MSE quantization."""
    indices: torch.Tensor       # (..., packed_len) uint8 bit-packed indices
    norms: torch.Tensor         # (...,) original L2 norms
    bits: int                   # number of bits per index (for unpacking)


class ProdQuantized(NamedTuple):
    """Output of TurboQuant inner-product quantization (regular channels only)."""
    mse_indices: torch.Tensor      # (..., packed_len) uint8 bit-packed MSE indices
    qjl_signs: torch.Tensor        # (..., packed_len) uint8 packed sign bits
    residual_norms: torch.Tensor   # (...,) L2 norms of residual vectors
    norms: torch.Tensor            # (...,) original L2 norms
    mse_bits: int                  # bits per MSE index (for unpacking)


class HybridQuantized(NamedTuple):
    """
    Hybrid quantization output: TurboQuant for regular channels,
    full-precision for outlier channels.
    
    outlier_data stores the actual fp16 values of outlier channels.
    regular_q stores TurboQuant-compressed regular channels only.
    regular_idx is a 1D int tensor of shape (d_regular,) mapping 
    regular channel positions back to the original d-dimensional space.
    outlier_idx is a 1D int tensor of shape (d_outlier,) for outlier positions.
    """
    regular_q: ProdQuantized           # compressed regular channels
    outlier_data: torch.Tensor         # (..., n_tokens, d_outlier) full precision
    regular_idx: torch.Tensor          # (d_regular,) original column indices
    outlier_idx: torch.Tensor          # (d_outlier,) original column indices
    d_total: int                       # original head_dim


def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Bit-pack integer indices (0..2^bits-1) into uint8 bytes.

    For bits=1: 8 values per byte
    For bits=2: 4 values per byte
    For bits=3: stored as 4-bit (2 per byte) for simplicity
    For bits=4: 2 values per byte
    """
    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]

    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4  # round up to 4-bit packing
    else:
        # Just store as uint8
        return indices.to(torch.uint8)

    # Pad to multiple of vals_per_byte
    padded_d = ((d + vals_per_byte - 1) // vals_per_byte) * vals_per_byte
    if padded_d > d:
        indices = F.pad(indices.to(torch.uint8), (0, padded_d - d), value=0)

    reshaped = indices.to(torch.uint8).reshape(*batch_shape, -1, vals_per_byte)
    shifts = torch.arange(vals_per_byte, device=indices.device, dtype=torch.uint8) * bits
    packed = (reshaped << shifts).sum(dim=-1, dtype=torch.uint8)
    return packed


def _unpack_indices(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """Unpack bit-packed indices back to integer tensor."""
    batch_shape = packed.shape[:-1]

    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return packed.long()

    mask = (1 << bits) - 1
    shifts = torch.arange(vals_per_byte, device=packed.device, dtype=torch.uint8) * bits
    unpacked = ((packed.unsqueeze(-1) >> shifts) & mask)
    unpacked = unpacked.reshape(*batch_shape, -1)
    return unpacked[..., :d].long()


class TurboQuantMSE(torch.nn.Module):
    """
    TurboQuant optimized for MSE (Algorithm 1).

    Quantize: y = Π·(x/||x||), then find nearest centroid per coordinate.
    Dequantize: look up centroids, rotate back, rescale by ||x||.
    """

    def __init__(
        self,
        dim: int,
        bits: int = 3,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        ip_optimized: bool = False,  # NEW: use IP-optimized codebook
        rotation_type: str = "dense",  # NEW: "dense" or "hadamard"
    ):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.n_clusters = 2**bits
        self.ip_optimized = ip_optimized  # Store flag
        self.rotation_type = rotation_type  # Store rotation type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Precompute rotation matrix
        self.register_buffer(
            "Pi", generate_rotation_matrix(dim, self.device, dtype, seed=seed, rotation_type=rotation_type)
        )

        # Precompute codebook (optionally IP-optimized)
        centroids, boundaries = get_codebook_tensors(dim, bits, self.device, dtype, 
                                                      ip_optimized=ip_optimized)
        self.register_buffer("centroids", centroids)      # (2^b,)
        self.register_buffer("boundaries", boundaries)    # (2^b + 1,)

        # Precompute interior boundaries for fast searchsorted
        # boundaries[1:-1] are the decision boundaries between clusters
        self.register_buffer("decision_boundaries", boundaries[1:-1].contiguous())

    def quantize(self, x: torch.Tensor) -> MSEQuantized:
        """
        Quantize vectors x of shape (..., d).

        Returns MSEQuantized with bit-packed indices and norms.
        """
        # Store norms for rescaling
        norms = x.norm(dim=-1, keepdim=False)
        # Normalize to unit sphere
        x_unit = x / (norms.unsqueeze(-1) + 1e-10)

        # Apply random rotation
        y = rotate_forward(x_unit.float(), self.Pi)  # (..., d)

        # Quantize each coordinate: find bucket via searchsorted
        indices = torch.searchsorted(self.decision_boundaries, y.contiguous())

        # Bit-pack the indices
        packed = _pack_indices(indices, self.bits)

        return MSEQuantized(indices=packed, norms=norms, bits=self.bits)

    def dequantize(self, q: MSEQuantized) -> torch.Tensor:
        """Reconstruct vectors from quantized representation."""
        # Unpack indices
        indices = _unpack_indices(q.indices, q.bits, self.dim)

        # Look up centroids
        y_hat = self.centroids[indices]  # (..., d)

        # Rotate back
        x_hat = rotate_backward(y_hat, self.Pi)  # (..., d)

        # Rescale by original norms
        x_hat = x_hat * q.norms.unsqueeze(-1)

        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and immediately dequantize (for testing)."""
        return self.dequantize(self.quantize(x))


class TurboQuantProd(torch.nn.Module):
    """
    TurboQuant optimized for inner products (Algorithm 2).

    Two-stage:
      1. Apply TurboQuant_MSE at (b-1) bits → get residual r = x - x̃
      2. Apply QJL to residual: sign(S·r) → 1 bit per coordinate
      3. Store ||r||₂ for rescaling

    The dequantized inner product estimate is:
      <y, x̃_mse> + ||r|| * sqrt(π/2)/d * <S^T · qjl_signs, y>
    which is unbiased: E[estimate] = <y, x>
    
    Outlier-aware extension:
      - Detect high-variance channels before rotation
      - Outliers: FP16 pass-through (outlier_bits>=8) or higher-bit quantization
      - Regular: Hadamard rotation + standard (bits-1)-bit MSE quantization + QJL
    """

    def __init__(
        self,
        dim: int,
        bits: int = 3,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        ip_optimized: bool = False,  # use IP-optimized codebook for MSE stage
        rotation_type: str = "dense",  # "dense" or "hadamard"
        outlier_ratio: float = 0.08,   # Fraction of channels to treat as outliers
        outlier_bits: float = 16.0,    # Bit-width for outliers: 16=FP16 pass-through
    ):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.ip_optimized = ip_optimized  # Store flag
        self.rotation_type = rotation_type  # Store rotation type
        self.outlier_ratio = outlier_ratio  # Store outlier ratio
        self.outlier_bits = outlier_bits    # Store outlier bits
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert bits >= 2, "Inner product TurboQuant requires at least 2 bits (1 for MSE + 1 for QJL)"

        # Stage 1: MSE quantizer at (b-1) bits (optionally IP-optimized)
        self.mse_quantizer = TurboQuantMSE(
            dim=dim, bits=bits - 1, device=self.device, dtype=dtype, seed=seed,
            ip_optimized=ip_optimized, rotation_type=rotation_type  # PASS DOWN
        )

        # Stage 2: QJL projection matrix S ∈ R^{d×d}
        self.register_buffer(
            "S", generate_qjl_matrix(dim, self.device, dtype, seed=seed + 1000)
        )

        # QJL dequantization constant
        self.qjl_scale = math.sqrt(math.pi / 2.0) / dim

    def _pack_qjl_signs(self, projected: torch.Tensor) -> torch.Tensor:
        """Pack sign bits into uint8 (8 signs per byte)."""
        signs = (projected > 0).to(torch.uint8)
        d = signs.shape[-1]
        if d % 8 != 0:
            signs = F.pad(signs, (0, 8 - d % 8), value=0)
        signs_reshaped = signs.reshape(*signs.shape[:-1], -1, 8)
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=signs.device, dtype=torch.uint8)
        return (signs_reshaped * powers).sum(dim=-1, dtype=torch.uint8)

    def _unpack_qjl_signs(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack sign bits from uint8 to float {-1, +1}."""
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=packed.device, dtype=torch.uint8)
        unpacked = ((packed.unsqueeze(-1) & powers) > 0).float()
        signs = unpacked.reshape(*packed.shape[:-1], -1)[..., :self.dim]
        return 2.0 * signs - 1.0

    def quantize(self, x: torch.Tensor) -> ProdQuantized:
        """
        Standard TurboQuant inner-product quantization (no outlier handling).
        
        Note: The outlier-aware logic has been moved to TurboQuantHybrid.
        This class now handles only regular channels for the hybrid approach.
        """
        # Store norms for rescaling
        norms = x.norm(dim=-1, keepdim=False)
        # Normalize to unit sphere
        x_unit = x / (norms.unsqueeze(-1) + 1e-10)

        # Stage 1: MSE quantize at (b-1) bits
        mse_q = self.mse_quantizer.quantize(x_unit)
        
        # Dequantize MSE to get residual
        x_mse = self.mse_quantizer.dequantize(mse_q)
        residual = x - x_mse
        residual_norms = residual.norm(dim=-1)
        
        # Stage 2: QJL on residual
        projected = torch.matmul(residual.float(), self.S.T)
        packed_signs = self._pack_qjl_signs(projected)
        
        return ProdQuantized(
            mse_indices=mse_q.indices,
            qjl_signs=packed_signs,
            residual_norms=residual_norms,
            norms=norms,
            mse_bits=mse_q.bits,
        )

    def dequantize(self, q: ProdQuantized) -> torch.Tensor:
        """Reconstruct vectors from quantized representation."""
        # Stage 1: MSE dequantize
        mse_q = MSEQuantized(indices=q.mse_indices, norms=q.norms, bits=q.mse_bits)
        x_mse = self.mse_quantizer.dequantize(mse_q)

        # Stage 2: QJL dequantize
        signs = self._unpack_qjl_signs(q.qjl_signs)

        # x̃_qjl = sqrt(π/2)/d * ||r|| * S^T @ signs
        x_qjl = torch.matmul(signs, self.S)
        x_qjl = x_qjl * (self.qjl_scale * q.residual_norms.unsqueeze(-1))

        return x_mse + x_qjl

    def attention_score(
        self,
        query: torch.Tensor,
        quantized_key: ProdQuantized,
    ) -> torch.Tensor:
        """
        Compute attention scores <query, key> using quantized keys.

        Args:
            query: (..., n_q, d)  — the query vectors
            quantized_key: ProdQuantized with shapes (..., n_k, ...) etc.

        Returns:
            scores: (..., n_q, n_k) — the attention logits
        """
        # Stage 1: MSE contribution
        mse_q = MSEQuantized(indices=quantized_key.mse_indices, norms=quantized_key.norms,
                             bits=quantized_key.mse_bits)
        k_mse = self.mse_quantizer.dequantize(mse_q)
        scores_mse = torch.matmul(query.float(), k_mse.float().transpose(-2, -1))

        # Stage 2: QJL contribution — asymmetric estimator
        q_sketched = torch.matmul(query.float(), self.S.T)
        signs = self._unpack_qjl_signs(quantized_key.qjl_signs)

        scores_qjl = torch.matmul(q_sketched, signs.transpose(-2, -1))
        scores_qjl = scores_qjl * (self.qjl_scale * quantized_key.residual_norms.unsqueeze(-2))

        return scores_mse + scores_qjl.to(scores_mse.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and immediately dequantize (for testing)."""
        return self.dequantize(self.quantize(x))


class OutlierDetector:
    """
    Identifies outlier channels from a running estimate of per-channel variance.
    Call update() on the first N tokens during warmup, then call finalize()
    to lock in the outlier/regular split. After finalize(), the split is fixed.
    """
    
    def __init__(self, d: int, outlier_ratio: float = 0.08, warmup_tokens: int = 256):
        self.d = d
        self.outlier_ratio = outlier_ratio
        self.warmup_tokens = warmup_tokens
        self.n_outliers = max(1, int(d * outlier_ratio))
        
        self._sum_sq = torch.zeros(d)   # running sum of x^2 per channel
        self._count = 0
        self._finalized = False
        
        # Set after finalize()
        self.outlier_idx: torch.Tensor = None   # (n_outliers,)
        self.regular_idx: torch.Tensor = None   # (d - n_outliers,)
    
    def update(self, x: torch.Tensor):
        """
        x: (..., d) — raw key vectors (NOT normalized).
        Accumulates per-channel variance estimate.
        Called only during warmup.
        Note: x can have shape (batch, tokens, d) or (1, H, tokens, d).
        We count actual tokens processed, not just batch calls.
        """
        if self._finalized:
            return
        # Flatten batch dims, keep channel dim
        x_flat = x.reshape(-1, self.d).float()
        n_tokens_in_batch = x_flat.shape[0]
        self._sum_sq = self._sum_sq.to(x_flat.device)
        self._sum_sq += (x_flat ** 2).mean(dim=0) * n_tokens_in_batch
        self._count += n_tokens_in_batch
        
        if self._count >= self.warmup_tokens:
            self.finalize()
    
    def finalize(self, device=None):
        """Lock in the outlier/regular split based on accumulated variance."""
        if self._finalized:
            return
        dev = device or self._sum_sq.device
        variance = self._sum_sq / max(self._count, 1)
        # Top-k channels by variance are outliers
        _, top_idx = torch.topk(variance, self.n_outliers)
        all_idx = torch.arange(self.d, device=dev)
        mask = torch.ones(self.d, dtype=torch.bool, device=dev)
        mask[top_idx] = False
        self.outlier_idx = top_idx.sort().values.to(dev)
        self.regular_idx = all_idx[mask].to(dev)
        self._finalized = True
    
    @property
    def is_ready(self) -> bool:
        return self._finalized


class TurboQuantHybrid(torch.nn.Module):
    """
    Hybrid quantizer: TurboQuant (Hadamard rotation) for regular channels,
    full fp16 pass-through for outlier channels.
    
    This directly addresses the 'outlier smearing' problem where Hadamard
    spreads high-variance outlier channels across all coordinates, degrading
    the distributional alignment that makes Hadamard better than dense QR.
    
    Usage:
        quantizer = TurboQuantHybrid(dim=128, bits=3, outlier_ratio=0.08)
        # During warmup (first ~256 tokens):
        quantizer.update_detector(raw_keys)
        # After warmup, quantizer.is_ready == True
        q = quantizer.quantize(keys)
        keys_recon = quantizer.dequantize(q)
    """
    
    def __init__(
        self,
        dim: int,
        bits: int = 3,
        outlier_ratio: float = 0.08,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        warmup_tokens: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.outlier_ratio = outlier_ratio
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.detector = OutlierDetector(dim, outlier_ratio, warmup_tokens)
        
        # The regular-channel quantizer is initialized with d_regular after warmup.
        # We defer its creation until finalize() so we know the exact d_regular.
        self._regular_quantizer: TurboQuantProd = None
        self._seed = seed
        self._dtype = dtype
    
    def update_detector(self, raw_keys: torch.Tensor):
        """
        Feed raw (unnormalized) key vectors to calibrate the outlier detector.
        raw_keys: (..., d)
        After warmup_tokens calls, the detector finalizes automatically.
        """
        self.detector.update(raw_keys)
        if self.detector.is_ready and self._regular_quantizer is None:
            self._build_regular_quantizer()
    
    def _build_regular_quantizer(self):
        """Create the TurboQuantProd for regular channels only."""
        d_reg = len(self.detector.regular_idx)
        self._regular_quantizer = TurboQuantProd(
            dim=d_reg,
            bits=self.bits,
            device=self.device,
            dtype=self._dtype,
            seed=self._seed,
            rotation_type="hadamard",   # Always Hadamard for regular channels
        )
    
    @property
    def is_ready(self) -> bool:
        return self.detector.is_ready and self._regular_quantizer is not None
    
    def quantize(self, x: torch.Tensor) -> HybridQuantized:
        """
        x: (..., n_tokens, d)
        Returns HybridQuantized with regular channels TQ-compressed and
        outlier channels stored as fp16.
        """
        assert self.is_ready, "Detector not finalized. Call update_detector() first."
        
        reg_idx = self.detector.regular_idx
        out_idx = self.detector.outlier_idx
        
        # Split channels
        x_regular = x[..., reg_idx]     # (..., n_tokens, d_regular)
        x_outlier = x[..., out_idx]     # (..., n_tokens, d_outlier)
        
        # Quantize regular channels with TurboQuant
        regular_q = self._regular_quantizer.quantize(x_regular)
        
        # Store outliers in fp16
        outlier_data = x_outlier.to(torch.float16)
        
        return HybridQuantized(
            regular_q=regular_q,
            outlier_data=outlier_data,
            regular_idx=reg_idx,
            outlier_idx=out_idx,
            d_total=self.dim,
        )
    
    def dequantize(self, q: HybridQuantized) -> torch.Tensor:
        """Reconstruct full d-dimensional vectors."""
        x_regular = self._regular_quantizer.dequantize(q.regular_q)
        x_outlier = q.outlier_data.to(x_regular.dtype)
        
        # Reconstruct in original channel order
        batch_shape = x_regular.shape[:-1]
        x_full = torch.empty(*batch_shape, q.d_total, 
                             dtype=x_regular.dtype, device=x_regular.device)
        x_full[..., q.regular_idx] = x_regular
        x_full[..., q.outlier_idx] = x_outlier
        return x_full
    
    def attention_score(
        self, 
        query: torch.Tensor, 
        quantized_key: HybridQuantized,
    ) -> torch.Tensor:
        """
        Compute attention scores <query, key> with hybrid quantized keys.
        
        This is the critical path. It splits the score into two parts:
          score = <q_reg, k_reg_recon> + <q_out, k_out>
        where k_reg_recon uses TurboQuant's unbiased estimator and
        k_out is just a direct fp16 dot product.
        
        query: (T_q, Q, d) or (T_q, H_kv, G, d) — typically (batch, num_query_heads, d)
               For decode: T_q=1, for prefill: T_q=num_prefill_tokens
        quantized_key: HybridQuantized with shapes (..., N, ...) where ... can be batch dims
                       Common patterns:
                       - Direct quantization: (batch, H_kv, N, d_xxx)
                       - After flatten: (H_kv, N, d_xxx)  [no leading batch dim]
        Returns: (T_q, ..., N) — scores per query token and key position
        """
        reg_idx = quantized_key.regular_idx
        out_idx = quantized_key.outlier_idx
        
        # Handle GQA: query can be (T_q, Q, d) or already split as (T_q, H_kv, G, d)
        if query.dim() == 3:
            # query: (T_q, Q, d) where Q = H_kv * gqa_ratio
            T_q, Q, d = query.shape
            # Reshape to (T_q, H_kv, G, d)
            # We need to infer H_kv from the quantized_key shapes
            # outlier_data shape: (..., N, d_out) - get H_kv from second-to-last dim if >2D, or assume 1 head
            if quantized_key.outlier_data.dim() == 3:
                H_kv = quantized_key.outlier_data.shape[0]
            else:
                H_kv = Q // (Q // 4)  # Assume GQA ratio of 8 by default, fallback
            G = Q // H_kv
            query_4d = query.view(T_q, H_kv, G, d)
        elif query.dim() == 4:
            # query: (T_q, H_kv, G, d) — already split for GQA
            T_q, H_kv, G, d = query.shape
            query_4d = query
        else:
            raise ValueError(f"Unexpected query shape: {query.shape}")
        
        # Average over G dimension to get per-KV-head queries
        q_regular = query_4d[..., reg_idx].mean(dim=2)   # (T_q, H_kv, d_regular) or (T_q, ..., d_regular)
        q_outlier = query_4d[..., out_idx].mean(dim=2)   # (T_q, H_kv, d_outlier) or (T_q, ..., d_outlier)
        
        # Dequantize regular keys
        k_regular = self._regular_quantizer.dequantize(quantized_key.regular_q)  # (..., N, d_reg)
        outlier_data = quantized_key.outlier_data  # (..., N, d_out)
        
        # Determine the batch shape of keys (everything except last two dims: N and d)
        key_batch_shape = k_regular.shape[:-2]
        N = k_regular.shape[-2]
        
        # Expand query to match key batch dimensions if needed
        # For simplicity, handle common cases:
        if len(key_batch_shape) == 0:
            # Keys are (N, d), no batch dims - shouldn't happen in practice
            scores_regular = torch.einsum("thd,nd->thn", q_regular.float(), k_regular.float())
            scores_outlier = torch.einsum("thd,nd->thn", q_outlier.float(), outlier_data.float())
            # Combine H_kv and G back to Q dimension
            scores = (scores_regular + scores_outlier).reshape(T_q, -1, N)
        elif len(key_batch_shape) == 1:
            # Keys are (H_kv, N, d)
            scores_regular = torch.einsum("thd,hnd->thn", q_regular.float(), k_regular.float())
            scores_outlier = torch.einsum("thd,hnd->thn", q_outlier.float(), outlier_data.float())
            # Reshape from (T_q, H_kv, N) to (T_q, Q, N) where Q = H_kv * G
            # We need to expand along G dimension
            scores = (scores_regular + scores_outlier).unsqueeze(2).expand(T_q, H_kv, G, N)
            scores = scores.reshape(T_q, H_kv * G, N)
        elif len(key_batch_shape) == 2:
            # Keys are (batch, H_kv, N, d) or (1, H_kv, N, d)
            if key_batch_shape[0] == 1:
                # Squeeze the leading 1 dim
                k_regular = k_regular.squeeze(0)  # (H_kv, N, d)
                outlier_data = outlier_data.squeeze(0)  # (H_kv, N, d_out)
                scores_regular = torch.einsum("thd,hnd->thn", q_regular.float(), k_regular.float())
                scores_outlier = torch.einsum("thd,hnd->thn", q_outlier.float(), outlier_data.float())
                # Reshape from (T_q, H_kv, N) to (T_q, Q, N)
                scores = (scores_regular + scores_outlier).unsqueeze(2).expand(T_q, H_kv, G, N)
                scores = scores.reshape(T_q, H_kv * G, N)
            else:
                # General case: use broadcasting
                # q: (T_q, H_kv, d), k: (B, H_kv, N, d)
                # Result: (T_q, B, H_kv, N)
                scores_regular = torch.einsum("thd,bhnd->tbhn", q_regular.float(), k_regular.float())
                scores_outlier = torch.einsum("thd,bhnd->tbhn", q_outlier.float(), outlier_data.float())
                scores = scores_regular + scores_outlier
                # Reshape to (T_q, B*H_kv*G, N) or keep as is depending on usage
                # For now, keep as (T_q, B, H_kv, N) - caller should handle
        else:
            raise ValueError(f"Unsupported key batch shape: {key_batch_shape}")
        
        return scores
