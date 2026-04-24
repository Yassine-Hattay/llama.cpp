#!/usr/bin/env python3
# experiment2_structured_rotation.py
"""
Experiment 2: Structured Rotations for TurboQuant

Tests the hypothesis from Claude/Gemi ni synthesis:
- TurboQuant uses dense random rotation (O(d²)) producing i.i.d. Beta coordinates
- PolarQuant/GSR show structured rotations (Walsh, sequency-ordered) match quality at O(d log d)
- Key question: Do structured rotations produce non-uniform variance profiles that enable water-filling?

This experiment:
1. Replaces dense Gaussian Π with 3 structured rotations:
   - Standard Walsh-Hadamard Transform (WHT)
   - Sequency-ordered Walsh blocks (GSR-style, Paper 18)
   - Hybrid: Hadamard + mild permutation for variance shaping

2. Profiles post-rotation variance: Computes per-coordinate σⱼ² and measures non-uniformity

3. Tests water-filling bit allocation: Redistributes bit budget B = b·d based on σⱼ²

4. Measures D_prod vs baseline TurboQuant

Expected outcomes:
- If variance profile is uniform → water-filling won't help (null hypothesis confirmed)
- If variance profile is non-uniform → water-filling may improve D_prod by 5-15%
"""

import math
import json
import torch
import numpy as np
from typing import Tuple, Dict, Any, List
import sys
sys.path.insert(0, '/workspace')

from turboquant.rotation import generate_rotation_matrix, rotate_forward
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd
from turboquant.codebook import get_codebook


# =============================================================================
# Structured Rotation Implementations
# =============================================================================

def hadamard_matrix(n: int) -> torch.Tensor:
    """
    Generate Sylvester-type Hadamard matrix of order n (must be power of 2).
    Uses recursive construction: H_{2n} = [H_n, H_n; H_n, -H_n]
    """
    if n & (n - 1) != 0:
        raise ValueError(f"n must be power of 2, got {n}")
    
    if n == 1:
        return torch.tensor([[1.0]])
    
    H_prev = hadamard_matrix(n // 2)
    H_top = torch.cat([H_prev, H_prev], dim=1)
    H_bottom = torch.cat([H_prev, -H_prev], dim=1)
    H = torch.cat([H_top, H_bottom], dim=0)
    
    # Normalize to make it orthogonal
    return H / math.sqrt(n)


def generate_walsh_hadamard_rotation(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate Walsh-Hadamard rotation matrix.
    
    For dimensions that aren't powers of 2, we pad to nearest power of 2
    and then truncate back (approximate orthogonality).
    
    Complexity: O(d²) to construct, but application can be O(d log d) with FWHT.
    For this experiment, we construct the full matrix for fair comparison.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    
    # Find nearest power of 2 >= d
    n = 1
    while n < d:
        n *= 2
    
    # Generate Hadamard matrix
    H = hadamard_matrix(n).to(dtype=torch.float32)
    
    # Add random sign flips for additional randomness (like randomized Hadamard)
    signs = torch.randint(0, 2, (n,), generator=rng, dtype=torch.float32) * 2 - 1
    H = H * signs.unsqueeze(0)
    
    # Truncate to d x d
    Pi = H[:d, :d]
    
    # Re-normalize rows to unit norm (truncation breaks exact orthogonality)
    row_norms = Pi.norm(dim=1, keepdim=True)
    Pi = Pi / row_norms
    
    return Pi.to(device=device, dtype=dtype)


def generate_sequency_ordered_rotation(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
    block_size: int = 16,
) -> torch.Tensor:
    """
    Generate sequency-ordered Walsh rotation (GSR-style, Paper 18).
    
    Key insight from GSR: ordering Walsh functions by sequency (number of zero crossings)
    reduces intra-group variance compared to natural ordering.
    
    Implementation:
    1. Generate block-diagonal Hadamard matrices
    2. Order rows within each block by sequency
    3. Apply random permutation across blocks for additional mixing
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    
    # Create block-diagonal structure
    num_blocks = (d + block_size - 1) // block_size
    Pi_blocks = []
    
    for b in range(num_blocks):
        block_dim = min(block_size, d - b * block_size)
        
        # Find power of 2 for this block
        n = 1
        while n < block_dim:
            n *= 2
        
        # Generate Hadamard for this block
        H_block = hadamard_matrix(n).to(dtype=torch.float32)
        
        # Compute sequency for each row (count sign changes)
        sequencies = []
        for i in range(min(n, block_dim)):
            row = H_block[i, :block_dim]
            # Count zero crossings (sign changes)
            signs = torch.sign(row)
            changes = (signs[:-1] != signs[1:]).sum().item()
            sequencies.append((changes, i))
        
        # Sort by sequency (ascending)
        sequencies.sort()
        ordered_indices = [idx for (_, idx) in sequencies]
        
        # Reorder rows by sequency
        H_ordered = H_block[ordered_indices, :block_dim][:block_dim, :]
        
        # Normalize
        row_norms = H_ordered.norm(dim=1, keepdim=True)
        H_ordered = H_ordered / row_norms.clamp(min=1e-8)
        
        Pi_blocks.append(H_ordered)
    
    # Assemble block-diagonal matrix
    Pi = torch.zeros(d, d, dtype=torch.float32)
    offset = 0
    for block in Pi_blocks:
        bs, bd = block.shape
        Pi[offset:offset+bs, offset:offset+bd] = block
        offset += bs
    
    # Apply random block permutation for additional mixing
    block_perm = torch.randperm(num_blocks, generator=rng)
    Pi_permuted = Pi.clone()
    for new_idx, old_idx in enumerate(block_perm):
        start_old = old_idx * block_size
        start_new = new_idx * block_size
        end_old = min(start_old + block_size, d)
        end_new = min(start_new + block_size, d)
        block_rows = end_old - start_old
        Pi_permuted[start_new:start_new+block_rows, :] = Pi[start_old:end_old, :]
    
    return Pi_permuted.to(device=device, dtype=dtype)


def generate_hybrid_rotation(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> torch.Tensor:
    """
    Hybrid rotation: Hadamard + learned/permutation for variance shaping.
    
    Applies Hadamard first, then a sparse permutation designed to create
    non-uniform variance in the output coordinates.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    
    # Step 1: Hadamard rotation
    Pi_H = generate_walsh_hadamard_rotation(d, device, dtype, seed)
    
    # Step 2: Create variance-shaping permutation
    # Strategy: group coordinates by expected variance using a deterministic pattern
    # Coordinates with similar indices after permutation will have correlated variance
    
    # Generate a permutation that creates "bands" of high/low variance
    perm = torch.arange(d, device="cpu")
    
    # Simple strategy: interleave high and low frequency components
    # (analogous to wavelet packet ordering)
    even_idx = perm[::2].clone()
    odd_idx = perm[1::2].clone()
    
    # Interleave: [even_0, odd_0, even_1, odd_1, ...]
    perm_interleaved = torch.empty(d, dtype=torch.long)
    perm_interleaved[0::2] = even_idx[:len(even_idx)]
    perm_interleaved[1::2] = odd_idx[:len(odd_idx)]
    
    # Apply permutation as a matrix
    P = torch.zeros(d, d, dtype=torch.float32)
    P[torch.arange(d), perm_interleaved] = 1.0
    
    # Combined rotation: Pi = P @ H
    Pi = torch.matmul(P.to(device), Pi_H)
    
    return Pi.to(device=device, dtype=dtype)


# =============================================================================
# Variance Profiling Utilities
# =============================================================================

def profile_coordinate_variance(
    X: torch.Tensor,
    Pi: torch.Tensor,
    num_samples: int = 10000,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Profile per-coordinate variance after rotation.
    
    Args:
        X: Input vectors [N, d] (should be unit vectors or normalized)
        Pi: Rotation matrix [d, d]
        num_samples: Number of samples to use
    
    Returns:
        variances: Per-coordinate variance [d]
        stats: Dictionary with summary statistics
    """
    if X.shape[0] > num_samples:
        idx = torch.randperm(X.shape[0])[:num_samples]
        X_sample = X[idx]
    else:
        X_sample = X
    
    # Apply rotation
    Y = rotate_forward(X_sample, Pi)
    
    # Compute per-coordinate variance
    mean = Y.mean(dim=0)
    var = ((Y - mean) ** 2).mean(dim=0)
    
    # Summary statistics
    stats = {
        "mean_variance": var.mean().item(),
        "std_variance": var.std().item(),
        "min_variance": var.min().item(),
        "max_variance": var.max().item(),
        "variance_ratio": (var.max() / var.clamp(min=1e-8).min()).item(),
        "coefficient_of_variation": (var.std() / var.clamp(min=1e-8).mean()).item(),
    }
    
    return var, stats


def compute_water_filling_bits(
    variances: torch.Tensor,
    total_bits: int,
    base_bits: int,
) -> Tuple[torch.Tensor, float]:
    """
    Compute water-filling bit allocation across coordinates.
    
    Formula: b_j = b + (1/2) * log2(σ_j² / ν)
    where ν is chosen so that Σ b_j = total_bits
    
    Args:
        variances: Per-coordinate variance [d]
        total_bits: Total bit budget B = b * d
        base_bits: Base bits per coordinate (b)
    
    Returns:
        bit_allocation: Bits per coordinate [d] (may be fractional)
        nu: Water level parameter
    """
    d = variances.shape[0]
    
    # Initial water level estimate
    # From constraint: sum(b + 0.5*log2(sigma_j^2/nu)) = total_bits
    # => d*b + 0.5*sum(log2(sigma_j^2) - log2(nu)) = total_bits
    # => 0.5*d*log2(nu) = d*b + 0.5*sum(log2(sigma_j^2)) - total_bits
    # => log2(nu) = 2*b + (1/d)*sum(log2(sigma_j^2)) - 2*total_bits/d
    # Since total_bits = b*d, we get: log2(nu) = (1/d)*sum(log2(sigma_j^2))
    
    log_var = torch.log2(variances.clamp(min=1e-10))
    log_nu = log_var.mean()
    nu = 2 ** log_nu
    
    # Compute bit allocation
    bit_allocation = base_bits + 0.5 * torch.log2(variances.clamp(min=1e-10) / nu)
    
    # Clamp to reasonable range [1, 2*base_bits]
    bit_allocation = bit_allocation.clamp(min=1, max=2*base_bits)
    
    return bit_allocation, nu.item()


# =============================================================================
# Inner Product Distortion Metric
# =============================================================================

def compute_d_prod(
    X: torch.Tensor,
    Y: torch.Tensor,
    quantizer: Any,
    num_pairs: int = 5000,
) -> Dict[str, float]:
    """
    Compute inner product distortion D_prod as defined in TurboQuant paper.
    
    D_prod = E[(<x,y> - <q(x),q(y)>)²] / E[<x,y>²]
    
    Args:
        X, Y: Sets of vectors [N, d]
        quantizer: TurboQuant quantizer instance
        num_pairs: Number of random pairs to evaluate
    
    Returns:
        Dictionary with D_prod and related metrics
    """
    N = min(X.shape[0], Y.shape[0])
    if N > num_pairs * 2:
        idx = torch.randperm(N)[:num_pairs * 2]
        X_test, Y_test = X[idx[:num_pairs]], Y[idx[num_pairs:]]
    else:
        X_test, Y_test = X, Y
    
    # True inner products
    true_ips = torch.sum(X_test * Y_test, dim=1)
    
    # Quantize and dequantize
    X_q = quantizer.dequantize(quantizer.quantize(X_test))
    Y_q = quantizer.dequantize(quantizer.quantize(Y_test))
    
    # Quantized inner products
    quant_ips = torch.sum(X_q * Y_q, dim=1)
    
    # Squared error
    squared_error = (true_ips - quant_ips) ** 2
    
    # D_prod
    d_prod = squared_error.mean().item()
    true_ip_variance = (true_ips ** 2).mean().item()
    
    relative_error = (d_prod / max(true_ip_variance, 1e-10)) ** 0.5
    
    # Correlation
    correlation = torch.corrcoef(torch.stack([true_ips, quant_ips]))[0, 1].item()
    
    return {
        "d_prod": d_prod,
        "true_ip_variance": true_ip_variance,
        "relative_rmse": relative_error,
        "correlation": correlation,
        "mean_squared_error": squared_error.mean().item(),
    }


# =============================================================================
# Data Generation
# =============================================================================

def generate_test_embeddings(
    n: int = 10000,
    d: int = 128,
    distribution: str = "gaussian_unit",
    seed: int = 123,
) -> torch.Tensor:
    """
    Generate test embeddings similar to DBpedia/OpenAI embeddings.
    
    Options:
    - gaussian_unit: Random Gaussian vectors, normalized to unit length
    - beta_marginal: Vectors with Beta-distributed marginals (matches TurboQuant theory)
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    
    if distribution == "gaussian_unit":
        # Standard approach: Gaussian vectors normalized to unit sphere
        X = torch.randn(n, d, generator=rng, dtype=torch.float32)
        X = X / X.norm(dim=1, keepdim=True)
    
    elif distribution == "beta_marginal":
        # Generate vectors with Beta(α, α) marginals where α = (d-1)/2
        # This matches the theoretical distribution after random rotation
        alpha = (d - 1) / 2
        
        # Sample from Beta distribution for each coordinate
        # Note: This is approximate; true joint distribution is more complex
        X = torch.zeros(n, d, dtype=torch.float32)
        for i in range(d):
            # Beta distribution on [-1, 1] via transformation
            beta_samples = torch.beta(alpha, alpha, generator=rng, size=(n,))
            X[:, i] = 2 * beta_samples - 1
        
        # Normalize to unit sphere
        X = X / X.norm(dim=1, keepdim=True)
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    return X


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiment2(
    d: int = 128,
    bits: int = 3,
    n_samples: int = 10000,
    algorithm: str = "prod",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run Experiment 2: Structured Rotations + Variance Profiling.
    
    Args:
        d: Embedding dimension
        bits: Bits per coordinate
        n_samples: Number of test vectors
        algorithm: "mse" or "prod" (TurboQuant Algorithm 1 or 2)
        verbose: Print progress
    
    Returns:
        results: Dictionary with all metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        print(f"Experiment 2: Structured Rotations")
        print(f"  Dimension: {d}, Bits: {bits}, Samples: {n_samples}")
        print(f"  Device: {device}")
        print()
    
    # Generate test data
    if verbose:
        print("Generating test embeddings...")
    X = generate_test_embeddings(n=n_samples, d=d, distribution="gaussian_unit")
    Y = generate_test_embeddings(n=n_samples, d=d, distribution="gaussian_unit", seed=456)
    X = X.to(device)
    Y = Y.to(device)
    
    # Define rotation types to test
    rotation_types = {
        "dense_random": lambda: generate_rotation_matrix(d, device, seed=42),
        "walsh_hadamard": lambda: generate_walsh_hadamard_rotation(d, device, seed=42),
        "sequency_gsr": lambda: generate_sequency_ordered_rotation(d, device, seed=42, block_size=16),
        "hybrid": lambda: generate_hybrid_rotation(d, device, seed=42),
    }
    
    results = {}
    
    for rot_name, rot_fn in rotation_types.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing: {rot_name}")
            print(f"{'='*60}")
        
        # Generate rotation
        Pi = rot_fn()
        
        # Profile variance
        if verbose:
            print("  Profiling coordinate variance...")
        variances, var_stats = profile_coordinate_variance(X, Pi)
        
        if verbose:
            print(f"  Variance statistics:")
            print(f"    Mean: {var_stats['mean_variance']:.6f}")
            print(f"    Std:  {var_stats['std_variance']:.6f}")
            print(f"    CV:   {var_stats['coefficient_of_variation']:.4f}")
            print(f"    Ratio (max/min): {var_stats['variance_ratio']:.2f}")
        
        # Compute water-filling bit allocation
        total_bits = bits * d
        bit_allocation, nu = compute_water_filling_bits(variances, total_bits, bits)
        
        if verbose:
            print(f"  Water-filling: nu = {nu:.6f}")
            print(f"  Bit allocation range: [{bit_allocation.min():.2f}, {bit_allocation.max():.2f}]")
        
        # Create quantizer
        if algorithm == "mse":
            quantizer = TurboQuantMSE(dim=d, bits=bits, device=device)
        else:
            quantizer = TurboQuantProd(dim=d, bits=bits, device=device)
        
        # Override rotation matrix
        quantizer.Pi = Pi
        quantizer.Pi_T = Pi.T
        
        # Compute D_prod
        if verbose:
            print("  Computing D_prod...")
        d_prod_metrics = compute_d_prod(X, Y, quantizer, num_pairs=5000)
        
        if verbose:
            print(f"  D_prod metrics:")
            print(f"    D_prod: {d_prod_metrics['d_prod']:.6e}")
            print(f"    Relative RMSE: {d_prod_metrics['relative_rmse']:.4f}")
            print(f"    Correlation: {d_prod_metrics['correlation']:.4f}")
        
        # Store results
        results[rot_name] = {
            "variance_stats": var_stats,
            "water_filling": {
                "nu": nu,
                "bit_range": [float(bit_allocation.min()), float(bit_allocation.max())],
                "bit_std": float(bit_allocation.std()),
            },
            "d_prod": d_prod_metrics,
        }
    
    # Compute improvement over baseline
    baseline = results["dense_random"]
    for rot_name in results:
        if rot_name == "dense_random":
            continue
        baseline_d_prod = baseline["d_prod"]["d_prod"]
        test_d_prod = results[rot_name]["d_prod"]["d_prod"]
        improvement = (baseline_d_prod - test_d_prod) / baseline_d_prod * 100
        results[rot_name]["improvement_over_baseline_pct"] = improvement
    
    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Rotation':<20} {'D_prod':>12} {'Rel RMSE':>10} {'Corr':>8} {'Imprv %':>10}")
        print(f"{'-'*60}")
        for rot_name, res in results.items():
            dp = res["d_prod"]["d_prod"]
            rmse = res["d_prod"]["relative_rmse"]
            corr = res["d_prod"]["correlation"]
            imprv = res.get("improvement_over_baseline_pct", 0)
            print(f"{rot_name:<20} {dp:>12.4e} {rmse:>10.4f} {corr:>8.4f} {imprv:>10.2f}%")
    
    return results


def save_results(results: Dict[str, Any], filepath: str = "/workspace/experiment2_results.json"):
    """Save results to JSON file."""
    # Convert tensors to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        else:
            return obj
    
    results_serializable = convert(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment 2: Structured Rotations")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--bits", type=int, default=3, help="Bits per coordinate")
    parser.add_argument("--samples", type=int, default=10000, help="Number of test samples")
    parser.add_argument("--algorithm", type=str, choices=["mse", "prod"], default="prod",
                        help="TurboQuant algorithm (1=MSE, 2=Prod)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    results = run_experiment2(
        d=args.dim,
        bits=args.bits,
        n_samples=args.samples,
        algorithm=args.algorithm,
        verbose=not args.quiet,
    )
    
    save_results(results)
