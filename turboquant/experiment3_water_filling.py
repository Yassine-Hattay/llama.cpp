#!/usr/bin/env python3
# experiment3_water_filling.py
"""
Experiment 3: Water-Filling Bit Allocation on Anisotropic Embeddings
=====================================================================
Tests whether non-uniform bit allocation (via water-filling) improves 
D_prod when post-rotation coordinate variances are non-uniform.

Precondition: Variance ratio > 1.5x after structured rotation (validated in Exp 2)

Building on Experiments 1 & 2:
- Exp 1: IP-optimized codebook showed +6.39% improvement on anisotropic data
- Exp 2: Hadamard rotation showed +6.06% improvement on anisotropic data
- Exp 3: Combine both with water-filling bit allocation for potential 10-15% total gain

Usage:
    python experiment3_water_filling.py --dim 384 --bits 3 --samples 5000
"""

import argparse
import json
import numpy as np
import torch
from scipy.optimize import bisect
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd
from turboquant.rotation import generate_rotation_matrix, rotate_forward
from scipy.linalg import hadamard


def load_anisotropic_embeddings(path: str = "/workspace/real_embeddings_anisotropic.npy"):
    """Load anisotropic test vectors generated in validation step."""
    vectors = np.load(path)
    print(f"✓ Loaded {vectors.shape[0]} anisotropic vectors (dim={vectors.shape[1]})")
    return vectors


def apply_rotation(vectors: np.ndarray, rot_type: str, seed: int = 42) -> np.ndarray:
    """Apply rotation to vectors: 'dense' (QR) or 'hadamard' (FWHT-style)."""
    n, d = vectors.shape
    device = torch.device('cpu')
    
    if rot_type == 'dense':
        Pi = generate_rotation_matrix(d, device=device, seed=seed).numpy()
    elif rot_type == 'hadamard':
        # Use nearest power-of-2 Hadamard, truncate/pad as needed
        pow2 = 2 ** int(np.ceil(np.log2(d)))
        H = hadamard(pow2) / np.sqrt(pow2)
        if pow2 > d:
            # Truncate to actual dimension
            Pi = H[:d, :d]
        else:
            Pi = H
    else:
        raise ValueError(f"Unknown rotation type: {rot_type}")
    
    return vectors @ Pi.T


def profile_coordinate_variance(rotated: np.ndarray) -> dict:
    """Compute per-coordinate variance statistics."""
    vars = np.var(rotated, axis=0)
    return {
        "mean": float(np.mean(vars)),
        "std": float(np.std(vars)),
        "min": float(np.min(vars)),
        "max": float(np.max(vars)),
        "ratio": float(np.max(vars) / (np.min(vars) + 1e-10)),
        "cv": float(np.std(vars) / (np.mean(vars) + 1e-10)),
        "full_array": vars.tolist()  # For water-filling
    }


def compute_water_filling_bits(variances: np.ndarray, total_bits: int, 
                                base_bits: float,
                                min_bits: float = 1.0, 
                                max_bits: float = 8.0) -> np.ndarray:
    """
    Water-filling bit allocation: b_j = b + 0.5 * log2(σ_j² / ν)
    
    Solve for ν such that Σ b_j = total_bits using binary search.
    
    Args:
        variances: per-coordinate variances σ_j²
        total_bits: total bit budget B = b·d
        base_bits: nominal bits per coordinate b
        min_bits, max_bits: clipping bounds for stability
    """
    d = len(variances)
    target_avg = total_bits / d
    
    def budget_residual(log_nu):
        nu = np.exp(log_nu)
        bits = base_bits + 0.5 * np.log2(np.array(variances) / (nu + 1e-10))
        bits_clipped = np.clip(bits, min_bits, max_bits)
        return np.sum(bits_clipped) - total_bits
    
    # Binary search for log(ν)
    log_nu_low, log_nu_high = -30, 30
    for _ in range(100):
        mid = (log_nu_low + log_nu_high) / 2
        residual = budget_residual(mid)
        if residual > 0:
            # Too many bits, need larger ν
            log_nu_low = mid
        else:
            # Too few bits, need smaller ν
            log_nu_high = mid
    
    nu = np.exp((log_nu_low + log_nu_high) / 2)
    bit_allocation = base_bits + 0.5 * np.log2(np.array(variances) / (nu + 1e-10))
    bit_allocation = np.clip(bit_allocation, min_bits, max_bits)
    
    # Verify budget constraint
    actual_total = np.sum(bit_allocation)
    print(f"  Water-filling: ν={nu:.6f}, allocated {actual_total:.1f} bits (target {total_bits})")
    
    return bit_allocation


def simulate_variable_bit_quantization(vectors: np.ndarray, bit_allocation: np.ndarray, 
                                        base_bits: int, ip_optimized: bool,
                                        device: torch.device) -> np.ndarray:
    """
    Simulate variable-bit quantization by scaling per-coordinate reconstruction error.
    
    The key insight: higher bit allocation → lower quantization error.
    Approximation: MSE ∝ 2^(-2b), so error scales as 2^(-b).
    
    For a proper implementation, we would need to:
    1. Generate separate codebooks for each bit-width
    2. Apply different codebooks per coordinate
    
    This simulation approximates the effect via error scaling relative to baseline.
    
    Args:
        vectors: input vectors to quantize
        bit_allocation: per-coordinate bit allocation b_j
        base_bits: baseline uniform bit-width
        ip_optimized: whether to use IP-optimized codebooks
        device: torch device
    """
    n, d = vectors.shape
    
    # Use baseline quantizer with base_bits
    quantizer = TurboQuantMSE(dim=d, bits=base_bits, device=device, ip_optimized=ip_optimized)
    
    # Quantize and dequantize
    x_tensor = torch.tensor(vectors, dtype=torch.float32, device=device)
    q_codes = quantizer.quantize(x_tensor)
    x_recon = quantizer.dequantize(q_codes).cpu().numpy()
    
    # Compute per-coordinate error
    errors = vectors - x_recon
    
    # Scale errors based on bit allocation difference
    # Higher bits → lower error: error_j ∝ 2^(-(b_j - base_bits))
    # We use sqrt because MSE ∝ 2^(-2b), so RMSE ∝ 2^(-b)
    bit_diff = bit_allocation - base_bits
    error_scale = np.power(2.0, -bit_diff)
    error_scale = np.clip(error_scale, 0.1, 2.0)  # Prevent extreme scaling
    
    # Apply per-coordinate scaling
    scaled_errors = errors * error_scale
    x_recon_scaled = vectors - scaled_errors
    
    return x_recon_scaled


def compute_d_prod(X: np.ndarray, Y: np.ndarray, X_q: np.ndarray, Y_q: np.ndarray) -> dict:
    """Compute inner product distortion D_prod = E[(<x,y> - <q(x),q(y)>)²]."""
    true_ips = np.sum(X * Y, axis=1)
    quant_ips = np.sum(X_q * Y_q, axis=1)
    
    mse = np.mean((true_ips - quant_ips) ** 2)
    true_var = np.mean(true_ips ** 2)
    rel_rmse = np.sqrt(mse / (true_var + 1e-10))
    corr = np.corrcoef(true_ips, quant_ips)[0, 1]
    
    return {
        "d_prod": float(mse),
        "true_ip_variance": float(true_var),
        "relative_rmse": float(rel_rmse),
        "correlation": float(corr if not np.isnan(corr) else 0.0)
    }


def run_experiment3(vectors: np.ndarray, bits: int = 3, rot_types: list = None,
                    ip_optimized: bool = True):
    """
    Main experiment loop testing water-filling vs uniform bit allocation.
    
    Args:
        vectors: anisotropic test vectors
        bits: base bit-width
        rot_types: list of rotation types to test
        ip_optimized: whether to use IP-optimized codebooks (from Exp 1)
    """
    if rot_types is None:
        rot_types = ['dense', 'hadamard']
    
    results = {}
    device = torch.device('cpu')
    
    # Use subset for speed while maintaining statistical power
    n_samples = min(2000, len(vectors) // 2)
    idx = np.random.choice(len(vectors), n_samples * 2, replace=False)
    X, Y = vectors[idx[:n_samples]], vectors[idx[n_samples:]]
    
    print(f"\n{'='*70}")
    print(f"Experiment 3: Water-Filling Bit Allocation")
    print(f"Config: {vectors.shape[0]} vectors, dim={vectors.shape[1]}, bits={bits}")
    print(f"IP-optimized codebooks: {ip_optimized}")
    print(f"{'='*70}")
    
    for rot_type in rot_types:
        print(f"\n{'─'*70}")
        print(f"Testing rotation: {rot_type.upper()}")
        print(f"{'─'*70}")
        
        # Apply rotation
        X_rot = apply_rotation(X, rot_type)
        Y_rot = apply_rotation(Y, rot_type)
        
        # Profile variance (this is the precondition for water-filling)
        var_stats = profile_coordinate_variance(X_rot)
        print(f"\nVariance Statistics:")
        print(f"  Mean: {var_stats['mean']:.6f}")
        print(f"  Std:  {var_stats['std']:.6f}")
        print(f"  Min:  {var_stats['min']:.6f}")
        print(f"  Max:  {var_stats['max']:.6f}")
        print(f"  Ratio (max/min): {var_stats['ratio']:.2f}x")
        print(f"  CV (σ/μ): {var_stats['cv']:.3f}")
        
        # Check precondition
        if var_stats['ratio'] < 1.5:
            print(f"  ⚠️  WARNING: Variance ratio {var_stats['ratio']:.2f}x < 1.5x")
            print(f"      Water-filling may have minimal impact")
        
        # Water-filling bit allocation
        total_bits = bits * X_rot.shape[1]
        bit_alloc = compute_water_filling_bits(
            np.array(var_stats['full_array']), 
            total_bits, 
            base_bits=float(bits)
        )
        print(f"\nBit Allocation:")
        print(f"  Uniform baseline: {bits} bits/coord (total {total_bits})")
        print(f"  Water-filling range: [{bit_alloc.min():.2f}, {bit_alloc.max():.2f}] bits")
        print(f"  Water-filling mean: {bit_alloc.mean():.3f} bits")
        print(f"  Water-filling std: {bit_alloc.std():.3f} bits")
        
        # Baseline: uniform bit allocation
        print(f"\nQuantizing with uniform bit allocation...")
        X_q_uniform = simulate_variable_bit_quantization(
            X_rot, np.full_like(bit_alloc, bits), bits, ip_optimized, device
        )
        Y_q_uniform = simulate_variable_bit_quantization(
            Y_rot, np.full_like(bit_alloc, bits), bits, ip_optimized, device
        )
        d_prod_uniform = compute_d_prod(X_rot, Y_rot, X_q_uniform, Y_q_uniform)
        
        # Water-filling: variable bit allocation
        print(f"Quantizing with water-filling bit allocation...")
        X_q_wf = simulate_variable_bit_quantization(
            X_rot, bit_alloc, bits, ip_optimized, device
        )
        Y_q_wf = simulate_variable_bit_quantization(
            Y_rot, bit_alloc, bits, ip_optimized, device
        )
        d_prod_wf = compute_d_prod(X_rot, Y_rot, X_q_wf, Y_q_wf)
        
        # Compare
        improvement = (d_prod_uniform['d_prod'] - d_prod_wf['d_prod']) / d_prod_uniform['d_prod'] * 100
        
        print(f"\nResults:")
        print(f"  D_prod (uniform):       {d_prod_uniform['d_prod']:.6e}")
        print(f"  D_prod (water-filling): {d_prod_wf['d_prod']:.6e}")
        print(f"  Improvement: {improvement:+.2f}%")
        print(f"  Relative RMSE: {d_prod_wf['relative_rmse']:.4f}")
        print(f"  Correlation: {d_prod_wf['correlation']:.4f}")
        
        results[rot_type] = {
            "variance_stats": var_stats,
            "bit_allocation": {
                "mean": float(np.mean(bit_alloc)),
                "std": float(np.std(bit_alloc)),
                "min": float(np.min(bit_alloc)),
                "max": float(np.max(bit_alloc)),
                "range": float(np.max(bit_alloc) - np.min(bit_alloc))
            },
            "d_prod_uniform": d_prod_uniform,
            "d_prod_water_filling": d_prod_wf,
            "improvement_pct": float(improvement),
            "precondition_met": var_stats['ratio'] >= 1.5
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Water-Filling Bit Allocation")
    parser.add_argument('--dim', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--bits', type=int, default=3, help='Base bit-width')
    parser.add_argument('--samples', type=int, default=5000, help='Number of test vectors')
    parser.add_argument('--data', type=str, default='/workspace/real_embeddings_anisotropic.npy',
                       help='Path to anisotropic embeddings')
    parser.add_argument('--no-ip-opt', action='store_true', 
                       help='Disable IP-optimized codebooks (use standard MSE)')
    parser.add_argument('--rotations', type=str, nargs='+', default=['dense', 'hadamard'],
                       choices=['dense', 'hadamard'],
                       help='Rotation types to test')
    args = parser.parse_args()
    
    # Load data
    vectors = load_anisotropic_embeddings(args.data)
    
    # Run experiment
    results = run_experiment3(
        vectors, 
        bits=args.bits, 
        rot_types=args.rotations,
        ip_optimized=not args.no_ip_opt
    )
    
    # Save results
    output_path = '/workspace/experiment3_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"✓ Results saved to {output_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    total_improvement = 0.0
    count = 0
    for rot, res in results.items():
        imp = res['improvement_pct']
        marker = "↑" if imp > 0 else "↓" if imp < 0 else "─"
        print(f"{rot:12s}: {imp:+.2f}% D_prod change {marker}")
        total_improvement += imp
        count += 1
    
    avg_improvement = total_improvement / count if count > 0 else 0.0
    print(f"\nAverage improvement: {avg_improvement:+.2f}%")
    
    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    
    if avg_improvement > 3.0:
        print("✓ Water-filling shows SIGNIFICANT improvement (>3%)")
        print("  → Recommended for integration into TurboQuant pipeline")
        print("  → Expected total gain (Exp1+Exp2+Exp3): ~10-15%")
    elif avg_improvement > 0.5:
        print("△ Water-filling shows MODEST improvement (0.5-3%)")
        print("  → May be worth integrating if engineering cost is low")
        print("  → Consider full implementation with per-coordinate codebooks")
    else:
        print("✗ Water-filling shows MINIMAL improvement (<0.5%)")
        print("  → Variance non-uniformity may not be sufficient")
        print("  → Consider alternative: refine QJL residual stage (Claude's Option B)")
    
    print(f"\nNext steps:")
    print(f"  1. Review experiment3_results.json for detailed metrics")
    print(f"  2. If improvement >3%, implement full variable-bit quantizer")
    print(f"  3. Otherwise, pivot to QJL residual optimization")


if __name__ == "__main__":
    main()
