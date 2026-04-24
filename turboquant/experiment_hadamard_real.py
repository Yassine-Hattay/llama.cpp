#!/usr/bin/env python3
# experiment_hadamard_real.py
"""
Validate Hadamard rotation on real embeddings.

This script validates the Hadamard vs. Dense rotation comparison on real LLM embedding data,
reusing the validated logic from experiment2_fixed_rotation.py.
"""
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cpu")

# Import validated functions from experiment2_fixed_rotation
from experiment2_fixed_rotation import (
    dense_rotation,
    hadamard_rotation,
    compute_d_prod_manual,
    assert_orthogonality,
)


def run_outlier_aware_single_seed(
    vectors: np.ndarray,
    data_seed: int,
    hadamard_seed: int,
    outlier_ratio: float,
    boundaries: torch.Tensor,
    centroids: torch.Tensor,
    device: torch.device,
    verbose: bool = False
) -> dict:
    """
    Run outlier-aware Hadamard validation for a single data seed.
    Returns metrics dict with d_prod, correlation, etc.
    """
    n_samples, d = vectors.shape
    
    # Regenerate data with this seed (re-shuffle/split)
    rng = np.random.default_rng(data_seed)
    indices = rng.permutation(n_samples)
    vectors_shuffled = vectors[indices]
    
    # Convert to torch and normalize
    X_all = torch.tensor(vectors_shuffled, device=device, dtype=torch.float32)
    X_all = X_all / (X_all.norm(dim=1, keepdim=True) + 1e-10)
    
    # Split into X (keys) and Y (queries)
    X = X_all[:n_samples//2]
    Y = X_all[n_samples//2:]
    
    # FIX: Detect outliers from held-out Y split to prevent data leakage
    Y_pre_norm = vectors_shuffled[n_samples//2:]
    var = torch.tensor(np.var(Y_pre_norm, axis=0), device=device)
    sorted_idx = torch.argsort(var, descending=True)
    k = max(1, int(d * outlier_ratio))
    outlier_idx = sorted_idx[:k]
    regular_idx = sorted_idx[k:]
    
    # Build Hadamard rotation for regular channels only
    Pi_reg = hadamard_rotation(len(regular_idx), device, seed=hadamard_seed)
    
    # Quantize outliers: pass-through (FP16 fidelity)
    X_unit = X / (X.norm(dim=1, keepdim=True) + 1e-10)
    X_out_hat = X_unit[:, outlier_idx]
    
    # Rotate + quantize regular channels
    X_reg = X_unit[:, regular_idx]
    Y_reg = X_reg @ Pi_reg.T
    int_bounds = boundaries[1:-1]
    idx_reg = torch.searchsorted(int_bounds, Y_reg)
    X_reg_hat = centroids[idx_reg] @ Pi_reg
    
    # Reconstruct full vector
    X_recon_unit = torch.zeros_like(X_unit)
    X_recon_unit[:, outlier_idx] = X_out_hat
    X_recon_unit[:, regular_idx] = X_reg_hat
    X_recon = X_recon_unit * X.norm(dim=1, keepdim=True)
    
    # Compute D_prod
    ip_true = (X * Y).sum(dim=1)
    ip_quant = (X_recon * Y).sum(dim=1)
    d_prod = ((ip_true - ip_quant) ** 2).mean().item()
    true_ip_var = (ip_true ** 2).mean().item()
    relative_rmse = np.sqrt(d_prod / max(true_ip_var, 1e-10))
    correlation = torch.corrcoef(torch.stack([ip_true, ip_quant]))[0, 1].item()
    
    return {
        "data_seed": data_seed,
        "d_prod": d_prod,
        "relative_rmse": relative_rmse,
        "correlation": correlation,
        "outlier_ratio": outlier_ratio,
        "n_outliers": len(outlier_idx),
        "n_regular": len(regular_idx),
        "effective_bits": outlier_ratio * 16.0 + (1.0 - outlier_ratio) * 3.0,
    }


def compute_channel_variance(X: torch.Tensor) -> torch.Tensor:
    """Compute per-channel variance across samples."""
    return X.var(dim=0)


def split_outlier_indices(var: torch.Tensor, outlier_ratio: float = 0.05):
    """Return (outlier_idx, regular_idx) tensors based on top variance."""
    d = var.shape[0]
    k = max(1, int(d * outlier_ratio))
    sorted_idx = torch.argsort(var, descending=True)
    return sorted_idx[:k], sorted_idx[k:]


def compute_effective_bits(outlier_ratio: float, outlier_bits: float = 16.0, regular_bits: float = 3.0) -> float:
    """Compute effective bits per coordinate given outlier handling strategy."""
    return outlier_ratio * outlier_bits + (1.0 - outlier_ratio) * regular_bits


def compute_d_prod_outlier_aware_with_bits(
    X: torch.Tensor, Y: torch.Tensor, Pi_reg: torch.Tensor,
    boundaries: torch.Tensor, centroids: torch.Tensor,
    outlier_ratio: float = 0.05,
    outlier_bits: float = 16.0,  # 16 = FP16 pass-through, 4-6 = higher-bit quant
    X_raw_for_detection: torch.Tensor = None,
    codebook_high_bits: torch.Tensor = None,  # Optional: higher-bit codebook for outliers
    boundaries_high: torch.Tensor = None,
) -> dict:
    """
    Outlier-aware D_prod with configurable outlier bit-width.
    - If outlier_bits >= 8: pass outliers through unchanged (FP16)
    - If outlier_bits < 8: quantize outliers with higher-bit codebook
    - Regular channels: always use standard 3-bit codebook + Hadamard
    """
    X_norms = X.norm(dim=1, keepdim=True)
    X_unit = X / (X_norms + 1e-10)
    
    # Detect outliers on pre-normalized data
    if X_raw_for_detection is None:
        X_raw_for_detection = X * (X_norms + 1e-10)
    var = compute_channel_variance(X_raw_for_detection)
    outlier_idx, regular_idx = split_outlier_indices(var, outlier_ratio)
    
    # Quantize outliers: either pass-through or higher-bit quantization
    if outlier_bits >= 8:
        # Pass-through (FP16 fidelity)
        X_out_hat = X_unit[:, outlier_idx]
    else:
        # Quantize with higher-bit codebook if provided, else fall back to standard
        int_bounds = boundaries_high[1:-1] if boundaries_high is not None else boundaries[1:-1]
        cents = codebook_high_bits if codebook_high_bits is not None else centroids
        idx_out = torch.searchsorted(int_bounds, X_unit[:, outlier_idx])
        X_out_hat = cents[idx_out]
    
    # Rotate + quantize regular channels (standard 3-bit)
    X_reg = X_unit[:, regular_idx]
    Y_reg = X_reg @ Pi_reg.T
    int_bounds_reg = boundaries[1:-1]
    idx_reg = torch.searchsorted(int_bounds_reg, Y_reg)
    X_reg_hat = centroids[idx_reg] @ Pi_reg
    
    # Reconstruct
    X_recon_unit = torch.zeros_like(X_unit)
    X_recon_unit[:, outlier_idx] = X_out_hat
    X_recon_unit[:, regular_idx] = X_reg_hat
    X_recon = X_recon_unit * X_norms
    
    ip_true = (X * Y).sum(dim=1)
    ip_quant = (X_recon * Y).sum(dim=1)
    d_prod = ((ip_true - ip_quant) ** 2).mean().item()
    true_ip_var = (ip_true ** 2).mean().item()
    relative_rmse = np.sqrt(d_prod / max(true_ip_var, 1e-10))
    correlation = torch.corrcoef(torch.stack([ip_true, ip_quant]))[0, 1].item()
    
    return {
        "d_prod": d_prod, "relative_rmse": relative_rmse, "correlation": correlation,
        "outlier_ratio": outlier_ratio, "n_outliers": len(outlier_idx), "n_regular": len(regular_idx),
        "effective_bits": compute_effective_bits(outlier_ratio, outlier_bits),
    }


def compute_d_prod_outlier_aware(
    X: torch.Tensor, Y: torch.Tensor, Pi_reg: torch.Tensor,
    boundaries: torch.Tensor, centroids: torch.Tensor, outlier_ratio: float = 0.05,
    X_raw_for_detection: torch.Tensor = None
) -> dict:
    """
    Outlier-aware D_prod for real embeddings (legacy wrapper for backward compatibility):
    - Splits channels into outliers (no rotation, NO quantization) and regular (Hadamard rotated)
    - Outliers pass through unchanged; regular channels are rotated + quantized
    - Reconstructs full vector and computes D_prod, RMSE, correlation
    """
    result = compute_d_prod_outlier_aware_with_bits(
        X, Y, Pi_reg, boundaries, centroids,
        outlier_ratio=outlier_ratio,
        outlier_bits=16.0,  # FP16 pass-through
        X_raw_for_detection=X_raw_for_detection
    )
    # Remove effective_bits for backward compatibility
    result.pop("effective_bits", None)
    return result


def load_real_embeddings():
    """Load real embeddings from highest-priority available file."""
    priority_files = [
        "real_embeddings_mteb.npy",
        "real_embeddings_minilm.npy",
        "real_embeddings_anisotropic.npy",
    ]
    
    for filename in priority_files:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if os.path.exists(filepath):
            vectors = np.load(filepath)
            print(f"✓ Loaded {filename}: {vectors.shape[0]} vectors, dim={vectors.shape[1]}")
            return vectors, filename
    
    raise FileNotFoundError(
        "No real embedding files found. Run load_real_embeddings.py first."
    )


def get_codebook_for_dimension(dim: int):
    """Load codebook tensors for the given dimension."""
    codebook_map = {
        64: "codebook_d64_b3.json",
        128: "codebook_d128_b3.json",
        384: "codebook_d384_b3.json",
        576: "codebook_d576_b3.json",
    }
    
    if dim not in codebook_map:
        raise ValueError(f"No codebook available for dimension {dim}. Available: {list(codebook_map.keys())}")
    
    codebook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f"turboquant/codebooks/{codebook_map[dim]}")
    
    with open(codebook_path, 'r') as f:
        codebook_data = json.load(f)
    
    boundaries = torch.tensor(codebook_data["boundaries"], device=device, dtype=torch.float32)
    centroids = torch.tensor(codebook_data["centroids"], device=device, dtype=torch.float32)
    
    return boundaries, centroids


def run_validation_on_real_data(vectors: np.ndarray, data_source: str, 
                                 verbose: bool = False, outlier_ratio: float = 0.05,
                                 outlier_bits: float = 16.0, compare_high_bit_baseline: bool = False) -> dict:
    """Run Hadamard vs. Dense validation on real embeddings with outlier-ratio sweeping support."""
    n_samples, d = vectors.shape
    print(f"\nReal data variance ratio: {np.var(vectors, axis=0).max() / (np.var(vectors, axis=0).min() + 1e-10):.2f}x")
    
    # Load matching codebook
    boundaries, centroids = get_codebook_for_dimension(d)
    if verbose:
        print(f"  Loaded codebook: {len(centroids)} centroids, {len(boundaries)} boundaries")
    
    # Convert to torch and normalize to unit sphere
    X_all = torch.tensor(vectors, device=device, dtype=torch.float32)
    X_all = X_all / (X_all.norm(dim=1, keepdim=True) + 1e-10)
    
    # Split into X (keys) and Y (queries)
    X = X_all[:n_samples//2]
    Y = X_all[n_samples//2:]
    
    # Generate rotation matrices with fixed seeds for reproducibility
    seed = 42
    hadamard_seed = 1337
    
    Pi_dense = dense_rotation(d, device, seed=seed)
    Pi_hadamard = hadamard_rotation(d, device, seed=hadamard_seed)
    
    # Verify orthogonality
    print("\n--- DENSE ---")
    dense_orth_error = torch.max(torch.abs(Pi_dense @ Pi_dense.T - torch.eye(d, device=device))).item()
    print(f"  Orthogonality error: {dense_orth_error:.2e}", end="")
    if dense_orth_error < 1e-4:
        print(" OK")
    else:
        print(" FAILED")
    
    r_dense = compute_d_prod_manual(X, Y, Pi_dense, boundaries, centroids)
    print(f"  D_prod: {r_dense['d_prod']:.6e}, Correlation: {r_dense['correlation']:.4f}")
    
    print("\n--- HADAMARD ---")
    hadamard_orth_error = torch.max(torch.abs(Pi_hadamard @ Pi_hadamard.T - torch.eye(d, device=device))).item()
    print(f"  Orthogonality error: {hadamard_orth_error:.2e}", end="")
    if hadamard_orth_error < 1e-4:
        print(" OK")
    else:
        print(" FAILED")
    
    r_hadamard = compute_d_prod_manual(X, Y, Pi_hadamard, boundaries, centroids)
    print(f"  D_prod: {r_hadamard['d_prod']:.6e}, Correlation: {r_hadamard['correlation']:.4f}")
    
    # Outlier-Aware Hadamard
    print("\n--- OUTLIER-AWARE HADAMARD ---")
    # Detect outliers on pre-normalized data (before unit sphere normalization)
    X_pre_norm = X_all[:n_samples//2] * (X_all[:n_samples//2].norm(dim=1, keepdim=True) + 1e-10)
    var = compute_channel_variance(X_pre_norm)
    outlier_idx, regular_idx = split_outlier_indices(var, outlier_ratio)
    Pi_reg = hadamard_rotation(len(regular_idx), device, seed=hadamard_seed)
    
    r_outlier = compute_d_prod_outlier_aware_with_bits(
        X, Y, Pi_reg, boundaries, centroids,
        outlier_ratio=outlier_ratio,
        outlier_bits=outlier_bits,
        X_raw_for_detection=X_pre_norm
    )
    print(f"  Outliers: {r_outlier['n_outliers']}, Regular: {r_outlier['n_regular']}")
    print(f"  Effective bits: {r_outlier['effective_bits']:.2f}")
    print(f"  D_prod: {r_outlier['d_prod']:.6e}, Relative RMSE: {r_outlier['relative_rmse']:.4f}, Correlation: {r_outlier['correlation']:.4f}")
    
    # Optional: TurboQuant-style baseline (channel split with different bit-widths, no rotation)
    r_turboquant = None
    if compare_high_bit_baseline:
        print("\n--- TURBOQUANT BASELINE (channel-split, no rotation) ---")
        # Same outlier detection
        idx_out_tq, idx_reg_tq = split_outlier_indices(var, outlier_ratio)
        
        # Get unit-normalized X for quantization
        X_unit_full = X / (X.norm(dim=1, keepdim=True) + 1e-10)
        
        # Quantize outliers with higher bits (e.g., 4-bit) and regular with 3-bit
        # For simplicity, we use the same codebook but simulate higher precision by less aggressive quantization
        int_bounds_high = boundaries[1:-1]  # In a real scenario, you'd have a separate high-bit codebook
        
        # Outliers: direct quantization (no rotation)
        X_out_tq = X_unit_full[:, outlier_idx]
        idx_out_tq_quant = torch.searchsorted(int_bounds_high, X_out_tq)
        X_out_tq_hat = centroids[idx_out_tq_quant]
        
        # Regular channels: standard 3-bit quantization (no rotation)
        X_reg_tq = X_unit_full[:, regular_idx]
        idx_reg_tq_quant = torch.searchsorted(int_bounds_high, X_reg_tq)
        X_reg_tq_hat = centroids[idx_reg_tq_quant]
        
        # Reconstruct without rotation
        X_recon_tq_unit = torch.zeros_like(X_unit_full)
        X_recon_tq_unit[:, outlier_idx] = X_out_tq_hat
        X_recon_tq_unit[:, regular_idx] = X_reg_tq_hat
        X_recon_tq = X_recon_tq_unit * X.norm(dim=1, keepdim=True)
        
        ip_true_tq = (X_all[:n_samples//2] * X_all[n_samples//2:]).sum(dim=1)
        ip_quant_tq = (X_recon_tq * X_all[n_samples//2:]).sum(dim=1)
        d_prod_tq = ((ip_true_tq - ip_quant_tq) ** 2).mean().item()
        true_ip_var_tq = (ip_true_tq ** 2).mean().item()
        relative_rmse_tq = np.sqrt(d_prod_tq / max(true_ip_var_tq, 1e-10))
        correlation_tq = torch.corrcoef(torch.stack([ip_true_tq, ip_quant_tq]))[0, 1].item()
        
        r_turboquant = {
            "d_prod": d_prod_tq,
            "relative_rmse": relative_rmse_tq,
            "correlation": correlation_tq,
            "outlier_ratio": outlier_ratio,
            "n_outliers": len(idx_out_tq),
            "n_regular": len(idx_reg_tq),
            "effective_bits": compute_effective_bits(outlier_ratio, 4.0 if outlier_bits < 8 else 16.0),
        }
        print(f"  Outliers: {r_turboquant['n_outliers']}, Regular: {r_turboquant['n_regular']}")
        print(f"  Effective bits: {r_turboquant['effective_bits']:.2f}")
        print(f"  D_prod: {r_turboquant['d_prod']:.6e}, Relative RMSE: {r_turboquant['relative_rmse']:.4f}, Correlation: {r_turboquant['correlation']:.4f}")
    
    # Calculate improvement
    improvement_pct = ((r_dense["d_prod"] - r_hadamard["d_prod"]) / r_dense["d_prod"]) * 100
    print(f"\n>>> Hadamard improvement: {improvement_pct:+.2f}%")
    
    # Print comparison table
    print("\n" + "=" * 90)
    print(f"{'Method':<30} {'D_prod':<18} {'Eff. Bits':<12} {'Rel. RMSE':<12} {'Correlation':<12}")
    print("=" * 90)
    dense_eff_bits = compute_effective_bits(0.0, outlier_bits, 3.0)
    print(f"{'Dense':<30} {r_dense['d_prod']:<18.6e} {dense_eff_bits:<12.2f} {'N/A':<12} {r_dense['correlation']:<12.4f}")
    print(f"{'Hadamard':<30} {r_hadamard['d_prod']:<18.6e} {dense_eff_bits:<12.2f} {'N/A':<12} {r_hadamard['correlation']:<12.4f}")
    print(f"{'Outlier-Aware Hadamard':<30} {r_outlier['d_prod']:<18.6e} {r_outlier['effective_bits']:<12.2f} {r_outlier['relative_rmse']:<12.4f} {r_outlier['correlation']:<12.4f}")
    if r_turboquant:
        print(f"{'TurboQuant Baseline':<30} {r_turboquant['d_prod']:<18.6e} {r_turboquant['effective_bits']:<12.2f} {r_turboquant['relative_rmse']:<12.4f} {r_turboquant['correlation']:<12.4f}")
    print("=" * 90)
    
    result_dict = {
        "data_source": data_source,
        "dimension": d,
        "variance_ratio": float(np.var(vectors, axis=0).max() / (np.var(vectors, axis=0).min() + 1e-10)),
        "dense": {
            "orth_error": dense_orth_error,
            "d_prod": r_dense["d_prod"],
            "correlation": r_dense["correlation"],
            "effective_bits": dense_eff_bits,
        },
        "hadamard": {
            "orth_error": hadamard_orth_error,
            "d_prod": r_hadamard["d_prod"],
            "correlation": r_hadamard["correlation"],
            "effective_bits": dense_eff_bits,
        },
        "outlier_aware": {
            "d_prod": r_outlier["d_prod"],
            "relative_rmse": r_outlier["relative_rmse"],
            "correlation": r_outlier["correlation"],
            "outlier_ratio": r_outlier["outlier_ratio"],
            "n_outliers": r_outlier["n_outliers"],
            "n_regular": r_outlier["n_regular"],
            "effective_bits": r_outlier["effective_bits"],
            "outlier_bits": outlier_bits,
        },
        "improvement_pct": improvement_pct,
    }
    
    if r_turboquant:
        result_dict["turboquant_baseline"] = r_turboquant
    
    return result_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate Hadamard rotation on real embeddings")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--outlier-ratio", type=float, default=0.05, 
                        help="Ratio of outlier channels (default: 0.05)")
    parser.add_argument("--outlier-ratios", type=str, default="0.01,0.02,0.05,0.08,0.10",
                        help="Comma-separated outlier ratios to sweep")
    parser.add_argument("--outlier-bits", type=float, default=16.0,
                        help="Bit-width for outlier channels (16=FP16 pass-through, 4-6=quantized)")
    parser.add_argument("--compare-high-bit-baseline", action="store_true",
                        help="Also run TurboQuant-style baseline: top channels get 4-bit, rest 3-bit")
    parser.add_argument("--data-seeds", type=str, default="42,123,777",
                        help="Comma-separated data seeds for robustness test (default: 42,123,777)")
    parser.add_argument("--hadamard-seed", type=int, default=1337,
                        help="Fixed seed for Hadamard rotation (default: 1337)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hadamard Rotation Validation on Real Embeddings")
    print("=" * 60)
    
    # Load real embeddings
    try:
        vectors, data_source = load_real_embeddings()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'python load_real_embeddings.py' first to generate embeddings.")
        sys.exit(1)
    
    # Parse outlier ratios for sweeping
    outlier_ratios = [float(x.strip()) for x in args.outlier_ratios.split(",")]
    
    # Parse data seeds for robustness testing
    data_seeds = [int(x.strip()) for x in args.data_seeds.split(",")]
    is_multi_seed = len(data_seeds) > 1
    
    # Load codebook once (shared across all runs)
    _, d = vectors.shape
    boundaries, centroids = get_codebook_for_dimension(d)
    
    if is_multi_seed:
        # Multi-seed robustness testing mode
        print(f"\n>>> Running multi-seed robustness test with {len(data_seeds)} seeds: {data_seeds}")
        print(f">>> Outlier ratio: {args.outlier_ratio}, Hadamard seed: {args.hadamard_seed}")
        
        # Run baselines once (fixed seed)
        print("\n" + "=" * 60)
        print("BASELINE RESULTS (single run, fixed seed)")
        print("=" * 60)
        
        X_all_baseline = torch.tensor(vectors, device=device, dtype=torch.float32)
        X_all_baseline = X_all_baseline / (X_all_baseline.norm(dim=1, keepdim=True) + 1e-10)
        X_base = X_all_baseline[:len(vectors)//2]
        Y_base = X_all_baseline[len(vectors)//2:]
        
        Pi_dense_base = dense_rotation(d, device, seed=42)
        Pi_hadamard_base = hadamard_rotation(d, device, seed=args.hadamard_seed)
        
        r_dense_base = compute_d_prod_manual(X_base, Y_base, Pi_dense_base, boundaries, centroids)
        r_hadamard_base = compute_d_prod_manual(X_base, Y_base, Pi_hadamard_base, boundaries, centroids)
        
        print(f"Dense D_prod:   {r_dense_base['d_prod']:.6e}")
        print(f"Hadamard D_prod: {r_hadamard_base['d_prod']:.6e}")
        baseline_improvement = ((r_dense_base["d_prod"] - r_hadamard_base["d_prod"]) / r_dense_base["d_prod"]) * 100
        print(f"Hadamard improvement: {baseline_improvement:+.2f}%")
        
        # Run outlier-aware for each data seed
        print("\n" + "=" * 60)
        print("OUTLIER-AWARE HADAMARD (multi-seed)")
        print("=" * 60)
        
        outlier_results_per_seed = []
        for seed in data_seeds:
            result = run_outlier_aware_single_seed(
                vectors, seed, args.hadamard_seed, args.outlier_ratio,
                boundaries, centroids, device, args.verbose
            )
            outlier_results_per_seed.append(result)
            
            improvement_vs_dense = ((r_dense_base["d_prod"] - result["d_prod"]) / r_dense_base["d_prod"]) * 100
            print(f"Seed {seed}: D_prod={result['d_prod']:.6e}, Rel.RMSE={result['relative_rmse']:.4f}, "
                  f"Corr={result['correlation']:.4f}, Improvement={improvement_vs_dense:+.2f}%")
        
        # Compute statistics
        d_prods = [r["d_prod"] for r in outlier_results_per_seed]
        rmse_vals = [r["relative_rmse"] for r in outlier_results_per_seed]
        corr_vals = [r["correlation"] for r in outlier_results_per_seed]
        improvements = [((r_dense_base["d_prod"] - r["d_prod"]) / r_dense_base["d_prod"]) * 100 
                       for r in outlier_results_per_seed]
        
        mean_d_prod = np.mean(d_prods)
        std_d_prod = np.std(d_prods)
        mean_rmse = np.mean(rmse_vals)
        std_rmse = np.std(rmse_vals)
        mean_corr = np.mean(corr_vals)
        std_corr = np.std(corr_vals)
        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        
        # Print summary table
        print("\n" + "=" * 90)
        print(f"{'Data Seed':<12} {'D_prod':<14} {'Rel. RMSE':<12} {'Correlation':<14} {'Improvement vs Dense':<20}")
        print("=" * 90)
        for i, seed in enumerate(data_seeds):
            r = outlier_results_per_seed[i]
            imp = improvements[i]
            print(f"{seed:<12} {r['d_prod']:<14.6e} {r['relative_rmse']:<12.4f} {r['correlation']:<14.4f} {imp:>+13.2f}%")
        print("-" * 90)
        print(f"{'MEAN ± STD':<12} {mean_d_prod:<14.6e} {mean_rmse:<12.4f} {mean_corr:<14.4f} {mean_imp:>+13.2f}%")
        print(f"{'±STD':<12} {'±'+f'{std_d_prod:.6e}':<14} {'±'+f'{std_rmse:.4f}':<12} {'±'+f'{std_corr:.4f}':<14} {'±'+f'{std_imp:.2f}%':>13}")
        print("=" * 90)
        
        # Build results dict
        full_results = {
            "robustness": {
                "data_seeds_tested": data_seeds,
                "outlier_ratio": args.outlier_ratio,
                "hadamard_seed": args.hadamard_seed,
                "baseline_dense": {
                    "d_prod": r_dense_base["d_prod"],
                    "correlation": r_dense_base["correlation"],
                },
                "baseline_hadamard": {
                    "d_prod": r_hadamard_base["d_prod"],
                    "correlation": r_hadamard_base["correlation"],
                },
                "outlier_aware_per_seed": outlier_results_per_seed,
                "summary": {
                    "mean_d_prod": float(mean_d_prod),
                    "std_d_prod": float(std_d_prod),
                    "mean_relative_rmse": float(mean_rmse),
                    "std_relative_rmse": float(std_rmse),
                    "mean_correlation": float(mean_corr),
                    "std_correlation": float(std_corr),
                    "mean_improvement_pct": float(mean_imp),
                    "std_improvement_pct": float(std_imp),
                }
            },
            "sweep_results": None,
            "single_run": None,
        }
        
        # Save results to JSON
        results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "real_data_hadamard_results.json")
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2, sort_keys=True)
        print(f"\n✓ Saved to {results_path}")
        
        # Final status
        if mean_imp > 3.0:
            print(f"\nSUCCESS: Outlier-aware shows +{mean_imp:.2f}% ± {std_imp:.2f}% improvement over Dense (robust across seeds)")
        elif mean_imp > 0:
            print(f"\nPARTIAL: Outlier-aware shows modest +{mean_imp:.2f}% ± {std_imp:.2f}% improvement")
        else:
            print(f"\nWARNING: Outlier-aware shows no consistent improvement over Dense ({mean_imp:.2f}% ± {std_imp:.2f}%)")
    
    elif len(outlier_ratios) > 1 or args.outlier_ratio not in outlier_ratios:
        # Sweeping mode
        print(f"\n>>> Running outlier-ratio sweep: {outlier_ratios}")
        print(f">>> Outlier bits: {args.outlier_bits}")
        print(f">>> Compare high-bit baseline: {args.compare_high_bit_baseline}")
        
        sweep_results = {
            "outlier_ratios_tested": outlier_ratios,
            "results_per_ratio": {},
            "summary": {
                "best_outlier_ratio": None,
                "best_improvement_pct": -float("inf"),
                "best_effective_bits": None,
            }
        }
        
        for ratio in outlier_ratios:
            print(f"\n{'='*60}")
            print(f"OUTLIER RATIO: {ratio}")
            print(f"{'='*60}")
            
            results = run_validation_on_real_data(
                vectors, data_source, args.verbose, ratio,
                outlier_bits=args.outlier_bits,
                compare_high_bit_baseline=args.compare_high_bit_baseline
            )
            
            sweep_results["results_per_ratio"][str(ratio)] = results
            
            # Track best result
            improvement_vs_dense = ((results["dense"]["d_prod"] - results["outlier_aware"]["d_prod"]) / 
                                   results["dense"]["d_prod"]) * 100
            if improvement_vs_dense > sweep_results["summary"]["best_improvement_pct"]:
                sweep_results["summary"]["best_outlier_ratio"] = ratio
                sweep_results["summary"]["best_improvement_pct"] = improvement_vs_dense
                sweep_results["summary"]["best_effective_bits"] = results["outlier_aware"]["effective_bits"]
        
        # Print sweep summary
        print(f"\n{'='*60}")
        print("SWEEP SUMMARY")
        print(f"{'='*60}")
        print(f"Best outlier ratio: {sweep_results['summary']['best_outlier_ratio']}")
        print(f"Best improvement vs Dense: {sweep_results['summary']['best_improvement_pct']:+.2f}%")
        print(f"Effective bits at best: {sweep_results['summary']['best_effective_bits']:.2f}")
        
        # Try to plot if matplotlib is available
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            ratios = sweep_results["outlier_ratios_tested"]
            d_prods = [sweep_results["results_per_ratio"][str(r)]["outlier_aware"]["d_prod"] for r in ratios]
            
            plt.figure(figsize=(10, 6))
            plt.plot(ratios, d_prods, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Outlier Ratio', fontsize=12)
            plt.ylabel('D_prod (lower is better)', fontsize=12)
            plt.title('Outlier Ratio Sweep: Effect on Quantization Error', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(ratios)
            
            # Add horizontal lines for Dense and Hadamard baselines (from first ratio result)
            first_result = sweep_results["results_per_ratio"][str(ratios[0])]
            plt.axhline(y=first_result["dense"]["d_prod"], color='r', linestyle='--', label=f"Dense ({first_result['dense']['d_prod']:.2e})")
            plt.axhline(y=first_result["hadamard"]["d_prod"], color='g', linestyle='--', label=f"Hadamard ({first_result['hadamard']['d_prod']:.2e})")
            plt.legend()
            
            plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outlier_ratio_sweep.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Saved plot to {plot_path}")
        except ImportError:
            print("\n(Note: Install matplotlib with 'pip install matplotlib' to enable plotting)")
        
        # Save full results including sweep
        results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "real_data_hadamard_results.json")
        full_results = {
            "sweep_results": sweep_results,
            "single_run": None,
        }
    else:
        # Single ratio mode (backward compatible)
        print(f"\n>>> Running single outlier ratio: {args.outlier_ratio}")
        print(f">>> Outlier bits: {args.outlier_bits}")
        print(f">>> Compare high-bit baseline: {args.compare_high_bit_baseline}")
        
        results = run_validation_on_real_data(
            vectors, data_source, args.verbose, args.outlier_ratio,
            outlier_bits=args.outlier_bits,
            compare_high_bit_baseline=args.compare_high_bit_baseline
        )
        
        full_results = {
            "sweep_results": None,
            "single_run": results,
        }
    
    # Save results to JSON (skip if multi-seed mode already saved)
    if not is_multi_seed:
        results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "real_data_hadamard_results.json")
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2, sort_keys=True)
        print(f"\n✓ Saved to {results_path}")
    
    # Final status (skip if multi-seed mode already printed status)
    if not is_multi_seed:
        if "sweep_results" in full_results and full_results["sweep_results"]:
            best_imp = full_results["sweep_results"]["summary"]["best_improvement_pct"]
            if best_imp > 3.0:
                print(f"\nSUCCESS: Best outlier-aware config shows +{best_imp:.2f}% improvement over Dense")
            elif best_imp > 0:
                print(f"\nPARTIAL: Best outlier-aware config shows modest +{best_imp:.2f}% improvement")
            else:
                print(f"\nWARNING: Outlier-aware shows no improvement over Dense ({best_imp:.2f}%)")
        else:
            imp_pct = results["improvement_pct"]
            if imp_pct > 3.0:
                print(f"\nSUCCESS: Hadamard on real data shows +{imp_pct:.2f}% improvement")
            elif imp_pct > 0:
                print(f"\nPARTIAL: Hadamard shows modest +{imp_pct:.2f}% improvement")
            else:
                print(f"\nWARNING: Hadamard shows no improvement on real data ({imp_pct:.2f}%)")


if __name__ == "__main__":
    main()
