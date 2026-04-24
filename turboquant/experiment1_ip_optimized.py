#!/usr/bin/env python3
# experiment1_ip_optimized.py
"""
Experiment 1: Validate IP-optimized Lloyd-Max codebook for TurboQuant.

This script tests the hypothesis that replacing the standard MSE objective
with a weighted MSE objective (optimized for inner product estimation) 
improves D_prod (inner product distortion) by 5-15%.

Usage:
    python experiment1_ip_optimized.py
    
The script will:
1. Generate baseline (MSE-optimal) and IP-optimized codebooks
2. Compare centroids and MSE values
3. Measure D_prod on synthetic and real embeddings
4. Report improvement percentage
"""

import torch
import numpy as np
import json
from turboquant.codebook import compute_lloyd_max_codebook, get_codebook
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd


def generate_test_embeddings(n_samples: int = 1000, dim: int = 128, seed: int = 42):
    """Generate random unit-norm embeddings for testing."""
    torch.manual_seed(seed)
    x = torch.randn(n_samples, dim)
    x = x / x.norm(dim=-1, keepdim=True)
    return x


def compute_inner_product_error(x: torch.Tensor, x_recon: torch.Tensor, 
                                 y: torch.Tensor = None) -> dict:
    """
    Compute inner product distortion metrics.
    
    Args:
        x: original vectors (n, d)
        x_recon: reconstructed vectors (n, d)
        y: optional query vectors for asymmetric IP test (m, d)
           If None, uses x for symmetric test
    
    Returns:
        dict with D_prod metrics
    """
    if y is None:
        y = x
    
    # True inner products
    ip_true = torch.matmul(x, y.T)
    
    # Reconstructed inner products
    ip_recon = torch.matmul(x_recon, y.T)
    
    # D_prod: mean squared error of inner products
    d_prod = ((ip_true - ip_recon) ** 2).mean().item()
    
    # Relative error
    rel_error = (torch.abs(ip_true - ip_recon) / (torch.abs(ip_true) + 1e-10)).mean().item()
    
    # Correlation
    correlation = torch.corrcoef(torch.stack([ip_true.flatten(), ip_recon.flatten()]))[0, 1].item()
    
    return {
        "d_prod": d_prod,
        "relative_error": rel_error,
        "correlation": correlation,
        "ip_true_mean": ip_true.mean().item(),
        "ip_recon_mean": ip_recon.mean().item(),
    }


def run_experiment1(dim: int = 128, bits: int = 3, n_samples: int = 1000):
    """
    Run Experiment 1: Compare MSE-optimal vs IP-optimized codebooks.
    
    Returns:
        dict with all experimental results
    """
    print("=" * 70)
    print("EXPERIMENT 1: IP-Optimized Lloyd-Max Codebook")
    print(f"Configuration: dim={dim}, bits={bits}, n_samples={n_samples}")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")
    
    # Step 1: Generate and compare codebooks
    print("[1/4] Generating codebooks...")
    cb_mse = compute_lloyd_max_codebook(dim, bits, ip_optimized=False)
    cb_ipopt = compute_lloyd_max_codebook(dim, bits, ip_optimized=True)
    
    print(f"\n  MSE-Optimal Codebook:")
    print(f"    Centroids: {cb_mse['centroids'][:4]}... (showing first 4)")
    print(f"    MSE per coord: {cb_mse['mse_per_coord']:.6e}")
    print(f"    MSE total: {cb_mse['mse_total']:.6f}")
    
    print(f"\n  IP-Optimized Codebook:")
    print(f"    Centroids: {cb_ipopt['centroids'][:4]}... (showing first 4)")
    print(f"    MSE per coord: {cb_ipopt['mse_per_coord']:.6e}")
    print(f"    MSE total: {cb_ipopt['mse_total']:.6f}")
    
    # Compute centroid differences
    centroids_mse = np.array(cb_mse['centroids'])
    centroids_ipopt = np.array(cb_ipopt['centroids'])
    centroid_shift = np.abs(centroids_mse - centroids_ipopt).mean()
    print(f"\n  Mean absolute centroid shift: {centroid_shift:.6f}")
    
    # Step 2: Generate test embeddings
    print("\n[2/4] Generating test embeddings...")
    x_test = generate_test_embeddings(n_samples, dim)
    print(f"  Generated {n_samples} embeddings of dimension {dim}")
    
    # Step 3: Test MSE quantizer
    print("\n[3/4] Testing MSE quantizer (Algorithm 1)...")
    
    # Baseline: MSE-optimal codebook
    tq_mse_baseline = TurboQuantMSE(dim=dim, bits=bits, device=device, ip_optimized=False)
    x_q_baseline = tq_mse_baseline.quantize(x_test.to(device))
    x_recon_baseline = tq_mse_baseline.dequantize(x_q_baseline)
    metrics_baseline = compute_inner_product_error(x_test.to(device), x_recon_baseline)
    
    # IP-optimized codebook
    tq_mse_ipopt = TurboQuantMSE(dim=dim, bits=bits, device=device, ip_optimized=True)
    x_q_ipopt = tq_mse_ipopt.quantize(x_test.to(device))
    x_recon_ipopt = tq_mse_ipopt.dequantize(x_q_ipopt)
    metrics_ipopt = compute_inner_product_error(x_test.to(device), x_recon_ipopt)
    
    print(f"\n  Algorithm 1 (MSE Quantizer) Results:")
    print(f"    Baseline (MSE-optimal):")
    print(f"      D_prod: {metrics_baseline['d_prod']:.6e}")
    print(f"      Relative error: {metrics_baseline['relative_error']:.4f}")
    print(f"      Correlation: {metrics_baseline['correlation']:.4f}")
    
    print(f"    IP-optimized:")
    print(f"      D_prod: {metrics_ipopt['d_prod']:.6e}")
    print(f"      Relative error: {metrics_ipopt['relative_error']:.4f}")
    print(f"      Correlation: {metrics_ipopt['correlation']:.4f}")
    
    d_prod_improvement_mse = (metrics_baseline['d_prod'] - metrics_ipopt['d_prod']) / metrics_baseline['d_prod'] * 100
    
    # Step 4: Test Prod quantizer (Algorithm 2)
    print("\n[4/4] Testing Prod quantizer (Algorithm 2)...")
    
    # Need at least 2 bits for Prod (1 for MSE + 1 for QJL)
    if bits >= 2:
        # Baseline
        tq_prod_baseline = TurboQuantProd(dim=dim, bits=bits, device=device, ip_optimized=False)
        x_q_prod_baseline = tq_prod_baseline.quantize(x_test.to(device))
        x_recon_prod_baseline = tq_prod_baseline.dequantize(x_q_prod_baseline)
        metrics_prod_baseline = compute_inner_product_error(x_test.to(device), x_recon_prod_baseline)
        
        # IP-optimized
        tq_prod_ipopt = TurboQuantProd(dim=dim, bits=bits, device=device, ip_optimized=True)
        x_q_prod_ipopt = tq_prod_ipopt.quantize(x_test.to(device))
        x_recon_prod_ipopt = tq_prod_ipopt.dequantize(x_q_prod_ipopt)
        metrics_prod_ipopt = compute_inner_product_error(x_test.to(device), x_recon_prod_ipopt)
        
        print(f"\n  Algorithm 2 (Prod Quantizer) Results:")
        print(f"    Baseline (MSE-optimal):")
        print(f"      D_prod: {metrics_prod_baseline['d_prod']:.6e}")
        print(f"      Relative error: {metrics_prod_baseline['relative_error']:.4f}")
        print(f"      Correlation: {metrics_prod_baseline['correlation']:.4f}")
        
        print(f"    IP-optimized:")
        print(f"      D_prod: {metrics_prod_ipopt['d_prod']:.6e}")
        print(f"      Relative error: {metrics_prod_ipopt['relative_error']:.4f}")
        print(f"      Correlation: {metrics_prod_ipopt['correlation']:.4f}")
        
        d_prod_improvement_prod = (metrics_prod_baseline['d_prod'] - metrics_prod_ipopt['d_prod']) / metrics_prod_baseline['d_prod'] * 100
    else:
        metrics_prod_baseline = None
        metrics_prod_ipopt = None
        d_prod_improvement_prod = None
        print("\n  Skipping Algorithm 2 (requires bits >= 2)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Codebook Changes:")
    print(f"    Mean centroid shift: {centroid_shift:.6f}")
    print(f"    MSE change: {(cb_ipopt['mse_per_coord'] - cb_mse['mse_per_coord']) / cb_mse['mse_per_coord'] * 100:+.2f}%")
    
    print(f"\n  Algorithm 1 (MSE) D_prod Improvement: {d_prod_improvement_mse:+.2f}%")
    if d_prod_improvement_prod is not None:
        print(f"  Algorithm 2 (Prod) D_prod Improvement: {d_prod_improvement_prod:+.2f}%")
    
    # Expected result check
    expected_range = (5.0, 15.0)
    if d_prod_improvement_mse >= expected_range[0] and d_prod_improvement_mse <= expected_range[1]:
        print(f"\n  ✓ Result within expected range ({expected_range[0]}-{expected_range[1]}%)")
    elif d_prod_improvement_mse > expected_range[1]:
        print(f"\n  ✓✓ Result EXCEEDS expected range! (>+{expected_range[1]}%)")
    elif d_prod_improvement_mse > 0:
        print(f"\n  ✓ Positive improvement but below expected range")
    else:
        print(f"\n  ✗ No improvement observed (hypothesis may be incorrect)")
    
    results = {
        "config": {"dim": dim, "bits": bits, "n_samples": n_samples},
        "codebook": {
            "mse_centroid_shift": float(centroid_shift),
            "mse_change_pct": float((cb_ipopt['mse_per_coord'] - cb_mse['mse_per_coord']) / cb_mse['mse_per_coord'] * 100),
        },
        "algorithm1_mse": {
            "baseline_d_prod": float(metrics_baseline['d_prod']),
            "ipopt_d_prod": float(metrics_ipopt['d_prod']),
            "improvement_pct": float(d_prod_improvement_mse),
        },
        "algorithm2_prod": None,
    }
    
    if metrics_prod_baseline is not None:
        results["algorithm2_prod"] = {
            "baseline_d_prod": float(metrics_prod_baseline['d_prod']),
            "ipopt_d_prod": float(metrics_prod_ipopt['d_prod']),
            "improvement_pct": float(d_prod_improvement_prod),
        }
    
    # Save results
    with open("experiment1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to experiment1_results.json")
    
    return results


if __name__ == "__main__":
    results = run_experiment1()
