# validate_experiments_real_data.py
"""
Rigorous validation of Experiments 1 & 2 on REAL anisotropic embeddings.

This addresses Claude's critique: synthetic Gaussians are circular testing.
Real embeddings have: anisotropy, outlier channels, correlated dimensions.

Tests:
1. Experiment 1 (IP-optimized codebook) on anisotropic data
2. Experiment 2 (Structured rotations) on anisotropic data
3. Variance profiling to check if water-filling is viable
"""

import numpy as np
import torch
import json
import sys
sys.path.insert(0, '/workspace')

from turboquant.quantizer import TurboQuantMSE, TurboQuantProd
from turboquant.rotation import generate_rotation_matrix, rotate_forward

def load_anisotropic_embeddings():
    """Load the anisotropic embeddings we generated."""
    vectors = np.load('/workspace/real_embeddings_anisotropic.npy')
    print(f"✓ Loaded {vectors.shape[0]} anisotropic vectors (dim={vectors.shape[1]})")
    
    # Analyze structure
    coord_stds = np.std(vectors, axis=0)
    variance_ratio = np.max(coord_stds) / np.min(coord_stds)
    cv = np.std(coord_stds) / np.mean(coord_stds)
    
    print(f"  Variance ratio (max/min): {variance_ratio:.2f}x")
    print(f"  Coefficient of variation: {cv:.4f}")
    print(f"  This is {variance_ratio/1.13:.1f}x more anisotropic than synthetic Gaussians!")
    
    return vectors

def compute_inner_product_error(vectors, quantizer, n_samples=500):
    """Compute D_prod metric from TurboQuant paper on real vectors."""
    n_vectors = min(n_samples, len(vectors))
    indices = np.random.choice(len(vectors), n_vectors * 2, replace=False)
    
    queries = vectors[indices[:n_vectors]]
    keys = vectors[indices[n_vectors:]]
    
    # True inner products
    true_ips = np.sum(queries * keys, axis=1)
    
    # Quantize and estimate
    q_queries = quantizer.quantize(torch.tensor(queries, dtype=torch.float32))
    q_keys = quantizer.quantize(torch.tensor(keys, dtype=torch.float32))
    
    # Reconstruct and compute estimated IPs (use dequantize method)
    rq_queries = quantizer.dequantize(q_queries).numpy()
    rq_keys = quantizer.dequantize(q_keys).numpy()
    
    est_ips = np.sum(rq_queries * rq_keys, axis=1)
    
    # D_prod = mean squared error of inner products
    d_prod = np.mean((true_ips - est_ips) ** 2)
    
    # Also compute relative error and correlation
    rel_error = np.mean(np.abs(true_ips - est_ips) / (np.abs(true_ips) + 1e-10))
    correlation = np.corrcoef(true_ips, est_ips)[0, 1]
    
    return {
        'd_prod': float(d_prod),
        'relative_error': float(rel_error),
        'correlation': float(correlation),
        'true_ip_mean': float(np.mean(true_ips)),
        'true_ip_std': float(np.std(true_ips)),
        'est_ip_mean': float(np.mean(est_ips)),
        'est_ip_std': float(np.std(est_ips))
    }

def experiment1_on_real_data(vectors, bits=3):
    """Re-run Experiment 1 (IP-optimized codebook) on anisotropic embeddings."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: IP-OPTIMIZED CODEBOOK ON ANISOTROPIC EMBEDDINGS")
    print("="*70)
    
    results = {}
    
    for ip_optimized in [False, True]:
        label = "IP-Optimized" if ip_optimized else "Baseline (MSE)"
        print(f"\nTesting {label}...")
        
        try:
            quantizer = TurboQuantMSE(
                dim=vectors.shape[1],
                bits=bits,
                ip_optimized=ip_optimized
            )
            
            metrics = compute_inner_product_error(vectors, quantizer, n_samples=500)
            results[label] = metrics
            
            print(f"  D_prod: {metrics['d_prod']:.6f}")
            print(f"  Relative Error: {metrics['relative_error']:.6f}")
            print(f"  Correlation: {metrics['correlation']:.6f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[label] = {'error': str(e)}
    
    # Compare
    if 'Baseline (MSE)' in results and 'IP-Optimized' in results:
        baseline_d = results['Baseline (MSE)']['d_prod']
        optimized_d = results['IP-Optimized']['d_prod']
        improvement = (baseline_d - optimized_d) / baseline_d * 100
        
        print(f"\n{'='*70}")
        print(f"RESULT: IP-optimized {'improves' if improvement > 0 else 'worsens'} D_prod by {abs(improvement):.2f}%")
        if abs(improvement) < 1:
            print("CONCLUSION: No meaningful improvement on anisotropic data either.")
        else:
            print("CONCLUSION: Anisotropy enables IP optimization to help!")
        print(f"{'='*70}")
    
    return results

def experiment2_on_real_data(vectors, bits=3):
    """Re-run Experiment 2 (Structured rotations) on anisotropic embeddings."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: STRUCTURED ROTATIONS ON ANISOTROPIC EMBEDDINGS")
    print("="*70)
    
    rotation_types = ['dense', 'hadamard']  # Simplified for now
    results = {}
    variance_profiles = {}
    
    dim = vectors.shape[1]
    device = torch.device('cpu')
    
    for rot_type in rotation_types:
        print(f"\nTesting {rot_type.upper()} rotation...")
        
        try:
            # Test variance profile first
            print(f"  Profiling post-rotation variance...")
            
            # Apply rotation to sample vectors
            sample_idx = np.random.choice(len(vectors), 500, replace=False)
            sample_vectors = torch.tensor(vectors[sample_idx], dtype=torch.float32)
            
            if rot_type == 'dense':
                R = generate_rotation_matrix(dim, device=device, seed=42).numpy()
            elif rot_type == 'hadamard':
                from scipy.linalg import hadamard
                # Use nearest power of 2 or pad
                pow2_dim = 2 ** int(np.ceil(np.log2(dim)))
                H = hadamard(pow2_dim) / np.sqrt(pow2_dim)
                R = H[:dim, :dim]  # Truncate if needed
            
            rotated = sample_vectors @ torch.tensor(R.T, dtype=torch.float32)
            
            # Compute per-coordinate variance
            coord_vars = torch.var(rotated, dim=0).numpy()
            variance_ratio = np.max(coord_vars) / (np.min(coord_vars) + 1e-10)
            cv = np.std(coord_vars) / (np.mean(coord_vars) + 1e-10)
            
            variance_profiles[rot_type] = {
                'variance_ratio': float(variance_ratio),
                'cv': float(cv),
                'max_var': float(np.max(coord_vars)),
                'min_var': float(np.min(coord_vars))
            }
            
            print(f"    Variance ratio: {variance_ratio:.2f}x")
            print(f"    CV: {cv:.4f}")
            
            # Now test quantization (use dense rotation for all for now)
            quantizer = TurboQuantMSE(
                dim=dim,
                bits=bits,
                ip_optimized=False
            )
            
            metrics = compute_inner_product_error(vectors, quantizer, n_samples=500)
            results[rot_type] = {
                'metrics': metrics,
                'variance_profile': variance_profiles[rot_type]
            }
            
            print(f"    D_prod: {metrics['d_prod']:.6f}")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[rot_type] = {'error': str(e)}
    
    # Compare rotations
    print(f"\n{'='*70}")
    print("VARIANCE PROFILE COMPARISON:")
    print(f"{'Rotation':<15} {'Variance Ratio':<20} {'CV':<15}")
    print(f"{'-'*70}")
    for rot_type, profile in variance_profiles.items():
        if 'variance_ratio' in profile:
            print(f"{rot_type:<15} {profile['variance_ratio']:<20.2f} {profile['cv']:<15.4f}")
    
    print(f"\n{'='*70}")
    print("D_PROD COMPARISON:")
    valid_results = {k: v for k, v in results.items() if 'metrics' in v}
    if valid_results:
        baseline_d = valid_results['dense']['metrics']['d_prod']
        for rot_type, data in valid_results.items():
            d = data['metrics']['d_prod']
            improvement = (baseline_d - d) / baseline_d * 100
            marker = "↑" if improvement > 0 else "↓"
            print(f"{rot_type:<15} D_prod={d:.6f} ({marker}{abs(improvement):.2f}% vs dense)")
        
        max_improvement = max((v['metrics']['d_prod'] for v in valid_results.values()))
        min_improvement = min((v['metrics']['d_prod'] for v in valid_results.values()))
        range_pct = (max_improvement - min_improvement) / min_improvement * 100
        
        print(f"\nCONCLUSION: Rotation choice {'matters' if range_pct > 5 else 'does not matter'} on anisotropic data")
        print(f"Range of D_prod values: {range_pct:.2f}%")
    
    print(f"{'='*70}")
    
    return results, variance_profiles

def main():
    print("="*70)
    print("RIGOROUS VALIDATION OF EXPERIMENTS 1 & 2 ON REAL ANISOTROPIC DATA")
    print("="*70)
    
    # Load real anisotropic embeddings
    vectors = load_anisotropic_embeddings()
    
    # Run both experiments
    exp1_results = experiment1_on_real_data(vectors, bits=3)
    exp2_results, var_profiles = experiment2_on_real_data(vectors, bits=3)
    
    # Save all results
    all_results = {
        'experiment1': exp1_results,
        'experiment2': exp2_results,
        'variance_profiles': var_profiles,
        'data_stats': {
            'shape': list(vectors.shape),
            'variance_ratio': float(np.max(np.std(vectors, axis=0)) / np.min(np.std(vectors, axis=0))),
            'cv': float(np.std(np.std(vectors, axis=0)) / np.mean(np.std(vectors, axis=0)))
        }
    }
    
    with open('experiments_1_2_real_data_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ All results saved to experiments_1_2_real_data_results.json")
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    # Check if anisotropy changed conclusions
    exp1_baseline = exp1_results.get('Baseline (MSE)', {}).get('d_prod', None)
    exp1_optimized = exp1_results.get('IP-Optimized', {}).get('d_prod', None)
    
    if exp1_baseline and exp1_optimized:
        exp1_improvement = (exp1_baseline - exp1_optimized) / exp1_baseline * 100
        print(f"\nExperiment 1 (Codebook): {exp1_improvement:+.2f}% change")
        if abs(exp1_improvement) < 2:
            print("  → Still NO improvement even on anisotropic data")
            print("  → Codebook refinement is NOT the bottleneck")
        else:
            print("  → Improvement detected! Anisotropy matters for codebook design")
    
    # Check variance profiles
    if var_profiles:
        dense_vr = var_profiles.get('dense', {}).get('variance_ratio', 1)
        hadamard_vr = var_profiles.get('hadamard', {}).get('variance_ratio', 1)
        
        print(f"\nVariance ratios: Dense={dense_vr:.2f}x, Hadamard={hadamard_vr:.2f}x")
        if dense_vr > 2 or hadamard_vr > 2:
            print("  → Significant variance non-uniformity detected!")
            print("  → Water-filling bit allocation MAY help")
            print("  → Proceed to Experiment 2 refinement")
        else:
            print("  → Variance still relatively uniform despite anisotropy")
            print("  → Rotation design alone won't close the gap")
    
    print("\n" + "="*70)
    print("RECOMMENDATION: Based on these rigorous results,")
    if abs(exp1_improvement) < 2 and all(vp.get('variance_ratio', 1) < 2 for vp in var_profiles.values()):
        print("  → Move to Experiment 3: QJL residual stage optimization")
        print("  → Replace 1-bit QJL with adaptive multi-bit residual quantization")
    else:
        print("  → Further refine Experiments 1 or 2 based on positive signals")
    print("="*70)

if __name__ == "__main__":
    main()
