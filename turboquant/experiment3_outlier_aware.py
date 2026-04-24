#!/usr/bin/env python3
"""
Validate outlier-aware hybrid quantization against plain Hadamard.
Tests on real_embeddings_anisotropic.npy (dim=384, var_ratio=19x).

Expected: outlier-aware should beat plain Hadamard by 5-15% on real data,
since we stop Hadamard from smearing the 5% outlier channels.
"""
import torch, numpy as np, json, sys, os

def hadamard_rotation(d, seed=1337):
    """Generate QR-orthogonalized Hadamard rotation."""
    n = 1
    while n < d: n *= 2
    def H(s):
        if s == 1: return torch.tensor([[1.0]])
        h = H(s//2)
        return torch.cat([torch.cat([h,h],1), torch.cat([h,-h],1)])
    Hm = H(n)
    rng = torch.Generator(); rng.manual_seed(seed)
    signs = (torch.randint(0,2,(n,),generator=rng).float()*2-1)
    Hm = Hm * signs.unsqueeze(0)
    Q, R = torch.linalg.qr(Hm[:d, :d])
    return Q * torch.sign(torch.diag(R)).unsqueeze(0)

def main():
    # Load data
    vectors = np.load("real_embeddings_anisotropic.npy")
    X_all = torch.tensor(vectors, dtype=torch.float32)
    X_all = X_all / (X_all.norm(dim=1, keepdim=True) + 1e-10)
    X, Y = X_all[:2500], X_all[2500:]
    d = X.shape[1]
    
    # Load codebooks
    with open("turboquant/codebooks/codebook_d384_b3.json") as f:
        cb = json.load(f)
    boundaries = torch.tensor(cb["boundaries"])
    centroids = torch.tensor(cb["centroids"])
    
    # Detect outliers from X variance (unnormalized)
    channel_var = torch.tensor(vectors[:2500]).var(dim=0)
    n_outliers = int(d * 0.05)  # 5%
    out_idx = channel_var.topk(n_outliers).indices.sort().values
    reg_idx = torch.tensor([i for i in range(d) if i not in out_idx.tolist()])
    print(f"Detected {n_outliers} outlier channels out of {d}")
    print(f"Outlier variance ratio: {channel_var[out_idx].mean() / channel_var[reg_idx].mean():.2f}x")
    
    # === Plain Hadamard (all channels) ===
    Pi_full = hadamard_rotation(d)
    X_unit = X / (X.norm(dim=1, keepdim=True) + 1e-10)
    Y_rot_full = X_unit @ Pi_full.T
    idx_full = torch.searchsorted(boundaries[1:-1], Y_rot_full)
    X_recon_full = (centroids[idx_full] @ Pi_full) * X.norm(dim=1, keepdim=True)
    ip_true = (X * Y).sum(dim=1)
    ip_plain = (X_recon_full * Y).sum(dim=1)
    d_prod_plain = ((ip_true - ip_plain)**2).mean().item()
    
    # === Hybrid: Hadamard on regular + pass-through outliers ===
    d_reg = len(reg_idx)
    Pi_reg = hadamard_rotation(d_reg)
    X_norms = X.norm(dim=1, keepdim=True)
    X_unit = X / (X_norms + 1e-10)
    X_reg = X_unit[:, reg_idx]
    X_out = X_unit[:, out_idx]
    Y_rot = X_reg @ Pi_reg.T
    idx = torch.searchsorted(boundaries[1:-1], Y_rot)
    X_reg_hat = centroids[idx] @ Pi_reg
    X_recon_unit = torch.zeros_like(X_unit)
    X_recon_unit[:, reg_idx] = X_reg_hat
    X_recon_unit[:, out_idx] = X_out   # pass-through
    X_recon = X_recon_unit * X_norms
    ip_hybrid = (X_recon * Y).sum(dim=1)
    d_prod_hybrid = ((ip_true - ip_hybrid)**2).mean().item()
    
    # Results
    improvement = (d_prod_plain - d_prod_hybrid) / d_prod_plain * 100
    print(f"\n{'='*60}")
    print(f"Plain Hadamard D_prod:      {d_prod_plain:.6e}")
    print(f"Hybrid (5% outlier) D_prod: {d_prod_hybrid:.6e}")
    print(f"Improvement:                {improvement:+.2f}%")
    print(f"{'='*60}")
    if improvement > 3.0:
        print("✓ SUCCESS: Outlier-aware hybrid shows significant improvement")
        return True
    elif improvement > 0:
        print("△ PARTIAL: Modest improvement detected")
        return True
    else:
        print("✗ WARNING: No improvement - outlier smearing hypothesis may be incorrect")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
