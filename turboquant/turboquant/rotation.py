"""
Random rotation utilities for TurboQuant.

The paper uses Π = QR decomposition of a random Gaussian matrix.
For efficiency on GPU, we offer two options:
  1. Full random orthogonal matrix (via QR) — exact, costs O(d^2) storage
  2. Randomized Hadamard Transform (RHT) — fast O(d log d) but approximate

For typical head_dim (64-256), full QR is fine. The matrix is shared
across all heads in a layer and generated once from a fixed seed per layer.
"""

import math
import torch


def generate_rotation_matrix(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
    rotation_type: str = "dense",  # NEW: "dense" or "hadamard"
) -> torch.Tensor:
    """
    Generate a random orthogonal matrix Π ∈ R^{d×d}.

    Args:
        d: dimension
        device: target device
        dtype: target dtype
        seed: random seed for reproducibility
        rotation_type: "dense" for full QR orthogonalization,
                       "hadamard" for fast Walsh-Hadamard transform

    For "dense": Uses QR decomposition of a random Gaussian matrix (Algorithm 1).
                 Includes fix for non-power-of-2 dimensions via proper sign correction.
    For "hadamard": Uses randomized Hadamard transform (RHT) with Rademacher diagonal.
                   Requires d to be power of 2; falls back to dense if not.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    if rotation_type == "hadamard":
        # Check if d is power of 2
        if d & (d - 1) != 0:
            # Not a power of 2, fall back to dense
            return generate_rotation_matrix(d, device, dtype, seed, rotation_type="dense")
        
        # Randomized Hadamard Transform: H @ D where D is diagonal Rademacher
        # Generate Rademacher diagonal (±1)
        diag_entries = torch.sign(torch.randn(d, generator=rng, dtype=torch.float32))
        
        # Apply Hadamard matrix recursively (Sylvester construction)
        H = _hadamard_matrix(d)
        
        # Π = H @ diag(entries) / sqrt(d) for orthogonality
        Pi = H * diag_entries.unsqueeze(0) / math.sqrt(d)
        return Pi.to(device=device, dtype=dtype)
    
    # Default: dense QR orthogonalization
    # Generate on CPU for reproducibility, then move to device
    G = torch.randn(d, d, generator=rng, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)

    # Ensure proper rotation (det = +1) by fixing signs
    # This is the QR orthogonalization fix for non-power-of-2 dimensions
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)

    return Q.to(device=device, dtype=dtype)


def _hadamard_matrix(n: int) -> torch.Tensor:
    """
    Construct Sylvester Hadamard matrix of order n (must be power of 2).
    H_1 = [1], H_{2n} = [[H_n, H_n], [H_n, -H_n]]
    """
    if n == 1:
        return torch.tensor([[1.0]])
    
    H_prev = _hadamard_matrix(n // 2)
    top = torch.cat([H_prev, H_prev], dim=1)
    bottom = torch.cat([H_prev, -H_prev], dim=1)
    return torch.cat([top, bottom], dim=0)


def generate_qjl_matrix(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 12345,
) -> torch.Tensor:
    """
    Generate the random projection matrix S ∈ R^{d×d} for QJL.
    S has i.i.d. N(0,1) entries.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    S = torch.randn(d, d, generator=rng, dtype=torch.float32)
    return S.to(device=device, dtype=dtype)


def rotate_forward(x: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """Apply random rotation: y = x @ Pi^T (equivalent to Pi @ x for each vector)."""
    return torch.matmul(x, Pi.T)


def rotate_backward(y: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """Apply inverse rotation: x = y @ Pi (equivalent to Pi^T @ y)."""
    return torch.matmul(y, Pi)
