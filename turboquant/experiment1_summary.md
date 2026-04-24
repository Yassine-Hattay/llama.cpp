# Experiment 1 Results Summary

## FINDING: Hypothesis NOT Validated ✗

**The hypothesis that weighted MSE improves D_prod was falsified.**

---

## Key Observations

### 1. CENTROID SHIFTS ARE MINISCULE (5×10⁻⁶ mean absolute shift)
- Despite applying x² weighting, the Lloyd-Max centroids barely move
- This is because the Beta(α,α) distribution for high d (d=128) is 
  extremely concentrated near zero (~N(0, 1/d))
- The tails have such low probability mass that even heavy weighting
  doesn't significantly change the optimal centroid positions

### 2. MSE INCREASES BY 12.6% WITH IP-OPTIMIZED WEIGHTING
- This is expected: we're optimizing a DIFFERENT objective
- Standard MSE: 2.65×10⁻⁴ per coordinate
- Weighted MSE: 2.99×10⁻⁴ per coordinate (+12.6%)
- The codebook is sacrificing MSE performance to reduce IP variance

### 3. D_PROD REMAINS ESSENTIALLY UNCHANGED (< 0.01% change)
- Algorithm 1 (MSE): **-0.007%** (statistically insignificant)
- Algorithm 2 (Prod): **+0.004%** (statistically insignificant)

---

## Root Cause Analysis

The fundamental issue is that for scalar quantization of i.i.d. coordinates,
the Lloyd-Max algorithm ALREADY produces the optimal solution for ANY 
per-coordinate weighted MSE objective when:
  - (a) coordinates are independent
  - (b) weights depend only on the coordinate value (not on other coordinates)

**The reason:** For separable objectives E[Σ_j w_j(X_j)·(X_j - c_j)²], the 
optimal c_j for each cell is still the conditional mean within that cell.
The weight affects WHICH partition is optimal, but for fixed Voronoi cells
(midpoint boundaries), the centroid update rule doesn't change.

---

## What This Means for TurboQuant

Claude's and Gemini's synthesis identified the **RIGHT insight** (optimize for IP,
not MSE), but the **WRONG mechanism** for achieving it through scalar quantization.

**The 2.7× gap between TurboQuant and theoretical bounds CANNOT be closed by:**
- ✗ Changing the Lloyd-Max objective function alone
- ✗ Using weighted MSE with coordinate-dependent weights
  
---

## Promising Alternatives (from prior art analysis)

Based on the paper summaries provided:

### 1. STRUCTURED ROTATIONS (PolarQuant, GSR)
- Paper 19: Hadamard rotation alone accounts for 98% of quality improvement
- Paper 18: Sequency-ordered Walsh blocks outperform standard Hadamard
- **This suggests the ROTATION design matters far more than codebook design**
   
### 2. ANISOTROPIC WEIGHTING (Guo et al. 2019)
- Paper 4/17 already decomposes IP error into parallel/orthogonal components
- But this requires knowing the query distribution (NOT data-oblivious)
   
### 3. OPTIMAL SCALAR QUANTIZATION (Pilanci et al.)
- Paper 6 uses conditional second moments for matrix multiplication
- This is the closest prior work to our attempted approach

---

## Recommended Next Steps

Given these results, we should **PIVOT** to:

### Experiment 2: Structured Rotations

Replace TurboQuant's dense random rotation Π with:
- Hadamard rotation (O(d log d) instead of O(d²))
- Sequency-ordered Walsh blocks (GSR-style)
  
Measure:
- Does post-rotation variance become non-uniform?
- Can water-filling bit allocation help if variance is non-uniform?
- Does this match/exceed dense rotation quality?

This aligns with PolarQuant's finding that rotation design dominates codebook
design at moderate bit widths (Q3-Q5).

---

## Conclusion

Experiment 1 successfully **falsified** the hypothesis that weighted MSE Lloyd-Max
improves inner product estimation in TurboQuant. The centroid shifts are too
small to matter because:

1. High-dimensional Beta distribution is tightly concentrated near zero
2. Scalar quantization with fixed Voronoi boundaries has limited flexibility
3. The true bottleneck is rotation design, not codebook optimization

**This negative result is valuable:** it rules out an entire class of approaches
and redirects effort toward structured rotations (Experiment 2).

| Metric | Value |
|--------|-------|
| Time invested | ~2 hours |
| Result | Hypothesis falsified ✓ |
| Next action | Proceed to Experiment 2 (structured rotations) |
