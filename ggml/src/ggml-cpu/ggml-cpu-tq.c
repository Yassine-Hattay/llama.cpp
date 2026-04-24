// TurboQuant-Hybrid CPU Implementation
// CPU-only kernels for hybrid KV cache quantization with outlier preservation

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "quants.h"
#include "ggml.h"

#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

// ============================================================================
// Configuration Constants
// ============================================================================

#define TQ_HYBRID_OUTLIER_RATIO 0.08f
#define TQ_HYBRID_BITS_REGULAR 3
#define TQ_HYBRID_CENTROIDS (1 << TQ_HYBRID_BITS_REGULAR)  // 8 centroids
#define TQ_HYBRID_MAX_OUTLIERS 64  // Support up to d=800 with 8% outliers
#define TQ_HYBRID_MAX_HEAD_DIM 512

// ============================================================================
// Precomputed Lloyd-Max Centroids for Beta(191.5, 191.5) Distribution
// Normalized to [-1, 1] range; scaled per-block during quantization
// ============================================================================

static const float ggml_tq_hybrid_centroids[TQ_HYBRID_CENTROIDS] = {
    -0.1879f, -0.1182f, -0.0661f, -0.0213f,
     0.0213f,  0.0661f,  0.1182f,  0.1879f
};

// ============================================================================
// Fast Walsh-Hadamard Transform (FWHT) - Iterative, In-Place
// Reference: https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform
// Complexity: O(d log d) where d is head dimension (must be power-of-2)
// ============================================================================

void ggml_fwht_cpu(float * x, int d) {
    assert(d > 0 && (d & (d - 1)) == 0);  // power of 2
    
    // Iterative Cooley-Tukey style FWHT
    for (int len = 1; len < d; len <<= 1) {
        for (int i = 0; i < d; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j]         = u + v;
                x[i + j + len]   = u - v;
            }
        }
    }
    
    // Normalize: FWHT is orthogonal up to scale sqrt(d)
    const float scale = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) {
        x[i] *= scale;
    }
}

// ============================================================================
// Helper: Unpack 3-bit Index from Packed Buffer
// ============================================================================

static inline int unpack_3bit_index(const uint8_t * packed, int * pos, int * bit_pos) {
    int idx = (packed[*pos] >> *bit_pos) & 0x07;
    *bit_pos += 3;
    if (*bit_pos >= 8) {
        (*pos)++;
        *bit_pos -= 8;
    }
    return idx;
}

// ============================================================================
// Quantize Single Float Row to TQ_HYBRID Format
// Input: x[d] - FP32 row vector
// Output: vy - block_tq_hybrid structure with packed data
// Parameters:
//   - outlier_indices: pre-computed static outlier channel indices for this layer/head
//   - n_outliers: number of static outliers
//   - d: head dimension (must be power-of-2, <= 512)
// ============================================================================

void quantize_row_tq_hybrid(
    const float * x,
    block_tq_hybrid * vy,
    const int16_t * outlier_indices,
    int n_outliers,
    int d
) {
    assert(d <= TQ_HYBRID_MAX_HEAD_DIM);
    assert(n_outliers <= TQ_HYBRID_MAX_OUTLIERS);
    
    // 1. Copy outlier values (FP16) and store indices
    vy->n_outliers = (uint8_t)n_outliers;
    for (int i = 0; i < n_outliers; i++) {
        vy->outlier_idx[i] = outlier_indices[i];
        vy->outlier_val[i] = GGML_FP32_TO_FP16(x[outlier_indices[i]]);
    }
    
    // 2. Extract regular channels and apply FWHT
    float regular_buf[TQ_HYBRID_MAX_HEAD_DIM];
    
    int reg_idx = 0;
    for (int ch = 0; ch < d; ch++) {
        // Skip if this channel is an outlier
        bool is_outlier = false;
        for (int o = 0; o < n_outliers; o++) {
            if (outlier_indices[o] == ch) {
                is_outlier = true;
                break;
            }
        }
        if (!is_outlier) {
            regular_buf[reg_idx++] = x[ch];
        }
    }
    assert(reg_idx == d - n_outliers);
    
    // Apply FWHT to regular channels
    ggml_fwht_cpu(regular_buf, reg_idx);
    
    // 3. Quantize rotated regular channels against codebook
    // Pack 3-bit indices: 8 indices per 3 bytes (since 8*3=24 bits = 3 bytes)
    uint8_t * packed = (uint8_t *)(vy + 1);  // Pack after struct
    int packed_pos = 0;
    uint8_t bit_buffer = 0;
    int bits_in_buffer = 0;
    
    for (int i = 0; i < reg_idx; i++) {
        float val = regular_buf[i];
        // Find nearest centroid (brute-force; 8 centroids is small)
        int best_idx = 0;
        float best_err = fabsf(val - ggml_tq_hybrid_centroids[0]);
        for (int c = 1; c < TQ_HYBRID_CENTROIDS; c++) {
            float err = fabsf(val - ggml_tq_hybrid_centroids[c]);
            if (err < best_err) {
                best_err = err;
                best_idx = c;
            }
        }
        // Pack 3-bit index
        bit_buffer |= (best_idx << bits_in_buffer);
        bits_in_buffer += 3;
        if (bits_in_buffer >= 8) {
            packed[packed_pos++] = bit_buffer;
            bit_buffer >>= 8;
            bits_in_buffer -= 8;
        }
    }
    // Flush remaining bits
    if (bits_in_buffer > 0) {
        packed[packed_pos++] = bit_buffer;
    }
}

// ============================================================================
// Dequantize TQ_HYBRID Block to FP32 (for Verification/Testing)
// Note: For attention, we don't inverse-FWHT here; we rotate Query instead
// ============================================================================

void dequantize_row_tq_hybrid(
    const block_tq_hybrid * vx,
    float * y,
    int d
) {
    // 1. Restore outliers
    for (int i = 0; i < vx->n_outliers; i++) {
        y[vx->outlier_idx[i]] = GGML_FP16_TO_FP32(vx->outlier_val[i]);
    }
    
    // 2. Dequantize regular channels
    const uint8_t * packed = (const uint8_t *)(vx + 1);
    int packed_pos = 0, bit_pos = 0;
    
    int reg_idx = 0;
    for (int ch = 0; ch < d; ch++) {
        // Skip outliers
        bool is_outlier = false;
        for (int o = 0; o < vx->n_outliers; o++) {
            if (vx->outlier_idx[o] == ch) {
                is_outlier = true;
                break;
            }
        }
        if (is_outlier) continue;
        
        // Unpack centroid index
        int centroid_idx = unpack_3bit_index(packed, &packed_pos, &bit_pos);
        
        // Note: For attention, we don't inverse-FWHT here; we rotate Query instead
        // This is just for roundtrip testing
        y[ch] = ggml_tq_hybrid_centroids[centroid_idx];
        reg_idx++;
    }
}

// ============================================================================
// Compute Attention Score: Q_query (FP32) @ K_stored (TQ_HYBRID)
// Implements the "Rotate Query" trick:
//   score = (H @ Q_query)^T @ dequantize(K_stored.regular) 
//         + Q_query[outliers]^T @ K_stored.outlier_vals
// This avoids dequantizing + rotating the entire KV cache
// ============================================================================

float ggml_compute_tq_hybrid_score(
    const float * q_query,           // [d] FP32 query vector
    const block_tq_hybrid * k_block, // Stored quantized key block
    const int16_t * outlier_indices, // Static outlier indices for this layer/head
    int n_outliers,
    int d                            // Head dimension (power-of-2)
) {
    float score = 0.0f;
    
    // 1. Rotate query by Hadamard (in-place on a copy)
    float q_rot[TQ_HYBRID_MAX_HEAD_DIM];
    assert(d <= TQ_HYBRID_MAX_HEAD_DIM);
    memcpy(q_rot, q_query, d * sizeof(float));
    ggml_fwht_cpu(q_rot, d);
    
    // 2. Compute dot product with regular (quantized) channels
    const uint8_t * packed = (const uint8_t *)(k_block + 1);
    int packed_pos = 0, bit_pos = 0;
    
    int reg_idx = 0;
    for (int ch = 0; ch < d; ch++) {
        // Skip outliers (handled separately)
        bool is_outlier = false;
        for (int o = 0; o < n_outliers; o++) {
            if (outlier_indices[o] == ch) {
                is_outlier = true;
                break;
            }
        }
        if (is_outlier) continue;
        
        // Unpack centroid index
        int centroid_idx = unpack_3bit_index(packed, &packed_pos, &bit_pos);
        
        // Dot product in rotated space: q_rot[reg_idx] * centroid
        score += q_rot[reg_idx] * ggml_tq_hybrid_centroids[centroid_idx];
        reg_idx++;
    }
    
    // 3. Add contribution from outlier channels (no rotation needed)
    for (int i = 0; i < k_block->n_outliers; i++) {
        int ch = k_block->outlier_idx[i];
        float k_val = GGML_FP16_TO_FP32(k_block->outlier_val[i]);
        score += q_query[ch] * k_val;  // Original query, not rotated
    }
    
    return score;
}

// ============================================================================
// GGML Compute Forward Function for TQ_DEQUANT_MAT Operation
// dst = src1^T @ src0, where src0 is TQ_HYBRID (KV cache), src1 is FP32 (Query)
// ============================================================================

static void ggml_compute_forward_tq_dequant_mat(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst
) {
    const struct ggml_tensor * src0 = dst->src[0];  // KV cache: [d, n_kv, n_head, n_stream]
    const struct ggml_tensor * src1 = dst->src[1];  // Query:    [d, n_tokens, n_head, n_stream]
    
    GGML_TENSOR_BINARY_OP_LOCALS;  // Expands ne0, ne1, nb0, nb1, etc.
    
    const int ith = params->ith;
    const int nth = params->nth;
    
    // Extract layer-specific outlier indices from op_params
    // op_params layout: [int16_t indices...][int32_t n_outliers]
    const int16_t * outlier_indices = (const int16_t *)src0->op_params;
    const int n_outliers = ((const int32_t *)src0->op_params)[32];  // Stored at offset 32
    
    // Iterate over output tokens (src1 columns)
    for (int i1 = ith; i1 < ne1; i1 += nth) {
        const float * q = (const float *)((char *)src1->data + i1 * nb1);
        float * s = (float *)((char *)dst->data + i1 * nb0);  // Output scores
        
        for (int i0 = 0; i0 < ne0; i0++) {
            const block_tq_hybrid * k = (const block_tq_hybrid *)
                ((char *)src0->data + i0 * nb0);
            
            s[i0] = ggml_compute_tq_hybrid_score(
                q, k, outlier_indices, n_outliers, ne00  // ne00 = head_dim
            );
        }
    }
}

// ============================================================================
// Type Traits Registration for TQ_HYBRID
// ============================================================================

const struct ggml_type_traits_cpu * ggml_get_type_traits_tq_hybrid(void) {
    static struct ggml_type_traits_cpu traits = {
        .from_float       = (ggml_from_float_t)quantize_row_tq_hybrid,
        .vec_dot          = NULL,  // Custom matmul via TQ_DEQUANT_MAT
        .vec_dot_type     = GGML_TYPE_F32,
        .nrows            = 1,
    };
    return &traits;
}
