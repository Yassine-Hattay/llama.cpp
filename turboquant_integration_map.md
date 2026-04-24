# TurboQuant-Hybrid Integration Map for llama.cpp Fork
## 🗂️ Candidate Integration Files
| File Path | Key Structs/Functions | Integration Strategy | Backend Notes | Risk/Complexity |
|---|---|---|---|---|
| `src/models/llama.cpp` | `llm_build_llama::build_qkv()`, `llm_build_llama::build_attn()` | Hook into QKV tensor creation and attention computation paths | CPU/CUDA/Both | Medium |
| `ggml/include/ggml.h` | `enum ggml_op`, `ggml_tensor struct` | Define new GGML_OP_TURBO_QUANT op and tensor metadata fields | Both | Low |
| `ggml/src/ggml.c` | `ggml_compute_forward()`, `ggml_graph_compute()` | Register and dispatch new quantization/dequantization ops | Both | High |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `ggml_cuda_set_tensor_split()`, CUDA kernel dispatchers | Implement CUDA kernels for TurboQuant operations | CUDA | High |
| `ggml/src/ggml-cpu/ggml-cpu.c` | `ggml_compute_forward_*` functions | Implement CPU kernels for quantization pipeline | CPU | High |
| `src/llama-graph.h` | `build_qkv()`, `build_attn_mha()`, `build_attn()` | Intercept KV cache write/read in graph construction | Both | Medium |
| `src/llama-graph.cpp` | `build_attn()`, `build_attn_mha()`, `cpy_k/cpy_v` calls | Insert quantization hooks at cache store/retrieve points | Both | Medium |
| `src/llama-kv-cache.h` | `class llama_kv_cache`, `get_k()`, `get_v()`, `cpy_k()`, `cpy_v()` | Modify KV storage layer to support hybrid quantized format | Both | High |
| `src/llama-kv-cache.cpp` | `ggml_gen_hadamard()`, `ggml_mul_mat_aux()` | Extend existing Hadamard rotation infrastructure for TurboQuant | Both | Medium |
---
## 🔗 Execution Flow Mapping
### KV Tensor Flow from `llama_decode()` → Cache Write → Attention Read → Output
```
llama_decode()
    │
    ▼
llama_context::decode() [in src/llama-context.cpp]
    │
    ▼
llm_build_llama<false>::llm_build_llama() constructor [src/models/llama.cpp:4-153]
    │
    ├── Line 46-47: build_qkv() creates Qcur, Kcur, Vcur tensors
    │       └── Qcur, Kcur, Vcur = [n_embd_head, n_head_kv, n_tokens]
    │
    ├── Line 49-59: ggml_rope_ext() applies rotary embeddings to Qcur, Kcur
    │
    ├── Line 65-71: Optional L2 norm on Qcur/Kcur (Llama4TextL2Norm)
    │
    └── Line 72-74: build_attn() handles KV-cache storage and attention
            │
            ▼
    llm_graph_context::build_attn(llm_graph_input_attn_kv*, ...) [src/llama-graph.cpp:2174-2247]
            │
            ├── Line 2189-2196: Optional rotation matrices applied
            │
            ├── Line 2201-2203: Expand q_cur, v_cur, k_cur into graph
            │
            ├── Line 2212-2213: ★ KV CACHE WRITE INTERCEPT POINT ★
            │       mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il)  ← Quantize here
            │       mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il)  ← Quantize here
            │
            ├── Line 2219-2220: Retrieve cached K, V tensors
            │       k = mctx_cur->get_k(ctx0, il)  ← Dequantize here
            │       v = mctx_cur->get_v(ctx0, il)  ← Dequantize here
            │
            └── Line 2222: build_attn_mha() computes Q @ K^T and softmax @ V
                    │
                    ▼
            llm_graph_context::build_attn_mha() [src/llama-graph.cpp:1932-2065]
                    │
                    ├── Line 1956-1978: Flash Attention path (if enabled)
                    │       ggml_flash_attn_ext() - fused QK^T + softmax + V
                    │
                    └── Line 1998-2054: Standard Attention path
                            ├── Line 1998: kq = ggml_mul_mat(k, q)  ← Q @ K^T
                            ├── Line 2032: ggml_soft_max_ext()
                            └── Line 2042: kqv = ggml_mul_mat(v, kq)  ← softmax @ V
```
### TurboQuant-Hybrid Pipeline Interception Points
| Stage | Location | Action |
|-------|----------|--------|
| **Cache Write (Prefill)** | `src/llama-graph.cpp:2212-2213` | After `cpy_k/cpy_v` calls, insert: outlier detection → Hadamard rotate → 3-bit Lloyd-Max quant |
| **Cache Storage** | `src/llama-kv-cache.cpp` | Modify `kv_layer` struct to hold hybrid format (FP16 outliers + quantized regular) |
| **Cache Read (Decode)** | `src/llama-graph.cpp:2219-2220` | Before `get_k/get_v`, insert: dequantize regular channels → reconstruct full tensor |
| **Attention Compute** | `src/llama-graph.cpp:1998, 2042` | Optionally fuse dequant+matmul into single custom op for efficiency |
---
## ⚠️ Critical Constraints & Gotchas
### 1. Memory Layout & Alignment Requirements
- **GGML Tensor Stride Rules** (`ggml/include/ggml.h:666-669`):
  ```cpp
  nb[0] = ggml_type_size(type)
  nb[1] = nb[0] * (ne[0] / ggml_blck_size(type)) + padding
  nb[i] = nb[i-1] * ne[i-1]
  ```
  - Hybrid quantization breaks contiguous storage assumptions
  - Custom `ggml_type` must define proper `blck_size` and `type_size`
- **Memory Pool Restrictions** (`src/llama-kv-cache.cpp:108-129`):
  - KV cache contexts are pre-allocated per buffer type
  - Hybrid format requires variable-size storage → may need separate memory pool
### 2. Graph Construction Limitations
- **Static Graph Per Model Load** (`src/llama-graph.cpp:2200-2203`):
  ```cpp
  // these nodes are added to the graph together so that they are not reordered
  ggml_build_forward_expand(gf, q_cur);
  ggml_build_forward_expand(gf, v_cur);
  ggml_build_forward_expand(gf, k_cur);
  ```
  - Dynamic quantization decisions must be baked into tensor metadata at graph build time
  - Cannot change quantization parameters between prefill and decode phases without rebuilding graph
### 3. KV Cache Format Assumptions
- **Contiguous FP16 Storage** (`src/llama-kv-cache.h:220-224`):
  ```cpp
  struct kv_layer {
      ggml_tensor * k;
      ggml_tensor * v;
      std::vector<ggml_tensor *> k_stream;
      std::vector<ggml_tensor *> v_stream;
  };
  ```
  - Current implementation assumes uniform `ggml_type` per layer
  - Hybrid format requires either:
    - Custom `ggml_type` with internal structure
    - Separate tensors for outliers + quantized data
### 4. CUDA-Specific Concerns
- **Memory Coalescing** (`ggml/src/ggml-cuda/ggml-cuda.cu`):
  - Structured Hadamard transforms must respect warp-level memory access patterns
  - 3-bit quantization requires careful bit-packing to avoid uncoalesced loads
- **Flash Attention Compatibility** (`src/llama-graph.cpp:1956-1978`):
  ```cpp
  if (use_flash_attn) {
      cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, ...);
  }
  ```
  - Flash attention expects FP16 inputs
  - Must dequantize before flash attn or implement custom `ggml_flash_attn_ext_turboquant`
### 5. Existing Hadamard Infrastructure
- **Pre-computed Rotation Matrices** (`src/llama-kv-cache.h:247-248`):
  ```cpp
  // pre-computed hadamard martrices
  std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;
  ```
  - Already exists for RoPE rotations - can extend for TurboQuant
  - Layer-specific head dimensions stored in `n_embd_head_k_all`, `n_embd_head_v_all`
- **Hadamard Generation Function** (`src/llama-kv-cache.cpp:22-58`):
  ```cpp
  static void ggml_gen_hadamard(ggml_tensor * tensor) {
      // Generates Walsh-Hadamard matrix for power-of-2 dimensions
  }
  ```
  - Can reuse/extend for TurboQuant structured rotation
### 6. Backend Dispatch Architecture
- **Operation Registration** (`ggml/src/ggml.c:970-1078`):
  ```cpp
  static const char * GGML_OP_NAME[GGML_OP_COUNT] = {...};
  ```
  - New `GGML_OP_TURBO_QUANT` must be added to enum and name tables
  - Dispatch happens in `ggml_compute_forward()` switch statement
- **CPU Backend** (`ggml/src/ggml-cpu/ggml-cpu.c`):
  - Forward/backward compute functions required for each new op
  - Multi-threading handled via `ggml_graph_compute_thread()`
- **CUDA Backend** (`ggml/src/ggml-cuda/ggml-cuda.cu`):
  - Kernel launchers follow pattern: `ggml_cuda_op_<op_name>()`
  - Tensor split handling via `ggml_cuda_set_tensor_split()`
### 7. Model Metadata Access
- **Layer Parameters** (`src/llama-graph.h:806-812`):
  ```cpp
  llm_graph_qkv build_qkv(
      const llama_layer & layer,
            ggml_tensor * cur,
                int64_t   n_embd_head,
                int64_t   n_head,
                int64_t   n_head_kv,
                    int   il) const;
  ```
  - `n_embd_head`, `n_head_kv` available for Hadamard matrix sizing
  - Precompute codebooks once per layer during model initialization
---
## 📄 Full File Contents
### File: `src/models/llama.cpp`
```cpp
#include "models.h"
template <bool embed>
llm_build_llama<embed>::llm_build_llama(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);
    ggml_tensor * cur;
    ggml_tensor * inpL;
    inpL = build_inp_embd(model.tok_embd);
    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();
    using inp_attn_type = std::conditional_t<embed, llm_graph_input_attn_no_cache, llm_graph_input_attn_kv>;
    inp_attn_type * inp_attn = nullptr;
    if constexpr (embed) {
        inp_attn = build_attn_inp_no_cache();
    } else {
        inp_attn = build_attn_inp_kv();
    }
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;
        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        // self-attention
        {
            // rope freq factors for llama3; may return nullptr for llama2 and other models
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);
            if (hparams.use_kq_norm) {
                // Llama4TextL2Norm
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
                cb(Qcur, "Qcur_normed", il);
                cb(Kcur, "Kcur_normed", il);
            }
            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            if (model.layers[il].wo_s) {
                cur = ggml_mul(ctx0, cur, model.layers[il].wo_s);
            }
            cb(cur, "attn_out", il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);
        // feed-forward network (non-MoE)
        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);
            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il,
                    nullptr, nullptr,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
            cb(cur, "ffn_moe_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        // input for next layer
        inpL = cur;
    }
    cur = inpL;
    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;
    if constexpr (!embed) {
        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }
    ggml_build_forward_expand(gf, cur);
}
template struct llm_build_llama<false>;
template struct llm_build_llama<true>;
```
---
### File: `ggml/include/ggml.h` (Relevant Sections)
#### Enum ggml_type (Lines 389-433)
```cpp
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // GGML_TYPE_Q4_0_4_8 = 32,
    // GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    // GGML_TYPE_IQ4_NL_4_4 = 36,
    // GGML_TYPE_IQ4_NL_4_8 = 37,
    // GGML_TYPE_IQ4_NL_8_8 = 38,
    GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
    GGML_TYPE_NVFP4   = 40, // NVFP4 (4 blocks, E4M3 scale)
    GGML_TYPE_Q1_0    = 41,
    GGML_TYPE_COUNT   = 42,
};
```
#### Enum ggml_op (Lines 473-581)
```cpp
enum ggml_op {
    GGML_OP_NONE = 0,
    GGML_OP_DUP,
    GGML_OP_ADD,
    GGML_OP_ADD_ID,
    GGML_OP_ADD1,
    GGML_OP_ACC,
    GGML_OP_SUB,
    GGML_OP_MUL,
    GGML_OP_DIV,
    GGML_OP_SQR,
    GGML_OP_SQRT,
    GGML_OP_LOG,
    GGML_OP_SIN,
    GGML_OP_COS,
    GGML_OP_SUM,
    GGML_OP_SUM_ROWS,
    GGML_OP_CUMSUM,
    GGML_OP_MEAN,
    GGML_OP_ARGMAX,
    GGML_OP_COUNT_EQUAL,
    GGML_OP_REPEAT,
    GGML_OP_REPEAT_BACK,
    GGML_OP_CONCAT,
    GGML_OP_SILU_BACK,
    GGML_OP_NORM, // normalize
    GGML_OP_RMS_NORM,
    GGML_OP_RMS_NORM_BACK,
    GGML_OP_GROUP_NORM,
    GGML_OP_L2_NORM,
    GGML_OP_MUL_MAT,
    GGML_OP_MUL_MAT_ID,
    GGML_OP_OUT_PROD,
    GGML_OP_SCALE,
    GGML_OP_SET,
    GGML_OP_CPY,
    GGML_OP_CONT,
    GGML_OP_RESHAPE,
    GGML_OP_VIEW,
    GGML_OP_PERMUTE,
    GGML_OP_TRANSPOSE,
    GGML_OP_GET_ROWS,
    GGML_OP_GET_ROWS_BACK,
    GGML_OP_SET_ROWS,
    GGML_OP_DIAG,
    GGML_OP_DIAG_MASK_INF,
    GGML_OP_DIAG_MASK_ZERO,
    GGML_OP_SOFT_MAX,
    GGML_OP_SOFT_MAX_BACK,
    GGML_OP_ROPE,
    GGML_OP_ROPE_BACK,
    GGML_OP_CLAMP,
    GGML_OP_CONV_TRANSPOSE_1D,
    GGML_OP_IM2COL,
    GGML_OP_IM2COL_BACK,
    GGML_OP_IM2COL_3D,
    GGML_OP_CONV_2D,
    GGML_OP_CONV_3D,
    GGML_OP_CONV_2D_DW,
    GGML_OP_CONV_TRANSPOSE_2D,
    GGML_OP_POOL_1D,
    GGML_OP_POOL_2D,
    GGML_OP_POOL_2D_BACK,
    GGML_OP_UPSCALE,
    GGML_OP_PAD,
    GGML_OP_PAD_REFLECT_1D,
    GGML_OP_ROLL,
    GGML_OP_ARANGE,
    GGML_OP_TIMESTEP_EMBEDDING,
    GGML_OP_ARGSORT,
    GGML_OP_TOP_K,
    GGML_OP_LEAKY_RELU,
    GGML_OP_TRI,
    GGML_OP_FILL,
    GGML_OP_FLASH_ATTN_EXT,
    GGML_OP_FLASH_ATTN_BACK,
    GGML_OP_SSM_CONV,
    GGML_OP_SSM_SCAN,
    GGML_OP_WIN_PART,
    GGML_OP_WIN_UNPART,
    GGML_OP_GET_REL_POS,
    GGML_OP_ADD_REL_POS,
    GGML_OP_RWKV_WKV6,
    GGML_OP_GATED_LINEAR_ATTN,
    GGML_OP_RWKV_WKV7,
    GGML_OP_SOLVE_TRI,
    GGML_OP_GATED_DELTA_NET,
    GGML_OP_UNARY,
    GGML_OP_MAP_CUSTOM1,
    GGML_OP_MAP_CUSTOM2,
    GGML_OP_MAP_CUSTOM3,
    GGML_OP_CUSTOM,
    GGML_OP_CROSS_ENTROPY_LOSS,
    GGML_OP_CROSS_ENTROPY_LOSS_BACK,
    GGML_OP_OPT_STEP_ADAMW,
    GGML_OP_OPT_STEP_SGD,
    GGML_OP_GLU,
    GGML_OP_COUNT,
};
```
#### Struct ggml_tensor (Lines 660-692)
```cpp
// n-dimensional tensor
struct ggml_tensor {
    enum ggml_type type;
    struct ggml_backend_buffer * buffer;
    int64_t ne[GGML_MAX_DIMS]; // number of elements
    size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                               // nb[0] = ggml_type_size(type)
                               // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                               // nb[i] = nb[i-1] * ne[i-1]
    // compute data
    enum ggml_op op;
    // op params - allocated as int32_t for alignment
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t flags;
    struct ggml_tensor * src[GGML_MAX_SRC];
    // source tensor and offset for views
    struct ggml_tensor * view_src;
    size_t               view_offs;
    void * data;
    char name[GGML_MAX_NAME];
    void * extra; // extra things e.g. for ggml-cuda.cu
    char padding[8];
};
```
---
### File: `src/llama-graph.h` (Relevant Sections)
#### build_qkv Declaration (Lines 806-812)
```cpp
// compute Q, K, V projections with optional bias and reshape
// supports both fused wqkv and separate wq/wk/wv paths
llm_graph_qkv build_qkv(
    const llama_layer & layer,
          ggml_tensor * cur,
              int64_t   n_embd_head,
              int64_t   n_head,
              int64_t   n_head_kv,
                  int   il) const;
```
#### build_attn_mha Declaration (Lines 896-905)
```cpp
ggml_tensor * build_attn_mha(
        ggml_tensor * q,       // [n_embd_head_q, n_head_q, n_tokens]
        ggml_tensor * k,       // [n_embd_head_k, n_head_k, n_tokens]
        ggml_tensor * v,       // [n_embd_head_v, n_head_v, n_tokens] (v_trans == false)
        ggml_tensor * kq_b,
        ggml_tensor * kq_mask,
        ggml_tensor * sinks,   // [n_head_q]
        ggml_tensor * v_mla,   // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
              float   kq_scale,
                int   il) const;
```
#### build_attn Declarations (Lines 909-970)
```cpp
llm_graph_input_attn_no_cache * build_attn_inp_no_cache() const;
ggml_tensor * build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
        ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
        ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
        ggml_tensor * kq_b,
        ggml_tensor * sinks, // [n_head_q]
        ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
              float   kq_scale,
                int   il) const;
llm_graph_input_attn_kv * build_attn_inp_kv() const;
ggml_tensor * build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
        ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
        ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
        ggml_tensor * kq_b,
        ggml_tensor * sinks, // [n_head_q]
        ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v] // TODO: remove
              float   kq_scale,
                int   il) const;
```
---
### File: `src/llama-graph.cpp` (Relevant Sections)
#### build_attn_mha Implementation (Lines 1932-2065)
```cpp
ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
               float   kq_scale,
                 int   il) const {
    const bool v_trans = v->nb[1] > v->nb[2];
    // split the batch into streams if needed
    const auto n_stream = k->ne[3];
    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);
    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);
    ggml_tensor * cur;
    const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
    if (use_flash_attn) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");
        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }
        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }
        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }
        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        cb(cur, LLAMA_TENSOR_NAME_FATTN, il);
        ggml_flash_attn_ext_add_sinks(cur, sinks);
        ggml_flash_attn_ext_set_prec (cur, GGML_PREC_F32);
        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);
        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below
            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il);
        }
        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il);
        }
        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il);
        }
        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);
        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }
        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cb(kqv, "kqv", il);
        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il);
        }
        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }
    ggml_build_forward_expand(gf, cur);
    return cur;
}
```
#### build_attn with KV Cache Implementation (Lines 2174-2247)
```cpp
ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla, // TODO: remove
            float     kq_scale,
            int       il) const {
    GGML_ASSERT(v_mla == nullptr);
    if (inp->self_k_rot) {
        q_cur = ggml_mul_mat_aux(ctx0, q_cur, inp->self_k_rot);
        k_cur = ggml_mul_mat_aux(ctx0, k_cur, inp->self_k_rot);
    }
    if (inp->self_v_rot) {
        v_cur = ggml_mul_mat_aux(ctx0, v_cur, inp->self_v_rot);
    }
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);
    const auto * mctx_cur = inp->mctx;
    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();
        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }
    const auto & kq_mask = inp->get_kq_mask();
    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);
    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);
    if (inp->self_v_rot) {
        cur = ggml_mul_mat_aux(ctx0, cur, inp->self_v_rot);
    }
    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }
    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }
    return cur;
}
```
---
### File: `src/llama-kv-cache.h` (Key Structures)
#### kv_layer Struct (Lines 215-225)
```cpp
struct kv_layer {
    // layer index in the model
    // note: can be different from the layer index in the KV cache
    uint32_t il;
    ggml_tensor * k;
    ggml_tensor * v;
    std::vector<ggml_tensor *> k_stream;
    std::vector<ggml_tensor *> v_stream;
};
```
#### Hadamard Matrix Storage (Lines 247-248)
```cpp
// pre-computed hadamard martrices
std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;
```
#### Cache Access Methods (Lines 165-170)
```cpp
// get views of the current state of the cache
ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
// store k_cur and v_cur in the cache based on the provided head location
ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;
```
---
### File: `src/llama-kv-cache.cpp` (Hadamard Infrastructure)
#### ggml_gen_hadamard Function (Lines 20-58)
```cpp
// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);
    const int n = tensor->ne[0];
    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);
    std::vector<float> data_f32;
    float * data = (float *) tensor->data;
    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }
    data[0*n + 0] = 1.0 / sqrtf(n);
    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];
                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }
    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}
```
#### ggml_mul_mat_aux Helper (Lines 60-73)
```cpp
static ggml_tensor * ggml_mul_mat_aux(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * rot) {
    const auto n = rot->ne[0];
    ggml_tensor * res;
    res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    res = ggml_mul_mat   (ctx, rot, res);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
    return res;
}
```
---
## 📌 Summary: Recommended Integration Sequence
1. **Phase 1: Define New ggml_type and ggml_op**
   - Add `GGML_TYPE_TURBO_HYBRID` to `ggml/include/ggml.h`
   - Add `GGML_OP_TURBO_QUANT_FWD` and `GGML_OP_TURBO_DEQUANT` to `enum ggml_op`
   - Update `GGML_TYPE_COUNT`
2. **Phase 2: Implement CPU Kernels**
   - Add `ggml_compute_forward_turbo_quant_fwd()` in `ggml/src/ggml-cpu/ggml-cpu.c`
   - Add `ggml_compute_forward_turbo_dequant()` in same file
   - Register in `ggml_compute_forward()` switch
3. **Phase 3: Extend KV Cache Layer**
   - Modify `kv_layer` struct to support hybrid format metadata
   - Update `cpy_k/cpy_v` to invoke quantization
   - Update `get_k/get_v` to invoke dequantization
4. **Phase 4: CUDA Backend**
   - Implement CUDA kernels in `ggml/src/ggml-cuda/ggml-cuda.cu`
   - Ensure coalesced memory access for 3-bit packed format
5. **Phase 5: Graph Integration**
   - Hook into `build_attn()` at cache write/read points
   - Precompute Hadamard matrices per layer during model init
   - Store outlier indices in tensor `extra` field or separate metadata
EOF
echo "File created successfully with $(wc -l < /workspace/turboquant_integration_map.md) lines"
I've successfully updated the turboquant_integration_map.md file with comprehensive contents including:
Complete integration file table - All 9 candidate files with their key functions, integration strategies, backend notes, and risk levels
Detailed execution flow mapping - A visual diagram showing how KV tensors flow from llama_decode() through cache write/read to attention computation, with specific line numbers and intercept points marked
Critical constraints and gotchas - Seven major sections covering:
Memory layout & alignment requirements
Graph construction limitations
KV cache format assumptions
CUDA-specific concerns
Existing Hadamard infrastructure
Backend dispatch architecture
Model metadata access
Full file contents for all requested files:
src/models/llama.cpp - Complete llm_build_llama template implementation
ggml/include/ggml.h - enum ggml_type, enum ggml_op, and struct ggml_tensor
src/llama-graph.h - build_qkv(), build_attn_mha(), and build_attn() declarations
src/llama-graph.cpp - Full implementations of build_attn_mha() and build_attn() with KV cache
src/llama-kv-cache.h - kv_layer struct, Hadamard storage, and cache access methods
src/llama-kv-cache.cpp - ggml_gen_hadamard() and ggml_mul_mat_aux() helper functions
Recommended integration sequence - Five-phase implementation plan from defining new types through graph integration