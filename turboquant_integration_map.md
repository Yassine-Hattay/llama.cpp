# TurboQuant-Hybrid Integration Map for llama.cpp

## 🗂️ Candidate Integration Files

| File Path | Key Structs/Functions | Integration Strategy | Backend Notes | Risk/Complexity |
|---|---|---|---|---|
| `src/models/llama.cpp` | `llm_build_llama::build_qkv()`, `llm_build_llama::build_attn()` | Hook into QKV tensor creation and attention computation paths | CPU/CUDA/Both | Medium |
| `src/ggml/ggml.h` | `enum ggml_op`, `ggml_tensor` struct | Define new `GGML_OP_TURBO_QUANT` op and tensor metadata fields | Both | Low |
| `src/ggml/ggml.c` | `ggml_compute_forward()`, `ggml_graph_compute()` | Register and dispatch new quantization/dequantization ops | Both | High |
| `src/ggml/src/ggml-cuda.cu` | `ggml_cuda_set_tensor_split()`, CUDA kernel dispatchers | Implement CUDA kernels for TurboQuant operations | CUDA | High |
| `src/ggml/src/ggml-cpu/ggml-cpu-ops.c` | `ggml_compute_forward_*` functions | Implement CPU kernels for quantization pipeline | CPU | High |

## 🔗 Execution Flow Mapping

KV tensors flow through the following path:

1. `llama_decode()` - Entry point for inference
2. `llm_build_llama<false>::llm_build_llama()` constructor - Graph construction begins
3. `build_qkv()` - Creates Kcur/Vcur tensors for current token
4. `build_attn()` - Handles:
   - KV-cache storage (write operation)
   - KV-cache retrieval (read operation) 
   - Q@K^T attention score computation
   - Softmax @ V weighted sum

**Interception Points:**
- **Cache Write:** Intercept in `build_attn()` after Kcur/Vcur creation, before storing to KV cache
- **Cache Read:** Intercept in `build_attn()` when retrieving historical K/V tensors
- **Attention Computation:** Replace standard matmul with fused TurboQuant dequantize + matmul op

## ⚠️ Critical Constraints & Gotchas

### Memory Layout Constraints
- Tensor alignment requirements in ggml backend memory pools may conflict with variable-size quantized storage
- KV-cache memory layout assumes contiguous fp16 storage - hybrid quantization requires custom tensor accessors

### Graph Construction Limitations
- Graph construction is static per model load
- Dynamic quantization decisions must be baked into tensor metadata at graph build time
- Cannot change quantization parameters during inference without rebuilding graph

### CUDA-Specific Concerns
- Strict memory coalescing requirements that structured Hadamard transforms must respect
- Custom kernels must follow ggml's tensor split and stream management patterns
- Shared memory limits may constrain outlier channel storage strategy

### General Architecture Notes
- Must preserve data-oblivious property (no per-sample calibration)
- Inner-product accuracy critical for attention mechanism
- Both CPU and CUDA backends must produce numerically consistent results
- Hadamard matrix precomputation should occur once per layer during model initialization
