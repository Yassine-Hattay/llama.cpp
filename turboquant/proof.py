#!/usr/bin/env python3
# proof.py
"""
TurboQuant definitive proof. Two separate subprocesses:
  1. Baseline vLLM
  2. TurboQuant + free_kv_cache
Hard numbers side by side.
"""
import os, sys, subprocess, json, argparse

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TP = int(os.environ.get("TP", "1"))
GPU_MEM = float(os.environ.get("GPU_MEM", "0.7"))  # Increased default from 0.5 to 0.9
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
GPUS = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
PYTHON = sys.executable

TQ_ROTATION = os.environ.get("TQ_ROTATION", "dense")  # default to dense
TQ_OUTLIER_RATIO = float(os.environ.get("TQ_OUTLIER_RATIO", "0.08"))
TQ_OUTLIER_BITS = float(os.environ.get("TQ_OUTLIER_BITS", "16.0"))


def run_phase(name, script):
    path = f"/tmp/tq_{name}.py"
    with open(path, "w") as f:
        f.write(script)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPUS
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__)) + ":" + env.get("PYTHONPATH", "")

    
    # Debug logging before subprocess call
    print(f"[DEBUG] Running: {path} with env CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[DEBUG] Model: {MODEL}, TP: {TP}, GPU_MEM: {GPU_MEM}, MAX_MODEL_LEN: {MAX_MODEL_LEN}", flush=True)
    
    # Increased timeout from 600 to 1800 seconds (30 minutes) for first run
    r = subprocess.run([PYTHON, path], capture_output=True, text=True, env=env, timeout=1800)
    
    if r.returncode != 0:
        print(f"=== {name} FAILED ===")
        # Print stderr for debugging
        print(f"[ERROR] stderr:\n{r.stderr}", flush=True)
        # Find the actual error
        for line in r.stderr.split("\n"):
            if "Error" in line or "error" in line:
                print(f"  {line.strip()}")
        return None
    for line in reversed(r.stdout.strip().split("\n")):
        try:
            return json.loads(line)
        except:
            continue
    return None


BASELINE = f'''
import os, json, subprocess, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

def main():
    import sys
    from vllm import LLM, SamplingParams
    
    print("[BASELINE] Starting...", flush=True)
    print(f"[BASELINE] Loading model: {{{{MODEL}}}}", flush=True)
    
    llm = LLM(
        model="{MODEL}", dtype="bfloat16",
        gpu_memory_utilization={GPU_MEM},
        max_model_len={MAX_MODEL_LEN},
        tensor_parallel_size={TP},
        trust_remote_code=True, max_num_seqs=1,
        enforce_eager=True,  # Add this to reduce compilation overhead
    )
    print("[BASELINE] Model loaded", flush=True)
    
    blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
    print(f"[BASELINE] Allocated {{blocks}} KV blocks", flush=True)

    r = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    vram = [int(l.split(",")[1].strip()) for l in r.stdout.strip().split("\\n") if l.strip()]

    print("[BASELINE] Generating...", flush=True)
    t0 = time.perf_counter()
    out = llm.generate(["Explain KV cache compression in LLM inference."],
        SamplingParams(temperature=0, max_tokens=64))
    t1 = time.perf_counter()
    print(f"[BASELINE] Generation took {{t1-t0:.2f}}s", flush=True)

    r2 = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    vram2 = [int(l.split(",")[1].strip()) for l in r2.stdout.strip().split("\\n") if l.strip()]

    print(json.dumps({{"blocks": blocks, "vram_load": vram, "vram_gen": vram2,
        "text": out[0].outputs[0].text[:100], "elapsed": round(t1-t0, 2)}}))

if __name__ == "__main__":
    main()
'''

def _get_tq_script(rotation_type: str, outlier_ratio: float, outlier_bits: float):
    return f'''
import os, json, subprocess, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
def main():
    import sys
    from vllm import LLM, SamplingParams
    print("[TQ] Starting...", flush=True)
    print(f"[TQ] Loading model: {MODEL}", flush=True)
    llm = LLM(
        model="{MODEL}", dtype="bfloat16",
        gpu_memory_utilization={GPU_MEM},
        max_model_len={MAX_MODEL_LEN},
        tensor_parallel_size={TP},
        trust_remote_code=True, max_num_seqs=1,
        enforce_eager=True,
    )
    print("[TQ] Model loaded", flush=True)
    blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
    print(f"[TQ] Allocated {{blocks}} KV blocks", flush=True)
    engine = llm.llm_engine
    core = getattr(engine, "engine_core", engine)
    inner = getattr(core, "engine_core", core)
    executor = inner.model_executor
    print("[TQ] Installing TurboQuant hooks...", flush=True)
    def _install(worker):
        from turboquant.vllm_attn_backend import install_turboquant_hooks, MODE_ACTIVE
        return len(install_turboquant_hooks(worker.model_runner, key_bits=3, value_bits=2,
        buffer_size=128, mode=MODE_ACTIVE, rotation_type="{rotation_type}",
        outlier_ratio={outlier_ratio}, outlier_bits={outlier_bits}))
    hooks = executor.collective_rpc(_install)
    print(f"[TQ] Hooks installed on {{hooks[0]}} layers", flush=True)
    print("[TQ] Generating...", flush=True)
    t0 = time.perf_counter()
    out = llm.generate(["Explain KV cache compression in LLM inference."],
    SamplingParams(temperature=0, max_tokens=64))
    t1 = time.perf_counter()
    print(f"[TQ] Generation took {{t1-t0:.2f}}s", flush=True)
    r = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],
    capture_output=True, text=True)
    vram_gen = [int(l.split(",")[1].strip()) for l in r.stdout.strip().split("\\n") if l.strip()]
    print("[TQ] Freeing KV cache...", flush=True)
    def _free(worker):
        from turboquant.vllm_attn_backend import free_kv_cache
        return free_kv_cache(worker.model_runner)
    freed = executor.collective_rpc(_free)
    print(f"[TQ] Freed {{sum(freed)/1e6:.0f}} MB total", flush=True)
    r2 = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],
    capture_output=True, text=True)
    vram_freed = [int(l.split(",")[1].strip()) for l in r2.stdout.strip().split("\\n") if l.strip()]
    print(json.dumps({{"blocks": blocks, "hooks": hooks[0], "vram_gen": vram_gen,
    "vram_freed": vram_freed, "freed_bytes": freed,
    "text": out[0].outputs[0].text[:100], "rotation_type": "{rotation_type}",
    "elapsed": round(t1-t0, 2)}}))
if __name__ == "__main__":
    main()
'''


def main():
    print(f"Model: {MODEL}")
    print(f"TP={TP}, GPU_MEM={GPU_MEM}, MAX_MODEL_LEN={MAX_MODEL_LEN}")
    print(f"GPUs: {GPUS}")
    print(f"TQ_ROTATION: {TQ_ROTATION}")
    print()

    # Handle --validate-real-data flag
    if hasattr(main, 'validate_real_data') and main.validate_real_data:
        print(">>> Running real data validation...", flush=True)
        import subprocess
        result = subprocess.run([sys.executable, "experiment_hadamard_real.py"], 
                               capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Validation failed: {result.stderr}")
        return

    print(">>> Phase 1: Baseline ...", flush=True)
    bl = run_phase("baseline", BASELINE)
    if not bl:
        print(">>> Baseline phase failed. Check stderr above.", flush=True)
        return
    print(f">>> Baseline complete. Blocks: {bl.get('blocks', 'N/A')}", flush=True)

    print(">>> Phase 2: TurboQuant ...", flush=True)
    tq = run_phase("tq", _get_tq_script(TQ_ROTATION, TQ_OUTLIER_RATIO, TQ_OUTLIER_BITS))
    if not tq:
        print(">>> TurboQuant phase failed. Check stderr above.", flush=True)
        return
    print(f">>> TurboQuant complete. Hooks: {tq.get('hooks', 'N/A')}", flush=True)

    n = len(GPUS.split(","))
    bl_v = bl["vram_gen"][:n]
    tq_v = tq["vram_gen"][:n]
    tq_f = tq["vram_freed"][:n]

    freed_total = sum(tq["freed_bytes"])
    freed_per = tq["freed_bytes"][0]

    block_size = 784  # Qwen3.5-27B: attention block aligned to mamba
    bl_tokens = bl["blocks"] * block_size
    # Extra capacity from freed KV cache
    # full_attn: 16 layers, kv_heads=1/gpu, head_dim=256, bf16=2, K+V=2
    bytes_per_block_full = 2 * 1 * 256 * 2 * block_size * tq["hooks"]
    extra_blocks = int(freed_per / max(bytes_per_block_full, 1))
    new_tokens = bl_tokens + extra_blocks * block_size

    print()
    print("=" * 70)
    print(f"  MODEL: {MODEL}")
    print(f"  TP={TP}, max_model_len={MAX_MODEL_LEN}, gpu_mem={GPU_MEM}")
    print(f"  Rotation: {tq.get('rotation_type', 'N/A')}")
    print()
    print(f"  BASELINE (vanilla vLLM)")
    print(f"    KV cache blocks:         {bl['blocks']}")
    print(f"    Max tokens:              {bl_tokens:,}")
    print(f"    VRAM/GPU after gen:      {bl_v} MB")
    print(f"    Generation time:         {bl.get('elapsed', 'N/A')}s")
    print()
    print(f"  TURBOQUANT (3-bit key, 2-bit value, {tq['hooks']} full_attn layers)")
    print(f"    KV cache blocks:         {tq['blocks']}  (same initial alloc)")
    print(f"    VRAM/GPU after gen:      {tq_v} MB")
    print(f"    VRAM/GPU after free:     {tq_f} MB")
    print(f"    Tensor freed/GPU:        {freed_per/1e6:.0f} MB")
    print(f"    Total tensor freed:      {freed_total/1e6:.0f} MB ({freed_total/1e9:.1f} GB)")
    print(f"    Generation time:         {tq.get('elapsed', 'N/A')}s")
    print()
    print(f"  RESULT")
    print(f"    KV VRAM saved/GPU:       {freed_per/1e6:.0f} MB")
    print(f"    Extra blocks possible:   {extra_blocks}")
    print(f"    Baseline capacity:       {bl_tokens:,} tokens")
    print(f"    With TQ capacity:        {new_tokens:,} tokens")
    print(f"    Improvement:             {new_tokens/bl_tokens:.2f}x context length")
    print()
    print(f"  OUTPUT COMPARISON")
    print(f"    Baseline: {bl['text']}")
    print(f"    TQ:       {tq['text']}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TurboQuant definitive proof benchmark")
    parser.add_argument("--rotation-type", type=str, default="dense", 
                       choices=["dense", "hadamard"],
                       help="Rotation type for TurboQuant (default: dense)")
    parser.add_argument("--outlier-ratio", type=float, default=0.08,
                       help="Fraction of channels to treat as outliers (default: 0.08)")
    parser.add_argument("--outlier-bits", type=float, default=16.0,
                       help="Bit-width for outliers: 16=FP16 pass-through (default: 16.0)")
    parser.add_argument("--model", type=str, default=None,
                       help="Override MODEL env var")
    parser.add_argument("--tp", type=int, default=None,
                       help="Override TP env var")
    parser.add_argument("--gpu-mem", type=float, default=None,
                       help="Override GPU_MEM env var (0.0-1.0)")
    parser.add_argument("--max-model-len", type=int, default=None,
                       help="Override MAX_MODEL_LEN env var")
    parser.add_argument("--validate-real-data", action="store_true",
                       help="Run Hadamard validation on real embeddings and exit")
    args = parser.parse_args()
    
    # Apply CLI overrides
    if args.model:
        MODEL = args.model
    if args.tp:
        TP = args.tp
    if args.gpu_mem:
        GPU_MEM = args.gpu_mem
    if args.max_model_len:
        MAX_MODEL_LEN = args.max_model_len
    
    # Update the global TQ_ROTATION for the f-string
    TQ_ROTATION = args.rotation_type
    TQ_OUTLIER_RATIO = args.outlier_ratio
    TQ_OUTLIER_BITS = args.outlier_bits
    
    # Set validate_real_data flag
    main.validate_real_data = args.validate_real_data
    
    main()