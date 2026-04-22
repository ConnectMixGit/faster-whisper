"""
GPU auto-configuration for optimal concurrency estimation.

Runs at worker startup before any model is loaded. Queries GPU hardware
via NVML and estimates the max number of concurrent whisper inference
calls the GPU can handle based on VRAM capacity, compute units, and
known model characteristics.
"""

# Known model VRAM footprints in GB (CTranslate2, float16)
MODEL_VRAM_GB = {
    "tiny": 0.15,
    "base": 0.3,
    "small": 0.5,
    "medium": 1.0,
    "large-v1": 3.0,
    "large-v2": 3.0,
    "large-v3": 3.0,
    "distil-large-v2": 1.5,
    "distil-large-v3": 1.5,
    "turbo": 1.6,
    "turbo-deepdml": 1.6,
}

# Per-concurrent-inference scratch memory in GB (encoder activations, buffers)
PER_INFERENCE_SCRATCH_GB = {
    "tiny": 0.1,
    "base": 0.15,
    "small": 0.2,
    "medium": 0.3,
    "large-v1": 0.5,
    "large-v2": 0.5,
    "large-v3": 0.5,
    "distil-large-v2": 0.4,
    "distil-large-v3": 0.4,
    "turbo": 0.35,
    "turbo-deepdml": 0.35,
}

# CUDA context + driver overhead in GB
CUDA_OVERHEAD_GB = 0.8

# SM count baselines for compute-based capping.
# Empirically, an RTX 4090 (128 SMs) handles ~6-8 concurrent large-v3 calls well.
# Scale linearly from that reference point.
REFERENCE_SM_COUNT = 128
REFERENCE_CONCURRENCY = 8

# Absolute bounds
MIN_CONCURRENCY = 1
MAX_CONCURRENCY = 32


def estimate_max_concurrency(target_model="turbo"):
    """
    Estimate the optimal number of concurrent inference calls for a given
    GPU and whisper model. Returns an integer concurrency value.

    This runs in <100ms with no model loading -- NVML queries only.
    Always returns a safe default (6) if anything goes wrong.
    """
    default = 6

    try:
        return _estimate(target_model, default)
    except Exception as e:
        _log(f"GPU auto-config failed ({e}), defaulting to concurrency={default}")
        return default


def _estimate(target_model, default):
    """Inner estimation logic. Raises on any failure so the caller can fallback."""
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception as e:
        _log(f"NVML init failed ({e}), defaulting to concurrency={default}")
        return default

    try:
        device_count = pynvml.nvmlDeviceGetCount()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8")

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_vram_gb = mem_info.total / (1024 ** 3)
        free_vram_gb = mem_info.free / (1024 ** 3)

        try:
            sm_count = pynvml.nvmlDeviceGetNumGpuCores(handle)
        except Exception:
            sm_count = None

        pynvml.nvmlShutdown()
    except Exception as e:
        _log(f"NVML query failed ({e}), defaulting to concurrency={default}")
        return default

    model_vram = MODEL_VRAM_GB.get(target_model, 1.6)
    scratch_per_call = PER_INFERENCE_SCRATCH_GB.get(target_model, 0.35)

    # VRAM-based estimate
    usable_vram = total_vram_gb - CUDA_OVERHEAD_GB - model_vram
    vram_concurrency = max(1, int(usable_vram / scratch_per_call))

    # Compute-based estimate (scale from reference GPU)
    if sm_count and sm_count > 0:
        compute_concurrency = max(1, int(
            REFERENCE_CONCURRENCY * (sm_count / REFERENCE_SM_COUNT)
        ))
    else:
        compute_concurrency = vram_concurrency

    # Take the lower of the two -- no point queuing more than either can handle
    optimal = min(vram_concurrency, compute_concurrency)
    optimal = max(MIN_CONCURRENCY, min(MAX_CONCURRENCY, optimal))

    _log_gpu_config(
        gpu_name=gpu_name,
        device_count=device_count,
        total_vram_gb=total_vram_gb,
        free_vram_gb=free_vram_gb,
        sm_count=sm_count,
        target_model=target_model,
        model_vram_gb=model_vram,
        scratch_per_call_gb=scratch_per_call,
        vram_concurrency=vram_concurrency,
        compute_concurrency=compute_concurrency,
        optimal=optimal,
    )

    return optimal


def _log_gpu_config(
    gpu_name,
    device_count,
    total_vram_gb,
    free_vram_gb,
    sm_count,
    target_model,
    model_vram_gb,
    scratch_per_call_gb,
    vram_concurrency,
    compute_concurrency,
    optimal,
):
    """Print GPU config and concurrency estimation to worker logs."""
    print("=" * 60)
    print("GPU AUTO-CONFIGURATION")
    print("=" * 60)
    print(f"  GPU detected     : {gpu_name}")
    print(f"  GPU count        : {device_count}")
    print(f"  Total VRAM       : {total_vram_gb:.1f} GB")
    print(f"  Free VRAM        : {free_vram_gb:.1f} GB")
    if sm_count:
        print(f"  CUDA cores       : {sm_count}")
    else:
        print(f"  CUDA cores       : (unavailable)")
    print("-" * 60)
    print(f"  Target model     : {target_model}")
    print(f"  Model VRAM       : {model_vram_gb:.1f} GB")
    print(f"  Scratch/call     : {scratch_per_call_gb:.2f} GB")
    print(f"  CUDA overhead    : {CUDA_OVERHEAD_GB:.1f} GB")
    print("-" * 60)
    print(f"  VRAM-based max   : {vram_concurrency}")
    print(f"  Compute-based max: {compute_concurrency}")
    print(f"  >> CONCURRENCY   : {optimal}")
    print("=" * 60)


def _log(msg):
    print(f"[gpu_config] {msg}")
