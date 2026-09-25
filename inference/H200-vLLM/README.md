# H200-vLLM/

Code that ran Llama-4-Maverick and Llama-4-Scout through vLLM on NVIDIA H200 GPUs.

## Why these two models are not on the `transformers` path

`transformers` decompresses a quantized checkpoint to bf16 before generation, whatever the hardware
supports. Maverick's fp8 checkpoint is 416.8 GB on disk and about 803 GB resident on that path,
which is more than any allocation available here can hold. vLLM's fused low-bit kernels consume the
fp8 weights directly, at roughly 418 GB resident, close to the on-disk footprint, and that is what
makes running the model possible at all.

Scout does fit on the `transformers` path, since it ships at bf16, but generates at roughly
37,900 ms per image there against roughly 24 ms per image under vLLM. That gap decides whether the
image variants and the sensitivity checks are affordable for it at all.

## Files

- **`worker_vllm.py`** is the driver. It uses `../A100-Transformers/config.py`,
  `../A100-Transformers/common.py`, and `load_image_pairs`, `select_determinism_subsample` and
  `make_row` from `../A100-Transformers/run_worker.py` unchanged, so the prompts, the image
  preprocessing, the parser and the resumability logic are the same as on that path. Only the
  generation mechanism differs: vLLM's `LLM.chat()` in place of `transformers`' `.generate()`. The
  script is not standalone, since it needs the `experiment1` package from `../A100-Transformers/`
  importable; it looks for that package under `$MLLM_LOCAL_ROOT`.
- **`run_maverick_vllm_fullgrid.sbatch`** is the job that produced Maverick's reported run,
  `full_grid_fixed`, at batch size 1.
- **`run_scout_vllm_exp2_full.sbatch`** is the job that produced Scout's reported base run,
  `exp2_original_base`. Scout is the one model whose reported base rows do not sit under
  `full_grid_fixed`: for Scout that run_type holds the `transformers` rows on A100, which are not
  reported, so the two must not be read interchangeably.

## Where the determinism check runs

vLLM's model load takes several minutes, much longer than `transformers`', so this script runs the
determinism check immediately after loading, as a cheap check before anything longer starts, rather
than in a separate invocation that would pay the load cost again. The full grid stays a separate,
deliberate invocation (`--full-grid-fixed`) and is not chained on automatically, so the check's
numbers are read before the grid is launched.

## No GPU telemetry

This path has no equivalent to `PowerSampler` or `torch.cuda.max_memory_allocated`. vLLM runs its
own generation loop internally, so there is no single `.generate()` call here to bracket with GPU
statistics the way the `transformers` path does. Maverick's and Scout's reported runs therefore
carry no peak-memory or power figures.
