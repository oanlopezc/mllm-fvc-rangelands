# H200-vLLM/

Code that ran Llama-4-Maverick and Llama-4-Scout through vLLM on NVIDIA H200 GPUs.

## Why vLLM

vLLM runs Maverick's fp8 checkpoint directly and serves both models with tensor parallelism across
the GPUs of one node, which the `transformers` path does not.

## Files

- **`worker_vllm.py`** is the driver. It uses `../A100-Transformers/config.py`,
  `../A100-Transformers/common.py`, and `load_image_pairs`, `select_determinism_subsample` and
  `make_row` from `../A100-Transformers/run_worker.py` unchanged, so the prompts, the image
  preprocessing, the parser and the resumability logic are the same as on that path. Only the
  generation mechanism differs: vLLM's `LLM.chat()` in place of `transformers`' `.generate()`. The
  script is not standalone, since it needs the `experiment1` package from `../A100-Transformers/`
  importable; set `PROJECT_ROOT` to the directory that contains it.
- **`run_maverick_vllm_fullgrid.sbatch`** is the job that produced Maverick's reported run,
  `full_grid_fixed`, at batch size 1.
- **`run_scout_vllm_exp2_full.sbatch`** is the job that produced Scout's runs on the three image
  variants, split into slices that each finish within an hour. Scout's rows for the unmodified
  photographs carry `run_type=exp2_original_base`; the other five models' carry `full_grid_fixed`.

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
