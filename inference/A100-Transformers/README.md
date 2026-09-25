# A100-Transformers/

Code that ran Gemma-3-12B, Gemma-3-27B, Mistral-Small-3.2 and Qwen-2.5, one model per single-GPU
job, on NVIDIA A100-SXM4-80GB, through Hugging Face `transformers`.

## Files

- **`config.py`** holds every constant the runs depend on: model paths and batch sizes (`MODELS`), the
  prompt text (`PROMPTS`), the generation settings (`MAX_NEW_TOKENS = 2048`, greedy decoding), the
  image preprocessing settings (`IMAGE_MAX_SIDE = 1536`, `IMAGE_RESAMPLE_NAME = "BICUBIC"`,
  `IMAGE_JPEG_QUALITY = 90`), the size of the subsample used for the determinism check
  (`DETERMINISM_N = 100`), and the seed (`SEED = 42`).
- **`common.py`** holds the shared logic: image preprocessing and its cache (`preprocess_image`,
  `preprocess_to_pil`), model loading, response parsing (`parse_response`, which delegates to
  `../../parser/fvc_parser.py`), retry with backoff, and GPU power sampling (`PowerSampler`, the
  source of the energy and latency figures).
- **`run_worker.py`** is the driver. One process serves one model on one GPU. It writes one row per
  (photograph, prompt) pair to that model's own log, and skips any pair the log already records as
  finished, so an interrupted job continues where it stopped when resubmitted. `--full-grid-fixed`
  runs the full grid reported in the paper (`run_type=full_grid_fixed`). The module and method
  docstrings describe the other modes it supports: the determinism check, the smaller validation
  subsamples, and the staged runs for the second prompt set.
- **`run_fullgrid_single_model.sbatch`** is the job script for the reported grid: one model, one GPU,
  `--full-grid-fixed`. The four models are independent workers writing to separate logs, so each one
  is submitted on its own. It first checks that the photograph-to-reference join holds exactly 1,155
  rows, then runs the grid.
- **`run_full_grid_fixed_bundled_withdrawn.sbatch`** is a variant that takes four A100s in a single
  allocation and runs the four workers in parallel inside it. It is not the script that collected
  the reported grid; the single-model script above is.

## What this folder does not cover

Response parsing is not specific to this path. `parse_response` here delegates to
`../../parser/fvc_parser.py`, the one parser behind every published estimate, on the local and the
hosted-API side alike: the published values come from running that parser over the stored responses,
not from anything a worker computed in passing. It scans a response for well-formed JSON objects and
takes the last one carrying both a vegetation percentage and a confidence value as numbers,
accepting `vegetation_cover` as well as `vegetation_percent` for the cover field. A response with no
complete object is recorded as missing and is not reconstructed from partial output. A response the
parser cannot resolve is not re-requested here, because greedy decoding would return the same text.

GPU memory and power telemetry (`PowerSampler`, `torch.cuda.max_memory_allocated`) exists only on
this path. The vLLM path (`../H200-vLLM/`) has no equivalent, so Maverick and Scout carry no
peak-memory or power figures.
