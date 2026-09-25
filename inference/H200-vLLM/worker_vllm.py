#!/usr/bin/env python3
"""vLLM worker for the two Llama-4 models on H200 GPUs, running the same stages as run_worker.py
(determinism_run1/2, validation_subsample_v2, full_grid_fixed) through vLLM rather than
transformers.

vLLM computes on the fp8 weights as they are stored: Maverick occupies about 418 GB resident,
close to its 417 GB on-disk footprint, where transformers decompresses every quantized tensor to
bf16 before generation and needs about 803 GB regardless of the hardware underneath.

Everything except the generation mechanism comes from the transformers path unchanged: the prompts,
the image preprocessing (common.preprocess_image, so vLLM receives the same resized JPEG every
other model received rather than the original photograph), the response parsing
(common.parse_response), the row schema (run_worker.make_row) and the resumability rule, which
skips any (photograph, prompt) pair already logged as settled. Only the model loading and
generation differ: vLLM's LLM.chat() in place of transformers' .generate().

One structural difference from run_worker.py. vLLM's model load takes several minutes, so this
script runs the determinism check immediately after loading, as a cheap check before anything
longer starts, rather than in a separate invocation that would pay the load cost again. The full
grid stays a separate, deliberate invocation (--full-grid-fixed) and is not chained on
automatically, so the check's numbers are read before the grid is launched.

Usage:
    python worker_vllm.py --model Llama-4-Maverick-17B-128E-Instruct-FP8 --tensor-parallel-size 4 --stop-after-determinism
    python worker_vllm.py --model Llama-4-Maverick-17B-128E-Instruct-FP8 --tensor-parallel-size 4 --validation-subsample-v2
    python worker_vllm.py --model Llama-4-Maverick-17B-128E-Instruct-FP8 --tensor-parallel-size 4 --full-grid-fixed --batch-size 8
    python worker_vllm.py --model Llama-4-Scout-17B-16E-Instruct --tensor-parallel-size 4 \
        --exp2-source rectified --exp2-limit-images 48 --batch-size 24
        (the base prompts against the rectified photographs on a small subset, to check the
        arrangement before committing to a full grid; run_type exp2_rectified_base, which no other
        stage writes to, so nothing already computed is redone)
    python worker_vllm.py --model Llama-4-Maverick-17B-128E-Instruct-FP8 --tensor-parallel-size 4 \
        --exp2-source rectified --exp2-prompts v1_point_hint,v3_detailed_ecology --batch-size 24
        (the full 1,155-photograph grid restricted to a subset of the prompts, which is how a grid
        is split across several jobs that each have to finish within the hour; see --exp2-prompts
        for why the split follows cost per prompt rather than the number of prompts)
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Root of the local-inference project, which supplies the `experiment1` package
# imported below. Override with MLLM_LOCAL_ROOT if it lives elsewhere.
sys.path.insert(0, os.environ.get("MLLM_LOCAL_ROOT", os.path.expanduser("~/MLLMs-Local")))

from experiment1 import common, config  # noqa: E402
from experiment1.run_worker import load_image_pairs, select_determinism_subsample, make_row  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--tensor-parallel-size", type=int, default=4)
parser.add_argument("--max-model-len", type=int, default=8192)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
parser.add_argument("--stop-after-determinism", action="store_true")
parser.add_argument("--validation-subsample-v2", action="store_true")
parser.add_argument("--full-grid-fixed", action="store_true")
parser.add_argument("--batch-size", type=int, default=1,
                     help="requests submitted per llm.chat() call for full_grid_fixed only -- "
                          "determinism/validation always use 1 for the cleanest reproducibility "
                          "signal, independent of this flag")
parser.add_argument("--limit-images", type=int, default=None)
parser.add_argument("--exp2-source", choices=["masked_gray", "rectified", "original"], default=None,
                     help="run the base prompts (config.PROMPT_IDS) against this image variant "
                          "instead of the stages above. run_type is exp2_{source}_base, the same "
                          "naming the transformers path uses for these variants, so the rows are "
                          "directly comparable. 'original' is the uncorrected photographs on this "
                          "engine, which is the comparison point the other variants need: for "
                          "Scout, full_grid_fixed holds transformers rows on A100 and so is not a "
                          "same-pipeline baseline for them")
parser.add_argument("--exp2-limit-images", type=int, default=None,
                     help="cap the number of images used for --exp2-source, for a bounded "
                          "verification run instead of the full 1,155-image grid")
parser.add_argument("--exp2-offset", type=int, default=0,
                     help="skip this many photographs from the start of the 1,155-photograph list "
                          "before applying --exp2-limit-images, which lets one prompt's work be "
                          "split across several jobs by image range as well as by prompt. Needed "
                          "where a single prompt does not fit in an hour on its own, as "
                          "v1_point_hint does not for Scout at batch_size=32")
parser.add_argument("--exp2-prompts", type=str, default=None,
                     help="comma-separated subset of config.PROMPT_IDS to run for --exp2-source "
                          "(default: all four). Lets a full 1,155-photograph grid be split across "
                          "several jobs that each finish within the hour. Cost per prompt is "
                          "uneven, v3 and v4 running roughly 2 to 7 times slower per photograph "
                          "than v1 and v2 even at batch_size=24, so the split should balance "
                          "measured cost across jobs rather than the number of prompts")
parser.add_argument("--ablation-arms", type=str, default=None,
                     help="comma-separated arm names, or 'all': one-factor-at-a-time sensitivity "
                          "checks on the 100-photograph determinism subsample, "
                          "run_type=ablation_<arm>. Run here rather than on the transformers path "
                          "so that each arm is measured on the engine that served these two "
                          "models. The arms that cap the pixels a transformers image_processor "
                          "keeps have no counterpart here, since vLLM receives a PIL image "
                          "object, and they apply to the Qwen2-VL family in any case.")
args = parser.parse_args()

if args.model not in config.MODELS:
    print(f"Unknown model {args.model!r}. Choices: {config.MODEL_NAMES}", file=sys.stderr)
    sys.exit(2)

if args.exp2_prompts:
    _exp2_prompt_ids = args.exp2_prompts.split(",")
    _bad = [p for p in _exp2_prompt_ids if p not in config.PROMPT_IDS]
    if _bad:
        print(f"Unknown --exp2-prompts entries {_bad}. Choices: {config.PROMPT_IDS}", file=sys.stderr)
        sys.exit(2)
else:
    _exp2_prompt_ids = list(config.PROMPT_IDS)

import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from PIL import Image  # noqa: E402

EXP2_SOURCE_DIRS = {
    "masked_gray": os.path.join(config.PROJECT_ROOT, "data/pics/masked_gray"),
    "rectified": os.path.join(config.PROJECT_ROOT, "data/pics/rectified"),
    "original": config.IMAGES_DIR,
}

# The sensitivity arms that have a vLLM equivalent. Arms that cap the pixels a transformers
# image_processor keeps (common.apply_image_max_pixels) have none: vLLM is handed a PIL image object
# instead (see VLLMWorker._build_conversation), and that cap belongs to the Qwen2-VL family, which
# neither model here is part of.
ABLATION_ARMS = {
    "baseline": ("the settings in config.py, control", {}),
    "image_first": ("content order image before text",
                     {"MESSAGE_ORDER": "image_first"}),
    "tokens_200": ("MAX_NEW_TOKENS lowered to 200", {"MAX_NEW_TOKENS": 200}),
    "no_newline": ("trailing newline stripped from all 4 prompts", {"_STRIP_PROMPT_NEWLINE": True}),
}

_ABLATION_ORIGINAL_PROMPTS = dict(config.PROMPTS)
_ABLATION_ORIGINAL = {k: getattr(config, k) for k in ("MESSAGE_ORDER", "MAX_NEW_TOKENS")}


def apply_ablation_arm(arm_name):
    """Sets config to this arm and returns a dict describing what changed, for the log.

    Every arm starts from the unmodified settings, so arms cannot accumulate on each other.
    VLLMWorker._build_conversation reads config.PROMPTS and config.MESSAGE_ORDER, and _generate
    reads config.MAX_NEW_TOKENS, at call time rather than at construction time, so setting them
    here takes effect on the next run_phase() call."""
    _, overrides = ABLATION_ARMS[arm_name]
    config.PROMPTS.update(_ABLATION_ORIGINAL_PROMPTS)
    for k, v in _ABLATION_ORIGINAL.items():
        setattr(config, k, v)

    applied = {}
    for key, value in overrides.items():
        if key == "_STRIP_PROMPT_NEWLINE":
            for pid in config.PROMPTS:
                config.PROMPTS[pid] = config.PROMPTS[pid].rstrip("\n")
            applied["prompts"] = "trailing newline stripped"
        else:
            setattr(config, key, value)
            applied[key] = value
    return applied


def exp2_preprocess_from_source(image_dir, filename, tag):
    """Preprocess one photograph from a named image variant, into a cache keyed on the variant.

    The masked-gray, rectified and original sets use the same filenames, so a cache keyed on the
    filename alone, as common.preprocess_image()'s is, would return whichever variant reached it
    first and say nothing about the substitution."""
    cache_dir = os.path.join(config.PREPROCESSED_DIR, f"{common.preprocessing_key()}_{tag}")
    cached_path = os.path.join(cache_dir, filename)
    if os.path.exists(cached_path):
        return cached_path
    os.makedirs(cache_dir, exist_ok=True)
    img = common.preprocess_to_pil(os.path.join(image_dir, filename))
    tmp_path = f"{cached_path}.tmp.{os.getpid()}"
    img.save(tmp_path, "JPEG", quality=config.IMAGE_JPEG_QUALITY)
    os.replace(tmp_path, cached_path)
    return cached_path


class VLLMWorker:
    def __init__(self, model_name, tensor_parallel_size, max_model_len, gpu_memory_utilization):
        self.model_name = model_name
        self.model_path = config.MODELS[model_name]["path"]
        self.log_path = os.path.join(
            config.RESULTS_RAW_DIR, f"{config.model_slug(model_name)}.jsonl"
        )
        self.tensor_parallel_size = tensor_parallel_size

        print(f"[{model_name}] loading via vLLM: tensor_parallel_size={tensor_parallel_size} "
              f"max_model_len={max_model_len} gpu_memory_utilization={gpu_memory_utilization}",
              flush=True)
        t0 = time.time()
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=tensor_parallel_size,
            enable_expert_parallel=True,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True,
        )
        print(f"[{model_name}] loaded in {time.time() - t0:.1f}s", flush=True)

        self.all_pairs = load_image_pairs(limit=args.limit_images)
        self.determinism_pairs = select_determinism_subsample(self.all_pairs)
        print(f"[{model_name}] {len(self.all_pairs)} images total, "
              f"{len(self.determinism_pairs)} in the determinism subsample", flush=True)

        existing = common.read_jsonl(self.log_path)
        self.done = {
            (r["image"], r["prompt_id"], r["run_type"])
            for r in existing
            if r.get("error", "") in ("", "JSON_PARSE_FAILURE")
        }
        print(f"[{model_name}] {len(self.done)} triplets already done, skipping those", flush=True)

        logged_metadata = common.read_jsonl(config.RUN_METADATA_PATH)
        self.metadata_done = {(r["model"], r["run_type"]) for r in logged_metadata}

    def _build_conversation(self, image_path, prompt_id):
        prompt_text = config.PROMPTS[prompt_id]
        image = Image.open(image_path)
        text_block = {"type": "text", "text": prompt_text}
        image_block = {"type": "image_pil", "image_pil": image}
        if config.MESSAGE_ORDER == "text_first":
            content = [text_block, image_block]
        else:
            content = [image_block, text_block]
        return [{"role": "user", "content": content}]

    def _preprocess_chunk(self, chunk, prompt_id):
        """CPU-only: build vLLM chat conversations for a chunk (image load/resize + prompt
        assembly). No GPU call here. Meant to run in a background thread while a PREVIOUS
        chunk's llm.chat() is still in flight, so image prep overlaps with GPU generation
        instead of leaving the GPU idle between chunks -- see run_phase(). This matters more,
        not less, at larger batch sizes: preprocessing N images synchronously before a single
        llm.chat() call creates a GPU-idle gap proportional to N unless overlapped."""
        return [
            self._build_conversation(
                common.preprocess_image(os.path.join(config.IMAGES_DIR, fn)), prompt_id
            )
            for fn, _ in chunk
        ]

    def _generate(self, conversations):
        """GPU-bound only: submit already-built conversations to vLLM. Kept separate from
        _preprocess_chunk so the two can be pipelined across chunks in run_phase()."""
        sampling_params = SamplingParams(
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            max_tokens=config.MAX_NEW_TOKENS,
        )
        t0 = time.time()
        outputs = self.llm.chat(conversations, sampling_params=sampling_params, use_tqdm=False)
        latency_ms = (time.time() - t0) * 1000
        raw_texts = [o.outputs[0].text for o in outputs]
        return raw_texts, latency_ms

    def run_chunk(self, chunk, conversations, prompt_id, run_type):
        try:
            raw_texts, latency_ms = self._generate(conversations)
        except Exception as exc:
            print(f"[{self.model_name}] INFERENCE_FAILURE on batch of {len(chunk)} "
                  f"({prompt_id}/{run_type}): {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            return [
                make_row(
                    fn, ref, prompt_id, self.model_name, run_type,
                    None, None, "", None, None, len(chunk), None,
                    f"INFERENCE_FAILURE: {type(exc).__name__}: {str(exc)[:200]}",
                )
                for fn, ref in chunk
            ]

        rows = []
        per_item_latency = latency_ms / max(len(chunk), 1)
        for (fn, ref), raw in zip(chunk, raw_texts):
            veg, conf, ok = common.parse_response(raw)
            rows.append(make_row(
                fn, ref, prompt_id, self.model_name, run_type,
                veg, conf, raw, per_item_latency, None, len(chunk), None,
                "" if ok else "JSON_PARSE_FAILURE",
            ))
        return rows

    def run_phase(self, run_type, pairs, prompt_ids, batch_size=1):
        t_start = common.utc_now_iso()
        phase_had_work = False
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            for prompt_id in prompt_ids:
                pending = [p for p in pairs if (p[0], prompt_id, run_type) not in self.done]
                if not pending:
                    continue
                phase_had_work = True
                chunks = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
                print(f"[{self.model_name}] {run_type}/{prompt_id}: {len(pending)} pending "
                      f"(batch_size={batch_size})", flush=True)

                next_conv_future = executor.submit(self._preprocess_chunk, chunks[0], prompt_id)
                for idx, chunk in enumerate(chunks):
                    conversations = next_conv_future.result()
                    if idx + 1 < len(chunks):
                        # kick off next chunk's CPU preprocessing before this chunk's blocking
                        # GPU call, so the two overlap instead of running back-to-back
                        next_conv_future = executor.submit(
                            self._preprocess_chunk, chunks[idx + 1], prompt_id
                        )
                    rows = self.run_chunk(chunk, conversations, prompt_id, run_type)
                    for row in rows:
                        common.append_jsonl(self.log_path, row)
                        if row["error"] in ("", "JSON_PARSE_FAILURE"):
                            self.done.add((row["image"], row["prompt_id"], row["run_type"]))
                    n_ok = sum(1 for r in rows if r["error"] == "")
                    print(f"[{self.model_name}] {run_type}/{prompt_id}: "
                          f"batch {idx + 1}/{len(chunks)} done ({n_ok}/{len(rows)} ok)", flush=True)
        finally:
            executor.shutdown(wait=False)

        phase_all_done_now = all(
            (fn, prompt_id, run_type) in self.done
            for prompt_id in prompt_ids for fn, _ in pairs
        )
        if not phase_had_work and (self.model_name, run_type) in self.metadata_done:
            pass  # already fully logged in a prior invocation, nothing to do
        elif phase_all_done_now and (self.model_name, run_type) not in self.metadata_done:
            self._write_run_metadata(run_type, t_start, common.utc_now_iso())
        elif not phase_all_done_now:
            print(f"[{self.model_name}] {run_type} not fully complete this invocation "
                  f"(walltime or failures) -- resubmit to continue", flush=True)

    def _write_run_metadata(self, run_type, t_start, t_end):
        row = {
            "model": self.model_name,
            "run_type": run_type,
            "gpu_model": "NVIDIA H200",
            "driver_version": None,
            "cuda_version": None,
            "framework": "vllm",
            "framework_version": vllm.__version__,
            "python_version": sys.version.split()[0],
            "model_path": self.model_path,
            "precision": "fp8",
            "temperature": config.TEMPERATURE,
            "top_p": config.TOP_P,
            "max_tokens": config.MAX_NEW_TOKENS,
            "start_time_utc": t_start,
            "end_time_utc": t_end,
            "num_gpus": self.tensor_parallel_size,
            "device_map": None,
            "max_memory": None,
            "engine": "vllm",
            "tensor_parallel_size": self.tensor_parallel_size,
            "enable_expert_parallel": True,
        }
        common.append_jsonl(config.RUN_METADATA_PATH, row)
        self.metadata_done.add((self.model_name, run_type))
        print(f"[{self.model_name}] wrote run_metadata for {run_type}", flush=True)


def main():
    worker = VLLMWorker(
        args.model, args.tensor_parallel_size, args.max_model_len, args.gpu_memory_utilization
    )

    # The run_type carries the engine (_vllm) rather than being the bare "determinism_run1/2" the
    # transformers path writes. A model with a determinism check on that path writes it to the same
    # log under the same name, and resumability keys on (image, prompt_id, run_type), so an
    # untagged check here would find nothing pending and skip, reporting nothing about whether vLLM
    # decodes deterministically while appearing to have run. Tagging is unconditional, so the
    # distinction holds for every model rather than only the ones that happen to have both.
    worker.run_phase("determinism_run1_vllm", worker.determinism_pairs, ["v2_short"], batch_size=1)
    worker.run_phase("determinism_run2_vllm", worker.determinism_pairs, ["v2_short"], batch_size=1)
    if args.stop_after_determinism:
        print(f"[{args.model}] --stop-after-determinism set, stopping here", flush=True)
        return

    if args.validation_subsample_v2:
        worker.run_phase(
            "validation_subsample_v2", worker.determinism_pairs, config.PROMPT_IDS, batch_size=1
        )
    elif args.full_grid_fixed:
        worker.run_phase(
            "full_grid_fixed", worker.all_pairs, config.PROMPT_IDS, batch_size=args.batch_size
        )
    elif args.exp2_source:
        image_dir = EXP2_SOURCE_DIRS[args.exp2_source]
        run_type = f"exp2_{args.exp2_source}_base"
        _sliced = worker.all_pairs[args.exp2_offset:]
        pairs = _sliced[:args.exp2_limit_images] if args.exp2_limit_images else _sliced

        if args.exp2_source == "original":
            # This variant is the directory (config.IMAGES_DIR) every other stage already reads
            # through the default common.preprocess_image(), so it needs no variant-tagged cache
            # and can reuse whatever that cache already holds. The masked-gray and rectified sets
            # are separate directories sharing the same filenames, and do need their own.
            print(f"[{args.model}] {run_type}: {len(pairs)} images x {_exp2_prompt_ids}, "
                  f"source={image_dir} (default cache), batch_size={args.batch_size}", flush=True)
            worker.run_phase(run_type, pairs, _exp2_prompt_ids, batch_size=args.batch_size)
        else:
            original_preprocess_image = common.preprocess_image

            def tagged_preprocess_image(src_path):
                filename = os.path.basename(src_path)
                return exp2_preprocess_from_source(image_dir, filename, args.exp2_source)

            common.preprocess_image = tagged_preprocess_image
            try:
                print(f"[{args.model}] {run_type}: {len(pairs)} images x {_exp2_prompt_ids}, "
                      f"source={image_dir}, batch_size={args.batch_size}", flush=True)
                worker.run_phase(run_type, pairs, _exp2_prompt_ids, batch_size=args.batch_size)
            finally:
                common.preprocess_image = original_preprocess_image
    elif args.ablation_arms:
        arms = list(ABLATION_ARMS) if args.ablation_arms == "all" else \
            [a.strip() for a in args.ablation_arms.split(",")]
        unknown = [a for a in arms if a not in ABLATION_ARMS]
        if unknown:
            print(f"Unknown ablation arm(s): {unknown}. Known: {list(ABLATION_ARMS)}", file=sys.stderr)
            sys.exit(2)
        n_per_arm = len(worker.determinism_pairs) * len(config.PROMPT_IDS)
        print(f"[{args.model}] ablation: {len(arms)} arms x {n_per_arm} inferences "
              f"= {len(arms) * n_per_arm} total, batch_size={args.batch_size}", flush=True)
        for arm in arms:
            applied = apply_ablation_arm(arm)
            run_type = f"ablation_{arm}"
            print(f"[{args.model}] === {run_type}: {ABLATION_ARMS[arm][0]} "
                  f"(overrides: {applied or 'none (control)'}) ===", flush=True)
            worker.run_phase(run_type, worker.determinism_pairs, config.PROMPT_IDS,
                              batch_size=args.batch_size)

    print(f"[{args.model}] worker finished this invocation.", flush=True)


if __name__ == "__main__":
    main()
