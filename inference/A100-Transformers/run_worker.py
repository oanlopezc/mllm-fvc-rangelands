#!/usr/bin/env python3
"""One process serves one model on one or more GPUs. It runs determinism_run1, then
determinism_run2, then full_grid, skipping any (photograph, prompt) pair this model's own log
already records as finished, which means either a response the parser resolved or a response
recorded as missing (JSON_PARSE_FAILURE). Both are settled: greedy decoding would return the same
text on a second attempt, so neither is worth re-requesting. The job is therefore safe to kill and
resubmit at any point.

Usage:
    python -m experiment1.run_worker --model Qwen2.5-VL-32B-Instruct [--gpu 0] [--limit-images N]

--gpu takes exactly one index and overrides config.MODELS[model] entirely, including for
multi-GPU models -- there's no CLI form for "give me GPUs 2 and 3". Omit --gpu to use whatever
config.MODELS[model] specifies (a single "gpu" index, or a "gpus" list for models too large for
one card, e.g. Llama-4-Scout).
"""
import argparse
import csv
import os
import random
import sys
import time

# CUDA_VISIBLE_DEVICES must be set before torch is imported anywhere (including transitively),
# so GPU resolution happens first, ahead of every other import.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--model", required=True)
_parser.add_argument("--gpu", type=int, default=None)
_early_args, _ = _parser.parse_known_args()

from experiment1 import config  # noqa: E402

if _early_args.model not in config.MODELS:
    print(f"Unknown model {_early_args.model!r}. Choices: {config.MODEL_NAMES}", file=sys.stderr)
    sys.exit(2)

_gpus = config.resolve_gpus(_early_args.model, _early_args.gpu)
_gpu = _gpus[0]  # kept as the single-value physical index for print messages / metadata lookups
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in _gpus)

import torch  # noqa: E402
import transformers  # noqa: E402

from experiment1 import common  # noqa: E402


def load_image_pairs(limit=None):
    with open(config.JOINED_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pairs = [(row["filename"], float(row["reference"])) for row in reader]
    pairs.sort(key=lambda p: p[0])
    if limit:
        pairs = pairs[:limit]
    return pairs


def select_determinism_subsample(all_pairs):
    idx = sorted(random.Random(config.SEED).sample(range(len(all_pairs)), config.DETERMINISM_N))
    return [all_pairs[i] for i in idx]


def load_exp2_split():
    """{image: split} from config.EXP2_SPLIT_CSV ('selection' or 'confirmation').

    Raises if the split file does not exist. The split is built once, outside this script, and is
    never regenerated here: regenerating it would move photographs between the two sets and leave
    the held-out design measuring something other than what earlier runs measured."""
    if not os.path.exists(config.EXP2_SPLIT_CSV):
        raise FileNotFoundError(
            f"{config.EXP2_SPLIT_CSV} not found -- run "
            f"`python -m experiment1.build_exp2_split` first to build the split."
        )
    with open(config.EXP2_SPLIT_CSV, newline="", encoding="utf-8") as f:
        return {r["image"]: r["split"] for r in csv.DictReader(f)}


def select_exp2_pilot_subsample(selection_pairs):
    """100 photographs drawn from the selection split only, stratified by cover bin.

    A fresh draw rather than select_determinism_subsample's seeded indices: those index the full
    1,155-photograph list, where this draws from the smaller selection-split list, so the same seed
    against a different population would not reproduce the same selection and there is nothing to
    be gained by pretending it would."""
    with open(config.JOINED_CSV, newline="", encoding="utf-8") as f:
        ref_by_image = {r["filename"]: float(r["reference"]) for r in csv.DictReader(f)}

    by_bin = {}
    for fn, ref in selection_pairs:
        bin_label = None
        for lo, hi in zip(config.EXP2_FVC_BIN_EDGES[:-1], config.EXP2_FVC_BIN_EDGES[1:]):
            if lo <= ref < hi or (hi == config.EXP2_FVC_BIN_EDGES[-1] and ref == hi):
                bin_label = f"{lo}-{hi}"
                break
        by_bin.setdefault(bin_label, []).append((fn, ref))

    rng = random.Random(config.EXP2_SPLIT_SEED)
    n_bins = len(by_bin)
    per_bin_n = config.EXP2_PILOT_N // n_bins
    remainder = config.EXP2_PILOT_N - per_bin_n * n_bins
    out = []
    for i, (bin_label, members) in enumerate(sorted(by_bin.items())):
        members = sorted(members)
        rng.shuffle(members)
        take = per_bin_n + (1 if i < remainder else 0)  # spread the remainder across the first
                                                          # few bins rather than dumping it in one
        out.extend(members[:min(take, len(members))])
    out.sort()
    return out


def make_row(image, reference, prompt_id, model_name, run_type, veg, conf, raw_text,
             latency_ms, peak_mem_mb, batch_size, power_w, error):
    return {
        "image": image,
        "prompt_id": prompt_id,
        "model": model_name,
        "model_local_path": config.MODELS[model_name]["path"],
        "vegetation_percent": veg,
        "confidence": conf,
        "raw_text": raw_text,
        "latency_ms": latency_ms,
        "error": error,
        "reference": reference,
        "run_type": run_type,
        "peak_gpu_mem_mb": peak_mem_mb,
        "batch_size": batch_size,
        "seed": config.SEED,
        "gpu_power_draw_w_mean": power_w,
    }


class Worker:
    def __init__(self, model_name, physical_gpus, batch_size, limit_images=None):
        self.model_name = model_name
        # `physical_gpus` may be a single index or a list of them; normalize to a list so
        # everything below has one shape to deal with.
        self.physical_gpus = [physical_gpus] if isinstance(physical_gpus, int) else physical_gpus
        self.batch_size = batch_size
        self.model_path = config.MODELS[model_name]["path"]
        self.log_path = os.path.join(
            config.RESULTS_RAW_DIR, f"{config.model_slug(model_name)}.jsonl"
        )
        device_map = config.MODELS[model_name].get("device_map")
        max_memory = config.MODELS[model_name].get("max_memory")

        print(f"[{model_name}] loading model from {self.model_path}, device_map="
              f"{device_map or {'': 0}} max_memory={max_memory} "
              f"(physical GPUs {self.physical_gpus})", flush=True)
        # config.IMAGE_MAX_PIXELS is None, so this is a no-op unless that setting is changed. The
        # cap is offered rather than applied: whether to limit the pixels a model's own image
        # processor keeps changes what the run measures, which is a decision about the experiment
        # and not something a worker should apply on its own.
        self.model, self.processor = common.load_model(
            self.model_path, image_max_pixels=config.IMAGE_MAX_PIXELS,
            device_map=device_map, max_memory=max_memory,
        )

        self.all_pairs = load_image_pairs(limit=limit_images)
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

    def _attempt(self, chunk, prompt_id):
        _t_pre0 = time.time()
        conversations = [
            common.build_conversation(
                config.PROMPTS[prompt_id], common.preprocess_image(os.path.join(config.IMAGES_DIR, fn))
            )
            for fn, _ in chunk
        ]
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to("cuda:0")
        _t_pre1 = time.time()
        # latency_ms below brackets model.generate() only, so it does not see the cost of building
        # the batch. That cost is not the same for every model: apply_chat_template does the image
        # encoding itself for MistralCommonBackend (see common.load_model's docstring), where for
        # every other model's AutoProcessor it is a fast tokenization step. This print reports the
        # figure when it is large enough to matter, so a model held up on CPU is not read as a
        # model that is simply slow to generate.
        _pre_ms = (_t_pre1 - _t_pre0) * 1000.0
        if _pre_ms > 1000:
            print(f"[_attempt] pre-generate (build_conversation+apply_chat_template) took "
                  f"{_pre_ms:.0f}ms for a chunk of {len(chunk)} ({prompt_id})", flush=True)

        # torch.cuda.device_count() reflects CUDA_VISIBLE_DEVICES, so this is one device for every
        # single-GPU model and covers every shard of a model placed with device_map="auto".
        n_devices = torch.cuda.device_count()
        for _d in range(n_devices):
            torch.cuda.reset_peak_memory_stats(_d)
        sampler = common.PowerSampler(self.physical_gpus)
        t0 = time.time()
        with sampler, torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=False,
            )
        latency_ms = (time.time() - t0) * 1000.0
        peak_mem_mb = sum(torch.cuda.max_memory_allocated(_d) for _d in range(n_devices)) / 1e6
        power_w = sampler.mean_watts()

        gen_tokens = out[:, inputs["input_ids"].shape[1]:]
        raw_texts = self.processor.batch_decode(gen_tokens, skip_special_tokens=True)
        return raw_texts, latency_ms, peak_mem_mb, power_w

    def run_chunk(self, chunk, prompt_id, run_type):
        result, exc = common.retry_with_backoff(lambda: self._attempt(chunk, prompt_id), max_retries=5)
        if exc is None:
            raw_texts, latency_ms, peak_mem_mb, power_w = result
            rows = []
            for (fn, ref), raw in zip(chunk, raw_texts):
                veg, conf, ok = common.parse_response(raw)
                rows.append(make_row(
                    fn, ref, prompt_id, self.model_name, run_type,
                    veg, conf, raw, latency_ms, peak_mem_mb, len(chunk), power_w,
                    "" if ok else "JSON_PARSE_FAILURE",
                ))
            return rows

        if len(chunk) == 1:
            fn, ref = chunk[0]
            print(f"[{self.model_name}] INFERENCE_FAILURE on {fn}/{prompt_id}/{run_type}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return [make_row(
                fn, ref, prompt_id, self.model_name, run_type,
                None, None, "", None, None, 1, None,
                f"INFERENCE_FAILURE: {type(exc).__name__}: {str(exc)[:200]}",
            )]

        mid = len(chunk) // 2
        return self.run_chunk(chunk[:mid], prompt_id, run_type) + \
            self.run_chunk(chunk[mid:], prompt_id, run_type)

    def run_phase(self, run_type, pairs, prompt_ids):
        phase_had_work = False
        phase_all_done_now = True
        t_start = common.utc_now_iso()
        any_processed = False

        for prompt_id in prompt_ids:
            pending = [p for p in pairs if (p[0], prompt_id, run_type) not in self.done]
            if not pending:
                continue
            phase_had_work = True
            print(f"[{self.model_name}] {run_type}/{prompt_id}: {len(pending)} pending "
                  f"(batch_size={self.batch_size})", flush=True)
            for i in range(0, len(pending), self.batch_size):
                chunk = pending[i:i + self.batch_size]
                rows = self.run_chunk(chunk, prompt_id, run_type)
                any_processed = True
                for row in rows:
                    common.append_jsonl(self.log_path, row)
                    if row["error"] in ("", "JSON_PARSE_FAILURE"):
                        self.done.add((row["image"], row["prompt_id"], row["run_type"]))
                n_ok = sum(1 for r in rows if r["error"] == "")
                print(f"[{self.model_name}] {run_type}/{prompt_id}: "
                      f"batch {i // self.batch_size + 1} done "
                      f"({n_ok}/{len(rows)} ok)", flush=True)

        # Did every (image, prompt_id) in this phase end up answered or recorded as missing?
        for prompt_id in prompt_ids:
            for fn, _ in pairs:
                if (fn, prompt_id, run_type) not in self.done:
                    phase_all_done_now = False

        if phase_all_done_now and (self.model_name, run_type) not in self.metadata_done:
            t_end = common.utc_now_iso()
            self._write_run_metadata(run_type, t_start, t_end)
        elif not phase_had_work and (self.model_name, run_type) in self.metadata_done:
            pass  # already fully logged in a prior submission, nothing to do
        elif not phase_all_done_now:
            print(f"[{self.model_name}] {run_type} not fully complete this invocation "
                  f"(walltime or failures) -- resubmit to continue", flush=True)

    def _write_run_metadata(self, run_type, t_start, t_end):
        try:
            import pynvml

            # The first GPU stands for all of them: the nodes used here are homogeneous, so every
            # card reports the same name and driver, and this metadata row has one field for it.
            h = pynvml.nvmlDeviceGetHandleByIndex(self.physical_gpus[0])
            gpu_name = pynvml.nvmlDeviceGetName(h)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode()
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode()
        except Exception:
            gpu_name, driver_version = None, None

        row = {
            "model": self.model_name,
            "run_type": run_type,
            "gpu_model": gpu_name,
            "driver_version": driver_version,
            "cuda_version": torch.version.cuda,
            "framework": "transformers",
            "framework_version": transformers.__version__,
            "python_version": sys.version.split()[0],
            "model_path": self.model_path,
            # Most checkpoints here run at bf16. A quantized checkpoint may override this only
            # once its representation at run time has been measured on the GPU architecture in
            # use, since a checkpoint's on-disk precision need not be the precision it computes in.
            "precision": config.MODELS[self.model_name].get("precision", "bf16"),
            "temperature": config.TEMPERATURE,
            "top_p": config.TOP_P,
            "max_tokens": config.MAX_NEW_TOKENS,
            "start_time_utc": t_start,
            "end_time_utc": t_end,
            # peak_gpu_mem_mb and gpu_power_draw_w_mean are summed across every GPU the model used
            # (see _attempt's n_devices loop and PowerSampler). Without the card count beside
            # them, a four-GPU model's figures would mean something different from a one-GPU
            # model's in the same column, and nothing in the column would say so.
            "num_gpus": len(self.physical_gpus),
            "device_map": config.MODELS[self.model_name].get("device_map"),
            "max_memory": config.MODELS[self.model_name].get("max_memory"),
        }
        common.append_jsonl(config.RUN_METADATA_PATH, row)
        self.metadata_done.add((self.model_name, run_type))
        print(f"[{self.model_name}] wrote run_metadata for {run_type}", flush=True)

    def run(self, stop_after_determinism=False):
        self.run_phase("determinism_run1", self.determinism_pairs, ["v2_short"])
        self.run_phase("determinism_run2", self.determinism_pairs, ["v2_short"])
        if stop_after_determinism:
            print(f"[{self.model_name}] --stop-after-determinism set, not starting full_grid",
                  flush=True)
            return
        self.run_phase("full_grid", self.all_pairs, config.PROMPT_IDS)

    def run_validation_subsample(self):
        """The same seeded 100-photograph subsample as the determinism check, against all four
        prompts rather than one, logged under run_type='validation_subsample'. Leaves
        determinism_run1/2 and full_grid untouched."""
        self.run_phase("validation_subsample", self.determinism_pairs, config.PROMPT_IDS)

    def run_validation_subsample_v2(self):
        """A second pass over the same 100 photographs and all four prompts, with the same seed and
        the same selection logic as validation_subsample rather than a fresh draw, so the two
        passes line up photograph for photograph. Its own run_type keeps them separable."""
        self.run_phase("validation_subsample_v2", self.determinism_pairs, config.PROMPT_IDS)

    def _exp2_pairs_for_split(self, split_name):
        exp2_split = load_exp2_split()
        pairs = [(fn, ref) for fn, ref in self.all_pairs if exp2_split.get(fn) == split_name]
        missing = [fn for fn, _ in self.all_pairs if fn not in exp2_split]
        if missing:
            print(f"[{self.model_name}] WARNING: {len(missing)} images have no exp2 split "
                  f"assignment (not in {config.EXP2_SPLIT_CSV}) -- excluded from this run",
                  flush=True)
        return pairs

    def run_exp2_pilot(self):
        """The four dormant-clause prompts on a 100-photograph subsample of the selection split,
        stratified by cover bin. run_type='exp2_pilot'. Needs the split file named by
        config.EXP2_SPLIT_CSV to exist already."""
        selection_pairs = self._exp2_pairs_for_split("selection")
        pilot_pairs = select_exp2_pilot_subsample(selection_pairs)
        print(f"[{self.model_name}] exp2_pilot: {len(pilot_pairs)} images "
              f"(stratified subsample of {len(selection_pairs)} selection-split images)",
              flush=True)
        self.run_phase("exp2_pilot", pilot_pairs, config.EXP2_PROMPT_IDS)

    def run_exp2_selection(self):
        """The four dormant-clause prompts on the rest of the selection split. The 100 pilot
        photographs are left out, since they are already logged under run_type='exp2_pilot';
        the exp2_pilot and exp2_selection rows together cover the whole selection split.
        run_type='exp2_selection'."""
        selection_pairs = self._exp2_pairs_for_split("selection")
        pilot_pairs = set(select_exp2_pilot_subsample(selection_pairs))
        remaining = [p for p in selection_pairs if p not in pilot_pairs]
        print(f"[{self.model_name}] exp2_selection: {len(remaining)} images "
              f"({len(selection_pairs)} selection-split total minus {len(pilot_pairs)} already "
              f"done in exp2_pilot)", flush=True)
        self.run_phase("exp2_selection", remaining, config.EXP2_PROMPT_IDS)

    def run_exp2_confirmation(self):
        """The four dormant-clause prompts on the whole confirmation split, the part of the dataset
        held back while the clause was chosen. run_type='exp2_confirmation'. This is the run that
        measures the clause, because it is the only one scored on photographs that played no part
        in selecting it."""
        confirmation_pairs = self._exp2_pairs_for_split("confirmation")
        print(f"[{self.model_name}] exp2_confirmation: {len(confirmation_pairs)} images",
              flush=True)
        self.run_phase("exp2_confirmation", confirmation_pairs, config.EXP2_PROMPT_IDS)

    def run_full_grid_fixed(self):
        """The full 1,155-photograph grid under the settings this code holds: text before image in
        the content list, MAX_NEW_TOKENS=2048, thumbnail()/BICUBIC resize, a trailing newline on
        every prompt, and the keyed preprocessing cache. This is the run the paper reports.

        It carries its own run_type because resumability keys on (image, prompt_id, run_type): rows
        logged under any other run_type, including the 100-photograph validation subsample, neither
        satisfy nor shadow this one, so the grid runs across all 1,155 photographs."""
        self.run_phase("full_grid_fixed", self.all_pairs, config.PROMPT_IDS)

    def run_determinism_fixed(self):
        """The determinism check under the same settings as run_full_grid_fixed, so that what it
        reports about repeatability is a statement about the grid the paper reports and not about
        some other arrangement of the pipeline.

        Same seeded 100-photograph subsample and the same single prompt (v2_short) as
        determinism_run1/2. Its own run_type, for the reason given in run_full_grid_fixed: rows
        under another run_type would satisfy the resumability key and the check would be skipped
        rather than run."""
        self.run_phase("determinism_run1_fixed", self.determinism_pairs, ["v2_short"])
        self.run_phase("determinism_run2_fixed", self.determinism_pairs, ["v2_short"])


def main():
    parser = argparse.ArgumentParser(parents=[_parser])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None,
                         help="dev/debug only: truncate the image list")
    parser.add_argument("--stop-after-determinism", action="store_true",
                         help="run the determinism check only, skip full_grid")
    parser.add_argument("--validation-subsample", action="store_true",
                         help="the seeded 100-image determinism subsample against all 4 prompts, "
                              "run_type=validation_subsample. Mutually exclusive with the normal "
                              "determinism+full_grid run.")
    parser.add_argument("--validation-subsample-v2", action="store_true",
                         help="a second pass over the same 100-image subsample and all 4 prompts, "
                              "run_type=validation_subsample_v2. Mutually exclusive with the "
                              "other modes.")
    parser.add_argument("--full-grid-fixed", action="store_true",
                         help="the full 1,155-image grid reported in the paper, "
                              "run_type=full_grid_fixed. Mutually exclusive with the other modes.")
    parser.add_argument("--determinism-fixed", action="store_true",
                         help="the determinism check (same 100-image subsample, v2_short only, "
                              "run1 then run2) under the same settings as --full-grid-fixed. "
                              "run_type=determinism_run1_fixed/determinism_run2_fixed. Mutually "
                              "exclusive with the other modes.")
    parser.add_argument("--exp2-pilot", action="store_true",
                         help="the 4 dormant-clause prompts on a 100-image subsample of the "
                              "selection split, stratified by cover bin. run_type=exp2_pilot. "
                              "Needs the split file to exist. Model must be in "
                              "config.EXP2_MODELS.")
    parser.add_argument("--exp2-selection", action="store_true",
                         help="the 4 dormant-clause prompts on the rest of the selection split "
                              "(pilot images excluded, already logged). "
                              "run_type=exp2_selection.")
    parser.add_argument("--exp2-confirmation", action="store_true",
                         help="the 4 dormant-clause prompts on the whole confirmation split, the "
                              "part of the dataset held back while the clause was chosen. "
                              "run_type=exp2_confirmation.")
    args = parser.parse_args()

    exp2_flags = [args.exp2_pilot, args.exp2_selection, args.exp2_confirmation]
    if sum(exp2_flags) > 1:
        print("--exp2-pilot / --exp2-selection / --exp2-confirmation are mutually exclusive "
              "(one stage per invocation).", file=sys.stderr)
        sys.exit(2)
    if any(exp2_flags) and args.model not in config.EXP2_MODELS:
        print(f"{args.model!r} is not in config.EXP2_MODELS {config.EXP2_MODELS} -- the "
              f"dormant-clause prompts are scoped to these 4 models; Scout and Maverick are "
              f"served through vLLM instead.", file=sys.stderr)
        sys.exit(2)

    batch_size = args.batch_size or config.MODELS[args.model]["batch_size"]
    worker = Worker(args.model, _gpus, batch_size, limit_images=args.limit_images)
    if args.validation_subsample:
        worker.run_validation_subsample()
    elif args.validation_subsample_v2:
        worker.run_validation_subsample_v2()
    elif args.full_grid_fixed:
        worker.run_full_grid_fixed()
    elif args.determinism_fixed:
        worker.run_determinism_fixed()
    elif args.exp2_pilot:
        worker.run_exp2_pilot()
    elif args.exp2_selection:
        worker.run_exp2_selection()
    elif args.exp2_confirmation:
        worker.run_exp2_confirmation()
    else:
        worker.run(stop_after_determinism=args.stop_after_determinism)
    print(f"[{args.model}] worker finished this invocation.", flush=True)


if __name__ == "__main__":
    main()
