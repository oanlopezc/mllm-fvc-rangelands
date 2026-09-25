"""Shared helpers for experiment1 workers: image preprocessing, JSON parsing, retry/backoff,
GPU power sampling, JSONL I/O, and generic VLM loading."""
import datetime
import hashlib
import json
import os
import re
import sys
import threading
import time

import PIL
from PIL import Image, ImageFile

from experiment1 import config

# Both match the API notebook's import-time setup (see ../API/). These are ordinary large
# field-camera photographs, not a decompression-bomb attack, and a slightly truncated file should
# decode rather than raise.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


# ---------------------------------------------------------------------------
# Image preprocessing (thumbnail to max-side 1536, re-encode JPEG q90), cached to disk once per
# image so it isn't redone across the 4 prompts x 4 models that share the same source image.
#
# The cache lives in a subdirectory named after a hash of every input that determines the bytes
# written -- filter, max side, JPEG quality, code path, and the Pillow version. Changing any of
# them lands in a different subdirectory, so a run can never read images written under settings
# other than its own.
# ---------------------------------------------------------------------------


def resample_filter():
    """The PIL resampling constant named by config.IMAGE_RESAMPLE_NAME."""
    return getattr(Image, config.IMAGE_RESAMPLE_NAME)


def preprocessing_params() -> dict:
    """Everything that determines a preprocessed file's bytes. Feeds the cache key -- add to this
    dict (never mutate a value silently) whenever preprocess_image()'s behaviour changes."""
    return {
        "method": "PIL.Image.thumbnail",  # see preprocess_to_pil() for why thumbnail, not resize
        "max_side": config.IMAGE_MAX_SIDE,
        "resample": config.IMAGE_RESAMPLE_NAME,
        "jpeg_quality": config.IMAGE_JPEG_QUALITY,
        "mode": "RGB",
        "pillow_version": PIL.__version__,  # resampling and the JPEG encoder both live here
    }


def preprocessing_key() -> str:
    blob = json.dumps(preprocessing_params(), sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def preprocessed_dir() -> str:
    return os.path.join(config.PREPROCESSED_DIR, preprocessing_key())


def _ensure_preprocessed_dir() -> str:
    """Creates the keyed cache directory and drops a params.json in it, so a stray directory of
    JPEGs on disk is always self-describing rather than an opaque hash."""
    d = preprocessed_dir()
    os.makedirs(d, exist_ok=True)
    marker = os.path.join(d, "params.json")
    if not os.path.exists(marker):
        tmp = f"{marker}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(preprocessing_params(), f, indent=2, sort_keys=True)
        os.replace(tmp, marker)
    return d


def preprocess_image(src_path: str) -> str:
    cache_dir = preprocessed_dir()
    cached_path = os.path.join(cache_dir, os.path.basename(src_path))
    if os.path.exists(cached_path):
        return cached_path

    _ensure_preprocessed_dir()
    img = preprocess_to_pil(src_path)

    tmp_path = f"{cached_path}.tmp.{os.getpid()}"
    img.save(tmp_path, "JPEG", quality=config.IMAGE_JPEG_QUALITY)
    os.replace(tmp_path, cached_path)  # atomic; safe if multiple workers race on the same image
    return cached_path


def preprocess_to_pil(src_path: str) -> Image.Image:
    """The decode and resize half of preprocess_image(), byte-for-byte what the API notebook's
    load_and_normalize_image() does (see ../API/). Split out so it can be checked against that
    notebook without writing to the cache."""
    img = Image.open(src_path)
    img = img.convert("RGB")
    # thumbnail() rather than resize(): it never upscales, and its default reducing_gap=2.0
    # box-reduces before resampling on the 12,000 x 9,000 photographs. Doing the scale arithmetic
    # by hand and calling resize() reproduces the output size but not those pixels.
    img.thumbnail((config.IMAGE_MAX_SIDE, config.IMAGE_MAX_SIDE), resample_filter())
    return img


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------
#
# This delegates to parser/fvc_parser.py, the single parser behind every
# published estimate. It scans the response for well-formed JSON objects and
# takes the last one carrying a vegetation percentage and a confidence as
# numbers, accepting "vegetation_cover" as well as "vegetation_percent" for the
# cover field.
#
# Every model writes its raw response to the row log, so the published values
# come from running this parser over those logs rather than from anything a
# worker computed in passing.

_PARSER_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "parser")
)
sys.path.insert(0, _PARSER_DIR)
from fvc_parser import parse_response as _parse_pair  # noqa: E402


def parse_response(raw_text: str):
    """Returns (vegetation_percent, confidence, ok: bool).

    Wraps the published parser in the three-value shape the workers unpack, so
    call sites in run_worker.py and worker_vllm.py are unchanged.
    """
    got = _parse_pair(raw_text)
    if got is None:
        return None, None, False
    return got[0], got[1], True


# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------


def retry_with_backoff(fn, max_retries: int = 5, base_delay: float = 2.0):
    """Calls fn() with exponential backoff on exception.

    Returns (result, None) on success, or (None, last_exception) once retries are exhausted.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn(), None
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is the retry boundary
            last_exc = e
            # A retry that eventually succeeds must announce itself. Its row logs error="", the
            # same as a clean first attempt, so without this line the 2 to 16 s backoff delays
            # are invisible in the row's latency_ms and in any progress check based on row
            # counts, and a model slowed by retries reads as a model that is simply slow.
            print(f"[retry_with_backoff] attempt {attempt + 1}/{max_retries} failed: "
                  f"{type(e).__name__}: {str(e)[:300]}", flush=True)
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
    return None, last_exc


# ---------------------------------------------------------------------------
# GPU power sampling (background thread, polls pynvml during a batch's forward/generate call)
# ---------------------------------------------------------------------------


class PowerSampler:
    """Samples GPU power draw on a background thread. `gpu_indices` must be *physical* device
    index/indices (pynvml does not respect CUDA_VISIBLE_DEVICES). Pass a single int for a model
    that fits one card, or a list for a model sharded across several, in which case each sample
    is the summed draw across all of them -- the model's total power, not one card's share."""

    def __init__(self, gpu_indices, interval: float = 0.2):
        if isinstance(gpu_indices, int):
            gpu_indices = [gpu_indices]
        self.interval = interval
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self._handles = []
        if _NVML_OK:
            for idx in gpu_indices:
                try:
                    self._handles.append(pynvml.nvmlDeviceGetHandleByIndex(idx))
                except Exception:
                    pass

    def _run(self):
        while not self._stop.is_set():
            if self._handles:
                try:
                    total_mw = sum(pynvml.nvmlDeviceGetPowerUsage(h) for h in self._handles)
                    self._samples.append(total_mw / 1000.0)
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def mean_watts(self):
        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples)


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def append_jsonl(path: str, row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn last line from a crash mid-write
    return rows


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Generic VLM loading -- all 4 models register under AutoModelForImageTextToText.
#
# Mistral-Small-3.2's local checkout ships no preprocessor_config.json/tokenizer_config.json --
# only its native tekken.json -- so AutoProcessor can't build a processor for it. In that case we
# fall back to AutoTokenizer, which (with mistral_common installed) resolves tekken.json to
# MistralCommonBackend: a tokenizer whose apply_chat_template() also does the image processing
# and returns pixel_values directly, making it a drop-in stand-in for a processor here.
# ---------------------------------------------------------------------------


def load_model(path: str, image_max_pixels: int = None, device_map=None, max_memory=None):
    """`device_map` defaults to pinning everything onto the single GPU CUDA_VISIBLE_DEVICES
    exposes, which is one GPU per model. Pass "auto" for a model too large for one card --
    accelerate then shards layers across every GPU CUDA_VISIBLE_DEVICES exposes.

    `max_memory` (dict, e.g. {0: "60GiB", "cpu": "250GiB"}) matters only alongside
    device_map="auto": left unset, accelerate's balanced-memory placement packs each visible GPU
    to its full reported capacity based on static weight size alone, with no headroom for
    decompression scratch (compressed-tensors FP8) or generation-time activations and KV-cache.
    Llama-4-Maverick then fills GPU 0 to 79.18 of its 79.25 GiB while decompressing weights,
    before inference runs at all. Pass an explicit max_memory below each GPU's true capacity to
    force real headroom and push the remainder to CPU."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    try:
        processor = AutoProcessor.from_pretrained(path)
    except OSError:
        processor = AutoTokenizer.from_pretrained(path)

    if image_max_pixels is not None:
        apply_image_max_pixels(processor, image_max_pixels)

    from_pretrained_kwargs = {}
    if max_memory is not None:
        from_pretrained_kwargs["max_memory"] = max_memory

    model = AutoModelForImageTextToText.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map=device_map if device_map is not None else {"": 0},
        attn_implementation="sdpa",
        **from_pretrained_kwargs,
    )
    model.eval()

    # Batched greedy generation needs left-padding on a decoder-only LM, and every tokenizer
    # here needs an explicit pad token (some VLM tokenizers don't ship one).
    # MistralCommonBackend *is* the tokenizer (subclasses PreTrainedTokenizerBase) but also
    # exposes its own inner `.tokenizer` (the raw mistral_common object) -- only descend into
    # `.tokenizer` for ProcessorMixin-style wrappers (Gemma3/Qwen2.5-VL), not for this.
    tok = processor if hasattr(processor, "pad_token") else getattr(processor, "tokenizer", processor)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return model, processor


def _size_field(size, key, default=None):
    """`ip.size` has held a plain dict in some transformers releases and a `SizeDict` (an object
    with attribute access, not dict-subclassed -- `isinstance(size, dict)` is False for it) in
    others. Handle both rather than guessing which one is installed."""
    if size is None:
        return default
    if isinstance(size, dict):
        return size.get(key, default)
    return getattr(size, key, default)


def _set_size_field(size, key, value):
    if isinstance(size, dict):
        size[key] = value
    else:
        setattr(size, key, value)


def read_image_max_pixels(processor):
    """The pixel budget the loaded processor is currently using, or None if it has no such knob."""
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return None
    longest_edge = _size_field(getattr(ip, "size", None), "longest_edge")
    if longest_edge is not None:
        return longest_edge
    return getattr(ip, "max_pixels", None)


def apply_image_max_pixels(processor, max_pixels: int):
    """Cap the model's own image processor (Qwen2-VL family only; a no-op elsewhere).

    Set on the loaded processor rather than passed to from_pretrained, because transformers has
    moved the knob between `max_pixels` and `size.longest_edge` across releases and both
    spellings need to agree or the older one silently wins.
    """
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        print(f"  image_max_pixels={max_pixels} requested but this processor has no "
              f"image_processor -- ignored", flush=True)
        return False

    touched = []
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = max_pixels
        touched.append("max_pixels")
    size = getattr(ip, "size", None)
    if _size_field(size, "longest_edge") is not None:
        _set_size_field(size, "longest_edge", max_pixels)
        touched.append("size.longest_edge")

    if not touched:
        print(f"  image_max_pixels={max_pixels} requested but {type(ip).__name__} exposes no "
              f"pixel budget -- ignored (expected for Gemma-3 / Mistral)", flush=True)
        return False
    print(f"  image_max_pixels={max_pixels} applied to {type(ip).__name__} via "
          f"{' and '.join(touched)} (now {read_image_max_pixels(processor)})", flush=True)
    return True


def build_conversation(prompt_text: str, image_path: str, order: str = None):
    """A single-message conversation; `image_path` is passed by path so apply_chat_template loads
    and processes it itself (works uniformly across AutoProcessor and MistralCommonBackend).

    `order` defaults to config.MESSAGE_ORDER. Every chat template in use here renders the content
    list in the order given, so this genuinely changes where the vision tokens land relative to
    the instruction text -- it is not cosmetic.
    """
    order = order or config.MESSAGE_ORDER
    text_block = {"type": "text", "text": prompt_text}
    image_block = {"type": "image", "path": image_path}
    if order == "text_first":
        content = [text_block, image_block]
    elif order == "image_first":
        content = [image_block, text_block]
    else:
        raise ValueError(f"unknown MESSAGE_ORDER {order!r}; use 'text_first' or 'image_first'")
    return [{"role": "user", "content": content}]
