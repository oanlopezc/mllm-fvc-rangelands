"""Prompts, model table and generation settings for the local runs.

The four prompts are held here as the exact strings sent to the models. Two details are easy to
lose to a reformat and both change the bytes a model receives: v2_short and v3_detailed_ecology
use a Unicode en-dash (U+2013) in "0-100" where v1_point_hint and v4_grid_overlay use a hyphen,
and all four end with a newline after the JSON template line. The asserts below hold the prompts
to that, because the same characters were sent through the API gateway and the two runs are
compared against each other.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Prompts -- exact, verbatim, do not alter.
# ---------------------------------------------------------------------------

PROMPTS = {
    "v1_point_hint": (
        "Estimate % vegetation cover inside the 1x1 m quadrat only.\n"
        "Imagine sampling many points uniformly and counting vegetation hits.\n"
        "\n"
        "Return STRICT JSON only:\n"
        '{"vegetation_percent": <number>, "confidence": <number>}\n'
    ),
    "v2_short": (
        "Inside the quadrat only: estimate vegetation cover percent (0–100).\n"
        "Return JSON only:\n"
        '{"vegetation_percent": <number>, "confidence": <number>}\n'
    ),
    "v3_detailed_ecology": (
        "You are an expert in vegetation ecology, trained in standard field methods such as "
        "quadrat sampling and point-intercept analysis.\n"
        "Your task is to estimate the percentage of ground area covered by live vegetation "
        "inside a 1x1 m quadrat.\n"
        "\n"
        "Definition of vegetation cover (based on ecological literature):\n"
        "- Include: living grasses, herbs, shrubs, tree seedlings, mosses, and any other green "
        "photosynthetic plant tissue.\n"
        "- Exclude: bare soil, litter (dead leaves, twigs), rocks, shadows, water, and man-made "
        "objects.\n"
        "- Count overlapping vegetation only once (do not double-count leaves stacked "
        "vertically).\n"
        "- Boundaries: only consider the area strictly inside the quadrat frame; ignore anything "
        "outside.\n"
        "\n"
        "Provide your best estimate of the proportion of ground covered by vegetation "
        "(0–100%).\n"
        "Return STRICT JSON only:\n"
        '{"vegetation_percent": <number>, "confidence": <number>}\n'
    ),
    "v4_grid_overlay": (
        "Estimate % vegetation cover inside the 1x1 m quadrat only.\n"
        "One way to do this is to imagine dividing the quadrat into a 10x10 grid (100 equal "
        "squares).\n"
        "For each square, decide if vegetation covers most of it or not, then sum the total.\n"
        "This approximates the area covered by vegetation.\n"
        "\n"
        "Return STRICT JSON only:\n"
        '{"vegetation_percent": <number>, "confidence": <number>}\n'
    ),
}

PROMPT_IDS = ["v1_point_hint", "v2_short", "v3_detailed_ecology", "v4_grid_overlay"]

# The en-dash belongs to exactly two of the four prompts.
assert "–" in PROMPTS["v2_short"]
assert "–" in PROMPTS["v3_detailed_ecology"]
assert "–" not in PROMPTS["v1_point_hint"]
assert "–" not in PROMPTS["v4_grid_overlay"]

# Every prompt in the API notebook ends with a newline after the JSON template line, and the local
# prompts must match it byte for byte.
assert all(p.endswith('{"vegetation_percent": <number>, "confidence": <number>}\n')
           for p in PROMPTS.values())

# ---------------------------------------------------------------------------
# Models -- the name used in logs -> local path, GPU index, batch size.
#
# GPU assignment and batch size affect throughput, never the estimate a model returns: generation
# is greedy and each image is a separate request. They are recorded here so a run can be
# reproduced at the same cost, not because a number depends on them.
# ---------------------------------------------------------------------------

MODELS = {
    "Qwen2.5-VL-32B-Instruct": {
        "path": os.path.join(PROJECT_ROOT, "models/qwen/Qwen2.5-VL-32B-Instruct"),
        "repo_id": "Qwen/Qwen2.5-VL-32B-Instruct",
        "gpu": 0,
        "batch_size": 2,
    },
    "Mistral-Small-3.2-24B-Instruct": {
        "path": os.path.join(
            PROJECT_ROOT, "models/mistralai/Mistral-Small-3.2-24B-Instruct-2506"
        ),
        "repo_id": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "gpu": 1,
        # batch_size=1, not 4: MistralCommonBackend's apply_chat_template() (see
        # common.load_model's docstring) stacks a batch's images into one array itself, where
        # every other model's AutoProcessor pads and resizes to a uniform shape first.
        # common.preprocess_to_pil() preserves aspect ratio, so the photographs arrive in
        # portrait, landscape and square shapes, and any batch mixing them raises "ValueError:
        # all input arrays must have the same shape". One image per batch is the only size that
        # always works for this model, and run_chunk's bisection fallback converges to it anyway
        # after spending the retry budget on the larger sizes first.
        "batch_size": 1,
    },
    "Gemma-3-12B": {
        "path": os.path.join(PROJECT_ROOT, "models/google/gemma-3-12b-it"),
        "repo_id": "google/gemma-3-12b-it",
        "gpu": 2,
        "batch_size": 8,
    },
    "Gemma-3-27B": {
        "path": os.path.join(PROJECT_ROOT, "models/google/gemma-3-27b-it"),
        "repo_id": "google/gemma-3-27b-it",
        "gpu": 3,
        "batch_size": 4,
    },
    # Llama-4-Scout and Llama-4-Maverick are listed here so every path shares one model table. Their
    # reported estimates come from vLLM on H200 GPUs (see ../H200-vLLM/).
    "Llama-4-Scout-17B-16E-Instruct": {
        "path": os.path.join(PROJECT_ROOT, "models/meta-llama/Llama-4-Scout-17B-16E-Instruct"),
        "repo_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "gpus": [0, 1, 2, 3],
        "device_map": "auto",
        "batch_size": 1,
    },
    "Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "path": os.path.join(
            PROJECT_ROOT, "models/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        ),
        "repo_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "gpus": [0, 1, 2, 3],
        "batch_size": 1,
    },
}

MODEL_NAMES = list(MODELS.keys())


def resolve_gpus(model_name: str, cli_override: int = None) -> list:
    """Physical GPU indices a worker process for this model should be given.

    `cli_override` (the CLI's single-value --gpu flag) always wins and always means "just this
    one GPU" -- it has no multi-GPU form. Otherwise: MODELS[model]["gpus"] (a list) for a model too
    large for one card, else the single MODELS[model]["gpu"] wrapped in a one-element list, which
    is what CUDA_VISIBLE_DEVICES receives either way.
    """
    if cli_override is not None:
        return [cli_override]
    m = MODELS[model_name]
    if "gpus" in m:
        return list(m["gpus"])
    return [m["gpu"]]

# ---------------------------------------------------------------------------
# Generation settings.
#
# These two are logged with every row and are what the paper reports. transformers.generate() is
# called with do_sample=False, which is greedy decoding and the local equivalent of temperature=0
# at an API.
# ---------------------------------------------------------------------------

TEMPERATURE = 0
TOP_P = 1

# Order of the blocks inside the user message's content list. "text_first" matches the payload the
# API notebook sends. Whether a hosted provider rendered that order or hoisted the image to the
# front cannot be seen from the client side, since serving stacks differ on it, so this stays a
# setting rather than a claim about what the API run did.
MESSAGE_ORDER = "text_first"

# Cap on the pixels the model's own image processor keeps, applied after the JPEG written by
# common.preprocess_image(). None keeps each checkpoint's default (Qwen2.5-VL: 12,845,056 pixels,
# so a 1536x1152 image passes through unchanged). Qwen2-VL family only; Gemma-3 and Mistral ignore it.
IMAGE_MAX_PIXELS = None
# Generous enough that no model's reasoning is cut off before its JSON answer. Several models
# narrate at length before producing the object the parser reads, and a response truncated mid-way
# carries no answer at all, so it would be recorded as missing rather than merely shortened.
MAX_NEW_TOKENS = 2048
SEED = 42

# ---------------------------------------------------------------------------
# Image preprocessing -- the same steps the API path applies, so the two runs see the same pixels.
#
# The API notebook's load_and_normalize_image() is exactly:
#     img = Image.open(path); img = img.convert("RGB")
#     img.thumbnail((1536, 1536), Image.BICUBIC)      # note: default reducing_gap=2.0
#     img.save(buf, format="JPEG", quality=90)
# common.preprocess_image() calls thumbnail() itself rather than doing the scale arithmetic and
# calling resize(), because reducing_gap is not a no-op on this dataset: 303 of the 1,155
# photographs are 12,000 x 9,000, and for those thumbnail() box-reduces by 3 before resampling
# where resize() does not. Both produce the same output size and different pixels.
# ---------------------------------------------------------------------------

IMAGE_MAX_SIDE = 1536
IMAGE_JPEG_QUALITY = 90
# Resolved to a PIL filter via getattr(PIL.Image, ...) in common.py, so config stays PIL-free
# and the filter has a stable string name to put in the preprocessing cache key.
IMAGE_RESAMPLE_NAME = "BICUBIC"

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

IMAGES_DIR = os.path.join(PROJECT_ROOT, "data/pics/used")
PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/pics/preprocessed")
REFS_CSV = os.path.join(PROJECT_ROOT, "data/refs/All.csv")
JOINED_CSV = os.path.join(PROJECT_ROOT, "data/refs/joined_1155.csv")
EXPECTED_JOIN_COUNT = 1155

# The API run's cleaned results, present locally so the local-against-API comparison can be
# computed here. Its `model` column holds OpenRouter ids, mapped back to the names used in these
# logs by API_MODEL_IDS below.
API_RESULTS_CSV = os.path.join(PROJECT_ROOT, "data/api/results_clean.csv")

API_MODEL_IDS = {
    "Qwen2.5-VL-32B-Instruct": "qwen/qwen2.5-vl-32b-instruct:free",
    "Mistral-Small-3.2-24B-Instruct": "mistralai/mistral-small-3.2-24b-instruct:free",
    "Gemma-3-12B": "google/gemma-3-12b-it:free",
    "Gemma-3-27B": "google/gemma-3-27b-it:free",
    "Llama-4-Scout-17B-16E-Instruct": "meta-llama/llama-4-scout:free",
    "Llama-4-Maverick-17B-128E-Instruct-FP8": "meta-llama/llama-4-maverick:free",
}

DETERMINISM_N = 100
RESULTS_RAW_DIR = os.path.join(PROJECT_ROOT, "results/raw")
RUN_METADATA_PATH = os.path.join(PROJECT_ROOT, "results/run_metadata.jsonl")
MERGED_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results/merged_output.csv")
# All six models' full_grid_fixed rows in one file, which is the run the paper reports. Produced
# by `python -m experiment1.merge_results --run-types full_grid_fixed --output <this path>`.
MERGED_OUTPUT_FIXED_PATH = os.path.join(PROJECT_ROOT, "results/merged_output_fixed.csv")


def model_slug(model_name: str) -> str:
    return model_name.lower().replace(".", "").replace(" ", "-")
