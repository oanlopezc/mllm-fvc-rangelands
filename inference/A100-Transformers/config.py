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

# Every prompt in the API notebook ends with a newline after the JSON template line. This assert
# is the only thing standing between a stray reformat and a one-character difference from the API
# run these results are compared against, which no output would reveal.
assert all(p.endswith('{"vegetation_percent": <number>, "confidence": <number>}\n')
           for p in PROMPTS.values())

# ---------------------------------------------------------------------------
# A second prompt set. Each prompt here is its base prompt above plus one inserted clause telling
# the model that vegetation cover includes dormant plants. v3 also needed two words changed, since
# its base text defines cover as living and green tissue and would otherwise contradict the new
# clause; nothing else in any base prompt changes.
#
# No estimate reported in the paper comes from these prompts. Table 1 lists the four above and
# nothing else.
# ---------------------------------------------------------------------------

_DORMANT_CLAUSE = (
    "Vegetation cover includes dormant plant material (alive but not currently green, e.g. due "
    "to seasonal dormancy) as well as green growth.\n"
)

PROMPTS["v1_point_hint_dormant"] = (
    "Estimate % vegetation cover inside the 1x1 m quadrat only.\n"
    "Imagine sampling many points uniformly and counting vegetation hits.\n"
    + _DORMANT_CLAUSE +
    "\n"
    "Return STRICT JSON only:\n"
    '{"vegetation_percent": <number>, "confidence": <number>}\n'
)
PROMPTS["v2_short_dormant"] = (
    "Inside the quadrat only: estimate vegetation cover percent (0–100).\n"
    + _DORMANT_CLAUSE +
    "Return JSON only:\n"
    '{"vegetation_percent": <number>, "confidence": <number>}\n'
)
PROMPTS["v3_detailed_ecology_dormant"] = (
    "You are an expert in vegetation ecology, trained in standard field methods such as "
    "quadrat sampling and point-intercept analysis.\n"
    "Your task is to estimate the percentage of ground area covered by vegetation "
    "inside a 1x1 m quadrat.\n"
    "\n"
    "Definition of vegetation cover (based on ecological literature):\n"
    "- Include: grasses, herbs, shrubs, tree seedlings, mosses, and any other plant tissue, "
    "living or dormant.\n"
    "- " + _DORMANT_CLAUSE +
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
)
PROMPTS["v4_grid_overlay_dormant"] = (
    "Estimate % vegetation cover inside the 1x1 m quadrat only.\n"
    "One way to do this is to imagine dividing the quadrat into a 10x10 grid (100 equal "
    "squares).\n"
    "For each square, decide if vegetation covers most of it or not, then sum the total.\n"
    "This approximates the area covered by vegetation.\n"
    + _DORMANT_CLAUSE +
    "\n"
    "Return STRICT JSON only:\n"
    '{"vegetation_percent": <number>, "confidence": <number>}\n'
)

EXP2_BASE_TO_DORMANT = {
    "v1_point_hint": "v1_point_hint_dormant",
    "v2_short": "v2_short_dormant",
    "v3_detailed_ecology": "v3_detailed_ecology_dormant",
    "v4_grid_overlay": "v4_grid_overlay_dormant",
}
EXP2_PROMPT_IDS = list(EXP2_BASE_TO_DORMANT.values())

# The same two details the four base prompts are held to.
assert "–" in PROMPTS["v2_short_dormant"]
assert "–" in PROMPTS["v3_detailed_ecology_dormant"]
assert "–" not in PROMPTS["v1_point_hint_dormant"]
assert "–" not in PROMPTS["v4_grid_overlay_dormant"]
assert all(p.endswith('{"vegetation_percent": <number>, "confidence": <number>}\n')
           for p in (PROMPTS[pid] for pid in EXP2_PROMPT_IDS))
assert all("dormant plant material" in PROMPTS[pid] for pid in EXP2_PROMPT_IDS)
# v3_dormant's specific self-contradiction fix -- make sure a future edit can't silently
# reintroduce "living"/"green" as an exclusive qualifier on the Include line.
assert "living grasses" not in PROMPTS["v3_detailed_ecology_dormant"]
assert "green photosynthetic" not in PROMPTS["v3_detailed_ecology_dormant"]

# ---------------------------------------------------------------------------
# A third prompt set, differing from the second by one word: "rooted dormant plants" in place of
# "dormant plant material". "Plant material" covers non-rooted plant material as readily as it
# covers dormant vegetation, so the clause was applied too broadly on sparse quadrats. "Rooted"
# names something a model can check against what it sees, since non-rooted material is by
# definition detached from the ground. Nothing else in any prompt changes.
#
# As with the second set, no estimate reported in the paper comes from these prompts.
# ---------------------------------------------------------------------------

_DORMANT_CLAUSE_ROOTED = (
    "Vegetation cover includes rooted dormant plants (alive but not currently green, e.g. due "
    "to seasonal dormancy) as well as green growth.\n"
)

PROMPTS["v1_point_hint_dormant_rooted"] = (
    "Estimate % vegetation cover inside the 1x1 m quadrat only.\n"
    "Imagine sampling many points uniformly and counting vegetation hits.\n"
    + _DORMANT_CLAUSE_ROOTED +
    "\n"
    "Return STRICT JSON only:\n"
    '{"vegetation_percent": <number>, "confidence": <number>}\n'
)
PROMPTS["v2_short_dormant_rooted"] = (
    "Inside the quadrat only: estimate vegetation cover percent (0–100).\n"
    + _DORMANT_CLAUSE_ROOTED +
    "Return JSON only:\n"
    '{"vegetation_percent": <number>, "confidence": <number>}\n'
)
PROMPTS["v3_detailed_ecology_dormant_rooted"] = (
    "You are an expert in vegetation ecology, trained in standard field methods such as "
    "quadrat sampling and point-intercept analysis.\n"
    "Your task is to estimate the percentage of ground area covered by vegetation "
    "inside a 1x1 m quadrat.\n"
    "\n"
    "Definition of vegetation cover (based on ecological literature):\n"
    "- Include: grasses, herbs, shrubs, tree seedlings, mosses, and any other plant tissue, "
    "living or dormant.\n"
    "- " + _DORMANT_CLAUSE_ROOTED +
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
)
PROMPTS["v4_grid_overlay_dormant_rooted"] = (
    "Estimate % vegetation cover inside the 1x1 m quadrat only.\n"
    "One way to do this is to imagine dividing the quadrat into a 10x10 grid (100 equal "
    "squares).\n"
    "For each square, decide if vegetation covers most of it or not, then sum the total.\n"
    "This approximates the area covered by vegetation.\n"
    + _DORMANT_CLAUSE_ROOTED +
    "\n"
    "Return STRICT JSON only:\n"
    '{"vegetation_percent": <number>, "confidence": <number>}\n'
)

EXP2_BASE_TO_ROOTED = {
    "v1_point_hint": "v1_point_hint_dormant_rooted",
    "v2_short": "v2_short_dormant_rooted",
    "v3_detailed_ecology": "v3_detailed_ecology_dormant_rooted",
    "v4_grid_overlay": "v4_grid_overlay_dormant_rooted",
}
EXP2_ROOTED_PROMPT_IDS = list(EXP2_BASE_TO_ROOTED.values())

assert "–" in PROMPTS["v2_short_dormant_rooted"]
assert "–" in PROMPTS["v3_detailed_ecology_dormant_rooted"]
assert "–" not in PROMPTS["v1_point_hint_dormant_rooted"]
assert "–" not in PROMPTS["v4_grid_overlay_dormant_rooted"]
assert all(p.endswith('{"vegetation_percent": <number>, "confidence": <number>}\n')
           for p in (PROMPTS[pid] for pid in EXP2_ROOTED_PROMPT_IDS))
assert all("rooted dormant plants" in PROMPTS[pid] for pid in EXP2_ROOTED_PROMPT_IDS)
assert all("dormant plant material" not in PROMPTS[pid] for pid in EXP2_ROOTED_PROMPT_IDS)
assert "living grasses" not in PROMPTS["v3_detailed_ecology_dormant_rooted"]
assert "green photosynthetic" not in PROMPTS["v3_detailed_ecology_dormant_rooted"]

# The four models the second and third prompt sets were run against. Scout and Maverick are absent
# because they are served through vLLM on different hardware (see ../H200-vLLM/).
EXP2_MODELS = [
    "Qwen2.5-VL-32B-Instruct",
    "Mistral-Small-3.2-24B-Instruct",
    "Gemma-3-12B",
    "Gemma-3-27B",
]

# The split those runs used, held out so a clause chosen on one set of images is measured on
# another. It is generated once and never regenerated: regenerating would move images between the
# two sets and leave every run already made against it measuring something else.
EXP2_SPLIT_RATIO = 0.7  # fraction assigned to "selection"
EXP2_SPLIT_SEED = 42
EXP2_PILOT_N = 100
EXP2_FVC_BIN_EDGES = [0, 20, 40, 60, 80, 100]  # the five cover bins used throughout

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
    # Llama-4-Scout and Llama-4-Maverick belong to the grid these constants describe, so their
    # settings live here with the rest of the model table. Their reported estimates come from
    # vLLM on H200s; ../H200-vLLM/ holds that code. What keeps them off this path is a property
    # of each checkpoint, stated below where the setting it explains sits.
    #
    # Scout is 203 GB on disk (bf16, 50 shards) and so does not fit one 80 GB A100 as the other
    # four models do. "gpus" (a list) rather than "gpu" (a single int) tells resolve_gpus() below
    # to hand a worker all four cards, and device_map="auto" lets accelerate shard the model
    # across them, which needs a dedicated allocation rather than the shared-node,
    # one-worker-per-GPU arrangement the other four use. Under transformers it generates at
    # roughly 37,900 ms per image, against roughly 24 ms under vLLM.
    "Llama-4-Scout-17B-16E-Instruct": {
        "path": os.path.join(PROJECT_ROOT, "models/meta-llama/Llama-4-Scout-17B-16E-Instruct"),
        "repo_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "gpus": [0, 1, 2, 3],
        "device_map": "auto",
        "batch_size": 1,
    },
    # Maverick is 416.8 GB on disk as FP8, and that figure does not describe what it needs to run
    # under transformers. An A100 has no FP8 tensor cores, so compressed-tensors decompresses every
    # quantized tensor to bf16 in memory before generate() is reached, and the resident size is the
    # bf16 size: 803 GB, matching the unquantized checkpoint. Weights alone therefore exceed a
    # 4-GPU node's whole capacity of roughly 774 GB, VRAM and system RAM together, which is a
    # ceiling no placement setting moves.
    #
    # max_memory is a placement budget, not a runtime one. It governs where accelerate puts each
    # tensor's compressed footprint while planning the device map, and decompression then grows
    # those tensors in place with nothing holding them to the figure that guided the placement. A
    # card sitting near its cap on compressed bytes lands past it once decompressed, and expert
    # placement across a mixture-of-experts model is uneven enough that one card takes more than
    # its share. The cap below is set far under each card's capacity for that reason, pushing the
    # remainder to CPU.
    "Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "path": os.path.join(
            PROJECT_ROOT, "models/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        ),
        "repo_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        # The 803 GB requirement is a property of transformers and compressed-tensors, not of the
        # card: on H200s the same decompression fills GPU 0 to 143,087 of 143,771 MiB at 8% of the
        # way through. Six H200s give roughly 842 GB, a margin of about 39 GB over the requirement,
        # so the caps below leave roughly 15 GiB of headroom per card (125 GiB of about 140.4) and
        # send the remaining 53 GB, about 6.6% of the model, to CPU. These settings let the model
        # load; Maverick's reported estimates come from vLLM (see ../H200-vLLM/).
        "gpus": [0, 1, 2, 3, 4, 5],
        "device_map": "auto",
        "max_memory": {
            0: "125GiB", 1: "125GiB", 2: "125GiB", 3: "125GiB", 4: "125GiB", 5: "125GiB",
            "cpu": "400GiB",
        },
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

# Cap on the pixels the *model's own* image processor keeps, applied after the JPEG written by
# common.preprocess_image(). None means each checkpoint's shipped default (Qwen2.5-VL: 12,845,056,
# so a 1536x1152 image passes through untouched at about 2,255 visual tokens). Hosted providers
# routinely set this far lower to control cost, and 1280*28*28 = 1,003,520 is the most common
# serving default. Qwen2-VL family only; Gemma-3 and Mistral ignore it.
IMAGE_MAX_PIXELS = None
QWEN_COMMON_SERVING_MAX_PIXELS = 1280 * 28 * 28  # 1,003,520
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

# Experiment 2 Stage 0 output -- built once by build_exp2_split.py, then read by run_worker.py
# and both report_exp2_*.py scripts. Columns: image, reference_fvc, fvc_bin, split.
EXP2_SPLIT_CSV = os.path.join(PROJECT_ROOT, "data/refs/selection_confirmation_split.csv")

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
