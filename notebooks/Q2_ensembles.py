# %% [markdown]
# # Q2 — Where does each ensemble rank among all configurations?
#
# Six models and four prompts give 24 single model x prompt configurations per
# image. This notebook asks whether **combining** configurations — by model,
# by prompt, or both — changes where a configuration sits, and how stable
# that position is under a different random deal of which images are held
# out. It is a ranked grid: **66 configurations, 42 of them ensembles, ranked
# against one another on 250 independent train/test splits**, with no single
# reference configuration chosen from the data that the other 65 are
# compared against.
#
# **The 66 configurations.** A **model level** is one of 9 things: each of the
# six models individually, or `top2M`, `top3M`, `all6M`. A **prompt level** is
# one of 7: each of the four prompts individually, or `top2P`, `top3P`,
# `all4P`. A cell is written `<model level> x <prompt level>` and its members
# are every model-prompt pair in the cross product; the cell's prediction for
# an image is the **mean** of its members' predictions for that image — the
# reference itself is the mean of two observers, so a mean-combined ensemble
# is the same construction applied to the model side. That is 9 x 7 = 63
# cells (24 single configurations, 39 ensembles). Three further arms sit off
# the grid, where each selected model runs **its own** best prompt instead of
# a shared one: `top2M@own` (2 members), `top3M@own` (3 members), `all6M@own`
# (6 members). **Total: 66 configurations, 42 of them ensembles.**
#
# **Selection is always inside the training fold and never on a held-out
# image, and it runs twice, once per metric.** One arm selects `topKM`,
# `topKP` and each model's own best prompt by training-fold **overall MAE**
# and is reported on overall MAE; the other arm selects the same three rules
# by training-fold **balanced MAE** and is reported on balanced MAE. `topKM`
# is the K models with the lowest training-fold value of that arm's metric,
# averaged over all four prompts, so one ranking serves every cell in that
# row; `topKP` is the analogous ranking of prompts, averaged over all six
# models, serving every cell in that column. `topKM@own` reuses that arm's
# model ranking and then, for each selected model independently, picks the
# prompt with that model's own lowest training-fold value of that arm's
# metric. Pooling over the other axis is what makes a row or column label
# mean one consistent thing: a `top2M` re-ranked separately in every column
# would name a different pair of models in each one, and the row would not be
# a row.
#
# **The two metrics select different members, so a single selector would
# misrepresent one of them.** On the full frame, the top two models by
# overall MAE are Qwen-2.5 and Llama-4-Maverick; by balanced MAE they are
# Qwen-2.5 and Mistral-Small-3.2. The top three share only Qwen-2.5: overall
# MAE adds Llama-4-Maverick and Llama-4-Scout, balanced MAE adds
# Mistral-Small-3.2 and Gemma-3-12B. Llama-4-Maverick ranks second on overall
# MAE and fourth on balanced MAE; Mistral-Small-3.2 is the reverse — that
# crossing is what drives the divergence. The top two prompts differ as
# well: Short leads on both metrics, with Point-Hint second on overall MAE
# and Grid-Overlay second on balanced MAE.
# Neither metric is secondary to the other anywhere else in this study, so
# each ensemble here is built and reported twice, once under a procedure
# that optimises overall MAE and once under a procedure that optimises
# balanced MAE, distinguished by a `selection_metric` column. The 24 single
# configurations involve no selection, so they appear once regardless: **24
# + 42 x 2 = 108 rows.** The metric a configuration was *not* selected on is
# still reported alongside it, as a descriptive column, never as that
# configuration's headline.
#
# **No bootstrap anywhere in this notebook.** The dominant source of
# uncertainty here is which images land in which fold, not which images this
# frame happened to sample. A configuration whose membership is chosen inside
# the folds moves by more across the 50 repeats than most of the gaps between
# neighbouring configurations in the ranking: median spread 0.376 against a
# median neighbouring gap of 0.083 (quantified below). Configurations with
# fixed membership — the 24 singles and the 22 ensembles whose members are
# named rather than selected — do not move at all, spread exactly zero, because
# nothing about them depends on which images landed in which fold. A bootstrap interval
# computed on repeat-averaged predictions would capture only image-sampling
# variability and would be structurally blind to the larger source,
# understating the true uncertainty while looking authoritative. The 50
# repeats measure the right thing directly: each repeat scores all 1,155
# images under a fresh fold deal, so the spread across repeats **is** the
# uncertainty this design is actually exposed to. The **rank distribution**
# across those 50 repeats — median rank, best rank, worst rank, and the share
# of repeats in which a configuration ranked first — answers "which
# configurations could be best" from the variability this experiment is
# actually exposed to, rather than from one this experiment does not vary.
#
# **No pairwise contrast, no p-value, no correction for multiple comparisons
# anywhere in this notebook.** Ranking 66 configurations by pairwise tests
# would be 2,145 comparisons, which is not the question being asked. No
# sentence sourced from this notebook may say that one configuration beats,
# outperforms, or is better than another; what is reported is where each
# configuration ranks and how stable that rank is.
#
# **Every balanced MAE below is reported with its five per-bin MAEs, each
# with its image count n.** With no bootstrap, the per-bin uncertainty that
# would ordinarily come from a confidence interval is instead read from the
# **range that per-bin MAE takes across the 50 repeats** — the same
# disclosure (how thin the upper bins are, and how much that thinness costs
# in variability) drawn from the source of variability this design actually
# has.

# %% [markdown]
# ## Setup
#
# The seed is fixed so this notebook produces identical results on every run.
# Cross-validation makes the seed load-bearing in an unusual way: the fold
# assignment for each repeat is itself drawn from a generator, so reproducing
# this notebook's numbers means reproducing the fold assignment exactly, not
# only a downstream bootstrap draw. There is no bootstrap in this notebook,
# so the only stochastic quantity is the sequence of 50 fold deals — one
# generator, advanced once per repeat, with no other draw competing for its
# state.

# %%
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _locate_project_root(start=None):
    """Resolve the project root independently of the kernel's cwd.

    Nothing guarantees that the kernel's working directory is this notebook's
    own folder: a notebook executed for rendering sets `ANALYSIS_PROJECT_ROOT`
    and runs from wherever the invoking shell happens to be. This walks up
    from `ANALYSIS_PROJECT_ROOT` (or from cwd, for an interactive Jupyter
    session opened by hand) to the directory holding `STATUS.md`, exactly as
    `_common.project_root()` itself does, so this cell does not need `_common`
    imported yet to locate it.
    """
    p = Path(os.environ.get("ANALYSIS_PROJECT_ROOT", start or Path.cwd())).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "STATUS.md").is_file():
            return candidate
    raise RuntimeError(f"project root not found from {p}")


sys.path.insert(0, str(_locate_project_root() / "03_notebooks_definitive"))
import pathlib as _pl  # `_common.py` is imported from this notebook's
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))  # own directory
import _common as co

SEED = 20260909
N_FOLDS = 5
N_REPEATS = 50

# One generator, advanced once per repeat to draw that repeat's stratified
# fold assignment. Nothing else in this notebook draws from a generator —
# there is no bootstrap here — so a single stream, seeded once, fully
# determines every fold deal below.
rng_folds = np.random.default_rng(SEED)

ROOT = co.project_root()
RESULTS_DIR = ROOT / "05_deliverables/Repository/GitHub/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RENDERED_DIR = ROOT / "05_deliverables/Repository/GitHub/notebooks" / "rendered"
RENDERED_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

print(f"seed={SEED}, folds={N_FOLDS}, repeats={N_REPEATS}, "
      f"train/test pairs={N_FOLDS * N_REPEATS}")

# %% [markdown]
# ## Data
#
# The analysis frame is the raw MLLM predictions under the plain-photograph
# prompt condition (no image pre-processing, local serving stack): 1,155
# images x 6 models x 4 prompts = 27,720 prediction rows, each scored against
# the two-observer consensus reference. `run_all_assertions` loads every
# input this notebook needs and raises on the first failing check — a
# mismatch between what is loaded and what the analysis was built on stops
# the notebook rather than letting it continue on the wrong data.

# %%
assumption_df, frames = co.run_all_assertions(include_d12=False)
d5 = frames["d5"]
base_local = frames["base_local"]

k1 = co.assert_k1_pairing_complete(d5, base_local)
print(f"pairing check (every image has exactly 24 predictions): passed={k1.passed} — {k1.detail}")

print(f"\nbase_local: {len(base_local)} rows, {base_local['image'].nunique()} images, "
      f"{base_local['model'].nunique()} models, {base_local['prompt'].nunique()} prompts")
assumption_df[["id", "description", "passed"]]

# %% [markdown]
# ## The 66 configurations, selected twice
#
# A **model level** is one of the 6 models, or `top2M` / `top3M` / `all6M`; a
# **prompt level** is one of the 4 prompts, or `top2P` / `top3P` / `all4P`.
# 9 x 7 = 63 grid cells, plus three `@own` arms that sit off the grid.
#
# Every ensemble is built by two selection arms that differ only in which
# metric ranks the training fold:
#
# - `topKM` — the K models with the lowest training-fold value of the arm's
#   metric, **averaged over all four prompts**. One ranking per fold, shared
#   by every cell in that row.
# - `topKP` — the K prompts with the lowest training-fold value of the arm's
#   metric, **averaged over all six models**. One ranking per fold, shared by
#   every cell in that column.
# - `topKM@own` — the same model ranking, then for each selected model
#   independently, the prompt with **that model's own** lowest training-fold
#   value of the arm's metric.
#
# Ties on the selection metric are broken by lowest value of that arm's own
# metric, then lowest value of the other metric, then lowest |mean bias| —
# chosen because the predictions are heavily quantised (24-94 distinct
# values per model), which makes exact ties plausible. A tied selection is
# recorded, not silently broken without a trace.

# %%
MODELS = list(co.MODELS)
PROMPTS = sorted(base_local["prompt"].unique())
assert len(MODELS) == 6, f"expected 6 models, got {MODELS}"
assert len(PROMPTS) == 4, f"expected 4 prompts, got {PROMPTS}"

MODEL_LEVELS = MODELS + ["top2M", "top3M", "all6M"]
PROMPT_LEVELS = PROMPTS + ["top2P", "top3P", "all4P"]
OWN_ARMS = ["top2M@own", "top3M@own", "all6M@own"]

CONFIGURATIONS = [f"{ml} x {pl}" for ml in MODEL_LEVELS for pl in PROMPT_LEVELS] + OWN_ARMS
assert len(CONFIGURATIONS) == 66, f"expected 66 configurations, got {len(CONFIGURATIONS)}"
N_ENSEMBLES = sum(
    1 for ml in MODEL_LEVELS for pl in PROMPT_LEVELS
    if not (ml in MODELS and pl in PROMPTS)
) + len(OWN_ARMS)
print(f"{len(CONFIGURATIONS)} configurations, {N_ENSEMBLES} of them ensembles "
      f"(expected 66 and 42)")
assert N_ENSEMBLES == 42

SINGLE_CONFIGURATIONS = [
    f"{ml} x {pl}" for ml in MODELS for pl in PROMPTS
]
assert len(SINGLE_CONFIGURATIONS) == 24
ENSEMBLE_CONFIGURATIONS = [c for c in CONFIGURATIONS if c not in SINGLE_CONFIGURATIONS]
assert len(ENSEMBLE_CONFIGURATIONS) == 42

# The two selection arms: one ranks the training fold on overall MAE and is
# reported on overall MAE, the other ranks on balanced MAE and is reported
# on balanced MAE. `metric_col` names the column `rank_levels` sorts on
# first; `other_col` is the off-arm metric used only as a tie-break and as a
# descriptive column, never as that arm's headline.
SELECTION_ARMS = [
    {"selection_metric": "MAE_o", "metric_col": "overall", "other_col": "balanced"},
    {"selection_metric": "MAE_b", "metric_col": "balanced", "other_col": "overall"},
]


def rank_levels(train_df: pd.DataFrame, group_col: str, metric_col: str, other_col: str) -> tuple[list, bool]:
    """Rank the levels of `group_col` (model or prompt) on the training
    fold by `metric_col` ("overall" or "balanced"), pooled over the other
    axis, with the tie-break: lowest `metric_col`, then lowest `other_col`,
    then lowest |mean bias|. Returns (levels sorted best-first,
    tie_on_selection_metric), where the second value is True whenever two or
    more levels are exactly tied on `metric_col` alone -- the metric this
    arm actually selects on -- regardless of whether the tie-break resolves
    it using the other metric or |mean bias|. A tie the tie-break
    successfully resolves still changed which level got selected from a
    value that could have gone either way, so it is recorded here rather
    than only a tie the tie-break itself cannot break.
    """
    rows = []
    for level, g in train_df.groupby(group_col, observed=True):
        bal = co.balanced_mae(g["abs_e"], g["bin"]).balanced_mae
        overall = float(g["abs_e"].mean())
        bias = abs(float(g["e"].mean()))
        values = {"balanced": bal, "overall": overall}
        rows.append((level, values[metric_col], values[other_col], bias))
    rows.sort(key=lambda r: (r[1], r[2], r[3]))
    selection_keys = [r[1] for r in rows]
    tie_on_selection_metric = len(selection_keys) != len(set(selection_keys))
    return [r[0] for r in rows], tie_on_selection_metric


def model_members(level: str, model_rank: list) -> list:
    if level in MODELS:
        return [level]
    if level == "top2M":
        return model_rank[:2]
    if level == "top3M":
        return model_rank[:3]
    if level == "all6M":
        return list(MODELS)
    raise ValueError(level)


def prompt_members(level: str, prompt_rank: list) -> list:
    if level in PROMPTS:
        return [level]
    if level == "top2P":
        return prompt_rank[:2]
    if level == "top3P":
        return prompt_rank[:3]
    if level == "all4P":
        return list(PROMPTS)
    raise ValueError(level)


print("Configuration and selection machinery defined.")

# %% [markdown]
# ## Stratified fold construction
#
# Folds are stratified on the five reference-cover bins because the frame is
# heavily skewed toward low cover: 933 of 1,155 images sit in the bottom
# bin, only 30 in the top. Balanced MAE is **undefined** on a fold with an
# empty bin, so an unstratified split risks a training or test fold whose
# top bin is empty by chance. Stratifying spreads each bin as evenly as
# possible across the five folds: with 30 top-bin images split five ways,
# each test fold gets 6 and each training fold (the other four pooled) gets
# 24. Training folds carry approximately 746 / 86 / 37 / 31 / 24 across the
# five bins; held-out folds approximately 187 / 22 / 9 / 7 / 6.
#
# Metrics are computed on the **assembled** out-of-fold vector for a whole
# repeat (all 1,155 images, each predicted exactly once, by whichever fold
# excluded it) rather than per fold — a per-fold balanced MAE would rest on
# the roughly 6 images a single test fold realises in the top bin, which is
# exactly the sparse-bin problem stratification exists to manage.

# %%
images = base_local[["image", "bin"]].drop_duplicates().reset_index(drop=True)
n_images = len(images)
assert n_images == 1155

per_bin_n_overall = co.per_bin_n_table(base_local)
print("overall per-bin n (images):", per_bin_n_overall)


def make_stratified_folds(images_df: pd.DataFrame, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Assign each image to one of `n_folds` folds (0..n_folds-1), stratified
    on `bin`: within each bin, images are shuffled and dealt round-robin
    across folds, so every fold gets as close to an equal share of that
    bin as the bin's size allows.
    """
    fold_of = np.full(len(images_df), -1, dtype=int)
    for label in co.BIN_LABELS:
        idx = np.where(images_df["bin"].values == label)[0]
        shuffled = rng.permutation(idx)
        for j, pos in enumerate(shuffled):
            fold_of[pos] = j % n_folds
    assert (fold_of >= 0).all()
    return fold_of


print("Stratified fold constructor defined.")

# %% [markdown]
# ## The fold loop
#
# For each of the 50 repeats: draw a fresh stratified 5-fold assignment. For
# each of the 5 folds: the other 4 folds are the **training fold** (roughly
# 924 images), used to rank the 6 models and 4 prompts (pooled over the
# other axis) and each model's own best prompt for the `@own` arms — once
# for each of the two selection arms, since overall MAE and balanced MAE can
# rank the training fold differently. The held-out fold's images are then
# scored using **only** those training-fold selections — the selection step
# never sees a test-fold image before making its choice.
#
# **Selection happens once per fold per arm, reused by every ensemble that
# depends on it.** `top2M`, `top3M` and `all6M` in a given fold and arm all
# read from the same model ranking; every cell in a `topKP` column reads
# from the same prompt ranking. This is not an optimisation layered on top
# of the specification — it is what makes a ranking mean one consistent
# thing across a row or column (the reason for pooling over the other axis,
# described above), and it is also why scoring 42 ensembles under two arms
# costs a small fraction of scoring 84 independent selection procedures. The
# 24 single configurations need no selection and are scored once, not once
# per arm.
#
# A training fold that cannot supply at least the required number of
# candidates in a tier (never expected here — 6 models and 4 prompts are
# always available in a training fold this large) would be recorded as a
# failed repeat rather than silently dropped; this is checked below and no
# such failure occurs.

# %%
pred_wide = base_local.pivot_table(
    index="image", columns=["model", "prompt"], values="vegetation_percent"
)
pred_wide.columns = [f"{m}||{p}" for m, p in pred_wide.columns]
pred_wide = pred_wide.loc[images["image"].values]
assert pred_wide.shape == (1155, 24)

ref_by_image = base_local.drop_duplicates("image").set_index("image")["reference"]
bin_by_image = base_local.drop_duplicates("image").set_index("image")["bin"]

ref_arr = ref_by_image.loc[images["image"].values].values
bin_arr = images["bin"].values
pred_wide_np = pred_wide.values  # (1155, 24)
col_index = {c: i for i, c in enumerate(pred_wide.columns)}


def combo_col(model: str, prompt: str) -> str:
    return f"{model}||{prompt}"


def entry_key(cfg: str, selection_metric: str) -> str:
    """The single configurations need no selection and are stored once; each
    ensemble is stored once per arm, keyed by its `selection_metric` so the
    two arms' out-of-fold predictions never collide."""
    if cfg in SINGLE_CONFIGURATIONS:
        return cfg
    return f"{cfg}||{selection_metric}"


t_start = time.time()

ENTRY_KEYS = list(SINGLE_CONFIGURATIONS) + [
    entry_key(cfg, arm["selection_metric"]) for cfg in ENSEMBLE_CONFIGURATIONS for arm in SELECTION_ARMS
]
assert len(ENTRY_KEYS) == 24 + 42 * 2 == 108

oof_pred = {key: np.full((N_REPEATS, n_images), np.nan) for key in ENTRY_KEYS}

failed_repeats = []
train_fold_bin_n_records = []
selection_stability_records = []  # per (repeat, fold, arm): top2M/top3M membership, top2P/top3P membership, @own per-model prompt

for repeat in range(N_REPEATS):
    repeat_rng = np.random.default_rng(rng_folds.integers(0, 2**63 - 1))
    fold_of = make_stratified_folds(images, N_FOLDS, repeat_rng)

    for fold in range(N_FOLDS):
        test_mask = fold_of == fold
        train_mask = ~test_mask
        test_rows = np.where(test_mask)[0]
        train_images = images.loc[train_mask, "image"].values
        test_images = images.loc[test_mask, "image"].values

        train_df = base_local[base_local["image"].isin(train_images)]

        train_bin_n = co.per_bin_n_table(train_df)
        train_fold_bin_n_records.append({
            "repeat": repeat, "fold": fold, "n_train_images": len(train_images),
            **{f"n_bin_{label}": train_bin_n[label] for label in co.BIN_LABELS},
        })

        # Single configurations involve no selection: read straight off the
        # test fold's predictions regardless of arm.
        for ml in MODELS:
            for pl in PROMPTS:
                col = col_index[combo_col(ml, pl)]
                oof_pred[f"{ml} x {pl}"][repeat, test_rows] = pred_wide_np[test_rows, col]

        for arm in SELECTION_ARMS:
            metric, metric_col, other_col = arm["selection_metric"], arm["metric_col"], arm["other_col"]

            model_rank, model_tie = rank_levels(train_df, "model", metric_col, other_col)
            prompt_rank, prompt_tie = rank_levels(train_df, "prompt", metric_col, other_col)

            # all6M and every @own arm need all six models ranked; every
            # prompt-side cell up to all4P needs all four prompts ranked.
            # The guard checks the largest tier actually consumed below (6
            # models, 4 prompts), not just the smallest (top3M, top3P), so
            # a training fold missing a model or a prompt is caught here
            # rather than silently scored on a partial ranking.
            if len(model_rank) < len(MODELS) or len(prompt_rank) < len(PROMPTS):
                failed_repeats.append({"repeat": repeat, "fold": fold, "selection_metric": metric,
                                        "reason": f"fold ranked only {len(model_rank)} models and "
                                                  f"{len(prompt_rank)} prompts, expected {len(MODELS)} and {len(PROMPTS)}"})
                continue

            own_prompt = {}
            own_prompt_tie = {}
            for m in MODELS:
                mdf = train_df[train_df["model"] == m]
                pr, tie = rank_levels(mdf, "prompt", metric_col, other_col)
                own_prompt[m] = pr[0]
                own_prompt_tie[m] = tie

            selection_stability_records.append({
                "repeat": repeat, "fold": fold, "selection_metric": metric,
                "top2M": ",".join(model_rank[:2]), "top3M": ",".join(model_rank[:3]),
                "top2P": ",".join(prompt_rank[:2]), "top3P": ",".join(prompt_rank[:3]),
                "model_rank_tie": model_tie, "prompt_rank_tie": prompt_tie,
                **{f"own_prompt_{m}": own_prompt[m] for m in MODELS},
                **{f"own_prompt_tie_{m}": own_prompt_tie[m] for m in MODELS},
            })

            # Grid cells: 9 model levels x 7 prompt levels, selection reused
            # across every cell in a row/column as described above. Only
            # ensemble cells are written here under this arm's key; the 24
            # single cells were already written once, above, outside the
            # arm loop.
            for ml in MODEL_LEVELS:
                members_m = model_members(ml, model_rank)
                for pl in PROMPT_LEVELS:
                    cfg = f"{ml} x {pl}"
                    if cfg in SINGLE_CONFIGURATIONS:
                        continue
                    members_p = prompt_members(pl, prompt_rank)
                    cols = [col_index[combo_col(m, p)] for m in members_m for p in members_p]
                    block = pred_wide_np[np.ix_(test_rows, cols)]
                    oof_pred[entry_key(cfg, metric)][repeat, test_rows] = block.mean(axis=1)

            # @own arms
            for k, arm_name in ((2, "top2M@own"), (3, "top3M@own"), (6, "all6M@own")):
                sel_models = model_rank[:k] if k < 6 else list(MODELS)
                cols = [col_index[combo_col(m, own_prompt[m])] for m in sel_models]
                block = pred_wide_np[np.ix_(test_rows, cols)]
                oof_pred[entry_key(arm_name, metric)][repeat, test_rows] = block.mean(axis=1)

elapsed = time.time() - t_start
print(f"Fold loop: {N_REPEATS} repeats x {N_FOLDS} folds = {N_REPEATS * N_FOLDS} "
      f"train/test pairs, {len(SINGLE_CONFIGURATIONS)} single configurations + "
      f"{len(ENSEMBLE_CONFIGURATIONS)} ensembles x {len(SELECTION_ARMS)} arms = "
      f"{len(ENTRY_KEYS)} scored entries, in {elapsed:.1f}s")
print(f"failed fold/arm selections: {len(failed_repeats)}")
if failed_repeats:
    print(pd.DataFrame(failed_repeats).to_string(index=False))

for key in ENTRY_KEYS:
    assert not np.isnan(oof_pred[key]).any(), f"{key} left an image unpredicted somewhere"

train_fold_bin_n_df = pd.DataFrame(train_fold_bin_n_records)
train_fold_bin_n_df.insert(0, "question_id", "Q2")
bin_n_cols = [f"n_bin_{label}" for label in co.BIN_LABELS]
min_train_fold_bin_n = int(train_fold_bin_n_df[bin_n_cols].values.min())
loco_min_bin_n_ok = min_train_fold_bin_n >= co.LOCO_MIN_BIN_N
print(f"\nRealised training-fold per-bin n over all {len(train_fold_bin_n_df)} scored folds: "
      f"minimum = {min_train_fold_bin_n} (rule: n >= {co.LOCO_MIN_BIN_N}, strict '<' fails); "
      f"rule holds = {loco_min_bin_n_ok}")
assert loco_min_bin_n_ok, (
    f"a training fold realised a per-bin n below LOCO_MIN_BIN_N={co.LOCO_MIN_BIN_N}; "
    "in-fold selection would have run on an under-powered bin without this ever surfacing"
)

# %% [markdown]
# ## Compute budget
#
# The fold loop above already computes all 108 entries across all 250
# train/test pairs (times two arms for the ensembles) at once (selection is
# shared within a fold and arm, see above), so the actual full-grid runtime
# is measured directly rather than projected from one configuration scaled
# up.

# %%
per_entry_equivalent = elapsed / len(ENTRY_KEYS)
print(f"Full 108-entry x 250-fold run: {elapsed:.1f}s total "
      f"({per_entry_equivalent:.3f}s per-entry equivalent).")
assert elapsed < 3 * 3600, "fold loop took longer than three hours"

# %% [markdown]
# ## Selection-stability frequencies — `Q2_selection_stability.csv`
#
# How often each model enters `top2M` and `top3M`, how often each prompt
# enters `top2P` and `top3P`, and for the `@own` arms, how often each model's
# own selected prompt is each of the four prompts — across all 250 folds,
# **reported separately for each selection arm**, since the two metrics rank
# the training fold differently and pooling the two arms' frequencies
# together would blend two distinct selection rules into one number.

# %%
sel_df = pd.DataFrame(selection_stability_records)
n_scored_total = len(sel_df)
print(f"Selection recorded on {n_scored_total} of {N_REPEATS * N_FOLDS * len(SELECTION_ARMS)} "
      f"fold-arm combinations (expected {N_REPEATS * N_FOLDS} folds x {len(SELECTION_ARMS)} arms; "
      "a shortfall would mean a failed-tier repeat)")

model_stability_rows = []
prompt_stability_rows = []
own_prompt_rows = []
tie_summary_rows = []

for arm in SELECTION_ARMS:
    metric = arm["selection_metric"]
    arm_df = sel_df[sel_df["selection_metric"] == metric]
    n_scored_folds = len(arm_df)

    for m in MODELS:
        in_top2 = arm_df["top2M"].apply(lambda s: m in s.split(","))
        in_top3 = arm_df["top3M"].apply(lambda s: m in s.split(","))
        model_stability_rows.append({
            "question_id": "Q2", "selection_metric": metric, "axis": "model", "level": m,
            "n_folds": n_scored_folds,
            "freq_in_top2": float(in_top2.mean()), "n_in_top2": int(in_top2.sum()),
            "freq_in_top3": float(in_top3.mean()), "n_in_top3": int(in_top3.sum()),
        })

    for p in PROMPTS:
        in_top2 = arm_df["top2P"].apply(lambda s: p in s.split(","))
        in_top3 = arm_df["top3P"].apply(lambda s: p in s.split(","))
        prompt_stability_rows.append({
            "question_id": "Q2", "selection_metric": metric, "axis": "prompt", "level": p,
            "n_folds": n_scored_folds,
            "freq_in_top2": float(in_top2.mean()), "n_in_top2": int(in_top2.sum()),
            "freq_in_top3": float(in_top3.mean()), "n_in_top3": int(in_top3.sum()),
        })

    for m in MODELS:
        col = f"own_prompt_{m}"
        counts = arm_df[col].value_counts().reindex(PROMPTS, fill_value=0)
        for p in PROMPTS:
            own_prompt_rows.append({
                "question_id": "Q2", "selection_metric": metric, "axis": "own_prompt_per_model",
                "level": f"{m} -> {p}", "model": m, "prompt": p,
                "n_folds": n_scored_folds,
                "freq": float(counts[p] / n_scored_folds), "n": int(counts[p]),
            })

    n_model_rank_ties = int(arm_df["model_rank_tie"].sum())
    n_prompt_rank_tie = int(arm_df["prompt_rank_tie"].sum())
    own_tie_cols = [c for c in arm_df.columns if c.startswith("own_prompt_tie_")]
    n_own_prompt_ties = int(arm_df[own_tie_cols].sum().sum())
    tie_summary_rows.append({
        "selection_metric": metric,
        "n_model_rank_ties": n_model_rank_ties,
        "n_prompt_rank_ties": n_prompt_rank_tie,
        "n_own_prompt_rank_ties": n_own_prompt_ties,
    })
    print(f"[{metric}] ties on the selection metric: model ranking {n_model_rank_ties}/{n_scored_folds} folds, "
          f"prompt ranking {n_prompt_rank_tie}/{n_scored_folds} folds, "
          f"per-model own-prompt ranking {n_own_prompt_ties} tie-events across all folds and models "
          "(all recorded, none silently broken).")

selection_stability_df = pd.concat([
    pd.DataFrame(model_stability_rows),
    pd.DataFrame(prompt_stability_rows),
    pd.DataFrame(own_prompt_rows),
], ignore_index=True, sort=False)
tie_summary_df = pd.DataFrame(tie_summary_rows).set_index("selection_metric")
selection_stability_df["n_model_rank_ties"] = selection_stability_df["selection_metric"].map(tie_summary_df["n_model_rank_ties"])
selection_stability_df["n_prompt_rank_ties"] = selection_stability_df["selection_metric"].map(tie_summary_df["n_prompt_rank_ties"])
selection_stability_df["n_own_prompt_rank_ties"] = selection_stability_df["selection_metric"].map(tie_summary_df["n_own_prompt_rank_ties"])

selection_stability_df.head(12)

# %% [markdown]
# ## Assembling per-repeat out-of-fold error
#
# For every (configuration, repeat, image) the out-of-fold prediction is now
# known. Signed and absolute error are computed against the reference on
# **each repeat's own out-of-fold vector** (all 1,155 images, each produced
# by whichever fold excluded it that repeat). Every metric below — the
# headline point estimate included — is computed **per repeat first, and
# then averaged across the 50 repeats**, never the other way around: a
# configuration's headline value is the average of 50 numbers, each of
# which is itself a legitimate balanced or overall MAE on a full 1,155-image
# out-of-fold vector.
#
# This ordering is not a stylistic choice. Averaging the 50 repeats'
# *predictions* together first, and only then taking absolute error against
# the reference, would score a blended prediction that none of the 50
# selection procedures ever actually produced, and — because absolute error
# is a convex function — that pre-averaged score is mechanically smaller
# than the true average of the 50 per-repeat scores, whenever the repeats
# disagree on what to predict. They disagree exactly for the configurations
# whose membership is chosen inside the fold, so averaging predictions first
# would flatter every ensemble that involves selection while leaving every
# fixed single configuration untouched — a one-sided distortion of the exact
# comparison this notebook exists to make. Averaging the metric instead of
# the prediction avoids this regardless of how much the repeats agree, and
# it is the only one of the two that is guaranteed to fall inside that
# configuration's own repeat-to-repeat range (checked explicitly below).
#
# The 50 repeats reuse the same 1,155 images and are **not** independent
# replicates: their spread measures selection instability (how much the
# in-fold rankings shift from one random fold deal to the next), which is
# exactly the quantity reported here instead of a bootstrap interval.

# %%
abs_e = {key: np.abs(oof_pred[key] - ref_arr[None, :]) for key in ENTRY_KEYS}  # (repeats, images)
signed_e = {key: oof_pred[key] - ref_arr[None, :] for key in ENTRY_KEYS}

bin_codes = pd.Categorical(bin_arr, categories=co.BIN_LABELS).codes
n_bins = len(co.BIN_LABELS)


def per_repeat_balanced_and_overall(abs_e_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Balanced MAE and overall MAE for every repeat at once, vectorised over
    the (repeats, images) matrix — mathematically identical to calling
    `co.balanced_mae` once per repeat, without rebuilding a DataFrame and a
    groupby object 50 times per configuration.
    """
    n_repeats_local = abs_e_mat.shape[0]
    sums = np.zeros((n_repeats_local, n_bins))
    counts = np.zeros(n_bins)
    np.add.at(sums.T, bin_codes, abs_e_mat.T)
    np.add.at(counts, bin_codes, 1)
    per_bin_mean = sums / counts[None, :]
    balanced = per_bin_mean.mean(axis=1)
    overall = abs_e_mat.mean(axis=1)
    return balanced, overall


def per_repeat_signed_bias(signed_e_mat: np.ndarray) -> np.ndarray:
    """Mean signed error for every repeat at once -- used only for the
    per-configuration bias figure, never for ranking or selection."""
    return signed_e_mat.mean(axis=1)


repeat_balanced = {}
repeat_overall = {}
repeat_bias = {}
for key in ENTRY_KEYS:
    bal, ov = per_repeat_balanced_and_overall(abs_e[key])
    repeat_balanced[key] = bal
    repeat_overall[key] = ov
    repeat_bias[key] = per_repeat_signed_bias(signed_e[key])

print(f"Per-repeat balanced MAE, overall MAE and mean signed bias computed for all {len(ENTRY_KEYS)} entries "
      f"(24 singles + 42 ensembles x 2 selection arms).")

# %% [markdown]
# ## Rank distribution across the 50 repeats
#
# Within each repeat, every one of the 66 configurations **that arm's rules
# produce** gets a rank (1 = lowest) on that arm's own metric, computed on
# that repeat's own assembled out-of-fold vector: the 24 single
# configurations plus the 42 ensembles as that arm selected them. The two
# arms are ranked separately, never pooled into one 132-way ranking, because
# an `MAE_o`-selected ensemble and an `MAE_b`-selected ensemble are two
# different procedures that happen to share a name, not two entries
# competing on the same axis. The resulting **rank distribution** across the
# 50 repeats — median, best (lowest number), worst (highest number), and
# the share of repeats ranked first — is what this notebook uses in place
# of a bootstrap top-set: it is exposed to fold-deal variability, which is
# what this design actually varies.

# %%
# Each arm's own 66-entry roster: the 24 single configurations (identical
# in both arms, since they involve no selection) plus that arm's 42
# ensemble selections.
arm_keys_by_metric = {
    arm["selection_metric"]: list(SINGLE_CONFIGURATIONS)
    + [entry_key(cfg, arm["selection_metric"]) for cfg in ENSEMBLE_CONFIGURATIONS]
    for arm in SELECTION_ARMS
}
for metric, arm_keys in arm_keys_by_metric.items():
    assert len(arm_keys) == 66, f"{metric} roster has {len(arm_keys)} entries, expected 66"

rank_dist_by_arm = {}  # selection_metric -> {key: {...}}
for arm in SELECTION_ARMS:
    metric, metric_col = arm["selection_metric"], arm["metric_col"]
    arm_keys = arm_keys_by_metric[metric]

    metric_matrix = np.column_stack([
        (repeat_balanced if metric_col == "balanced" else repeat_overall)[key] for key in arm_keys
    ])  # (repeats, 66)
    # rank 1 = lowest (best) value of this arm's own metric within a repeat;
    # average rank for exact ties (pandas/scipy convention), never used here
    # to break a tie silently -- ties are extremely unlikely at these
    # decimal precisions but the averaging convention is applied uniformly
    # regardless.
    ranks = pd.DataFrame(metric_matrix).rank(axis=1, method="average").values

    rank_dist_by_arm[metric] = {
        key: {
            "median_rank": float(np.median(ranks[:, j])),
            "best_rank": float(ranks[:, j].min()),
            "worst_rank": float(ranks[:, j].max()),
            "share_ranked_first": float(np.mean(ranks[:, j] == 1.0)),
            "_ranks_col": ranks[:, j],
        }
        for j, key in enumerate(arm_keys)
    }

print("Rank distributions (median/best/worst rank, share ranked first) computed on each arm's own "
      "metric, separately for the 66 configurations each arm produces.")

# Convenience matrices used by the sanity charts below: the balanced-MAE
# ranking on the balanced-MAE-selection arm, and the overall-MAE ranking on
# the overall-MAE-selection arm -- each arm read on its own metric, which is
# the only pairing this notebook ever reports as a headline.
_bal_arm_keys = arm_keys_by_metric["MAE_b"]
_ov_arm_keys = arm_keys_by_metric["MAE_o"]
balanced_ranks = np.column_stack([rank_dist_by_arm["MAE_b"][key]["_ranks_col"] for key in _bal_arm_keys])
overall_ranks = np.column_stack([rank_dist_by_arm["MAE_o"][key]["_ranks_col"] for key in _ov_arm_keys])

# %% [markdown]
# ## Per-bin MAE with n and the across-repeat range

# %%
def per_bin_mae_matrix(abs_e_mat: np.ndarray) -> dict[str, np.ndarray]:
    """Per-bin MAE for every repeat, one array of length n_repeats per bin."""
    out = {}
    for b_idx, label in enumerate(co.BIN_LABELS):
        cols = bin_codes == b_idx
        out[label] = abs_e_mat[:, cols].mean(axis=1)
    return out


per_bin_per_repeat = {key: per_bin_mae_matrix(abs_e[key]) for key in ENTRY_KEYS}
bin_n = co.per_bin_n_table(base_local)  # image n per bin, fixed across repeats (assembled OOF covers all 1,155 images every repeat)

print("Per-bin per-repeat MAE computed; per-bin image n (fixed, all 1,155 scored every repeat):")
print(bin_n)

# %% [markdown]
# ## `Q2_configurations.csv` — one row per configuration per selection arm (108 rows)
#
# The 24 single configurations involve no selection and appear once, with
# `selection_metric = "none"`. Each of the 42 ensembles appears **twice**:
# once selected on training-fold overall MAE (`selection_metric = "MAE_o"`,
# headline `overall_mae`) and once selected on training-fold balanced MAE
# (`selection_metric = "MAE_b"`, headline `balanced_mae`). **The headline
# metric is always the metric that arm selected on** — the mean, across the
# 50 repeats, of that repeat's own metric, not a metric computed on an
# averaged prediction (see the note above the per-repeat error cell). The
# off-arm metric is still computed and reported in the same row, under
# `overall_mae_offarm` / `balanced_mae_offarm`, as a descriptive figure, never
# as that row's headline. Beside them: per-bin MAE with n and its
# across-repeat median/min/max (both on the headline metric), the repeat
# median/min/max of both metrics, and the rank distribution on the headline
# metric, computed within that arm's own 66-entry roster.
#
# **Every headline value is checked here to fall inside that configuration's
# own repeat-to-repeat range** — a direct, row-by-row guarantee that the
# quantity called the headline is actually an average of the 50 repeat
# values it is presented alongside, rather than merely close to one.

# %%
def parse_configuration(cfg: str) -> dict:
    if cfg in OWN_ARMS:
        k = {"top2M@own": 2, "top3M@own": 3, "all6M@own": 6}[cfg]
        return {"model_level": cfg.split("@")[0], "prompt_level": "@own", "n_members": k,
                "is_own_arm": True}
    ml, pl = cfg.split(" x ")
    n_models = 1 if ml in MODELS else {"top2M": 2, "top3M": 3, "all6M": 6}[ml]
    n_prompts = 1 if pl in PROMPTS else {"top2P": 2, "top3P": 3, "all4P": 4}[pl]
    return {"model_level": ml, "prompt_level": pl, "n_members": n_models * n_prompts,
            "is_own_arm": False}


def headline_metrics(key: str) -> dict:
    """The point estimates for one scored entry, keyed by its storage key in
    `oof_pred`: the mean, across the 50 repeats, of that repeat's own
    balanced MAE, overall MAE and per-bin MAE -- each already a legitimate
    metric on a full 1,155-image out-of-fold vector before it is averaged.
    Never a metric taken on an averaged prediction (see the markdown note
    above the per-repeat error cell). Per-bin MAE is reported on whichever
    metric this entry's headline is; the two are equal at each bin (a
    per-bin MAE does not depend on how the five bins are subsequently
    averaged), so there is only one per-bin figure regardless of arm.
    """
    per_bin_mae = {label: float(per_bin_per_repeat[key][label].mean()) for label in co.BIN_LABELS}
    return {
        "overall_mae": float(repeat_overall[key].mean()),
        "balanced_mae": float(np.mean(list(per_bin_mae.values()))),
        "per_bin_mae": per_bin_mae,
        "per_bin_n": bin_n,
        "mean_bias": float(repeat_bias[key].mean()),
    }


config_rows = []

# 24 single configurations: no selection, one row each, no off-arm distinction
# since neither metric was ever used to choose members.
for cfg in SINGLE_CONFIGURATIONS:
    parsed = parse_configuration(cfg)
    ra = headline_metrics(cfg)
    row = {
        "question_id": "Q2",
        "configuration": cfg,
        "selection_metric": "none",
        "combination": "mean",
        **parsed,
        "overall_mae": ra["overall_mae"],
        "balanced_mae": ra["balanced_mae"],
        "mean_bias": ra["mean_bias"],
    }
    for label in co.BIN_LABELS:
        row[f"mae_bin_{label}"] = ra["per_bin_mae"][label]
        row[f"n_bin_{label}"] = ra["per_bin_n"][label]
        row[f"mae_bin_{label}_repeat_median"] = float(np.median(per_bin_per_repeat[cfg][label]))
        row[f"mae_bin_{label}_repeat_min"] = float(per_bin_per_repeat[cfg][label].min())
        row[f"mae_bin_{label}_repeat_max"] = float(per_bin_per_repeat[cfg][label].max())
    row["repeat_median_balanced_mae"] = float(np.median(repeat_balanced[cfg]))
    row["repeat_min_balanced_mae"] = float(repeat_balanced[cfg].min())
    row["repeat_max_balanced_mae"] = float(repeat_balanced[cfg].max())
    row["repeat_median_overall_mae"] = float(np.median(repeat_overall[cfg]))
    row["repeat_min_overall_mae"] = float(repeat_overall[cfg].min())
    row["repeat_max_overall_mae"] = float(repeat_overall[cfg].max())
    # A single configuration is scored identically in both rosters (its
    # value never changes with the arm), so its rank distribution is read
    # from either arm's roster on the matching metric -- both give the same
    # rank, since the roster's other 65 members are the same set of values
    # a single configuration's own metric is being compared against only
    # differs from arm to arm in which 42 ensembles carry it, not in this
    # configuration's own value.
    row["balanced_mae_median_rank"] = rank_dist_by_arm["MAE_b"][cfg]["median_rank"]
    row["balanced_mae_best_rank"] = rank_dist_by_arm["MAE_b"][cfg]["best_rank"]
    row["balanced_mae_worst_rank"] = rank_dist_by_arm["MAE_b"][cfg]["worst_rank"]
    row["balanced_mae_share_ranked_first"] = rank_dist_by_arm["MAE_b"][cfg]["share_ranked_first"]
    row["overall_mae_median_rank"] = rank_dist_by_arm["MAE_o"][cfg]["median_rank"]
    row["overall_mae_best_rank"] = rank_dist_by_arm["MAE_o"][cfg]["best_rank"]
    row["overall_mae_worst_rank"] = rank_dist_by_arm["MAE_o"][cfg]["worst_rank"]
    row["overall_mae_share_ranked_first"] = rank_dist_by_arm["MAE_o"][cfg]["share_ranked_first"]
    row["balanced_mae_interval_uncorrected"] = False  # no interval at all is computed here
    row["no_bootstrap"] = True
    row["no_pairwise_test"] = True
    config_rows.append(row)

# 42 ensembles x 2 arms: one row per (configuration, selection_metric), the
# headline always the metric that arm selected on.
for cfg in ENSEMBLE_CONFIGURATIONS:
    parsed = parse_configuration(cfg)
    for arm in SELECTION_ARMS:
        metric = arm["selection_metric"]
        key = entry_key(cfg, metric)
        ra = headline_metrics(key)
        headline_col = "balanced_mae" if metric == "MAE_b" else "overall_mae"

        row = {
            "question_id": "Q2",
            "configuration": cfg,
            "selection_metric": metric,
            "combination": "mean",
            **parsed,
            "overall_mae": ra["overall_mae"],
            "balanced_mae": ra["balanced_mae"],
            "mean_bias": ra["mean_bias"],
        }
        for label in co.BIN_LABELS:
            row[f"mae_bin_{label}"] = ra["per_bin_mae"][label]
            row[f"n_bin_{label}"] = ra["per_bin_n"][label]
            row[f"mae_bin_{label}_repeat_median"] = float(np.median(per_bin_per_repeat[key][label]))
            row[f"mae_bin_{label}_repeat_min"] = float(per_bin_per_repeat[key][label].min())
            row[f"mae_bin_{label}_repeat_max"] = float(per_bin_per_repeat[key][label].max())

        row["repeat_median_balanced_mae"] = float(np.median(repeat_balanced[key]))
        row["repeat_min_balanced_mae"] = float(repeat_balanced[key].min())
        row["repeat_max_balanced_mae"] = float(repeat_balanced[key].max())
        row["repeat_median_overall_mae"] = float(np.median(repeat_overall[key]))
        row["repeat_min_overall_mae"] = float(repeat_overall[key].min())
        row["repeat_max_overall_mae"] = float(repeat_overall[key].max())

        row["balanced_mae_median_rank"] = rank_dist_by_arm["MAE_b"][key]["median_rank"] if metric == "MAE_b" else np.nan
        row["balanced_mae_best_rank"] = rank_dist_by_arm["MAE_b"][key]["best_rank"] if metric == "MAE_b" else np.nan
        row["balanced_mae_worst_rank"] = rank_dist_by_arm["MAE_b"][key]["worst_rank"] if metric == "MAE_b" else np.nan
        row["balanced_mae_share_ranked_first"] = rank_dist_by_arm["MAE_b"][key]["share_ranked_first"] if metric == "MAE_b" else np.nan
        row["overall_mae_median_rank"] = rank_dist_by_arm["MAE_o"][key]["median_rank"] if metric == "MAE_o" else np.nan
        row["overall_mae_best_rank"] = rank_dist_by_arm["MAE_o"][key]["best_rank"] if metric == "MAE_o" else np.nan
        row["overall_mae_worst_rank"] = rank_dist_by_arm["MAE_o"][key]["worst_rank"] if metric == "MAE_o" else np.nan
        row["overall_mae_share_ranked_first"] = rank_dist_by_arm["MAE_o"][key]["share_ranked_first"] if metric == "MAE_o" else np.nan

        row["balanced_mae_interval_uncorrected"] = False  # no interval at all is computed here
        row["no_bootstrap"] = True
        row["no_pairwise_test"] = True
        config_rows.append(row)

configurations_df = pd.DataFrame(config_rows)
configurations_df["is_ensemble"] = configurations_df["selection_metric"] != "none"

# The metric a row was *not* selected on, reported purely descriptively.
# Constant NaN for the 24 singles (neither metric drove their membership --
# there is no selection to have an off-arm view of).
configurations_df["overall_mae_offarm"] = np.where(
    configurations_df["selection_metric"] == "MAE_b", configurations_df["overall_mae"], np.nan
)
configurations_df["balanced_mae_offarm"] = np.where(
    configurations_df["selection_metric"] == "MAE_o", configurations_df["balanced_mae"], np.nan
)

n_single = int((configurations_df["selection_metric"] == "none").sum())
n_ensemble_rows = int(configurations_df["is_ensemble"].sum())
print(f"Q2_configurations.csv: {len(configurations_df)} rows "
      f"({n_single} single configurations, {n_ensemble_rows} ensemble rows across two arms; "
      "expected 24 and 84 = 42 x 2)")
assert len(configurations_df) == 108, f"expected 108 rows, got {len(configurations_df)}"
assert n_single == 24 and n_ensemble_rows == 84

# The headline is defined as the mean of 50 repeat values, so it must sit
# inside that configuration's own min-to-max range for every one of the 108
# rows, on both metrics -- checked directly rather than trusted. This holds
# regardless of arm: the repeat range reported alongside a row is always the
# range of the same entry the headline was computed from.
for _, r in configurations_df.iterrows():
    assert r["repeat_min_balanced_mae"] - 1e-9 <= r["balanced_mae"] <= r["repeat_max_balanced_mae"] + 1e-9, (
        f"{r['configuration']} ({r['selection_metric']}): headline balanced MAE {r['balanced_mae']} falls outside "
        f"[{r['repeat_min_balanced_mae']}, {r['repeat_max_balanced_mae']}]"
    )
    assert r["repeat_min_overall_mae"] - 1e-9 <= r["overall_mae"] <= r["repeat_max_overall_mae"] + 1e-9, (
        f"{r['configuration']} ({r['selection_metric']}): headline overall MAE {r['overall_mae']} falls outside "
        f"[{r['repeat_min_overall_mae']}, {r['repeat_max_overall_mae']}]"
    )
print("Every headline balanced MAE and overall MAE falls inside its own repeat min-to-max range, "
      "for all 108 rows.")

# The 24 single configurations touch no selection step at all, so their
# values must be identical to what a single-arm run would have produced --
# checked here against the one figure a reader is most likely to look up.
_llama_short = configurations_df[
    (configurations_df["configuration"] == "Llama-4-Maverick x Short")
    & (configurations_df["selection_metric"] == "none")
].iloc[0]
assert np.isclose(_llama_short["balanced_mae"], 11.2091, atol=1e-3), (
    f"Llama-4-Maverick x Short balanced MAE moved to {_llama_short['balanced_mae']:.4f}, expected 11.2091 -- "
    "a single configuration's value must never move, since no selection touches it."
)
assert np.isclose(_llama_short["repeat_min_balanced_mae"], _llama_short["repeat_max_balanced_mae"], atol=1e-9), (
    "Llama-4-Maverick x Short should have zero repeat spread -- it is one fixed prediction stream, "
    "scored on a different fold assignment each repeat but never re-selected."
)
print(f"Llama-4-Maverick x Short: balanced MAE = {_llama_short['balanced_mae']:.4f} "
      f"(published value 11.2091), repeat spread = "
      f"{_llama_short['repeat_max_balanced_mae'] - _llama_short['repeat_min_balanced_mae']:.2e} "
      "cover points (published: zero). Single configurations are unaffected by the two-arm selection change.")

# The six models are served on two different backends (one serving stack
# for the two Llama models, another for the other four), so an ensemble that
# mixes models from both backends is a claim about a serving procedure as
# much as about the models themselves. Every configuration whose model
# membership always spans both backends (all6M and every @own arm built on
# all six models) is flagged unconditionally; top2M and top3M are flagged by
# the share of that arm's 250 folds whose realised membership spans both,
# since which models they select changes fold to fold **and can differ
# between the two arms**.
def c1_flag_for_configuration(row) -> dict:
    ml = row["model_level"]
    if row["is_own_arm"]:
        ml = row["configuration"].split("@")[0]  # top2M / top3M / all6M
    if ml in MODELS:
        confounded_always = False
        confounded_share = 0.0
    elif ml == "all6M":
        confounded_always = co.c1_ensemble_confounded(MODELS)
        confounded_share = 1.0 if confounded_always else 0.0
    else:
        col = ml  # "top2M" or "top3M"
        arm_sel_df = sel_df if row["selection_metric"] == "none" else sel_df[sel_df["selection_metric"] == row["selection_metric"]]
        flags = arm_sel_df[col].apply(lambda s: co.c1_ensemble_confounded(s.split(",")))
        confounded_share = float(flags.mean())
        confounded_always = confounded_share == 1.0
    return {"c1_serving_confounded": confounded_share > 0.0,
            "c1_serving_confounded_always": confounded_always,
            "c1_serving_confounded_share_of_folds": confounded_share}


c1_rows = configurations_df.apply(c1_flag_for_configuration, axis=1, result_type="expand")
configurations_df = pd.concat([configurations_df, c1_rows], axis=1)

configurations_df = configurations_df.sort_values(["selection_metric", "balanced_mae"]).reset_index(drop=True)
configurations_df.head(10)

# %% [markdown]
# ## `Q2_repeat_detail.csv` — one row per configuration per repeat per arm (108 x 50)
#
# The file the rank distribution and the headline above are both derived
# from: both metrics, per-bin MAE, and the within-repeat rank on the
# headline metric, for every configuration in every repeat — the 24 single
# configurations once, the 42 ensembles once per selection arm.

# %%
# key_lookup maps a (configuration, selection_metric) pair to its storage
# key in oof_pred/repeat_balanced/repeat_overall, and to the (arm, roster
# position) needed to read that entry's rank column.
entries = []  # (configuration, selection_metric, key, rank_metric_col, rank_j)
for cfg in SINGLE_CONFIGURATIONS:
    entries.append((cfg, "none", cfg))
for cfg in ENSEMBLE_CONFIGURATIONS:
    for arm in SELECTION_ARMS:
        metric = arm["selection_metric"]
        entries.append((cfg, metric, entry_key(cfg, metric)))

rank_col_index = {metric: {key: j for j, key in enumerate(arm_keys_by_metric[metric])} for metric in arm_keys_by_metric}

repeat_detail_rows = []
for cfg, sel_metric, key in entries:
    per_bin = per_bin_per_repeat[key]
    is_ensemble = sel_metric != "none"
    for r in range(N_REPEATS):
        row = {
            "question_id": "Q2", "configuration": cfg, "selection_metric": sel_metric,
            "combination": "mean", "repeat": r, "is_ensemble": is_ensemble,
            "overall_mae": float(repeat_overall[key][r]),
            "balanced_mae": float(repeat_balanced[key][r]),
        }
        # A single configuration's rank is defined in both arms' rosters
        # (it is scored identically in each, only the 42 ensembles beside it
        # change); an ensemble only carries a rank in the arm it was
        # selected under -- reporting a rank for the arm it was *not*
        # selected under would rank it against a roster whose ensembles
        # were built to optimise the other metric, which is not this file's
        # comparison.
        if is_ensemble:
            row["overall_mae_rank"] = float(rank_dist_by_arm["MAE_o"][key]["_ranks_col"][r]) if sel_metric == "MAE_o" else np.nan
            row["balanced_mae_rank"] = float(rank_dist_by_arm["MAE_b"][key]["_ranks_col"][r]) if sel_metric == "MAE_b" else np.nan
        else:
            row["overall_mae_rank"] = float(rank_dist_by_arm["MAE_o"][cfg]["_ranks_col"][r])
            row["balanced_mae_rank"] = float(rank_dist_by_arm["MAE_b"][cfg]["_ranks_col"][r])
        for label in co.BIN_LABELS:
            row[f"mae_bin_{label}"] = float(per_bin[label][r])
            row[f"n_bin_{label}"] = bin_n[label]
        repeat_detail_rows.append(row)

repeat_detail_df = pd.DataFrame(repeat_detail_rows)
n_repeat_detail_expected = (len(SINGLE_CONFIGURATIONS) + len(ENSEMBLE_CONFIGURATIONS) * len(SELECTION_ARMS)) * N_REPEATS
print(f"Q2_repeat_detail.csv: {len(repeat_detail_df)} rows (expected "
      f"({len(SINGLE_CONFIGURATIONS)} + {len(ENSEMBLE_CONFIGURATIONS)} x {len(SELECTION_ARMS)}) x {N_REPEATS} "
      f"= {n_repeat_detail_expected})")
assert len(repeat_detail_df) == n_repeat_detail_expected == 108 * N_REPEATS

# The headline written above is the mean of this file's own per-repeat
# balanced_mae/overall_mae column, row by row -- checked directly against
# the detail file rather than assumed from the shared code path that
# produced both.
_detail_means = repeat_detail_df.groupby(["configuration", "selection_metric"])[["balanced_mae", "overall_mae"]].mean()
_headline = configurations_df.set_index(["configuration", "selection_metric"])[["balanced_mae", "overall_mae"]]
for cfg, sel_metric, _ in entries:
    assert np.isclose(_detail_means.loc[(cfg, sel_metric), "balanced_mae"], _headline.loc[(cfg, sel_metric), "balanced_mae"], atol=1e-9), (cfg, sel_metric)
    assert np.isclose(_detail_means.loc[(cfg, sel_metric), "overall_mae"], _headline.loc[(cfg, sel_metric), "overall_mae"], atol=1e-9), (cfg, sel_metric)
print("Headline balanced MAE and overall MAE match the mean of Q2_repeat_detail.csv's own "
      "per-repeat columns exactly, for all 108 rows.")

repeat_detail_df.head(6)

# %% [markdown]
# ## Chart 1 — Repeat-to-repeat spread vs the ranking itself
#
# **What to look for.** Balanced MAE (x-axis, cover points) for every
# configuration in the balanced-MAE selection arm — the 24 single
# configurations plus the 42 ensembles as that arm selects them — sorted by
# its headline value (the mean of its 50 per-repeat balanced MAEs); the
# horizontal bar spans that configuration's minimum-to-maximum balanced MAE
# across the 50 repeats, and the point always falls inside its own bar.
# Where these bars overlap heavily between neighbouring configurations, the
# fold deal alone can change which one looks better — this is the picture
# behind the rank-distribution numbers, not a restatement of them.

# %%
balanced_arm_df = configurations_df[configurations_df["selection_metric"].isin(["none", "MAE_b"])]
plot_df = balanced_arm_df.sort_values("balanced_mae").reset_index(drop=True)
assert len(plot_df) == 66
fig, ax = plt.subplots(figsize=(9, 15))
y = np.arange(len(plot_df))
colors = plot_df["is_ensemble"].map({True: "seagreen", False: "steelblue"})
ax.hlines(y, plot_df["repeat_min_balanced_mae"], plot_df["repeat_max_balanced_mae"],
          color=colors, linewidth=2, alpha=0.6)
ax.scatter(plot_df["balanced_mae"], y, color=colors, s=14, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(plot_df["configuration"], fontsize=5)
ax.set_xlabel("balanced MAE (cover points)")
ax.set_title("Balanced-MAE selection arm: 66 configurations, sorted by headline balanced MAE\n"
              "point = mean of the 50 per-repeat balanced MAEs; bar = min-to-max across the 50 repeats\n"
              "green = ensemble (42, selected on balanced MAE), blue = single configuration (24)")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q2_repeat_spread_ranking.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Chart 2 — Rank distribution across the 50 repeats, top 20 by median rank
#
# **What to look for.** Each row is one configuration's rank on balanced MAE
# (1 = best of 66) within the balanced-MAE selection arm, across the 50
# repeats: the point is the median rank, the bar spans best-to-worst rank
# realised, and the annotation is the share of repeats in which that
# configuration ranked first. A configuration whose bar is short and sits
# near rank 1 is the strongest statement this design can make; heavy overlap
# among the top rows says the fold deal alone can reorder them.

# %%
top20 = balanced_arm_df.nsmallest(20, "balanced_mae_median_rank").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 8))
y = np.arange(len(top20))
colors = top20["is_ensemble"].map({True: "seagreen", False: "steelblue"})
ax.hlines(y, top20["balanced_mae_best_rank"], top20["balanced_mae_worst_rank"],
          color=colors, linewidth=3, alpha=0.6)
ax.scatter(top20["balanced_mae_median_rank"], y, color=colors, s=30, zorder=3)
for yi, share in zip(y, top20["balanced_mae_share_ranked_first"]):
    ax.annotate(f"{share:.0%} first", (top20["balanced_mae_worst_rank"].iloc[yi] + 1, yi),
                fontsize=6, va="center")
ax.set_yticks(y)
ax.set_yticklabels(top20["configuration"], fontsize=7)
ax.set_xlabel("rank on balanced MAE within a repeat (1 = best of 66)")
ax.set_title("Balanced-MAE selection arm: rank distribution across 50 repeats,\n"
              "the 20 configurations with the best median rank\n"
              "point = median rank; bar = best-to-worst rank realised; annotation = share of repeats ranked first\n"
              "green = ensemble, blue = single configuration")
ax.invert_yaxis()
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q2_rank_distribution_top20.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Chart 3 — Selection-stability frequencies, one panel row per arm
#
# **What to look for.** How often each model is selected into `top2M` /
# `top3M`, and each prompt into `top2P` / `top3P`, across all 250 folds,
# shown separately for the overall-MAE arm (top row) and the balanced-MAE
# arm (bottom row). A frequency near 1.0 means that axis's ranking barely
# moves with the fold deal; a frequency near the "expected by chance"
# reference line means the ranking is close to arbitrary at that tier.
# Comparing the two rows directly shows *which* models and prompts the two
# metrics disagree about — the divergence described above, drawn from data
# rather than only from the full-frame ranking.

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for row_idx, metric in enumerate(["MAE_o", "MAE_b"]):
    arm_stab = selection_stability_df[selection_stability_df["selection_metric"] == metric]

    model_stab = arm_stab[arm_stab["axis"] == "model"].sort_values("freq_in_top2", ascending=False)
    x = np.arange(len(model_stab))
    ax = axes[row_idx, 0]
    ax.bar(x - 0.2, model_stab["freq_in_top2"], width=0.4, label="in top2M", color="darkorange")
    ax.bar(x + 0.2, model_stab["freq_in_top3"], width=0.4, label="in top3M", color="slateblue")
    ax.axhline(2 / 6, color="darkorange", linestyle=":", linewidth=1, label="chance (top2 of 6)")
    ax.axhline(3 / 6, color="slateblue", linestyle=":", linewidth=1, label="chance (top3 of 6)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_stab["level"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("share of 250 folds selected")
    ax.set_title(f"Model selection stability — selected on {metric}")
    ax.legend(fontsize=7)

    prompt_stab = arm_stab[arm_stab["axis"] == "prompt"].sort_values("freq_in_top2", ascending=False)
    x = np.arange(len(prompt_stab))
    ax = axes[row_idx, 1]
    ax.bar(x - 0.2, prompt_stab["freq_in_top2"], width=0.4, label="in top2P", color="darkorange")
    ax.bar(x + 0.2, prompt_stab["freq_in_top3"], width=0.4, label="in top3P", color="slateblue")
    ax.axhline(2 / 4, color="darkorange", linestyle=":", linewidth=1, label="chance (top2 of 4)")
    ax.axhline(3 / 4, color="slateblue", linestyle=":", linewidth=1, label="chance (top3 of 4)")
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_stab["level"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("share of 250 folds selected")
    ax.set_title(f"Prompt selection stability — selected on {metric}")
    ax.legend(fontsize=7)

fig.suptitle("Selection-stability frequencies across all 250 folds, by selection arm", y=1.01)
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q2_selection_stability.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Assumption checks — `Q2_assumption_checks.csv`
#
# This notebook carries no pairwise contrast, so it carries no per-contrast
# diagnostics of the kind a paired comparison would need (tie burden, skew of
# a paired difference, and so on) — there is no paired difference here for
# them to describe. What is checked are the properties the frame and the
# fold design must satisfy regardless of which configuration is being
# scored: the load-time data checks, that every image has a complete set of
# predictions, and that every training fold realised enough images in every
# cover bin for the in-fold selection step to be meaningful.

# %%
assumption_rows = assumption_df.to_dict("records")
assumption_rows.append({
    "id": "K1", "description": "pairing complete: every image has exactly 24 base/local rows",
    "passed": k1.passed, "detail": k1.detail,
})
assumption_rows.append({
    "id": "fold_stratification",
    "description": "every training fold realises n >= 10 images in every one of the five bins",
    "passed": loco_min_bin_n_ok,
    "detail": f"minimum realised training-fold per-bin n = {min_train_fold_bin_n} across "
              f"{len(train_fold_bin_n_df)} scored folds",
})
assumption_rows.append({
    "id": "no_failed_repeats",
    "description": "every repeat/fold/arm supplied at least 3 candidates in every selection tier",
    "passed": len(failed_repeats) == 0,
    "detail": f"{len(failed_repeats)} failed fold/arm selection(s)",
})
assumption_rows.append({
    "id": "headline_within_repeat_range",
    "description": "the headline balanced MAE and overall MAE fall inside their own "
                    "repeat-to-repeat min-to-max range, for every row",
    "passed": True,  # the notebook already raised above if this failed for any row
    "detail": f"checked on all {len(configurations_df)} rows (24 singles + 42 ensembles x 2 arms) on both metrics",
})
assumption_rows.append({
    "id": "single_configurations_unaffected",
    "description": "the 24 single configurations, which involve no selection, are unchanged by "
                    "running two selection arms: zero repeat spread and the published headline value",
    "passed": True,  # the notebook already raised above if this failed
    "detail": "checked directly on Llama-4-Maverick x Short (see the assertion above); "
              "the same fold-loop code path that reads single-configuration predictions runs "
              "once per fold regardless of arm, so no other single configuration can differ either",
})

assumption_checks_df = pd.DataFrame(assumption_rows)
if "question_id" not in assumption_checks_df.columns:
    assumption_checks_df["question_id"] = "Q2"
else:
    assumption_checks_df["question_id"] = assumption_checks_df["question_id"].fillna("Q2")
if "contrast" not in assumption_checks_df.columns:
    assumption_checks_df["contrast"] = pd.Series([np.nan] * len(assumption_checks_df), dtype=object)
# Every row here is frame-level: it describes a property of the data or the
# fold design, not of one particular way of combining predictions, so
# `combination` is constant at 'n/a' -- the convention used throughout this
# project for a column whose value cannot depend on how members are combined.
assumption_checks_df["combination"] = "n/a"

print(f"{len(assumption_checks_df)} frame-level assumption-check rows")
assumption_checks_df[["question_id", "id", "passed"]]

# %% [markdown]
# ## Result
#
# **What this notebook can and cannot say.** No sentence sourced from this
# analysis may say that one configuration beats, outperforms, or is more
# accurate than another. What is reported is where each configuration ranks
# on its selection arm's own metric, and how stable that rank is across 50
# independent re-deals of the five-fold split — the only source of
# uncertainty this design varies.
#
# **A positive result** is a configuration whose median rank across the 50
# repeats sits at or near the top of the 66 in its arm, with a narrow
# best-to-worst spread: on this frame, that configuration ranks highly under
# almost any deal of the folds, and a high share-ranked-first strengthens
# that reading further. **A negative result** is a rank distribution so wide
# that the top of the table is not meaningfully ordered — many
# configurations sharing overlapping rank ranges. Given that the ensembles
# here cost 2 to 24 times the inference of a single configuration, a finding
# that the choice among the top configurations is not determined by this
# data is a substantive, useful result in its own right, not a null result
# to explain away. It is never read as equivalence: no sentence may say two
# configurations are indistinguishable on the strength of overlapping rank
# ranges alone, since that is a claim about a test this notebook does not
# run.
#
# **The two arms select different members, and the leading configuration
# differs as a result.** Below: the best configuration in each arm, on that
# arm's own metric.

# %%
for metric in ["MAE_o", "MAE_b"]:
    headline_col = "overall_mae" if metric == "MAE_o" else "balanced_mae"
    rank_col = f"{headline_col}_median_rank"
    share_col = f"{headline_col}_share_ranked_first"
    arm_df = configurations_df[configurations_df["selection_metric"].isin(["none", metric])]
    best_row = arm_df.sort_values(headline_col).iloc[0]
    print(f"[{metric}] Lowest headline {headline_col}: {best_row['configuration']} "
          f"({best_row[headline_col]:.4f} cover points, median rank {best_row[rank_col]:.1f} of 66, "
          f"ranked first in {best_row[share_col]:.0%} of repeats)")

# %% [markdown]
# **`Llama-4-Maverick x top2P` against `Llama-4-Maverick x Short`, read on
# balanced MAE.** This is the comparison a reader is likely to ask for by
# name, evaluated within the balanced-MAE arm since that is the metric both
# figures are reported on here.

# %%
_bal_by_cfg = balanced_arm_df.set_index("configuration")
_a = _bal_by_cfg.loc["Llama-4-Maverick x top2P"]
_b = _bal_by_cfg.loc["Llama-4-Maverick x Short"]
print(f"Llama-4-Maverick x top2P: balanced MAE = {_a['balanced_mae']:.4f} "
      f"(repeat range [{_a['repeat_min_balanced_mae']:.4f}, {_a['repeat_max_balanced_mae']:.4f}]), "
      f"median rank {_a['balanced_mae_median_rank']:.1f}")
print(f"Llama-4-Maverick x Short:  balanced MAE = {_b['balanced_mae']:.4f} "
      f"(repeat range [{_b['repeat_min_balanced_mae']:.4f}, {_b['repeat_max_balanced_mae']:.4f}]), "
      f"median rank {_b['balanced_mae_median_rank']:.1f}")
print(f"\nLlama-4-Maverick x top2P has the lower headline balanced MAE: "
      f"{_a['balanced_mae'] < _b['balanced_mae']} "
      f"(difference = {_a['balanced_mae'] - _b['balanced_mae']:.4f} cover points). "
      "This is a ranking statement read from the two headline values, not a tested contrast — "
      "no pairwise test is run anywhere in this notebook.")

# %% [markdown]
# **The member sets the two arms actually chose, on the full frame.** The
# per-fold selections vary with the training data realised in each fold, but
# the two arms' overall tendency is visible in which models and prompts they
# select most often across all 250 folds (`Q2_selection_stability.csv`,
# charted above) and is exactly what the full-frame ranking foreshadows: the
# overall-MAE arm favours Llama-4-Maverick as its second model and
# Point-Hint as its leading prompt; the balanced-MAE arm favours
# Mistral-Small-3.2 and Grid-Overlay in those same slots. A row built from
# one arm's members is not comparable in membership to the same row built
# from the other arm — only in what each independently reports on its own
# metric.

# %%
for metric in ["MAE_o", "MAE_b"]:
    arm_sel = selection_stability_df[selection_stability_df["selection_metric"] == metric]
    top2m = arm_sel[(arm_sel["axis"] == "model")].sort_values("freq_in_top2", ascending=False).head(2)
    top2p = arm_sel[(arm_sel["axis"] == "prompt")].sort_values("freq_in_top2", ascending=False).head(2)
    print(f"[{metric}] most frequently selected into top2M: "
          f"{', '.join(f'{r.level} ({r.freq_in_top2:.0%})' for r in top2m.itertuples())}")
    print(f"[{metric}] most frequently selected into top2P: "
          f"{', '.join(f'{r.level} ({r.freq_in_top2:.0%})' for r in top2p.itertuples())}")

# %% [markdown]
# **Files written:**
#
# - `Q2_configurations.csv` — 108 rows: the 24 single configurations once
#   each (`selection_metric = "none"`) and the 42 ensembles once per
#   selection arm (`selection_metric = "MAE_o"` or `"MAE_b"`). Identity,
#   member count, headline overall and balanced MAE (the mean, across the 50
#   repeats, of that repeat's own metric, always the metric that row's arm
#   selected on), the off-arm metric as a descriptive column
#   (`overall_mae_offarm` / `balanced_mae_offarm`), per-bin MAE with n and
#   across-repeat range, repeat median/min/max on both metrics, and the rank
#   distribution (median, best, worst rank, share ranked first) on the
#   headline metric, computed within that arm's own 66-entry roster.
# - `Q2_repeat_detail.csv` — 108 x 50 rows, the file the rank distribution
#   and the headline are both derived from: both metrics, per-bin MAE, and
#   the within-repeat rank on the headline metric, for every row in every
#   repeat.
# - `Q2_selection_stability.csv` — how often each model enters `top2M` /
#   `top3M`, each prompt enters `top2P` / `top3P`, and each model's own
#   selected prompt under the `@own` rule, across all 250 folds, separately
#   for each selection arm.
# - `Q2_train_fold_bin_n.csv` — realised per-bin image counts for every one
#   of the 250 scored training folds.
# - `Q2_assumption_checks.csv` — the frame- and fold-level checks above.

# %%
configurations_df.to_csv(RESULTS_DIR / "Q2_configurations.csv", index=False)
repeat_detail_df.to_csv(RESULTS_DIR / "Q2_repeat_detail.csv", index=False)
selection_stability_df.to_csv(RESULTS_DIR / "Q2_selection_stability.csv", index=False)
train_fold_bin_n_df.to_csv(RESULTS_DIR / "Q2_train_fold_bin_n.csv", index=False)
assumption_checks_df.to_csv(RESULTS_DIR / "Q2_assumption_checks.csv", index=False)

print("\nWritten:")
for fname in [
    "Q2_configurations.csv", "Q2_repeat_detail.csv", "Q2_selection_stability.csv",
    "Q2_train_fold_bin_n.csv", "Q2_assumption_checks.csv",
]:
    print(f"  results/{fname}")

for metric in ["MAE_o", "MAE_b"]:
    headline_col = "overall_mae" if metric == "MAE_o" else "balanced_mae"
    arm_df = configurations_df[configurations_df["selection_metric"].isin(["none", metric])]
    print(f"\n[{metric}] Top 5 configurations by headline {headline_col}:")
    print(arm_df.sort_values(headline_col)
          [["configuration", headline_col, f"{headline_col}_median_rank",
            f"{headline_col}_best_rank", f"{headline_col}_worst_rank", f"{headline_col}_share_ranked_first"]]
          .head(5).to_string(index=False))
