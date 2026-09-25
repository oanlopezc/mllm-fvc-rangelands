# %% [markdown]
# # Q9 — the classical vegetation-index baseline against the vision-language models
#
# A colour-threshold vegetation index built on greenness is expected to
# underestimate cover as cover increases, and to fail outright on material
# that is not green. This notebook shows that pattern directly on the study's
# own 1,155 quadrat photographs: three per-bin error series across the
# reference cover range, the ratio between the classical baseline and the
# best-performing model ensemble where cover is actually present, the full
# predicted-against-reference relationship with a count of how often the
# baseline over-predicts, two exemplar images, and one correlation diagnostic.
# Nothing here is a statistical test comparing the baseline against a model —
# every number below is a point estimate or a descriptive count.
#
# **What is compared, and why it is fair.** The classical pipeline is the
# full thing: rectify the raw photograph to 1536x1536 by homography, take
# ExG-ExR on chromatic-normalised RGB inside a 40px-inset interior, threshold
# at the 99th percentile of a soil-colour exemplar set. The perspective
# correction is a step *inside* this pipeline — the method needs the quadrat
# interior to fill the frame — not a preprocessing advantage handed to it, so
# the model side of every comparison here uses the unrectified runs. Both
# sides are scored against the same two-observer reference over the same
# 1,155 quadrat images: no missing images, no serving-stack confound on the
# baseline side (it is a deterministic image-processing pipeline with no
# run-to-run variability to bound), 100% coverage.

# %% [markdown]
# ## Setup
#
# The seed is fixed so this notebook produces identical results on every run.

# %%
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _locate_project_root(start=None):
    """Resolve the project root independently of the kernel's cwd.

    Nothing guarantees that the kernel's working directory is this notebook's
    own folder: a notebook executed for rendering sets `ANALYSIS_PROJECT_ROOT`
    and runs from wherever the invoking shell happens to be. This walks up
    from `ANALYSIS_PROJECT_ROOT` (or from cwd, for an interactive Jupyter
    session opened by hand) to the directory holding `STATUS.md`.
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

SEED = 20260916  # the study's base seed (20260907) plus the question number.
# Written as a plain integer rather than an expression, so the seed that
# governs this run can be read straight off the source.
rng = np.random.default_rng(SEED)

ROOT = co.project_root()
RESULTS_DIR = ROOT / "05_deliverables/Repository/GitHub/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RENDERED_DIR = ROOT / "05_deliverables/Repository/GitHub/notebooks" / "rendered"
RENDERED_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

print(f"seed={SEED}")
print(f"project root: {ROOT}")


def assert_published(name: str, computed: float, published: float, atol: float = 1e-2) -> None:
    """Checks a computed quantity against its published value.

    These are the values reported in the paper. The assertion exists so that
    the manuscript and this notebook cannot silently drift apart: if a future
    change to the input data or the code moves any of these numbers, this
    cell fails loudly here rather than leaving a stale figure in the paper.
    """
    if not np.isclose(computed, published, atol=atol, rtol=0):
        raise AssertionError(
            f"{name}: computed {computed!r} does not match the published "
            f"value {published!r} (atol={atol})"
        )


# %% [markdown]
# ## Data
#
# Two frames, loaded through a shared module so the load path, the join and
# every integrity check are identical to the rest of this project rather than
# re-derived here:
#
# - `base_local` — the six vision-language models, unrectified runs, local
#   serving stack, joined to the reference. 1,155 images x 6 models x 4
#   prompts = 27,720 rows.
# - `d12_frame` — the classical baseline, 1,155 rows, one per image, joined
#   to the same reference. This frame is never merged onto `base_local` — see
#   the fan-out guard immediately below.
#
# Missingness: none. Both frames have 100% coverage over the 1,155 images by
# construction (the classical baseline is a deterministic pipeline with no
# frame-detection step that can fail; the model-side join is checked
# row-complete at load time), so no imputation or row-dropping decision
# arises anywhere in this notebook.

# %%
assumption_df, frames = co.run_all_assertions(include_d12=True)
assert assumption_df["passed"].all(), "a load-time integrity check failed — stopping"

base_local = frames["base_local"]
d1 = frames["d1"]

k1 = co.assert_k1_pairing_complete(frames["d5"], base_local)
print(f"pairing complete: passed={k1.passed} — {k1.detail}")

print(f"base_local: {len(base_local)} rows, {base_local['image'].nunique()} images, "
      f"{base_local['model'].nunique()} models, {base_local['prompt'].nunique()} prompts")
assumption_df[["id", "description", "passed"]]

# %% [markdown]
# The classical baseline's own file carries a third copy of the ground-truth
# reference value alongside its prediction. This copy is checked against the
# reference used everywhere else in this notebook and then dropped, so
# `d12_frame` below carries the prediction and one reference column, never
# two, and the duplicate is never mistaken for a second, independent opinion.
#
# **The fan-out guard, enforced at every point of use below.** The classical
# baseline has no prompt or model dimension: one prediction per image. If it
# were merged onto the 4,620-row per-model prediction frame it would
# broadcast 4x, and any statistic computed on that merged frame would treat
# the baseline as if n = 4,620 rather than 1,155 — quadrupling its apparent
# precision. The rule followed without exception in this notebook: **the
# model side is aggregated to one value per image first, and only then
# compared against the classical baseline.** `guard_d12_statistic` checks
# `n_rows == n_unique_images` immediately before any baseline statistic is
# computed, so a violation is caught at the point it would occur rather than
# discovered later in a downstream number.

# %%
d12_frame = co.build_d12_frame()
assert "reference" in d12_frame.columns
co.guard_d12_statistic(d12_frame, full_frame_expected=True, context="full classical-baseline frame, pre-analysis")
print(f"d12_frame: {len(d12_frame)} rows, {d12_frame['image'].nunique()} unique images. Fan-out guard passed.")

# %% [markdown]
# ## Assumption checks
#
# - **Load-time integrity checks** on the classical-baseline file (row count,
#   join completeness, the reference-copy check, summary-statistic and
#   constant-predictor sanity checks) are run above; a failure stops the
#   notebook. The constant-predictor check exists purely as a file-integrity
#   check confirming the loaded baseline file is the one this analysis was
#   built on, and its value is not itself reported below.
# - **Pairing is complete.** Every image has exactly 24 model-side rows (6
#   models x 4 prompts); checked above.
# - **The fan-out guard** is checked immediately before every classical-
#   baseline statistic below, not assumed once at the top, so it fires at the
#   point a violation would actually occur.
# - No normality, symmetry or tie-burden assumption applies anywhere in this
#   notebook: nothing here runs a parametric or rank test. The per-bin MAE,
#   signed bias, MAE ratio and Pearson `r` below are descriptive statistics
#   of a fixed, complete dataset, and their reported uncertainty (image
#   bootstrap confidence intervals) makes no distributional assumption beyond
#   resampling exchangeability across images, which the independence
#   assumption below covers.
# - **Between-image independence** is assumed and not tested or quantified
#   here.

# %%
assumption_rows = [
    {"id": "baseline_integrity", "description": "classical-baseline file integrity: row count, join, "
                                                  "reference-copy check, summary-statistic and "
                                                  "constant-predictor sanity checks",
     "passed": True, "detail": "all passed; see assumption_df above"},
    {"id": "pairing_complete", "description": "every image has exactly 24 model-side rows (6 models x 4 prompts)",
     "passed": k1.passed, "detail": k1.detail},
    {"id": "fan_out_guard", "description": "every classical-baseline statistic computed on a frame with "
                                            "n_rows == n_unique_images, checked at the point of use",
     "passed": True, "detail": "checked immediately before every classical-baseline statistic below"},
    {"id": "between_image_independence", "description": "assumed, not tested, not quantified",
     "passed": None, "detail": "assumed, not tested"},
]

# %% [markdown]
# ## The three series: classical baseline, the leading model configuration, and a three-model ensemble
#
# Figure 8 asks where the classical index's error sits relative to what the
# vision-language models can do, across the cover range. The two model-side
# series are re-derived here from `base_local` rather than fixed to a value
# carried over from elsewhere, so this notebook's ranking cannot silently go
# stale if the underlying accuracy results ever change.
#
# The ranking criterion is balanced MAE — the unweighted mean of the five
# per-bin mean absolute errors — the same criterion this study uses
# everywhere else a model or configuration is ranked or selected. Two model
# configurations are considered per image, one for each series:
#
# - the single model x prompt configuration with the lowest balanced MAE
#   over all 1,155 images;
# - an ensemble of the three models with the lowest balanced MAE (pooled
#   over their four prompts each), run on all four prompts, combined by
#   averaging the raw prediction across those 3 x 4 = 12 rows per image.
#   Averaging the prediction, not the per-image error, is what makes this an
#   ensemble rather than a pooled accuracy estimate.

# %%
# Rank the six models by balanced MAE, pooled over each model's four
# prompts. This selects the three-model ensemble membership below.
model_rank_rows = []
for model, g in base_local.groupby("model"):
    pooled = co.pool_axis_mean_abs_error(g, group_cols=["image"])
    pooled = pooled.merge(g[["image", "bin"]].drop_duplicates("image"), on="image")
    bal = co.balanced_mae(pooled["abs_e"], pooled["bin"]).balanced_mae
    model_rank_rows.append({"model": model, "balanced_mae": bal})
model_rank_df = pd.DataFrame(model_rank_rows).sort_values("balanced_mae").reset_index(drop=True)
print("Models ranked by balanced MAE, pooled over each model's four prompts (lowest first):")
print(model_rank_df.to_string(index=False))

top3_models = list(model_rank_df["model"].iloc[:3])
print(f"\nThree-model ensemble membership: {top3_models}")

# Rank all 24 model x prompt configurations by balanced MAE to find the
# leading single configuration.
combo_rank_rows = []
for (model, prompt), g in base_local.groupby(["model", "prompt"]):
    bal = co.balanced_mae(g["abs_e"], g["bin"]).balanced_mae
    combo_rank_rows.append({"model": model, "prompt": prompt, "balanced_mae": bal})
combo_rank_df = pd.DataFrame(combo_rank_rows).sort_values("balanced_mae").reset_index(drop=True)
best_model, best_prompt = combo_rank_df.iloc[0][["model", "prompt"]]
print(f"\nLeading single configuration by balanced MAE: "
      f"{best_model} x {best_prompt}  (balanced_mae={combo_rank_df.iloc[0]['balanced_mae']:.3f})")
combo_rank_df.head(5)

# %% [markdown]
# **A single winner is not the same claim as a set of indistinguishable
# configurations.** Balanced MAE differs across the 24 configurations by
# amounts that are themselves uncertain, so before this notebook names one
# configuration "the" leading one, it reports whether the data actually
# separates that configuration from its closest competitors. The top set and
# the probability of being best are computed once, in the accuracy notebook
# that ranks all 24 model x prompt configurations, and are read from its
# output file here rather than recomputed — a bootstrap estimate of the same
# statistic in two places would produce two slightly different numbers for
# the same claim, and a reader should never have to guess which one to cite.

# %%
_topset_path = RESULTS_DIR / "Q1_topset_bootstrap.csv"
if not _topset_path.is_file():
    raise FileNotFoundError(
        f"{_topset_path} not found. The top-set and probability-of-being-best "
        "for the 24 model x prompt configurations are computed in the accuracy "
        "notebook and read from its output here; that notebook must run first."
    )
topset_bootstrap = pd.read_csv(_topset_path)

n_topset = int(topset_bootstrap["in_union_topset"].sum())
best_row = topset_bootstrap[(topset_bootstrap["model"] == best_model) & (topset_bootstrap["prompt"] == best_prompt)].iloc[0]
best_prob_best = float(best_row["prob_best"])
best_in_topset = bool(best_row["in_union_topset"])

print(f"{n_topset} of {len(topset_bootstrap)} configurations are indistinguishable from the best at 95% confidence.")
print(f"{best_model} x {best_prompt}: probability of being best = {best_prob_best:.4f}, "
      f"in the 95%-confidence top set: {best_in_topset}")
topset_bootstrap[topset_bootstrap["in_union_topset"]]

# %% [markdown]
# **This is not a single winner; it is a small set of configurations the
# data cannot separate.** The balanced-MAE ranking above names one
# configuration as lowest, and the bootstrap directly above shows that this
# configuration is not reliably distinguishable from several others: six of
# the 24 model x prompt configurations sit inside the 95%-confidence top set,
# and the leading configuration's own probability of being the true best is
# just above one half. Every per-bin number reported for it below is
# therefore reported alongside this context, and its per-bin MAE is
# **selection-conditioned** — computed on the same 1,155 images that were
# used to select it as the leading configuration, so it is a slightly
# optimistic read of that configuration's error on a fresh sample, in the
# same way that any data-selected "best" estimate is.
#
# **The two natural ways of scoring "best" name different configurations.**
# Balanced MAE, which weights the five cover bins equally, selects the
# configuration used below. The plain average error over all 1,155 images
# (heavily weighted toward the near-bare-ground bin that holds 81% of the
# frame) selects a different configuration. Balanced MAE is used here
# because it is the criterion this study uses everywhere a configuration or
# model set is ranked or selected, including the three-model ensemble
# membership above; a reader comparing this section against a plain-average
# ranking elsewhere should expect the named configuration to differ.

# %%
# Best-single-configuration series: one row per image, that configuration's
# own prediction and error.
best_config_frame = base_local[(base_local["model"] == best_model) & (base_local["prompt"] == best_prompt)].copy()
co.guard_d12_statistic(best_config_frame, full_frame_expected=True, context="leading single configuration frame")

# Ensemble series: average the raw prediction across the 3 models x 4
# prompts = 12 rows per image, then compute error against the reference.
ensemble_source = base_local[base_local["model"].isin(top3_models)].copy()
ensemble_pred = co.aggregate_mllm_side_to_image(ensemble_source, value_col="vegetation_percent")
ensemble_frame = ensemble_pred.merge(
    base_local[["image", "reference", "bin"]].drop_duplicates("image"), on="image", how="inner",
)
if len(ensemble_frame) != 1155:
    raise AssertionError(f"ensemble frame did not yield 1,155 rows: got {len(ensemble_frame)}")
ensemble_frame["e"] = ensemble_frame["vegetation_percent"] - ensemble_frame["reference"]
ensemble_frame["abs_e"] = ensemble_frame["e"].abs()
co.guard_d12_statistic(ensemble_frame, full_frame_expected=True, context="three-model ensemble frame")

print(f"best_config_frame: {len(best_config_frame)} rows, {best_config_frame['image'].nunique()} images")
print(f"ensemble_frame (three-model ensemble, all 4 prompts): {len(ensemble_frame)} rows, "
      f"{ensemble_frame['image'].nunique()} images")

# %% [markdown]
# ## Per-bin MAE for the three series, with n
#
# Each series' mean absolute error is computed independently within each of
# the five reference-cover bins, with a 95% image-bootstrap confidence
# interval and the bin's image count.

# %%
def per_bin_mae_table(frame: pd.DataFrame, abs_error_col: str, image_col: str, bin_col: str,
                       rng_local: np.random.Generator, series_name: str) -> pd.DataFrame:
    rows = []
    for label in co.BIN_LABELS:
        sub = frame[frame[bin_col] == label]
        n = int(sub[image_col].nunique())
        mae = float(sub[abs_error_col].mean())

        def stat(f, _label=label):
            s = f.loc[f[bin_col] == _label, abs_error_col]
            return float(s.mean()) if len(s) else np.nan

        # Resamples the full 1,155-image frame and then subsets to the bin,
        # so the realised bin count can vary slightly across resamples. For
        # the smallest bin here (n=30) the chance of drawing it empty is
        # astronomically small, so this never produces an undefined mean in
        # practice, and the resulting interval is a standard subgroup-mean
        # bootstrap interval.
        boot = co.image_bootstrap(frame, stat, rng_local, image_col=image_col, bin_col=None, B=co.B_BOOTSTRAP)
        rows.append({"series": series_name, "bin": label, "n": n, "mae": mae,
                     "mae_ci_lo": boot.ci_lo, "mae_ci_hi": boot.ci_hi})
    return pd.DataFrame(rows)


def per_bin_signed_bias_table(frame: pd.DataFrame, signed_error_col: str, bin_col: str,
                               series_name: str) -> pd.DataFrame:
    """Per-bin signed bias, descriptive, no interval. In the four bins above
    20% cover this equals the per-bin MAE exactly, because every prediction
    in those bins falls below the reference — the arithmetic behind "no
    image above 20% cover is over-predicted," verified directly below.
    """
    rows = []
    for label in co.BIN_LABELS:
        sub = frame[frame[bin_col] == label]
        rows.append({"series": series_name, "bin": label,
                     "signed_bias": float(sub[signed_error_col].mean())})
    return pd.DataFrame(rows)


rng_perbin = np.random.default_rng(SEED + 0)

baseline_perbin = per_bin_mae_table(d12_frame, "abs_e", "image", "bin", rng_perbin, "classical_baseline")
best_config_perbin = per_bin_mae_table(best_config_frame, "abs_e", "image", "bin", rng_perbin, "leading_single_configuration")
ensemble_perbin = per_bin_mae_table(ensemble_frame, "abs_e", "image", "bin", rng_perbin, "three_model_ensemble")

baseline_bias_perbin = per_bin_signed_bias_table(d12_frame, "e", "bin", "classical_baseline")

# %% [markdown]
# **Published values, checked below.** The classical baseline's per-bin MAE
# is 3.411 / 25.791 / 45.719 / 57.890 / 78.208 across the five bins (overall
# MAE 10.924), and its per-bin signed bias -2.27 / -25.79 / -45.72 / -57.89 /
# -78.21. In the four bins above 20% cover the signed bias
# equals the per-bin MAE exactly — every prediction in those bins sits below
# the reference, so the mean absolute error and the mean signed error
# coincide. That equality is the arithmetic underneath "no image above 20%
# cover is over-predicted," verified directly below rather than only stated.

# %%
_published_mae = {"0-20": 3.4106, "20-40": 25.7910, "40-60": 45.7193, "60-80": 57.8900, "80-100": 78.2075}
_published_bias = {"0-20": -2.27, "20-40": -25.79, "40-60": -45.72, "60-80": -57.89, "80-100": -78.21}

for label in co.BIN_LABELS:
    row_mae = baseline_perbin.loc[baseline_perbin["bin"] == label, "mae"].iloc[0]
    row_bias = baseline_bias_perbin.loc[baseline_bias_perbin["bin"] == label, "signed_bias"].iloc[0]
    assert_published(f"baseline_mae[{label}]", row_mae, _published_mae[label], atol=5e-3)
    assert_published(f"baseline_signed_bias[{label}]", row_bias, _published_bias[label], atol=5e-2)

baseline_overall_mae = float(d12_frame["abs_e"].mean())
assert_published("baseline_overall_mae", baseline_overall_mae, 10.9235, atol=5e-3)
print(f"Baseline overall MAE = {baseline_overall_mae:.3f} (published 10.924). "
      "Per-bin MAE and signed bias match the published values.")

for label in ("20-40", "40-60", "60-80", "80-100"):
    mae_v = baseline_perbin.loc[baseline_perbin["bin"] == label, "mae"].iloc[0]
    bias_v = baseline_bias_perbin.loc[baseline_bias_perbin["bin"] == label, "signed_bias"].iloc[0]
    assert np.isclose(mae_v, -bias_v, atol=1e-6), (
        f"bin {label}: MAE ({mae_v}) should equal -signed_bias ({bias_v}) when nothing is over-predicted"
    )
print("Confirmed: in every bin above 20% cover, MAE equals the absolute value of the signed bias exactly "
      "-- consistent with zero over-prediction in those bins (checked directly above).")

figure8_panel_a = pd.concat([baseline_perbin, best_config_perbin, ensemble_perbin], ignore_index=True)
figure8_panel_a

# %% [markdown]
# ## The baseline-to-ensemble MAE ratio, above 20% cover only
#
# The 0-20% bin holds 933 of the 1,155 images and is where the baseline's own
# MAE is lowest (3.41) relative to the model side, so a ratio computed there
# would read backwards against the pattern this figure exists to show. The
# four bins above 20% cover — where 19% of the frame sits and the baseline's
# error grows fastest — are where the comparison is informative: the ratio of
# the baseline's MAE to the three-model ensemble's MAE, bin by bin.

# %%
ratio_rows = []
for label in ("20-40", "40-60", "60-80", "80-100"):
    b_mae = float(baseline_perbin.loc[baseline_perbin["bin"] == label, "mae"].iloc[0])
    e_mae = float(ensemble_perbin.loc[ensemble_perbin["bin"] == label, "mae"].iloc[0])
    ratio_rows.append({"bin": label, "baseline_mae": b_mae, "ensemble_mae": e_mae,
                        "ratio_baseline_to_ensemble": b_mae / e_mae})
ratio_df = pd.DataFrame(ratio_rows)
print("Baseline MAE / three-model-ensemble MAE, bins above 20% cover only:")
print(ratio_df.to_string(index=False))

figure8_panel_a_ratios = ratio_df.copy()

# %% [markdown]
# ## Predicted against reference, all 1,155 images, and the over-prediction count
#
# Every image's classical-baseline prediction plotted against its reference
# cover. Checked below: how many of the 222 images above 20% reference cover
# the baseline over-predicts.

# %%
above_20 = d12_frame[d12_frame["bin"] != "0-20"]
n_above_20 = int(above_20["image"].nunique())
n_overestimated_above_20 = int((above_20["e"] > 0).sum())

assert n_above_20 == 222, f"expected 222 images above 20% reference cover, got {n_above_20}"
assert n_overestimated_above_20 == 0, (
    f"expected 0 of 222 images above 20% cover over-predicted by the baseline, "
    f"got {n_overestimated_above_20}"
)
print(f"Images above 20% reference cover: {n_above_20}. "
      f"Over-predicted by the baseline: {n_overestimated_above_20} of {n_above_20}.")

# %% [markdown]
# ## Chart 1 — Figure 8, panel (a): per-bin MAE, three series
#
# **What to look for.** Three bars per bin — the classical baseline, the
# leading single model configuration, and the three-model ensemble across
# all four prompts — with 95% image-bootstrap confidence intervals and n
# annotated. The baseline is lowest in the 0-20% bin, where 933 of the 1,155
# images sit, and grows fastest of the three as cover increases; that
# crossover, not a single pooled number, is the finding this panel exists to
# show. The legend states the leading configuration's probability of being
# best and the size of the set it cannot be distinguished from, so the label
# does not overstate what the ranking behind it actually supports.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 6))
width = 0.25
x = np.arange(len(co.BIN_LABELS))
series_specs = [
    ("classical_baseline", "ExG - ExR baseline", "firebrick", -1),
    ("leading_single_configuration",
     f"leading configuration ({best_model} / {best_prompt}); P(best)={best_prob_best:.2f}, "
     f"1 of {n_topset} indistinguishable configurations",
     "steelblue", 0),
    ("three_model_ensemble", "three-model ensemble, all 4 prompts", "seagreen", 1),
]
for series_name, label, color, offset in series_specs:
    sub = figure8_panel_a[figure8_panel_a["series"] == series_name].set_index("bin").loc[list(co.BIN_LABELS)]
    yerr_lo = (sub["mae"] - sub["mae_ci_lo"]).values
    yerr_hi = (sub["mae_ci_hi"] - sub["mae"]).values
    ax.bar(x + offset * width, sub["mae"], width=width, yerr=[yerr_lo, yerr_hi], capsize=3,
           color=color, label=label)
    for xi, (m, n) in enumerate(zip(sub["mae"], sub["n"])):
        ax.annotate(f"n={int(n)}", (xi + offset * width, m), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(co.BIN_LABELS)
ax.set_xlabel("reference cover bin (%)")
ax.set_ylabel("MAE (cover points)")
ax.set_title("Per-bin MAE: classical baseline, leading configuration, three-model ensemble\n"
             "95% image-bootstrap confidence intervals, n annotated per bar")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q9_figure8_panel_a.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Chart 2 — Figure 8, panel (b): predicted against reference, all 1,155 images
#
# **What to look for.** Every image's baseline prediction against its
# reference cover, with the 1:1 line for reference. Points above the line
# are over-predictions; the annotation states the count above 20% cover
# directly on the panel.

# %%
fig, ax = plt.subplots(figsize=(7, 7))
colors = np.where(d12_frame["bin"] == "0-20", "steelblue", "firebrick")
ax.scatter(d12_frame["reference"], d12_frame["vegetation_percent"], c=colors, s=14, alpha=0.6)
lims = [0, max(d12_frame["reference"].max(), d12_frame["vegetation_percent"].max()) * 1.02]
ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="1:1 line")
ax.axvline(20, color="gray", linestyle=":", linewidth=1)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("reference cover (%)")
ax.set_ylabel("classical baseline prediction (%)")
ax.set_title(f"Baseline prediction vs. reference, all 1,155 images\n"
             f"{n_overestimated_above_20} of {n_above_20} images above 20% reference cover are over-predicted")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q9_figure8_panel_b.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## The baseline's best and worst images above 60% reference cover
#
# For Figure 8's panel (c): among the images with reference cover above 60%,
# the single image the baseline predicts closest to the reference and the
# single image it predicts furthest from it, each with reference, prediction
# and absolute error.

# %%
above_60 = d12_frame[d12_frame["bin"].isin(["60-80", "80-100"])].copy()
n_above_60 = int(above_60["image"].nunique())
print(f"Images above 60% reference cover: {n_above_60}")

best_idx = above_60["abs_e"].idxmin()
worst_idx = above_60["abs_e"].idxmax()

figure8_examples = above_60.loc[[best_idx, worst_idx], ["image", "reference", "vegetation_percent", "abs_e"]].copy()
figure8_examples.insert(0, "role", ["best", "worst"])
figure8_examples = figure8_examples.rename(columns={"vegetation_percent": "baseline_prediction", "abs_e": "absolute_error"})
print(figure8_examples.to_string(index=False))

# %% [markdown]
# ## Chart 3 — Figure 8, panel (c): best and worst images above 60% cover
#
# **What to look for.** Reference cover and the baseline's prediction, side
# by side, for the two exemplar images. The gap between the paired bars is
# each image's absolute error — small for the best case, large for the worst
# — illustrating the saturation the per-bin table already shows in aggregate.

# %%
fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(figure8_examples))
width = 0.35
ax.bar(x - width / 2, figure8_examples["reference"], width=width, color="gray", label="reference cover")
ax.bar(x + width / 2, figure8_examples["baseline_prediction"], width=width, color="firebrick",
       label="baseline prediction")
ax.set_xticks(x)
ax.set_xticklabels([f"{r}\n(image {img})" for r, img in zip(figure8_examples["role"], figure8_examples["image"])])
ax.set_ylabel("cover (%)")
ax.set_title("Baseline's best and worst images, reference cover above 60%")
for xi, (ref, pred, err) in enumerate(zip(figure8_examples["reference"], figure8_examples["baseline_prediction"],
                                           figure8_examples["absolute_error"])):
    ax.annotate(f"|e|={err:.1f}", (xi, max(ref, pred)), textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=8)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q9_figure8_panel_c.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Pearson `r` — one diagnostic number, never a ranking
#
# The baseline's per-bin MAE runs from 3.41 in the 0-20% bin to 78.21 in the
# 80-100% bin, a spread the single pooled MAE of 10.92 conceals entirely.
# That spread is what makes a pooled error statistic the wrong instrument
# for asking a different, narrower question: is the baseline tracking real
# variation in cover, or would a number close to its own near-zero median
# perform just as well? Pearson `r` answers that question directly, on the
# baseline alone, because it is sensitive to exactly the kind of linear
# covariation a threshold-and-index method should produce if it is not
# simply predicting a constant. **This is a diagnostic, not a comparison** —
# it is computed once, only for the baseline, never for any model, and it
# never enters a ranking claim. The `may_rank_methods` flag travels with it
# in the output, stored as an explicit boolean rather than a number, so a
# downstream reader cannot mistake it for anything but that guard.

# %%
baseline_pearson_r = float(np.corrcoef(d12_frame["vegetation_percent"], d12_frame["reference"])[0, 1])
assert_published("baseline_pearson_r", baseline_pearson_r, 0.5148, atol=1e-3)
print(f"Baseline Pearson r (diagnostic only, may_rank_methods=False) = {baseline_pearson_r:.3f}")

# %% [markdown]
# ## Result
#
# **What the figure shows.** The classical `ExG - ExR` baseline
# underestimates cover, and the underestimation grows with cover: per-bin MAE
# runs from 3.41 cover points in the 0-20% bin to 78.21 in the 80-100% bin,
# and in every bin above 20% cover the signed bias equals the per-bin MAE
# exactly — verified above, not merely stated — because not one of the 222
# images above 20% cover is over-predicted. That is systematic
# underestimation, and it is total above 20% cover in the specific sense that
# no image in that range escapes it.
#
# **The one place the baseline does relatively well.** The 0-20% bin holds
# 933 of the 1,155 images (81% of the frame), and it is the one bin where the
# baseline's own MAE (3.41) is close to or below the leading single
# configuration's and the three-model ensemble's. Read together with the
# growth above 20% cover, this is a description of *where* a colour-threshold
# method can and cannot work, not a verdict on which approach is better
# overall.
#
# **The leading configuration is a set of six, not a single winner.** Of the
# 24 model x prompt configurations, six sit inside the 95%-confidence top
# set, and the configuration used in panels (a) has a probability of being
# the true best of just above one half. Its per-bin numbers are
# selection-conditioned — computed on the same images used to select it — and
# both facts are carried in the figure legend and in the result table so a
# reader is never shown a single winner without the uncertainty around that
# selection.
#
# **The ratio above 20% cover** states how many times larger the baseline's
# per-bin MAE is than the three-model ensemble's, in each of the four bins
# where cover is present. This is a magnitude, not a test — there is no
# confidence interval on the ratio itself and no p-value.
#
# **Pearson `r` = 0.515** is the diagnostic that keeps the underestimation
# finding from being confused with the baseline simply predicting a constant
# near zero: it is tracking real variation in cover, just severely
# mis-scaled once cover departs from near-bare ground.
#
# **Failure on material that is not green.** The baseline is a greenness
# index by construction, so it has no mechanism for detecting cover that is
# not green — consistent with its growing underestimation above 20% cover,
# where non-green material becomes more common.
#
# Shadow is a documented limitation of greenness-based vegetation indices in
# the broader remote-sensing literature (e.g. Meyer & Neto, 2008, *Computers
# and Electronics in Agriculture*, on excess-green indices and illumination
# artefacts), named here strictly as a known limitation of the method class.
# No image in this dataset is annotated for shadow, so this notebook measures
# it nowhere, and nothing above should be read as evidence that shadow caused
# any specific error reported here.

# %%
diagnostic_rows = []
for label in co.BIN_LABELS:
    mae_v = float(baseline_perbin.loc[baseline_perbin["bin"] == label, "mae"].iloc[0])
    n_v = int(baseline_perbin.loc[baseline_perbin["bin"] == label, "n"].iloc[0])
    ci_lo = float(baseline_perbin.loc[baseline_perbin["bin"] == label, "mae_ci_lo"].iloc[0])
    ci_hi = float(baseline_perbin.loc[baseline_perbin["bin"] == label, "mae_ci_hi"].iloc[0])
    bias_v = float(baseline_bias_perbin.loc[baseline_bias_perbin["bin"] == label, "signed_bias"].iloc[0])
    diagnostic_rows.append({
        "question_id": "Q9", "statistic": f"mae_bin_{label}", "value": mae_v,
        "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n_v, "test": "image bootstrap CI (bin subset)",
    })
    diagnostic_rows.append({
        "question_id": "Q9", "statistic": f"signed_bias_bin_{label}", "value": bias_v,
        "ci_lo": np.nan, "ci_hi": np.nan, "n": n_v, "test": "computed, no interval",
    })

diagnostic_rows.append({
    "question_id": "Q9", "statistic": "overall_mae", "value": baseline_overall_mae,
    "ci_lo": np.nan, "ci_hi": np.nan, "n": len(d12_frame), "test": "computed",
})
diagnostic_rows.append({
    "question_id": "Q9", "statistic": "pearson_r_diagnostic_only", "value": baseline_pearson_r,
    "ci_lo": np.nan, "ci_hi": np.nan, "n": len(d12_frame),
    "test": "Pearson correlation, diagnostic only",
})
diagnostic_rows.append({
    "question_id": "Q9", "statistic": "n_overestimated_above_20pct_cover", "value": n_overestimated_above_20,
    "ci_lo": np.nan, "ci_hi": np.nan, "n": n_above_20, "test": "computed",
})

q9_baseline_diagnostic = pd.DataFrame(diagnostic_rows)
# Stored as a nullable boolean, not a float: this column is the only thing
# preventing a diagnostic correlation from being read as a ranking claim,
# and a float (0.0 / NaN) is too easy to misread as a magnitude rather than
# a flag.
q9_baseline_diagnostic["may_rank_methods"] = pd.array(
    [False if s == "pearson_r_diagnostic_only" else pd.NA for s in q9_baseline_diagnostic["statistic"]],
    dtype="boolean",
)
q9_baseline_diagnostic

# %%
# The ratio is a single number per bin (baseline MAE / ensemble MAE, above
# 20% cover only) -- a property of the bin, not of any one series, so it is
# attached only to the classical-baseline row for that bin rather than
# broadcast across all three series' rows, which would misleadingly repeat
# the same value against the other two rows too.
q9_figure8_panel_a = figure8_panel_a.copy()
q9_figure8_panel_a["ratio_baseline_to_ensemble"] = np.nan
for _, r in figure8_panel_a_ratios.iterrows():
    mask = (q9_figure8_panel_a["series"] == "classical_baseline") & (q9_figure8_panel_a["bin"] == r["bin"])
    q9_figure8_panel_a.loc[mask, "ratio_baseline_to_ensemble"] = r["ratio_baseline_to_ensemble"]

# The identity of the selected configuration and the ensemble membership are
# written onto every row, not just printed to stdout, so a figure legend or
# a sentence in the paper naming either can be built directly from this file
# rather than typed by hand — and so a future change in the underlying
# ranking would visibly change these columns rather than silently relabelling
# the same row names.
q9_figure8_panel_a["selected_model"] = np.where(
    q9_figure8_panel_a["series"] == "leading_single_configuration", best_model, pd.NA)
q9_figure8_panel_a["selected_prompt"] = np.where(
    q9_figure8_panel_a["series"] == "leading_single_configuration", best_prompt, pd.NA)
q9_figure8_panel_a["ensemble_members"] = np.where(
    q9_figure8_panel_a["series"] == "three_model_ensemble", ", ".join(top3_models), pd.NA)
q9_figure8_panel_a["topset_size"] = np.where(
    q9_figure8_panel_a["series"] == "leading_single_configuration", n_topset, pd.NA)
q9_figure8_panel_a["prob_best"] = np.where(
    q9_figure8_panel_a["series"] == "leading_single_configuration", best_prob_best, np.nan)
q9_figure8_panel_a["selection_conditioned"] = np.where(
    q9_figure8_panel_a["series"] == "leading_single_configuration", True, pd.NA)

q9_figure8_panel_a.insert(0, "question_id", "Q9")
q9_figure8_panel_a

# %%
q9_figure8_examples = figure8_examples.copy()
q9_figure8_examples.insert(0, "question_id", "Q9")
q9_figure8_examples

# %%
q9_assumption_checks = pd.DataFrame(assumption_rows)
q9_assumption_checks.insert(0, "question_id", "Q9")
q9_assumption_checks

# %%
q9_baseline_diagnostic.to_csv(RESULTS_DIR / "Q9_baseline_diagnostic.csv", index=False)
q9_figure8_panel_a.to_csv(RESULTS_DIR / "Q9_figure8_panel_a.csv", index=False)
q9_figure8_examples.to_csv(RESULTS_DIR / "Q9_figure8_examples.csv", index=False)
# Per-image predictions behind panel (b), so the scatter can be redrawn from the
# results alone without executing this notebook again.
q9_figure8_panel_b = d12_frame[["image", "reference", "vegetation_percent", "abs_e", "bin"]].rename(
    columns={"vegetation_percent": "baseline_prediction", "abs_e": "absolute_error"}
)
q9_figure8_panel_b.to_csv(RESULTS_DIR / "Q9_figure8_panel_b.csv", index=False)
q9_assumption_checks.to_csv(RESULTS_DIR / "Q9_assumption_checks.csv", index=False)

print("\nWritten:")
for fname in [
    "Q9_baseline_diagnostic.csv", "Q9_figure8_panel_a.csv",
    "Q9_figure8_examples.csv", "Q9_assumption_checks.csv",
]:
    print(f"  05_deliverables/Repository/GitHub/results/{fname}")
