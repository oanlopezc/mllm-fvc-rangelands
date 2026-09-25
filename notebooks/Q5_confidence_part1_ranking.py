# %% [markdown]
# # Q5 — Confidence as a ranking: does it track error, and does it track scene difficulty?
#
# Every model in this study reports a self-assessed confidence alongside its
# vegetation-cover estimate. Two different questions can be asked of that
# number, and they are not the same question. Confidence can track a
# configuration's **error** — which is what would make it useful for triage,
# flagging the estimates worth a second look — and, independently, it can
# track **scene difficulty**, whether or not the estimate happened to come out
# right. The second question is invisible if confidence is judged only by
# whether it flags a large error: a low-confidence *correct* estimate counts
# against a model under that framing, when it may simply be the model
# responding honestly to material this project's Q3 already shows is
# genuinely hard to score. This notebook keeps the two questions separate
# throughout.
#
# **The primary unit is the model x prompt cell — 24 of them.** Averaging a
# model's four cell-level rho values *on paper* would describe nothing real:
# Gemma-3-27B's own confidence-error rho runs from +0.128 under *Point-Hint*
# to -0.382 under *Detailed*, two opposite verdicts a hand-computed average
# would collapse into one middling number. Every cell-level estimate below is
# computed at the cell and carries its own `n` — usable confidence per cell
# ranges from 1,109 to 1,155 of 1,155 images, and seventeen cells lose
# nothing at all. Separately, and later in this notebook, the model and the prompt are
# also each treated as their own unit of analysis: a model row pools all
# 1,155 images across that model's four prompts and computes a fresh
# image-level rank correlation directly on the pooled rows, rather than
# averaging the four cell-level rho values. The pooled rho is not the same
# quantity as an average over that model's four cells: pooling four responses per image
# denoises both series first, so the pooled model-level rho can be, and
# sometimes is, more extreme than any of that model's own four cells — and it
# is reported and read as a verdict in its own right, not as a summary of the
# cell-level table.
#
# **No test anywhere in this notebook.** The associations below run as
# strong as `rho = -0.77`; a p-value adds nothing to that, and the cells
# where the answer is genuinely uncertain are the ones near zero, where an
# interval already says everything a p-value would. Every quantity below is
# an estimate with a bootstrap interval, never a corrected test. Because
# nothing here is a pre-declared test, the 24 per-cell verdicts are not
# multiplicity-corrected — that is stated once here, and again at each table,
# rather than left to be discovered.
#
# **What this notebook does not compute.** Self-reported confidence
# (`D5.v5`/`D6.v4`) is an ordinal scale: nothing establishes that these
# numbers are probabilities, or that they mean the same thing from one model
# to the next. ECE, Brier score, log-loss and any calibration metric that
# assumes a probability scale are inadmissible here and appear nowhere in this
# notebook, and neither does a mean or a Pearson correlation of confidence.
# `rooted_dead_alike_plants` (`D2.v3`) is nominal, its three levels carrying
# no order, so no rank correlation, no trend test and no 0/1/2 score is
# computed on it anywhere below.
#
# **Four layers, in the order a reader should take them.** Reported first,
# ahead of every association: confidence's own **granularity** — how many
# distinct values a cell actually emits, because a coarse confidence signal
# changes how every later result should be read. **Layer 1** asks whether
# confidence tracks error (tie-corrected Spearman rho, pooled and inside the
# 0-20% cover bin, to check the pooled figure is not just borrowed from
# cover). **Layer 2** asks whether that association is actionable for
# triage — a per-cell ROC against the same error threshold used throughout
# this analysis. **Layer 3** asks what confidence responds to regardless of
# whether the estimate was right — five scene features, compared the only
# way that is legal on a nominal variable. **Layer 4** is the operational
# sweep: what filtering on confidence actually buys, and what it costs, at
# every cutoff the data can realise.

# %% [markdown]
# ## Setup
#
# The seed is fixed so every resampling step below reproduces identically on
# every run.

# %%
import os
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


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

SEED = 20260907 + 5  # this question's own fixed seed, offset from the study's base seed

# Each bootstrap quantity below draws from its own generator, offset from
# SEED by a fixed integer declared once here — never from a single shared
# stream — so that adding, removing or reordering one block can never shift
# the draws consumed by another. The offsets are recorded here and nowhere
# else:
#   +0  Layer 1 -- per-cell Spearman rho (pooled and within the 0-20% bin)
#   +3  Layer 3 -- five-feature pairwise percentages, per cell
#   +4  Layer 4 -- confidence-filtering sweep, MAE_o and MAE_b on retained set
rng_rank = np.random.default_rng(SEED + 0)
rng_feature = np.random.default_rng(SEED + 3)
rng_sweep = np.random.default_rng(SEED + 4)

ROOT = co.project_root()
RESULTS_DIR = ROOT / "05_deliverables/Repository/GitHub/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RENDERED_DIR = ROOT / "05_deliverables/Repository/GitHub/notebooks" / "rendered"
RENDERED_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

MODEL_COLORS = {
    "Qwen-2.5": "#4477AA",
    "Mistral-Small-3.2": "#228833",
    "Llama-4-Maverick": "#EE6677",
    "Llama-4-Scout": "#66CCEE",
    "Gemma-3-12B": "#CCBB44",
    "Gemma-3-27B": "#AA3377",
}

print(f"seed={SEED}")

# %% [markdown]
# ## Data
#
# The main-path analysis frame is `variant == 'base'`, local stack: 1,155
# images x 6 models x 4 prompts = 27,720 prediction rows, each scored against
# `D1.v7` (the mean-of-two-observers reference). Layer 3 additionally needs
# the two material annotations (`D2.v3`, `D2.v4`) and obliquity (`D4.v2`),
# joined on `image`.

# %%
assumption_df, frames = co.run_all_assertions(include_d12=False)
base_local = frames["base_local"].copy()
d2 = frames["d2"]
d4 = frames["d4"]

k1 = co.assert_k1_pairing_complete(frames["d5"], base_local)
print(f"K1 (pairing complete): passed={k1.passed} — {k1.detail}")

base_local = base_local.merge(
    d2[["image", "rooted_dead_alike_plants", "non_rooted_plant_material"]],
    on="image", how="left", validate="many_to_one",
)
base_local = base_local.merge(d4[["image", "obliquity"]], on="image", how="left", validate="many_to_one")

# Confidence is missing not at random by construction: it is missing exactly
# when the parse method is `ambiguous_unresolved`. It is never listwise-deleted from any
# vegetation_percent analysis — only from the confidence-specific quantities
# below, and every one of those states its own usable n.
base_local["confidence_unusable"] = (
    base_local["confidence_parse_method"].eq("ambiguous_unresolved") | base_local["confidence"].isna()
)

a13 = co.assert_a13(base_local, frames["d5"])
print(f"A13 (per-cell unusable confidence, base/local frame only): {a13.detail}")

print(f"base_local: {len(base_local)} rows, {base_local['image'].nunique()} images, "
      f"{base_local['model'].nunique()} models, {base_local['prompt'].nunique()} prompts")


def complete_case(frame: pd.DataFrame) -> pd.DataFrame:
    """Usable-confidence rows only, for a confidence-specific quantity."""
    return frame.loc[~frame["confidence_unusable"]].copy()


CELLS = [(m, p) for m in co.MODELS for p in sorted(base_local["prompt"].unique())]


def cell_bootstrap(
    frame_cell: pd.DataFrame,
    row_statistic: "Callable[[np.ndarray], float]",
    rng: np.random.Generator,
    B: int = co.B_BOOTSTRAP,
    alpha: float = 0.05,
) -> co.BootstrapResult:
    """Image bootstrap specialised to a single model x prompt cell.

    `_common.image_bootstrap` resamples named image IDs and rebuilds the
    frame via a groupby on every one of its B draws, because in general one
    image can carry several rows in the frame it is handed (e.g. every
    model, every prompt at once). A single cell of this notebook's frame is
    guaranteed by K1 to hold **exactly one row per image**, so resampling row
    positions with `rng.integers` is exactly equivalent to resampling image
    IDs with replacement here, and is roughly two orders of magnitude
    faster — material at the number of per-cell bootstraps this notebook
    runs. `row_statistic` receives a 1-D numpy array of whatever
    per-row quantity the caller has already extracted (never a DataFrame),
    and the same BCa construction `_common._bca_interval` implements is
    reused unchanged, so every interval below is on identical footing with
    every other bootstrap interval in this project.
    """
    n = len(frame_cell)
    theta_hat = row_statistic(np.arange(n))
    boot_vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot_vals[b] = row_statistic(idx)

    jack_vals = np.empty(n)
    all_idx = np.arange(n)
    for i in range(n):
        jack_vals[i] = row_statistic(np.delete(all_idx, i))

    lo, hi, used_bca = co._bca_interval(theta_hat, boot_vals, jack_vals, alpha=alpha)
    ci_method = "BCa"
    if not used_bca:
        lo = float(np.percentile(boot_vals, 100 * alpha / 2))
        hi = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
        ci_method = "percentile (BCa fallback)"

    return co.BootstrapResult(
        estimate=theta_hat, ci_lo=lo, ci_hi=hi, ci_method=ci_method,
        boot_values=boot_vals, n_empty_bin_violations=0, p_two_sided=None,
    )


# %% [markdown]
# ## The outlier definition: an upper Tukey fence on absolute error, 20.94
#
# A prediction is flagged as an outlier by a threshold computed once,
# directly from this study's own error distribution: the upper Tukey fence,
# `Q3 + 1.5 x IQR`, applied to the **mean absolute error of each of the 1,155
# images, averaged across the 24 model x prompt cells**. This is a
# data-driven choice, not a non-arbitrary one — the cut comes from this
# study's own distribution, but the 1.5 multiplier remains a convention
# calibrated for roughly symmetric data, while absolute error here is
# right-skewed with a floor at zero, so "data-driven" is what is claimed and
# no more.
#
# The fence is computed on **absolute** error, upper side only. A signed,
# two-fence rule is not used: the cover scale is bounded,
# so a lower fence on signed error mostly detects that a quadrat at 5% cover
# cannot be underestimated by more than 5 points, and the fences would absorb
# the models' mean overestimation bias rather than measuring anything about
# confidence. Direction is not lost by this choice — it is already reported
# per bin in Q1's mean-bias table.
#
# The same constant is used at both levels this notebook needs: it defines
# the **hard-image set at image level** (used nowhere else in this notebook,
# but computed here so its value is verifiable) and the **positive class for
# the per-cell ROC at response level**. Methods states plainly that the
# threshold is derived from the image-level distribution and then applied to
# individual responses — never pretended to be a response-level fence in its
# own right. The most principled outlier definition would compare a model's
# miss against the disagreement between the two trained observers on the same
# quadrat, but the input data carries only the mean of the two observers
# (`D1.v7`), not the two readings separately, so that comparison cannot be
# built from this data.

# %%
image_mean_abs_e = base_local.groupby("image")["abs_e"].mean()
q1_iqr, q3_iqr = image_mean_abs_e.quantile([0.25, 0.75])
iqr = q3_iqr - q1_iqr
OUTLIER_FENCE = float(q3_iqr + 1.5 * iqr)
n_hard_images = int((image_mean_abs_e > OUTLIER_FENCE).sum())
skew_image_mae = float(stats.skew(image_mean_abs_e))

print(f"Q1={q1_iqr:.4f}, Q3={q3_iqr:.4f}, IQR={iqr:.4f}")
print(f"Upper Tukey fence on image-level mean |e| = {OUTLIER_FENCE:.4f} cover points")
print(f"Hard images at image level: {n_hard_images} ({n_hard_images/1155*100:.1f}%)")
print(f"Skewness of the image-level mean |e| distribution: {skew_image_mae:.2f}")

base_local["positive"] = base_local["abs_e"] > OUTLIER_FENCE
response_level_flag_rate = float(base_local["positive"].mean())
positives_per_cell = base_local.groupby(["model", "prompt"])["positive"].sum()
print(f"Response-level flag rate at this fence: {response_level_flag_rate*100:.1f}%")
print(f"Positives per cell: {int(positives_per_cell.min())} to {int(positives_per_cell.max())}")

# %% [markdown]
# ## Confidence granularity, reported first
#
# Confidence granularity is not a caveat to mention after the fact — it is
# the constraint that shapes every result below, so it is reported before any
# association. For each of the 24 cells: how many distinct confidence values
# it emits, and what share of its usable responses sit on the single most
# common value.

# %%
granularity_rows = []
for model, prompt in CELLS:
    g = base_local[(base_local["model"] == model) & (base_local["prompt"] == prompt)]
    usable = g.loc[~g["confidence_unusable"], "confidence"]
    k8 = co.check_k8_confidence_degeneracy(usable)
    granularity_rows.append({
        "question_id": "Q5",
        "model": model,
        "prompt": prompt,
        "n_usable": len(usable),
        "n_excluded_unusable_confidence": int(g["confidence_unusable"].sum()),
        "n_total": len(g),
        "n_distinct_values": k8["n_levels"],
        "max_single_level_mass": k8["max_single_level_mass"],
        "degenerate_k8": k8["degenerate"],
    })

granularity_df = pd.DataFrame(granularity_rows)
print(f"Distinct values per cell: {granularity_df['n_distinct_values'].min()} to "
      f"{granularity_df['n_distinct_values'].max()}")
print(f"Max single-level mass across cells: {granularity_df['max_single_level_mass'].max()*100:.1f}%")
print(f"Usable n per cell: {granularity_df['n_usable'].min()} to {granularity_df['n_usable'].max()}")

worked_case = granularity_df[(granularity_df["model"] == "Qwen-2.5") & (granularity_df["prompt"] == "Grid-Overlay")].iloc[0]
print(f"\nWorked case — Qwen-2.5 x Grid-Overlay: {worked_case['n_distinct_values']} distinct values, "
      f"{worked_case['max_single_level_mass']*100:.1f}% of responses on the single most common value.")

granularity_df.sort_values("max_single_level_mass", ascending=False).head(8)

# %% [markdown]
# Ten cells lose no rows to unusable confidence at all; the worst are
# Gemma-3-12B under *Point-Hint* and Llama-4-Scout under *Point-Hint* and
# *Detailed*. Coarse confidence recurs throughout this notebook: it pulls
# Layer 3's pairwise percentages toward 50% (K8), and it is why Layer 2's ROC
# for some cells is a step curve with few distinct points rather than a
# smooth one. Every later table is read against this one, not the reverse.

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5.5))
for model in co.MODELS:
    sub = granularity_df[granularity_df["model"] == model].sort_values("prompt")
    ax.scatter(sub["n_distinct_values"], sub["max_single_level_mass"] * 100,
               color=MODEL_COLORS[model], label=model, s=70, edgecolor="white", linewidth=0.6)
ax.set_xlabel("distinct confidence values emitted (usable rows only)")
ax.set_ylabel("share of responses on the single most common value (%)")
ax.set_title("Confidence granularity, 24 model x prompt cells")
ax.legend(fontsize=8, loc="upper right", ncol=2)
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q5_granularity.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# **What to look for.** Each point is one cell. Points toward the top-left
# (few distinct values, most mass on one of them) are the cells whose Layer 3
# percentages are most compressed toward 50% by ties — read those against
# this chart, not as an association that failed.

# %% [markdown]
# ## Layer 1 — does confidence track error
#
# Tie-corrected Spearman `rho` between confidence and absolute error, per
# cell, with a BCa interval from the paired image bootstrap. A direction is
# claimed only where the interval excludes zero. The same statistic is then
# recomputed inside the 0-20% cover bin alone (933 images) — confidence
# correlates with reference cover itself at -0.47 to -0.66, and error rises
# with cover, so in principle the pooled association could be almost entirely
# a cover effect rather than a genuine error signal. It is not: the
# within-bin association below is at least as strong as the pooled one in
# every cell that shows an association at all, so the pooled headline
# understates rather than overstates what confidence is doing.
#
# Unusable confidence is bounded, not ignored: `rho` is recomputed twice more
# under the two extreme assignments a genuinely unknown value could take —
# every unusable row given first the lowest possible confidence rank, then
# the highest — and the two results bound what the missing rows could do to
# the answer.

# %%
def spearman_rho_stat(frame_cc: pd.DataFrame) -> float:
    rho, _ = co.spearman_tie_corrected(frame_cc["confidence"].values, frame_cc["abs_e"].values)
    return rho


def mnar_extreme_rho(frame_all: pd.DataFrame, *, unusable_get_lowest: bool) -> float:
    """Recompute rho with every unusable row assigned an extreme confidence
    rank: `unusable_get_lowest=True` sends them below every observed value
    (the assignment most favourable to a negative rho), `False` sends them
    above every observed value (least favourable). The two together bound
    what the unknown values could do to the pooled estimate.
    """
    filled = frame_all.copy()
    mask = filled["confidence_unusable"]
    if not mask.any():
        return spearman_rho_stat(filled)
    obs = filled.loc[~mask, "confidence"]
    fill_value = float(obs.min()) - 1.0 if unusable_get_lowest else float(obs.max()) + 1.0
    filled.loc[mask, "confidence"] = fill_value
    return spearman_rho_stat(filled)


rank_rows = []
for model, prompt in CELLS:
    g = base_local[(base_local["model"] == model) & (base_local["prompt"] == prompt)].reset_index(drop=True)
    g_cc = complete_case(g)

    rho_pooled = spearman_rho_stat(g_cc)

    unusable_arr = g["confidence_unusable"].to_numpy()
    conf_arr = g["confidence"].to_numpy(dtype=float)
    abs_e_arr = g["abs_e"].to_numpy(dtype=float)

    def rho_pooled_row_stat(idx, _unusable=unusable_arr, _conf=conf_arr, _abs_e=abs_e_arr):
        u, c, a = _unusable[idx], _conf[idx], _abs_e[idx]
        usable_mask = ~u
        if usable_mask.sum() < 2:
            return np.nan
        rho, _ = co.spearman_tie_corrected(c[usable_mask], a[usable_mask])
        return rho

    boot_pooled = cell_bootstrap(g, rho_pooled_row_stat, rng_rank, B=co.B_BOOTSTRAP)

    g020 = g[g["bin"] == "0-20"].reset_index(drop=True)
    g020_cc = complete_case(g020)
    rho_020 = spearman_rho_stat(g020_cc) if len(g020_cc) >= 2 else np.nan

    if len(g020_cc) >= 2:
        unusable_020 = g020["confidence_unusable"].to_numpy()
        conf_020 = g020["confidence"].to_numpy(dtype=float)
        abs_e_020 = g020["abs_e"].to_numpy(dtype=float)

        def rho_020_row_stat(idx, _unusable=unusable_020, _conf=conf_020, _abs_e=abs_e_020):
            u, c, a = _unusable[idx], _conf[idx], _abs_e[idx]
            usable_mask = ~u
            if usable_mask.sum() < 2:
                return np.nan
            rho, _ = co.spearman_tie_corrected(c[usable_mask], a[usable_mask])
            return rho

        boot_020 = cell_bootstrap(g020, rho_020_row_stat, rng_rank, B=co.B_BOOTSTRAP)
    else:
        boot_020 = None

    rho_mnar_lo = mnar_extreme_rho(g, unusable_get_lowest=True)
    rho_mnar_hi = mnar_extreme_rho(g, unusable_get_lowest=False)
    mnar_bound_lo = min(rho_mnar_lo, rho_mnar_hi)
    mnar_bound_hi = max(rho_mnar_lo, rho_mnar_hi)

    rank_rows.append({
        "question_id": "Q5",
        "model": model,
        "prompt": prompt,
        "n_pooled": len(g_cc),
        "n_excluded_unusable_confidence_pooled": int(g["confidence_unusable"].sum()),
        "rho_pooled": rho_pooled,
        "rho_pooled_ci_lo": boot_pooled.ci_lo,
        "rho_pooled_ci_hi": boot_pooled.ci_hi,
        "rho_pooled_ci_method": boot_pooled.ci_method,
        "direction_claimed_pooled": bool(boot_pooled.ci_lo > 0 or boot_pooled.ci_hi < 0),
        "n_0_20_bin": len(g020_cc),
        "rho_0_20_bin": rho_020,
        "rho_0_20_bin_ci_lo": boot_020.ci_lo if boot_020 is not None else np.nan,
        "rho_0_20_bin_ci_hi": boot_020.ci_hi if boot_020 is not None else np.nan,
        "rho_0_20_bin_ci_method": boot_020.ci_method if boot_020 is not None else "not computed (n<2)",
        "direction_claimed_0_20_bin": bool(
            boot_020 is not None and (boot_020.ci_lo > 0 or boot_020.ci_hi < 0)
        ),
        "rho_mnar_bound_lo": mnar_bound_lo,
        "rho_mnar_bound_hi": mnar_bound_hi,
        "mnar_bound_width": mnar_bound_hi - mnar_bound_lo,
        "k8_degenerate": bool(
            granularity_df[(granularity_df["model"] == model) & (granularity_df["prompt"] == prompt)]
            ["degenerate_k8"].iloc[0]
        ),
        "multiplicity_note": "24-cell interval verdicts, not corrected for multiplicity",
    })

rank_association_df = pd.DataFrame(rank_rows)
print(f"Pooled rho range: {rank_association_df['rho_pooled'].min():.3f} to "
      f"{rank_association_df['rho_pooled'].max():.3f}")
print(f"Within-0-20-bin rho range: {rank_association_df['rho_0_20_bin'].min():.3f} to "
      f"{rank_association_df['rho_0_20_bin'].max():.3f}")

# Two cells where the within-bin figure reads very differently from the
# pooled one.
for m, p in [("Mistral-Small-3.2", "Detailed"), ("Gemma-3-12B", "Short")]:
    row = rank_association_df[(rank_association_df["model"] == m) & (rank_association_df["prompt"] == p)].iloc[0]
    print(f"{m} x {p}: pooled rho={row['rho_pooled']:.3f}, within-0-20 rho={row['rho_0_20_bin']:.3f}")

rank_association_df.sort_values("rho_pooled").head(6)

# %% [markdown]
# Mistral-Small-3.2 under *Detailed* looks incoherent pooled (near zero) but
# negative inside the 0-20% bin, and Gemma-3-12B under *Short* moves from a
# small pooled association to a clearer negative one within the bin. Both are
# cells where restricting to the sparsest bin — which removes most of the
# image-to-image cover range, though far from all of it — strengthens rather
# than weakens the association, so the pooled figure alone understates what
# confidence is doing in these two cells. This is not the same claim as
# confidence being independent of cover within the bin: confidence still
# tracks reference cover inside the 0-20% bin in most cells (Layer 3's
# obliquity check below runs into the same residual cover dependence), so the
# within-bin figures show the pooled association is not merely an artefact of
# pooling across cover, without showing that cover has been held fixed.

# %%
fig, ax = plt.subplots(figsize=(9, 6))
for model in co.MODELS:
    sub = rank_association_df[rank_association_df["model"] == model].sort_values("prompt")
    ax.scatter(sub["rho_pooled"], sub["rho_0_20_bin"], color=MODEL_COLORS[model],
               label=model, s=70, edgecolor="white", linewidth=0.6)
lims = [-0.85, 0.65]
ax.plot(lims, lims, color="grey", linestyle=":", linewidth=1, label="pooled = within-bin")
ax.axhline(0, color="black", linewidth=0.6)
ax.axvline(0, color="black", linewidth=0.6)
ax.set_xlabel("Spearman rho(confidence, |e|), pooled over all 1,155 images")
ax.set_ylabel("Spearman rho(confidence, |e|), within the 0-20% cover bin (n=933)")
ax.set_title("Layer 1 — the within-bin check moves several cells further negative")
ax.legend(fontsize=7, loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q5_layer1_rank_association.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# **What to look for.** A point below the diagonal has a more negative
# within-bin association than pooled — the pooled figure understated the
# effect. Points in the lower-left quadrant are the strongest, most
# unambiguous confidence-tracks-error cells; points near the origin are the
# ones a coarse-confidence cell (see the granularity chart) pulls toward
# zero regardless of any real association.

# %% [markdown]
# ## Layer 2 — can confidence be used to triage bad estimates
#
# A rank correlation alone hides the false-positive cost of a triage rule —
# a configuration can post a strong `rho` while a usable rule still requires
# flagging almost every image. Qwen-2.5 under *Grid-Overlay* is exactly this
# case: `rho = -0.611`, yet reaching a true-positive rate of 1.0 means
# flagging the great majority of its images, because its worst and its
# ordinary images share the same lowest emitted confidence level. A ROC
# makes that cost visible directly: at every confidence level a cell
# actually emits, the true-positive rate and false-positive rate against the
# same 20.94-cover-point fence defined above, applied at response level. It
# carries no test, and it is read as a curve with operating points, not as a
# bare summary statistic.

# %%
def roc_points(g: pd.DataFrame) -> pd.DataFrame:
    """Realisable-cutoff TPR/FPR points: predict positive for confidence at
    or below a given usable level, unusable-confidence rows never predicted
    positive (they cannot be triaged at all, so the rule that never fires on
    them is the honest one).
    """
    usable = g.loc[~g["confidence_unusable"]].copy()
    n_pos = int(g["positive"].sum())
    n_neg = int((~g["positive"]).sum())
    rows = []
    for lev in sorted(usable["confidence"].unique()):
        below_or_eq = (~g["confidence_unusable"]) & (g["confidence"] <= lev)
        tp = int((g["positive"] & below_or_eq).sum())
        fp = int((~g["positive"] & below_or_eq).sum())
        rows.append({
            "confidence_level": lev,
            "tpr": tp / n_pos if n_pos else np.nan,
            "fpr": fp / n_neg if n_neg else np.nan,
            "n_at_level": int((usable["confidence"] == lev).sum()),
        })
    return pd.DataFrame(rows)


roc_rows = []
for model, prompt in CELLS:
    g = base_local[(base_local["model"] == model) & (base_local["prompt"] == prompt)]
    pts = roc_points(g)
    n_pos = int(g["positive"].sum())
    n_neg = int((~g["positive"]).sum())
    for _, r in pts.iterrows():
        roc_rows.append({
            "question_id": "Q5", "model": model, "prompt": prompt,
            "confidence_level": r["confidence_level"], "tpr": r["tpr"], "fpr": r["fpr"],
            "n_at_level": int(r["n_at_level"]), "n_positives": n_pos, "n_negatives": n_neg,
        })

roc_df = pd.DataFrame(roc_rows)
print(f"{len(roc_df)} operating points across 24 cells "
      f"({roc_df.groupby(['model','prompt']).size().mean():.1f} points/cell on average)")

qwen_grid = roc_df[(roc_df["model"] == "Qwen-2.5") & (roc_df["prompt"] == "Grid-Overlay")]
tpr1_row = qwen_grid[np.isclose(qwen_grid["tpr"], 1.0)].iloc[0]
print(f"Qwen-2.5 x Grid-Overlay: reaching TPR=1.0 requires flagging "
      f"{tpr1_row['fpr']*qwen_grid['n_negatives'].iloc[0] + qwen_grid['n_positives'].iloc[0]:.0f} of "
      f"{qwen_grid['n_positives'].iloc[0] + qwen_grid['n_negatives'].iloc[0]:.0f} images "
      f"(FPR={tpr1_row['fpr']:.3f})")

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
for ax, model in zip(axes.ravel(), co.MODELS):
    for prompt in sorted(base_local["prompt"].unique()):
        sub = roc_df[(roc_df["model"] == model) & (roc_df["prompt"] == prompt)].sort_values("fpr")
        ax.plot(sub["fpr"], sub["tpr"], marker="o", markersize=2, linewidth=1.2, label=prompt)
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
    ax.set_title(model, fontsize=10)
    ax.set_xlabel("false positive rate")
axes[0, 0].set_ylabel("true positive rate")
axes[1, 0].set_ylabel("true positive rate")
axes[0, 0].legend(fontsize=7, loc="lower right")
fig.suptitle("Layer 2 — ROC, positive class = |e| > 20.94 at response level, no test", y=1.01)
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q5_roc.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# **What to look for.** A curve with few distinct points rather than a
# smooth one is coarse confidence again, exactly as in the granularity chart
# above, and a curve that must climb close to FPR=1 before reaching TPR=1 is
# the signature of a configuration whose worst and its ordinary images share
# a confidence level — the Qwen-2.5 case above.

# %% [markdown]
# ## Layer 3 — what the models respond to
#
# Five scene features, one statistic, no threshold and no adjustment. For
# each cell and each feature: build every pair of one image carrying the
# feature and one image without it, count the pairs where the feature-bearing
# image received the **lower** confidence (ties counting as half), and
# divide by the number of pairs. **50% means no association; above 50% means
# confidence runs lower when the feature is present.**
#
# Counting pairs never orders the levels of a variable, which is exactly what
# makes this admissible on `rooted_dead_alike_plants`. That variable is
# nominal, which rules out any rank correlation, trend test or 0/1/2
# score, but a pairwise count imposes no such ordering. Splitting the
# variable into two present-or-absent contrasts (counted off vs. counted in)
# also recovers a distinction a single coefficient would average away.
#
# Four of the five features are present-or-absent contrasts against **that
# annotation's own zero level** — `rooted_dead_alike_plants == 0` for the two
# rooted rows, `non_rooted_plant_material == 0` for the two non-rooted rows —
# never against the `clean` subset where both annotations are zero at once.
# This is a precise choice, not a loose one, for two reasons. A nominal
# variable admits **category contrasts**, and level 1 against level 0 of the
# same variable is exactly that; level 1 against a cross-classified cell (both
# annotations zero at once) is a different construct, and not one this
# variable's levels define. And restricting to `clean` compresses the one distinction the
# non-rooted rows exist to expose — the scattered-versus-spread gap narrows
# from 7.3 points to 6.2 under `clean`, blurring exactly the contrast this
# layer is built to show. The fifth feature, obliquity, has no natural
# "absent" state, so it instead contrasts, over every possible pair of
# images, the more oblique member against the less oblique one.
#
# **Confidence is compared raw, with no adjustment for the model's own error,
# and this is deliberate.** It is tempting to control for absolute error
# first, on the grounds that images carrying this material are also images
# the model got more wrong. That would be the wrong move. Error and
# confidence are two outputs of the same process, not a cause and an effect,
# so error is not a confounder here, and adjusting for it conditions on a
# sibling outcome rather than removing a nuisance variable. It would also
# compare unusual
# subsets against each other — inside the 0-20% cover bin, images carrying
# counted-off material run roughly 12 cover points of overestimation against
# about 7 for the counted-in ones, so matching on error would pair the
# feature images against a slice of the reference group that is itself
# atypical. The question this layer asks is whether confidence shows a
# pattern when the model faces this material; the raw, unadjusted comparison
# is what answers it.
#
# Each row here is a **marginal** comparison, not one feature's effect with
# the other held fixed: a substantial share of material-bearing images carry
# both the rooted and the non-rooted kind, so the four material contrasts
# are not mutually adjusted for each other. And a cell with very coarse
# confidence is mechanically pulled toward 50% by ties — such a row is read
# against the granularity table above, never as evidence of no effect (K8).

# %%
def pct_pairs_lower(conf_feature: np.ndarray, conf_ref: np.ndarray) -> float:
    """% of (feature, reference) pairs where the feature-bearing image has
    the strictly lower confidence, ties counting as half. 50 = no
    association; above 50 = confidence lower when the feature is present.
    """
    a = np.asarray(conf_feature, dtype=float)
    b = np.asarray(conf_ref, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.nan
    less = (a[:, None] < b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return float((less + 0.5 * eq) / (a.size * b.size) * 100)


def pct_pairs_more_oblique_lower(conf: np.ndarray, obliquity: np.ndarray) -> float:
    """Over every pair of images with distinct obliquity, % of pairs where
    the MORE oblique member has the lower confidence, ties counting as half.
    Pairs tied on obliquity itself contribute nothing (there is no defined
    'more oblique' member), matching the case count reported alongside.

    Mathematically identical to building the full pairwise comparison
    matrix and counting `hi_obliquity_confidence < lo_obliquity_confidence`
    (plus half the ties), but computed in O(n log K) with no Python-level
    loop over images at all — at this notebook's B=10,000 resamples x 24
    cells x 2 obliquity views, the O(n^2) matrix form is too slow to render
    and even a per-image Python loop (a Fenwick-tree sweep) is a material
    cost at that call volume. The trick: sort images by obliquity once, then
    for every image the count of strictly-lower-obliquity predecessors at
    each confidence level is a running per-level tally — computed for every
    image at once via a cumulative sum over a (n_images x n_confidence
    levels) one-hot matrix, corrected so that images tied on obliquity with
    each other are excluded from each other's tally (obliquity is continuous
    so such ties are rare, but the correction handles them exactly). This is
    cross-checked against the direct O(n^2) definition at the point estimate
    for every cell, below.
    """
    conf = np.asarray(conf, dtype=float)
    obliq = np.asarray(obliquity, dtype=float)
    n = len(conf)
    if n < 2:
        return np.nan

    order = np.argsort(obliq, kind="mergesort")
    sorted_obliq = obliq[order]
    sorted_conf = conf[order]

    _, conf_rank = np.unique(sorted_conf, return_inverse=True)
    K = int(conf_rank.max()) + 1

    onehot = np.zeros((n, K), dtype=np.int64)
    onehot[np.arange(n), conf_rank] = 1
    # cum_before[i, k] = count of rows 0..i-1 (in obliquity order) at confidence rank k
    cum_before = np.cumsum(onehot, axis=0) - onehot

    # A row's true predecessor tally must only include STRICTLY smaller
    # obliquity, so same-obliquity tie-blocks share the tally value taken at
    # the block's first row (before any member of the block was counted).
    is_block_start = np.empty(n, dtype=bool)
    is_block_start[0] = True
    is_block_start[1:] = sorted_obliq[1:] != sorted_obliq[:-1]
    block_id = np.cumsum(is_block_start) - 1
    first_row_of_block = np.flatnonzero(is_block_start)
    predecessor_tally = cum_before[first_row_of_block][block_id]  # (n, K)

    r = conf_rank
    rows = np.arange(n)
    eq_cnt = predecessor_tally[rows, r]
    cum_by_rank = np.cumsum(predecessor_tally, axis=1)
    total_prior = cum_by_rank[:, -1]
    le_cnt = cum_by_rank[rows, r]
    higher_cnt = total_prior - le_cnt

    favor = float(np.sum(higher_cnt + 0.5 * eq_cnt))
    n_pairs = int(np.sum(total_prior))
    if n_pairs == 0:
        return np.nan
    return float(favor / n_pairs * 100)


def _pct_pairs_more_oblique_lower_bruteforce(conf: np.ndarray, obliquity: np.ndarray) -> float:
    """Direct O(n^2) definition, kept only as a correctness check on the
    Fenwick-tree implementation above (used once per cell at the point
    estimate, never inside a bootstrap replicate).
    """
    conf = np.asarray(conf, dtype=float)
    obliq = np.asarray(obliquity, dtype=float)
    n = len(conf)
    iu = np.triu_indices(n, k=1)
    oi, oj = obliq[iu[0]], obliq[iu[1]]
    ci, cj = conf[iu[0]], conf[iu[1]]
    valid = oi != oj
    oi, oj, ci, cj = oi[valid], oj[valid], ci[valid], cj[valid]
    if oi.size == 0:
        return np.nan
    hi_conf = np.where(oi > oj, ci, cj)
    lo_conf = np.where(oi > oj, cj, ci)
    favor = np.sum(hi_conf < lo_conf) + 0.5 * np.sum(hi_conf == lo_conf)
    return float(favor / oi.size * 100)


FEATURES = [
    ("rooted_off", "rooted dead look-alike, counted off", "rooted_dead_alike_plants", 1),
    ("rooted_in", "rooted dead look-alike, counted in", "rooted_dead_alike_plants", 2),
    ("nonrooted_scattered", "non-rooted material, scattered", "non_rooted_plant_material", 1),
    ("nonrooted_spread", "non-rooted material, spread", "non_rooted_plant_material", 2),
]

feature_rows = []
for model, prompt in CELLS:
    g = base_local[(base_local["model"] == model) & (base_local["prompt"] == prompt)].reset_index(drop=True)
    g_cc = complete_case(g)

    unusable_arr = g["confidence_unusable"].to_numpy()
    conf_arr = g["confidence"].to_numpy(dtype=float)
    rooted_arr = g["rooted_dead_alike_plants"].to_numpy()
    nonrooted_arr = g["non_rooted_plant_material"].to_numpy()
    obliquity_arr = g["obliquity"].to_numpy(dtype=float)

    for feature_id, feature_label, col, level in FEATURES:
        # The reference is that annotation's own zero level -- D2.v3 == 0 for
        # the two rooted rows, D2.v4 == 0 for the two non-rooted rows -- never
        # the cross-classified `clean` subset (both annotations zero at
        # once). Level 1 vs level 0 of the same nominal variable is a category
        # contrast, which the variable's scale admits; level 1 vs a
        # cross-classified cell is not.
        own_zero_conf = g_cc.loc[g_cc[col] == 0, "confidence"]
        feat_conf = g_cc.loc[g_cc[col] == level, "confidence"]
        pct = pct_pairs_lower(feat_conf.values, own_zero_conf.values)
        col_arr = rooted_arr if col == "rooted_dead_alike_plants" else nonrooted_arr

        def feature_row_stat(idx, _col_arr=col_arr, _level=level,
                              _unusable=unusable_arr, _conf=conf_arr):
            u = _unusable[idx]
            usable_mask = ~u
            c, col_r = _conf[idx][usable_mask], _col_arr[idx][usable_mask]
            feat_mask = col_r == _level
            own_zero_mask = col_r == 0
            return pct_pairs_lower(c[feat_mask], c[own_zero_mask])

        boot = cell_bootstrap(g, feature_row_stat, rng_feature, B=co.B_BOOTSTRAP)
        assert np.isclose(boot.estimate, pct, atol=1e-6), (
            f"feature association mismatch for {model}/{prompt}/{feature_id}: {boot.estimate} vs {pct}"
        )
        feature_rows.append({
            "question_id": "Q5", "model": model, "prompt": prompt,
            "feature": feature_id, "feature_label": feature_label,
            "pct_lower_confidence": pct,
            "ci_lo": boot.ci_lo, "ci_hi": boot.ci_hi, "ci_method": boot.ci_method,
            "n_with_feature": int(len(feat_conf)), "n_without_material": int(len(own_zero_conf)),
            "restricted_to_0_20_bin": False,
            "clearly_above_50": bool(boot.ci_lo > 50),
            "clearly_below_50": bool(boot.ci_hi < 50),
            "multiplicity_note": "24-cell interval verdicts, not corrected for multiplicity",
        })

    # Obliquity: all pairs, all images in the cell (no reference-set
    # restriction — every image participates as either the more or less
    # oblique member of a pair).
    g_cc_obliq_conf = g_cc["confidence"].to_numpy(dtype=float)
    g_cc_obliq = g.loc[~g["confidence_unusable"], "obliquity"].to_numpy(dtype=float)
    pct_obliquity = pct_pairs_more_oblique_lower(g_cc_obliq_conf, g_cc_obliq)
    pct_obliquity_check = _pct_pairs_more_oblique_lower_bruteforce(g_cc_obliq_conf, g_cc_obliq)
    assert np.isclose(pct_obliquity, pct_obliquity_check, atol=1e-6), (
        f"obliquity Fenwick/brute-force mismatch for {model}/{prompt}: {pct_obliquity} vs {pct_obliquity_check}"
    )

    def obliquity_row_stat(idx, _unusable=unusable_arr, _conf=conf_arr, _obliq=obliquity_arr):
        u = _unusable[idx]
        usable_mask = ~u
        return pct_pairs_more_oblique_lower(_conf[idx][usable_mask], _obliq[idx][usable_mask])

    boot_obliquity = cell_bootstrap(g, obliquity_row_stat, rng_feature, B=co.B_BOOTSTRAP)
    feature_rows.append({
        "question_id": "Q5", "model": model, "prompt": prompt,
        "feature": "obliquity_all", "feature_label": "obliquity, all images",
        "pct_lower_confidence": pct_obliquity,
        "ci_lo": boot_obliquity.ci_lo, "ci_hi": boot_obliquity.ci_hi, "ci_method": boot_obliquity.ci_method,
        "n_with_feature": int(len(g_cc)), "n_without_material": np.nan,
        "restricted_to_0_20_bin": False,
        "clearly_above_50": bool(boot_obliquity.ci_lo > 50),
        "clearly_below_50": bool(boot_obliquity.ci_hi < 50),
        "multiplicity_note": "24-cell interval verdicts, not corrected for multiplicity",
    })

    # Obliquity, restricted to the 0-20% cover bin -- reported both ways
    # because the restriction changes the answer here (unlike the four
    # material features, which agree within 2.5 points across the
    # restriction and so are reported only on the full frame).
    g020 = g[g["bin"] == "0-20"].reset_index(drop=True)
    g020_cc = complete_case(g020)
    pct_obliquity_020 = pct_pairs_more_oblique_lower(
        g020_cc["confidence"].to_numpy(dtype=float), g020_cc["obliquity"].to_numpy(dtype=float)
    )

    unusable_020 = g020["confidence_unusable"].to_numpy()
    conf_020 = g020["confidence"].to_numpy(dtype=float)
    obliq_020 = g020["obliquity"].to_numpy(dtype=float)

    def obliquity_020_row_stat(idx, _unusable=unusable_020, _conf=conf_020, _obliq=obliq_020):
        u = _unusable[idx]
        usable_mask = ~u
        return pct_pairs_more_oblique_lower(_conf[idx][usable_mask], _obliq[idx][usable_mask])

    boot_obliquity_020 = cell_bootstrap(g020, obliquity_020_row_stat, rng_feature, B=co.B_BOOTSTRAP)
    feature_rows.append({
        "question_id": "Q5", "model": model, "prompt": prompt,
        "feature": "obliquity_0_20_bin", "feature_label": "obliquity, 0-20% bin only",
        "pct_lower_confidence": pct_obliquity_020,
        "ci_lo": boot_obliquity_020.ci_lo, "ci_hi": boot_obliquity_020.ci_hi,
        "ci_method": boot_obliquity_020.ci_method,
        "n_with_feature": int(len(g020_cc)), "n_without_material": np.nan,
        "restricted_to_0_20_bin": True,
        "clearly_above_50": bool(boot_obliquity_020.ci_lo > 50),
        "clearly_below_50": bool(boot_obliquity_020.ci_hi < 50),
        "multiplicity_note": "24-cell interval verdicts, not corrected for multiplicity",
    })

feature_association_df = pd.DataFrame(feature_rows)

pooled_summary = (
    feature_association_df.groupby(["feature", "feature_label"])
    .agg(
        median_pct=("pct_lower_confidence", "median"),
        min_pct=("pct_lower_confidence", "min"),
        max_pct=("pct_lower_confidence", "max"),
        n_clearly_above_50=("clearly_above_50", "sum"),
        n_clearly_below_50=("clearly_below_50", "sum"),
        n_cells=("pct_lower_confidence", "size"),
    )
    .reset_index()
)
pooled_summary["question_id"] = "Q5"
print(pooled_summary.to_string(index=False))

# Worked per-cell checks, all 1,155 images, own-zero reference.
_worked_cells = [
    ("Llama-4-Maverick", "Grid-Overlay", "rooted_off", 73.6, "rooted_in", 77.6),
    ("Llama-4-Scout", "Short", "rooted_off", 72.7, "rooted_in", 75.2),
]
print("\nWorked per-cell check (all images, own-zero reference):")
for model, prompt, feat_a, expect_a, feat_b, expect_b in _worked_cells:
    row_a = feature_association_df[
        (feature_association_df["model"] == model) & (feature_association_df["prompt"] == prompt)
        & (feature_association_df["feature"] == feat_a)
    ].iloc[0]
    row_b = feature_association_df[
        (feature_association_df["model"] == model) & (feature_association_df["prompt"] == prompt)
        & (feature_association_df["feature"] == feat_b)
    ].iloc[0]
    print(f"  {model} x {prompt}: {feat_a}={row_a['pct_lower_confidence']:.1f} (expect ~{expect_a}), "
          f"{feat_b}={row_b['pct_lower_confidence']:.1f} (expect ~{expect_b})")
    assert np.isclose(row_a["pct_lower_confidence"], expect_a, atol=0.5), (
        f"{model}/{prompt}/{feat_a} does not match the published value"
    )
    assert np.isclose(row_b["pct_lower_confidence"], expect_b, atol=0.5), (
        f"{model}/{prompt}/{feat_b} does not match the published value"
    )

# %% [markdown]
# **Reading the pooled table.** Both kinds of rooted dead look-alike material
# pull confidence down by nearly identical amounts — the median percentage
# for "counted off" and "counted in" are within a point of each other, and
# the per-cell values track closely — so the models see the material and
# respond to its presence, but do not register the observers' distinction
# between what was counted as vegetation and what was not. That distinction
# is exactly where the practical damage sits: **inside the 0-20% cover bin**
# the counted-off images carry roughly 12 cover points of overestimation
# against about 7 for the counted-in ones (reported in Q3), so a model
# treating the two identically is blind to which error it is more prone to.
# Non-rooted material shows the
# same pattern in a different shape: it registers as a lower-confidence
# signal only when spread as a continuous layer, not when scattered as loose
# pieces largely invisible to the models. Obliquity is the reportable
# negative — a small residue above 50% on the full frame that collapses
# toward 50% once cover is held fixed in the 0-20% bin, consistent with
# oblique images simply tending to be denser rather than the model
# responding to viewing geometry itself. Whatever obliquity does to error,
# it does not show up as a confidence response.
#
# **The claim boundary.** This layer says confidence responds to the
# material present in a scene. It does not say confidence responds to the
# material *beyond* responding to the model's own error on that scene — the
# two are not disentangled here, deliberately, for the reason given above.

# %%
fig, ax = plt.subplots(figsize=(9, 5.5))
feature_order = ["rooted_off", "rooted_in", "nonrooted_scattered", "nonrooted_spread",
                  "obliquity_all", "obliquity_0_20_bin"]
labels_for_plot = {
    "rooted_off": "rooted dead look-alike\n(counted off)",
    "rooted_in": "rooted dead look-alike\n(counted in)",
    "nonrooted_scattered": "non-rooted material\n(scattered)",
    "nonrooted_spread": "non-rooted material\n(spread)",
    "obliquity_all": "obliquity\n(all images)",
    "obliquity_0_20_bin": "obliquity\n(0-20% bin only)",
}
positions = np.arange(len(feature_order))
for i, feat in enumerate(feature_order):
    sub = feature_association_df[feature_association_df["feature"] == feat]
    ax.scatter(np.full(len(sub), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(sub)),
               sub["pct_lower_confidence"], color="steelblue", alpha=0.6, s=25)
    med = pooled_summary.loc[pooled_summary["feature"] == feat, "median_pct"].iloc[0]
    ax.scatter([i], [med], color="firebrick", marker="D", s=60, zorder=5)
ax.axhline(50, color="grey", linestyle=":", linewidth=1)
ax.set_xticks(positions)
ax.set_xticklabels([labels_for_plot[f] for f in feature_order], fontsize=8)
ax.set_ylabel("% of pairs where the feature image had lower confidence")
ax.set_title("Layer 3 — 24 per-cell values (blue) and the median across cells (red diamond)")
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q5_layer3_feature_association.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# **What to look for.** The dotted line at 50% is "no association." The four
# material features cluster clearly above it; obliquity on the full frame
# sits just above 50% and collapses toward it once restricted to the 0-20%
# bin — the visual signature of a cover-mediated residue rather than a
# genuine response to viewing geometry.

# %% [markdown]
# ## Layer 4 — the operational sweep, and missing confidence
#
# What does filtering on confidence actually buy? At every realisable
# cutoff (every distinct confidence value a cell emits, applied as "retain
# images with confidence at or above this level"), both `MAE_o` (overall MAE)
# and `MAE_b` (balanced MAE) are recomputed on what remains, alongside the
# retained fraction and the cover composition of the retained sample.
# **Balanced MAE becomes undefined the instant a bin empties** — this is
# reported directly as `NaN` rather than argued around, since it is exactly
# the trap a naive "just filter on confidence" strategy walks into once a
# sparse high-cover bin loses its last image.
#
# **Unusable confidence is never dropped.** An image with no usable
# confidence cannot be triaged at all, so it counts as **unflagged and
# unverified**: it is never removed by any cutoff (it stays in the retained
# set at every threshold), on the reasoning that a configuration should not
# be rewarded for failing to emit a value on its hardest images.

# %%
def sweep_at_cutoff(g: pd.DataFrame, cutoff: float) -> dict:
    """Retain rows with usable confidence >= cutoff, plus every row with
    unusable confidence (unflagged and unverified -- never dropped).
    """
    keep = (g["confidence_unusable"]) | (g["confidence"] >= cutoff)
    retained = g.loc[keep]
    mae_o = float(retained["abs_e"].mean()) if len(retained) else np.nan
    bin_counts = retained["bin"].value_counts().reindex(co.BIN_LABELS, fill_value=0)
    if (bin_counts == 0).any():
        mae_b = np.nan
    else:
        mae_b = co.balanced_mae(retained["abs_e"], retained["bin"]).balanced_mae
    return {
        "n_retained": len(retained), "retained_fraction": len(retained) / len(g),
        "mae_o": mae_o, "mae_b": mae_b,
        **{f"n_retained_bin_{b}": int(bin_counts[b]) for b in co.BIN_LABELS},
    }


def sweep_bootstrap_vectorized(
    g: pd.DataFrame, cutoffs: list[float], rng: np.random.Generator, B: int = co.B_BOOTSTRAP,
) -> dict[float, dict]:
    """Bootstrap MAE_o and MAE_b at every cutoff for one cell, in a single
    pass over B resamples rather than one `image_bootstrap` call per cutoff.

    Each cell has one row per image (24 cells, one prediction row per image
    each), so resampling images with replacement is exactly resampling rows
    with replacement here -- no groupby/rebuild is needed the way the
    general-purpose `image_bootstrap` requires for a multi-row-per-image
    frame. This keeps the sweep's cost to O(B x n) per cell instead of
    O(B x n_cutoffs) frame rebuilds, since the codebase's own per-combo MCB
    bootstrap (Q1) uses the identical fast-path justification.
    """
    n = len(g)
    unusable = g["confidence_unusable"].to_numpy()
    conf = g["confidence"].to_numpy(dtype=float)
    abs_e = g["abs_e"].to_numpy(dtype=float)
    bin_codes = pd.Categorical(g["bin"], categories=co.BIN_LABELS).codes
    n_bins = len(co.BIN_LABELS)

    out = {c: {"mae_o": np.empty(B), "mae_b": np.full(B, np.nan)} for c in cutoffs}
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        u_b, c_b, a_b, bin_b = unusable[idx], conf[idx], abs_e[idx], bin_codes[idx]
        for cutoff in cutoffs:
            keep = u_b | (c_b >= cutoff)
            a_keep = a_b[keep]
            out[cutoff]["mae_o"][b] = a_keep.mean() if a_keep.size else np.nan
            bin_keep = bin_b[keep]
            sums = np.zeros(n_bins)
            counts = np.zeros(n_bins)
            np.add.at(sums, bin_keep, a_keep)
            np.add.at(counts, bin_keep, 1)
            if (counts == 0).any():
                out[cutoff]["mae_b"][b] = np.nan
            else:
                out[cutoff]["mae_b"][b] = (sums / counts).mean()
    return out


def percentile_ci(vals: np.ndarray, alpha: float = 0.05) -> tuple[float, float, str]:
    valid = vals[np.isfinite(vals)]
    if len(valid) < 50:
        return np.nan, np.nan, "not computed (too many empty-bin resamples)"
    return (float(np.percentile(valid, 100 * alpha / 2)),
            float(np.percentile(valid, 100 * (1 - alpha / 2))),
            "percentile (bin can empty on resample)")


sweep_rows = []
for model, prompt in CELLS:
    g = base_local[(base_local["model"] == model) & (base_local["prompt"] == prompt)].reset_index(drop=True)
    usable = g.loc[~g["confidence_unusable"]]
    cutoffs = sorted(usable["confidence"].unique())
    n_excluded = int(g["confidence_unusable"].sum())

    boot_by_cutoff = sweep_bootstrap_vectorized(g, cutoffs, rng_sweep, B=co.B_BOOTSTRAP)

    for cutoff in cutoffs:
        point = sweep_at_cutoff(g, cutoff)
        boot_o_vals = boot_by_cutoff[cutoff]["mae_o"]
        boot_b_vals = boot_by_cutoff[cutoff]["mae_b"]

        o_ci_lo, o_ci_hi, o_ci_method = percentile_ci(boot_o_vals)
        mae_b_point = point["mae_b"]
        if np.isfinite(mae_b_point):
            frac_undefined = float(np.isnan(boot_b_vals).mean())
            b_ci_lo, b_ci_hi, b_ci_method = percentile_ci(boot_b_vals)
        else:
            frac_undefined, b_ci_lo, b_ci_hi, b_ci_method = (
                1.0, np.nan, np.nan, "not computed (bin empty at point estimate)"
            )

        sweep_rows.append({
            "question_id": "Q5", "model": model, "prompt": prompt, "cutoff": cutoff,
            "n_total": len(g), "n_excluded_unusable_confidence": n_excluded,
            "n_retained": point["n_retained"], "retained_fraction": point["retained_fraction"],
            "mae_o": point["mae_o"], "mae_o_ci_lo": o_ci_lo, "mae_o_ci_hi": o_ci_hi,
            "mae_o_ci_method": o_ci_method,
            "mae_b": mae_b_point, "mae_b_ci_lo": b_ci_lo, "mae_b_ci_hi": b_ci_hi,
            "mae_b_ci_method": b_ci_method, "mae_b_undefined_share_of_resamples": frac_undefined,
            **{k: point[k] for k in point if k.startswith("n_retained_bin_")},
        })

sweep_df = pd.DataFrame(sweep_rows)
print(f"{len(sweep_df)} sweep rows across 24 cells "
      f"({sweep_df.groupby(['model','prompt']).size().mean():.1f} cutoffs/cell on average)")
n_mae_b_undefined = int(sweep_df["mae_b"].isna().sum())
print(f"Rows where MAE_b is undefined at the point estimate (an emptied bin): {n_mae_b_undefined} of {len(sweep_df)}")

sweep_df.sort_values(["model", "prompt", "cutoff"]).head(8)

# %% [markdown]
# **What to look for.** `mae_o` should fall as the cutoff rises (the
# remaining images are the ones confidence rated highest) whenever Layer 1's
# association is genuinely negative; a flat or rising `mae_o` under
# filtering is the sweep's own evidence that a cell's confidence is not
# informative. `mae_b` turning to `NaN` partway through the sweep is not a
# computation failure — it is balanced MAE's definition failing exactly when
# a sparse high-cover bin loses its last retained image, which a
# practitioner filtering on confidence alone would hit without warning.

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
for ax, model in zip(axes.ravel(), co.MODELS):
    for prompt in sorted(base_local["prompt"].unique()):
        sub = sweep_df[(sweep_df["model"] == model) & (sweep_df["prompt"] == prompt)].sort_values("retained_fraction")
        ax.plot(sub["retained_fraction"], sub["mae_o"], marker="o", markersize=2, linewidth=1.2, label=prompt)
    ax.set_title(model, fontsize=10)
    ax.set_xlabel("retained fraction")
axes[0, 0].set_ylabel("MAE_o on retained images (cover points)")
axes[1, 0].set_ylabel("MAE_o on retained images (cover points)")
axes[0, 0].legend(fontsize=7, loc="upper left")
fig.suptitle("Layer 4 — overall MAE under confidence filtering, unusable confidence never dropped", y=1.01)
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q5_layer4_filtering_sweep.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Reading Layer 1 and Layer 3 by model, and by prompt
#
# Every result above is reported at the model x prompt cell, because that is
# the unit this notebook's design actually supports. But the Results section
# of the manuscript reads confidence **by model**: five of the six rank their
# own errors in the right direction at the model level, and Gemma-3-12B is the
# one that ranks them backwards. The reversal is not uniform across its own
# prompts either: under *Short* its cell-level rho is -0.062 [-0.124, -0.002],
# the correct direction, while the other three run positive. Averaging
# four cell-level numbers on paper, in prose, would put a median into the
# manuscript with no interval and no code behind it — nothing enters the
# result unless a computation here produced it, so the model view and the
# prompt view are computed directly, the same way every cell-level number
# above was.
#
# Both axes are views of the **same 27,720 base/local rows**. The six model
# rows below are six different subsets of all 1,155 images (one subset per
# model, each covering all 1,155 images pooled over its four prompts); the
# four prompt rows are four different subsets of the same 1,155 images
# (pooled over the six models). A model row and a prompt row therefore never
# add information the cell-level tables above did not already contain — they
# only regroup it along the axis a reader actually wants to compare. Because
# every level is a reshuffling of the same underlying responses rather than
# an independent draw, a difference between two model rows, or two prompt
# rows, is not a hypothesis test, and no interval below should be read as
# one — the same "not corrected for multiplicity" caveat that applies to the
# 24 per-cell verdicts above applies here for the same reason: descriptive
# intervals around a fixed pooling of the same data, not the outcome of a
# pre-declared test.
#
# Pooling four or six cells' worth of responses together also means
# `confidence` is being averaged within a level, at the image level, before
# the rank correlation is computed. Averaging an ordinal score is already a
# convention this notebook does not extend past its safest use: it is
# defensible **within one model**, because every response being averaged sits
# on that one model's own confidence scale, but it is never used to compare
# raw confidence values *between* models anywhere in this notebook — nothing
# below computes a cross-model difference in mean confidence, only a
# within-level rank correlation between that level's own mean confidence and
# its own mean absolute error.

# %%
rng_rank_by_axis = np.random.default_rng(SEED + 5)
rng_feature_by_axis = np.random.default_rng(SEED + 6)

AXIS_LEVELS = [("model", m) for m in co.MODELS] + [("prompt", p) for p in sorted(base_local["prompt"].unique())]


def per_image_means_for_level(frame_level: pd.DataFrame) -> pd.DataFrame:
    """Collapse a level's rows (one model's four prompts, or one prompt's
    six models) to one row per image: that image's mean confidence and mean
    absolute error over the level's remaining usable-confidence rows. The
    unit of Layer 1's rank correlation is the image, never the response, so
    a photograph that contributes several rows to a level (up to four for a
    model, up to six for a prompt) must enter the correlation exactly once.
    """
    return (
        frame_level.groupby("image", as_index=False)
        .agg(confidence=("confidence", "mean"), abs_e=("abs_e", "mean"))
    )


rank_by_axis_rows = []
for axis, level in AXIS_LEVELS:
    g_all = base_local[base_local[axis] == level].reset_index(drop=True)
    n_images_all = g_all["image"].nunique()
    g_cc = complete_case(g_all)
    n_excluded = int(g_all["confidence_unusable"].sum())

    per_image = per_image_means_for_level(g_cc)
    n_images_used = len(per_image)

    conf_img = per_image["confidence"].to_numpy(dtype=float)
    abs_e_img = per_image["abs_e"].to_numpy(dtype=float)
    rho_pooled = co.spearman_tie_corrected(conf_img, abs_e_img)[0] if n_images_used >= 2 else np.nan

    def rho_pooled_row_stat(idx, _conf=conf_img, _abs_e=abs_e_img):
        if len(idx) < 2:
            return np.nan
        rho, _ = co.spearman_tie_corrected(_conf[idx], _abs_e[idx])
        return rho

    n_img_all = len(conf_img)
    boot_pooled = cell_bootstrap(
        pd.DataFrame({"_i": np.arange(n_img_all)}), rho_pooled_row_stat, rng_rank_by_axis, B=co.B_BOOTSTRAP,
    )

    g020_level = g_all[g_all["bin"] == "0-20"].reset_index(drop=True)
    g020_cc = complete_case(g020_level)
    per_image_020 = per_image_means_for_level(g020_cc)
    n_020 = len(per_image_020)
    conf_020 = per_image_020["confidence"].to_numpy(dtype=float)
    abs_e_020 = per_image_020["abs_e"].to_numpy(dtype=float)
    rho_020 = co.spearman_tie_corrected(conf_020, abs_e_020)[0] if n_020 >= 2 else np.nan

    if n_020 >= 2:
        def rho_020_row_stat(idx, _conf=conf_020, _abs_e=abs_e_020):
            if len(idx) < 2:
                return np.nan
            rho, _ = co.spearman_tie_corrected(_conf[idx], _abs_e[idx])
            return rho

        boot_020 = cell_bootstrap(
            pd.DataFrame({"_i": np.arange(n_020)}), rho_020_row_stat, rng_rank_by_axis, B=co.B_BOOTSTRAP,
        )
    else:
        boot_020 = None

    rank_by_axis_rows.append({
        "question_id": "Q5", "axis": axis, "level": level,
        "n_images": n_images_all, "n_excluded_unusable_confidence": n_excluded,
        "rho_pooled": rho_pooled,
        "rho_pooled_ci_lo": boot_pooled.ci_lo, "rho_pooled_ci_hi": boot_pooled.ci_hi,
        "rho_pooled_ci_method": boot_pooled.ci_method,
        "rho_0_20_bin": rho_020,
        "rho_0_20_bin_ci_lo": boot_020.ci_lo if boot_020 is not None else np.nan,
        "rho_0_20_bin_ci_hi": boot_020.ci_hi if boot_020 is not None else np.nan,
        "rho_0_20_bin_ci_method": boot_020.ci_method if boot_020 is not None else "not computed (n<2)",
        "n_0_20_bin": n_020,
        "direction_claimed": bool(boot_pooled.ci_lo > 0 or boot_pooled.ci_hi < 0),
        "descriptive_uncorrected": True,
        "note": (
            "the six model rows are six views of the same 1,155 images, and the four "
            "prompt rows are likewise four views of the same 1,155 images, so a "
            "difference between two levels of either axis is not a test and no interval "
            "here supports one; per-image mean confidence averages an ordinal score, "
            "which is defensible within one model's own scale, and no comparison of raw "
            "confidence values between models is made anywhere in this notebook"
        ),
    })

rank_association_by_axis_df = pd.DataFrame(rank_by_axis_rows)
print(rank_association_by_axis_df[["axis", "level", "n_images", "rho_pooled", "rho_pooled_ci_lo", "rho_pooled_ci_hi"]]
      .to_string(index=False))

# %% [markdown]
# **Reading the model rows.** Five of the six models post a negative pooled
# rho whose interval excludes zero — confidence tracks error in the
# direction that would make it useful for triage — but the magnitude varies
# widely, from -0.22 (Mistral-Small-3.2) to -0.72 (Llama-4-Maverick), so
# "excludes zero" alone does not separate a strong association from a weak
# one. Gemma-3-12B is the only model whose pooled rho sits on the positive
# side: its confidence runs *backwards*, rating scenes it gets more wrong as
# more, not less, confident. This is read directly from the model rows
# below, not asserted from the cell-level table above.

# %% [markdown]
# ## Layer 2's ROC, by model and by prompt
#
# The same realisable-cutoff ROC Layer 2 builds per cell, rebuilt on the same
# ten pooled levels: for one level, every base-variant response belonging to
# that model (its four prompts) or that prompt (its six models) is pooled,
# unusable-confidence responses stay in both denominators and are never
# predicted positive at any threshold, exactly as `roc_points` already
# handles them, and the true-positive / false-positive rate is read off at
# each confidence value the pooled level actually emits, against the same
# 20.94-cover-point fence defined once above. `roc_points` is reused
# unchanged — pooling six or four cells together changes nothing about what
# counts as a threshold or a positive, only how many responses are pooled
# before the curve is built.

# %%
roc_by_axis_rows = []
for axis, level in AXIS_LEVELS:
    g_all = base_local[base_local[axis] == level]
    pts = roc_points(g_all)
    n_pos = int(g_all["positive"].sum())
    n_neg = int((~g_all["positive"]).sum())
    note = (
        f"pooled over the {'four prompts' if axis == 'model' else 'six models'} of "
        "the same 1,155 images; this curve describes the level as a whole and is not "
        "a paired comparison against any other level"
    )
    for _, r in pts.iterrows():
        roc_by_axis_rows.append({
            "question_id": "Q5", "axis": axis, "level": level,
            "confidence_level": r["confidence_level"], "tpr": r["tpr"], "fpr": r["fpr"],
            "n_at_level": int(r["n_at_level"]), "n_positives": n_pos, "n_negatives": n_neg,
            "descriptive_uncorrected": True, "note": note,
        })

roc_association_by_axis_df = pd.DataFrame(roc_by_axis_rows)
print(f"{len(roc_association_by_axis_df)} operating points across {len(AXIS_LEVELS)} levels "
      f"({roc_association_by_axis_df.groupby(['axis', 'level']).size().mean():.1f} points/level on average)")

# %% [markdown]
# **Reading the model rows.** Pooling a model's four prompts together gives
# each model's own ROC curve the same shape the per-cell curves above already
# hinted at. By trapezoid area under the pooled curve, Llama-4-Maverick leads
# (0.660), followed by Qwen-2.5 (0.628), Gemma-3-27B (0.584) and Llama-4-Scout
# (0.575) in a middle band, with Mistral-Small-3.2 sitting almost exactly on
# the diagonal (0.503). Gemma-3-12B is the outlier at the other end, not the
# curve closest to chance: its area is 0.384, the only one **below** the
# diagonal. In distance from the diagonal it is third, behind Llama-4-Maverick
# (0.160) and Qwen-2.5 (0.128); what sets it apart is the side it falls on — its ranking rule
# performs worse than a coin flip, the visual counterpart of its
# backwards-signed pooled rho above.

# %% [markdown]
# ## The same six features, by model and by prompt
#
# Layer 3's pairwise feature association, recomputed on the same ten levels.
# The unit is the image, exactly as the respec requires and exactly as the
# model/prompt rank correlation above already does: a level's rows are first
# collapsed to one row per image (that image's mean confidence over the
# level's usable-confidence rows), because `rooted_dead_alike_plants`,
# `non_rooted_plant_material` and `obliquity` are themselves per-image
# constants — every response for a given image carries the same value on all
# three — so pairing at the response level would count one photograph as
# though it were several independent observations of its own feature value.
# The pairwise statistic itself, `pct_pairs_lower` /
# `pct_pairs_more_oblique_lower`, is reused unchanged; only its two input
# arrays are now per-image means rather than per-response values.

# %%
def per_image_mean_confidence_for_level(frame_level: pd.DataFrame) -> pd.DataFrame:
    """Collapse a level's usable-confidence rows to one row per image (mean
    confidence), carrying the image's own fixed feature columns along
    unchanged since `rooted_dead_alike_plants`, `non_rooted_plant_material`
    and `obliquity` never vary within an image.
    """
    return frame_level.groupby("image", as_index=False).agg(
        confidence=("confidence", "mean"),
        rooted_dead_alike_plants=("rooted_dead_alike_plants", "first"),
        non_rooted_plant_material=("non_rooted_plant_material", "first"),
        obliquity=("obliquity", "first"),
    )


feature_by_axis_rows = []
for axis, level in AXIS_LEVELS:
    g_all = base_local[base_local[axis] == level].reset_index(drop=True)
    g_cc = complete_case(g_all)
    per_image_all = per_image_mean_confidence_for_level(g_cc)

    conf_img = per_image_all["confidence"].to_numpy(dtype=float)
    rooted_img = per_image_all["rooted_dead_alike_plants"].to_numpy()
    nonrooted_img = per_image_all["non_rooted_plant_material"].to_numpy()
    obliquity_img = per_image_all["obliquity"].to_numpy(dtype=float)

    for feature_id, feature_label, col, feat_level in FEATURES:
        col_img = rooted_img if col == "rooted_dead_alike_plants" else nonrooted_img
        own_zero_conf = conf_img[col_img == 0]
        feat_conf = conf_img[col_img == feat_level]
        pct = pct_pairs_lower(feat_conf, own_zero_conf)

        def feature_row_stat(idx, _col_img=col_img, _level=feat_level, _conf=conf_img):
            c, col_r = _conf[idx], _col_img[idx]
            feat_mask = col_r == _level
            own_zero_mask = col_r == 0
            return pct_pairs_lower(c[feat_mask], c[own_zero_mask])

        boot = cell_bootstrap(per_image_all, feature_row_stat, rng_feature_by_axis, B=co.B_BOOTSTRAP)
        feature_by_axis_rows.append({
            "question_id": "Q5", "axis": axis, "level": level,
            "feature": feature_id, "feature_label": feature_label,
            "pct_lower_confidence": pct,
            "ci_lo": boot.ci_lo, "ci_hi": boot.ci_hi, "ci_method": boot.ci_method,
            "n_with_feature": int(len(feat_conf)), "n_without_material": int(len(own_zero_conf)),
            "restricted_to_0_20_bin": False,
            "clearly_above_50": bool(boot.ci_lo > 50),
            "clearly_below_50": bool(boot.ci_hi < 50),
            "descriptive_uncorrected": True,
            "multiplicity_note": "24-cell interval verdicts, not corrected for multiplicity",
            "note": (
                "the six model rows are six views of the same 1,155 images, and the four "
                "prompt rows are likewise four views of the same 1,155 images, so a "
                "difference between two levels of either axis is not a test and no interval "
                "here supports one; the unit is the image, one row per image collapsed from "
                "the level's usable-confidence responses, and no comparison of raw confidence "
                "values between models is made anywhere in this notebook"
            ),
        })

    pct_obliquity = pct_pairs_more_oblique_lower(conf_img, obliquity_img)

    def obliquity_row_stat(idx, _conf=conf_img, _obliq=obliquity_img):
        return pct_pairs_more_oblique_lower(_conf[idx], _obliq[idx])

    boot_obliquity = cell_bootstrap(per_image_all, obliquity_row_stat, rng_feature_by_axis, B=co.B_BOOTSTRAP)
    feature_by_axis_rows.append({
        "question_id": "Q5", "axis": axis, "level": level,
        "feature": "obliquity_all", "feature_label": "obliquity, all images",
        "pct_lower_confidence": pct_obliquity,
        "ci_lo": boot_obliquity.ci_lo, "ci_hi": boot_obliquity.ci_hi, "ci_method": boot_obliquity.ci_method,
        "n_with_feature": int(len(per_image_all)), "n_without_material": np.nan,
        "restricted_to_0_20_bin": False,
        "clearly_above_50": bool(boot_obliquity.ci_lo > 50),
        "clearly_below_50": bool(boot_obliquity.ci_hi < 50),
        "descriptive_uncorrected": True,
        "multiplicity_note": "24-cell interval verdicts, not corrected for multiplicity",
        "note": (
            "the six model rows are six views of the same 1,155 images, and the four "
            "prompt rows are likewise four views of the same 1,155 images, so a "
            "difference between two levels of either axis is not a test and no interval "
            "here supports one; the unit is the image, one row per image collapsed from "
            "the level's usable-confidence responses, and no comparison of raw confidence "
            "values between models is made anywhere in this notebook"
        ),
    })

    g020_level = g_all[g_all["bin"] == "0-20"].reset_index(drop=True)
    g020_cc = complete_case(g020_level)
    per_image_020 = per_image_mean_confidence_for_level(g020_cc)
    conf_020 = per_image_020["confidence"].to_numpy(dtype=float)
    obliq_020 = per_image_020["obliquity"].to_numpy(dtype=float)
    pct_obliquity_020 = pct_pairs_more_oblique_lower(conf_020, obliq_020)

    def obliquity_020_row_stat(idx, _conf=conf_020, _obliq=obliq_020):
        return pct_pairs_more_oblique_lower(_conf[idx], _obliq[idx])

    boot_obliquity_020 = cell_bootstrap(per_image_020, obliquity_020_row_stat, rng_feature_by_axis, B=co.B_BOOTSTRAP)
    feature_by_axis_rows.append({
        "question_id": "Q5", "axis": axis, "level": level,
        "feature": "obliquity_0_20_bin", "feature_label": "obliquity, 0-20% bin only",
        "pct_lower_confidence": pct_obliquity_020,
        "ci_lo": boot_obliquity_020.ci_lo, "ci_hi": boot_obliquity_020.ci_hi,
        "ci_method": boot_obliquity_020.ci_method,
        "n_with_feature": int(len(per_image_020)), "n_without_material": np.nan,
        "restricted_to_0_20_bin": True,
        "clearly_above_50": bool(boot_obliquity_020.ci_lo > 50),
        "clearly_below_50": bool(boot_obliquity_020.ci_hi < 50),
        "descriptive_uncorrected": True,
        "multiplicity_note": "24-cell interval verdicts, not corrected for multiplicity",
        "note": (
            "the six model rows are six views of the same 1,155 images, and the four "
            "prompt rows are likewise four views of the same 1,155 images, so a "
            "difference between two levels of either axis is not a test and no interval "
            "here supports one; the unit is the image, one row per image collapsed from "
            "the level's usable-confidence responses, and no comparison of raw confidence "
            "values between models is made anywhere in this notebook"
        ),
    })

feature_association_by_axis_df = pd.DataFrame(feature_by_axis_rows)
print(f"{len(feature_association_by_axis_df)} rows across {len(AXIS_LEVELS)} levels x 6 features")
print(
    feature_association_by_axis_df[feature_association_by_axis_df["axis"] == "model"]
    .pivot(index="level", columns="feature", values="pct_lower_confidence")
    .round(1)
)

# %% [markdown]
# **Reading the model rows.** Five of the six models register both forms of
# rooted dead look-alike material by lowering confidence when it is present.
# Gemma-3-12B is the exception, and it runs the other way rather than merely
# failing to show the pattern: its confidence is *higher* on images carrying
# either form of the material (pct_lower_confidence in the 20s, against the
# high 50s to mid 70s for the other five models), consistent with the same
# model's backwards-signed pooled rho above. Non-rooted material spread as a
# continuous layer moves confidence down for most models; scattered as loose
# pieces moves it far less, clearly above 50% for only three models
# (Gemma-3-27B, Mistral-Small-3.2 and Qwen-2.5) and near the null for the rest
# including Gemma-3-12B, so the reversal is specific to the rooted material
# rather than a general property of this model's confidence. Obliquity's small residual sits just above 50%
# for most models on the full frame, consistent with the cell-level reading
# above: a cover-mediated residue rather than a response to viewing geometry
# in its own right.

# %% [markdown]
# ## Assumption checks
#
# No test is run anywhere in this notebook, so there is no test-validity
# assumption to check in the usual sense. What is checked instead is the
# machinery every layer above depends on: K8's confidence-degeneracy flag,
# which cell-level rows are read against rather than treated as a null; the
# MNAR accounting for unusable confidence, both as a per-cell count and as
# the bound on Layer 1's rho; and the unflagged-and-unverified rule that
# Layers 2 and 4 both implement.

# %%
assumption_rows = []
for model, prompt in CELLS:
    g = base_local[(base_local["model"] == model) & (base_local["prompt"] == prompt)]
    gran_row = granularity_df[(granularity_df["model"] == model) & (granularity_df["prompt"] == prompt)].iloc[0]
    rank_row = rank_association_df[(rank_association_df["model"] == model) & (rank_association_df["prompt"] == prompt)].iloc[0]
    assumption_rows.append({
        "id": "K8", "question_id": "Q5", "model": model, "prompt": prompt,
        "description": "confidence granularity — reported first, read against every later layer, never gating a test",
        "n_distinct_values": gran_row["n_distinct_values"],
        "max_single_level_mass": gran_row["max_single_level_mass"],
        "degenerate": gran_row["degenerate_k8"],
        "passed": True,
        "detail": f"n_distinct={gran_row['n_distinct_values']}, max_mass={gran_row['max_single_level_mass']:.3f}",
    })
    assumption_rows.append({
        "id": "MNAR", "question_id": "Q5", "model": model, "prompt": prompt,
        "description": "unusable confidence is missing not at random by design; bounded, never dropped from vegetation_percent analyses",
        "n_distinct_values": np.nan, "max_single_level_mass": np.nan, "degenerate": np.nan,
        "passed": True,
        "detail": f"n_excluded={int(g['confidence_unusable'].sum())}, "
                  f"rho_mnar_bound=[{rank_row['rho_mnar_bound_lo']:.3f}, {rank_row['rho_mnar_bound_hi']:.3f}]",
    })

assumption_rows.append({
    "id": "UNFLAGGED_UNVERIFIED", "question_id": "Q5", "model": np.nan, "prompt": np.nan,
    "description": "images with no usable confidence are counted as unflagged and unverified in the ROC and the filtering sweep, never silently dropped",
    "n_distinct_values": np.nan, "max_single_level_mass": np.nan, "degenerate": np.nan,
    "passed": True,
    "detail": "ROC: an unusable-confidence row is never predicted positive at any threshold; "
              "sweep: unusable rows retained at every cutoff, never removed",
})
assumption_rows.append({
    "id": "OUTLIER_FENCE_SOURCE", "question_id": "Q5", "model": np.nan, "prompt": np.nan,
    "description": "the 20.94 fence is computed once on the pooled image-level mean |e| distribution and applied unchanged to response-level rows for the ROC positive class -- never a per-configuration fence",
    "n_distinct_values": np.nan, "max_single_level_mass": np.nan, "degenerate": np.nan,
    "passed": True,
    "detail": f"fence={OUTLIER_FENCE:.4f}, n_hard_images={n_hard_images}, "
              f"response_level_flag_rate={response_level_flag_rate*100:.2f}%, "
              f"positives_per_cell=[{int(positives_per_cell.min())}, {int(positives_per_cell.max())}]",
})
assumption_rows.append({
    "id": "NO_MULTIPLICITY_CORRECTION", "question_id": "Q5", "model": np.nan, "prompt": np.nan,
    "description": "24-cell interval verdicts across every layer are not corrected for multiplicity -- disclosed once here",
    "n_distinct_values": np.nan, "max_single_level_mass": np.nan, "degenerate": np.nan,
    "passed": True,
    "detail": "this analysis performs no hypothesis test; every interval below is read one at a time",
})

# The three checks below cover the by-axis reading added alongside the
# per-cell tables above: that every level actually is a full view of the
# 1,155 images, that each axis's levels partition the 27,720 base/local
# responses exactly once (nothing double-counted, nothing dropped), and
# which CI method governed each of the by-axis rows, since a percentile
# fallback carries a weaker coverage guarantee than the primary BCa
# construction and a reader should be able to see how often it fired.
axis_coverage_ok = all(
    int(base_local.loc[base_local[axis] == level, "image"].nunique()) == 1155
    for axis, level in AXIS_LEVELS
)
assumption_rows.append({
    "id": "AXIS_LEVEL_COVERAGE", "question_id": "Q5", "model": np.nan, "prompt": np.nan,
    "description": "every model level and every prompt level covers all 1,155 images before the unusable-confidence exclusion",
    "n_distinct_values": np.nan, "max_single_level_mass": np.nan, "degenerate": np.nan,
    "passed": bool(axis_coverage_ok),
    "detail": ", ".join(
        f"{axis}={level}: n_images={int(base_local.loc[base_local[axis] == level, 'image'].nunique())}"
        for axis, level in AXIS_LEVELS
    ),
})

model_partition_ok = (
    base_local.groupby("image")["model"].apply(lambda s: sorted(s.unique()) == sorted(co.MODELS)).all()
    and len(base_local) == 27720
)
prompt_levels_sorted = sorted(base_local["prompt"].unique())
prompt_partition_ok = (
    base_local.groupby("image")["prompt"].apply(lambda s: sorted(s.unique()) == prompt_levels_sorted).all()
    and len(base_local) == 27720
)
assumption_rows.append({
    "id": "AXIS_PARTITION", "question_id": "Q5", "model": np.nan, "prompt": np.nan,
    "description": "the model axis and the prompt axis each partition the 27,720 base/local responses exactly once -- every response belongs to exactly one model level and exactly one prompt level",
    "n_distinct_values": np.nan, "max_single_level_mass": np.nan, "degenerate": np.nan,
    "passed": bool(model_partition_ok and prompt_partition_ok),
    "detail": f"n_responses={len(base_local)} (expect 27720), "
              f"model_axis_partition_ok={model_partition_ok}, prompt_axis_partition_ok={prompt_partition_ok}",
})

n_rank_percentile_fallback = int(
    (rank_association_by_axis_df["rho_pooled_ci_method"] != "BCa").sum()
    + (rank_association_by_axis_df["rho_0_20_bin_ci_method"] != "BCa").sum()
)
n_feature_percentile_fallback = int((feature_association_by_axis_df["ci_method"] != "BCa").sum())
assumption_rows.append({
    "id": "AXIS_CI_METHOD", "question_id": "Q5", "model": np.nan, "prompt": np.nan,
    "description": "CI method recorded per by-axis row (BCa primary, percentile on fallback); fallback count reported here rather than left implicit",
    "n_distinct_values": np.nan, "max_single_level_mass": np.nan, "degenerate": np.nan,
    "passed": True,
    "detail": f"rank_association_by_axis: {n_rank_percentile_fallback} of "
              f"{2 * len(rank_association_by_axis_df)} rho columns used a percentile fallback; "
              f"feature_association_by_axis: {n_feature_percentile_fallback} of "
              f"{len(feature_association_by_axis_df)} rows used a percentile fallback",
})

assumption_checks_df = pd.DataFrame(assumption_rows)
assumption_checks_df.head(10)

# %% [markdown]
# ## Result
#
# **Granularity comes first because it explains everything else.** Cells
# emit between 3 and 10 distinct confidence values, and one cell puts 93.2%
# of its responses on a single value — yet the worked case, Qwen-2.5 under
# *Grid-Overlay*, still reaches a pooled rho of -0.611 with 78.3% of its mass
# on one level. Coarse confidence bounds how fine an association *can* look;
# it does not prevent a real one from showing up.
#
# **Layer 1.** Confidence tracks error in most cells, and restricting to the
# sparsest cover bin — which removes most, though not all, of the
# image-to-image cover range — makes the association stronger rather than
# weaker everywhere it was already negative, which makes it implausible that
# the pooled result is an artefact of pooling across cover.
#
# **Layer 2.** A rank correlation alone hides what a triage rule costs in
# false positives — a strong rho can still demand flagging nearly every
# image before catching every bad estimate, when confidence is coarse enough
# that the worst and the ordinary images share a level. The ROC makes that
# cost visible directly, at every confidence level a cell actually emits.
#
# **Layer 3.** Confidence responds to two kinds of scene material — both
# forms of rooted dead look-alike material lower it by nearly the same
# amount, and non-rooted material lowers it only when spread as a continuous
# layer — while obliquity's small residual effect collapses once cover is
# held fixed, a reportable negative that narrows what "confidence tracks
# difficulty" is allowed to mean. None of this is claimed to hold independent
# of the model's own error, by design.
#
# **Layer 4.** The operational sweep shows what filtering buys and what it
# costs in the same table: overall MAE falls under filtering in the cells
# where Layer 1's association holds, and balanced MAE simply stops being
# computable once filtering empties a sparse high-cover bin — the sweep
# states that trap directly rather than arguing around it, and no image
# without usable confidence is ever dropped from either calculation.

# %%
granularity_df.to_csv(RESULTS_DIR / "Q5_confidence_granularity.csv", index=False)
rank_association_df.to_csv(RESULTS_DIR / "Q5_rank_association.csv", index=False)
roc_df.to_csv(RESULTS_DIR / "Q5_roc.csv", index=False)
feature_association_df.to_csv(RESULTS_DIR / "Q5_feature_association.csv", index=False)
sweep_df.to_csv(RESULTS_DIR / "Q5_confidence_filtering.csv", index=False)
assumption_checks_df.to_csv(RESULTS_DIR / "Q5_assumption_checks.csv", index=False)
rank_association_by_axis_df.to_csv(RESULTS_DIR / "Q5_rank_association_by_axis.csv", index=False)
feature_association_by_axis_df.to_csv(RESULTS_DIR / "Q5_feature_association_by_axis.csv", index=False)
roc_association_by_axis_df.to_csv(RESULTS_DIR / "Q5_roc_by_axis.csv", index=False)

# The bounds on what the unusable confidence values could have done are in
# `Q5_rank_association.csv`, as the `rho_mnar_bound_lo`/`_hi` and
# `mnar_bound_width` columns, beside the estimate each one bounds.

print("\nWritten:")
for fname in [
    "Q5_confidence_granularity.csv", "Q5_rank_association.csv",
    "Q5_roc.csv", "Q5_feature_association.csv", "Q5_confidence_filtering.csv",
    "Q5_assumption_checks.csv", "Q5_rank_association_by_axis.csv",
    "Q5_feature_association_by_axis.csv", "Q5_roc_by_axis.csv",
]:
    print(f"  05_deliverables/Repository/GitHub/results/{fname}")
