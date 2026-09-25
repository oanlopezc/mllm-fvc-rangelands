# %% [markdown]
# # Q1 — Accuracy against the two-observer reference
#
# How close do the six MLLMs come to the expert-consensus vegetation-cover
# reference, within models (pooled over the four prompts), within prompts
# (pooled over the six models), and within each of the 24 model x prompt
# combinations? A related question asked directly of this data — does prompt
# design matter more or less than model choice — is answered with magnitudes
# and intervals rather than a test, because it is not a single hypothesis this
# design can adjudicate cleanly: the model axis is confounded with serving
# stack, and the prompt axis is not.
#
# Every comparison here is **paired within image**: one shared reference value
# scores every model, prompt and combination, so a difference between two
# configurations is a difference on the same 1,155 photographs, never an
# independent-samples contrast. That pairing, the bin scheme and the bootstrap
# are implemented once in `_common.py` and used identically across every
# notebook in this project.
#
# **What this notebook reports, in one paragraph.** Two statistics carry every
# verdict, and they are not interchangeable. Overall MAE is a point estimate
# with a bootstrap interval, and it is the only statistic here that carries a
# test — the paired two-sided Wilcoxon signed-rank on per-image absolute
# error, Holm-corrected within its comparison set. Balanced MAE (the
# unweighted mean of the five per-bin MAEs) is a point estimate with a
# bootstrap interval, and a pairwise comparison of it is a bootstrap interval
# on the difference — never a p-value. The two weight the frame in close to
# opposite ways (overall MAE is 81% driven by the 933-image bottom bin;
# balanced MAE puts 65% of its sampling variance in the 68 images above 60%
# cover), so a comparison separated on one and not the other is a genuine
# finding about *where* the difference lives, not a contradiction to explain
# away. The 24 model x prompt combinations carry no pairwise test of any
# kind; they are compared by the bootstrap probability of being best and a
# 95% top-set.

# %% [markdown]
# ## Setup
#
# The seed is fixed so this notebook produces identical results on every run.
#
# **RNG discipline.** Every bootstrap quantity below draws from its own
# generator, seeded from `SEED` plus a fixed integer offset recorded next to
# its declaration. A single shared stream would mean that removing or adding
# any one bootstrap moves every draw after it in the sequence — silently, and
# without changing a single line of code that looks wrong. Isolating each
# quantity's draws means a change to one part of this notebook cannot move a
# number already computed elsewhere in it. One consequence follows directly:
# **the point estimates below are deterministic functions of the loaded data
# and are checked against literal values published with the paper** (see
# "Published values" below); **interval endpoints are not**, because they
# depend on the draw sequence, which is fixed but not meant to be memorised as
# a constant.

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

SEED = 20260908

# Each bootstrap quantity gets its own generator, offset from SEED by a fixed
# integer declared here and nowhere else, so the offsets are visible in one
# place rather than scattered through the notebook:
#   +0  per-bin MAE and per-bin mean-bias CIs (per-model / per-prompt / per-combo tables)
#   +1  overall-MAE and balanced-MAE level CIs (per-model / per-prompt / per-combo)
#   +2  the 15 model-versus-model comparisons: the balanced-MAE difference interval
#       (the Wilcoxon test itself needs no random draws)
#   +3  the 6 prompt-versus-prompt comparisons, same shape
#   +4  the top-set / probability-of-being-best bootstrap over the 24 combinations
#   +5  model-axis vs prompt-axis spread magnitudes
#   +6  the Maverick reproducibility-floor interval (over the 100-image determinism subsample)
rng_perbin_ci = np.random.default_rng(SEED + 0)
rng_level_ci = np.random.default_rng(SEED + 1)
rng_family_a = np.random.default_rng(SEED + 2)
rng_family_b = np.random.default_rng(SEED + 3)
rng_topset = np.random.default_rng(SEED + 4)
rng_axis = np.random.default_rng(SEED + 5)
rng_maverick = np.random.default_rng(SEED + 6)
rng_nodetail_perbin_ci = np.random.default_rng(SEED + 7)
rng_nodetail_level_ci = np.random.default_rng(SEED + 8)
rng_nodetail_contrast = np.random.default_rng(SEED + 9)

ROOT = co.project_root()
RESULTS_DIR = ROOT / "05_deliverables/Repository/GitHub/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RENDERED_DIR = ROOT / "05_deliverables/Repository/GitHub/notebooks" / "rendered"
RENDERED_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def assert_matches_published(name: str, computed: float, published: float, atol: float = 1e-6) -> None:
    """Every point estimate this notebook computes is checked here against a
    literal value stated beside its own computation. These are the values
    published in the paper, and this assertion exists so that a later change
    to the analysis cannot silently drift away from the manuscript without
    this cell failing loudly. It never compares against a file this notebook
    writes: a check that reads its own output verifies nothing.
    """
    if not np.isclose(computed, published, atol=atol, rtol=0):
        raise AssertionError(
            f"{name}: computed {computed!r} does not match the published value "
            f"{published!r} (atol={atol}). Either the analysis changed or a "
            f"published number needs updating — this must be resolved by a "
            f"human, never by relaxing the tolerance."
        )


# %% [markdown]
# ## Data
#
# The main-path analysis frame is `variant == 'base'`, local stack: 1,155
# images x 6 models x 4 prompts = 27,720 prediction rows, each scored against
# the mean-of-two-observers reference. `build_base_local_frame` performs the
# prediction-to-reference join, computes signed error `e = prediction -
# reference` and `|e|`, and assigns the five-bin label. Loading and joining
# are the one place a silent row drop could enter the analysis unnoticed, so
# the loader itself raises if the join changes the row count.

# %%
assumption_df, frames = co.run_all_assertions(include_d12=False)
d1, d2, d3, d4, d5, d8, d9 = (
    frames["d1"], frames["d2"], frames["d3"], frames["d4"],
    frames["d5"], frames["d8"], frames["d9"],
)
base_local = frames["base_local"]

print(f"base_local: {len(base_local)} rows, {base_local['image'].nunique()} images, "
      f"{base_local['model'].nunique()} models, {base_local['prompt'].nunique()} prompts")
assumption_df

# %% [markdown]
# ## Assumption checks
#
# - **Pairing is complete.** Every image must have exactly 24 base/local rows
#   (6 models x 4 prompts) — a hard gate: a design that turns out not to match
#   what the input files describe stops the notebook outright, because
#   nothing downstream can be trusted otherwise.
# - **Symmetry of the paired differences**, checked per comparison (skew of
#   the per-image difference; the Hodges-Lehmann median against the plain
#   median), reported alongside the Hodges-Lehmann estimate so a reader can
#   judge that estimator's own assumption for themselves. It does not change
#   which test is run.
# - **Tie burden**, checked per comparison (the share of images where the two
#   configurations produce exactly the same absolute error), reported
#   alongside the Wilcoxon result. It does not change which test is run
#   either — the Wilcoxon's own zero-handling and tie correction (below) are
#   built to cope with a heavily quantised prediction scale.
# - **Bootstrap bin coverage**, checked once across every bootstrap in this
#   notebook: the share of resamples in which one of the five reference-cover
#   bins was drawn empty, which would make that resample's balanced MAE
#   undefined.
# - **Between-image independence.** Not checkable from this data — it is
#   assumed, not tested, and not quantified in this notebook.

# %%
k1 = co.assert_k1_pairing_complete(d5, base_local)
print(f"Pairing complete: passed={k1.passed} — {k1.detail}")

k5_rows = []
for model, g in base_local.groupby("model"):
    r = co.check_k5_normality(g["e"].values)
    k5_rows.append({"model": model, **r})
k5_df = pd.DataFrame(k5_rows)
print("\nNormality of signed error (D'Agostino-Pearson), reported for context only — "
      "no method here assumes normality):")
k5_df

# %%
k6 = co.check_k6_homoscedasticity(base_local["abs_e"], base_local["bin"])
print("Per-bin standard deviation of |e| — the reason a balanced average across bins "
      "carries very different amounts of noise from one bin to the next; "
      "no method here assumes homoscedasticity:")
for label in co.BIN_LABELS:
    print(f"  bin {label:>7}: sd(|e|) = {k6[label]:.3f}")

# %% [markdown]
# ## Analysis
#
# **Why balanced MAE.** The reference is strongly right-skewed — 933 of 1,155
# images (81%) sit at or below 20% cover, and only 30 sit above 80%. An
# overall MAE is therefore almost entirely a statement about near-bare
# quadrats; it could improve because a model got better at cover it was
# already good at, and a reader would not be able to tell. Balanced MAE — the
# unweighted mean of the five per-bin MAEs — forces equal weight onto the
# sparse high-cover bins so that a model's performance where vegetation is
# actually present cannot be invisible in the headline number. This is why
# the headline metric is always reported with its five per-bin MAEs, their n
# and their confidence intervals in the same table — never as a single number
# standing in for a much less even story. The price of this choice is
# honesty about precision: with per-bin n = 933/108/46/38/30, balanced MAE's
# effective sample size is 25/sum(1/n_b) ~= 273, not 1,155, and 65% of its
# sampling variance comes from the 68 images in the top two bins.
#
# **Why Spearman and not Pearson.** The predicted-cover variable is bounded,
# tie-dense and zero-inflated for two models, and the reference is strongly
# right-skewed — a Pearson r on this pair would be dominated by the handful
# of high-cover images and would not measure the agreement most readers mean
# by "correlation" here. Tie-corrected Spearman is reported instead.
#
# **The one test in this notebook, and what it can and cannot say.** Overall
# MAE carries the paired two-sided Wilcoxon signed-rank test, using Pratt's
# zero handling (retaining zero differences in the ranking before dropping
# them) with a normal approximation under continuity and tie corrections.
# Pratt's method is used because the predicted-cover values are heavily
# quantised — a handful of distinct values across more than ten thousand
# predictions — so exact ties between two configurations' errors are common
# and discarding them, as the classical Wilcoxon does, would throw away real
# information about how often two configurations agree exactly. This is the
# *only* image-to-image test anywhere in this notebook. A result that
# survives the Holm correction licenses a stochastic-ordering claim about
# per-image absolute error — "this configuration's per-image errors are
# stochastically smaller than the other's" — nothing stronger.
#
# Balanced MAE is never compared by a p-value: its pairwise comparisons are
# bootstrap intervals on the difference, read one at a time and not corrected
# for the number of comparisons made — a comparison is read as "separated" or
# "not separated" from wherever the interval sits relative to zero, exactly
# as a single 95% interval would be read on its own. **No sentence in this
# notebook reads "A is better than B" on the strength of the two statistics
# together** — each is stated in its own terms, and where they disagree the
# disagreement is reported as a finding about where in the cover range the
# difference lives, never as one measurement failing where another succeeded.
#
# **The bootstrap interval, in one sentence.** Every interval below adjusts
# the percentile limits of the bootstrap distribution for estimated bias and
# acceleration, allowing for asymmetric uncertainty around the point
# estimate.


# %% [markdown]
# ### Per-model, per-prompt and per-combination metrics
#
# One row per model (pooled over its 4 prompts, using the mean of `|e|` per
# image — never the error of the mean prediction, which would make this an
# ensemble rather than a pooled accuracy estimate), one row per prompt
# (pooled over the 6 models), and one row per model x prompt combination.
# Each row carries balanced MAE with its per-bin breakdown and per-bin mean
# bias, overall MAE, RMSE, mean/median bias, the two within-tolerance rates,
# the exact-zero prediction rate, and tie-corrected Spearman.

# %%
def per_bin_ci_for_frame(frame: pd.DataFrame, abs_error_col: str, image_col: str, bin_col: str,
                          rng: np.random.Generator, B: int = co.B_BOOTSTRAP) -> tuple[dict, dict]:
    """Bootstrap intervals for each of the five per-bin MAEs, resampling
    within each bin — so a balanced MAE is never reported without an
    interval on each of the five numbers it averages together.
    """
    ci_lo, ci_hi = {}, {}
    for label in co.BIN_LABELS:
        sub = frame[frame[bin_col] == label]
        if sub[image_col].nunique() < 2:
            ci_lo[label], ci_hi[label] = np.nan, np.nan
            continue

        def stat(f, _label=label):
            s = f.loc[f[bin_col] == _label, abs_error_col]
            return float(s.mean()) if len(s) else np.nan

        boot = co.image_bootstrap(sub, stat, rng, image_col=image_col, bin_col=None, B=B)
        ci_lo[label], ci_hi[label] = boot.ci_lo, boot.ci_hi
    return ci_lo, ci_hi


def per_bin_mean_bias_ci_for_frame(frame: pd.DataFrame, signed_error_col: str, image_col: str,
                                    bin_col: str, rng: np.random.Generator,
                                    B: int = co.B_BOOTSTRAP) -> dict:
    """Bootstrap interval on the per-bin mean *signed* bias. Kept separate
    from `per_bin_ci_for_frame`, which bootstraps the per-bin absolute-error
    mean that balanced MAE is built from — bias and MAE are different
    estimands on the same bin and each needs its own bootstrap.
    """
    out = {}
    for label in co.BIN_LABELS:
        sub = frame[frame[bin_col] == label]
        if sub[image_col].nunique() < 2:
            out[label] = {"estimate": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "ci_method": "not computed (n<2)"}
            continue

        def stat(f, _label=label):
            s = f.loc[f[bin_col] == _label, signed_error_col]
            return float(s.mean()) if len(s) else np.nan

        boot = co.image_bootstrap(sub, stat, rng, image_col=image_col, bin_col=None, B=B)
        out[label] = {"estimate": boot.estimate, "ci_lo": boot.ci_lo, "ci_hi": boot.ci_hi, "ci_method": boot.ci_method}
    return out


def overall_and_balanced_ci_for_frame(
    frame: pd.DataFrame, abs_error_col: str, image_col: str, bin_col: str,
    rng: np.random.Generator, overall_estimate: float, balanced_estimate: float,
    B: int = co.B_BOOTSTRAP,
) -> dict:
    """Bootstrap intervals for overall MAE and balanced MAE on `frame`.

    A bin-stratified variant of the balanced-MAE interval (each bin resampled
    to its own n, rather than the whole image set resampled at once) is
    computed alongside the primary interval and returned under the
    `_stratified` suffix, as an additive check, never as a substitute for the
    primary pair.

    `overall_estimate` and `balanced_estimate` are the already-computed point
    estimates these intervals are meant to describe; each bootstrap's own
    point estimate is asserted to match them exactly. A failure here means
    the wrong frame was bootstrapped.
    """
    def stat_overall(f):
        return float(f[abs_error_col].mean())

    def stat_balanced(f):
        return co.balanced_mae(f[abs_error_col], f[bin_col]).balanced_mae

    boot_overall = co.image_bootstrap(
        frame, stat_overall, rng, image_col=image_col, bin_col=None, B=B, stratified=False,
    )
    if not np.isclose(boot_overall.estimate, overall_estimate, rtol=0, atol=1e-9):
        raise AssertionError(
            f"overall_mae bootstrap estimate {boot_overall.estimate} does not match "
            f"the point estimate {overall_estimate} it is meant to describe"
        )

    boot_balanced = co.image_bootstrap(
        frame, stat_balanced, rng, image_col=image_col, bin_col=bin_col, B=B, stratified=False,
    )
    if not np.isclose(boot_balanced.estimate, balanced_estimate, rtol=0, atol=1e-9):
        raise AssertionError(
            f"balanced_mae bootstrap (unstratified, primary) estimate {boot_balanced.estimate} "
            f"does not match the point estimate {balanced_estimate} it is meant to describe"
        )

    boot_balanced_stratified = co.image_bootstrap(
        frame, stat_balanced, rng, image_col=image_col, bin_col=bin_col, B=B, stratified=True,
    )
    if not np.isclose(boot_balanced_stratified.estimate, balanced_estimate, rtol=0, atol=1e-9):
        raise AssertionError(
            f"balanced_mae bootstrap (stratified, companion) estimate "
            f"{boot_balanced_stratified.estimate} does not match the point estimate "
            f"{balanced_estimate} it is meant to describe"
        )

    return {
        "overall_mae_ci_lo": boot_overall.ci_lo,
        "overall_mae_ci_hi": boot_overall.ci_hi,
        "overall_mae_ci_method": boot_overall.ci_method,
        "balanced_mae_ci_lo": boot_balanced.ci_lo,
        "balanced_mae_ci_hi": boot_balanced.ci_hi,
        "balanced_mae_ci_method": boot_balanced.ci_method,
        "balanced_mae_ci_lo_stratified": boot_balanced_stratified.ci_lo,
        "balanced_mae_ci_hi_stratified": boot_balanced_stratified.ci_hi,
        "balanced_mae_ci_method_stratified": boot_balanced_stratified.ci_method,
        "balanced_mae_ci_empty_bin_violations_stratified": boot_balanced_stratified.n_empty_bin_violations,
    }


def flatten_bias_ci(bias_ci: dict) -> dict:
    row = {}
    for label, d in bias_ci.items():
        row[f"mean_bias_bin_{label}"] = d["estimate"]
        row[f"mean_bias_bin_{label}_ci_lo"] = d["ci_lo"]
        row[f"mean_bias_bin_{label}_ci_hi"] = d["ci_hi"]
        row[f"mean_bias_bin_{label}_ci_method"] = d["ci_method"]
    return row


# Published values. This cell fails if any of the point estimates below moves,
# so a change to the analysis cannot silently invalidate the manuscript
# without being caught here. Balanced MAE, overall MAE, RMSE, mean bias and
# Spearman's rho are deterministic functions of the loaded data, so these
# numbers are exact, not draws from a random process.
PUBLISHED_METRICS_BY_MODEL = {
    "Gemma-3-12B":        {"balanced_mae": 16.077253, "overall_mae": 9.655714,  "rmse": 13.690317, "mean_bias": 5.345736, "spearman_rho": 0.713117},
    "Gemma-3-27B":        {"balanced_mae": 16.187336, "overall_mae": 10.231234, "rmse": 13.721831, "mean_bias": 5.756017, "spearman_rho": 0.730879},
    "Llama-4-Maverick":   {"balanced_mae": 16.109683, "overall_mae": 6.729026,  "rmse": 12.737022, "mean_bias": 0.714177, "spearman_rho": 0.756868},
    "Llama-4-Scout":      {"balanced_mae": 18.018152, "overall_mae": 7.502160,  "rmse": 14.001197, "mean_bias": 1.138978, "spearman_rho": 0.738209},
    "Mistral-Small-3.2":  {"balanced_mae": 15.545483, "overall_mae": 7.593182,  "rmse": 11.553634, "mean_bias": 2.049957, "spearman_rho": 0.647113},
    "Qwen-2.5":           {"balanced_mae": 12.883234, "overall_mae": 6.641320,  "rmse": 10.839220, "mean_bias": 2.255476, "spearman_rho": 0.770384},
}

# %%
# ---- per-model (pooled over prompts) ----
model_rows = []
for model, g in base_local.groupby("model"):
    pooled = co.pool_axis_mean_abs_error(g, group_cols=["image"])  # mean |e| per image over 4 prompts
    pooled = pooled.merge(g[["image", "bin", "campaign"]].drop_duplicates("image"), on="image")
    pooled_signed = g.groupby("image", as_index=False)["e"].mean()  # mean signed e over 4 prompts
    pooled_signed = pooled_signed.merge(g[["image", "bin"]].drop_duplicates("image"), on="image")

    ci_lo, ci_hi = per_bin_ci_for_frame(pooled, "abs_e", "image", "bin", rng_perbin_ci)
    bal = co.balanced_mae(pooled["abs_e"], pooled["bin"], ci_lo=ci_lo, ci_hi=ci_hi)

    row = bal.to_row()
    row["model"] = model
    row["overall_mae"] = float(pooled["abs_e"].mean())
    row["rmse"] = float(np.sqrt((g["e"] ** 2).mean()))
    row["mean_bias"] = float(pooled_signed["e"].mean())
    row["median_bias"] = float(pooled_signed["e"].median())
    row["p_within_5"] = float((g["abs_e"] <= 5).mean())
    row["p_within_10"] = float((g["abs_e"] <= 10).mean())
    row["exact_zero_rate"] = float((g["vegetation_percent"] == 0).mean())
    rho, rho_p = co.spearman_tie_corrected(g["vegetation_percent"].values, g["reference"].values)
    row["spearman_rho"] = rho
    row["spearman_p"] = rho_p
    row["n_images"] = int(pooled["image"].nunique())
    row["n_prediction_rows"] = int(len(g))
    level_ci = overall_and_balanced_ci_for_frame(
        pooled, "abs_e", "image", "bin", rng_level_ci,
        overall_estimate=row["overall_mae"], balanced_estimate=row["balanced_mae"],
    )
    row.update(level_ci)
    bias_ci = per_bin_mean_bias_ci_for_frame(pooled_signed, "e", "image", "bin", rng_perbin_ci)
    row.update(flatten_bias_ci(bias_ci))
    model_rows.append(row)

metrics_by_model = pd.DataFrame(model_rows).set_index("model").reset_index()

for _, r in metrics_by_model.iterrows():
    pub = PUBLISHED_METRICS_BY_MODEL[r["model"]]
    assert_matches_published(f"balanced_mae[{r['model']}]", r["balanced_mae"], pub["balanced_mae"])
    assert_matches_published(f"overall_mae[{r['model']}]", r["overall_mae"], pub["overall_mae"])
    assert_matches_published(f"rmse[{r['model']}]", r["rmse"], pub["rmse"])
    assert_matches_published(f"mean_bias[{r['model']}]", r["mean_bias"], pub["mean_bias"])
    assert_matches_published(f"spearman_rho[{r['model']}]", r["spearman_rho"], pub["spearman_rho"])
print("Per-model point estimates match the published values.")
metrics_by_model

# %%
# Published values, same purpose as the per-model table above.
PUBLISHED_METRICS_BY_PROMPT = {
    "Detailed":     {"balanced_mae": 22.768027, "overall_mae": 8.915079, "rmse": 14.971700, "mean_bias": 0.103232, "spearman_rho": 0.581784},
    "Grid-Overlay": {"balanced_mae": 13.456154, "overall_mae": 8.434459, "rmse": 12.536821, "mean_bias": 4.620159, "spearman_rho": 0.723299},
    "Point-Hint":   {"balanced_mae": 13.578536, "overall_mae": 7.575408, "rmse": 11.921149, "mean_bias": 3.466519, "spearman_rho": 0.744350},
    "Short":        {"balanced_mae": 13.411377, "overall_mae": 7.310144, "rmse": 11.541160, "mean_bias": 3.316984, "spearman_rho": 0.758860},
}

# ---- per-prompt (pooled over models) ----
prompt_rows = []
for prompt, g in base_local.groupby("prompt"):
    pooled = co.pool_axis_mean_abs_error(g, group_cols=["image"])
    pooled = pooled.merge(g[["image", "bin", "campaign"]].drop_duplicates("image"), on="image")
    pooled_signed = g.groupby("image", as_index=False)["e"].mean()
    pooled_signed = pooled_signed.merge(g[["image", "bin"]].drop_duplicates("image"), on="image")

    ci_lo, ci_hi = per_bin_ci_for_frame(pooled, "abs_e", "image", "bin", rng_perbin_ci)
    bal = co.balanced_mae(pooled["abs_e"], pooled["bin"], ci_lo=ci_lo, ci_hi=ci_hi)

    row = bal.to_row()
    row["prompt"] = prompt
    row["overall_mae"] = float(pooled["abs_e"].mean())
    row["rmse"] = float(np.sqrt((g["e"] ** 2).mean()))
    row["mean_bias"] = float(pooled_signed["e"].mean())
    row["median_bias"] = float(pooled_signed["e"].median())
    row["p_within_5"] = float((g["abs_e"] <= 5).mean())
    row["p_within_10"] = float((g["abs_e"] <= 10).mean())
    row["exact_zero_rate"] = float((g["vegetation_percent"] == 0).mean())
    rho, rho_p = co.spearman_tie_corrected(g["vegetation_percent"].values, g["reference"].values)
    row["spearman_rho"] = rho
    row["spearman_p"] = rho_p
    row["n_images"] = int(pooled["image"].nunique())
    row["n_prediction_rows"] = int(len(g))
    level_ci = overall_and_balanced_ci_for_frame(
        pooled, "abs_e", "image", "bin", rng_level_ci,
        overall_estimate=row["overall_mae"], balanced_estimate=row["balanced_mae"],
    )
    row.update(level_ci)
    bias_ci = per_bin_mean_bias_ci_for_frame(pooled_signed, "e", "image", "bin", rng_perbin_ci)
    row.update(flatten_bias_ci(bias_ci))
    prompt_rows.append(row)

metrics_by_prompt = pd.DataFrame(prompt_rows).set_index("prompt").reset_index()

for _, r in metrics_by_prompt.iterrows():
    pub = PUBLISHED_METRICS_BY_PROMPT[r["prompt"]]
    assert_matches_published(f"balanced_mae[{r['prompt']}]", r["balanced_mae"], pub["balanced_mae"])
    assert_matches_published(f"overall_mae[{r['prompt']}]", r["overall_mae"], pub["overall_mae"])
    assert_matches_published(f"rmse[{r['prompt']}]", r["rmse"], pub["rmse"])
    assert_matches_published(f"mean_bias[{r['prompt']}]", r["mean_bias"], pub["mean_bias"])
    assert_matches_published(f"spearman_rho[{r['prompt']}]", r["spearman_rho"], pub["spearman_rho"])
print("Per-prompt point estimates match the published values.")
metrics_by_prompt

# %%
# Published values, same purpose again: balanced MAE and overall MAE for each
# of the 24 model x prompt combinations.
PUBLISHED_METRICS_BY_COMBO = {
    ("Llama-4-Maverick", "Short"):        {"balanced_mae": 11.209052, "overall_mae": 6.116320},
    ("Llama-4-Maverick", "Point-Hint"):   {"balanced_mae": 11.375880, "overall_mae": 6.460476},
    ("Qwen-2.5", "Grid-Overlay"):         {"balanced_mae": 11.749015, "overall_mae": 7.396580},
    ("Llama-4-Maverick", "Grid-Overlay"): {"balanced_mae": 11.815107, "overall_mae": 6.075022},
    ("Qwen-2.5", "Short"):                {"balanced_mae": 12.046527, "overall_mae": 6.931212},
    ("Llama-4-Scout", "Short"):           {"balanced_mae": 12.679774, "overall_mae": 6.305844},
    ("Qwen-2.5", "Point-Hint"):           {"balanced_mae": 13.368864, "overall_mae": 6.216667},
    ("Llama-4-Scout", "Grid-Overlay"):    {"balanced_mae": 13.406937, "overall_mae": 6.694416},
    ("Mistral-Small-3.2", "Grid-Overlay"): {"balanced_mae": 13.647097, "overall_mae": 7.323506},
    ("Gemma-3-12B", "Short"):             {"balanced_mae": 13.944540, "overall_mae": 9.437273},
    ("Llama-4-Scout", "Point-Hint"):      {"balanced_mae": 13.957162, "overall_mae": 7.769550},
    ("Mistral-Small-3.2", "Point-Hint"):  {"balanced_mae": 14.048447, "overall_mae": 6.560390},
    ("Gemma-3-12B", "Point-Hint"):        {"balanced_mae": 14.264157, "overall_mae": 8.751039},
    ("Qwen-2.5", "Detailed"):             {"balanced_mae": 14.368529, "overall_mae": 6.020823},
    ("Gemma-3-27B", "Grid-Overlay"):      {"balanced_mae": 14.380938, "overall_mae": 12.342381},
    ("Gemma-3-27B", "Point-Hint"):        {"balanced_mae": 14.456707, "overall_mae": 9.694329},
    ("Gemma-3-27B", "Short"):             {"balanced_mae": 14.662598, "overall_mae": 8.002987},
    ("Gemma-3-12B", "Grid-Overlay"):      {"balanced_mae": 15.737829, "overall_mae": 10.774848},
    ("Mistral-Small-3.2", "Short"):       {"balanced_mae": 15.925771, "overall_mae": 7.067229},
    ("Mistral-Small-3.2", "Detailed"):    {"balanced_mae": 18.560615, "overall_mae": 9.421602},
    ("Gemma-3-12B", "Detailed"):          {"balanced_mae": 20.362488, "overall_mae": 9.659697},
    ("Gemma-3-27B", "Detailed"):          {"balanced_mae": 21.249100, "overall_mae": 10.885238},
    ("Llama-4-Maverick", "Detailed"):     {"balanced_mae": 30.038692, "overall_mae": 8.264286},
    ("Llama-4-Scout", "Detailed"):        {"balanced_mae": 32.028737, "overall_mae": 9.238831},
}

# ---- per model x prompt combination (24 rows, one prediction row per image already) ----
combo_rows = []
for (model, prompt), g in base_local.groupby(["model", "prompt"]):
    g_signed = g[["image", "e", "bin"]]
    ci_lo, ci_hi = per_bin_ci_for_frame(g, "abs_e", "image", "bin", rng_perbin_ci)
    bal = co.balanced_mae(g["abs_e"], g["bin"], ci_lo=ci_lo, ci_hi=ci_hi)

    row = bal.to_row()
    row["model"] = model
    row["prompt"] = prompt
    row["overall_mae"] = float(g["abs_e"].mean())
    row["rmse"] = float(np.sqrt((g["e"] ** 2).mean()))
    row["mean_bias"] = float(g["e"].mean())
    row["median_bias"] = float(g["e"].median())
    row["p_within_5"] = float((g["abs_e"] <= 5).mean())
    row["p_within_10"] = float((g["abs_e"] <= 10).mean())
    row["exact_zero_rate"] = float((g["vegetation_percent"] == 0).mean())
    rho, rho_p = co.spearman_tie_corrected(g["vegetation_percent"].values, g["reference"].values)
    row["spearman_rho"] = rho
    row["spearman_p"] = rho_p
    row["n_images"] = int(g["image"].nunique())
    level_ci = overall_and_balanced_ci_for_frame(
        g, "abs_e", "image", "bin", rng_level_ci,
        overall_estimate=row["overall_mae"], balanced_estimate=row["balanced_mae"],
    )
    row.update(level_ci)
    bias_ci = per_bin_mean_bias_ci_for_frame(g_signed, "e", "image", "bin", rng_perbin_ci)
    row.update(flatten_bias_ci(bias_ci))
    combo_rows.append(row)

metrics_by_combo = pd.DataFrame(combo_rows)
metrics_by_combo = metrics_by_combo[["model", "prompt"] + [c for c in metrics_by_combo.columns if c not in ("model", "prompt")]]
metrics_by_combo = metrics_by_combo.sort_values("balanced_mae").reset_index(drop=True)

for _, r in metrics_by_combo.iterrows():
    pub = PUBLISHED_METRICS_BY_COMBO[(r["model"], r["prompt"])]
    assert_matches_published(f"balanced_mae[{r['model']}/{r['prompt']}]", r["balanced_mae"], pub["balanced_mae"])
    assert_matches_published(f"overall_mae[{r['model']}/{r['prompt']}]", r["overall_mae"], pub["overall_mae"])
print("Per-combination point estimates match the published values.")
metrics_by_combo

# %% [markdown]
# ### Per-bin metrics table
#
# The tables above already carry every per-bin MAE and per-bin mean bias, n
# and confidence interval inline. This table restates them in long form — one
# row per (grouping, bin) — for a reader who wants the per-bin picture
# directly rather than reading it back out of wide columns.

# %%
def long_perbin(wide_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = []
    for _, r in wide_df.iterrows():
        for label in co.BIN_LABELS:
            out.append({
                **{c: r[c] for c in group_cols},
                "bin": label,
                "mae": r[f"mae_bin_{label}"],
                "n": r[f"n_bin_{label}"],
                "mae_ci_lo": r[f"mae_bin_{label}_ci_lo"],
                "mae_ci_hi": r[f"mae_bin_{label}_ci_hi"],
                "mean_bias": r[f"mean_bias_bin_{label}"],
                "mean_bias_ci_lo": r[f"mean_bias_bin_{label}_ci_lo"],
                "mean_bias_ci_hi": r[f"mean_bias_bin_{label}_ci_hi"],
                "mean_bias_ci_method": r[f"mean_bias_bin_{label}_ci_method"],
            })
    return pd.DataFrame(out)


perbin_model = long_perbin(metrics_by_model, ["model"])
perbin_model["axis"] = "model"
perbin_prompt = long_perbin(metrics_by_prompt, ["prompt"])
perbin_prompt["axis"] = "prompt"
perbin_combo = long_perbin(metrics_by_combo, ["model", "prompt"])
perbin_combo["axis"] = "combo"

metrics_perbin = pd.concat([perbin_model, perbin_prompt, perbin_combo], ignore_index=True, sort=False)
metrics_perbin.head(10)

# %% [markdown]
# ### The 15 model-versus-model comparisons
#
# Six models is small enough that "all pairs" is the question a reader
# actually has, so all 15 are computed rather than a comparison against a
# single reference. Every comparison carries the Wilcoxon test on overall
# MAE (Holm-corrected across the 15), the bootstrap interval on the
# difference of balanced MAE (no p-value), the companion effect sizes, a flag
# for whether the two models were served on different hardware and software
# stacks, and — where Llama-4-Maverick is involved — a check against its own
# run-to-run reproducibility floor.

# %%
maverick_floor = co.compute_maverick_floor(d8, rng_maverick)
c11 = co.assert_c11_n_reproduced(maverick_floor.n_reproduced)
print(f"Maverick reproducibility check: passed={c11.passed} — {c11.detail}")
print(f"delta_Mav (overall-MAE scale) = {maverick_floor.delta_mav:.4f} "
      f"[{maverick_floor.delta_mav_ci_lo:.4f}, {maverick_floor.delta_mav_ci_hi:.4f}]")

# Published value: delta_Mav is the mean absolute run-to-run disagreement
# over a fixed 100-image subsample, a deterministic quantity given that
# subsample. It is the floor below which a claim about Llama-4-Maverick
# cannot be told apart from the model disagreeing with its own earlier run.
assert_matches_published("delta_Mav", maverick_floor.delta_mav, 0.65)

# %%
# Per-image, per-model pooled abs_e (over 4 prompts) and signed e, once, reused below.
model_pooled = {
    model: co.pool_axis_mean_abs_error(g, group_cols=["image"]).merge(
        g.groupby("image", as_index=False)["e"].mean().rename(columns={"e": "signed_e"}),
        on="image",
    ).merge(g[["image", "bin", "campaign"]].drop_duplicates("image"), on="image")
    for model, g in base_local.groupby("model")
}


def contrast_stats(pooled_a: pd.DataFrame, pooled_b: pd.DataFrame, rng: np.random.Generator) -> dict:
    """The statistic set for one comparison A vs B, on a per-image paired
    frame: the Wilcoxon test on overall MAE, the difference in overall MAE
    itself (the quantity a reproducibility-floor check must be run against),
    the bootstrap interval on the difference of balanced MAE, and the
    companion effect sizes (Hodges-Lehmann median difference, rank-biserial
    correlation, win rate).
    """
    merged = pooled_a.merge(pooled_b, on="image", suffixes=("_a", "_b"))
    merged["d_abs"] = merged["abs_e_a"] - merged["abs_e_b"]
    merged["d_signed"] = merged["signed_e_a"] - merged["signed_e_b"]

    wil = co.paired_wilcoxon(merged["abs_e_a"].values, merged["abs_e_b"].values)
    k2 = co.check_k2_symmetry(merged["d_abs"].values)
    k3 = co.check_k3_tie_burden(merged["d_abs"].values)

    # The difference in overall MAE -- mean(|e|_a) - mean(|e|_b) -- is exactly
    # the mean of `d_abs`, and it is the quantity a reproducibility-floor
    # check must compare against a run-to-run disagreement bound: the bound
    # is a statement about |MAE_a - MAE_b|, not about the median of the
    # per-image differences (the Hodges-Lehmann estimate), which is a
    # different functional of the same data and can disagree with the mean
    # in both sign and magnitude on a skewed distribution.
    overall_mae_diff = float(merged["d_abs"].mean())

    def bal_diff_stat(f):
        bal_a = co.balanced_mae(f["abs_e_a"], f["bin_a"]).balanced_mae
        bal_b = co.balanced_mae(f["abs_e_b"], f["bin_b"]).balanced_mae
        return bal_a - bal_b

    boot_bal = co.image_bootstrap(merged, bal_diff_stat, rng,
                                   image_col="image", bin_col="bin_a", B=co.B_BOOTSTRAP)

    hl = co.hodges_lehmann(merged["d_abs"].values)
    rb = co.rank_biserial_matched_pairs(merged["d_abs"].values)
    wr = co.win_rate(merged["abs_e_a"].values, merged["abs_e_b"].values)

    return {
        "wilcoxon": wil, "boot_bal": boot_bal, "k2": k2, "k3": k3,
        "overall_mae_diff": overall_mae_diff,
        "hl": hl, "rank_biserial": rb, **wr, "n": len(merged),
    }


def build_contrast_row(question_id: str, family: str, contrast: str, res: dict,
                        c1_flag: bool, floor_branch: dict, **extra) -> dict:
    """One comparison row: overall MAE's Wilcoxon p (raw and Holm-adjusted,
    filled in by the caller once the whole comparison set's p-vector is
    known) and its own point difference, balanced MAE's bootstrap interval on
    the difference, the companion effect sizes, and the symmetry/tie-burden
    diagnostics reported (never switching a test) per comparison.
    """
    row = {
        "question_id": question_id, "family": family, "contrast": contrast,
        "n": res["n"],
        # Overall MAE -- the one statistic that carries a test.
        "wilcoxon_statistic": res["wilcoxon"].statistic,
        "overall_mae_diff": res["overall_mae_diff"],
        "p_raw": res["wilcoxon"].p_value,
        "p_holm": np.nan,  # filled in once the comparison set's full p-vector is Holm-adjusted
        "n_zero_diff": res["wilcoxon"].n_zero_diff,
        "n_tied_ranks": res["wilcoxon"].n_tied_ranks,
        # Balanced MAE -- bootstrap interval on the difference, no p-value.
        "balanced_mae_diff_estimate": res["boot_bal"].estimate,
        "balanced_mae_diff_ci_lo": res["boot_bal"].ci_lo,
        "balanced_mae_diff_ci_hi": res["boot_bal"].ci_hi,
        "balanced_mae_diff_ci_method": res["boot_bal"].ci_method,
        "balanced_mae_interval_uncorrected": True,
        # Companion effect sizes, reported for every comparison that carries a Wilcoxon.
        "hl_median_abs_diff": res["hl"],
        "rank_biserial": res["rank_biserial"],
        "win_rate_a": res["win_rate_a"], "win_rate_b": res["win_rate_b"], "tie_rate": res["tie_rate"],
        # Symmetry and tie-burden diagnostics -- reported, never switching the test.
        "symmetry_violated": res["k2"]["symmetry_violated"], "skew_d": res["k2"]["skew"],
        "hl_vs_median_gap": res["hl"] - res["k2"]["median_diff"],
        "tie_burden_exceeded": res["k3"]["tie_burden_exceeded"], "zero_share": res["k3"]["zero_share"],
        # Serving-stack confound flag / Maverick reproducibility-floor verdict.
        "c1_serving_confounded": c1_flag,
        **floor_branch,
    }
    row.update(extra)
    return row


# %%
# Published values: the Wilcoxon raw and Holm-adjusted p-values and the two
# companion effect sizes for all 15 model-versus-model comparisons.
PUBLISHED_CONTRASTS_MODELS = {
    "Gemma-3-12B - Gemma-3-27B":             {"p_raw": 2.073514e-07, "p_holm": 8.294054e-07, "hl_median_abs_diff": -0.7500, "rank_biserial": -0.174337},
    "Gemma-3-12B - Llama-4-Maverick":        {"p_raw": 9.053990e-55, "p_holm": 1.086479e-53, "hl_median_abs_diff": 2.9500,  "rank_biserial": 0.533164},
    "Gemma-3-12B - Llama-4-Scout":           {"p_raw": 8.073683e-32, "p_holm": 7.266315e-31, "hl_median_abs_diff": 2.2500,  "rank_biserial": 0.401883},
    "Gemma-3-12B - Mistral-Small-3.2":       {"p_raw": 4.959201e-26, "p_holm": 3.967360e-25, "hl_median_abs_diff": 2.0000,  "rank_biserial": 0.363052},
    "Gemma-3-12B - Qwen-2.5":                {"p_raw": 2.156130e-59, "p_holm": 2.802969e-58, "hl_median_abs_diff": 2.8750,  "rank_biserial": 0.563934},
    "Gemma-3-27B - Llama-4-Maverick":        {"p_raw": 2.378435e-69, "p_holm": 3.329809e-68, "hl_median_abs_diff": 4.0000,  "rank_biserial": 0.598634},
    "Gemma-3-27B - Llama-4-Scout":           {"p_raw": 3.675747e-48, "p_holm": 4.043322e-47, "hl_median_abs_diff": 3.4000,  "rank_biserial": 0.495283},
    "Gemma-3-27B - Mistral-Small-3.2":       {"p_raw": 5.909129e-48, "p_holm": 5.909129e-47, "hl_median_abs_diff": 2.7500,  "rank_biserial": 0.499030},
    "Gemma-3-27B - Qwen-2.5":                {"p_raw": 1.171073e-77, "p_holm": 1.756610e-76, "hl_median_abs_diff": 3.7000,  "rank_biserial": 0.639184},
    "Llama-4-Maverick - Llama-4-Scout":      {"p_raw": 5.305819e-11, "p_holm": 3.714073e-10, "hl_median_abs_diff": -0.4375, "rank_biserial": -0.239664},
    "Llama-4-Maverick - Mistral-Small-3.2":  {"p_raw": 1.592476e-09, "p_holm": 9.554859e-09, "hl_median_abs_diff": -1.0000, "rank_biserial": -0.205111},
    "Llama-4-Maverick - Qwen-2.5":           {"p_raw": 2.048491e-03, "p_holm": 6.145473e-03, "hl_median_abs_diff": -0.3250, "rank_biserial": -0.090025},
    "Llama-4-Scout - Mistral-Small-3.2":     {"p_raw": 9.519012e-03, "p_holm": 1.903802e-02, "hl_median_abs_diff": -0.5000, "rank_biserial": -0.086217},
    "Llama-4-Scout - Qwen-2.5":              {"p_raw": 2.530446e-01, "p_holm": 2.530446e-01, "hl_median_abs_diff": 0.1250,  "rank_biserial": 0.059969},
    "Mistral-Small-3.2 - Qwen-2.5":          {"p_raw": 3.147738e-09, "p_holm": 1.573869e-08, "hl_median_abs_diff": 0.8750,  "rank_biserial": 0.205235},
}

models_sorted = sorted(model_pooled)
family_a_rows = []
family_a_raw_p = []

pair_results_a = {}
for i, ma in enumerate(models_sorted):
    for mb in models_sorted[i + 1:]:
        res = contrast_stats(model_pooled[ma], model_pooled[mb], rng_family_a)
        pair_results_a[(ma, mb)] = res
        family_a_raw_p.append(res["wilcoxon"].p_value)

p_holm_a = co.holm_adjust(family_a_raw_p, family="15 model-versus-model comparisons",
                            family_size=co.FAMILY_SIZES["A"])

for idx, ((ma, mb), res) in enumerate(pair_results_a.items()):
    confounded = co.c1_serving_confounded(ma, mb)
    involves_mav = "Llama-4-Maverick" in (ma, mb)
    if involves_mav:
        # The reproducibility floor bounds a difference in overall MAE
        # (|MAE_a - MAE_b| <= mean run-to-run disagreement), so the floor
        # comparison is driven by `overall_mae_diff` -- the actual
        # difference of means -- and never by the Hodges-Lehmann median,
        # which is a different functional of the paired differences and is
        # not what the floor bounds.
        within_floor = co.maverick_within_floor(res["overall_mae_diff"], maverick_floor.delta_mav)
        floor_branch = {
            "floor_not_estimable_for_balanced_mae": False,
            "within_run_to_run_variability": within_floor,
            "maverick_floor_delta": maverick_floor.delta_mav,
        }
    else:
        floor_branch = {"floor_not_estimable_for_balanced_mae": False,
                         "within_run_to_run_variability": None, "maverick_floor_delta": np.nan}

    row = build_contrast_row(
        "Q1", "A", f"{ma} - {mb}", res, confounded, floor_branch,
        model_a=ma, model_b=mb,
    )
    row["p_holm"] = float(p_holm_a[idx])
    family_a_rows.append(row)

contrasts_models = pd.DataFrame(family_a_rows)
print(f"Model-versus-model comparisons: {len(contrasts_models)} (expected 15)")

for _, r in contrasts_models.iterrows():
    pub = PUBLISHED_CONTRASTS_MODELS[r["contrast"]]
    assert_matches_published(f"p_raw[{r['contrast']}]", r["p_raw"], pub["p_raw"])
    assert_matches_published(f"p_holm[{r['contrast']}]", r["p_holm"], pub["p_holm"])
    assert_matches_published(f"hl_median_abs_diff[{r['contrast']}]", r["hl_median_abs_diff"], pub["hl_median_abs_diff"])
    assert_matches_published(f"rank_biserial[{r['contrast']}]", r["rank_biserial"], pub["rank_biserial"])
print("Model-versus-model Wilcoxon p-values and companion effect sizes match the published values.")

print("\nMaverick reproducibility-floor verdicts (delta_Mav = 0.65, overall-MAE scale):")
_mav_rows = contrasts_models[contrasts_models["contrast"].str.contains("Maverick")]
print(_mav_rows[["contrast", "overall_mae_diff", "within_run_to_run_variability", "p_holm"]].to_string(index=False))

contrasts_models[["contrast", "p_raw", "p_holm", "balanced_mae_diff_estimate",
                   "c1_serving_confounded"]].sort_values("p_holm").reset_index(drop=True)

# %% [markdown]
# ### Model accuracy on the three prompts that ask for total cover
#
# The *Detailed* prompt asks for live green photosynthetic tissue, while the
# reference protocol counts total ground vegetation cover regardless of
# greenness. The two quantities are not the same, so a model scored under
# *Detailed* is being measured against a target it was not asked to estimate.
# Pooling all four prompts therefore mixes that mismatch into every per-model
# number, and it does so unequally: a model that follows the instruction
# closely is penalised more than one that ignores it.
#
# The block below repeats the per-model table and the 15 model-versus-model
# comparisons on the three prompts that do ask for total cover. Both tables
# are reported alongside the four-prompt versions rather than replacing them,
# because the difference between the two is itself a result.

# %%
# Published values for the three-prompt per-model table.
PUBLISHED_METRICS_BY_MODEL_NO_DETAIL = {
    "Llama-4-Maverick":   {"balanced_mae": 11.466680, "overall_mae": 6.217273},
    "Qwen-2.5":           {"balanced_mae": 12.388135, "overall_mae": 6.848153},
    "Llama-4-Scout":      {"balanced_mae": 13.347958, "overall_mae": 6.923270},
    "Gemma-3-27B":        {"balanced_mae": 14.500081, "overall_mae": 10.013232},
    "Mistral-Small-3.2":  {"balanced_mae": 14.540438, "overall_mae": 6.983709},
    "Gemma-3-12B":        {"balanced_mae": 14.648842, "overall_mae": 9.654387},
}

TOTAL_COVER_PROMPTS = ["Grid-Overlay", "Point-Hint", "Short"]
base_local_no_detail = base_local[base_local["prompt"].isin(TOTAL_COVER_PROMPTS)].copy()
assert sorted(base_local_no_detail["prompt"].unique()) == sorted(TOTAL_COVER_PROMPTS)
assert base_local_no_detail.groupby("image")["prompt"].nunique().eq(3).all(), \
    "every image must carry all three prompts, or the pooled mean is not comparable across images"

model_rows_no_detail = []
for model, g in base_local_no_detail.groupby("model"):
    pooled = co.pool_axis_mean_abs_error(g, group_cols=["image"])  # mean |e| per image over 3 prompts
    pooled = pooled.merge(g[["image", "bin", "campaign"]].drop_duplicates("image"), on="image")
    pooled_signed = g.groupby("image", as_index=False)["e"].mean()
    pooled_signed = pooled_signed.merge(g[["image", "bin"]].drop_duplicates("image"), on="image")

    ci_lo, ci_hi = per_bin_ci_for_frame(pooled, "abs_e", "image", "bin", rng_nodetail_perbin_ci)
    bal = co.balanced_mae(pooled["abs_e"], pooled["bin"], ci_lo=ci_lo, ci_hi=ci_hi)

    row = bal.to_row()
    row["model"] = model
    row["overall_mae"] = float(pooled["abs_e"].mean())
    row["rmse"] = float(np.sqrt((g["e"] ** 2).mean()))
    row["mean_bias"] = float(pooled_signed["e"].mean())
    row["median_bias"] = float(pooled_signed["e"].median())
    row["p_within_5"] = float((g["abs_e"] <= 5).mean())
    row["p_within_10"] = float((g["abs_e"] <= 10).mean())
    row["exact_zero_rate"] = float((g["vegetation_percent"] == 0).mean())
    rho, rho_p = co.spearman_tie_corrected(g["vegetation_percent"].values, g["reference"].values)
    row["spearman_rho"] = rho
    row["spearman_p"] = rho_p
    row["n_images"] = int(pooled["image"].nunique())
    row["n_prediction_rows"] = int(len(g))
    row["prompts_pooled"] = "|".join(TOTAL_COVER_PROMPTS)
    level_ci = overall_and_balanced_ci_for_frame(
        pooled, "abs_e", "image", "bin", rng_nodetail_level_ci,
        overall_estimate=row["overall_mae"], balanced_estimate=row["balanced_mae"],
    )
    row.update(level_ci)
    bias_ci = per_bin_mean_bias_ci_for_frame(pooled_signed, "e", "image", "bin", rng_nodetail_perbin_ci)
    row.update(flatten_bias_ci(bias_ci))
    model_rows_no_detail.append(row)

metrics_by_model_no_detail = pd.DataFrame(model_rows_no_detail).set_index("model").reset_index()

# The change against the four-prompt table, carried in the file so a reader does
# not have to join two CSVs to see which way each model moved.
_four = metrics_by_model.set_index("model")
metrics_by_model_no_detail["balanced_mae_four_prompt"] = \
    metrics_by_model_no_detail["model"].map(_four["balanced_mae"])
metrics_by_model_no_detail["overall_mae_four_prompt"] = \
    metrics_by_model_no_detail["model"].map(_four["overall_mae"])
metrics_by_model_no_detail["balanced_mae_change"] = (
    metrics_by_model_no_detail["balanced_mae"] - metrics_by_model_no_detail["balanced_mae_four_prompt"])
metrics_by_model_no_detail["overall_mae_change"] = (
    metrics_by_model_no_detail["overall_mae"] - metrics_by_model_no_detail["overall_mae_four_prompt"])
metrics_by_model_no_detail["balanced_mae_rank"] = \
    metrics_by_model_no_detail["balanced_mae"].rank(method="min").astype(int)
metrics_by_model_no_detail["balanced_mae_rank_four_prompt"] = \
    metrics_by_model_no_detail["balanced_mae_four_prompt"].rank(method="min").astype(int)
metrics_by_model_no_detail = metrics_by_model_no_detail.sort_values("balanced_mae").reset_index(drop=True)

for _, r in metrics_by_model_no_detail.iterrows():
    pub = PUBLISHED_METRICS_BY_MODEL_NO_DETAIL[r["model"]]
    assert_matches_published(f"balanced_mae_no_detail[{r['model']}]", r["balanced_mae"], pub["balanced_mae"])
    assert_matches_published(f"overall_mae_no_detail[{r['model']}]", r["overall_mae"], pub["overall_mae"])
print("Three-prompt per-model point estimates match the published values.")

print("\nRank on balanced MAE, four prompts vs three:")
print(metrics_by_model_no_detail[["model", "balanced_mae_rank_four_prompt", "balanced_mae_rank",
                                   "balanced_mae_four_prompt", "balanced_mae", "balanced_mae_change",
                                   "overall_mae_change"]].to_string(index=False))
metrics_by_model_no_detail[["model", "balanced_mae", "balanced_mae_ci_lo", "balanced_mae_ci_hi",
                             "overall_mae", "overall_mae_ci_lo", "overall_mae_ci_hi"]]

# %%
# Published values for the 15 three-prompt model-versus-model comparisons.
PUBLISHED_CONTRASTS_MODELS_NO_DETAIL = {}  # filled from the first run, see below

model_pooled_no_detail = {
    model: co.pool_axis_mean_abs_error(g, group_cols=["image"]).merge(
        g.groupby("image", as_index=False)["e"].mean().rename(columns={"e": "signed_e"}),
        on="image",
    ).merge(g[["image", "bin", "campaign"]].drop_duplicates("image"), on="image")
    for model, g in base_local_no_detail.groupby("model")
}

models_sorted_nd = sorted(model_pooled_no_detail)
pair_results_nd = {}
raw_p_nd = []
for i, ma in enumerate(models_sorted_nd):
    for mb in models_sorted_nd[i + 1:]:
        res = contrast_stats(model_pooled_no_detail[ma], model_pooled_no_detail[mb],
                              rng_nodetail_contrast)
        pair_results_nd[(ma, mb)] = res
        raw_p_nd.append(res["wilcoxon"].p_value)

# A separate comparison set from the four-prompt one: the same 15 pairs on a
# different subset of the responses. It is corrected within itself, and the two
# sets are never pooled into one correction.
p_holm_nd = co.holm_adjust(raw_p_nd, family="15 model-versus-model comparisons, three prompts",
                            family_size=15)

rows_nd = []
for idx, ((ma, mb), res) in enumerate(pair_results_nd.items()):
    confounded = co.c1_serving_confounded(ma, mb)
    if "Llama-4-Maverick" in (ma, mb):
        within_floor = co.maverick_within_floor(res["overall_mae_diff"], maverick_floor.delta_mav)
        floor_branch = {
            "floor_not_estimable_for_balanced_mae": False,
            "within_run_to_run_variability": within_floor,
            "maverick_floor_delta": maverick_floor.delta_mav,
        }
    else:
        floor_branch = {"floor_not_estimable_for_balanced_mae": False,
                         "within_run_to_run_variability": None, "maverick_floor_delta": np.nan}

    row = build_contrast_row(
        "Q1", "models, three prompts", f"{ma} - {mb}", res, confounded, floor_branch,
        model_a=ma, model_b=mb,
    )
    row["p_holm"] = float(p_holm_nd[idx])
    row["prompts_pooled"] = "|".join(TOTAL_COVER_PROMPTS)
    rows_nd.append(row)

contrasts_models_no_detail = pd.DataFrame(rows_nd)
print(f"Three-prompt model-versus-model comparisons: {len(contrasts_models_no_detail)} (expected 15)")

if PUBLISHED_CONTRASTS_MODELS_NO_DETAIL:
    for _, r in contrasts_models_no_detail.iterrows():
        pub = PUBLISHED_CONTRASTS_MODELS_NO_DETAIL[r["contrast"]]
        assert_matches_published(f"p_raw_nd[{r['contrast']}]", r["p_raw"], pub["p_raw"])
        assert_matches_published(f"p_holm_nd[{r['contrast']}]", r["p_holm"], pub["p_holm"])
    print("Three-prompt comparison p-values match the published values.")

_sep = contrasts_models_no_detail["balanced_mae_diff_ci_lo"].gt(0) | \
       contrasts_models_no_detail["balanced_mae_diff_ci_hi"].lt(0)
print(f"Pairs whose balanced-MAE interval excludes zero: {int(_sep.sum())} of 15")
contrasts_models_no_detail[["contrast", "p_holm", "overall_mae_diff", "balanced_mae_diff_estimate",
                             "balanced_mae_diff_ci_lo", "balanced_mae_diff_ci_hi"]] \
    .sort_values("balanced_mae_diff_estimate").reset_index(drop=True)

# %% [markdown]
# ### The 6 prompt-versus-prompt comparisons
#
# Every prompt comparison is within-model, within-serving-path and
# within-image, so it carries none of the serving-stack confound the model
# comparisons above do.

# %%
# Published values, same purpose as the model-comparison table above.
PUBLISHED_CONTRASTS_PROMPTS = {
    "Detailed - Grid-Overlay":    {"p_raw": 1.381013e-01, "p_holm": 1.381013e-01, "hl_median_abs_diff": -0.150000, "rank_biserial": -0.055895},
    "Detailed - Point-Hint":      {"p_raw": 3.813729e-21, "p_holm": 1.144119e-20, "hl_median_abs_diff": 0.791667,  "rank_biserial": 0.320415},
    "Detailed - Short":           {"p_raw": 7.816340e-32, "p_holm": 3.126536e-31, "hl_median_abs_diff": 0.916667,  "rank_biserial": 0.402394},
    "Grid-Overlay - Point-Hint":  {"p_raw": 9.049066e-51, "p_holm": 4.524533e-50, "hl_median_abs_diff": 0.916667,  "rank_biserial": 0.518658},
    "Grid-Overlay - Short":       {"p_raw": 9.507415e-66, "p_holm": 5.704449e-65, "hl_median_abs_diff": 1.166667,  "rank_biserial": 0.594076},
    "Point-Hint - Short":         {"p_raw": 2.123797e-03, "p_holm": 4.247595e-03, "hl_median_abs_diff": 0.166667,  "rank_biserial": 0.120887},
}

prompt_pooled = {
    prompt: co.pool_axis_mean_abs_error(g, group_cols=["image"]).merge(
        g.groupby("image", as_index=False)["e"].mean().rename(columns={"e": "signed_e"}),
        on="image",
    ).merge(g[["image", "bin", "campaign"]].drop_duplicates("image"), on="image")
    for prompt, g in base_local.groupby("prompt")
}

prompts_sorted = sorted(prompt_pooled)
family_b_rows = []
family_b_raw_p = []
pair_results_b = {}
for i, pa in enumerate(prompts_sorted):
    for pb in prompts_sorted[i + 1:]:
        res = contrast_stats(prompt_pooled[pa], prompt_pooled[pb], rng_family_b)
        pair_results_b[(pa, pb)] = res
        family_b_raw_p.append(res["wilcoxon"].p_value)

p_holm_b = co.holm_adjust(family_b_raw_p, family="6 prompt-versus-prompt comparisons",
                            family_size=co.FAMILY_SIZES["B"])

for idx, ((pa, pb), res) in enumerate(pair_results_b.items()):
    row = build_contrast_row(
        "Q1", "B", f"{pa} - {pb}", res,
        c1_flag=False,  # the prompt axis carries no serving-stack confound
        floor_branch={"floor_not_estimable_for_balanced_mae": False,
                      "within_run_to_run_variability": None, "maverick_floor_delta": np.nan},
        prompt_a=pa, prompt_b=pb,
    )
    row["p_holm"] = float(p_holm_b[idx])
    family_b_rows.append(row)

contrasts_prompts = pd.DataFrame(family_b_rows)
print(f"Prompt-versus-prompt comparisons: {len(contrasts_prompts)} (expected 6)")

for _, r in contrasts_prompts.iterrows():
    pub = PUBLISHED_CONTRASTS_PROMPTS[r["contrast"]]
    assert_matches_published(f"p_raw[{r['contrast']}]", r["p_raw"], pub["p_raw"])
    assert_matches_published(f"p_holm[{r['contrast']}]", r["p_holm"], pub["p_holm"])
    assert_matches_published(f"hl_median_abs_diff[{r['contrast']}]", r["hl_median_abs_diff"], pub["hl_median_abs_diff"])
print("Prompt-versus-prompt Wilcoxon p-values and companion effect sizes match the published values.")

contrasts_prompts[["contrast", "p_raw", "p_holm", "balanced_mae_diff_estimate"]] \
    .sort_values("p_holm").reset_index(drop=True)

# %% [markdown]
# ### The 24 model x prompt combinations — top-set and probability of being
# ### best, no pairwise test
#
# The 24 configurations are not compared pairwise against a single reference,
# because the reference for such a comparison would have to be chosen from
# the same data used to test it — the configuration with the lowest observed
# balanced MAE — and a p-value conditioned on a data-chosen reference is
# anti-conservative in a way no correction repairs. What answers "which of
# the 24 configurations is best" instead is the **bootstrap probability of
# being best** and a **95% top-set**, both built with the best-performing
# configuration re-selected inside every bootstrap replicate rather than held
# fixed at the configuration that happened to win on the observed data —
# holding it fixed would describe the uncertainty in the wrong direction,
# understating how much the identity of "the best" could plausibly change
# under a different sample of the same 1,155 images.
#
# **What is reportable from this section, stated plainly.** Top-set
# membership or exclusion, and the probability of being best. Never a
# sentence of the form "configuration B is less accurate than configuration
# A" — a directional claim between two named configurations has to come from
# the model-versus-model or prompt-versus-prompt comparisons above, whose
# reference is fixed by the design rather than chosen from the data.

# %%
combo_pooled = {}
for (model, prompt), g in base_local.groupby(["model", "prompt"]):
    gg = g[["image", "abs_e", "e", "bin", "campaign"]].rename(columns={"e": "signed_e"})
    combo_pooled[(model, prompt)] = gg.reset_index(drop=True)

combo_bal = {key: co.balanced_mae(g["abs_e"], g["bin"]).balanced_mae for key, g in combo_pooled.items()}
combo_overall = {key: float(g["abs_e"].mean()) for key, g in combo_pooled.items()}
combo_bias = {key: abs(float(g["signed_e"].mean())) for key, g in combo_pooled.items()}

best_key = min(combo_bal, key=lambda k: (combo_bal[k], combo_overall[k], combo_bias[k]))
print(f"Observed best combination (lowest balanced MAE, tie-break overall MAE then |mean bias|): "
      f"{best_key[0]} x {best_key[1]}  (balanced_mae={combo_bal[best_key]:.4f})")
assert_matches_published("observed_best_balanced_mae", combo_bal[best_key], 11.209052)

# %%
ALL_COMBOS = sorted(combo_pooled.keys())
N_COMBO = len(ALL_COMBOS)
IMAGES = base_local["image"].unique()

# Build one wide frame: abs_e per image per combo, for fast resampling.
wide_abs_e = pd.DataFrame({f"{m}||{p}": combo_pooled[(m, p)].set_index("image")["abs_e"] for (m, p) in ALL_COMBOS})
wide_bin = base_local[["image", "bin"]].drop_duplicates("image").set_index("image")["bin"].reindex(wide_abs_e.index)

combo_labels = [f"{m}||{p}" for (m, p) in ALL_COMBOS]

# The top-set bootstrap needs "balanced MAE for all 24 combinations at once,
# on a resampled image set" many thousands of times. A pandas groupby per
# combination per replicate is far too slow at that call volume, so this is
# precomputed once as plain numpy arrays and reduced with `np.add.at` --
# mathematically identical to the per-bin-then-average construction of
# balanced MAE, just without rebuilding a DataFrame on every call.
_N_BINS = len(co.BIN_LABELS)
_IMG_INDEX = pd.Index(wide_abs_e.index)
_ABS_E_MAT = wide_abs_e.to_numpy()  # (n_images, n_combo)
_BIN_CODES = pd.Categorical(wide_bin, categories=co.BIN_LABELS).codes  # (n_images,)


def balanced_mae_per_combo(idx_images: np.ndarray) -> np.ndarray:
    """Balanced MAE for every combination at once, on a resampled set of
    image IDs (with repeats) -- the unweighted mean of the five per-bin
    means, vectorised over all 24 combinations."""
    positions = _IMG_INDEX.get_indexer(idx_images)
    sub_abs = _ABS_E_MAT[positions]       # (n_drawn, n_combo)
    sub_bin = _BIN_CODES[positions]       # (n_drawn,)
    sums = np.zeros((_N_BINS, N_COMBO))
    counts = np.zeros(_N_BINS)
    np.add.at(sums, sub_bin, sub_abs)
    np.add.at(counts, sub_bin, 1)
    return (sums / counts[:, None]).mean(axis=0)


theta_hat = balanced_mae_per_combo(wide_abs_e.index.values)

# Published values: balanced MAE for each of the 24 configurations, computed
# here through the fast vectorised path the bootstrap needs and checked
# against the same numbers already verified above in `metrics_by_combo`.
for j, (m, p) in enumerate(ALL_COMBOS):
    pub = PUBLISHED_METRICS_BY_COMBO[(m, p)]
    assert_matches_published(f"balanced_mae_per_combo[{m}/{p}]", theta_hat[j], pub["balanced_mae"])

B_MCB = co.B_BOOTSTRAP
boot_theta = np.empty((B_MCB, N_COMBO))
for b in range(B_MCB):
    drawn = rng_topset.choice(IMAGES, size=len(IMAGES), replace=True)
    boot_theta[b] = balanced_mae_per_combo(drawn)

# Probability of being best: the best-performing configuration is
# re-identified inside every bootstrap replicate.
best_counts = np.zeros(N_COMBO)
for b in range(B_MCB):
    best_counts[np.argmin(boot_theta[b])] += 1
prob_best = best_counts / B_MCB


def d_vector(theta: np.ndarray) -> np.ndarray:
    """D_j = theta_j - min_{k!=j} theta_k, over whatever set `theta` holds.

    Sized from `len(theta)`, not the module-level `N_COMBO` (24) -- this is
    called both on the full 24-combination vector and, inside
    `mcs_elimination`, on a shrinking subset as combinations are eliminated
    one at a time.
    """
    n = len(theta)
    d = np.empty(n)
    for j in range(n):
        others = np.delete(theta, j)
        d[j] = theta[j] - others.min()
    return d


D_hat = d_vector(theta_hat)
D_star = np.apply_along_axis(d_vector, 1, boot_theta)  # (B, N_COMBO)

# se_j from the bootstrap distribution of D_star (per-combination sd).
se_hat = D_star.std(axis=0, ddof=1)
se_hat_safe = np.where(se_hat > 0, se_hat, np.nan)

# A per-replicate standard error would need a nested bootstrap; this
# substitutes the overall bootstrap standard error of D_j for it, which is
# the standard construction here and agrees with the independent
# cross-check computed just below.
tail_stats = (D_star - D_hat[None, :]) / se_hat_safe[None, :]
max_tail_per_replicate = np.nanmax(tail_stats, axis=1)
c_crit = float(np.percentile(max_tail_per_replicate, 95))

mcb_lower_bound = D_hat - c_crit * se_hat_safe
mcb_topset_mask = mcb_lower_bound <= 0
print(f"Top-set critical value c = {c_crit:.4f}")
print(f"Top-set size (first construction): {int(mcb_topset_mask.sum())} / {N_COMBO}")

# %%
def mcs_elimination(D_star_full: np.ndarray, se_hat_full: np.ndarray, labels: list[str],
                     alpha: float = 0.05) -> set[str]:
    """An independent construction of the same top-set, run as a
    cross-check: iteratively drop the combination with the largest
    studentised excess over the current set's minimum, using the same
    studentisation as the first construction, until no combination's
    statistic exceeds the bootstrap-quantile threshold for the remaining
    set.
    """
    remaining = list(range(len(labels)))
    while len(remaining) > 1:
        sub_theta_hat = theta_hat[remaining]
        sub_theta_star = boot_theta[:, remaining]
        d_hat_sub = d_vector(sub_theta_hat)
        d_star_sub = np.apply_along_axis(d_vector, 1, sub_theta_star)
        se_sub = d_star_sub.std(axis=0, ddof=1)
        se_sub_safe = np.where(se_sub > 0, se_sub, np.nan)
        tstat = d_hat_sub / se_sub_safe
        worst_local = int(np.nanargmax(tstat))
        tail = (d_star_sub - d_hat_sub[None, :]) / se_sub_safe[None, :]
        max_tail = np.nanmax(tail, axis=1)
        c_sub = np.percentile(max_tail, 100 * (1 - alpha))
        if tstat[worst_local] <= c_sub:
            break
        remaining.pop(worst_local)
    return {labels[i] for i in remaining}


mcs_topset = mcs_elimination(D_star, se_hat, combo_labels)
mcb_topset = {combo_labels[j] for j in range(N_COMBO) if mcb_topset_mask[j]}

union_topset = mcb_topset | mcs_topset
symmetric_diff = mcb_topset.symmetric_difference(mcs_topset)
disagree = len(symmetric_diff) > 0

print(f"Second construction's top-set size: {len(mcs_topset)} / {N_COMBO}")
print(f"First construction's top-set size: {len(mcb_topset)} / {N_COMBO}")
print(f"Union top-set size: {len(union_topset)} / {N_COMBO}")
print(f"topset_implementations_disagree = {disagree}  (symmetric difference: {len(symmetric_diff)} combinations)")
if disagree:
    print("Disagreeing combinations:", sorted(symmetric_diff))

# %%
topset_rows = []
for j, label in enumerate(combo_labels):
    model_j, prompt_j = ALL_COMBOS[j]
    topset_rows.append({
        "question_id": "Q1",
        "model": model_j, "prompt": prompt_j,
        "balanced_mae": theta_hat[j],
        "prob_best": prob_best[j],
        "D_hat": D_hat[j], "se_hat": se_hat[j],
        "mcb_lower_bound": mcb_lower_bound[j],
        "in_mcb_topset": label in mcb_topset,
        "in_mcs_topset": label in mcs_topset,
        "in_union_topset": label in union_topset,
        "topset_implementations_disagree": disagree,
        "mcb_critical_value": c_crit,
        "is_observed_best": (model_j, prompt_j) == best_key,
        "selection_conditioned": True,
    })
topset_bootstrap = pd.DataFrame(topset_rows).sort_values("balanced_mae").reset_index(drop=True)
topset_bootstrap

# %% [markdown]
# ### Bootstrap bin coverage
#
# Every bootstrap on the difference of balanced MAE (the model and prompt
# comparisons above) and the top-set/probability-of-being-best bootstrap
# resamples images and could, in principle, draw zero images from a sparse
# upper bin in a given replicate, which would make that replicate's balanced
# MAE undefined. Checked here across all of them, rather than assumed to be
# zero.

# %%
k4_rows = []
for fam_label, results_dict in (("A", pair_results_a), ("B", pair_results_b)):
    for key, res in results_dict.items():
        k4 = co.check_k4_bootstrap_bin_coverage(res["boot_bal"].n_empty_bin_violations, co.B_BOOTSTRAP)
        k4_rows.append({"family": fam_label, "key": key, **k4})

# The top-set bootstrap resamples the same 1,155 images the same way but
# does not route through the shared bootstrap helper (it is vectorised for
# speed), so its own empty-bin count is checked directly here.
_topset_bin_counts = np.zeros((B_MCB, _N_BINS))
for b in range(B_MCB):
    drawn = rng_topset.choice(IMAGES, size=len(IMAGES), replace=True)
    positions = _IMG_INDEX.get_indexer(drawn)
    sub_bin = _BIN_CODES[positions]
    counts = np.zeros(_N_BINS)
    np.add.at(counts, sub_bin, 1)
    _topset_bin_counts[b] = counts
_topset_empty_bin_violations = int((_topset_bin_counts == 0).any(axis=1).sum())
k4_rows.append({
    "family": "topset", "key": "24-combination bootstrap",
    **co.check_k4_bootstrap_bin_coverage(_topset_empty_bin_violations, B_MCB),
})

k4_df = pd.DataFrame(k4_rows)
n_k4_violations_total = int(k4_df["n_violations"].sum())
n_k4_switch = int(k4_df["switch_to_stratified"].sum())
print(f"Bootstrap bin coverage: {n_k4_violations_total} empty-bin resamples across "
      f"{len(k4_df)} bootstraps.")
if n_k4_switch > 0:
    print(f"{n_k4_switch} bootstrap(s) exceed the 1% threshold and would require the "
          f"bin-stratified bootstrap in place of the primary one.")
else:
    print("No bootstrap exceeds the 1% threshold; the primary image bootstrap is used as reported.")

# %% [markdown]
# ## Chart 1 — Balanced MAE by model, with per-bin breakdown
#
# **What to look for.** Each panel is one model's five per-bin MAEs (cover
# points, y-axis) with 95% bootstrap intervals; the dashed line marks that
# model's balanced MAE (the unweighted mean of the five bars). Bar width does
# not encode n — n is annotated on each bar — precisely because the top bins
# are so much sparser than the bottom one, and a reader should not infer
# precision from bar width here.

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
for ax, model in zip(axes.flat, sorted(metrics_by_model["model"])):
    row = metrics_by_model.set_index("model").loc[model]
    maes = [row[f"mae_bin_{b}"] for b in co.BIN_LABELS]
    los = [row[f"mae_bin_{b}_ci_lo"] for b in co.BIN_LABELS]
    his = [row[f"mae_bin_{b}_ci_hi"] for b in co.BIN_LABELS]
    ns = [int(row[f"n_bin_{b}"]) for b in co.BIN_LABELS]
    yerr_lo = [m - l if np.isfinite(l) else 0 for m, l in zip(maes, los)]
    yerr_hi = [h - m if np.isfinite(h) else 0 for m, h in zip(maes, his)]
    ax.bar(co.BIN_LABELS, maes, yerr=[yerr_lo, yerr_hi], capsize=3, color="steelblue")
    ax.axhline(row["balanced_mae"], color="firebrick", linestyle="--", linewidth=1,
                label=f"balanced MAE = {row['balanced_mae']:.2f}")
    for x, (m, n) in enumerate(zip(maes, ns)):
        ax.annotate(f"n={n}", (x, m), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7)
    ax.set_title(model, fontsize=10)
    ax.set_xlabel("reference cover bin (%)")
    ax.legend(fontsize=7, loc="upper left")
axes[0, 0].set_ylabel("MAE (cover points)")
axes[1, 0].set_ylabel("MAE (cover points)")
fig.suptitle("Per-bin MAE by model, base/local, 95% image-bootstrap confidence intervals, n annotated per bar", y=1.02)
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q1_perbin_by_model.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Chart 2 — The 24 model x prompt combinations, and the ones that cannot
# ## be told apart from the best
#
# **What to look for.** Balanced MAE (lower is better) for all 24
# combinations, sorted low to high. Green points are the ones **inside** the
# 95% top-set — the lowest-lying cluster, including the observed best — and
# cannot be distinguished from it at 95%; gray points are excluded. There is
# no drawn threshold line; membership is read from the underlying table
# directly and encoded only by colour. Read this as "how many configurations
# are statistically tied for best," not as a clean ranking of 24.

# %%
fig, ax = plt.subplots(figsize=(11, 6))
plot_df = topset_bootstrap.copy()
plot_df["label"] = plot_df["model"] + " / " + plot_df["prompt"]
colors = plot_df["in_union_topset"].map({True: "seagreen", False: "lightgray"})
ax.scatter(range(len(plot_df)), plot_df["balanced_mae"], c=colors, s=40, zorder=3)
ax.plot(range(len(plot_df)), plot_df["balanced_mae"], color="gray", linewidth=0.5, zorder=2)
ax.set_xticks(range(len(plot_df)))
ax.set_xticklabels(plot_df["label"], rotation=90, fontsize=7)
ax.set_ylabel("balanced MAE (cover points)")
ax.set_title(f"24 model x prompt combinations, sorted by balanced MAE\n"
             f"green = the {int(plot_df['in_union_topset'].sum())} configurations in the 95% top-set, "
             f"indistinguishable from the best; gray = excluded")
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q1_topset_ranking.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Model choice versus prompt choice — magnitudes, not a test
#
# Does prompt design matter more or less than model choice? This is answered
# **descriptively**, with estimates and intervals and no p-value, because it
# is not a single well-posed hypothesis this design can test cleanly: the
# model axis is a model x framework x precision x GPU axis, while the prompt
# axis is clean of that confound but is inflated for a different, definitional
# reason (only one of the four prompts states an inclusion rule for
# vegetation, and it excludes dead material the reference sometimes counts).

# %%
combo_bal_wide = pd.DataFrame(
    [{"model": m, "prompt": p, "balanced_mae": combo_bal[(m, p)]} for (m, p) in ALL_COMBOS]
)

model_axis_per_prompt = combo_bal_wide.groupby("prompt")["balanced_mae"].agg(["max", "min", "std"])
model_axis_per_prompt["spread"] = model_axis_per_prompt["max"] - model_axis_per_prompt["min"]

prompt_axis_per_model = combo_bal_wide.groupby("model")["balanced_mae"].agg(["max", "min", "std"])
prompt_axis_per_model["spread"] = prompt_axis_per_model["max"] - prompt_axis_per_model["min"]

model_axis_summary = float(model_axis_per_prompt["spread"].mean())
prompt_axis_summary = float(prompt_axis_per_model["spread"].mean())

assert_matches_published("model_axis_spread", model_axis_summary, 7.361642)
assert_matches_published("prompt_axis_spread", prompt_axis_summary, 9.832957)
print("Model-axis and prompt-axis point spreads match the published values.")

print(f"Model-axis spread (mean over 4 prompts of max-min balanced MAE across 6 models): "
      f"{model_axis_summary:.3f} cover points")
print(f"Prompt-axis spread (mean over 6 models of max-min balanced MAE across 4 prompts): "
      f"{prompt_axis_summary:.3f} cover points")


def model_axis_spread_stat(image_ids: np.ndarray) -> float:
    theta = balanced_mae_per_combo(image_ids)
    df = pd.DataFrame({"model": [c[0] for c in ALL_COMBOS], "prompt": [c[1] for c in ALL_COMBOS], "bal": theta})
    per_prompt_spread = df.groupby("prompt")["bal"].agg(lambda s: s.max() - s.min())
    return float(per_prompt_spread.mean())


def prompt_axis_spread_stat(image_ids: np.ndarray) -> float:
    theta = balanced_mae_per_combo(image_ids)
    df = pd.DataFrame({"model": [c[0] for c in ALL_COMBOS], "prompt": [c[1] for c in ALL_COMBOS], "bal": theta})
    per_model_spread = df.groupby("model")["bal"].agg(lambda s: s.max() - s.min())
    return float(per_model_spread.mean())


def paired_diff_spread_stat(image_ids: np.ndarray) -> float:
    return model_axis_spread_stat(image_ids) - prompt_axis_spread_stat(image_ids)


axis_frame_stub = pd.DataFrame({"image": IMAGES})
boot_model_axis = co.image_bootstrap(axis_frame_stub, lambda f: model_axis_spread_stat(f["image"].values),
                                      rng_axis, bin_col=None, B=co.B_BOOTSTRAP)
boot_prompt_axis = co.image_bootstrap(axis_frame_stub, lambda f: prompt_axis_spread_stat(f["image"].values),
                                       rng_axis, bin_col=None, B=co.B_BOOTSTRAP)
boot_diff_axis = co.image_bootstrap(axis_frame_stub, lambda f: paired_diff_spread_stat(f["image"].values),
                                     rng_axis, bin_col=None, B=co.B_BOOTSTRAP)

print(f"\nModel-axis spread: {boot_model_axis.estimate:.3f} "
      f"[{boot_model_axis.ci_lo:.3f}, {boot_model_axis.ci_hi:.3f}] (95% bootstrap interval)")
print(f"Prompt-axis spread: {boot_prompt_axis.estimate:.3f} "
      f"[{boot_prompt_axis.ci_lo:.3f}, {boot_prompt_axis.ci_hi:.3f}] (95% bootstrap interval)")
print(f"Paired difference (model - prompt): {boot_diff_axis.estimate:.3f} "
      f"[{boot_diff_axis.ci_lo:.3f}, {boot_diff_axis.ci_hi:.3f}] (95% bootstrap interval, no p-value)")

# %%
combo_bal_wide_ex_detailed = combo_bal_wide[combo_bal_wide["prompt"] != "Detailed"]
prompt_axis_per_model_ex = combo_bal_wide_ex_detailed.groupby("model")["balanced_mae"].agg(["max", "min"])
prompt_axis_per_model_ex["spread"] = prompt_axis_per_model_ex["max"] - prompt_axis_per_model_ex["min"]
prompt_axis_summary_ex_detailed = float(prompt_axis_per_model_ex["spread"].mean())
assert_matches_published("prompt_axis_spread_excl_detailed", prompt_axis_summary_ex_detailed, 1.309486)
print(f"Prompt-axis spread excluding the Detailed prompt: {prompt_axis_summary_ex_detailed:.3f} cover points "
      f"(vs {prompt_axis_summary:.3f} including it)")

# %%
def variance_components(frame: pd.DataFrame) -> dict:
    """ANOVA-style moment-estimator decomposition of per-image |e| into
    image, model, prompt and model x prompt contributions."""
    grand_mean = frame["abs_e"].mean()
    n_img, n_model, n_prompt = frame["image"].nunique(), frame["model"].nunique(), frame["prompt"].nunique()

    image_means = frame.groupby("image")["abs_e"].mean()
    model_means = frame.groupby("model")["abs_e"].mean()
    prompt_means = frame.groupby("prompt")["abs_e"].mean()
    cell_means = frame.groupby(["model", "prompt"])["abs_e"].mean()

    ss_image = n_model * n_prompt * ((image_means - grand_mean) ** 2).sum()
    ss_model = n_img * n_prompt * ((model_means - grand_mean) ** 2).sum()
    ss_prompt = n_img * n_model * ((prompt_means - grand_mean) ** 2).sum()

    interaction_terms = []
    for (m, p), cm in cell_means.items():
        interaction_terms.append((cm - model_means[m] - prompt_means[p] + grand_mean) ** 2)
    ss_interaction = n_img * float(np.sum(interaction_terms))

    ss_total = ((frame["abs_e"] - grand_mean) ** 2).sum()
    ss_resid = ss_total - ss_image - ss_model - ss_prompt - ss_interaction

    shares = {
        "image": ss_image / ss_total, "model": ss_model / ss_total,
        "prompt": ss_prompt / ss_total, "model_x_prompt": ss_interaction / ss_total,
        "residual": max(ss_resid, 0.0) / ss_total,
    }
    # Renormalise the model / prompt / interaction shares over the three
    # non-image components only. `residual` is excluded from both the
    # renormalising denominator and the renormalised output: a share
    # renormalised over (model + prompt + interaction) but still including
    # `residual` in the numerator can exceed 1 (it does here -- `residual`
    # alone is larger than the other three combined).
    non_image_total = shares["model"] + shares["prompt"] + shares["model_x_prompt"]
    renorm = {f"{k}_renorm_over_non_image": (v / non_image_total if non_image_total > 0 else np.nan)
              for k, v in shares.items() if k not in ("image", "residual")}
    return {**shares, **renorm}


vc = variance_components(base_local)
assert_matches_published("variance_share_model_x_prompt_renorm_over_non_image",
                          vc["model_x_prompt_renorm_over_non_image"], 0.22799, atol=1e-4)
assert_matches_published("variance_share_prompt_renorm_over_non_image",
                          vc["prompt_renorm_over_non_image"], 0.13726, atol=1e-4)
print("\nVariance-component decomposition of per-image |e| (share of total sum of squares):")
for k, v in vc.items():
    print(f"  {k:35s}: {v:.4f}")

axis_magnitudes = pd.DataFrame([{
    "question_id": "Q1",
    "model_axis_spread": boot_model_axis.estimate,
    "model_axis_ci_lo": boot_model_axis.ci_lo, "model_axis_ci_hi": boot_model_axis.ci_hi,
    "prompt_axis_spread": boot_prompt_axis.estimate,
    "prompt_axis_ci_lo": boot_prompt_axis.ci_lo, "prompt_axis_ci_hi": boot_prompt_axis.ci_hi,
    "prompt_axis_spread_excl_detailed": prompt_axis_summary_ex_detailed,
    "paired_diff_model_minus_prompt": boot_diff_axis.estimate,
    "paired_diff_ci_lo": boot_diff_axis.ci_lo, "paired_diff_ci_hi": boot_diff_axis.ci_hi,
    **{f"variance_share_{k}": v for k, v in vc.items()},
    "note": (
        "Estimates with 95% bootstrap confidence intervals, no p-value. "
        "Two caveats travel with these numbers: (1) the model axis mixes model "
        "identity with serving stack/precision/GPU, so model_axis_spread is "
        "not attributable to model choice alone; (2) prompt_axis_spread is "
        "dominated by the one prompt whose instructions define a "
        "vegetation-inclusion rule the other three do not state and "
        "that excludes dead material the reference sometimes counts -- a "
        "construct mismatch, not a wording effect."
    ),
}])
axis_magnitudes

# %% [markdown]
# ## Chart 3 — Model-axis vs prompt-axis spread

# %%
fig, ax = plt.subplots(figsize=(6, 5))
axes_names = ["model axis", "prompt axis"]
ests = [boot_model_axis.estimate, boot_prompt_axis.estimate]
los = [boot_model_axis.ci_lo, boot_prompt_axis.ci_lo]
his = [boot_model_axis.ci_hi, boot_prompt_axis.ci_hi]
yerr = [[e - l for e, l in zip(ests, los)], [h - e for e, h in zip(ests, his)]]
ax.bar(axes_names, ests, yerr=yerr, capsize=5, color=["darkorange", "slateblue"])
ax.set_ylabel("spread of balanced MAE (cover points), mean across the other axis")
ax.set_title("Model-axis vs prompt-axis spread\n"
              "model axis mixes serving stack; prompt axis inflated by one prompt's construct mismatch\n"
              "estimates with 95% bootstrap confidence interval, not a hypothesis test")
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q1_axis_magnitudes.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Chain-of-thought sensitivity and determinism
#
# A handful of Llama-4-Scout Grid-Overlay predictions were recovered from
# chain-of-thought text rather than a direct structured response. Checked
# here directly, on this notebook's own frame, rather than assumed small.

# %%
scout_cot_rows = d9[(d9["in_scope"] == True) & (d9["stack"] == "local")  # noqa: E712
                    & (d9["model"] == "Llama-4-Scout") & (d9["error"] == "JSON_PARSE_FAILURE")
                    & (d9["prompt"] == "Grid-Overlay") & (d9["variant"] == "base")]
print(f"Llama-4-Scout Grid-Overlay base rows recovered from chain-of-thought text: "
      f"{len(scout_cot_rows)} of 27,720 base/local prediction rows")

determinism_floor = pd.DataFrame([{
    "question_id": "Q1",
    "model": model, "n_reproduced": n, "n_total": 100,
} for model, n in maverick_floor.n_reproduced.items()])
determinism_floor["delta_mav"] = np.where(
    determinism_floor["model"] == "Llama-4-Maverick", maverick_floor.delta_mav, np.nan)
determinism_floor["delta_mav_ci_lo"] = np.where(
    determinism_floor["model"] == "Llama-4-Maverick", maverick_floor.delta_mav_ci_lo, np.nan)
determinism_floor["delta_mav_ci_hi"] = np.where(
    determinism_floor["model"] == "Llama-4-Maverick", maverick_floor.delta_mav_ci_hi, np.nan)
determinism_floor["median_nonzero_disagreement"] = np.where(
    determinism_floor["model"] == "Llama-4-Maverick", maverick_floor.median_nonzero_disagreement, np.nan)
determinism_floor["max_nonzero_disagreement"] = np.where(
    determinism_floor["model"] == "Llama-4-Maverick", maverick_floor.max_nonzero_disagreement, np.nan)
determinism_floor["scout_cot_rows_in_q1_frame"] = len(scout_cot_rows)
determinism_floor

# %%
print(f"delta_Mav (overall-MAE scale) = {maverick_floor.delta_mav:.4f} cover points, for scale against "
      f"the median |difference in balanced MAE| across the 15 model-versus-model comparisons of "
      f"{float(contrasts_models['balanced_mae_diff_estimate'].abs().median()):.4f} cover points.")

# %% [markdown]
# ## Assumption-check summary written to CSV

# %%
assumption_rows = assumption_df.to_dict("records")
assumption_rows.append({"id": "K1", "description": "pairing complete: every image has exactly 24 base/local rows",
                         "passed": k1.passed, "detail": k1.detail})
for _, r in k5_df.iterrows():
    assumption_rows.append({"id": "K5", "description": f"D'Agostino-Pearson normality ({r['model']}), context only",
                             "passed": True, "detail": f"statistic={r['statistic']:.3f}, p={r['p_value']:.3g}"})
for label in co.BIN_LABELS:
    assumption_rows.append({"id": "K6", "description": f"per-bin sd(|e|), bin {label}, context only",
                             "passed": True, "detail": f"sd={k6[label]:.3f}"})
assumption_rows.append({"id": "K10", "description": "between-image independence -- not checkable, "
                                                      "assumed and not quantified in this notebook",
                         "passed": None, "detail": "assumed, not tested"})
assumption_rows.append({"id": "C11", "description": "n_reproduced 92 (Maverick) / 100 (others)",
                         "passed": c11.passed, "detail": c11.detail})
assumption_rows.append({
    "id": "K4",
    "description": "bootstrap bin coverage: empty-bin resamples across the model, prompt and top-set bootstraps",
    "passed": n_k4_switch == 0,
    "detail": f"n_violations={n_k4_violations_total}; {n_k4_switch} bootstrap(s) over the 1% switch threshold",
})

# Per-comparison symmetry and tie-burden diagnostics -- reported, never
# switching a test.
for fam_df, fam_label in ((contrasts_models, "A"), (contrasts_prompts, "B")):
    for _, r in fam_df.iterrows():
        assumption_rows.append({
            "id": "K2", "description": f"symmetry of paired differences (Hodges-Lehmann reading only), {r['contrast']}",
            "passed": not r["symmetry_violated"], "detail": f"skew={r['skew_d']:.3f}, hl_vs_median_gap={r['hl_vs_median_gap']:.3f}",
        })
        assumption_rows.append({
            "id": "K3", "description": f"tie burden, {r['contrast']}",
            "passed": not r["tie_burden_exceeded"], "detail": f"zero_share={r['zero_share']:.3f}",
        })

assumption_checks = pd.DataFrame(assumption_rows)

# `passed` holds True/False/None and is therefore `object` dtype, not `bool`
# -- `~` on an object-dtype Series performs Python's integer bitwise
# negation on each element, not logical negation, and silently returns
# nonsense rather than raising. Casting to `bool` first makes `~` the
# logical negation the check actually needs.
k2_mask = assumption_checks["id"] == "K2"
k3_mask = assumption_checks["id"] == "K3"
k2_passed_bool = assumption_checks.loc[k2_mask, "passed"].astype(bool)
k3_passed_bool = assumption_checks.loc[k3_mask, "passed"].astype(bool)
n_k2_violations = int((~k2_passed_bool).sum())
n_k3_violations = int((~k3_passed_bool).sum())
print(f"Symmetry diagnostic flagged on {n_k2_violations} of {int(k2_mask.sum())} model/prompt comparisons")
print(f"Tie-burden diagnostic flagged on {n_k3_violations} of {int(k3_mask.sum())} model/prompt comparisons")
print("Neither diagnostic changes which test is run: the paired Wilcoxon and the "
      "Hodges-Lehmann reading are used throughout, on every comparison.")

assumption_checks.tail(10)

# %% [markdown]
# ## Result
#
# **The realised top-set.** The configurations in the top-set cannot be
# distinguished from the observed best at 95% (the observed-best
# configuration and the top-set size are printed below); membership and the
# probability of being best for all 24 configurations are in
# `Q1_topset_bootstrap.csv`. Any claim of a single best configuration on this
# frame is properly a claim about that set, not about the single observed
# winner.
#
# **Headline.** Balanced MAE (with its five per-bin MAEs, n and confidence
# intervals) and overall MAE (with its own interval), RMSE, mean and median
# bias (mean bias with a per-bin interval), the two within-tolerance rates,
# the exact-zero rate and tie-corrected Spearman are reported for all 6
# models, all 4 prompts and all 24 combinations in `Q1_metrics_by_model.csv`,
# `Q1_metrics_by_prompt.csv`, `Q1_metrics_by_combo.csv` and
# `Q1_metrics_perbin.csv`.
#
# **The one test in this notebook.** `Q1_contrasts_models.csv` (15
# comparisons) and `Q1_contrasts_prompts.csv` (6 comparisons) carry the
# paired Wilcoxon signed-rank test on overall MAE, Holm-adjusted within each
# set of comparisons, alongside the bootstrap interval on the difference of
# balanced MAE (`balanced_mae_interval_uncorrected = TRUE`) and the companion
# effect sizes (Hodges-Lehmann median difference, rank-biserial correlation,
# win rate). Symmetry and tie-burden diagnostics are reported per comparison
# but never switch which test is run.
#
# **The disclosing clause that travels with every balanced-MAE comparison.**
# Balanced MAE's pairwise comparisons are 95% bootstrap intervals on the
# difference, read one at a time, and they are **not corrected for the
# number of comparisons made**. This is a property of the statistic: there
# is no p-value for balanced MAE anywhere in this notebook, so there is
# nothing to correct.
#
# **The 24 configurations carry no pairwise test at all.** What is
# reportable is top-set membership or exclusion and the bootstrap probability
# of being best — never a sentence comparing two specific configurations by
# name. That comparison, if wanted, comes from the model-versus-model or
# prompt-versus-prompt comparisons above.
#
# **The effective sample size behind every balanced-MAE statement is ~273,
# not 1,155.** Because the top two bins carry 65% of balanced MAE's sampling
# variance, this design detects large differences reliably in that estimand
# and small ones only occasionally, while the Wilcoxon on overall MAE is 81%
# driven by the 933-image bottom bin. A comparison separated on one and not
# the other is read as a statement about where in the cover range the
# difference lives, never as one measurement failing where another
# succeeded.
#
# **Model choice vs prompt choice — read with both caveats, not one number.**
# `Q1_axis_magnitudes.csv` reports the prompt-axis spread and the model-axis
# spread as estimates with 95% bootstrap intervals and their paired
# difference, with no p-value. Read on its own, a larger prompt-axis spread
# says prompt choice moves balanced MAE more than model choice does on this
# frame; it should not be read on its own, because one of the four prompts
# states an inclusion rule for vegetation that excludes dead material the
# reference sometimes counts — a construct mismatch, not a wording effect —
# and excluding that one prompt collapses most of the prompt-axis spread. The
# model axis carries its own caveat in the other direction: it is confounded
# with serving stack, precision and GPU (8 of the 15 model-versus-model
# comparisons cross that confound), so a model-axis spread is not
# attributable to "model" alone either. Neither comparison is a hypothesis
# test, and this frame does not resolve which axis "really" matters more once
# both caveats are held fixed.
#
# **Determinism.** `Q1_determinism_floor.csv` carries the 92/100-vs-100/100
# reproducibility table and δ_Mav = 0.65 cover points. Every model-versus-
# model comparison involving Llama-4-Maverick is evaluated against δ_Mav on
# the overall-MAE scale — the scale the Wilcoxon test runs on and exactly the
# scale δ_Mav bounds, because δ_Mav is derived from the run-to-run
# disagreement in the same absolute-error quantity overall MAE averages, and
# it bounds |MAE_a − MAE_b| directly. Two of the five Maverick comparisons —
# Llama-4-Maverick vs Llama-4-Scout (overall-MAE difference −0.7731) and
# Llama-4-Maverick vs Qwen-2.5 (overall-MAE difference +0.0877) — sit on
# opposite sides of that floor: the Scout comparison's effect exceeds δ_Mav
# and is claimable (its Holm-adjusted p is 3.71e-10), while the Qwen-2.5
# comparison's effect is smaller than Maverick's own run-to-run disagreement
# and is not claimable regardless of its p-value. The printed table above
# gives the verdict for all five.

# %%
best_model, best_prompt = best_key
print(f"Observed best configuration: {best_model} x {best_prompt}")
print(f"Union top-set size: {len(union_topset)} of 24 configurations")

metrics_by_model.to_csv(RESULTS_DIR / "Q1_metrics_by_model.csv", index=False)
metrics_by_prompt.to_csv(RESULTS_DIR / "Q1_metrics_by_prompt.csv", index=False)
metrics_by_combo.to_csv(RESULTS_DIR / "Q1_metrics_by_combo.csv", index=False)
metrics_perbin.to_csv(RESULTS_DIR / "Q1_metrics_perbin.csv", index=False)
contrasts_models.to_csv(RESULTS_DIR / "Q1_contrasts_models.csv", index=False)
contrasts_prompts.to_csv(RESULTS_DIR / "Q1_contrasts_prompts.csv", index=False)
metrics_by_model_no_detail.to_csv(RESULTS_DIR / "Q1_metrics_by_model_no_detail.csv", index=False)
contrasts_models_no_detail.to_csv(RESULTS_DIR / "Q1_contrasts_models_no_detail.csv", index=False)
topset_bootstrap.to_csv(RESULTS_DIR / "Q1_topset_bootstrap.csv", index=False)
axis_magnitudes.to_csv(RESULTS_DIR / "Q1_axis_magnitudes.csv", index=False)
determinism_floor.to_csv(RESULTS_DIR / "Q1_determinism_floor.csv", index=False)
assumption_checks.to_csv(RESULTS_DIR / "Q1_assumption_checks.csv", index=False)

print("\nWritten:")
for fname in [
    "Q1_metrics_by_model.csv", "Q1_metrics_by_prompt.csv", "Q1_metrics_by_combo.csv",
    "Q1_metrics_perbin.csv", "Q1_contrasts_models.csv", "Q1_contrasts_prompts.csv",
    "Q1_topset_bootstrap.csv", "Q1_axis_magnitudes.csv",
    "Q1_determinism_floor.csv", "Q1_assumption_checks.csv",
    "Q1_metrics_by_model_no_detail.csv", "Q1_contrasts_models_no_detail.csv",
]:
    print(f"  results/{fname}")
