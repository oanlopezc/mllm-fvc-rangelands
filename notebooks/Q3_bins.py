# %% [markdown]
# # Q3 — Error against reference cover: profile, trend, and three covariates
#
# The reference used throughout is a two-observer consensus estimate of
# vegetation cover, `D1.v7`, bounded between 0 and 100 cover points. Absolute
# error against a bounded reference is floor-constrained: a model predicting
# near 0 against a reference of 0 has almost no error available to it, while
# against a reference of 90 an under-prediction can be as large as 90 points.
# Any absolute-error metric therefore trends upward with the reference *by
# construction*, whatever the model does. This notebook reports that profile
# without treating a rising trend in raw absolute error as evidence about
# model behaviour on its own — the finding it licenses is a description of
# where in the cover range the error and its sign sit, not a verdict on
# whether the models are "really" worse at high cover.
#
# Three things follow. First, the per-bin profile of mean absolute error and
# signed bias across five reference-cover bins, with the trend in each
# tested by a permutation Jonckheere-Terpstra test. Second, how much three
# per-image covariates — rooted dead look-alike plants
# (`rooted_dead_alike_plants`), non-rooted plant material
# (`non_rooted_plant_material`), and quadrat-plane obliquity — add to
# per-image error once cover bin is already accounted for, both pooled and
# broken out by model and by prompt. Third, the relationship between
# obliquity and cover itself, reported as a limitation on what the pooled
# obliquity coefficient can mean rather than as a finding about photographs.

# %% [markdown]
# ## Setup
#
# The seed is fixed so this notebook produces identical results on every
# run.
#
# **RNG discipline.** Every bootstrap quantity below draws from its own
# generator, seeded from `SEED` plus a fixed integer offset declared next to
# its use, so that adding or removing one bootstrap in this notebook cannot
# silently move the draw sequence of another. One consequence follows
# directly: **the deterministic quantities below — MAE, signed bias, the
# Jonckheere-Terpstra z-statistics, ΔR², and the OLS and median-regression
# coefficients — are checked against literal published values**, embedded in
# this notebook rather than read from any file, because they are the numbers
# that appear in the manuscript and this cell exists so a change to the
# analysis cannot silently invalidate it. **Interval endpoints, and every
# p-value built from a permutation or bootstrap draw, are not checked against
# a literal** — they are stochastic by construction, and a resampling draw
# that differs from a previous run by less than Monte Carlo noise is
# expected, not a defect.

# %%
import os
import sys
import warnings
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.quantile_regression import QuantReg


def _locate_project_root(start=None):
    """Resolve the project root independently of the kernel's working
    directory, so this notebook runs identically whether launched by the
    render pipeline or opened by hand.
    """
    p = Path(os.environ.get("ANALYSIS_PROJECT_ROOT", start or Path.cwd())).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "STATUS.md").is_file():
            return candidate
    raise RuntimeError(f"project root not found from {p}")


sys.path.insert(0, str(_locate_project_root() / "03_notebooks_definitive"))
import pathlib as _pl  # `_common.py` is imported from this notebook's
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))  # own directory
import _common as C

SEED = 20260910

# Each bootstrap-consuming block gets its own generator, offset from SEED by
# a fixed integer declared here and nowhere else, so that adding a bootstrap
# to one block never shifts the draw sequence of another:
#   +0  per-bin error table (MAE, mean/median bias, sAPE, MAE/midpoint)
#   +1  headline balanced-MAE bootstrap and its empty-bin coverage check
#   +2  per-bin skill ratio
#   +3  OLS image-bootstrap coefficient CIs (pooled additive covariate model)
#   +4  within-bin permutation covariate-block tests
#   +5  signed and absolute error by rooted dead look-alike level, pooled and by prompt
#   +6  per-model / per-prompt covariate contrasts
#   +7  obliquity-cover Spearman BCa interval
TREND_PERM_SEED = SEED + 100  # the two pooled trend tests
rng_perbin = np.random.default_rng(SEED + 0)
rng_headline_bmae = np.random.default_rng(SEED + 1)
rng_skill_ratio = np.random.default_rng(SEED + 2)
rng_ols_boot = np.random.default_rng(SEED + 3)
rng_block_perm = np.random.default_rng(SEED + 4)
rng_rooted_level = np.random.default_rng(SEED + 5)
rng_by_configuration = np.random.default_rng(SEED + 6)
rng_obliquity = np.random.default_rng(SEED + 7)
rng_trend_jt = np.random.default_rng(TREND_PERM_SEED)

ROOT = C.project_root()
RESULTS_DIR = ROOT / "05_deliverables/Repository/GitHub/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RENDERED_DIR = ROOT / "05_deliverables/Repository/GitHub/notebooks" / "rendered"
RENDERED_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

# The five pre-declared cover-gradient tests (two trend tests plus three
# covariate-block tests) are corrected together, Holm-Bonferroni, so that a
# reader is not shown five separately-flagged 5% tests, which would overstate
# how many of them are genuinely significant.
N_COVER_GRADIENT_TESTS = 5
HOLM_FIRST_THRESHOLD = 0.05 / N_COVER_GRADIENT_TESTS

BIN_MIDPOINTS = {"0-20": 10.0, "20-40": 30.0, "40-60": 50.0, "60-80": 70.0, "80-100": 90.0}

print(f"seed={SEED}, five cover-gradient tests, Holm first threshold = {HOLM_FIRST_THRESHOLD:.5f}")


def assert_matches_published(name: str, computed: float, published: float, atol: float = 1e-6) -> None:
    """The deterministic quantities in this notebook (MAE, signed bias, JT
    z-statistics, delta R^2, OLS/median-regression coefficients) are
    published values. This check fails if any of them moves, so a change to
    the analysis cannot silently invalidate the manuscript.
    """
    if not np.isclose(computed, published, atol=atol, rtol=0):
        raise AssertionError(
            f"{name}: computed {computed!r} does not match the published value "
            f"{published!r} (atol={atol})"
        )

# %% [markdown]
# ## Data
#
# The analysis frame is `variant == 'base'`, local stack: 1,155 images x 6
# models x 4 prompts = 27,720 prediction rows. Every trend test and
# covariate model pools over the 24 configurations to one row per image
# (per-image mean absolute error, mean signed error, or mean relative
# skill), so the unit of every test below is the **image** (n = 1,155) — the
# same photograph is never counted twice inside one test. All load-time
# checks are run and must pass before anything downstream is computed; the
# check that every image carries exactly 24 prediction rows is run
# explicitly alongside them.

# %%
checks_df, frames = C.run_all_assertions(include_d12=False)
d1, d2, d3, d4, d5 = frames["d1"], frames["d2"], frames["d3"], frames["d4"], frames["d5"]
base_local = frames["base_local"]

k1_result = C.assert_k1_pairing_complete(d5, base_local)
checks_df = pd.concat([checks_df, pd.DataFrame([k1_result.__dict__])], ignore_index=True)

print(checks_df[["id", "description", "passed"]].to_string(index=False))
assert checks_df["passed"].all(), "one or more load-time checks failed"

print(f"\nbase_local: {len(base_local)} rows, {base_local['image'].nunique()} images")

# %% [markdown]
# **Missingness strategy.** The reference, the model predictions, and the
# three covariates are all complete at intake (0.0% missing on every one of
# them), and the join check above confirms no row is silently dropped by any
# join this notebook performs. No imputation is used anywhere in this
# notebook because none is needed; if a future data refresh introduces
# missingness, the checks above fail loudly rather than the analysis
# silently proceeding on a smaller n.

# %% [markdown]
# ## Building the per-image analysis frame
#
# Every trend test and covariate model pools over the six models and four
# prompts: the per-image value is the mean of the row-level quantity
# (absolute error, signed error) over the 24 configurations for that image —
# the mean of the per-configuration quantity, never the error of a pooled or
# ensembled prediction. Both bin (ordinal, by construction) and the three
# covariates (constant within an image) attach cleanly to this one-row-per-
# image frame.

# %%
per_image = (
    base_local.groupby("image", as_index=False)
    .agg(reference=("reference", "first"), campaign=("campaign", "first"),
         mean_abs_e=("abs_e", "mean"), mean_e=("e", "mean"))
)
per_image["bin"] = C.assign_bins(per_image["reference"])

per_image = per_image.merge(
    d2[["image", "rooted_dead_alike_plants", "non_rooted_plant_material"]],
    on="image", how="left", validate="one_to_one",
)
per_image = per_image.merge(d4[["image", "obliquity"]], on="image", how="left", validate="one_to_one")

assert per_image["image"].is_unique and len(per_image) == 1155
assert per_image[["rooted_dead_alike_plants", "non_rooted_plant_material", "obliquity"]].isna().sum().sum() == 0

per_bin_n = C.per_bin_n_table(per_image)
print("per-bin n (images):", per_bin_n)

# %% [markdown]
# ## A floor-matched constant baseline, used only to define relative skill
#
# `B_median` always predicts the reference's median value (computed from
# `D1.v7` over the full 1,155-image frame, never quoted from memory). It
# exists solely so that a scale-free relative-skill quantity can be defined
# below — it inherits the same floor constraint as every model, which is
# what makes the comparison scale-free rather than a claim that the baseline
# is a reasonable predictor in its own right.

# %%
b_median = float(d1["reference"].median())
b_mean = float(d1["reference"].mean())
b_zero = 0.0

per_image["e_base_median"] = b_median - per_image["reference"]
per_image["e_base_mean"] = b_mean - per_image["reference"]
per_image["e_base_zero"] = b_zero - per_image["reference"]
for name in ("median", "mean", "zero"):
    per_image[f"abs_e_base_{name}"] = per_image[f"e_base_{name}"].abs()

baseline_rows = []
for name, e_col, abs_col in (
    ("B_median", "e_base_median", "abs_e_base_median"),
    ("B_mean", "e_base_mean", "abs_e_base_mean"),
    ("B_zero", "e_base_zero", "abs_e_base_zero"),
):
    bmae_res = C.balanced_mae(per_image[abs_col], per_image["bin"])
    overall_mae = float(per_image[abs_col].mean())
    baseline_rows.append({
        "question_id": "Q3", "baseline": name,
        "baseline_value": {"B_median": b_median, "B_mean": b_mean, "B_zero": b_zero}[name],
        "overall_mae": overall_mae, "balanced_mae": bmae_res.balanced_mae,
        **{f"mae_bin_{lbl}": bmae_res.per_bin_mae[lbl] for lbl in C.BIN_LABELS},
        **{f"n_bin_{lbl}": bmae_res.per_bin_n[lbl] for lbl in C.BIN_LABELS},
        "is_oracle_baseline": True,
        "note": "computed from D1.v7 over the full 1,155-image frame; calibrated on the same "
                "frame it is scored on, so harder to beat than a baseline built without seeing "
                "this data.",
    })
baseline_df = pd.DataFrame(baseline_rows)

# Published values (deterministic functions of the loaded reference column).
PUBLISHED_BASELINE_MAE = {
    "B_median": {"overall_mae": 11.855367965367966, "balanced_mae": 47.1354980271095},
    "B_mean": {"overall_mae": 14.918576038679934, "balanced_mae": 40.76193059347132},
    "B_zero": {"overall_mae": 12.478831168831169, "balanced_mae": 49.54709502603769},
}
for _, r in baseline_df.iterrows():
    pub = PUBLISHED_BASELINE_MAE[r["baseline"]]
    assert_matches_published(f"balanced_mae[{r['baseline']}]", r["balanced_mae"], pub["balanced_mae"])
    assert_matches_published(f"overall_mae[{r['baseline']}]", r["overall_mae"], pub["overall_mae"])
print("Baseline point estimates match the published values.")
baseline_df

# %% [markdown]
# ## Relative skill against the floor-matched baseline — sign convention
#
# `skill_i = |e_base,i| - |e_model,i|` against `B_median`: positive means
# the model's error on that image is smaller than the baseline's.
# `r_i = (|e_base,i| - |e_model,i|) / (|e_base,i| + |e_model,i|)`, `r_i = 0`
# when both errors are exactly zero, is the scale-free form: it is bounded in
# `[-1, 1]` and a common multiplicative change in both errors cancels in the
# ratio, so `r_i` cannot trend with cover purely because errors get larger in
# absolute terms everywhere. The sign convention is verified arithmetically
# below rather than trusted from the formula alone.

# %%
per_image["e_model"] = per_image["mean_abs_e"]
model_smaller = per_image["e_model"] < per_image["abs_e_base_median"]
model_larger = per_image["e_model"] > per_image["abs_e_base_median"]


def relative_skill(abs_e_base: np.ndarray, abs_e_model: np.ndarray) -> np.ndarray:
    """r_i = (|e_base| - |e_model|) / (|e_base| + |e_model|); r_i = 0 when both are 0."""
    num = abs_e_base - abs_e_model
    den = abs_e_base + abs_e_model
    return np.where(den > 0, num / den, 0.0)


per_image["skill"] = per_image["abs_e_base_median"] - per_image["e_model"]
per_image["r"] = relative_skill(per_image["abs_e_base_median"].values, per_image["e_model"].values)

sign_check_ok = (
    (per_image.loc[model_smaller, "r"] > 0).all()
    and (per_image.loc[model_larger, "r"] < 0).all()
    and (per_image.loc[model_smaller, "skill"] > 0).all()
    and (per_image.loc[model_larger, "skill"] < 0).all()
    and per_image["r"].between(-1, 1).all()
)
print(f"sign check: model-beats-baseline rows all have r>0 and skill>0; "
      f"baseline-beats-model rows all have r<0 and skill<0; r in [-1,1]: {sign_check_ok}")
assert sign_check_ok, "sign convention violated — stop, do not proceed"

# %% [markdown]
# ### Tie share of `r_i` per bin
#
# `r_i` is heavily tied at ±1 whenever one of the two errors (model or
# baseline) is exactly zero; the tie share per bin is reported so a bin's
# mean `r` driven mostly by a pile-up at one extreme reads differently from
# one where `r` is smoothly distributed.

# %%
r_tie_minus1 = per_image["r"] == -1.0
r_tie_plus1 = per_image["r"] == 1.0
tie_share_rows = []
for lbl in C.BIN_LABELS:
    sub = per_image[per_image["bin"] == lbl]
    n_bin = len(sub)
    n_minus1 = int((sub["r"] == -1.0).sum())
    n_plus1 = int((sub["r"] == 1.0).sum())
    tie_share_rows.append({
        "bin": lbl, "n_images": n_bin,
        "n_r_eq_minus1": n_minus1, "share_r_eq_minus1": n_minus1 / n_bin if n_bin else np.nan,
        "n_r_eq_plus1": n_plus1, "share_r_eq_plus1": n_plus1 / n_bin if n_bin else np.nan,
    })
r_tie_share_df = pd.DataFrame(tie_share_rows)
print(f"r tie share per bin: total r=-1 count = {int(r_tie_minus1.sum())}, "
      f"total r=+1 count = {int(r_tie_plus1.sum())}")
print(r_tie_share_df.to_string(index=False))

# %% [markdown]
# ## Per-bin error tables
#
# Mean absolute error, signed bias (mean and median), median symmetric
# absolute percentage error, and MAE divided by the bin midpoint, each with
# n and a 95% BCa confidence interval per bin. Per-bin CIs come from the
# bin-stratified image bootstrap: looped per bin, resampling images within
# that bin to its own n, carrying every configuration of a drawn image
# together.

# %%
def per_bin_ci(values: pd.Series, bins: pd.Series, images: pd.Series, statfunc, rng) -> tuple[dict, dict, dict]:
    ci_lo, ci_hi, ci_method = {}, {}, {}
    tmp = pd.DataFrame({"v": values.values, "bin": bins.values, "image": images.values})
    for lbl in C.BIN_LABELS:
        sub = tmp[tmp["bin"] == lbl]
        if sub["image"].nunique() < 2:
            ci_lo[lbl], ci_hi[lbl], ci_method[lbl] = np.nan, np.nan, "not computed (n < 2 images)"
            continue
        boot = C.image_bootstrap(sub, lambda f: statfunc(f["v"]), rng, image_col="image", bin_col=None, B=C.B_BOOTSTRAP)
        ci_lo[lbl], ci_hi[lbl], ci_method[lbl] = boot.ci_lo, boot.ci_hi, boot.ci_method
    return ci_lo, ci_hi, ci_method


mae_ci_lo, mae_ci_hi, mae_ci_method = per_bin_ci(per_image["mean_abs_e"], per_image["bin"], per_image["image"], np.mean, rng_perbin)
model_bmae = C.balanced_mae(per_image["mean_abs_e"], per_image["bin"], ci_lo=mae_ci_lo, ci_hi=mae_ci_hi)

bias_mean_ci_lo, bias_mean_ci_hi, bias_mean_ci_method = per_bin_ci(per_image["mean_e"], per_image["bin"], per_image["image"], np.mean, rng_perbin)
bias_median_ci_lo, bias_median_ci_hi, bias_median_ci_method = per_bin_ci(per_image["mean_e"], per_image["bin"], per_image["image"], np.median, rng_perbin)

base_local = base_local.copy()
pred = base_local["vegetation_percent"]
ref = base_local["reference"]
denom = (pred.abs() + ref.abs()) / 2.0
sape = np.where(denom > 0, base_local["abs_e"] / denom, 0.0)
base_local["sape"] = sape
per_image_sape = base_local.groupby("image")["sape"].mean().rename("mean_sape")
per_image = per_image.merge(per_image_sape, on="image", how="left")

per_image["bin_midpoint"] = per_image["bin"].map(BIN_MIDPOINTS)
per_image["mae_over_midpoint"] = per_image["mean_abs_e"] / per_image["bin_midpoint"]

sape_ci_lo, sape_ci_hi, sape_ci_method = per_bin_ci(per_image["mean_sape"], per_image["bin"], per_image["image"], np.median, rng_perbin)
maemid_ci_lo, maemid_ci_hi, maemid_ci_method = per_bin_ci(per_image["mae_over_midpoint"], per_image["bin"], per_image["image"], np.mean, rng_perbin)

perbin_rows = []
for lbl in C.BIN_LABELS:
    sub = per_image[per_image["bin"] == lbl]
    perbin_rows.append({
        "question_id": "Q3", "bin": lbl, "bin_midpoint": BIN_MIDPOINTS[lbl], "n_images": len(sub),
        "mae": model_bmae.per_bin_mae[lbl], "mae_ci_lo": mae_ci_lo[lbl], "mae_ci_hi": mae_ci_hi[lbl],
        "mae_ci_method": mae_ci_method[lbl],
        "mean_signed_bias": float(sub["mean_e"].mean()) if len(sub) else np.nan,
        "mean_signed_bias_ci_lo": bias_mean_ci_lo[lbl], "mean_signed_bias_ci_hi": bias_mean_ci_hi[lbl],
        "mean_signed_bias_ci_method": bias_mean_ci_method[lbl],
        "median_signed_bias": float(sub["mean_e"].median()) if len(sub) else np.nan,
        "median_signed_bias_ci_lo": bias_median_ci_lo[lbl], "median_signed_bias_ci_hi": bias_median_ci_hi[lbl],
        "median_signed_bias_ci_method": bias_median_ci_method[lbl],
        "median_sape": float(sub["mean_sape"].median()) if len(sub) else np.nan,
        "median_sape_ci_lo": sape_ci_lo[lbl], "median_sape_ci_hi": sape_ci_hi[lbl],
        "median_sape_ci_method": sape_ci_method[lbl],
        "mae_over_midpoint": float(sub["mae_over_midpoint"].mean()) if len(sub) else np.nan,
        "mae_over_midpoint_ci_lo": maemid_ci_lo[lbl], "mae_over_midpoint_ci_hi": maemid_ci_hi[lbl],
        "mae_over_midpoint_ci_method": maemid_ci_method[lbl],
    })
perbin_df = pd.DataFrame(perbin_rows)

# Published values: mean absolute error and mean signed bias per bin.
PUBLISHED_PERBIN = {
    "0-20": {"mae": 6.3120266166488035, "mean_signed_bias": 5.402487495534119,
             "median_sape": 1.0651792944682879, "mae_over_midpoint": 0.6312026616648804},
    "20-40": {"mae": 10.503001543209878, "mean_signed_bias": 0.8876466049382714,
              "median_sape": 0.3770992234355196, "mae_over_midpoint": 0.35010005144032924},
    "40-60": {"mae": 16.029891304347824, "mean_signed_bias": -9.361413043478262,
              "median_sape": 0.39045186864895787, "mae_over_midpoint": 0.3205978260869565},
    "60-80": {"mae": 20.23519736842105, "mean_signed_bias": -16.150767543859647,
              "median_sape": 0.3535314414872217, "mae_over_midpoint": 0.28907424812030075},
    "80-100": {"mae": 25.9375, "mean_signed_bias": -25.64722222222222,
               "median_sape": 0.3550538346360522, "mae_over_midpoint": 0.2881944444444445},
}
for _, r in perbin_df.iterrows():
    pub = PUBLISHED_PERBIN[r["bin"]]
    assert_matches_published(f"mae[{r['bin']}]", r["mae"], pub["mae"])
    assert_matches_published(f"mean_signed_bias[{r['bin']}]", r["mean_signed_bias"], pub["mean_signed_bias"])
    assert_matches_published(f"median_sape[{r['bin']}]", r["median_sape"], pub["median_sape"])
    assert_matches_published(f"mae_over_midpoint[{r['bin']}]", r["mae_over_midpoint"], pub["mae_over_midpoint"])
print("Per-bin point estimates match the published values.")
print(f"balanced MAE (pooled across 24 configurations) = {model_bmae.balanced_mae:.3f}")
perbin_df

# %% [markdown]
# ## Normality and homoscedasticity, reported for context
#
# Neither the trend tests nor the covariate model below assumes normal or
# homoscedastic residuals — the trend tests are permutation-based and the
# covariate model's uncertainty comes from an image bootstrap. These two
# checks are reported because they are the reason those choices were made,
# not because anything downstream depends on their outcome.

# %%
k5 = C.check_k5_normality(per_image["mean_e"].values)
k6 = C.check_k6_homoscedasticity(per_image["mean_e"], per_image["bin"])
print(f"D'Agostino-Pearson normality test on per-image signed bias: statistic={k5['statistic']:.2f}, "
      f"p={k5['p_value']:.2e} (reported for context; no method here assumes normality)")
print(f"Per-bin sd of signed bias: {k6}")

k9_v3 = C.check_k9_level_counts(per_image["rooted_dead_alike_plants"])
k9_v4 = C.check_k9_level_counts(per_image["non_rooted_plant_material"])
print(f"rooted_dead_alike_plants level counts: {k9_v3['counts']}, any_below_30={k9_v3['any_below_min']}")
print(f"non_rooted_plant_material level counts: {k9_v4['counts']}, any_below_30={k9_v4['any_below_min']}")

assumption_rows = [
    {"question_id": "Q3", "check": "normality", "statistic": k5["statistic"], "p_value": k5["p_value"],
     "detail": "D'Agostino-Pearson on per-image mean signed bias; context only, no gating action"},
    {"question_id": "Q3", "check": "homoscedasticity", "statistic": np.nan, "p_value": np.nan,
     "detail": f"per-bin sd of signed bias: {k6}"},
    {"question_id": "Q3", "check": "rooted_dead_alike_plants_level_counts", "statistic": np.nan, "p_value": np.nan,
     "detail": f"counts={k9_v3['counts']}, any_below_30={k9_v3['any_below_min']}"},
    {"question_id": "Q3", "check": "non_rooted_plant_material_level_counts", "statistic": np.nan, "p_value": np.nan,
     "detail": f"counts={k9_v4['counts']}, any_below_30={k9_v4['any_below_min']}"},
]

# %% [markdown]
# ## The two trend tests
#
# A permutation Jonckheere-Terpstra test asks whether a response is
# stochastically ordered across ordered groups — legal here because the
# five reference-cover bins are ordered by construction and are disjoint
# image sets. The null is permuted by shuffling bin labels across images
# (10,000 permutations, mid-rank tie handling). Two quantities are tested
# this way: **mean absolute error** (the floor-constrained quantity — a
# rising trend here is expected by construction and is not on its own
# evidence of behaviour) and **signed bias** (whether the model over- or
# under-predicts, which the floor does not force in either direction).
# These two, together with three covariate-block tests further below, are
# the five cover-gradient tests corrected together by Holm-Bonferroni.

# %%
trend_specs = [
    ("signed_bias", per_image["mean_e"], "not forced by the error floor in either direction"),
    ("raw_abs_error", per_image["mean_abs_e"], "expected increasing — forced by the error floor regardless "
                                                "of model behaviour"),
]


def format_floored_p(p: float, n_perm: int = 10_000) -> str:
    floor = 1.0 / (n_perm + 1)
    return "< 1e-4" if p <= floor else f"{p:.5f}"


trend_results = {}
for name, series, note in trend_specs:
    jt_res = C.permutation_jt(
        series.to_numpy(), per_image["bin"].to_numpy(), list(C.BIN_LABELS), rng_trend_jt,
        alpha=HOLM_FIRST_THRESHOLD,
    )
    trend_results[name] = {"z_obs": jt_res.z, "p_raw": jt_res.p_perm, "direction": jt_res.direction,
                            "jt_result": jt_res, "note": note}
    print(f"{name:20s} z={jt_res.z:+.3f}  p={format_floored_p(jt_res.p_perm):>8s}  direction={jt_res.direction}")

# Published values: the JT z-statistic is a deterministic function of the
# loaded data (it does not depend on the permutation draws, only the
# permutation p-value does).
PUBLISHED_TREND_Z = {
    "signed_bias": -15.797488, "raw_abs_error": 18.324252,
}
for name in PUBLISHED_TREND_Z:
    assert_matches_published(f"z_obs[{name}]", trend_results[name]["z_obs"], PUBLISHED_TREND_Z[name], atol=1e-4)
print("Trend-test z-statistics match the published values.")

# %% [markdown]
# ## The per-bin skill ratio
#
# `1 - MAE_model,b / MAE_base,b` per bin, with an image-bootstrap BCa
# interval — the aggregate form of relative skill, reported alongside the
# per-bin MAE table. Positive means this bin's pooled MAE is below the
# floor-matched baseline's MAE in that bin.

# %%
def skill_ratio_stat(f: pd.DataFrame) -> float:
    mae_model = f["mean_abs_e"].mean()
    mae_base = f["abs_e_base_median"].mean()
    return 1.0 - mae_model / mae_base if mae_base > 0 else np.nan


skill_ratio_rows = []
for lbl in C.BIN_LABELS:
    sub = per_image[per_image["bin"] == lbl]
    point = skill_ratio_stat(sub)
    if sub["image"].nunique() >= 2:
        boot = C.image_bootstrap(sub, skill_ratio_stat, rng_skill_ratio, image_col="image", bin_col=None, B=C.B_BOOTSTRAP)
        lo, hi = boot.ci_lo, boot.ci_hi
    else:
        lo, hi = np.nan, np.nan
    skill_ratio_rows.append({"bin": lbl, "n_images": len(sub), "skill_ratio": point, "ci_lo": lo, "ci_hi": hi})
skill_ratio_df = pd.DataFrame(skill_ratio_rows)

PUBLISHED_SKILL_RATIO = {
    "0-20": -0.7616538785603537, "20-40": 0.6035217872538741, "40-60": 0.667249548736462,
    "60-80": 0.7057261768082663, "80-100": 0.7074718045112782,
}
for _, r in skill_ratio_df.iterrows():
    assert_matches_published(f"skill_ratio[{r['bin']}]", r["skill_ratio"], PUBLISHED_SKILL_RATIO[r["bin"]])
print("Per-bin skill-ratio point estimates match the published values.")
skill_ratio_df

# %% [markdown]
# ## The covariate model: how much do three covariates add net of bin?
#
# Outcome: per-image mean absolute error over the 24 configurations.
# Predictors: four bin dummies (the 0-20 bin is the reference level), two
# `rooted_dead_alike_plants` dummies (nominal — dummy-coded, never entered
# as a 0/1/2 score, since level identity carries no order), two
# `non_rooted_plant_material` dummies (ordinal, also dummy-coded here so
# its effect is not forced to be linear; monotonicity is asked separately
# below by a rank method), and standardised obliquity. The `clean` indicator
# is never entered — it is a deterministic function of the other two
# annotation columns and would be perfectly collinear with them.
#
# For each of the three covariate blocks, the rows of that block are
# permuted **within reference bin** (10,000 permutations), which isolates
# the block's own contribution from one that is really just the block being
# correlated with bin. Effect size is the change in R² from adding the
# block, with a bootstrap confidence interval, alongside each coefficient in
# cover points.

# %%
design = pd.DataFrame({"image": per_image["image"]})
bin_dummies = pd.get_dummies(per_image["bin"], prefix="bin", drop_first=True).astype(float)
design = pd.concat([design, bin_dummies], axis=1)

v3_dummies = pd.get_dummies(per_image["rooted_dead_alike_plants"].astype(int), prefix="rooted").astype(float)
v3_dummies = v3_dummies.drop(columns=[c for c in v3_dummies.columns if c.endswith("_0")])
v4_dummies = pd.get_dummies(per_image["non_rooted_plant_material"].astype(int), prefix="nonrooted").astype(float)
v4_dummies = v4_dummies.drop(columns=[c for c in v4_dummies.columns if c.endswith("_0")])

obliquity_z = (per_image["obliquity"] - per_image["obliquity"].mean()) / per_image["obliquity"].std()
design["obliquity_z"] = obliquity_z.values

blocks = {
    "bin": list(bin_dummies.columns),
    "rooted_dead_alike_plants": list(v3_dummies.columns),
    "non_rooted_plant_material": list(v4_dummies.columns),
    "obliquity": ["obliquity_z"],
}
full_design = pd.concat([design.drop(columns=["image"]), v3_dummies, v4_dummies], axis=1)

c9 = C.check_c9_clean_never_with_v3v4(full_design.columns)
print(f"clean is never entered alongside the two annotation dummy blocks: {c9}")
assert not c9["c9_violated"]

# %% [markdown]
# ### Obliquity, continuous vs quartile-binned — a linearity check
#
# Obliquity enters the covariate model continuously. This block checks
# whether that is a reasonable approximation by refitting with obliquity cut
# into quartiles instead: if the continuous fit is adequate, the quartile
# model should not improve R² materially. Descriptive only — obliquity's
# permutation test below uses the continuous specification.

# %%
obliquity_quartile = pd.qcut(per_image["obliquity"], q=4, labels=["q1", "q2", "q3", "q4"])
per_image["obliquity_quartile"] = obliquity_quartile
obliquity_q_dummies = pd.get_dummies(obliquity_quartile, prefix="obliquity").astype(float)
obliquity_q_dummies = obliquity_q_dummies.drop(columns=["obliquity_q1"])

other_blocks_cols = blocks["bin"] + list(v3_dummies.columns) + list(v4_dummies.columns)
design_continuous_obliquity = pd.concat(
    [full_design[other_blocks_cols], full_design[["obliquity_z"]]], axis=1,
)
design_quartile_obliquity = pd.concat(
    [full_design[other_blocks_cols], obliquity_q_dummies], axis=1,
)

y_obliquity_check = per_image["mean_abs_e"].to_numpy()
r2_cont = sm.OLS(y_obliquity_check,
                  sm.add_constant(design_continuous_obliquity.astype(float).to_numpy())).fit().rsquared
fit_quart = sm.OLS(y_obliquity_check,
                    sm.add_constant(design_quartile_obliquity.astype(float).to_numpy())).fit()
r2_quart = fit_quart.rsquared

print(f"obliquity linearity check: R^2 continuous = {r2_cont:.4f}, R^2 quartile-dummy = "
      f"{r2_quart:.4f} (delta = {r2_quart - r2_cont:+.4f})")

assumption_rows.append({
    "question_id": "Q3", "check": "obliquity_quartile_linearity", "statistic": np.nan, "p_value": np.nan,
    "detail": f"R^2 continuous={r2_cont:.4f}, R^2 quartile-dummy={r2_quart:.4f}, "
              f"delta={r2_quart - r2_cont:+.4f}; descriptive only, not one of the five cover-gradient tests",
})

# %% [markdown]
# ### Collinearity check
#
# The variance inflation factor is computed over the full design (bin
# dummies plus both annotation blocks plus obliquity). Any column with a
# VIF above 10 would flag its whole block as not separately identified from
# the rest of the design.

# %%
k7 = C.check_k7_collinearity(full_design, blocks)
print("per-column VIF:")
for col, vif in k7["per_column_vif"].items():
    print(f"  {col:20s} {vif:8.3f}")
print("block flags (VIF > 10 anywhere in block):", k7["block_not_separately_identified"])

assumption_rows.append({
    "question_id": "Q3", "check": "collinearity", "statistic": np.nan, "p_value": np.nan,
    "detail": f"per_column_vif={k7['per_column_vif']}, "
              f"block_not_separately_identified={k7['block_not_separately_identified']}",
})

# %% [markdown]
# ### The pooled covariate model: OLS with image-bootstrap BCa intervals,
# median regression as a robustness check
#
# The image bootstrap resamples images with replacement and refits OLS each
# time; the interval is BCa (bias-corrected and accelerated — the percentile
# limits are adjusted for estimated bias and skew in the resample
# distribution, using a leave-one-image-out jackknife for the acceleration
# term), which is this project's declared method throughout and is used here
# rather than a plain percentile interval. Median regression is reported
# alongside as a robustness check: it is not the primary fit because
# absolute error has a large point mass at exactly 0 (two of the six models
# predict 0 on roughly a third of rows), which destabilises a median fit,
# and averaging over 24 configurations first — as the outcome variable
# already does — substantially smooths that mass.

# %%
y = per_image["mean_abs_e"].to_numpy()
X = sm.add_constant(full_design.astype(float).to_numpy())
ols_full = sm.OLS(y, X).fit()
coef_names = ["const"] + list(full_design.columns)

# The image bootstrap needs one row-level frame carrying the outcome and
# every predictor column, so each resample can rebuild the design matrix
# exactly as the observed fit does, and a BCa interval with its own
# leave-one-image-out jackknife (image_bootstrap's declared method) rather
# than a plain percentile interval.
ols_boot_frame = pd.concat(
    [per_image[["image"]].reset_index(drop=True), full_design.reset_index(drop=True),
     pd.Series(y, name="mean_abs_e")],
    axis=1,
)
ols_coef_cis = {}
for i, name in enumerate(coef_names):
    def ols_coef_stat(f: pd.DataFrame, _i=i) -> float:
        Xf = sm.add_constant(f[full_design.columns].astype(float).to_numpy())
        return float(sm.OLS(f["mean_abs_e"].to_numpy(), Xf).fit().params[_i])

    boot = C.image_bootstrap(ols_boot_frame, ols_coef_stat, rng_ols_boot, image_col="image", bin_col=None, B=C.B_BOOTSTRAP)
    ols_coef_cis[name] = boot

qr_fit = QuantReg(y, X).fit(q=0.5)

covariate_model_rows = []
for i, name in enumerate(coef_names):
    boot = ols_coef_cis[name]
    covariate_model_rows.append({
        "question_id": "Q3", "term": name,
        "ols_coef": ols_full.params[i], "ols_ci_lo": boot.ci_lo, "ols_ci_hi": boot.ci_hi,
        "ols_ci_method": boot.ci_method,
        "qr_median_coef": float(qr_fit.params[i]),
        "qr_median_p": float(qr_fit.pvalues[i]),
        "qr_median_p_note": "descriptive robustness check, not one of the five cover-gradient tests; "
                             "no correction applies to this column",
        "estimator": "OLS (image-bootstrap BCa interval, primary) / median regression (robustness)",
    })
covariate_model_df = pd.DataFrame(covariate_model_rows)

PUBLISHED_OLS_COEF = {
    "const": 4.717397276202704, "bin_20-40": 2.0889256059039116, "bin_40-60": 8.150818664375873,
    "bin_60-80": 12.711568691648027, "bin_80-100": 18.33788346665877, "obliquity_z": 0.5624205869773633,
    "rooted_1": 5.649072898161939, "rooted_2": 2.8421432230148147,
    "nonrooted_1": 0.292263524924465, "nonrooted_2": 3.3552026652737172,
}
PUBLISHED_QR_COEF = {
    "const": 3.657843177894094, "bin_20-40": 2.925736777187659, "bin_40-60": 8.516838297200463,
    "bin_60-80": 12.203238850444832, "bin_80-100": 20.008963087372848, "obliquity_z": 0.649283647118196,
    "rooted_1": 4.174447978850651, "rooted_2": 2.021116676365722,
    "nonrooted_1": 0.8165389629246107, "nonrooted_2": 3.7041837252065477,
}
for _, r in covariate_model_df.iterrows():
    assert_matches_published(f"ols_coef[{r['term']}]", r["ols_coef"], PUBLISHED_OLS_COEF[r["term"]])
    assert_matches_published(f"qr_median_coef[{r['term']}]", r["qr_median_coef"], PUBLISHED_QR_COEF[r["term"]])
print("OLS and median-regression coefficients match the published values.")
print(f"OLS R^2 = {ols_full.rsquared:.4f}, n = {len(y)}")
covariate_model_df

# %% [markdown]
# ### Within-bin permutation tests, one per covariate block
#
# For each block, the model is fit twice: once with the bin dummies alone,
# once with the bin dummies plus that block. The difference in R² is the
# block's contribution. The permutation null shuffles the block's rows
# **within each reference bin** (never across bins), refits both models, and
# recomputes the R² difference each time.

# %%
def fit_r2(X_cols: pd.DataFrame, y: np.ndarray) -> float:
    Xm = sm.add_constant(X_cols.astype(float).to_numpy())
    return sm.OLS(y, Xm).fit().rsquared


def within_bin_permute(block_cols: list[str], bins: pd.Series, y: np.ndarray, base_cols: pd.DataFrame,
                        full_cols: pd.DataFrame, rng: np.random.Generator, n_perm: int = 10_000) -> dict:
    r2_base = fit_r2(base_cols, y)
    r2_full = fit_r2(full_cols, y)
    delta_obs = r2_full - r2_base

    block_vals = full_cols[block_cols].to_numpy()
    bins_arr = bins.to_numpy()
    delta_perm = np.empty(n_perm)
    for p in range(n_perm):
        permuted_block = block_vals.copy()
        for lbl in C.BIN_LABELS:
            mask = bins_arr == lbl
            idx = np.where(mask)[0]
            if len(idx) > 1:
                shuffled = rng.permutation(idx)
                permuted_block[idx] = block_vals[shuffled]
        perm_full = full_cols.copy()
        perm_full[block_cols] = permuted_block
        delta_perm[p] = fit_r2(perm_full, y) - r2_base

    p_raw = (1 + np.sum(delta_perm >= delta_obs)) / (n_perm + 1)
    return {"delta_r2": float(delta_obs), "p_raw": float(p_raw), "delta_perm": delta_perm}


base_cols_only_bin = full_design[blocks["bin"]]
block_test_results = {}
block_boot_frames = {}
for block_name in ("rooted_dead_alike_plants", "non_rooted_plant_material", "obliquity"):
    other_cols = blocks["bin"] + [c for name2, cols in blocks.items() if name2 not in ("bin", block_name) for c in cols]
    full_cols_for_block = full_design[blocks["bin"] + blocks[block_name] +
                                       [c for name2, cols in blocks.items()
                                        if name2 not in ("bin", block_name) for c in cols]]
    base_cols = full_design[other_cols]
    res = within_bin_permute(blocks[block_name], per_image["bin"], y, base_cols, full_cols_for_block, rng_block_perm)
    block_test_results[block_name] = res

    # BCa bootstrap CI on the R^2 difference: the image bootstrap needs one
    # frame carrying the outcome and every column either design uses, so it
    # can rebuild both the base and full design matrices from the same
    # resampled rows on every draw.
    boot_frame = pd.concat(
        [per_image[["image"]].reset_index(drop=True), full_cols_for_block.reset_index(drop=True),
         pd.Series(y, name="mean_abs_e")],
        axis=1,
    )
    block_boot_frames[block_name] = boot_frame

    def delta_r2_stat(f: pd.DataFrame, _base_cols=list(base_cols.columns), _full_cols=list(full_cols_for_block.columns)) -> float:
        yb = f["mean_abs_e"].to_numpy()
        return fit_r2(f[_full_cols], yb) - fit_r2(f[_base_cols], yb)

    boot = C.image_bootstrap(boot_frame, delta_r2_stat, rng_block_perm, image_col="image", bin_col=None, B=C.B_BOOTSTRAP)
    block_test_results[block_name]["ci_lo"] = boot.ci_lo
    block_test_results[block_name]["ci_hi"] = boot.ci_hi
    block_test_results[block_name]["ci_method"] = boot.ci_method
    print(f"{block_name:28s} delta_R2={res['delta_r2']:+.5f}  p={res['p_raw']:.5f}  "
          f"95% {boot.ci_method} CI [{boot.ci_lo:.5f}, {boot.ci_hi:.5f}]")

PUBLISHED_DELTA_R2 = {
    "rooted_dead_alike_plants": 0.05312843785054888,
    "non_rooted_plant_material": 0.029238722446982335,
    "obliquity": 0.005055020455352777,
}
for block_name, res in block_test_results.items():
    assert_matches_published(f"delta_r2[{block_name}]", res["delta_r2"], PUBLISHED_DELTA_R2[block_name])
print("Covariate-block delta R^2 values match the published values.")

# %% [markdown]
# ### Two things this covariate result cannot mean
#
# First, `rooted_dead_alike_plants == 2` is precisely the material the
# reference counted as vegetation, while one of the four prompts instructs
# the model to exclude it — a coefficient on that level is therefore partly
# a definitional artefact of that prompt's wording, not purely an
# image-difficulty effect. Second, if the annotation levels are themselves
# associated with the reference bin, part of what looks like an annotation
# effect could be bin structure showing through the annotation — reported
# here (chi-square, legal for a nominal variable) precisely so the
# within-bin permutation test above, which nets the block out from bin, is
# the number that actually distinguishes the two.

# %%
rooted2_images = per_image.loc[per_image["rooted_dead_alike_plants"] == 2, "image"]
rooted2_by_prompt = (
    base_local[base_local["image"].isin(rooted2_images)]
    .groupby("prompt")["abs_e"].agg(["mean", "count"]).rename(columns={"mean": "mean_abs_e", "count": "n_rows"})
)
rooted2_by_prompt["n_images"] = len(rooted2_images)
print(f"rooted_dead_alike_plants == 2: {len(rooted2_images)} images (descriptive only, not a test). "
      f"Mean row-level absolute error by prompt on this subset:")
print(rooted2_by_prompt)

crosstab_v3_bin = pd.crosstab(per_image["rooted_dead_alike_plants"], per_image["bin"])
chi2_v3, chi2_p_v3, _, _ = stats.chi2_contingency(crosstab_v3_bin)
print(f"\nrooted_dead_alike_plants x bin chi-square: statistic={chi2_v3:.3f}, p={chi2_p_v3:.2e} "
      f"(context for the within-bin permutation test above; some cells are small, so this is not "
      f"quoted as a precise p-value)")

assumption_rows.append({
    "question_id": "Q3", "check": "rooted_dead_alike_plants_x_bin_association", "statistic": chi2_v3, "p_value": chi2_p_v3,
    "detail": f"chi-square association between rooted_dead_alike_plants and reference bin, reported "
              f"so the within-bin permutation test (which nets the block out from bin) is the number "
              f"that distinguishes an annotation effect from bin structure showing through the "
              f"annotation. rooted==2 by-prompt mean absolute error: "
              f"{rooted2_by_prompt['mean_abs_e'].round(4).to_dict()}",
})

# %% [markdown]
# ## The five cover-gradient tests, corrected together
#
# Two trend tests and three covariate-block tests, Holm-Bonferroni across
# the five, first threshold 0.05/5 = 0.01.

# %%
gradient_pvalues = {
    "trend_signed_bias": trend_results["signed_bias"]["p_raw"],
    "trend_raw_abs_error": trend_results["raw_abs_error"]["p_raw"],
    "covariate_rooted_dead_alike_plants": block_test_results["rooted_dead_alike_plants"]["p_raw"],
    "covariate_non_rooted_plant_material": block_test_results["non_rooted_plant_material"]["p_raw"],
    "covariate_obliquity": block_test_results["obliquity"]["p_raw"],
}
assert len(gradient_pvalues) == N_COVER_GRADIENT_TESTS

holm_adj = C.holm_adjust(list(gradient_pvalues.values()), family="cover_gradient", family_size=N_COVER_GRADIENT_TESTS)
gradient_holm = dict(zip(gradient_pvalues.keys(), holm_adj))

print(f"Holm-Bonferroni across the five cover-gradient tests (first threshold {HOLM_FIRST_THRESHOLD:.5f}):")
for k in gradient_pvalues:
    sig = "***" if gradient_holm[k] < 0.05 else ""
    print(f"  {k:40s} raw p={gradient_pvalues[k]:.5f}  Holm-adjusted p={gradient_holm[k]:.5f} {sig}")

for name, key in (("signed_bias", "trend_signed_bias"),
                   ("raw_abs_error", "trend_raw_abs_error")):
    holm_p = gradient_holm[key]
    z_obs = trend_results[name]["z_obs"]
    trend_results[name]["direction"] = (
        ("increasing" if z_obs > 0 else "decreasing") if holm_p < 0.05 else "null"
    )

PUBLISHED_TREND_DIRECTION = {
    "signed_bias": "decreasing", "raw_abs_error": "increasing",
}
for name, direction in PUBLISHED_TREND_DIRECTION.items():
    assert trend_results[name]["direction"] == direction, (
        f"direction for {name} does not match the published value: "
        f"{trend_results[name]['direction']} vs {direction}"
    )
print("Trend-test directions under the Holm-adjusted threshold match the published values.")

PUBLISHED_COVARIATE_SIGNIFICANT = {
    "covariate_rooted_dead_alike_plants": True,
    "covariate_non_rooted_plant_material": True,
    "covariate_obliquity": True,
}
for key, significant in PUBLISHED_COVARIATE_SIGNIFICANT.items():
    assert (gradient_holm[key] < 0.05) == significant, (
        f"significance verdict for {key} does not match the published value"
    )
print("Covariate-block significance verdicts under the Holm-adjusted threshold match the published values.")

# %% [markdown]
# ## Monotonicity of `non_rooted_plant_material`, within bin
#
# `non_rooted_plant_material` is ordinal, so it is legitimate to ask whether
# its effect on error is monotone via a rank method — a Jonckheere-Terpstra
# test of absolute error across its three ordered levels, run **within
# bin** and then combined into one stratified statistic (summing each bin's
# J and its null variance before standardising), so this does not simply
# re-detect the pooled bin trend through the annotation's own correlation
# with bin. Descriptive — not one of the five cover-gradient tests.

# %%
per_image["non_rooted_ordinal"] = per_image["non_rooted_plant_material"].astype(int)

nonrooted_within_bin_rows = []
J_total, meanJ_total, varJ_total = 0.0, 0.0, 0.0
for lbl in C.BIN_LABELS:
    sub = per_image[per_image["bin"] == lbl]
    values_b = sub["mean_abs_e"].to_numpy()
    groups_b = sub["non_rooted_ordinal"].to_numpy()
    levels_present = sorted(set(groups_b))
    if len(levels_present) < 2 or len(sub) < 2:
        nonrooted_within_bin_rows.append({
            "bin": lbl, "n_images": len(sub), "levels_present": levels_present,
            "J": np.nan, "mean_J": np.nan, "var_J": np.nan, "z": np.nan,
            "note": "fewer than 2 levels or fewer than 2 images in this bin; excluded from the "
                    "stratified combination",
        })
        continue
    core_b = C.jonckheere_terpstra_statistic(values_b, groups_b, levels_present, variance_method="textbook")
    J_b, mean_J_b, var_J_b = core_b["J"], core_b["mean_J"], core_b["var_J"]
    z_b = (J_b - mean_J_b) / np.sqrt(var_J_b) if var_J_b > 0 else np.nan
    nonrooted_within_bin_rows.append({
        "bin": lbl, "n_images": len(sub), "levels_present": levels_present,
        "J": J_b, "mean_J": mean_J_b, "var_J": var_J_b, "z": z_b, "note": "",
    })
    J_total += J_b
    meanJ_total += mean_J_b
    varJ_total += var_J_b

z_nonrooted_within_bin = (J_total - meanJ_total) / np.sqrt(varJ_total) if varJ_total > 0 else np.nan
nonrooted_within_bin_df = pd.DataFrame(nonrooted_within_bin_rows)
print(f"stratified combination across bins: z = {z_nonrooted_within_bin:+.3f}")

z_nonrooted_pooled, _, _ = C.jonckheere_terpstra_z(
    per_image["mean_abs_e"].to_numpy(), per_image["non_rooted_ordinal"].to_numpy(),
    sorted(per_image["non_rooted_ordinal"].unique()), variance_method="textbook",
)
print(f"(for comparison only — confounded with bin, not used for any claim: pooled-over-bins "
      f"z = {z_nonrooted_pooled:+.3f})")

assert_matches_published("non_rooted_within_bin_z", z_nonrooted_within_bin, 14.148406615165301)
print("Within-bin stratified test statistic on non_rooted_plant_material matches the published value.")

assumption_rows.append({
    "question_id": "Q3", "check": "non_rooted_plant_material_within_bin_monotonicity", "statistic": z_nonrooted_within_bin,
    "p_value": np.nan,
    "detail": f"stratified (within-bin, then combined) Jonckheere-Terpstra test on "
              f"non_rooted_plant_material levels; descriptive, not one of the six cover-gradient "
              f"tests; pooled-over-bins z (confounded with bin, not used for any claim) = "
              f"{z_nonrooted_pooled:+.3f} for comparison; "
              f"per-bin breakdown: {nonrooted_within_bin_df.to_dict(orient='records')}",
})

# %% [markdown]
# ## Covariate effects by model and by prompt
#
# The pooled covariate model above averages every coefficient over six
# models and four prompts. Whether the material costs the same error to
# every model, and under every prompt, is a separate question the pooled
# fit cannot answer. This block re-estimates the **same bin-adjusted
# coefficient** the pooled model reports — an OLS coefficient net of the
# four bin dummies and the other two covariate blocks, not a raw pooled
# mean difference — **per model** (6) and **per prompt** (4), each on that
# configuration's own per-image absolute error, with image-bootstrap BCa
# intervals. Every row in this table is the same estimand: `bin_adjusted_ols
# _coefficient`, matching the column the pooled model reports in the
# covariate-model table above, so the two tables can be read side by side
# without the same term name carrying two different quantities.
#
# **Descriptive, uncorrected — no p-value, no correction.** Nothing here is
# a test.
#
# **The note that governs how this table may be read.** Every one of the
# per-model and per-prompt estimates for the two `rooted_dead_alike_plants`
# contrasts is built from **the same 94 images** with
# `rooted_dead_alike_plants == 1` — that group is fixed by the annotation,
# not by model or prompt — so the six per-model coefficients are six
# different models' readings of the same 94 photographs, not six
# independent samples. Two things follow, both binding: **the spread across
# models is not sampling variation**, and **no difference between two
# per-model coefficients is a test** — there is no test here, and the two
# estimates are not independent. `n_rooted_1 = 94` and
# `paired_views_of_one_frame = True` are written on every row so this
# constraint travels with the data, not only with this notebook's prose.

# %%
n_rooted_1_check = int((per_image["rooted_dead_alike_plants"] == 1).sum())
assert n_rooted_1_check == 94, f"expected 94 rooted_1 images, found {n_rooted_1_check}"

CONTRAST_SPECS = [
    ("rooted_dead_alike_plants", 1, "rooted_1"),
    ("rooted_dead_alike_plants", 2, "rooted_2"),
    ("non_rooted_plant_material", 1, "nonrooted_1"),
    ("non_rooted_plant_material", 2, "nonrooted_2"),
]


def bin_adjusted_covariate_coefficients(frame: pd.DataFrame, rng: np.random.Generator) -> dict:
    """The bin-adjusted OLS coefficient on each of the four annotation dummy
    columns and on standardised obliquity, fit on this configuration's own
    per-image absolute error, controlling for bin and the other two
    covariate blocks — the identical design the pooled model uses, refit on
    the configuration's own rows so every coefficient is specific to it.
    This is the same estimand the pooled `Q3_covariate_models.csv` table
    reports (`bin_adjusted_ols_coefficient`), not the raw, bin-unadjusted
    mean difference between two groups — the two are not interchangeable,
    because the annotation levels are themselves associated with cover bin.
    """
    sub = frame.copy()
    sub_bin_dummies = pd.get_dummies(sub["bin"], prefix="bin", drop_first=True).astype(float)
    sub_v3 = pd.get_dummies(sub["rooted_dead_alike_plants"].astype(int), prefix="rooted").astype(float)
    sub_v3 = sub_v3.drop(columns=[c for c in sub_v3.columns if c.endswith("_0")])
    sub_v4 = pd.get_dummies(sub["non_rooted_plant_material"].astype(int), prefix="nonrooted").astype(float)
    sub_v4 = sub_v4.drop(columns=[c for c in sub_v4.columns if c.endswith("_0")])
    obl_z = (sub["obliquity"] - sub["obliquity"].mean()) / sub["obliquity"].std()
    design_sub = pd.concat([sub_bin_dummies, sub_v3, sub_v4], axis=1)
    design_sub["obliquity_z"] = obl_z.values
    for missing_col in ["rooted_1", "rooted_2", "nonrooted_1", "nonrooted_2"]:
        if missing_col not in design_sub.columns:
            design_sub[missing_col] = 0.0
    design_sub = design_sub[[c for c in sub_bin_dummies.columns] + ["rooted_1", "rooted_2",
                                                                      "nonrooted_1", "nonrooted_2", "obliquity_z"]]

    y_sub = sub["mean_abs_e"].to_numpy()
    X_sub = sm.add_constant(design_sub.astype(float).to_numpy())
    coef_names_sub = ["const"] + list(design_sub.columns)
    fit_sub = sm.OLS(y_sub, X_sub).fit()
    points = {name: float(fit_sub.params[i]) for i, name in enumerate(coef_names_sub)}

    boot_frame = pd.concat(
        [sub[["image"]].reset_index(drop=True), design_sub.reset_index(drop=True),
         pd.Series(y_sub, name="mean_abs_e")],
        axis=1,
    )

    results = {}
    for i, name in enumerate(coef_names_sub):
        if name == "const":
            continue

        def coef_stat(f: pd.DataFrame, _i=i) -> float:
            Xf = sm.add_constant(f[design_sub.columns].astype(float).to_numpy())
            return float(sm.OLS(f["mean_abs_e"].to_numpy(), Xf).fit().params[_i])

        boot = C.image_bootstrap(boot_frame, coef_stat, rng, image_col="image", bin_col=None, B=C.B_BOOTSTRAP)
        results[name] = {"estimate": points[name], "ci_lo": boot.ci_lo, "ci_hi": boot.ci_hi,
                          "ci_method": boot.ci_method, "n": len(sub)}
    return results


by_configuration_rows = []
for axis, levels in (("model", C.MODELS), ("prompt", sorted(base_local["prompt"].unique()))):
    for level_value in levels:
        sub_bl = base_local[base_local[axis] == level_value]
        per_img_sub = sub_bl.groupby("image", as_index=False).agg(
            reference=("reference", "first"), mean_abs_e=("abs_e", "mean"),
        )
        per_img_sub["bin"] = C.assign_bins(per_img_sub["reference"])
        per_img_sub = per_img_sub.merge(
            per_image[["image", "rooted_dead_alike_plants", "non_rooted_plant_material", "obliquity"]],
            on="image", how="left",
        )
        coefs = bin_adjusted_covariate_coefficients(per_img_sub, rng_by_configuration)
        for covariate, level, contrast_name in CONTRAST_SPECS:
            res = coefs[contrast_name]
            by_configuration_rows.append({
                "question_id": "Q3", "axis": axis, "configuration": level_value,
                "covariate": covariate, "contrast": contrast_name,
                "estimand": "bin_adjusted_ols_coefficient",
                "estimate": res["estimate"], "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
                "ci_method": res["ci_method"], "n_images": res["n"],
                "n_rooted_1": 94, "paired_views_of_one_frame": True,
                "posthoc": True, "family": "none (descriptive, post-hoc)",
            })
        obl_res = coefs["obliquity_z"]
        by_configuration_rows.append({
            "question_id": "Q3", "axis": axis, "configuration": level_value,
            "covariate": "obliquity", "contrast": "obliquity_z",
            "estimand": "bin_adjusted_ols_coefficient",
            "estimate": obl_res["estimate"], "ci_lo": obl_res["ci_lo"], "ci_hi": obl_res["ci_hi"],
            "ci_method": obl_res["ci_method"], "n_images": obl_res["n"],
            "n_rooted_1": 94, "paired_views_of_one_frame": True,
            "posthoc": True, "family": "none (descriptive, post-hoc)",
        })

covariate_by_configuration_df = pd.DataFrame(by_configuration_rows)
covariate_by_configuration_df["note"] = (
    "Every row is the bin-adjusted OLS coefficient (net of the four bin dummies and the other two "
    "covariate blocks), the same estimand Q3_covariate_models.csv reports pooled -- never the raw, "
    "bin-unadjusted mean difference between two groups. Every row built from the rooted_1 contrast "
    "uses the same 94 images (rooted_dead_alike_plants == 1 is fixed by the annotation, independent "
    "of model or prompt); the per-model and per-prompt estimates are paired views of one 1,155-image "
    "frame, not independent estimates. No sentence may treat the spread across models or prompts as "
    "sampling variation, and no difference between two per-model or per-prompt coefficients is a test."
)

# The pooled bin-adjusted coefficient (Q3_covariate_models.csv) must be
# reproducible as a weighted combination of the same design fit on the
# whole frame -- checked directly by refitting the pooled model here with
# the identical helper used per configuration, rather than trusting that
# the two code paths agree by inspection.
_pooled_check = bin_adjusted_covariate_coefficients(per_image, np.random.default_rng(SEED + 999))
for contrast_name in ("rooted_1", "rooted_2", "nonrooted_1", "nonrooted_2", "obliquity_z"):
    assert_matches_published(f"pooled_check[{contrast_name}]", _pooled_check[contrast_name]["estimate"],
                              PUBLISHED_OLS_COEF[contrast_name])
print("The per-configuration estimator reproduces the pooled bin-adjusted coefficients exactly when "
      "run on the whole frame, confirming the per-configuration rows are the same estimand as the "
      "pooled covariate model.")

print(f"Q3_covariate_by_configuration.csv: {len(covariate_by_configuration_df)} rows "
      f"({covariate_by_configuration_df['axis'].nunique()} axes x "
      f"{covariate_by_configuration_df.groupby('axis')['configuration'].nunique().to_dict()} levels x 5 contrasts)")
covariate_by_configuration_df.pivot_table(
    index=["axis", "configuration"], columns="contrast", values="estimate",
).round(3)

# %% [markdown]
# ## The obliquity-cover association — a limitation, not a result
#
# Obliquity is not independent of cover on this frame, and that fact belongs
# here as a constraint on what the pooled obliquity coefficient means, not
# as a finding about photographs. This block computes exactly two things:
# the tie-corrected Spearman correlation between obliquity and reference
# cover with its BCa confidence interval, and the five per-bin obliquity
# means.
#
# **Why a finer question — does obliquity cost more error specifically in
# dense scenes? — cannot be answered here, and the reason is not the one a
# reader would guess.** It is not collinearity: the collinearity check above
# gives `obliquity_z` a variance inflation factor of 1.22, far under any
# threshold that would flag it as inseparable from the bin dummies, so the
# model is well identified. The problem is what the pooled coefficient
# **is**: a within-bin average dominated by the 933 images in the bottom
# bin, because that is where almost all of the obliquity variation the fit
# can see actually sits. Answering the finer question needs obliquity
# variation *inside* bins holding 46, 38 and 30 images, and three bins of
# that size cannot supply it. No obliquity-by-material cross-tabulation is
# added here — it would invite a three-way reading this design cannot
# support and would turn a one-line constraint into an analysis.

# %%
obliquity_bin_means = per_image.groupby("bin")["obliquity"].mean().reindex(C.BIN_LABELS)
print("mean obliquity per bin:")
print(obliquity_bin_means)

PUBLISHED_OBLIQUITY_BIN_MEANS = {"0-20": 0.0309, "20-40": 0.0478, "40-60": 0.0580, "60-80": 0.0660, "80-100": 0.0848}
for lbl, expected in PUBLISHED_OBLIQUITY_BIN_MEANS.items():
    assert_matches_published(f"obliquity_mean[{lbl}]", obliquity_bin_means[lbl], expected, atol=0.001)
print("Per-bin obliquity means match the published values.")

rho_obliquity, rho_obliquity_p = C.spearman_tie_corrected(
    per_image["obliquity"].to_numpy(), per_image["reference"].to_numpy(),
)
assert_matches_published("obliquity_cover_rho", rho_obliquity, 0.2523, atol=0.001)
print(f"Spearman correlation between obliquity and reference cover = {rho_obliquity:.4f} "
      f"(p={rho_obliquity_p:.2e}), matching the published value")


def obliquity_rho_stat(f: pd.DataFrame) -> float:
    rho, _ = C.spearman_tie_corrected(f["obliquity"].to_numpy(), f["reference"].to_numpy())
    return rho


rho_boot = C.image_bootstrap(
    per_image[["image", "obliquity", "reference", "bin"]], obliquity_rho_stat, rng_obliquity,
    image_col="image", bin_col=None, B=C.B_BOOTSTRAP,
)
print(f"95% BCa CI on the correlation: [{rho_boot.ci_lo:.4f}, {rho_boot.ci_hi:.4f}] (method: {rho_boot.ci_method})")

vif_obliquity = float(k7["per_column_vif"]["obliquity_z"])
print(f"Variance inflation factor for obliquity in the pooled covariate design = {vif_obliquity:.3f} "
      f"(not the binding constraint here — see the prose above)")

obliquity_structure_df = pd.DataFrame([{
    "question_id": "Q3",
    "spearman_rho": rho_obliquity, "spearman_rho_ci_lo": rho_boot.ci_lo, "spearman_rho_ci_hi": rho_boot.ci_hi,
    "spearman_rho_ci_method": rho_boot.ci_method,
    **{f"mean_obliquity_bin_{lbl}": float(obliquity_bin_means[lbl]) for lbl in C.BIN_LABELS},
    "vif_obliquity": vif_obliquity,
    "note": (
        "Reported as a limitation on what the pooled obliquity coefficient means, not as a result. "
        "Denser quadrats were photographed from slightly further off square, monotonically across all "
        "five bins. A finer question -- does obliquity cost more error specifically in dense scenes -- "
        "cannot be answered here: it is not collinearity (the variance inflation factor is 1.22, so the "
        "coefficient is cleanly separated from the bin dummies), but that the pooled coefficient is a "
        "within-bin average dominated by the 933 images in the sparse bottom bin, and answering the "
        "finer question needs obliquity variation inside bins holding 46, 38 and 30 images, which this "
        "design cannot supply. No obliquity-by-material cross-tabulation is added."
    ),
}])
obliquity_structure_df

# %% [markdown]
# ## Charts

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

ax = axes[0, 0]
xpos = np.arange(len(C.BIN_LABELS))
maes = [model_bmae.per_bin_mae[b] for b in C.BIN_LABELS]
lo = [mae_ci_lo[b] for b in C.BIN_LABELS]
hi = [mae_ci_hi[b] for b in C.BIN_LABELS]
err = [[m - l for m, l in zip(maes, lo)], [h - m for m, h in zip(maes, hi)]]
ax.bar(xpos, maes, yerr=err, capsize=4, color="steelblue")
for x, b in zip(xpos, C.BIN_LABELS):
    ax.text(x, maes[xpos.tolist().index(x)] + err[1][xpos.tolist().index(x)] + 0.3,
            f"n={per_bin_n[b]}", ha="center", fontsize=9)
ax.set_xticks(xpos); ax.set_xticklabels(C.BIN_LABELS)
ax.set_xlabel("reference cover bin (cover points)")
ax.set_ylabel("mean absolute error (cover points)")
ax.set_title("Per-bin MAE, pooled over 24 configurations\n(95% BCa CI; n = images per bin)")

ax = axes[0, 1]
bias_vals = [perbin_df.loc[perbin_df["bin"] == b, "mean_signed_bias"].iloc[0] for b in C.BIN_LABELS]
bias_lo = [perbin_df.loc[perbin_df["bin"] == b, "mean_signed_bias_ci_lo"].iloc[0] for b in C.BIN_LABELS]
bias_hi = [perbin_df.loc[perbin_df["bin"] == b, "mean_signed_bias_ci_hi"].iloc[0] for b in C.BIN_LABELS]
err2 = [[m - l for m, l in zip(bias_vals, bias_lo)], [h - m for m, h in zip(bias_vals, bias_hi)]]
ax.axhline(0, color="grey", linewidth=0.8)
ax.errorbar(xpos, bias_vals, yerr=err2, fmt="o-", color="darkorange", capsize=4)
ax.set_xticks(xpos); ax.set_xticklabels(C.BIN_LABELS)
ax.set_xlabel("reference cover bin (cover points)")
ax.set_ylabel("mean signed bias, prediction - reference (cover points)")
ax.set_title("Signed bias per bin\n(positive = over-prediction; look for a sign reversal)")

ax = axes[1, 0]
deltas = [block_test_results[b]["delta_r2"] for b in ("rooted_dead_alike_plants", "non_rooted_plant_material", "obliquity")]
cis_lo = [block_test_results[b]["ci_lo"] for b in ("rooted_dead_alike_plants", "non_rooted_plant_material", "obliquity")]
cis_hi = [block_test_results[b]["ci_hi"] for b in ("rooted_dead_alike_plants", "non_rooted_plant_material", "obliquity")]
err3 = [[d - l for d, l in zip(deltas, cis_lo)], [h - d for d, h in zip(deltas, cis_hi)]]
xpos2 = np.arange(3)
ax.bar(xpos2, deltas, yerr=err3, capsize=4, color="indianred")
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_xticks(xpos2); ax.set_xticklabels(["rooted_dead\n_alike_plants", "non_rooted_\nplant_material", "obliquity"])
ax.set_ylabel("change in R^2 (block contribution, net of bin)")
ax.set_title("Covariate block contribution\n(within-bin permutation, 95% BCa bootstrap CI)")

ax = axes[1, 1]
obl_means = [obliquity_bin_means[b] for b in C.BIN_LABELS]
ax.plot(xpos, obl_means, "s-", color="seagreen")
ax.set_xticks(xpos); ax.set_xticklabels(C.BIN_LABELS)
ax.set_xlabel("reference cover bin (cover points)")
ax.set_ylabel("mean obliquity (unitless, 0 = perfect square)")
ax.set_title(f"Obliquity by cover bin — a limitation, not a result\n"
             f"(Spearman correlation={rho_obliquity:.3f}, 95% BCa CI [{rho_boot.ci_lo:.3f}, {rho_boot.ci_hi:.3f}])")

plt.tight_layout()
fig_path = RENDERED_DIR / "Q3_bins_charts.png"
plt.savefig(fig_path, dpi=110)
plt.show()
print(f"chart saved to {fig_path}")

# %% [markdown]
# **What to look for.** Panel 1 shows mean absolute error rising with cover
# — but the topmost bin's estimate rests on only 30 images against 933 in
# the bottom bin, so its interval is much wider, and that width is part of
# the honest picture, not a footnote to it. Panel 2 shows the sign of the
# error reversing: models over-predict near-bare quadrats and under-predict
# dense ones. Panel 3 shows how much of the per-image error each covariate
# block explains once bin is already accounted for. Panel 4 is the
# limitation: obliquity rises with cover across all five bins, which is why
# a single coefficient describing "the effect of obliquity" is, in practice,
# mostly describing the 933 images in the bottom bin.

# %% [markdown]
# ## Result
#
# **The per-bin profile.** Mean absolute error runs from
# {mae at 0-20} up to {mae at 80-100} cover points across the five bins
# (Jonckheere-Terpstra test, increasing, Holm-adjusted p reported below),
# and the sign of the mean bias reverses over the same range: models
# over-predict near-bare quadrats and under-predict dense ones
# (Jonckheere-Terpstra test, decreasing). Both trends are reported as
# what they are — a rank-ordered pattern across five bins, on a metric
# that is bounded below by the reference and therefore expected to rise
# with it — and neither is read as a verdict on model behaviour beyond
# that profile; this notebook does not contain an instrument that would
# license such a verdict.
#
# **The three covariate blocks.** Each adds detectable explanatory power
# to per-image error, net of cover bin: rooted dead look-alike plants,
# non-rooted plant material, and obliquity all clear the Holm-adjusted
# threshold across the five cover-gradient tests. The coefficient on
# `rooted_dead_alike_plants == 2` is read with a construct-validity
# caveat: that level is precisely the material the reference counted as
# vegetation while one of the four prompts instructs a model to exclude
# it, so part of the coefficient reflects that prompt's own wording.
#
# **Obliquity's two streams do not agree, and both are reported rather
# than picking one.** The within-bin permutation test finds obliquity's
# block contribution significant; the OLS coefficient's own bootstrap
# interval straddles zero. The permutation test asks whether the block
# improves fit at all, net of bin; the interval asks whether the linear
# slope is distinguishable from zero under resampling — a block can add
# detectable explanatory power while its single coefficient is not
# tightly pinned down.
#
# **Covariate effects by model and by prompt** are read as six, and
# separately four, views of the same 94 images carrying
# `rooted_dead_alike_plants == 1` — not as independent estimates. The
# spread across models is not sampling variation, and no pairwise
# difference in this table is a test.
#
# **The obliquity-cover association is a limitation, not a result.**
# Denser quadrats were photographed from slightly further off square,
# monotonically across all five bins. This is not a collinearity problem
# — the variance inflation factor for obliquity is 1.22 — it is that the
# pooled coefficient is a within-bin average dominated by the 933 images
# in the sparse bottom bin, and a finer question about dense scenes needs
# obliquity variation inside bins holding 46, 38 and 30 images that this
# design cannot supply.
#
# **What this does not establish.** The Jonckheere-Terpstra test asks
# about stochastic ordering across bins, not about means, and not about
# the shape or size of the change at any one cover level. Independence
# between images is assumed, not testable.

# %%
from IPython.display import Markdown, display

result_summary = {
    "raw_abs_error_direction": trend_results["raw_abs_error"]["direction"],
    "raw_abs_error_holm_p": gradient_holm["trend_raw_abs_error"],
    "signed_bias_direction": trend_results["signed_bias"]["direction"],
    "signed_bias_holm_p": gradient_holm["trend_signed_bias"],
    "rooted_delta": block_test_results["rooted_dead_alike_plants"]["delta_r2"],
    "rooted_holm_p": gradient_holm["covariate_rooted_dead_alike_plants"],
    "nonrooted_delta": block_test_results["non_rooted_plant_material"]["delta_r2"],
    "nonrooted_holm_p": gradient_holm["covariate_non_rooted_plant_material"],
    "obliquity_delta": block_test_results["obliquity"]["delta_r2"],
    "obliquity_holm_p": gradient_holm["covariate_obliquity"],
    "obliquity_ols_ci_lo": float(covariate_model_df.set_index("term").loc["obliquity_z", "ols_ci_lo"]),
    "obliquity_ols_ci_hi": float(covariate_model_df.set_index("term").loc["obliquity_z", "ols_ci_hi"]),
    "mae_bin0": perbin_df.loc[perbin_df["bin"] == "0-20", "mae"].iloc[0],
    "mae_bin4": perbin_df.loc[perbin_df["bin"] == "80-100", "mae"].iloc[0],
    "bias_bin0": perbin_df.loc[perbin_df["bin"] == "0-20", "mean_signed_bias"].iloc[0],
    "bias_bin4": perbin_df.loc[perbin_df["bin"] == "80-100", "mean_signed_bias"].iloc[0],
}
for k, v in result_summary.items():
    print(f"{k}: {v}")

# %%
_result_md = f"""
**The per-bin profile.** Mean absolute error runs
**{result_summary['mae_bin0']:.2f} → {result_summary['mae_bin4']:.2f} cover
points** from the 0-20 to the 80-100 bin
(Jonckheere-Terpstra direction: **{result_summary['raw_abs_error_direction']}**,
Holm-adjusted p = {result_summary['raw_abs_error_holm_p']:.4f}), and the sign
of the mean bias reverses over the same range, from
**{result_summary['bias_bin0']:+.2f} to {result_summary['bias_bin4']:+.2f}**
points (Jonckheere-Terpstra direction:
**{result_summary['signed_bias_direction']}**, Holm-adjusted p =
{result_summary['signed_bias_holm_p']:.4f}) — over-prediction on near-bare
quadrats, under-prediction on dense ones. Both are rank-ordered patterns
across the five bins on a metric that is bounded below by the reference and
therefore expected to rise with it; neither is read here as a verdict on
model behaviour beyond that profile.

**The three covariate blocks**, contribution to explained variance net of
bin, via the within-bin permutation test: `rooted_dead_alike_plants`
change in R² = {result_summary['rooted_delta']:.4f} (Holm-adjusted p =
{result_summary['rooted_holm_p']:.4f}), `non_rooted_plant_material` change
in R² = {result_summary['nonrooted_delta']:.4f} (Holm-adjusted p =
{result_summary['nonrooted_holm_p']:.4f}), obliquity change in R² =
{result_summary['obliquity_delta']:.4f} (Holm-adjusted p =
{result_summary['obliquity_holm_p']:.4f}). All three clear the Holm-adjusted
threshold across the five cover-gradient tests. The
`rooted_dead_alike_plants == 2` coefficient is read with the
construct-validity caveat above — it is partly a definitional artefact of
one prompt's exclusion instruction, not purely an image-difficulty effect.

**Obliquity's two streams do not agree, and both are reported rather than
picking one.** The within-bin permutation test finds obliquity's
contribution significant; the OLS image-bootstrap BCa interval on the
standardised-obliquity coefficient in the same design is
[{result_summary['obliquity_ols_ci_lo']:.4f}, {result_summary['obliquity_ols_ci_hi']:.4f}],
which straddles zero. The permutation test asks whether the block improves
fit at all, net of bin; the interval asks whether the linear slope is
distinguishable from zero under resampling — a block can add detectable
explanatory power while its single linear coefficient is not tightly
pinned down. Neither stream is discarded in favour of the other.

**Covariate effects by model and by prompt** (`Q3_covariate_by_configuration.csv`)
are the same bin-adjusted coefficient the pooled model reports, re-estimated
per model and per prompt, and are read as six, and separately four, views
of the same 94 images carrying `rooted_dead_alike_plants == 1` — not as
independent estimates. The spread across models is not sampling variation
and no pairwise difference in this table is a test.

**The obliquity-cover association is a limitation, not a result**
(`Q3_obliquity_structure.csv`). Denser quadrats were photographed from
slightly further off square, monotonically across all five bins (Spearman
correlation = {rho_obliquity:.4f}, 95% BCa CI
[{rho_boot.ci_lo:.4f}, {rho_boot.ci_hi:.4f}]). This is not a collinearity
problem — the variance inflation factor for obliquity is
{vif_obliquity:.3f} — it is that the pooled coefficient is a within-bin
average dominated by the 933 images in the sparse bottom bin, and a finer
question about dense scenes needs obliquity variation inside bins holding
46, 38 and 30 images that this design cannot supply.

**What this does not establish.** The Jonckheere-Terpstra test asks about
stochastic ordering across bins, not about means, and not about the shape
or size of the change at any one cover level. Independence between images
is assumed, not testable.
"""
display(Markdown(_result_md))

# %% [markdown]
# ## Signed error by rooted dead look-alike level
#
# `Q3_covariate_models.csv` already shows that rooted dead look-alike plants
# raise **absolute** error net of cover bin, and that the coefficient is
# roughly twice as large when the plants were recorded present but *not*
# counted as reference cover (`rooted_1`, level 1) as when they were counted
# (`rooted_2`, level 2). Absolute error cannot say which direction that extra
# error runs in, and it cannot say whether the pattern looks the same under
# every prompt's own wording. Both questions need a signed quantity, so this
# block reports the mean **signed** error — prediction minus reference,
# positive meaning over-prediction, the same convention as every other signed
# quantity in this study — at each of the three annotation levels, pooled and
# broken out by prompt, alongside the mean absolute error and the mean
# reference cover at that level. The four prompt scopes are four readings of
# the same 1,155 images, not four independent samples, so their intervals
# move together and a difference between two prompt scopes is not a test.
#
# **This is descriptive, not a sixth cover-gradient test.** No p-value is
# computed and no correction applies; the interval is there to show how
# precisely each mean is pinned down, not to support a significance claim.
# The cover bin is **not** held fixed here — these are marginal means over
# whatever images carry that annotation level, unlike the bin-adjusted
# coefficients in `Q3_covariate_models.csv` and
# `Q3_covariate_by_configuration.csv`. `mean_reference_cover` is reported on
# every row precisely so a reader can see that this is a marginal comparison:
# level 2 (rooted dead look-alike material the reference counted as cover)
# tends to sit at different cover on the gradient than levels 0 and 1, and
# that placement, not just the annotation itself, is part of why its mean
# error differs.
#
# The pooled scope uses `per_image`, whose per-image value already averages
# over the 24 model x prompt configurations. Each prompt's scope instead
# averages over that prompt's six models only, built fresh from `base_local`,
# so that no photograph is counted twice inside one interval and the unit
# stays the image throughout.

# %%
ROOTED_LEVEL_LABELS = {
    0: "absent",
    1: "present, not counted as cover",
    2: "present, counted as cover",
}
# Named explicitly, in the study's own prompt order, rather than sorted --
# alphabetical order would put Detailed first and Short last. The assertion
# guards against a silent reorder if the data ever gains or drops a prompt.
PROMPT_NAMES = ("Short", "Point-Hint", "Grid-Overlay", "Detailed")
assert set(PROMPT_NAMES) == set(base_local["prompt"].unique()), (
    f"PROMPT_NAMES {PROMPT_NAMES} does not match the prompts present in base_local: "
    f"{sorted(base_local['prompt'].unique())}"
)


def rooted_level_scope_frame(scope: str) -> pd.DataFrame:
    """Per-image frame for one scope: 'pooled' reuses per_image directly
    (already averaged over all 24 configurations); a prompt name rebuilds a
    per-image mean from that prompt's six model rows only, so the image
    remains the unit and no photograph enters an interval twice.
    """
    if scope == "pooled":
        return per_image[["image", "reference", "mean_e", "mean_abs_e", "rooted_dead_alike_plants"]].copy()
    sub_bl = base_local[base_local["prompt"] == scope]
    out = sub_bl.groupby("image", as_index=False).agg(
        reference=("reference", "first"), mean_e=("e", "mean"), mean_abs_e=("abs_e", "mean"),
    )
    out = out.merge(d2[["image", "rooted_dead_alike_plants"]], on="image", how="left", validate="one_to_one")
    return out


def mean_stat_factory(col: str) -> Callable[[pd.DataFrame], float]:
    def _stat(f: pd.DataFrame) -> float:
        return float(f[col].mean())
    return _stat


rooted_level_rows = []
for scope in ("pooled", *PROMPT_NAMES):
    scope_frame = rooted_level_scope_frame(scope)
    for level in (0, 1, 2):
        sub = scope_frame[scope_frame["rooted_dead_alike_plants"] == level]
        n_images = len(sub)
        if n_images >= 2:
            signed_boot = C.image_bootstrap(
                sub[["image", "mean_e"]], mean_stat_factory("mean_e"), rng_rooted_level,
                image_col="image", bin_col=None, B=C.B_BOOTSTRAP,
            )
            abs_boot = C.image_bootstrap(
                sub[["image", "mean_abs_e"]], mean_stat_factory("mean_abs_e"), rng_rooted_level,
                image_col="image", bin_col=None, B=C.B_BOOTSTRAP,
            )
            signed_est, signed_lo, signed_hi, signed_method = (
                signed_boot.estimate, signed_boot.ci_lo, signed_boot.ci_hi, signed_boot.ci_method,
            )
            abs_est, abs_lo, abs_hi, abs_method = (
                abs_boot.estimate, abs_boot.ci_lo, abs_boot.ci_hi, abs_boot.ci_method,
            )
        else:
            signed_est = float(sub["mean_e"].mean()) if n_images else np.nan
            abs_est = float(sub["mean_abs_e"].mean()) if n_images else np.nan
            signed_lo = signed_hi = abs_lo = abs_hi = np.nan
            signed_method = abs_method = "not computed (n < 2 images)"

        rooted_level_rows.append({
            "question_id": "Q3", "scope": scope, "rooted_level": level,
            "rooted_level_label": ROOTED_LEVEL_LABELS[level],
            "n_images": n_images,
            "mean_signed_error": signed_est, "mean_signed_error_ci_lo": signed_lo,
            "mean_signed_error_ci_hi": signed_hi, "mean_signed_error_ci_method": signed_method,
            "mean_abs_error": abs_est, "mean_abs_error_ci_lo": abs_lo,
            "mean_abs_error_ci_hi": abs_hi, "mean_abs_error_ci_method": abs_method,
            "mean_reference_cover": float(sub["reference"].mean()) if n_images else np.nan,
            "part_of_cover_gradient_tests": False,
            "note": "Descriptive interval, not a test -- no p-value, no multiplicity correction. The "
                    "reference cover bin is not held fixed here; this is a marginal mean over whatever "
                    "images carry this annotation level in this scope. The bin-adjusted coefficient on "
                    "rooted_dead_alike_plants is in Q3_covariate_models.csv. mean_reference_cover is "
                    "reported so a reader can see how far this level sits from the other two on the "
                    "cover gradient itself.",
        })

rooted_level_df = pd.DataFrame(rooted_level_rows)

# Each scope's three level counts must sum to the full 1,155-image frame: the
# annotation is complete (0% missing, checked at load time) and every image
# in a scope carries exactly one of the three levels, so a mismatch here
# would mean a level was dropped somewhere in the per-scope frame above.
for scope in ("pooled", *PROMPT_NAMES):
    scope_total = int(rooted_level_df.loc[rooted_level_df["scope"] == scope, "n_images"].sum())
    assert scope_total == 1155, f"scope {scope!r}: rooted-level counts sum to {scope_total}, expected 1155"
print("Every scope's three rooted-level image counts sum to 1,155 (pooled and each of the four prompts).")

n_rooted_level_percentile_fallback = int(
    (rooted_level_df["mean_signed_error_ci_method"] == "percentile (BCa fallback)").sum()
    + (rooted_level_df["mean_abs_error_ci_method"] == "percentile (BCa fallback)").sum()
)
assumption_rows.append({
    "question_id": "Q3", "check": "rooted_level_scope_counts_sum_to_1155", "statistic": np.nan, "p_value": np.nan,
    "detail": "pooled scope and each of the four prompt scopes: the three rooted_dead_alike_plants "
              "level counts sum to 1,155 images, confirmed by assertion above.",
})
assumption_rows.append({
    "question_id": "Q3", "check": "rooted_level_ci_method", "statistic": n_rooted_level_percentile_fallback,
    "p_value": np.nan,
    "detail": f"ci_method is recorded on every row of Q3_rooted_level_error.csv; "
              f"{n_rooted_level_percentile_fallback} of "
              f"{2 * len(rooted_level_df)} intervals (signed + absolute, across all rows) fell back "
              f"to the plain percentile interval rather than BCa.",
})

print(f"Q3_rooted_level_error.csv: {len(rooted_level_df)} rows "
      f"({rooted_level_df['scope'].nunique()} scopes x 3 rooted levels)")
rooted_level_df[["scope", "rooted_level", "rooted_level_label", "n_images",
                  "mean_signed_error", "mean_abs_error", "mean_reference_cover"]]

# %% [markdown]
# **What to look for.** Level 2 (rooted dead look-alike material the
# reference counted as cover) should sit at higher mean reference cover than
# levels 0 and 1 across every scope — that gradient placement, not the
# annotation alone, is part of why its error differs from the other two
# levels. A negative `mean_signed_error` at a level means the models
# under-predict there on average; a positive value means over-prediction. The
# absolute-error column here is the same marginal (not bin-adjusted) quantity
# and will not match `Q3_covariate_models.csv`'s bin-adjusted coefficients —
# the two are different estimands by design, as the note column on every row
# states. The pooled row and the four prompt rows at a given level all draw
# on the same 1,155 images, so reading the four prompt scopes side by side is
# reading four views of one frame, not four independent samples — no
# difference between two prompt scopes here is a test.

# %% [markdown]
# ## Writing the output CSVs

# %%
perbin_out = perbin_df.copy()
perbin_out.to_csv(RESULTS_DIR / "Q3_perbin_error.csv", index=False)

trend_rows_out = []
for name, spec in trend_results.items():
    p_holm = gradient_holm[f"trend_{name}"]
    trend_rows_out.append({
        "question_id": "Q3", "model": np.nan, "trend": name,
        "estimate": spec["z_obs"], "ci_lo": np.nan, "ci_hi": np.nan,
        "p_raw": spec["p_raw"],
        "p_holm": p_holm,
        "direction": spec["direction"],
        "note_on_scope": spec["note"],
        "part_of_cover_gradient_tests": True,
        "n_permutations": 10_000, "n_images": len(per_image),
    })

trend_tests_df = pd.DataFrame(trend_rows_out)
trend_tests_df.to_csv(RESULTS_DIR / "Q3_trend_tests.csv", index=False)

baseline_out = baseline_df.copy()
skill_ratio_out = skill_ratio_df.copy()
skill_ratio_out.insert(0, "question_id", "Q3")
skill_ratio_out["quantity"] = "skill_ratio_1_minus_mae_model_over_mae_base"
skill_ratio_out["is_oracle_baseline"] = True
skill_ratio_out["baseline"] = "B_median"
skill_ratio_out = skill_ratio_out.merge(r_tie_share_df, on="bin", suffixes=("", "_rtie"))
baseline_full = pd.concat([
    baseline_out,
    skill_ratio_out.rename(columns={"skill_ratio": "value", "n_images_rtie": "n_images_check"}),
], axis=0, ignore_index=True, sort=False)
baseline_full.to_csv(RESULTS_DIR / "Q3_baseline_skill.csv", index=False)

covariate_model_df.to_csv(RESULTS_DIR / "Q3_covariate_models.csv", index=False)

covariate_rows_out = []
for block_name in ("rooted_dead_alike_plants", "non_rooted_plant_material", "obliquity"):
    res = block_test_results[block_name]
    p_holm = gradient_holm[f"covariate_{block_name}"]
    covariate_rows_out.append({
        "question_id": "Q3", "covariate_block": block_name,
        "estimate": res["delta_r2"], "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
        "ci_method": res["ci_method"],
        "p_raw": res["p_raw"], "p_holm": p_holm,
        "part_of_cover_gradient_tests": True,
        "vif_flagged_not_separately_identified": k7["block_not_separately_identified"].get(block_name, False),
        "n_images": len(per_image),
        "note": "within-bin permutation p-value on the change in R^2 from adding this covariate "
                "block (10,000 permutations, shuffled within reference bin); the confidence "
                "interval is an image-bootstrap BCa interval on the same change in R^2.",
    })
covariate_block_tests_df = pd.DataFrame(covariate_rows_out)
covariate_block_tests_df.to_csv(RESULTS_DIR / "Q3_covariate_block_tests.csv", index=False)

covariate_by_configuration_df.to_csv(RESULTS_DIR / "Q3_covariate_by_configuration.csv", index=False)

obliquity_structure_df.to_csv(RESULTS_DIR / "Q3_obliquity_structure.csv", index=False)

rooted_level_df.to_csv(RESULTS_DIR / "Q3_rooted_level_error.csv", index=False)

# Empty-bin coverage of the headline balanced-MAE bootstrap: how often did a
# plain (non-stratified) resample of the full 1,155-image pool leave one of
# the five bins empty. The bottom bin holds 933 images and the top holds
# 30, so an empty-bin draw is expected to be extremely rare; checked rather
# than assumed.
def balanced_mae_stat(f: pd.DataFrame) -> float:
    return C.balanced_mae(f["mean_abs_e"], f["bin"]).balanced_mae


headline_bmae_boot = C.image_bootstrap(
    per_image[["image", "bin", "mean_abs_e"]], balanced_mae_stat, rng_headline_bmae,
    image_col="image", bin_col="bin", B=C.B_BOOTSTRAP,
)
assert_matches_published("balanced_mae_pooled", headline_bmae_boot.estimate, model_bmae.balanced_mae)
k4 = C.check_k4_bootstrap_bin_coverage(headline_bmae_boot.n_empty_bin_violations, C.B_BOOTSTRAP)
print(f"Empty-bin coverage of the balanced-MAE bootstrap: {k4['n_violations']}/{k4['B']} resamples "
      f"had an empty bin (share={k4['violation_share']:.4%}), switch_to_stratified={k4['switch_to_stratified']}")
assumption_rows.append({
    "question_id": "Q3", "check": "bootstrap_bin_coverage", "statistic": k4["n_violations"], "p_value": np.nan,
    "detail": f"balanced-MAE headline bootstrap; violation_share={k4['violation_share']:.4%}, "
              f"switch_to_stratified={k4['switch_to_stratified']}",
})

assumption_rows.append({
    "question_id": "Q3", "check": "pairing_complete", "statistic": np.nan, "p_value": np.nan,
    "detail": k1_result.detail,
})
assumption_checks_df = pd.DataFrame(assumption_rows)
assumption_checks_df.to_csv(RESULTS_DIR / "Q3_assumption_checks.csv", index=False)

print("wrote:")
for f in ("Q3_perbin_error.csv", "Q3_trend_tests.csv", "Q3_baseline_skill.csv",
          "Q3_covariate_models.csv", "Q3_covariate_block_tests.csv",
          "Q3_covariate_by_configuration.csv", "Q3_obliquity_structure.csv",
          "Q3_rooted_level_error.csv", "Q3_assumption_checks.csv"):
    print(f"  {RESULTS_DIR / f}")
