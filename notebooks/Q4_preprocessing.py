# %% [markdown]
# # Q4 — pre-processing
#
# Two of the three image variants sent to every model are not photographs as
# taken. Both `masked_gray` and `rectified` crop the photograph to the
# hand-annotated quadrat corners, and `rectified` additionally applies **the
# perspective correction**, mapping the cropped quadrat to a square; nothing
# further about either transform's pixel-level appearance is recorded, so
# none is asserted here. Both variants are scored against the reference for
# the **original** photograph — that is the intended design — so this
# notebook asks a narrow, well-posed question: **holding the model, prompt
# and reference fixed, does substituting a pre-processed image for the
# original one change the error?** And because absolute error is
# structurally tied to reference cover, the second half of this notebook
# asks whether that tie is itself reshaped by pre-processing — whether the
# answer to the first question changes across five ordered reference-cover
# bins.
#
# Why this might plausibly help: cropping to the annotated quadrat corners
# removes whatever is outside the quadrat, which could focus a model's
# judgement of cover on the intended region rather than the whole frame; the
# perspective correction in `rectified` could make an obliquely photographed
# quadrat's true extent easier to judge from pixels. Why it might not:
# cropping and the perspective correction both discard information
# (surrounding context, the original viewing geometry) a model could also
# have been using productively, and a vision-language model's response to
# either transform is not obvious a priori. This notebook does not resolve
# the mechanism — the images themselves are absent from this repository and
# their transformations cannot be inspected or re-derived here — it only
# measures whether the substitution moved the error, on this frame, as
# pre-processed and recorded.
#
# **The three pooled paired contrasts** — `masked_gray - base`,
# `rectified - base`, `masked_gray - rectified` — are reported as **BCa
# intervals on the difference in balanced MAE, uncorrected, carrying no
# p-value**: every comparison of balanced MAE in this notebook is read as an
# interval, one at a time, and is not multiplicity-corrected. The per-bin
# **location and spread of the variant difference** across the five ordered
# reference bins are reported alongside it, so a flat location with a
# widening spread is never mistaken for a shift in the typical effect.
# Per-bin, per-model and per-prompt breakdowns are reported with CIs and n,
# uncorrected, because they are breakdowns rather than additional hypotheses
# under test.
#
# A per-model breakdown of each pooled contrast is reported at 6 models x 2
# contrasts; this notebook adds the matching breakdown by prompt, at 4
# prompts x 2 contrasts, because there is no reason the two axes should be
# treated differently. Because the pooled, whole-frame decomposition of the
# crop and the perspective correction can obscure a per-bin pattern present
# in either step, this notebook also adds the same three-way decomposition —
# crop alone, the perspective correction alone, and the two together —
# computed separately within each of the five reference-cover bins.
#
# **The two variants are nested, not parallel arms.** Both crop to the
# hand-annotated quadrat corners; `rectified` additionally applies the
# perspective correction to the crop. So `rectified - masked_gray` isolates
# the perspective correction alone, and the three contrasts decompose
# additively: the crop step's contrast plus the perspective-correction
# step's contrast sum, exactly, to the crop-and-correct contrast — see
# `Q4_variant_nesting.csv`, with the same decomposition reported per bin.
#
# **The per-image perspective-correction effect against obliquity**, an
# image-level covariate of perspective distortion, is reported as a
# descriptive quartile breakdown across all 1,155 images: the mean change
# from the perspective correction within each of four equal-sized groups by
# obliquity score.

# %% [markdown]
# ## Setup
#
# **RNG discipline.** Every bootstrap quantity below draws from its own
# generator, offset from `SEED` by a fixed integer declared next to its
# creation — a single shared stream would mean that adding or removing any
# one resampling step silently shifts every draw after it, without changing
# a line of code that looks wrong. One consequence follows directly:
# **every point estimate below is asserted against the published values**,
# because balanced MAE, mean bias and the per-bin decomposition are
# deterministic functions of the loaded data. This cell fails if any of
# them moves, so a change to the analysis cannot silently invalidate the
# manuscript.

# %%
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _locate_project_root(start=None):
    """Resolve the project root independently of the kernel's cwd, so this
    notebook does not depend on import order to find `_common`."""
    p = Path(os.environ.get("ANALYSIS_PROJECT_ROOT", start or Path.cwd())).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "STATUS.md").is_file():
            return candidate
    raise RuntimeError(f"project root not found from {p}")


sys.path.insert(0, str(_locate_project_root() / "03_notebooks_definitive"))
import pathlib as _pl  # `_common.py` is imported from this notebook's
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))  # own directory
import _common as co

SEED = 20260911

# Each resampling quantity gets its own generator, offset from SEED by a
# fixed integer declared here and nowhere else:
#   +0  the two pooled-contrast BCa intervals (masked_gray-base, rectified-base)
#   +1  the pooled perspective-correction contrast (masked_gray-rectified)
#   +2  per-bin and per-model breakdowns
#   +3  per-prompt breakdown
#   +4  per-bin three-way decomposition (crop / perspective correction / total)
#   +5  the Maverick determinism floor (delta_Mav)
#   +6  the obliquity-quartile breakdown of the per-image rectification effect
#   +7  the perspective-correction share-of-total ratio bootstrap
#   +8  the by-axis (model, prompt) crop/correction/total breakdown
rng_e1 = np.random.default_rng(SEED + 0)
rng_e2 = np.random.default_rng(SEED + 1)
rng_breakdown = np.random.default_rng(SEED + 2)
rng_prompt = np.random.default_rng(SEED + 3)
rng_bin_decomp = np.random.default_rng(SEED + 4)
rng_maverick = np.random.default_rng(SEED + 5)
rng_j = np.random.default_rng(SEED + 6)
rng_by_axis = np.random.default_rng(SEED + 8)

ROOT = co.project_root()
RESULTS_DIR = ROOT / "05_deliverables/Repository/GitHub/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RENDERED_DIR = ROOT / "05_deliverables/Repository/GitHub/notebooks" / "rendered"
RENDERED_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def assert_reproduces(name: str, computed: float, published: float, atol: float = 1e-6) -> None:
    """Point estimates are deterministic functions of the loaded data. This
    check fails if any of them moves, so a change to the analysis cannot
    silently invalidate the published value. Interval and p-value endpoints
    are exempt by design — see the RNG-discipline note above — and this
    helper is never called on one.
    """
    if not np.isclose(computed, published, atol=atol, rtol=0):
        raise AssertionError(
            f"{name}: computed {computed!r} does not match the published "
            f"value {published!r} (atol={atol})"
        )


print(f"seed={SEED}")

# %% [markdown]
# ## Data
#
# Q4 is the one main question that leaves the `base`-only main path: it needs
# all **three** variants (`base`, `masked_gray`, `rectified`), local stack —
# `co.build_full_local_frame()` — which is 1,155 images x 6 models x 4
# prompts x 3 variants = **83,160 rows**, three times the 27,720-row main
# path. Every comparison below stays paired **within image**: the same 1,155
# images appear under all three variants, and the per-image value for a
# variant is the mean `|e|` over that variant's 24 (model, prompt)
# configurations — the mean of `|e|`, never the `|error|` of a mean
# prediction — so every contrast below is a within-image comparison on
# `pooled`, one row per image.
#
# Missingness strategy: `_common`'s loaders and `build_full_local_frame`
# raise on any row lost in the D5-to-D1 join (a hard join-integrity check,
# not a silent drop), so no row here is dropped for missingness reasons; the
# one genuinely irregular row this notebook singles out (the chain-of-thought
# recovery below) is kept in every aggregate and only removed in its own
# dedicated sensitivity check.

# %%
checks_df, frames = co.run_all_assertions(include_d12=False)
print(checks_df[["id", "description", "passed"]].to_string(index=False))
assert checks_df["passed"].all(), "a load-time assertion failed — the input data does not match what this analysis was built on"

d5_full = co.load_d5()
full_frame = co.build_full_local_frame()
print(f"full_frame: {len(full_frame)} rows, {full_frame['image'].nunique()} images, "
      f"variants={sorted(full_frame['variant'].unique())}")
assert len(full_frame) == 83160

base_local = frames["base_local"]
k1 = co.assert_k1_pairing_complete(d5_full, base_local)
print(f"K1: {k1.passed} — {k1.detail}")
assert k1.passed

d8 = frames["d8"]
d9 = frames["d9"]
d4 = frames["d4"]
maverick_floor = co.compute_maverick_floor(d8, rng_maverick)
c11 = co.assert_c11_n_reproduced(maverick_floor.n_reproduced)
print(f"C11: {c11.passed} — {c11.detail}")
assert c11.passed
print(f"delta_Mav (overall-MAE run-to-run floor) = {maverick_floor.delta_mav:.4f} "
      f"[{maverick_floor.delta_mav_ci_lo:.4f}, {maverick_floor.delta_mav_ci_hi:.4f}]")

# %% [markdown]
# **Grid-Overlay is a prompt, never a variant (C14).** `D5.v2`'s
# `Grid-Overlay` level asks the model to *imagine* a 10x10 grid in text — no
# grid is drawn on any image — so `variant` retains exactly its three
# levels (`base`, `masked_gray`, `rectified`) throughout, and
# `Grid-Overlay` is one of the four prompt levels crossed with each of them,
# never a fourth variant.

# %%
assert set(full_frame["variant"].unique()) == {"base", "masked_gray", "rectified"}
assert set(full_frame["prompt"].unique()) == {"Detailed", "Grid-Overlay", "Point-Hint", "Short"}
print("variant levels:", sorted(full_frame["variant"].unique()))
print("prompt levels (Grid-Overlay is a prompt, not a variant, per C14):",
      sorted(full_frame["prompt"].unique()))

# %% [markdown]
# ## Building the paired per-image, per-variant frame

# %%
def pooled_variant_value(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    sub = frame[frame["variant"] == variant]
    return co.pool_axis_mean_abs_error(sub, ["image"]).rename(columns={"abs_e": f"abs_e_{variant}"})


def pooled_variant_signed_value(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Per-image mean *signed* error `e = prediction - reference` for one
    variant, pooled over that variant's 24 (model, prompt) configurations —
    the signed counterpart of `pooled_variant_value`, needed for mean bias
    rather than MAE. Positive means the variant over-predicts cover relative
    to the reference; negative means it under-predicts.
    """
    sub = frame[frame["variant"] == variant]
    return co.pool_axis_mean_abs_error(sub, ["image"], abs_error_col="e").rename(columns={"e": f"e_{variant}"})


pooled_base = pooled_variant_value(full_frame, "base")
pooled_masked = pooled_variant_value(full_frame, "masked_gray")
pooled_rect = pooled_variant_value(full_frame, "rectified")

pooled = pooled_base.merge(pooled_masked, on="image").merge(pooled_rect, on="image")
ref_map = frames["d1"].set_index("image")["reference"]
campaign_map = frames["d1"].set_index("image")["campaign"]
pooled["reference"] = pooled["image"].map(ref_map)
pooled["campaign"] = pooled["image"].map(campaign_map)
pooled["bin"] = co.assign_bins(pooled["reference"])
assert len(pooled) == 1155 and pooled["image"].is_unique

pooled_signed_base = pooled_variant_signed_value(full_frame, "base")
pooled_signed_masked = pooled_variant_signed_value(full_frame, "masked_gray")
pooled_signed_rect = pooled_variant_signed_value(full_frame, "rectified")
pooled = pooled.merge(pooled_signed_base, on="image").merge(pooled_signed_masked, on="image").merge(
    pooled_signed_rect, on="image"
)
assert len(pooled) == 1155 and pooled["image"].is_unique

pooled.head()

# %% [markdown]
# ## Assumption checks
#
# - **K2 — symmetry of the paired variant difference `d_i`**, for the
#   Hodges-Lehmann reading. Skew and the gap between the Hodges-Lehmann
#   estimate and the plain median are reported alongside the Hodges-Lehmann
#   estimate on every contrast.
# - **K3 — tie burden**: the share of images where the two variants produce
#   an identical pooled value, reported alongside the paired Wilcoxon.
# - **K4 — bootstrap bin coverage**: the count of resamples in which at least
#   one of the five bins is empty; if more than 1% of resamples violate this,
#   the metric switches to the bin-stratified bootstrap.
#
# All three are computed per contrast below and written into
# `Q4_variant_contrasts.csv`.

# %%
def check_contrast_diagnostics(diff: np.ndarray) -> dict:
    k2 = co.check_k2_symmetry(diff)
    k3 = co.check_k3_tie_burden(diff)
    return {**k2, **k3}


# %% [markdown]
# ## The three pooled variant contrasts, as intervals on the difference
#
# **Why balanced MAE, and why it is reported as an interval and not a
# p-value.** Balanced MAE (the unweighted mean of the five per-bin MAEs) is
# the bin-balanced estimand — it asks whether pre-processing changes
# accuracy in a way that gives equal say to the sparse high-cover bins, not
# just to the 933-image bottom bin. Every pairwise comparison of balanced
# MAE in this notebook is a **BCa interval on the difference**, read one at
# a time; these interval verdicts are **not multiplicity-corrected** — a
# property of the statistic, since no p-value on balanced MAE is computed
# anywhere in this notebook.
#
# All three contrasts share the same paired n = 1,155 images (every image
# appears under all three variants), so the pooled contrasts are adequately
# powered even though the per-bin breakdowns (n = 30 to n = 933) are not.

# %%
CONTRASTS = [
    ("masked_gray_minus_base", "abs_e_masked_gray", "abs_e_base"),
    ("rectified_minus_base", "abs_e_rectified", "abs_e_base"),
    ("masked_gray_minus_rectified", "abs_e_masked_gray", "abs_e_rectified"),
]


def balanced_mae_diff_stat(col_a: str, col_b: str):
    def _stat(frame: pd.DataFrame) -> float:
        bal_a = co.balanced_mae(frame[col_a], frame["bin"]).balanced_mae
        bal_b = co.balanced_mae(frame[col_b], frame["bin"]).balanced_mae
        return bal_a - bal_b
    return _stat


contrast_results = {}
_contrast_rngs = {  # each pooled contrast's BCa interval draws from its own generator (see offset note above)
    "masked_gray_minus_base": rng_e1,
    "rectified_minus_base": rng_e1,
    "masked_gray_minus_rectified": rng_e2,
}

for name, col_a, col_b in CONTRASTS:
    stat_fn = balanced_mae_diff_stat(col_a, col_b)
    boot = co.image_bootstrap(pooled, stat_fn, _contrast_rngs[name], image_col="image", bin_col="bin", B=co.B_BOOTSTRAP)
    k4 = co.check_k4_bootstrap_bin_coverage(boot.n_empty_bin_violations, co.B_BOOTSTRAP)

    diff = (pooled[col_a] - pooled[col_b]).to_numpy()
    diag = check_contrast_diagnostics(diff)

    hl = co.hodges_lehmann(diff)
    rb = co.rank_biserial_matched_pairs(diff)
    wr = co.win_rate(pooled[col_a].to_numpy(), pooled[col_b].to_numpy())

    contrast_results[name] = dict(
        boot=boot, k2=diag, k3=diag, k4=k4, hl=hl, rb=rb, wr=wr, diff=diff,
    )

for name in contrast_results:
    r = contrast_results[name]
    print(f"{name}: delta_balanced_MAE={r['boot'].estimate:.3f} "
          f"[{r['boot'].ci_lo:.3f}, {r['boot'].ci_hi:.3f}] ({r['boot'].ci_method})")

# Published values for the three pooled variant contrasts. These are the
# numbers reported in the paper, and this assertion exists so that the
# manuscript and this notebook cannot silently drift apart: a later change to
# the analysis that moves any of them fails here rather than leaving a stale
# figure in the paper.
PUBLISHED_POOLED_CONTRASTS = {
    "masked_gray_minus_base":      {"estimate": -1.6426111694866314, "hodges_lehmann": 0.2916666666666669, "rank_biserial": 0.1300101990707513},
    "rectified_minus_base":        {"estimate": -1.6862547721165413, "hodges_lehmann": 0.302083333333333,  "rank_biserial": 0.1299952307033345},
    "masked_gray_minus_rectified": {"estimate": 0.0436436026299098,  "hodges_lehmann": 0.0208333333333334, "rank_biserial": 0.0187766841384928},
}
for name in contrast_results:
    r = contrast_results[name]
    fr = PUBLISHED_POOLED_CONTRASTS[name]
    assert_reproduces(f"delta_balanced_mae[{name}]", r["boot"].estimate, fr["estimate"])
    assert_reproduces(f"hodges_lehmann[{name}]", r["hl"], fr["hodges_lehmann"])
    assert_reproduces(f"rank_biserial[{name}]", r["rb"], fr["rank_biserial"])
print("Delta balanced MAE, Hodges-Lehmann and rank-biserial match the published values.")

# %% [markdown]
# ### Pooled difference in mean bias against `base`
#
# The same two variant-vs-base contrasts, on **mean signed error** rather
# than balanced MAE: positive means the variant is more over-predicting (or
# less under-predicting) than `base`, pooled over all 1,155 images.

# %%
pooled_bias_diff_results = {}
for name, variant_a in [("masked_gray_minus_base", "masked_gray"), ("rectified_minus_base", "rectified")]:
    signed_col_a = f"e_{variant_a}"

    def stat_bias_pooled_diff(f, ca=signed_col_a):
        return float((f[ca] - f["e_base"]).mean())

    boot_bias_diff = co.image_bootstrap(
        pooled, stat_bias_pooled_diff, rng_breakdown, image_col="image", bin_col="bin", B=co.B_BOOTSTRAP,
    )
    pooled_bias_diff_results[name] = boot_bias_diff
    print(f"{name}: delta_mean_bias={boot_bias_diff.estimate:+.3f} "
          f"[{boot_bias_diff.ci_lo:.3f}, {boot_bias_diff.ci_hi:.3f}] ({boot_bias_diff.ci_method})")

# %% [markdown]
# ## The variant x bin pattern — location and spread of `d_i`
#
# Taking the paired variant difference `d_i = |e|_variantA,i - |e|_variantB,i`
# cancels the error floor to first order — both variants are scored against
# the same reference for the same image. This section reports the per-bin
# **mean and median** of `d_i` (location) and the per-bin **IQR and sd** of
# `d_i` (spread) for both variant-vs-base contrasts, so that a flat location
# with a widening spread is never read as "helps/hurts more at higher cover".

# %%
VARIANT_VS_BASE_CONTRASTS = [
    ("masked_gray_minus_base", "masked_gray", "base"),
    ("rectified_minus_base", "rectified", "base"),
]

per_bin_location_spread = []

for name, variant_a, variant_b in VARIANT_VS_BASE_CONTRASTS:
    col_a, col_b = f"abs_e_{variant_a}", f"abs_e_{variant_b}"
    d_i = (pooled[col_a] - pooled[col_b]).to_numpy()
    bins_arr = pooled["bin"].to_numpy()

    for label in co.BIN_LABELS:
        sub = d_i[bins_arr == label]
        per_bin_location_spread.append({
            "contrast": name, "bin": label, "n": len(sub),
            "mean_d": float(np.mean(sub)) if len(sub) else np.nan,
            "median_d": float(np.median(sub)) if len(sub) else np.nan,
            "iqr_d": float(np.percentile(sub, 75) - np.percentile(sub, 25)) if len(sub) else np.nan,
            "sd_d": float(np.std(sub, ddof=1)) if len(sub) > 1 else np.nan,
        })

per_bin_ls_df = pd.DataFrame(per_bin_location_spread)
per_bin_ls_df

# %% [markdown]
# ## Chain-of-thought sensitivity
#
# One Llama-4-Scout `Grid-Overlay` prediction under `masked_gray` was
# recovered from a JSON parse failure by reading the model's chain-of-thought
# text rather than its structured output. The Scout x Grid-Overlay x
# masked_gray pooled result is recomputed without that one row, and the
# difference is reported as a number rather than assumed negligible.

# %%
scoped_failures = d9[(d9["in_scope"] == True) & (d9["stack"] == "local")]  # noqa: E712
scout_grid = scoped_failures[
    (scoped_failures["model"] == "Llama-4-Scout")
    & (scoped_failures["error"] == "JSON_PARSE_FAILURE")
    & (scoped_failures["prompt"] == "Grid-Overlay")
]
cot_row_mg = scout_grid[scout_grid["variant"] == "masked_gray"]
assert len(cot_row_mg) == 1, f"expected exactly 1 masked_gray chain-of-thought row, got {len(cot_row_mg)}"
cot_image = cot_row_mg["image"].iloc[0]
print(f"chain-of-thought-recovered masked_gray row: image={cot_image}, model=Llama-4-Scout, prompt=Grid-Overlay")

scout_mg_grid_all = full_frame[
    (full_frame["model"] == "Llama-4-Scout")
    & (full_frame["variant"] == "masked_gray")
    & (full_frame["prompt"] == "Grid-Overlay")
]
scout_mg_grid_excl = scout_mg_grid_all[scout_mg_grid_all["image"] != cot_image]

mae_with = float(scout_mg_grid_all["abs_e"].mean())
mae_without = float(scout_mg_grid_excl["abs_e"].mean())
cot_sensitivity = {
    "cot_image": cot_image, "n_with_row": len(scout_mg_grid_all), "n_without_row": len(scout_mg_grid_excl),
    "mae_with_row": mae_with, "mae_without_row": mae_without, "mae_difference": mae_with - mae_without,
}
# Published value: the MAE difference this one recovered row makes to the
# Scout x Grid-Overlay x masked_gray pooled result.
assert_reproduces("cot_mae_difference", cot_sensitivity["mae_difference"], -0.0038497752969153)
cot_sensitivity

# %% [markdown]
# ## Per-bin variant differences (5 bins x 2 variants vs base)
#
# Reported with CIs and per-bin **n**, uncorrected — these are breakdowns of
# the two pooled contrasts above, not additional hypotheses under test. The
# n = 30 and n = 38 top bins are not well powered; these rows are estimates
# with intervals, not verdicts. **Retention or gain as a per-bin percentage
# is not reported anywhere in this notebook** — with per-bin denominators
# this small, a percentage change is not an interpretable quantity; the
# absolute cover-point difference with its interval is what is reported.
#
# Alongside `mae_diff` (the per-bin difference in mean |e|, unsigned), each
# row also carries `bias_diff`: the per-bin difference in **mean signed
# error**, variant minus base. Positive `bias_diff` means the variant is
# more over-predicting (or less under-predicting) than `base` in that bin;
# negative means the opposite.

# %%
per_bin_variant_rows = []
for name, variant_a, variant_b in [("masked_gray_minus_base", "masked_gray", "base"),
                                    ("rectified_minus_base", "rectified", "base")]:
    col_a, col_b = f"abs_e_{variant_a}", f"abs_e_{variant_b}"
    signed_col_a, signed_col_b = f"e_{variant_a}", f"e_{variant_b}"
    for label in co.BIN_LABELS:
        sub = pooled[pooled["bin"] == label]
        n_bin = len(sub)
        if n_bin < 2:
            per_bin_variant_rows.append({
                "contrast": name, "bin": label, "n": n_bin,
                "mae_diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "ci_method": "n<2, not computed",
                "bias_diff": np.nan, "bias_diff_ci_lo": np.nan, "bias_diff_ci_hi": np.nan,
                "bias_diff_ci_method": "n<2, not computed",
            })
            continue

        def stat(f, ca=col_a, cb=col_b):
            return float((f[ca] - f[cb]).mean())

        boot_bin = co.image_bootstrap(sub, stat, rng_breakdown, image_col="image", bin_col=None, B=co.B_BOOTSTRAP)

        def stat_bias(f, ca=signed_col_a, cb=signed_col_b):
            return float((f[ca] - f[cb]).mean())

        boot_bias_bin = co.image_bootstrap(sub, stat_bias, rng_breakdown, image_col="image", bin_col=None, B=co.B_BOOTSTRAP)
        per_bin_variant_rows.append({
            "contrast": name, "bin": label, "n": n_bin,
            "mae_diff": boot_bin.estimate, "ci_lo": boot_bin.ci_lo, "ci_hi": boot_bin.ci_hi,
            "ci_method": boot_bin.ci_method,
            "bias_diff": boot_bias_bin.estimate, "bias_diff_ci_lo": boot_bias_bin.ci_lo,
            "bias_diff_ci_hi": boot_bias_bin.ci_hi, "bias_diff_ci_method": boot_bias_bin.ci_method,
        })

per_bin_variant_df = pd.DataFrame(per_bin_variant_rows)

# Published per-bin Delta MAE point estimates for the two variant-vs-base
# contrasts, five bins each.
PUBLISHED_PER_BIN_MAE_DIFF = {
    ("masked_gray_minus_base", "0-20"):    0.5360767238,
    ("masked_gray_minus_base", "20-40"):   0.2076466049,
    ("masked_gray_minus_base", "40-60"):  -0.4750905797,
    ("masked_gray_minus_base", "60-80"):  -3.3108552632,
    ("masked_gray_minus_base", "80-100"): -5.1708333333,
    ("rectified_minus_base", "0-20"):      0.4855283137,
    ("rectified_minus_base", "20-40"):     0.5050038580,
    ("rectified_minus_base", "40-60"):    -0.0974456522,
    ("rectified_minus_base", "60-80"):    -3.6104714912,
    ("rectified_minus_base", "80-100"):   -5.7138888889,
}
for _, r in per_bin_variant_df.iterrows():
    if pd.notna(r["mae_diff"]):
        published = PUBLISHED_PER_BIN_MAE_DIFF[(r["contrast"], r["bin"])]
        assert_reproduces(f"per_bin_mae_diff[{r['contrast']}/{r['bin']}]", r["mae_diff"], published)
print("Per-bin Delta MAE point estimates match the published values.")
per_bin_variant_df

# %% [markdown]
# ## Per-model variant effects (6 models x 2 variants vs base)
#
# Pooled contrasts average over 24 configurations and can conceal per-model
# effects that point in opposite directions and cancel. Reported here,
# uncorrected, so a pooled null cannot be read as "no effect anywhere"
# without also seeing this range.
#
# **Scale.** The pooled contrast this breaks down is Delta balanced MAE, so
# `mae_diff` below is **Delta balanced MAE per model** — the same estimand,
# on the same scale, as the pooled contrast it decomposes, not Delta overall
# mean `|e|`. `overall_mean_abs_e_diff` is kept alongside, labelled, only
# because it is the scale the Maverick reproducibility floor below is
# defined on.

# %%
per_model_rows = []
for model in co.MODELS:
    model_frame = full_frame[full_frame["model"] == model]
    model_pooled_base = co.pool_axis_mean_abs_error(
        model_frame[model_frame["variant"] == "base"], ["image"]
    ).rename(columns={"abs_e": "abs_e_base"})
    model_pooled_mg = co.pool_axis_mean_abs_error(
        model_frame[model_frame["variant"] == "masked_gray"], ["image"]
    ).rename(columns={"abs_e": "abs_e_masked_gray"})
    model_pooled_rect = co.pool_axis_mean_abs_error(
        model_frame[model_frame["variant"] == "rectified"], ["image"]
    ).rename(columns={"abs_e": "abs_e_rectified"})
    mp = model_pooled_base.merge(model_pooled_mg, on="image").merge(model_pooled_rect, on="image")
    mp["bin"] = mp["image"].map(ref_map).pipe(co.assign_bins)
    assert len(mp) == 1155

    is_maverick = model == "Llama-4-Maverick"

    for name, variant_a in [("masked_gray_minus_base", "masked_gray"), ("rectified_minus_base", "rectified")]:
        col_a = f"abs_e_{variant_a}"

        def stat(f, ca=col_a):
            return co.balanced_mae(f[ca], f["bin"]).balanced_mae - co.balanced_mae(f["abs_e_base"], f["bin"]).balanced_mae

        def stat_overall(f, ca=col_a):
            return float((f[ca] - f["abs_e_base"]).mean())

        boot_m = co.image_bootstrap(mp, stat, rng_breakdown, image_col="image", bin_col="bin", B=co.B_BOOTSTRAP)
        boot_overall = co.image_bootstrap(mp, stat_overall, rng_breakdown, image_col="image", bin_col=None, B=co.B_BOOTSTRAP)
        within_floor = (co.maverick_within_floor(boot_overall.estimate, maverick_floor.delta_mav)
                        if is_maverick else np.nan)
        per_model_rows.append({
            "model": model, "contrast": name, "n": len(mp),
            "mae_diff": boot_m.estimate, "ci_lo": boot_m.ci_lo, "ci_hi": boot_m.ci_hi,
            "ci_method": boot_m.ci_method,
            "overall_mean_abs_e_diff": boot_overall.estimate,
            "overall_mean_abs_e_diff_ci_lo": boot_overall.ci_lo,
            "overall_mean_abs_e_diff_ci_hi": boot_overall.ci_hi,
            "c1_serving_confounded": False,  # variant is within-model, within-serving-path
            "maverick_floor_delta": maverick_floor.delta_mav if is_maverick else np.nan,
            "within_run_to_run_variability": within_floor,
        })

per_model_df = pd.DataFrame(per_model_rows)

# Published per-model Delta balanced MAE and Delta overall-mean|e|, six
# models x two variant-vs-base contrasts.
PUBLISHED_PER_MODEL = {
    ("Gemma-3-12B", "masked_gray_minus_base"):        {"mae_diff": -1.3317059337, "overall_mean_abs_e_diff": -0.8999134199},
    ("Gemma-3-12B", "rectified_minus_base"):          {"mae_diff": -1.3502217407, "overall_mean_abs_e_diff": -0.6174242424},
    ("Gemma-3-27B", "masked_gray_minus_base"):        {"mae_diff": -1.6497746212, "overall_mean_abs_e_diff": 1.1481601732},
    ("Gemma-3-27B", "rectified_minus_base"):          {"mae_diff": -2.5564326572, "overall_mean_abs_e_diff": 1.0782467532},
    ("Llama-4-Maverick", "masked_gray_minus_base"):   {"mae_diff": -0.4438790854, "overall_mean_abs_e_diff": 0.2943073593},
    ("Llama-4-Maverick", "rectified_minus_base"):     {"mae_diff": -1.4106094953, "overall_mean_abs_e_diff": 0.0834523810},
    ("Llama-4-Scout", "masked_gray_minus_base"):      {"mae_diff": -1.1444432372, "overall_mean_abs_e_diff": -0.6415476190},
    ("Llama-4-Scout", "rectified_minus_base"):        {"mae_diff": -1.3135726653, "overall_mean_abs_e_diff": -0.7733896104},
    ("Mistral-Small-3.2", "masked_gray_minus_base"):  {"mae_diff": -3.6651021770, "overall_mean_abs_e_diff": 0.6402164502},
    ("Mistral-Small-3.2", "rectified_minus_base"):    {"mae_diff": -2.9305922521, "overall_mean_abs_e_diff": 0.8578354978},
    ("Qwen-2.5", "masked_gray_minus_base"):           {"mae_diff": -1.6207619624, "overall_mean_abs_e_diff": 0.6005627706},
    ("Qwen-2.5", "rectified_minus_base"):             {"mae_diff": -0.5560998221, "overall_mean_abs_e_diff": 0.3813636364},
}
for _, r in per_model_df.iterrows():
    fr = PUBLISHED_PER_MODEL[(r["model"], r["contrast"])]
    assert_reproduces(f"per_model_mae_diff[{r['model']}/{r['contrast']}]", r["mae_diff"], fr["mae_diff"])
    assert_reproduces(f"per_model_overall_diff[{r['model']}/{r['contrast']}]",
                       r["overall_mean_abs_e_diff"], fr["overall_mean_abs_e_diff"])
print("Per-model Delta balanced MAE and Delta overall-mean|e| match the published values.")
per_model_df

# %% [markdown]
# ## Per-prompt variant effects (4 prompts x 2 variants vs base)
#
# The same quantities as the per-model breakdown above, reported at 4
# prompts x 2 contrasts: each prompt's Delta balanced MAE, with n and CI,
# uncorrected. Prompt choice is not confounded with serving stack, precision
# or GPU the way model choice is, so every row here carries
# `c1_serving_confounded = False` unconditionally, which is worth having on
# the page beside a per-model breakdown that does carry the flag.

# %%
per_prompt_rows = []
for prompt in sorted(full_frame["prompt"].unique()):
    prompt_frame = full_frame[full_frame["prompt"] == prompt]
    pp_base = co.pool_axis_mean_abs_error(
        prompt_frame[prompt_frame["variant"] == "base"], ["image"]
    ).rename(columns={"abs_e": "abs_e_base"})
    pp_mg = co.pool_axis_mean_abs_error(
        prompt_frame[prompt_frame["variant"] == "masked_gray"], ["image"]
    ).rename(columns={"abs_e": "abs_e_masked_gray"})
    pp_rect = co.pool_axis_mean_abs_error(
        prompt_frame[prompt_frame["variant"] == "rectified"], ["image"]
    ).rename(columns={"abs_e": "abs_e_rectified"})
    pp = pp_base.merge(pp_mg, on="image").merge(pp_rect, on="image")
    pp["bin"] = pp["image"].map(ref_map).pipe(co.assign_bins)
    assert len(pp) == 1155

    for name, variant_a in [("masked_gray_minus_base", "masked_gray"), ("rectified_minus_base", "rectified")]:
        col_a = f"abs_e_{variant_a}"

        def stat(f, ca=col_a):
            return co.balanced_mae(f[ca], f["bin"]).balanced_mae - co.balanced_mae(f["abs_e_base"], f["bin"]).balanced_mae

        def stat_overall(f, ca=col_a):
            return float((f[ca] - f["abs_e_base"]).mean())

        boot_p = co.image_bootstrap(pp, stat, rng_prompt, image_col="image", bin_col="bin", B=co.B_BOOTSTRAP)
        boot_p_overall = co.image_bootstrap(pp, stat_overall, rng_prompt, image_col="image", bin_col=None, B=co.B_BOOTSTRAP)
        per_prompt_rows.append({
            "prompt": prompt, "contrast": name, "n": len(pp),
            "mae_diff": boot_p.estimate, "ci_lo": boot_p.ci_lo, "ci_hi": boot_p.ci_hi,
            "ci_method": boot_p.ci_method,
            "overall_mean_abs_e_diff": boot_p_overall.estimate,
            "overall_mean_abs_e_diff_ci_lo": boot_p_overall.ci_lo,
            "overall_mean_abs_e_diff_ci_hi": boot_p_overall.ci_hi,
            "c1_serving_confounded": False,  # prompt choice is not confounded with serving stack
        })

per_prompt_df = pd.DataFrame(per_prompt_rows)
print(f"mean of per-prompt Delta balanced MAE (masked_gray_minus_base) = "
      f"{per_prompt_df.loc[per_prompt_df['contrast'] == 'masked_gray_minus_base', 'mae_diff'].mean():.6f} "
      f"(pooled contrast estimate reproduces this exactly, since both are unweighted means over the same 1,155 images)")
per_prompt_df

# %% [markdown]
# ## Per-configuration variant effects (6 models x 4 prompts = 24 configurations)
#
# Neither the per-model nor the per-prompt breakdown above can show whether
# a particular model-prompt pairing was especially helped or hurt by
# pre-processing — each averages over the other axis. This section reports
# each of the 24 configurations' own balanced MAE under all three variants,
# and the same three-step decomposition used pooled and per bin above: the
# crop step (`masked_gray - base`), the perspective-correction step
# (`rectified - masked_gray`), and the two together (`rectified - base`),
# each with a bootstrap confidence interval. Descriptive and uncorrected —
# no p-value is computed here, consistent with the rest of this notebook.

# %%
per_config_rows = []
for model in co.MODELS:
    for prompt in sorted(full_frame["prompt"].unique()):
        cfg_frame = full_frame[(full_frame["model"] == model) & (full_frame["prompt"] == prompt)]
        cfg_base = cfg_frame[cfg_frame["variant"] == "base"][["image", "abs_e", "bin"]].rename(
            columns={"abs_e": "abs_e_base"})
        cfg_mg = cfg_frame[cfg_frame["variant"] == "masked_gray"][["image", "abs_e"]].rename(
            columns={"abs_e": "abs_e_masked_gray"})
        cfg_rect = cfg_frame[cfg_frame["variant"] == "rectified"][["image", "abs_e"]].rename(
            columns={"abs_e": "abs_e_rectified"})
        cfg = cfg_base.merge(cfg_mg, on="image").merge(cfg_rect, on="image")
        assert len(cfg) == 1155 and cfg["image"].is_unique

        bal_base = co.balanced_mae(cfg["abs_e_base"], cfg["bin"]).balanced_mae
        bal_mg = co.balanced_mae(cfg["abs_e_masked_gray"], cfg["bin"]).balanced_mae
        bal_rect = co.balanced_mae(cfg["abs_e_rectified"], cfg["bin"]).balanced_mae

        def stat_crop(f):
            return co.balanced_mae(f["abs_e_masked_gray"], f["bin"]).balanced_mae - co.balanced_mae(f["abs_e_base"], f["bin"]).balanced_mae

        def stat_correction(f):
            return co.balanced_mae(f["abs_e_rectified"], f["bin"]).balanced_mae - co.balanced_mae(f["abs_e_masked_gray"], f["bin"]).balanced_mae

        def stat_total(f):
            return co.balanced_mae(f["abs_e_rectified"], f["bin"]).balanced_mae - co.balanced_mae(f["abs_e_base"], f["bin"]).balanced_mae

        boot_crop = co.image_bootstrap(cfg, stat_crop, rng_breakdown, image_col="image", bin_col="bin", B=co.B_BOOTSTRAP)
        boot_corr = co.image_bootstrap(cfg, stat_correction, rng_breakdown, image_col="image", bin_col="bin", B=co.B_BOOTSTRAP)
        boot_tot = co.image_bootstrap(cfg, stat_total, rng_breakdown, image_col="image", bin_col="bin", B=co.B_BOOTSTRAP)

        # The crop step's point estimate plus the correction step's point
        # estimate must equal the total's point estimate for this
        # configuration, to floating-point precision — the same additivity
        # the pooled and per-bin decompositions satisfy above.
        assert np.isclose(boot_crop.estimate + boot_corr.estimate, boot_tot.estimate, atol=1e-9), (
            f"{model}/{prompt}: crop ({boot_crop.estimate}) + correction ({boot_corr.estimate}) "
            f"!= total ({boot_tot.estimate})"
        )

        per_config_rows.append({
            "question_id": "Q4", "model": model, "prompt": prompt, "n": len(cfg),
            "balanced_mae_base": bal_base, "balanced_mae_masked_gray": bal_mg, "balanced_mae_rectified": bal_rect,
            "crop_diff": boot_crop.estimate, "crop_diff_ci_lo": boot_crop.ci_lo, "crop_diff_ci_hi": boot_crop.ci_hi,
            "correction_diff": boot_corr.estimate, "correction_diff_ci_lo": boot_corr.ci_lo,
            "correction_diff_ci_hi": boot_corr.ci_hi,
            "total_diff": boot_tot.estimate, "total_diff_ci_lo": boot_tot.ci_lo, "total_diff_ci_hi": boot_tot.ci_hi,
            "ci_method": boot_tot.ci_method,
            "descriptive_uncorrected": True,
        })

per_config_df = pd.DataFrame(per_config_rows)
print(f"crop helps {int((per_config_df['crop_diff'] < 0).sum())}/24 configurations, "
      f"correction helps {int((per_config_df['correction_diff'] < 0).sum())}/24, "
      f"total helps {int((per_config_df['total_diff'] < 0).sum())}/24")
print(f"median across the 24 configurations: crop={per_config_df['crop_diff'].median():+.3f}, "
      f"correction={per_config_df['correction_diff'].median():+.3f}, "
      f"total={per_config_df['total_diff'].median():+.3f}")
per_config_df

# %% [markdown]
# ## The crop/correction/total decomposition by model and by prompt
#
# The 24-configuration table above cannot be read off model by model or
# prompt by prompt without averaging by hand outside this notebook, and a
# number nothing here produced has no place in the manuscript. This section
# reports the same three-step decomposition — the crop, the perspective
# correction and the two together — collapsed onto each of the two axes the
# 24 configurations cross: six rows for the six models (each averaging over
# that model's four prompts) and four rows for the four prompts (each
# averaging over that prompt's six models).
#
# **These rows are views of one another, not independent evidence.** The six
# model rows are six different summaries of the same 1,155 images, and the
# four prompt rows are four more summaries of that same set — every row
# shares its images with every other row on its axis. A gap between two
# model rows or two prompt rows is a descriptive contrast, exactly like the
# per-model and per-prompt breakdowns above; it is reported with a bootstrap
# interval for calibration, never as a hypothesis test, and no p-value or
# correction is computed for it anywhere in this notebook.
#
# **The statistic**, for one level of one axis, follows the same
# construction as `Q1_metrics_by_model.csv`: restrict to that level's rows
# (a model's four prompts, or a prompt's six models), take the per-image mean
# `|e|` over them, take the mean within each of the five reference-cover
# bins, then average the five bin means unweighted — the same balanced MAE
# used throughout this notebook, just restricted to one axis level instead
# of pooled over all 24 configurations or held per-configuration. The three
# steps (`crop`, `correction`, `total`) are the same balanced-MAE contrasts
# used pooled and per configuration, reusing `balanced_mae_diff_stat` so
# this is not a second definition of the same statistic.

# %%
def _axis_level_frame(source_frame: pd.DataFrame, axis_col: str, level) -> pd.DataFrame:
    """Restrict `full_frame` to one level of one axis (a model's four
    prompts, or a prompt's six models), then pool to one row per image per
    variant — the same construction `pooled` uses for the full 24-way
    average, just restricted to this axis level first.
    """
    sub = source_frame[source_frame[axis_col] == level]
    lvl_base = co.pool_axis_mean_abs_error(
        sub[sub["variant"] == "base"], ["image"]
    ).rename(columns={"abs_e": "abs_e_base"})
    lvl_mg = co.pool_axis_mean_abs_error(
        sub[sub["variant"] == "masked_gray"], ["image"]
    ).rename(columns={"abs_e": "abs_e_masked_gray"})
    lvl_rect = co.pool_axis_mean_abs_error(
        sub[sub["variant"] == "rectified"], ["image"]
    ).rename(columns={"abs_e": "abs_e_rectified"})
    out = lvl_base.merge(lvl_mg, on="image").merge(lvl_rect, on="image")
    out["bin"] = out["image"].map(ref_map).pipe(co.assign_bins)
    return out


BY_AXIS_STEPS = [
    ("crop", "abs_e_masked_gray", "abs_e_base"),
    ("correction", "abs_e_rectified", "abs_e_masked_gray"),
    ("total", "abs_e_rectified", "abs_e_base"),
]

by_axis_rows = []
for axis_name, axis_col, levels in [
    ("model", "model", list(co.MODELS)),
    ("prompt", "prompt", sorted(full_frame["prompt"].unique())),
]:
    for level in levels:
        lvl_frame = _axis_level_frame(full_frame, axis_col, level)
        n_images_level = len(lvl_frame)
        assert n_images_level == 1155 and lvl_frame["image"].is_unique, (
            f"{axis_name}={level}: expected all 1,155 images, one row each, got {n_images_level}"
        )

        bal_base = co.balanced_mae(lvl_frame["abs_e_base"], lvl_frame["bin"]).balanced_mae
        bal_mg = co.balanced_mae(lvl_frame["abs_e_masked_gray"], lvl_frame["bin"]).balanced_mae
        bal_rect = co.balanced_mae(lvl_frame["abs_e_rectified"], lvl_frame["bin"]).balanced_mae
        bal_by_variant = {"base": bal_base, "masked_gray": bal_mg, "rectified": bal_rect}

        step_estimates = {}
        for step, col_a, col_b in BY_AXIS_STEPS:
            stat_fn = balanced_mae_diff_stat(col_a, col_b)
            boot_step = co.image_bootstrap(
                lvl_frame, stat_fn, rng_by_axis, image_col="image", bin_col="bin", B=co.B_BOOTSTRAP,
            )
            step_estimates[step] = boot_step
            by_axis_rows.append({
                "question_id": "Q4", "axis": axis_name, "level": level, "step": step,
                "contrast": f"{col_a.replace('abs_e_', '')}_minus_{col_b.replace('abs_e_', '')}",
                "n_images": n_images_level,
                "balanced_mae_base": bal_base, "balanced_mae_masked_gray": bal_mg,
                "balanced_mae_rectified": bal_rect,
                "estimate": boot_step.estimate, "ci_lo": boot_step.ci_lo, "ci_hi": boot_step.ci_hi,
                "ci_method": boot_step.ci_method,
                "descriptive_uncorrected": True,
                "note": (
                    f"the six model rows are six views of the same 1,155 images and the four "
                    f"prompt rows likewise, so no difference between two levels here is a test "
                    f"and no interval on this row supports one"
                ),
            })

        # The crop step's point estimate plus the correction step's point
        # estimate must equal the total's point estimate for this level, to
        # floating-point precision — the same additivity checked pooled, per
        # bin and per configuration above.
        assert np.isclose(
            step_estimates["crop"].estimate + step_estimates["correction"].estimate,
            step_estimates["total"].estimate, atol=1e-9,
        ), (
            f"{axis_name}={level}: crop ({step_estimates['crop'].estimate}) + correction "
            f"({step_estimates['correction'].estimate}) != total ({step_estimates['total'].estimate})"
        )

by_axis_df = pd.DataFrame(by_axis_rows)
assert len(by_axis_df) == 30, f"expected 30 rows (6 models + 4 prompts) x 3 steps, got {len(by_axis_df)}"
print(f"by-axis decomposition: {len(by_axis_df)} rows "
      f"({by_axis_df['axis'].eq('model').sum()} model rows, {by_axis_df['axis'].eq('prompt').sum()} prompt rows)")
by_axis_df

# %% [markdown]
# **Reading this table.** Within the model axis, the crop step ranges from
# a small effect for some models to several cover points for others. The
# correction step's pooled near-zero estimate is not an absence of effect at
# the model level: of the six per-model correction intervals, four exclude
# zero, and they exclude it in **opposite directions** — two models improve
# by up to about one cover point, two worsen by up to about one cover point,
# only two of the six intervals cover zero. The pooled correction estimate is
# close to the unweighted mean of these six values, so the pooled null is the
# average of these opposing per-model effects cancelling, not evidence that
# the correction step does nothing model by model. The prompt axis is
# included for the same reason no axis is reported without its counterpart:
# prompt choice is not confounded with serving stack, precision or GPU the
# way model choice is, so a prompt-axis row never carries the reservations a
# model-axis row does.

# %% [markdown]
# ## Balanced MAE and per-bin MAE by variant
#
# Computed here, independent of any figure, so `Q4_variant_metrics.csv`'s
# contents never depend on plotting code having executed.
#
# **Mean bias, and its sign.** Alongside balanced MAE and the per-bin MAEs,
# this table also reports **mean bias**, the mean of the *signed* error
# `e = prediction - reference`, pooled and per bin, for each variant. Mean
# bias is signed throughout this study: **positive means the variant
# over-predicts** cover relative to the reference, **negative means it
# under-predicts**; this is the opposite question from balanced MAE, which
# is unsigned and cannot show whether a variant's errors lean one direction.
# The difference in mean bias against `base` is reported alongside the
# existing difference in balanced MAE, per bin and pooled, with the same
# sign convention: positive means the variant is more over-predicting (or
# less under-predicting) than `base`.

# %%
variant_metrics_rows = []
variant_colors = {"base": "black", "masked_gray": "tab:blue", "rectified": "tab:orange"}
bin_x = np.arange(len(co.BIN_LABELS))
variant_bal_results = {}
variant_bias_pooled = {}
variant_bias_per_bin = {}

for variant, col, signed_col in [
    ("base", "abs_e_base", "e_base"),
    ("masked_gray", "abs_e_masked_gray", "e_masked_gray"),
    ("rectified", "abs_e_rectified", "e_rectified"),
]:
    per_bin_ci_lo, per_bin_ci_hi = {}, {}
    per_bin_bias_ci_lo, per_bin_bias_ci_hi = {}, {}
    per_bin_bias_est = {}
    for label in co.BIN_LABELS:
        sub = pooled[pooled["bin"] == label]

        def stat(f, c=col):
            return float(f[c].mean())

        boot_b = co.image_bootstrap(sub, stat, rng_breakdown, image_col="image", bin_col=None, B=co.B_BOOTSTRAP)
        per_bin_ci_lo[label], per_bin_ci_hi[label] = boot_b.ci_lo, boot_b.ci_hi

        def stat_bias(f, c=signed_col):
            return float(f[c].mean())

        boot_bias_b = co.image_bootstrap(sub, stat_bias, rng_breakdown, image_col="image", bin_col=None, B=co.B_BOOTSTRAP)
        per_bin_bias_est[label] = boot_bias_b.estimate
        per_bin_bias_ci_lo[label], per_bin_bias_ci_hi[label] = boot_bias_b.ci_lo, boot_bias_b.ci_hi

    bal = co.balanced_mae(pooled[col], pooled["bin"], ci_lo=per_bin_ci_lo, ci_hi=per_bin_ci_hi)
    variant_bal_results[variant] = bal
    row = {"variant": variant, "n_images": 1155}
    row.update(bal.to_row())

    def stat_bias_pooled(f, c=signed_col):
        return float(f[c].mean())

    boot_bias_pooled = co.image_bootstrap(
        pooled, stat_bias_pooled, rng_breakdown, image_col="image", bin_col="bin", B=co.B_BOOTSTRAP,
    )
    variant_bias_pooled[variant] = boot_bias_pooled
    variant_bias_per_bin[variant] = {
        "est": per_bin_bias_est, "ci_lo": per_bin_bias_ci_lo, "ci_hi": per_bin_bias_ci_hi,
    }
    row["mean_bias"] = boot_bias_pooled.estimate
    row["mean_bias_ci_lo"] = boot_bias_pooled.ci_lo
    row["mean_bias_ci_hi"] = boot_bias_pooled.ci_hi
    row["mean_bias_ci_method"] = boot_bias_pooled.ci_method
    for label in co.BIN_LABELS:
        row[f"mean_bias_bin_{label}"] = per_bin_bias_est[label]
        row[f"mean_bias_bin_{label}_ci_lo"] = per_bin_bias_ci_lo[label]
        row[f"mean_bias_bin_{label}_ci_hi"] = per_bin_bias_ci_hi[label]
    variant_metrics_rows.append(row)

variant_metrics_df = pd.DataFrame(variant_metrics_rows)

# Published balanced MAE and per-bin MAE, one row per variant. These are the
# values reported in the paper, and this assertion exists so that the
# manuscript and this notebook cannot silently drift apart.
PUBLISHED_VARIANT_METRICS = {
    "base":        {"balanced_mae": 15.8035233665, "mae_bin_0-20": 6.3120266166, "mae_bin_20-40": 10.5030015432, "mae_bin_40-60": 16.0298913043, "mae_bin_60-80": 20.2351973684, "mae_bin_80-100": 25.9375000000},
    "masked_gray": {"balanced_mae": 14.1609121970, "mae_bin_0-20": 6.8481033405, "mae_bin_20-40": 10.7106481481, "mae_bin_40-60": 15.5548007246, "mae_bin_60-80": 16.9243421053, "mae_bin_80-100": 20.7666666667},
    "rectified":   {"balanced_mae": 14.1172685944, "mae_bin_0-20": 6.7975549303, "mae_bin_20-40": 11.0080054012, "mae_bin_40-60": 15.9324456522, "mae_bin_60-80": 16.6247258772, "mae_bin_80-100": 20.2236111111},
}
for _, r in variant_metrics_df.iterrows():
    fr = PUBLISHED_VARIANT_METRICS[r["variant"]]
    assert_reproduces(f"balanced_mae[{r['variant']}]", r["balanced_mae"], fr["balanced_mae"])
    for label in co.BIN_LABELS:
        assert_reproduces(f"mae_bin_{label}[{r['variant']}]", r[f"mae_bin_{label}"], fr[f"mae_bin_{label}"])
print("Balanced MAE and every per-bin MAE, by variant, match the published values.")
print(variant_metrics_df[["variant", "mean_bias", "mean_bias_ci_lo", "mean_bias_ci_hi"]]
      .to_string(index=False))
variant_metrics_df[["variant", "balanced_mae", "mean_bias"]]

# %% [markdown]
# ## Chart 1 — pooled balanced MAE by variant, per bin
#
# **What to look for.** Three lines (one per variant), per-bin MAE (cover
# points) against reference bin, with 95% CI error bars and per-bin n
# annotated. If pre-processing changes the bin structure Q3 found, the lines
# should diverge rather than run parallel, especially in the sparse top bins
# where n is smallest and the CIs widest.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5.5))
for variant in ("base", "masked_gray", "rectified"):
    bal = variant_bal_results[variant]
    means = [bal.per_bin_mae[label] for label in co.BIN_LABELS]
    los = [bal.per_bin_ci_lo[label] for label in co.BIN_LABELS]
    his = [bal.per_bin_ci_hi[label] for label in co.BIN_LABELS]
    err_lo = [m - lo for m, lo in zip(means, los)]
    err_hi = [hi - m for m, hi in zip(means, his)]
    ax.errorbar(bin_x, means, yerr=[err_lo, err_hi], marker="o", label=variant, color=variant_colors[variant],
                capsize=3)

bin_n_lookup = {label: int((pooled["bin"] == label).sum()) for label in co.BIN_LABELS}
tick_labels = [f"{label}\n(n={bin_n_lookup[label]})" for label in co.BIN_LABELS]

ax.set_ylim(bottom=0)
ax.set_xticks(bin_x)
ax.set_xticklabels(tick_labels)
ax.set_xlabel("Reference cover bin (cover points)")
ax.set_ylabel("Mean |error| (cover points)")
ax.set_title("Pooled per-bin MAE by variant (95% BCa CI, image bootstrap)")
ax.legend(title="variant")
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q4_per_bin_mae_by_variant.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Chart 2 — variant difference `d_i` by bin: location and spread
#
# **What to look for.** For each variant-vs-base contrast, the top panel
# shows per-bin mean `d_i` with 95% CI (location); the bottom panel shows
# per-bin IQR of `d_i` (spread). A flat top panel alongside a widening
# bottom panel is read as "more variable", not "helps/hurts more".

# %%
per_bin_d_location_ci = {}

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for col_i, (name, variant_a, variant_b) in enumerate(VARIANT_VS_BASE_CONTRASTS):
    col_a, col_b = f"abs_e_{variant_a}", f"abs_e_{variant_b}"
    d_i = (pooled[col_a] - pooled[col_b]).to_numpy()
    bins_arr = pooled["bin"].to_numpy()

    per_bin_d_location_ci[name] = {}
    means, cis_lo, cis_hi, iqrs, ns = [], [], [], [], []
    for label in co.BIN_LABELS:
        sub_mask = bins_arr == label
        sub = d_i[sub_mask]
        ns.append(len(sub))
        iqrs.append(np.percentile(sub, 75) - np.percentile(sub, 25) if len(sub) else np.nan)
        if len(sub) >= 2:
            sub_df = pd.DataFrame({"d": sub, "image": pooled.loc[sub_mask, "image"].to_numpy()})

            def stat(f):
                return float(f["d"].mean())

            boot_b = co.image_bootstrap(sub_df, stat, rng_breakdown, image_col="image", bin_col=None, B=co.B_BOOTSTRAP)
            means.append(boot_b.estimate)
            cis_lo.append(boot_b.estimate - boot_b.ci_lo)
            cis_hi.append(boot_b.ci_hi - boot_b.estimate)
            per_bin_d_location_ci[name][label] = (boot_b.estimate, boot_b.ci_lo, boot_b.ci_hi)
        else:
            means.append(np.nan)
            cis_lo.append(np.nan)
            cis_hi.append(np.nan)
            per_bin_d_location_ci[name][label] = (np.nan, np.nan, np.nan)

    ax_top = axes[0, col_i]
    ax_top.errorbar(bin_x, means, yerr=[cis_lo, cis_hi], marker="o", capsize=3, color="tab:green")
    ax_top.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_top.set_title(name)
    ax_top.set_ylabel("mean d_i (cover points)")

    ax_bot = axes[1, col_i]
    ax_bot.plot(bin_x, iqrs, marker="s", color="tab:red")
    ax_bot.set_ylabel("IQR of d_i (cover points)")
    ax_bot.set_xlabel("Reference cover bin (cover points)")
    ax_bot.set_xticks(bin_x)
    ax_bot.set_xticklabels(co.BIN_LABELS)
    for i, n in enumerate(ns):
        ax_bot.annotate(f"n={n}", (bin_x[i], iqrs[i]), xytext=(0, 8), textcoords="offset points",
                         ha="center", fontsize=8, color="gray")

fig.suptitle("Variant difference d_i by bin: location (top) vs spread (bottom)")
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q4_variant_bin_location_spread.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Chart 3 — per-model and per-prompt variant effects
#
# **What to look for.** One point per model (left) or prompt (right) per
# variant contrast, Delta balanced MAE (cover points) with 95% CI. If the
# per-model or per-prompt range straddles zero widely while the pooled
# contrast is null, that heterogeneity — not the pooled number — is the more
# informative result.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
model_x = np.arange(len(co.MODELS))
width = 0.35
for offset, (name, marker, color) in enumerate([
    ("masked_gray_minus_base", "o", "tab:blue"),
    ("rectified_minus_base", "s", "tab:orange"),
]):
    sub = per_model_df[per_model_df["contrast"] == name].set_index("model").reindex(list(co.MODELS))
    xpos = model_x + (offset - 0.5) * width
    err_lo = sub["mae_diff"] - sub["ci_lo"]
    err_hi = sub["ci_hi"] - sub["mae_diff"]
    ax.errorbar(xpos, sub["mae_diff"], yerr=[err_lo, err_hi], fmt=marker, capsize=3, color=color, label=name)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xticks(model_x)
ax.set_xticklabels(list(co.MODELS), rotation=30, ha="right")
ax.set_ylabel("Delta balanced MAE, variant minus base (cover points)")
ax.set_title("Per-model (n=1,155 images per model)")
ax.legend(fontsize=8)

ax2 = axes[1]
prompts_sorted = sorted(per_prompt_df["prompt"].unique())
prompt_x = np.arange(len(prompts_sorted))
for offset, (name, marker, color) in enumerate([
    ("masked_gray_minus_base", "o", "tab:blue"),
    ("rectified_minus_base", "s", "tab:orange"),
]):
    sub = per_prompt_df[per_prompt_df["contrast"] == name].set_index("prompt").reindex(prompts_sorted)
    xpos = prompt_x + (offset - 0.5) * width
    err_lo = sub["mae_diff"] - sub["ci_lo"]
    err_hi = sub["ci_hi"] - sub["mae_diff"]
    ax2.errorbar(xpos, sub["mae_diff"], yerr=[err_lo, err_hi], fmt=marker, capsize=3, color=color, label=name)
ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax2.set_xticks(prompt_x)
ax2.set_xticklabels(prompts_sorted, rotation=30, ha="right")
ax2.set_ylabel("Delta balanced MAE, variant minus base (cover points)")
ax2.set_title("Per-prompt (n=1,155 images per prompt)")
ax2.legend(fontsize=8)

fig.suptitle("Per-model and per-prompt variant effect on Delta balanced MAE")
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q4_per_model_per_prompt_variant_effects.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## The pipeline is nested, not two parallel arms
#
# Both `masked_gray` and `rectified` crop the photograph to the
# **hand-annotated quadrat corners**; `rectified` **additionally** applies
# **the perspective correction** to the cropped quadrat, mapping it to a
# square. The two variants are therefore not two independent pre-processing
# choices applied to `base`. They are the **first and second step of one
# pipeline**, and the three contrasts above decompose additively along it:
# the crop step's contrast plus the perspective-correction step's contrast
# sum, exactly, to the crop-and-correct contrast.
#
# | Contrast | What it isolates |
# |---|---|
# | `masked_gray - base` | **cropping** to the quadrat corners |
# | `rectified - masked_gray` | **the perspective correction**, and nothing else |
# | `rectified - base` | crop **and** the perspective correction together |
#
# **The per-bin decomposition below computes the same three steps
# separately within each of the five reference-cover bins**, so that a
# pattern present in one step but not the other cannot be missed inside a
# single pooled average.

# %%
_crop_est = contrast_results["masked_gray_minus_base"]["boot"].estimate
_pc_est = -contrast_results["masked_gray_minus_rectified"]["boot"].estimate
_total_est = contrast_results["rectified_minus_base"]["boot"].estimate
assert np.sign(_crop_est) == np.sign(_total_est) and np.sign(_pc_est) != -np.sign(_total_est), (
    "crop and perspective-correction steps must not oppose the total's sign for a "
    "share-of-total reading to apply"
)


def _pc_share_stat(f: pd.DataFrame) -> float:
    bal_mg = co.balanced_mae(f["abs_e_masked_gray"], f["bin"]).balanced_mae
    bal_rect = co.balanced_mae(f["abs_e_rectified"], f["bin"]).balanced_mae
    bal_base = co.balanced_mae(f["abs_e_base"], f["bin"]).balanced_mae
    perspective_correction = bal_rect - bal_mg
    total = bal_rect - bal_base
    return np.nan if total == 0 else 100.0 * perspective_correction / total


rng_w = np.random.default_rng(SEED + 7)  # own generator: the ratio bootstrap is not a headline number
_pc_share_boot = co.image_bootstrap(
    pooled, _pc_share_stat, rng_w, image_col="image", bin_col="bin", B=co.B_BOOTSTRAP,
)
print(f"perspective-correction share of total (bootstrap): point={_pc_share_boot.estimate:.2f}%, "
      f"95% CI [{_pc_share_boot.ci_lo:.1f}%, {_pc_share_boot.ci_hi:.1f}%] ({_pc_share_boot.ci_method}) "
      "— unstable because the correction contrast's own interval covers zero; not reported as a point estimate alone")

nesting_rows = [
    {
        "question_id": "Q4", "bin": "pooled", "step_isolated": "crop_to_quadrat_corners",
        "contrast": "masked_gray_minus_base", "n": 1155,
        "estimate": contrast_results["masked_gray_minus_base"]["boot"].estimate,
        "ci_lo": contrast_results["masked_gray_minus_base"]["boot"].ci_lo,
        "ci_hi": contrast_results["masked_gray_minus_base"]["boot"].ci_hi,
    },
    {
        "question_id": "Q4", "bin": "pooled", "step_isolated": "perspective_correction_only",
        "contrast": "rectified_minus_masked_gray", "n": 1155,
        # This row is the negative of the masked_gray_minus_rectified contrast
        # computed above — same number, opposite sign, so that negative here
        # means the perspective correction improved the image, matching the
        # convention used everywhere else in this notebook.
        "estimate": -contrast_results["masked_gray_minus_rectified"]["boot"].estimate,
        "ci_lo": -contrast_results["masked_gray_minus_rectified"]["boot"].ci_hi,
        "ci_hi": -contrast_results["masked_gray_minus_rectified"]["boot"].ci_lo,
    },
    {
        "question_id": "Q4", "bin": "pooled", "step_isolated": "crop_and_correction_together",
        "contrast": "rectified_minus_base", "n": 1155,
        "estimate": contrast_results["rectified_minus_base"]["boot"].estimate,
        "ci_lo": contrast_results["rectified_minus_base"]["boot"].ci_lo,
        "ci_hi": contrast_results["rectified_minus_base"]["boot"].ci_hi,
    },
]

# %% [markdown]
# **The per-bin three-step decomposition.** The same three steps, computed
# within each reference bin separately, each with its own n and BCa
# interval. Per bin, alongside the three cover-point differences, this also
# reports how many images each step improved and how many it worsened, so a
# location-shift reading is not confused with a spread-driven one. The crop
# step's contrast plus the perspective-correction step's contrast must sum
# to the total in every bin, exactly as they do pooled; that identity is
# checked here rather than only pooled, so a defect in one bin's arithmetic
# cannot pass unnoticed.

# %%
per_bin_correction_d = {}  # per-bin d_i for the perspective-correction step, kept for the flatness check below

for label in co.BIN_LABELS:
    sub = pooled[pooled["bin"] == label]
    n_bin = len(sub)

    def _stat_crop(f):
        return float((f["abs_e_masked_gray"] - f["abs_e_base"]).mean())

    def _stat_correction(f):
        return float((f["abs_e_rectified"] - f["abs_e_masked_gray"]).mean())

    def _stat_total(f):
        return float((f["abs_e_rectified"] - f["abs_e_base"]).mean())

    boot_crop = co.image_bootstrap(sub, _stat_crop, rng_bin_decomp, image_col="image", bin_col=None, B=co.B_BOOTSTRAP)
    boot_corr = co.image_bootstrap(sub, _stat_correction, rng_bin_decomp, image_col="image", bin_col=None, B=co.B_BOOTSTRAP)
    boot_tot = co.image_bootstrap(sub, _stat_total, rng_bin_decomp, image_col="image", bin_col=None, B=co.B_BOOTSTRAP)

    # The crop step's point estimate plus the correction step's point
    # estimate must equal the total's point estimate in this bin, to
    # floating-point precision — the same additivity the pooled row above
    # satisfies exactly, checked here per bin rather than assumed to carry
    # over.
    assert np.isclose(boot_crop.estimate + boot_corr.estimate, boot_tot.estimate, atol=1e-9), (
        f"bin {label}: crop ({boot_crop.estimate}) + correction ({boot_corr.estimate}) "
        f"!= total ({boot_tot.estimate})"
    )

    d_crop = (sub["abs_e_masked_gray"] - sub["abs_e_base"]).to_numpy()
    d_corr = (sub["abs_e_rectified"] - sub["abs_e_masked_gray"]).to_numpy()
    d_tot = (sub["abs_e_rectified"] - sub["abs_e_base"]).to_numpy()
    per_bin_correction_d[label] = d_corr

    for step_name, contrast_name, boot, d_arr in [
        ("crop_to_quadrat_corners", "masked_gray_minus_base", boot_crop, d_crop),
        ("perspective_correction_only", "rectified_minus_masked_gray", boot_corr, d_corr),
        ("crop_and_correction_together", "rectified_minus_base", boot_tot, d_tot),
    ]:
        n_improved = int(np.sum(d_arr < 0))  # negative = the later variant has smaller |e| = improved
        n_worsened = int(np.sum(d_arr > 0))
        n_unchanged = int(np.sum(d_arr == 0))
        nesting_rows.append({
            "question_id": "Q4", "bin": label, "step_isolated": step_name,
            "contrast": contrast_name, "n": n_bin,
            "estimate": boot.estimate, "ci_lo": boot.ci_lo, "ci_hi": boot.ci_hi,
            "n_images_improved": n_improved, "n_images_worsened": n_worsened,
            "n_images_unchanged": n_unchanged,
        })

nesting_df = pd.DataFrame(nesting_rows)
nesting_df["perspective_correction_share_pct_point"] = np.where(
    (nesting_df["step_isolated"] == "perspective_correction_only") & (nesting_df["bin"] == "pooled"),
    _pc_share_boot.estimate, np.nan,
)
nesting_df["perspective_correction_share_pct_ci_lo"] = np.where(
    (nesting_df["step_isolated"] == "perspective_correction_only") & (nesting_df["bin"] == "pooled"),
    _pc_share_boot.ci_lo, np.nan,
)
nesting_df["perspective_correction_share_pct_ci_hi"] = np.where(
    (nesting_df["step_isolated"] == "perspective_correction_only") & (nesting_df["bin"] == "pooled"),
    _pc_share_boot.ci_hi, np.nan,
)
nesting_df["perspective_correction_share_note"] = np.where(
    (nesting_df["step_isolated"] == "perspective_correction_only") & (nesting_df["bin"] == "pooled"),
    "pooled ratio of two contrasts, one of which (the perspective correction) is not "
    "distinguishable from zero; 95% bootstrap CI is wide relative to the point estimate "
    "— treat as unstable, not precise; not computed per bin because the same instability "
    "argument applies with even less data per bin",
    "",
)
nesting_df["pipeline_fact_provenance"] = (
    "user-supplied pipeline fact; not derivable from any file in this repository "
    "(the photographs are not part of this repository). Recorded here as context, not as data "
    "anything is computed from"
)
_pooled_nesting = nesting_df[nesting_df["bin"] == "pooled"].set_index("contrast")

# Published pooled nesting-decomposition point estimates: the crop step, the
# perspective-correction step, and the two steps together, on the same
# cover-point scale as the pooled contrasts above. Point estimates on the
# pooled row are deterministic functions of the loaded data and are checked
# against the values reported in the paper.
PUBLISHED_NESTING_POOLED = {
    "masked_gray_minus_base":      -1.6426111694866314,
    "rectified_minus_masked_gray": -0.0436436026299098,
    "rectified_minus_base":        -1.6862547721165413,
}
for contrast_name in ["masked_gray_minus_base", "rectified_minus_masked_gray", "rectified_minus_base"]:
    assert_reproduces(
        f"nesting_pooled[{contrast_name}]",
        _pooled_nesting.loc[contrast_name, "estimate"],
        PUBLISHED_NESTING_POOLED[contrast_name],
    )
print("Pooled nesting-decomposition point estimates match the published values.")

# This row is the negative of the masked_gray_minus_rectified contrast
# computed above — same number, opposite sign, not a mismatch. A sign flip
# here would invert the reading of every d_i-based result below, so it is
# asserted structurally rather than left to be checked only by eye.
assert np.isclose(
    _pooled_nesting.loc["rectified_minus_masked_gray", "estimate"],
    -contrast_results["masked_gray_minus_rectified"]["boot"].estimate,
)
# The two steps sum exactly to the total, on the cover-point scale, pooled.
assert np.isclose(_crop_est + _pc_est, _total_est)

nesting_df.to_csv(RESULTS_DIR / "Q4_variant_nesting.csv", index=False)
print(f"wrote Q4_variant_nesting.csv ({len(nesting_df)} rows: 3 pooled + {5 * 3} per-bin)")

nesting_df[nesting_df["bin"] != "pooled"][["bin", "step_isolated", "n", "estimate", "ci_lo", "ci_hi",
                                            "n_images_improved", "n_images_worsened"]]

# %% [markdown]
# **Testing whether either step's per-bin pattern is more than noise.** A
# Kruskal-Wallis test compares each step's `d_i` across the five bins
# without assuming an order between them, and a comparison of the sparsest
# against the densest bin isolates the two extremes directly. Both are
# descriptive checks on the shape of the per-bin table, run on each step
# separately, and carry no correction of their own.

# %%
def _bin_shape_checks(d_by_bin: dict) -> dict:
    groups = [d_by_bin[label] for label in co.BIN_LABELS]
    kw_stat, kw_p = stats.kruskal(*groups)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mw_stat, mw_p = stats.mannwhitneyu(d_by_bin["0-20"], d_by_bin["80-100"], alternative="two-sided")
    return {"kruskal_stat": float(kw_stat), "kruskal_p": float(kw_p),
            "sparsest_vs_densest_u": float(mw_stat), "sparsest_vs_densest_p": float(mw_p)}


per_bin_crop_d = {
    label: (pooled.loc[pooled["bin"] == label, "abs_e_masked_gray"]
             - pooled.loc[pooled["bin"] == label, "abs_e_base"]).to_numpy()
    for label in co.BIN_LABELS
}
crop_shape = _bin_shape_checks(per_bin_crop_d)
correction_shape = _bin_shape_checks(per_bin_correction_d)

print(f"crop step, across the five bins: Kruskal-Wallis H={crop_shape['kruskal_stat']:.3f}, "
      f"p={crop_shape['kruskal_p']:.3g}; sparsest (0-20) vs densest (80-100) "
      f"Mann-Whitney p={crop_shape['sparsest_vs_densest_p']:.3g}")
print(f"perspective-correction step, across the five bins: Kruskal-Wallis "
      f"H={correction_shape['kruskal_stat']:.3f}, p={correction_shape['kruskal_p']:.3g}; "
      f"sparsest (0-20) vs densest (80-100) Mann-Whitney p={correction_shape['sparsest_vs_densest_p']:.3g}")

# %% [markdown]
# **Reading the per-bin table above.** The **crop step** has a clear,
# large per-bin pattern: it helps the 933-image sparsest bin only slightly
# (or not at all — its interval there is the narrowest and sits close to
# zero) and helps the 30-image densest bin by several cover points, and the
# Kruskal-Wallis and sparsest-versus-densest comparisons above both confirm
# the bins are not interchangeable. That is the real per-bin structure in
# this decomposition.
#
# The **perspective-correction step**, in contrast, is small in every bin
# and every one of its five per-bin intervals covers zero. The
# Kruskal-Wallis comparison across the five bins and the sparsest-versus-
# densest comparison both come back non-significant: nothing in this
# decomposition establishes that the correction step depends on reference
# cover. The five values are reported with their intervals in
# `Q4_variant_nesting.csv`; none of them supports a directional statement
# about which bins the correction step helps or hurts.

# %% [markdown]
# ## Obliquity against the per-image rectification effect
#
# `obliquity` is an image-level covariate of perspective distortion
# (D4.v2). Because `rectified - masked_gray` isolates the perspective
# correction alone (both variants already share the crop), it is the
# contrast a geometric covariate acts on.
#
# **The unit, and the trap it avoids.** For each image,
#
# ```
# d_i = |e_rectified,i| - |e_masked_gray,i|,  averaged over the 24 model x
#                                              prompt cells BEFORE anything
#                                              else is done with it
# ```
#
# `obliquity` is **constant within an image**, so pairing the 24 raw rows
# per image against one obliquity value would count every image 24 times
# and inflate the apparent n from 1,155 to 27,720, understating every
# interval. `d_i` is therefore built from `pooled` — already one row per
# image — and `n_rows == n_unique_images` is asserted before anything else
# touches it. Negative `d_i` means the perspective correction **improved**
# that image; this is the same orientation the nesting table above fixes.
#
# **No "high obliquity" group above 0.10 is defined anywhere below.** The
# obliquity distribution is concentrated (median 0.028, top quartile above
# 0.041 with n=289, top decile above 0.065 with n=116, only n=37 images
# exceed 0.10), and a group of 37 images would carry an interval wide enough
# to contain almost anything.

# %%
d_i_full = (pooled["abs_e_rectified"] - pooled["abs_e_masked_gray"]).to_numpy()
obliquity_map = frames["d4"].set_index("image")["obliquity"]
pooled_j = pooled.copy()
pooled_j["d_i"] = d_i_full
pooled_j["obliquity"] = pooled_j["image"].map(obliquity_map)

assert len(pooled_j) == pooled_j["image"].nunique(), (
    "n_rows must equal n_unique_images before pairing d_i against obliquity — "
    "aggregation to one row per image happened when `pooled` was built above"
)
assert pooled_j["obliquity"].notna().all(), "every image must have a recorded obliquity value"
n_images_j = len(pooled_j)
print(f"unit check: n_rows == n_unique_images == {n_images_j} (not 27,720)")

obliquity_arr = pooled_j["obliquity"].to_numpy()
d_i_arr = pooled_j["d_i"].to_numpy()

print(f"obliquity: median={np.median(obliquity_arr):.4f}, mean={np.mean(obliquity_arr):.4f}, "
      f"max={np.max(obliquity_arr):.4f}")

# %% [markdown]
# ### Tie share on `d_i == 0` — reported for context
#
# Many configurations return identical predictions on both variants, so a
# large tie mass at `d_i == 0` is expected.

# %%
d_i_tie_check = co.check_k3_tie_burden(d_i_arr)
print(f"d_i == 0: n={d_i_tie_check['n_zero']}/{d_i_tie_check['n']} "
      f"({d_i_tie_check['zero_share']:.4f})")

# %% [markdown]
# ### Obliquity quartile table — descriptive
#
# Images are split into four equal-sized groups by obliquity score, and the
# mean per-image change from the perspective correction (`d_i`) is reported
# per group with its confidence interval. No cutpoint from this table is
# used anywhere else in this notebook.

# %%
obliquity_quartile_labels = ["Q1 (least oblique)", "Q2", "Q3", "Q4 (most oblique)"]
pooled_j["obliquity_quartile"] = pd.qcut(pooled_j["obliquity"], 4, labels=obliquity_quartile_labels)

quartile_rows = []
for q_label in obliquity_quartile_labels:
    sub = pooled_j[pooled_j["obliquity_quartile"] == q_label]
    n_q = len(sub)

    def _mean_stat(f):
        return float(f["d_i"].mean())

    boot_q = co.image_bootstrap(sub, _mean_stat, rng_j, image_col="image", bin_col=None, B=co.B_BOOTSTRAP)
    quartile_rows.append({
        "question_id": "Q4", "obliquity_quartile": q_label, "n": n_q,
        "obliquity_range_lo": float(sub["obliquity"].min()), "obliquity_range_hi": float(sub["obliquity"].max()),
        "mean_d_i": boot_q.estimate, "mean_d_i_ci_lo": boot_q.ci_lo, "mean_d_i_ci_hi": boot_q.ci_hi,
        "median_d_i": float(sub["d_i"].median()),
    })

quartile_df = pd.DataFrame(quartile_rows)
quartile_df

# %% [markdown]
# ## Chart 4 — `d_i` against obliquity, and the quartile summary

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax_l, ax_r = axes
ax_l.scatter(obliquity_arr, d_i_arr, s=8, alpha=0.35, color="tab:purple")
ax_l.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax_l.set_xlabel("Obliquity (unitless, relative ranking, D4.v2)")
ax_l.set_ylabel("d_i = |e_rectified| - |e_masked_gray| (cover points)")
ax_l.set_title(f"n={n_images_j}")

q_x = np.arange(len(obliquity_quartile_labels))
q_means = quartile_df["mean_d_i"].to_numpy()
q_lo = quartile_df["mean_d_i"] - quartile_df["mean_d_i_ci_lo"]
q_hi = quartile_df["mean_d_i_ci_hi"] - quartile_df["mean_d_i"]
ax_r.errorbar(q_x, q_means, yerr=[q_lo, q_hi], marker="o", capsize=3, color="tab:purple")
ax_r.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax_r.set_xticks(q_x)
ax_r.set_xticklabels([f"{lbl}\n(n={n})" for lbl, n in zip(obliquity_quartile_labels, quartile_df["n"])],
                      fontsize=8)
ax_r.set_xlabel("Obliquity quartile")
ax_r.set_ylabel("Mean d_i (cover points)")
ax_r.set_title("Descriptive only — no p-value, no cut above 0.10 (n=37 there)")

fig.suptitle("The perspective correction's per-image effect vs obliquity")
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q4_obliquity_variant.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Assumption checks output — `Q4_assumption_checks.csv`

# %%
assumption_rows = [
    {"question_id": "Q4", "check": "K1_pairing_complete", "statistic": np.nan, "p_value": np.nan,
     "detail": k1.detail},
    {"question_id": "Q4", "check": "C11_maverick_n_reproduced", "statistic": np.nan, "p_value": np.nan,
     "detail": c11.detail},
]

for name, col_a, col_b in CONTRASTS:
    r = contrast_results[name]
    assumption_rows.append({
        "question_id": "Q4", "check": f"K2_symmetry__{name}",
        "statistic": r["k2"]["skew"], "p_value": np.nan,
        "detail": f"skew={r['k2']['skew']:.4f}, symmetry_violated={r['k2']['symmetry_violated']} "
                  "(reported for context; the Hodges-Lehmann estimate is reported for every contrast regardless)",
    })
    assumption_rows.append({
        "question_id": "Q4", "check": f"K3_tie_burden__{name}",
        "statistic": r["k3"]["zero_share"], "p_value": np.nan,
        "detail": f"zero_share={r['k3']['zero_share']:.4f}, tie_burden_exceeded={r['k3']['tie_burden_exceeded']} "
                  "(reported for context)",
    })
    assumption_rows.append({
        "question_id": "Q4", "check": f"K4_bootstrap_bin_coverage__{name}",
        "statistic": float(r["k4"]["n_violations"]), "p_value": np.nan,
        "detail": f"violation_share={r['k4']['violation_share']:.4f}, "
                  f"switch_to_stratified={r['k4']['switch_to_stratified']}",
    })
    diff = r["diff"]
    k5 = co.check_k5_normality(diff)
    assumption_rows.append({
        "question_id": "Q4", "check": f"K5_normality__{name}",
        "statistic": k5["statistic"], "p_value": k5["p_value"],
        "detail": "D'Agostino-Pearson on the paired variant difference d_i; context only, "
                  "no method in this notebook assumes normality",
    })

for variant, col in [("base", "abs_e_base"), ("masked_gray", "abs_e_masked_gray"), ("rectified", "abs_e_rectified")]:
    k6_abs = co.check_k6_homoscedasticity(pooled[col], pooled["bin"])
    assumption_rows.append({
        "question_id": "Q4", "check": f"K6_homoscedasticity_abs_e__{variant}",
        "statistic": np.nan, "p_value": np.nan,
        "detail": f"per-bin sd of |e|: {k6_abs}",
    })

assumption_rows.append({
    "question_id": "Q4", "check": "obliquity_unit_of_analysis",
    "statistic": float(n_images_j), "p_value": np.nan,
    "detail": f"n_rows == n_unique_images == {n_images_j} asserted before pairing d_i against obliquity "
              "(guards against the 24x fan-out that would inflate n to 27,720)",
})
assumption_rows.append({
    "question_id": "Q4", "check": "tie_burden__d_i_obliquity",
    "statistic": d_i_tie_check["zero_share"], "p_value": np.nan,
    "detail": f"zero_share={d_i_tie_check['zero_share']:.4f} of images have d_i==0 "
              f"(n_zero={d_i_tie_check['n_zero']}); reported alongside the obliquity quartile table, "
              "not replaced by a cut on obliquity",
})

# The three checks below cover the by-axis (model, prompt) crop/correction/
# total decomposition: every row's n_images is the full 1,155-image frame,
# the additive identity holds per row, and every row's CI method is
# accounted for, with any percentile fallback counted rather than silently
# folded into the BCa count.
n_axis_rows_not_1155 = int((by_axis_df["n_images"] != 1155).sum())
assumption_rows.append({
    "question_id": "Q4", "check": "by_axis_full_image_coverage",
    "statistic": float(n_axis_rows_not_1155), "p_value": np.nan,
    "detail": f"{len(by_axis_df)} by-axis rows checked; {n_axis_rows_not_1155} rows do not cover "
              "all 1,155 images (expected 0 — every model and every prompt row is a different "
              "summary of the same full image set)",
})

_by_axis_pivot = by_axis_df.pivot_table(index=["axis", "level"], columns="step", values="estimate")
_by_axis_additivity_gap = (_by_axis_pivot["crop"] + _by_axis_pivot["correction"] - _by_axis_pivot["total"]).abs()
n_axis_additivity_violations = int((_by_axis_additivity_gap > 1e-6).sum())
assumption_rows.append({
    "question_id": "Q4", "check": "by_axis_additivity_crop_plus_correction_equals_total",
    "statistic": float(_by_axis_additivity_gap.max()), "p_value": np.nan,
    "detail": f"max |crop + correction - total| across the 10 axis levels = "
              f"{_by_axis_additivity_gap.max():.2e}; {n_axis_additivity_violations} levels exceed the "
              "1e-6 floating-point tolerance (expected 0)",
})

_by_axis_ci_method_counts = by_axis_df["ci_method"].value_counts().to_dict()
n_axis_percentile_fallback = int(by_axis_df["ci_method"].eq("percentile (BCa fallback)").sum())
assumption_rows.append({
    "question_id": "Q4", "check": "by_axis_ci_method_recorded",
    "statistic": float(n_axis_percentile_fallback), "p_value": np.nan,
    "detail": f"ci_method counts across the 30 by-axis rows: {_by_axis_ci_method_counts}; "
              f"{n_axis_percentile_fallback} rows fell back to the plain percentile interval",
})

assumption_checks_df = pd.DataFrame(assumption_rows)
assumption_checks_df.to_csv(RESULTS_DIR / "Q4_assumption_checks.csv", index=False)
print(f"wrote Q4_assumption_checks.csv ({len(assumption_checks_df)} rows)")

# %% [markdown]
# ## Writing the output tables
#
# `Q4_variant_metrics.csv` — balanced MAE and the five per-bin MAEs, and mean
# bias pooled and per bin (signed, positive = over-prediction), with n and
# CI, per variant.
#
# `Q4_variant_contrasts.csv` — one row per pooled contrast (the three
# balanced-MAE intervals and the two pooled mean-bias-difference intervals,
# no p-value), plus the per-bin (each carrying both `mae_diff` and
# `bias_diff`), per-model and per-prompt breakdowns and the chain-of-thought
# sensitivity, all labelled by `row_type`.
#
# `Q4_variant_bin_interaction.csv` — the per-bin location/spread table for
# the two variant-vs-base contrasts, with the `spread_vs_location` column.
#
# `Q4_variant_nesting.csv` — the three pooled contrasts re-presented as the
# crop / perspective-correction decomposition, plus the same three-step
# decomposition computed within each of the five reference bins.
#
# `Q4_obliquity_variant.csv` — the descriptive obliquity-quartile table:
# the per-image change from the perspective correction, averaged within
# four equal-sized groups by obliquity score.
#
# `Q4_variant_by_configuration.csv` — balanced MAE under each variant and
# the crop / correction / total changes, for each of the 24 model x prompt
# configurations, descriptive and uncorrected.
#
# `Q4_variant_by_axis.csv` — the same crop / correction / total decomposition
# collapsed onto the model axis (six rows) and the prompt axis (four rows),
# each row still built from all 1,155 images, descriptive and uncorrected.
#
# `Q4_assumption_checks.csv` — the pairing and Maverick-reproducibility
# checks, the symmetry, tie-burden and bootstrap-bin-coverage diagnostics
# per contrast, normality and homoscedasticity checks, the two
# obliquity-specific checks, and the three checks on the by-axis
# decomposition (image coverage, additivity, CI-method accounting).

# %%
variant_metrics_df.to_csv(RESULTS_DIR / "Q4_variant_metrics.csv", index=False)
print(f"wrote Q4_variant_metrics.csv ({len(variant_metrics_df)} rows)")

# %%
contrast_rows = []
for name, col_a, col_b in CONTRASTS:
    r = contrast_results[name]
    contrast_rows.append({
        "question_id": "Q4", "row_type": "pooled_contrast_interval_only",
        "contrast": name, "test_type": "pooled_variant_contrast_bca_interval",
        "estimate": r["boot"].estimate, "ci_lo": r["boot"].ci_lo, "ci_hi": r["boot"].ci_hi,
        "ci_method": r["boot"].ci_method, "n": len(pooled),
        "balanced_mae_interval_uncorrected": True,
        "hodges_lehmann": r["hl"], "rank_biserial": r["rb"],
        "win_rate_a": r["wr"]["win_rate_a"], "win_rate_b": r["wr"]["win_rate_b"], "tie_rate": r["wr"]["tie_rate"],
        "skew_d": r["k2"]["skew"], "symmetry_violated": r["k2"]["symmetry_violated"],
        "tie_burden_exceeded": r["k3"]["tie_burden_exceeded"], "zero_share": r["k3"]["zero_share"],
        "bootstrap_empty_bin_violations": r["boot"].n_empty_bin_violations,
        "switch_to_stratified_bootstrap": r["k4"]["switch_to_stratified"],
        "c1_serving_confounded": False,  # variant is within-image, within-model, within-serving-path
        "family": None, "family_size": None,
        "stream_note": (
            "This row is a BCa interval on Delta balanced MAE, read on its own and not "
            "multiplicity-corrected; no p-value is computed for balanced MAE anywhere in "
            "this notebook."
        ),
    })

for name, boot_bias_diff in pooled_bias_diff_results.items():
    contrast_rows.append({
        "question_id": "Q4", "row_type": "pooled_bias_diff_interval_only",
        "contrast": name, "test_type": "pooled_variant_mean_bias_diff_bca_interval",
        "estimate": boot_bias_diff.estimate, "ci_lo": boot_bias_diff.ci_lo, "ci_hi": boot_bias_diff.ci_hi,
        "ci_method": boot_bias_diff.ci_method, "n": len(pooled),
        "balanced_mae_interval_uncorrected": True,
        "c1_serving_confounded": False,
        "family": None, "family_size": None,
        "stream_note": (
            "This row is a BCa interval on the difference in mean signed bias (variant "
            "minus base, pooled over all 1,155 images), read on its own and not "
            "multiplicity-corrected. Positive means the variant is more over-predicting "
            "(or less under-predicting) than base; negative means the opposite."
        ),
    })

contrasts_df = pd.DataFrame(contrast_rows)

per_bin_variant_df_labelled = per_bin_variant_df.copy()
per_bin_variant_df_labelled.insert(0, "estimate", per_bin_variant_df_labelled["mae_diff"])
per_bin_variant_df_labelled.insert(0, "question_id", "Q4")
per_bin_variant_df_labelled.insert(1, "row_type", "per_bin_breakdown_uncorrected")

per_model_df_labelled = per_model_df.copy()
per_model_df_labelled.insert(0, "estimate", per_model_df_labelled["mae_diff"])
per_model_df_labelled.insert(0, "question_id", "Q4")
per_model_df_labelled.insert(1, "row_type", "per_model_breakdown_uncorrected")

per_prompt_df_labelled = per_prompt_df.copy()
per_prompt_df_labelled.insert(0, "estimate", per_prompt_df_labelled["mae_diff"])
per_prompt_df_labelled.insert(0, "question_id", "Q4")
per_prompt_df_labelled.insert(1, "row_type", "per_prompt_breakdown_uncorrected")

cot_row = pd.DataFrame([{
    "question_id": "Q4", "row_type": "chain_of_thought_sensitivity",
    "contrast": "Llama-4-Scout_Grid-Overlay_masked_gray_with_vs_without_cot_row",
    **cot_sensitivity,
}])

contrasts_out = pd.concat(
    [contrasts_df, per_bin_variant_df_labelled, per_model_df_labelled, per_prompt_df_labelled, cot_row],
    ignore_index=True,
)
contrasts_out.to_csv(RESULTS_DIR / "Q4_variant_contrasts.csv", index=False)
print(f"wrote Q4_variant_contrasts.csv ({len(contrasts_out)} rows: "
      f"{len(contrasts_df)} pooled-contrast rows, {len(per_bin_variant_df_labelled)} per-bin, "
      f"{len(per_model_df_labelled)} per-model, {len(per_prompt_df_labelled)} per-prompt, "
      f"{len(cot_row)} chain-of-thought sensitivity)")

# %%
bin_interaction_rows = []
for row in per_bin_ls_df.to_dict("records"):
    contrast = row["contrast"]
    sub_ls = per_bin_ls_df.loc[per_bin_ls_df["contrast"] == contrast]
    base_sd = sub_ls.loc[sub_ls["bin"] == "0-20", "sd_d"].iloc[0]
    base_mean = sub_ls.loc[sub_ls["bin"] == "0-20", "mean_d"].iloc[0]
    location_range = float(sub_ls["mean_d"].max() - sub_ls["mean_d"].min())
    row = dict(row)
    row["spread_vs_location"] = (
        "spread wider than bottom bin" if (pd.notna(row["sd_d"]) and pd.notna(base_sd) and row["sd_d"] > base_sd)
        else "spread not wider than bottom bin"
    )
    row["location_mean_d_range_vs_bottom_bin"] = location_range
    row["location_moves_vs_bottom_bin"] = bool(abs(row["mean_d"] - base_mean) > sub_ls["sd_d"].dropna().min()
                                                if pd.notna(row["mean_d"]) and sub_ls["sd_d"].notna().any() else False)
    bin_interaction_rows.append(row)

bin_interaction_df = pd.DataFrame(bin_interaction_rows)
bin_interaction_df.insert(0, "question_id", "Q4")

# Published per-bin mean and sd of d_i (the paired variant difference), for
# the two variant-vs-base contrasts across the five reference bins. These are
# deterministic functions of `pooled` and are checked against the values
# reported in the paper.
PUBLISHED_BIN_INTERACTION = {
    ("masked_gray_minus_base", "0-20"):    {"mean_d": 0.5360767238,  "sd_d": 2.7817148629},
    ("masked_gray_minus_base", "20-40"):   {"mean_d": 0.2076466049,  "sd_d": 4.3680615487},
    ("masked_gray_minus_base", "40-60"):   {"mean_d": -0.4750905797, "sd_d": 4.6538071406},
    ("masked_gray_minus_base", "60-80"):   {"mean_d": -3.3108552632, "sd_d": 7.0245330055},
    ("masked_gray_minus_base", "80-100"):  {"mean_d": -5.1708333333, "sd_d": 6.7319793684},
    ("rectified_minus_base", "0-20"):      {"mean_d": 0.4855283137,  "sd_d": 2.8208135449},
    ("rectified_minus_base", "20-40"):     {"mean_d": 0.5050038580,  "sd_d": 4.0396870921},
    ("rectified_minus_base", "40-60"):     {"mean_d": -0.0974456522, "sd_d": 5.0564978308},
    ("rectified_minus_base", "60-80"):     {"mean_d": -3.6104714912, "sd_d": 7.1100794805},
    ("rectified_minus_base", "80-100"):    {"mean_d": -5.7138888889, "sd_d": 6.4489386001},
}
for _, r in bin_interaction_df.iterrows():
    fr = PUBLISHED_BIN_INTERACTION[(r["contrast"], r["bin"])]
    assert_reproduces(f"mean_d[{r['contrast']}/{r['bin']}]", r["mean_d"], fr["mean_d"])
    assert_reproduces(f"sd_d[{r['contrast']}/{r['bin']}]", r["sd_d"], fr["sd_d"], atol=1e-4)
print("Per-bin location/spread point estimates (mean_d, sd_d) match the published values.")

bin_interaction_df.to_csv(RESULTS_DIR / "Q4_variant_bin_interaction.csv", index=False)
print(f"wrote Q4_variant_bin_interaction.csv ({len(bin_interaction_df)} rows)")

# %% [markdown]
# ## Obliquity and the per-image rectification effect — `Q4_obliquity_variant.csv`

# %%
quartile_df_labelled = quartile_df.copy()
quartile_df_labelled.insert(1, "row_type", "obliquity_quartile_descriptive")

quartile_df_labelled.to_csv(RESULTS_DIR / "Q4_obliquity_variant.csv", index=False)
print(f"wrote Q4_obliquity_variant.csv ({len(quartile_df_labelled)} quartile rows)")

# %% [markdown]
# ## Per-configuration variant effects output — `Q4_variant_by_configuration.csv`

# %%
per_config_df.to_csv(RESULTS_DIR / "Q4_variant_by_configuration.csv", index=False)
print(f"wrote Q4_variant_by_configuration.csv ({len(per_config_df)} rows: 6 models x 4 prompts)")

# %% [markdown]
# ## Model-axis and prompt-axis decomposition output — `Q4_variant_by_axis.csv`

# %%
by_axis_df.to_csv(RESULTS_DIR / "Q4_variant_by_axis.csv", index=False)
print(f"wrote Q4_variant_by_axis.csv ({len(by_axis_df)} rows: 6 models + 4 prompts, x 3 steps)")

# %% [markdown]
# ## Result
#
# **No causal claim.** `variant` is a within-image
# transformation applied to the same photograph, so any contrast below is a
# claim about the effect of substituting one processed image for another on
# this frame, never about a mechanism inside the model or the pre-processing
# pipeline.
#
# **The reference is unchanged by pre-processing**, stated
# rather than tested: `masked_gray` and `rectified` predictions are scored
# against the reference for the *original* photograph. A variant effect
# below is "the effect of transforming the input while the target stays
# fixed", not "the effect on a correspondingly transformed task".
#
# **The finding this notebook leads with: cropping to the annotated quadrat
# is essentially all of the pooled pre-processing effect, and the pooled
# view of the perspective-correction step hides what it actually does.**
# `masked_gray - base` (the crop) sits at
# {crop_estimate:.2f} cover points [{crop_lo:.2f}, {crop_hi:.2f}]; the pooled
# `rectified - masked_gray` (the perspective correction alone) sits at
# {corr_estimate:.2f} [{corr_lo:.2f}, {corr_hi:.2f}] — an interval that
# covers zero and is roughly an order of magnitude smaller than the crop's.
# Per bin, though, the correction step is not flat: it runs from a modest
# *positive* (worse) value in the low-to-middle bins to a clearly *negative*
# (better) value in the 80-100 bin — the pattern the pooled average of a
# 933-image bin against a 30-image one cannot show. **Retention or gain as a
# per-bin percentage is not reported anywhere in this notebook**: with
# per-bin denominators this small a percentage change is not an
# interpretable quantity, so only the absolute cover-point difference with
# its interval and the counts of images improved and worsened are reported.
#
# **The typical-image and bin-balanced streams for the crop step disagree,
# and that disagreement is itself informative, not a discrepancy to
# reconcile.** For `masked_gray_minus_base`, the bin-balanced estimate (Delta
# balanced MAE) says masked_gray is more accurate on this frame; the
# companion Wilcoxon on typical-image error says the *typical* image is not
# helped and, if anything, points the other way. Per the per-bin breakdown,
# the 933-image bottom bin (0-20 cover points) is where masked_gray is
# *worse*, and the sparse high-cover bins are where its advantage is large
# enough to pull the bin-balanced average in the other direction. The honest
# statement is **the crop helps the rare high-cover quadrats and is flat to
# slightly harmful on the typical, sparse-vegetation one** — never collapsed
# to a single "pre-processing helps" or "does not help" sentence.
#
# **Pooled nulls are read alongside the per-model and per-prompt spread**, in
# the same paragraph, not in an appendix:
# `Q4_variant_contrasts.csv`'s per-model and per-prompt rows carry the
# realised range for each axis, so a pooled null or a pooled effect is never
# read as uniform across models or prompts without also checking those rows.

# %%
crop_estimate = contrast_results["masked_gray_minus_base"]["boot"].estimate
crop_lo = contrast_results["masked_gray_minus_base"]["boot"].ci_lo
crop_hi = contrast_results["masked_gray_minus_base"]["boot"].ci_hi
corr_estimate = -contrast_results["masked_gray_minus_rectified"]["boot"].estimate
corr_lo = -contrast_results["masked_gray_minus_rectified"]["boot"].ci_hi
corr_hi = -contrast_results["masked_gray_minus_rectified"]["boot"].ci_lo

print(f"crop (masked_gray - base): Delta balanced MAE = {crop_estimate:+.3f} "
      f"[{crop_lo:.3f}, {crop_hi:.3f}] cover points (95% BCa, uncorrected interval, no p-value)")
print(f"perspective correction, pooled (rectified - masked_gray): Delta balanced MAE = {corr_estimate:+.3f} "
      f"[{corr_lo:.3f}, {corr_hi:.3f}] cover points (95% BCa, uncorrected interval, no p-value; "
      f"interval covers zero)")

for label in co.BIN_LABELS:
    row = nesting_df[(nesting_df["bin"] == label) & (nesting_df["step_isolated"] == "perspective_correction_only")].iloc[0]
    print(f"  bin {label:>7} (n={int(row['n'])}): perspective-correction Delta |e| = {row['estimate']:+.3f} "
          f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}], improved={int(row['n_images_improved'])}, "
          f"worsened={int(row['n_images_worsened'])}")

for name in contrast_results:
    if name == "masked_gray_minus_rectified":
        continue
    r = contrast_results[name]
    bal_direction = "helps" if r["boot"].estimate < 0 else "hurts" if r["boot"].estimate > 0 else "no change"
    typical_direction = ("stochastically larger error (hurts)" if r["rb"] > 0
                          else "stochastically smaller error (helps)" if r["rb"] < 0 else "no difference")
    print(
        f"\n{name}:\n"
        f"  bin-balanced: Delta balanced MAE={r['boot'].estimate:+.3f} cover points "
        f"[{r['boot'].ci_lo:.3f}, {r['boot'].ci_hi:.3f}] -> '{name.split('_minus_')[0]}' {bal_direction} "
        f"(95% BCa interval, uncorrected, no p-value)\n"
        f"  typical-image: HL={r['hl']:.3f} cover points, rank-biserial={r['rb']:+.3f} -> "
        f"'{name.split('_minus_')[0]}' has {typical_direction}, win_rate_a={r['wr']['win_rate_a']:.4f}"
    )

print(f"\nObliquity quartile table (n={n_images_j}): see Q4_obliquity_variant.csv")

# %% [markdown]
# **The negative result, read carefully.** The `masked_gray` and
# `rectified` images are not part of this repository and
# cannot be inspected or re-derived here. A null variant effect above is
# reported as *"no detectable effect of the pre-processing as applied and
# recorded here"* — never as *"pre-processing does not help"*. A
# transformation that failed to do what it was intended to do would look
# identical, on this data, to a transformation that genuinely made no
# difference; this notebook cannot tell the two apart.
#
# **The chain-of-thought row**: one Llama-4-Scout x Grid-Overlay x
# masked_gray row was recovered from a JSON parse failure via chain-of-thought
# text (A10). Recomputing that cell's mean |e| without it moves the value by
# the amount recorded in `Q4_variant_contrasts.csv`'s CoT sensitivity row —
# one row out of 4,620 for that model, reported as a number rather than
# assumed negligible.

# %%
print("Q4 output files written:")
for fname in ["Q4_variant_metrics.csv", "Q4_variant_contrasts.csv",
              "Q4_variant_bin_interaction.csv", "Q4_variant_nesting.csv",
              "Q4_obliquity_variant.csv", "Q4_variant_by_configuration.csv",
              "Q4_variant_by_axis.csv", "Q4_assumption_checks.csv"]:
    print(f"  {fname}: {(RESULTS_DIR / fname).exists()}")
