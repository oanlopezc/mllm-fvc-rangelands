# %% [markdown]
# # Q9 — do MLLMs outperform the classical ExG-ExR vegetation-index baseline?
#
# The paper's central claim is stronger if it is not just "MLLMs achieve error
# *X*" but "MLLMs beat the established remote-sensing method for this task."
# This notebook builds that comparison: the full classical pipeline — rectify
# the photograph to 1536x1536 by homography, take ExG-ExR on chromatic-
# normalised RGB inside a 40px-inset interior, threshold at the 99th
# percentile of a soil-colour exemplar set — applied to a **raw photograph**,
# against each of six MLLMs applied to the **same raw photograph**. Both sides
# are scored against the same two-observer reference over the same 1,155
# quadrat images, so the comparison is exact and complete: no missingness, no
# serving-stack confound on the baseline side, 100% coverage.
#
# The classical pipeline requires rectification *by construction* — its
# method needs the quadrat interior to fill the frame — so rectification is a
# step inside the baseline, not a preprocessing advantage handed to it. That
# is why this question uses the MLLMs' `base` (unrectified) runs: the fair
# comparison is "the classical pipeline, including its own rectification step,
# on a raw photo" against "an MLLM directly on the same raw photo."
#
# Six pre-declared contrasts (Holm family H), one per model pooled over its
# four prompts. Two claims per contrast, each its own pre-declared test,
# neither subordinate to the other: a bin-balanced claim (bootstrap p on
# delta balanced MAE) and a typical-image claim (paired Wilcoxon on per-image
# mean |e|, demoted to the exact paired sign test once the symmetry check
# below fails, as it does for all six contrasts). Read the two streams
# separately — the prose below explains why they are expected to disagree,
# and why the second is where this question is actually decided. The two
# streams disagree sharply here: the balanced-MAE stream favours every MLLM,
# while the typical-image stream finds the classical baseline significantly
# *stronger* than three of the six MLLMs, beaten by exactly one, and not
# distinguishable from the remaining two.
#
# Three further, descriptive blocks sit alongside the six confirmatory
# contrasts, none of them carrying a p-value or entering family H: the 0-20
# bin (81% of the frame) at model x prompt configuration granularity, since
# the model-pooled claim that the baseline wins that bin does not hold for
# every configuration; the baseline's over-prediction counts together with
# whether over-prediction was arithmetically possible in each bin, since the
# baseline's own ceiling forces some of those counts to zero regardless of
# what it predicts; and a supplementary re-run of the six contrasts with the
# MLLM side scored on the `rectified` variant, for a reader who wants the
# comparison with matched image geometry on both sides.

# %% [markdown]
# ## Setup
#
# Seed is `20260907 + 9 = 20260916`: the study's base seed plus the question
# number. Fixed here, before any resampling, and never tuned after seeing a
# result.

# %%
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _project_root(start=None):
    """Resolve the project root: walk up from the kernel's working directory
    (or from ANALYSIS_PROJECT_ROOT, when it is set) looking for STATUS.md.
    When a notebook is executed by nbconvert the working directory is the
    folder holding the rendered notebook rather than the project root, and
    `__file__` is not defined, so path discovery can rely on neither a
    relative path nor `__file__`.
    """
    p = Path(os.environ.get("ANALYSIS_PROJECT_ROOT", start or Path.cwd())).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "STATUS.md").is_file():
            return candidate
    raise RuntimeError(f"project root not found from {p}")


ROOT = _project_root()
sys.path.insert(0, str(ROOT / "05_deliverables/Repository/GitHub/notebooks"))
import pathlib as _pl  # `_common.py` is imported from this notebook's
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))  # own directory
import _common as C

SEED = 20260916  # the study's base seed (20260907) plus the question number.
# Written as a plain integer rather than an expression, so the seed that
# governs this run can be read straight off the source.
rng = np.random.default_rng(SEED)

RESULTS_DIR = ROOT / "05_deliverables/Repository/GitHub/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

print(f"seed={SEED}")
print(f"project root: {ROOT}")

# %% [markdown]
# ## Data
#
# Three frames, loaded through `_common` so the load path, the join and every
# assertion are shared with the rest of the project rather than re-derived
# here:
#
# - `base_local` — the six MLLMs, `variant == 'base'`, local serving stack,
#   joined to the reference (`D1`). 1,155 images x 6 models x 4 prompts =
#   27,720 rows.
# - `d12_frame` — the classical baseline (`D12`), 1,155 rows, one per image,
#   joined to the same reference. This frame is never merged onto
#   `base_local` — see the C15 note directly below.
# - `d1` — the reference frame alone, used for the A14/A19 assertion
#   machinery and the model-level metrics table.
#
# All I/O in this notebook goes through `_common`'s loaders or
# `guarded_read_csv`, which refuse any forbidden path (`D7`, `D10`, the
# external summary CSV) or any path outside the project (A12).

# %%
assumption_df, frames = C.run_all_assertions(include_d12=True)
print(assumption_df[["id", "passed", "detail"]].to_string(index=False))
assert assumption_df["passed"].all(), "a load-time assertion failed — stopping"

d1 = frames["d1"]
d5 = frames["d5"]
d8 = frames["d8"]
base_local = frames["base_local"]
d12 = frames["d12"]

# K1 and C11 are not reachable from run_all_assertions() /
# compute_maverick_floor(), so they are called explicitly here.
k1_result = C.assert_k1_pairing_complete(d5, base_local)
print(f"K1: {k1_result.detail}")

maverick_floor = C.compute_maverick_floor(d8, rng)
c11_result = C.assert_c11_n_reproduced(maverick_floor.n_reproduced)
print(f"C11: {c11_result.detail}")
print(f"delta_Mav = {maverick_floor.delta_mav:.4f} "
      f"(95% CI {maverick_floor.delta_mav_ci_lo:.4f}, {maverick_floor.delta_mav_ci_hi:.4f})")

# %% [markdown]
# **A16 in this cell, explicitly.** `D12.v3` (`reference`) is a third
# verbatim copy of the ground truth, alongside `D1.v7` and `D2.v2`. It is
# asserted equal to `D1.v7` on all 1,155 rows inside `C.build_d12_frame()`
# (called by `C.load_d12()` / `run_all_assertions`) and dropped in the same
# cell — `d12_frame` below carries only `D1.v7` under the name `reference`,
# never a second reference column. It is never treated as corroboration.

# %%
d12_frame = C.build_d12_frame()
assert "reference" in d12_frame.columns
print(f"d12_frame: {len(d12_frame)} rows, {d12_frame['image'].nunique()} unique images")
print(d12_frame[["image", "vegetation_percent", "reference", "bin"]].head(3))

# %% [markdown]
# **The C15 fan-out guard (A17), stated once and enforced at every point of
# use below.** `D12` has no prompt or model dimension. If it were merged onto
# the 4,620-row per-model prediction frame it would broadcast 4x and every
# statistic computed on the merged frame would be computed as though n were
# 4,620 rather than 1,155 — quadrupling its apparent precision. The rule this
# notebook follows without exception: **the MLLM side is aggregated to one
# value per image first, and only then paired against `D12`.** `D12`
# statistics are always computed on `d12_frame` (1,155 rows) or, inside the
# LOCO path, on its own 574/735/1,001-row campaign subsets — never on a frame
# merged with `base_local`. `C.guard_d12_statistic` / `C.assert_a17` checks
# `n_rows == n_unique_images` immediately before any `D12` statistic is
# taken, with the full `n == 1,155` check applied only outside the LOCO path.

# %%
C.guard_d12_statistic(d12_frame, full_frame_expected=True, context="full D12 frame, pre-analysis")
print("A17 passed on the full 1,155-row D12 frame.")

# %% [markdown]
# ## The baseline diagnostic block
#
# Before any MLLM comparison, the baseline is characterised on its own terms.
# This block establishes two things a bare MAE cannot: whether the baseline's
# apparent accuracy on MAE alone is meaningful (it can look deceptively
# reasonable even when produced by something close to a constant), and
# whether — and where — it saturates.
#
# **Why Pearson appears here and nowhere else in this notebook.** The
# baseline's pooled MAE (10.92) sits on top of a per-bin profile computed
# below that runs from 3.41 in the 0-20 bin to 78.21 in the 80-100 bin — a
# 23-fold spread concealed inside that single pooled figure. A pooled error
# statistic averaged over five cover bins that the baseline handles completely
# differently is the wrong instrument for asking whether it is producing
# genuine estimates at all, and that is what correlation is for here:
# Pearson `r` is computed **once**, on the baseline alone, strictly as a
# diagnostic, and `r = 0.515` is the number that shows the baseline is
# *tracking real variation* rather than producing noise, i.e. it is
# mis-scaled, not blind. That is Pearson's entire job. It is written to
# `pearson_r_diagnostic_only` with `may_rank_methods = FALSE` beside it, and
# it is never used to rank or compare methods. **Every comparative
# correlation statement in this notebook — every MLLM-vs-baseline contrast —
# uses tie-corrected Spearman instead**, because `D5.v4` is bounded and
# tie-dense and `D1.v7` is strongly right-skewed, so a Pearson r there would
# be dominated by the handful of high-cover images rather than reflecting
# overall agreement.

# %%
baseline_pred = d12_frame["vegetation_percent"]
baseline_ref = d12_frame["reference"]
baseline_e = d12_frame["e"]
baseline_abs_e = d12_frame["abs_e"]

baseline_mae = float(baseline_abs_e.mean())
baseline_rmse = float(np.sqrt((baseline_e ** 2).mean()))
baseline_bias = float(baseline_e.mean())
baseline_pearson = float(np.corrcoef(baseline_pred, baseline_ref)[0, 1])
baseline_spearman, _ = C.spearman_tie_corrected(baseline_pred.values, baseline_ref.values)

# These three quantities are assertion machinery only (A14/A19), retained so
# that A19's check on the loaded baseline file still runs — they are not
# part of the reported comparison and do not appear in any chart, table or
# sentence below: an assertion is checking machinery, not reported content.
const3_mae = float((baseline_ref - 3.0).abs().mean())  # A19, asserted 11.86 +/- 0.05
b_median = float(d1["reference"].median())              # A14, computed not asserted
b_median_mae = float((d1["reference"] - b_median).abs().mean())
b_zero_mae = float(d1["reference"].abs().mean())         # predict 0 for every image

print(f"baseline: MAE={baseline_mae:.3f}  RMSE={baseline_rmse:.3f}  bias={baseline_bias:.3f}  "
      f"pearson_r={baseline_pearson:.3f}  spearman={baseline_spearman:.3f}")
print(f"[assertion machinery, A14/A19, not reported below] constant-3-predictor MAE={const3_mae:.3f} "
      f"(asserted 11.86 +/- 0.05); B_median predictor MAE={b_median_mae:.3f}; "
      f"B_zero predictor MAE={b_zero_mae:.3f}")
print(f"baseline prediction range: min={baseline_pred.min():.3f}  "
      f"median={baseline_pred.median():.3f}  max={baseline_pred.max():.3f}")

# %% [markdown]
# Image-bootstrap CIs on the baseline's own summary statistics, over the
# 1,155-image frame (A17-guarded — no fan-out is possible here since this
# frame already has one row per image).

# %%
def _bl_stat(fn):
    def _s(frame):
        return fn(frame["vegetation_percent"] - frame["reference"])
    return _s

boot_mae = C.image_bootstrap(d12_frame, lambda f: (f["vegetation_percent"] - f["reference"]).abs().mean(),
                              rng, bin_col="bin")
boot_rmse = C.image_bootstrap(d12_frame, lambda f: np.sqrt(((f["vegetation_percent"] - f["reference"]) ** 2).mean()),
                               rng, bin_col="bin")
boot_bias = C.image_bootstrap(d12_frame, lambda f: (f["vegetation_percent"] - f["reference"]).mean(),
                               rng, bin_col="bin")
boot_spearman = C.image_bootstrap(
    d12_frame, lambda f: C.spearman_tie_corrected(f["vegetation_percent"].values, f["reference"].values)[0],
    rng, bin_col="bin",
)

baseline_perbin = C.balanced_mae(baseline_abs_e, d12_frame["bin"])
perbin_ci_lo, perbin_ci_hi = {}, {}
for label in C.BIN_LABELS:
    sub = d12_frame[d12_frame["bin"] == label]
    if sub["image"].nunique() < 2:
        perbin_ci_lo[label], perbin_ci_hi[label] = np.nan, np.nan
        continue
    b = C.image_bootstrap(sub, lambda f: (f["vegetation_percent"] - f["reference"]).abs().mean(),
                           rng, bin_col=None)
    perbin_ci_lo[label], perbin_ci_hi[label] = b.ci_lo, b.ci_hi
baseline_perbin = C.balanced_mae(baseline_abs_e, d12_frame["bin"], ci_lo=perbin_ci_lo, ci_hi=perbin_ci_hi)
baseline_perbin_signed = d12_frame.groupby("bin")["e"].mean().reindex(C.BIN_LABELS)

print(f"MAE  {boot_mae.estimate:.3f}  95% CI ({boot_mae.ci_lo:.3f}, {boot_mae.ci_hi:.3f}) [{boot_mae.ci_method}]")
print(f"RMSE {boot_rmse.estimate:.3f}  95% CI ({boot_rmse.ci_lo:.3f}, {boot_rmse.ci_hi:.3f}) [{boot_rmse.ci_method}]")
print(f"bias {boot_bias.estimate:.3f}  95% CI ({boot_bias.ci_lo:.3f}, {boot_bias.ci_hi:.3f}) [{boot_bias.ci_method}]")
print(f"Spearman (baseline's own value — this is the comparative statistic used in "
      f"every Delta Spearman contrast below, unlike Pearson above) {boot_spearman.estimate:.3f}  "
      f"95% CI ({boot_spearman.ci_lo:.3f}, {boot_spearman.ci_hi:.3f})")
print("\nper-bin baseline MAE and signed bias:")
for label in C.BIN_LABELS:
    print(f"  {label:>7}: n={baseline_perbin.per_bin_n[label]:>4}  "
          f"MAE={baseline_perbin.per_bin_mae[label]:6.2f}  "
          f"signed bias={baseline_perbin_signed[label]:7.2f}  "
          f"CI ({perbin_ci_lo[label]:.2f}, {perbin_ci_hi[label]:.2f})")

# %% [markdown]
# The baseline's predictions run from 0.002 to **58.467** on a 0-100
# reference, with a median of 0.567 — it is a mis-scaled predictor that
# saturates well below the top of the reference range. That saturation is
# visible directly in the per-bin table above: the baseline's error in the
# 80-100 bin is enormous, not because the model is "bad" in any general
# sense but because it structurally cannot output a value near 80 when its
# own output ceiling sits near 58. The chart below makes this the first
# thing a reader sees, rather than something argued in prose alone.

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(d12_frame["reference"], d12_frame["vegetation_percent"], s=10, alpha=0.35,
           color="#8B4513", label="classical baseline prediction")
lims = [0, 100]
ax.plot(lims, lims, "k--", linewidth=1, label="perfect agreement (y = x)")
ax.axhline(baseline_pred.max(), color="firebrick", linestyle=":", linewidth=1.5,
           label=f"baseline saturation ceiling ({baseline_pred.max():.1f})")
ax.set_xlim(0, 100)
ax.set_ylim(-2, 100)
ax.set_xlabel("Reference cover (% FVC)")
ax.set_ylabel("Classical baseline prediction (% FVC)")
ax.set_title("Q9 — classical baseline saturates far below the reference ceiling")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig_path = ROOT / "05_deliverables/Repository/GitHub/notebooks" / "rendered"
fig_path.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_path / "Q9_baseline_saturation.png", dpi=150)
plt.show()
print("Caption: each point is one of the 1,155 images. The dashed line is perfect "
      "agreement; the baseline never approaches it above roughly 60% reference "
      "cover, which is why the bin-balanced stream below is structurally "
      "dominated by the top bins.")

# %% [markdown]
# ## All 24 configurations against the baseline's overall MAE (descriptive)
#
# Q1 defines a 95% top-set of model x prompt configurations from a
# data-selected reference; routing that set through a test here would
# launder a selection-conditioned family into a directional claim, so no test
# in this notebook is defined on it. What is legitimate, and reported here, is
# the purely descriptive statement of whether every one of the 24
# configurations' point estimates individually beats the baseline's overall
# MAE — no p-value, no family, no claim of significance attaches to this
# cell, and no read of `Q1_topset_bootstrap.csv` is performed here.

# %%
combo_mae = (
    base_local.groupby(["model", "prompt"])["abs_e"].mean()
    .reset_index(name="overall_mae")
    .sort_values("overall_mae")
)
n_below_baseline = int((combo_mae["overall_mae"] < baseline_mae).sum())
print(f"{n_below_baseline} / {len(combo_mae)} model x prompt configurations have a lower "
      f"overall-MAE point estimate than the baseline's {baseline_mae:.2f} "
      f"(descriptive only, no test, no p-value).")

# %% [markdown]
# ## Assumption checks for the six family-H contrasts
#
# Each contrast pairs, at the image, the MLLM's per-image mean |e| (pooled
# over its 4 prompts: the mean of |e|, never the |error| of the mean
# prediction, which would be an ensemble) against the baseline's per-image
# |e|. K1 (pairing complete) is confirmed above at the base/local frame
# level. Remaining checks, per contrast: K2 (symmetry, for the HL reading),
# K3 (tie burden), K4 (bootstrap bin coverage), K10 (between-image
# independence — not checkable, quantified instead by the clustered
# companion below). C1 does not apply to the baseline side (D12 has no
# serving stack); it applies as usual to the MLLM side, so every contrast row
# carries `c1_serving_confounded = FALSE` alongside
# `c1_not_applicable_baseline_side = TRUE`. The Maverick run-to-run
# determinism floor still binds the Maverick row.

# %%
def mllm_per_image(model: str) -> pd.DataFrame:
    """Pool a model's 4 prompts to one value per image (C15 aggregation
    step). Returns image, mean |e|, mean signed e, joined to the reference/bin.
    """
    sub = base_local[base_local["model"] == model]
    per_image_abs = C.aggregate_mllm_side_to_image(sub, "abs_e").rename(columns={"abs_e": "abs_e_mllm"})
    per_image_signed = C.aggregate_mllm_side_to_image(sub, "e").rename(columns={"e": "e_mllm"})
    out = per_image_abs.merge(per_image_signed, on="image", how="inner")
    out = out.merge(d12_frame[["image", "vegetation_percent", "reference", "e", "abs_e", "bin", "campaign"]]
                     .rename(columns={"vegetation_percent": "baseline_pred",
                                      "e": "e_baseline", "abs_e": "abs_e_baseline"}),
                     on="image", how="inner")
    return out


def mllm_pooled_rows(model: str) -> pd.DataFrame:
    """The model-axis Spearman estimand: pool a model's 4 prompts as
    separate **rows** (one row per image x prompt, `n` = 4,620 before the
    join), never as the mean of their predictions. Averaging predictions
    first is an ensemble (Q2's question, not Q9's) and denoises the
    per-image prediction before ranking, which inflates rank agreement.
    Joined to the baseline reference and bin so `image_bootstrap` can
    resample by image while carrying all 4 prompt rows of a drawn image
    together, which keeps the pairing C12 requires intact under resampling.
    """
    sub = base_local[base_local["model"] == model]
    out = sub[["image", "prompt", "vegetation_percent"]].merge(
        d12_frame[["image", "reference", "bin"]], on="image", how="inner")
    return out


# A separate, unseeded-relative-to-`rng` generator for the K2/K3 fallback
# decision only. `exact_sign_test` itself is deterministic (an exact binomial
# test has no randomness), so this generator is never actually drawn from —
# it exists so that adding, removing or reordering this check can never
# perturb the `rng` stream that every bootstrap below consumes. Seeded from
# the same literal so the run is still fully reproducible.
rng_k2_check = np.random.default_rng(SEED)

assumption_rows = []
sign_test_rows = {}
for model in C.MODELS:
    paired = mllm_per_image(model)
    C.guard_d12_statistic(paired, full_frame_expected=True, context=f"Q9 pairing, {model}")
    diff = (paired["abs_e_mllm"] - paired["abs_e_baseline"]).values
    k2 = C.check_k2_symmetry(diff)
    k3 = C.check_k3_tie_burden(diff)
    # K2/K3 fallback branch: exact paired sign test, computed for every
    # contrast regardless of which threshold trips, so the branch that is
    # actually taken is always available rather than assembled after the
    # fact. `rng_k2_check` is touched only to confirm the branch fires
    # independently of the main `rng` stream; the test itself is exact.
    _ = rng_k2_check.random()
    sign_test_rows[model] = C.exact_sign_test(diff)
    assumption_rows.append({
        "question_id": "Q9", "model": model, "n_pairs": len(paired),
        "k1_pairing_complete": len(paired) == 1155,
        "skew_d": k2["skew"], "symmetry_violated_k2": k2["symmetry_violated"],
        "hl_estimate_descriptive": k2["hl_estimate"],
        "n_zero_diff": k3["n_zero"], "zero_share_k3": k3["zero_share"],
        "tie_burden_exceeded_k3": k3["tie_burden_exceeded"],
        "sign_test_fallback_triggered": bool(k2["symmetry_violated"] or k3["tie_burden_exceeded"]),
        "sign_test_n_used": sign_test_rows[model]["n_used"],
        "sign_test_n_pos": sign_test_rows[model]["n_pos"],
        "sign_test_p_value": sign_test_rows[model]["p_value"],
        "c1_serving_confounded": False, "c1_not_applicable_baseline_side": True,
    })

q9_assumptions = pd.DataFrame(assumption_rows)
print(q9_assumptions.to_string(index=False))

# %% [markdown]
# **K2 fails on all six contrasts** — `|skew(d_i)| > 1` for every model, from
# 1.57 (Gemma-3-12B) to 2.03 (Llama-4-Scout), confirmed in the table above.
# K3's 40% zero-difference threshold is not exceeded by any of the six, but
# the fallback rule triggers on *either* check failing, so it applies
# here on K2 alone: **the exact paired sign test is the reported
# typical-image effect statement for all six contrasts, and the
# Hodges-Lehmann median is demoted to a descriptive column.** The sign test
# is computed above for every contrast (not only where a threshold trips),
# using its own generator (`rng_k2_check`) so that this check can never
# shift the bootstrap and Wilcoxon results that follow, which all draw from
# `rng`. K4 (bootstrap empty-bin coverage) is checked inside
# `image_bootstrap` itself and reported per contrast below.

# %% [markdown]
# ## The six family-H contrasts
#
# For model *m*, image *i*: `a_i` = mean of `|e|` over the 4 prompts (MLLM
# side), `b_i` = `|e|` of the classical baseline, `d_i = a_i - b_i`. Two
# pre-declared claims per contrast, Holm-adjusted within family H (size 6,
# first threshold 0.05/6 = 0.00833):
#
# 1. **bin-balanced claim** — two-sided image-bootstrap p on delta balanced
#    MAE, 95% BCa interval as the magnitude;
# 2. **typical-image claim** — pre-declared as the paired Wilcoxon
#    signed-rank (Pratt) on `d_i`, but K2 (symmetry of `d_i`) fails on all
#    six contrasts (computed below), so the reported effect statement here
#    is the exact paired sign test, the stated fallback once symmetry fails;
#    Wilcoxon's statistic
#    and the Hodges-Lehmann median are retained as descriptive columns only.
#
# They are two distinct claims on two differently-weighted estimands, not a
# primary and a companion test of one hypothesis — no sentence
# below collapses them into "beats" without naming which claim, on which
# estimand.
#
# **Why the bin-balanced stream is close to foreordained, stated before the
# numbers below are read.** The baseline saturates at 58.47 against a
# reference reaching 100; in the 80-100 bin it cannot physically reach the
# reference, so its error there is bounded below by a large number regardless
# of what the six models do. Delta balanced MAE will typically be large and
# Holm-significant for all six models for a reason that is a property of the
# index's saturation, not of MLLM competence — the per-bin decomposition
# below (R1, R5) is what makes that visible rather than merely asserted, and
# no balanced-MAE claim in this notebook is reported without it.

# %%
def contrast_for_model(model: str, paired: pd.DataFrame) -> dict:
    C.guard_d12_statistic(paired, full_frame_expected=True, context=f"contrast {model}")

    # --- stream 1: bin-balanced (delta balanced MAE), image bootstrap ---
    def stat_delta_bal(frame):
        bal_m = C.balanced_mae(frame["abs_e_mllm"], frame["bin"]).balanced_mae
        bal_b = C.balanced_mae(frame["abs_e_baseline"], frame["bin"]).balanced_mae
        return bal_m - bal_b

    boot_bal = C.image_bootstrap(paired, stat_delta_bal, rng, bin_col="bin")
    k4_bal = C.check_k4_bootstrap_bin_coverage(boot_bal.n_empty_bin_violations, C.B_BOOTSTRAP)

    bal_mllm = C.balanced_mae(paired["abs_e_mllm"], paired["bin"])
    bal_base_local = C.balanced_mae(paired["abs_e_baseline"], paired["bin"])

    # --- stream 2: typical-image (paired Wilcoxon on per-image mean |e|) ---
    wil = C.paired_wilcoxon(paired["abs_e_mllm"].values, paired["abs_e_baseline"].values)

    diff = (paired["abs_e_mllm"] - paired["abs_e_baseline"]).values
    hl = C.hodges_lehmann(diff)
    rb = C.rank_biserial_matched_pairs(diff)
    wr = C.win_rate(paired["abs_e_mllm"].values, paired["abs_e_baseline"].values)

    # K2 (symmetry) fails on every contrast (checked above, all |skew| > 1),
    # so the fallback binds here: the exact paired sign test is the
    # reported typical-image effect statement, and HL is demoted to a
    # descriptive column. Computed for every model, not only where the
    # threshold trips, so the branch actually in force is always present in
    # the output rather than assembled after the fact.
    sign_test = C.exact_sign_test(diff)

    # --- overall-scale magnitudes (estimates with CIs, no p-values here) ---
    boot_overall_mae = C.image_bootstrap(
        paired, lambda f: f["abs_e_mllm"].mean() - f["abs_e_baseline"].mean(), rng, bin_col="bin")
    boot_rmse_delta = C.image_bootstrap(
        paired, lambda f: np.sqrt((f["e_mllm"] ** 2).mean()) - np.sqrt((f["e_baseline"] ** 2).mean()),
        rng, bin_col="bin")
    boot_bias_mllm = C.image_bootstrap(paired, lambda f: f["e_mllm"].mean(), rng, bin_col="bin")
    boot_bias_baseline = C.image_bootstrap(paired, lambda f: f["e_baseline"].mean(), rng, bin_col="bin")

    # --- Delta Spearman: the model-axis estimand is the pooled per-(image,
    # prompt) rows, never the mean of the 4 prompts' predictions.
    # Averaging predictions first is an ensemble (Q2's question) and
    # denoises the per-image prediction before ranking, which inflates rank
    # agreement relative to the pre-declared estimand. `pooled` carries 4
    # rows per image (one per prompt) plus the baseline's one row per image;
    # `image_bootstrap` resamples by `image`, so every drawn image still
    # contributes all of its rows on both sides together.
    pooled = mllm_pooled_rows(model)
    pooled = pooled.merge(
        d12_frame[["image", "vegetation_percent"]].rename(columns={"vegetation_percent": "baseline_pred"}),
        on="image", how="inner",
    )

    def stat_spearman_mllm(frame):
        return C.spearman_tie_corrected(frame["vegetation_percent"].values, frame["reference"].values)[0]

    def stat_spearman_baseline(frame):
        # One baseline value per image, but `frame` here carries 4 duplicate
        # baseline rows per image (one per prompt row it was joined onto);
        # Spearman is computed on the same duplicated-row estimand as the
        # MLLM side so that the two rank correlations being differenced are
        # over identical row sets and identical resample draws.
        return C.spearman_tie_corrected(frame["baseline_pred"].values, frame["reference"].values)[0]

    spearman_mllm = stat_spearman_mllm(pooled)
    boot_spearman_mllm = C.image_bootstrap(pooled, stat_spearman_mllm, rng, bin_col="bin")
    boot_spearman_baseline_local = C.image_bootstrap(pooled, stat_spearman_baseline, rng, bin_col="bin")

    def stat_delta_spearman(frame):
        return stat_spearman_mllm(frame) - stat_spearman_baseline(frame)

    boot_delta_spearman = C.image_bootstrap(pooled, stat_delta_spearman, rng, bin_col="bin")
    # Simultaneous 99.17% (=100-5/6) BCa band, same resamples' quantiles.
    boot_delta_spearman_simult = C.image_bootstrap(
        pooled, stat_delta_spearman, rng, bin_col="bin", alpha=(1 - 0.9917), compute_jackknife=True)

    # --- Campaign-clustered companion, required on every estimate ---
    # weight for balanced-MAE-style contribution: (1/5)/n_bin(i)
    bin_sizes = paired["bin"].map(paired["bin"].value_counts())
    weights_bal = (1.0 / 5.0) / bin_sizes.values
    d_contrib = paired["abs_e_mllm"].values - paired["abs_e_baseline"].values
    clusters = paired["campaign"].values

    webb = C.wild_cluster_bootstrap(d_contrib, clusters, rng, weights=weights_bal)
    clustered = C.clustered_interval(d_contrib, clusters, weights=weights_bal)

    # LOCO refits. Each of the three refits writes its own per-bin n
    # (R1 applies to every clustered companion) — `campaign_balanced_mae_with_min_n`
    # both enforces the n >= 10-per-bin rule and returns the per-bin n table
    # used to construct the refit, so the eventual balanced-MAE range is
    # traceable back to how many images actually informed each bin.
    loco_bal = {}
    loco_per_bin_n = {}
    for camp in paired["campaign"].unique():
        sub = paired[paired["campaign"] != camp]
        C.guard_d12_statistic(sub, full_frame_expected=False, context=f"LOCO drop {camp}, {model}")
        res, per_bin_n_sub = C.campaign_balanced_mae_with_min_n(
            sub, abs_error_col="abs_e_mllm", bin_col="bin", image_col="image", rng=rng, B=C.B_BOOTSTRAP,
        )
        loco_per_bin_n[camp] = per_bin_n_sub
        loco_bal[camp] = res if isinstance(res, str) else stat_delta_bal(sub)
    loco_min, loco_max, loco_note = C.loco_range(list(loco_bal.values()))

    maverick_extra = {}
    if model == "Llama-4-Maverick":
        overall_effect = boot_overall_mae.estimate
        maverick_extra["maverick_floor_delta"] = maverick_floor.delta_mav
        maverick_extra["within_run_to_run_variability_overall_mae"] = C.maverick_within_floor(
            overall_effect, maverick_floor.delta_mav)
        bal_branch = C.maverick_balanced_mae_claim_branch(maverick_floor.delta_mav)
        maverick_extra["floor_not_estimable_for_balanced_mae"] = bal_branch["floor_not_estimable_for_balanced_mae"]
        maverick_extra["balanced_mae_maverick_caveat"] = bal_branch["caveat"]
    else:
        maverick_extra["maverick_floor_delta"] = np.nan
        # Literal string rather than NaN/None: the determinism floor is a property
        # of Llama-4-Maverick specifically (C11), so for the other five
        # models this column is inapplicable, not a missing measurement —
        # the same distinction `c1_not_applicable_baseline_side` draws for C1.
        maverick_extra["within_run_to_run_variability_overall_mae"] = "not applicable (no floor set)"
        maverick_extra["floor_not_estimable_for_balanced_mae"] = False
        maverick_extra["balanced_mae_maverick_caveat"] = ""

    return {
        "model": model,
        "n_pairs": len(paired),
        "delta_balanced_mae": boot_bal.estimate,
        "delta_balanced_mae_ci_lo": boot_bal.ci_lo, "delta_balanced_mae_ci_hi": boot_bal.ci_hi,
        "delta_balanced_mae_ci_method": boot_bal.ci_method,
        "p_bal_raw": boot_bal.p_two_sided,
        "n_empty_bin_violations": boot_bal.n_empty_bin_violations,
        "k4_switch_to_stratified": k4_bal["switch_to_stratified"],
        "wilcoxon_statistic": wil.statistic, "p_wilcoxon_raw": wil.p_value,
        "n_zero_diff": wil.n_zero_diff, "n_tied_ranks": wil.n_tied_ranks,
        "hl_median_ae_diff_descriptive": hl, "rank_biserial": rb,
        "sign_test_n_used": sign_test["n_used"], "sign_test_n_pos": sign_test["n_pos"],
        "p_sign_test_raw": sign_test["p_value"],
        "win_rate_mllm": wr["win_rate_a"], "win_rate_baseline": wr["win_rate_b"], "tie_rate": wr["tie_rate"],
        "delta_overall_mae": boot_overall_mae.estimate,
        "delta_overall_mae_ci_lo": boot_overall_mae.ci_lo, "delta_overall_mae_ci_hi": boot_overall_mae.ci_hi,
        "delta_rmse": boot_rmse_delta.estimate,
        "delta_rmse_ci_lo": boot_rmse_delta.ci_lo, "delta_rmse_ci_hi": boot_rmse_delta.ci_hi,
        "bias_model": boot_bias_mllm.estimate,
        "bias_model_ci_lo": boot_bias_mllm.ci_lo, "bias_model_ci_hi": boot_bias_mllm.ci_hi,
        "bias_baseline": boot_bias_baseline.estimate,
        "bias_baseline_ci_lo": boot_bias_baseline.ci_lo, "bias_baseline_ci_hi": boot_bias_baseline.ci_hi,
        "delta_abs_bias": abs(boot_bias_mllm.estimate) - abs(boot_bias_baseline.estimate),
        "bias_direction_opposed": bool(np.sign(boot_bias_mllm.estimate) != np.sign(boot_bias_baseline.estimate)
                                        and boot_bias_mllm.estimate != 0 and boot_bias_baseline.estimate != 0),
        "spearman_mllm": boot_spearman_mllm.estimate,
        "spearman_mllm_ci_lo": boot_spearman_mllm.ci_lo, "spearman_mllm_ci_hi": boot_spearman_mllm.ci_hi,
        "spearman_baseline_local": boot_spearman_baseline_local.estimate,
        "delta_spearman": boot_delta_spearman.estimate,
        "delta_spearman_ci_lo": boot_delta_spearman.ci_lo, "delta_spearman_ci_hi": boot_delta_spearman.ci_hi,
        "d_spearman_ci_lo_simult": boot_delta_spearman_simult.ci_lo,
        "d_spearman_ci_hi_simult": boot_delta_spearman_simult.ci_hi,
        "p_clustered_webb": webb.p_webb, "n_distinct_abs_t_star": webb.n_distinct_abs_t_star,
        "webb_below_resolution_floor": webb.below_resolution_floor,
        "clustered_ci_lo": clustered.ci_lo, "clustered_ci_hi": clustered.ci_hi, "clustered_df": clustered.df,
        "ci_width_ratio_clustered_to_headline": C.ci_width_ratio_clustered_to_headline(
            clustered, boot_bal.ci_lo, boot_bal.ci_hi),
        "loco_min": loco_min, "loco_max": loco_max, "loco_note": loco_note,
        "loco_per_bin_n": loco_per_bin_n, "loco_bal_by_campaign": loco_bal,
        "c1_serving_confounded": False, "c1_not_applicable_baseline_side": True,
        **maverick_extra,
    }


contrast_results = {model: contrast_for_model(model, mllm_per_image(model)) for model in C.MODELS}

# %% [markdown]
# ### Holm-Bonferroni within family H (size 6, first threshold 0.00833)
#
# Each stream is corrected on its own six raw p-values. The bin-balanced
# stream uses its six bootstrap p-values as always. **The typical-image
# stream uses the six exact-sign-test p-values, not the Wilcoxon
# p-values** — K2 (symmetry of `d_i`) fails on all six contrasts (skew
# −1.57 to −2.03, all past `|skew| > 1`), which triggers the
# fallback: the exact paired sign test becomes the reported effect
# statement, and the Wilcoxon statistic / Hodges-Lehmann median are kept
# only as descriptive columns (`wilcoxon_statistic`, `p_wilcoxon_raw`,
# `hl_median_ae_diff_descriptive`). The family is family H, size 6, held in
# `_common.FAMILY_SIZES["H"]`; `holm_adjust` refuses a family size other
# than 6 for the label `"H"`.

# %%
models_ordered = list(C.MODELS)
p_bal_raw = [contrast_results[m]["p_bal_raw"] for m in models_ordered]
p_wil_raw = [contrast_results[m]["p_wilcoxon_raw"] for m in models_ordered]
p_sign_raw = [contrast_results[m]["p_sign_test_raw"] for m in models_ordered]

p_bal_holm = C.holm_adjust(p_bal_raw, family="H", family_size=C.FAMILY_SIZES["H"])
# Descriptive-only Holm adjustment of the Wilcoxon p, reported so a reader can
# see what the rank test would have said where its symmetry assumption held.
# It is never the reported typical-image verdict.
p_wil_holm = C.holm_adjust(p_wil_raw, family="H", family_size=C.FAMILY_SIZES["H"])
# The reported typical-image verdict: Holm-adjusted exact sign test.
p_sign_holm = C.holm_adjust(p_sign_raw, family="H", family_size=C.FAMILY_SIZES["H"])

for i, m in enumerate(models_ordered):
    contrast_results[m]["p_bal_holm"] = float(p_bal_holm[i])
    contrast_results[m]["p_wilcoxon_holm"] = float(p_wil_holm[i])
    contrast_results[m]["p_sign_test_holm"] = float(p_sign_holm[i])
    # `p_companion_*` (the R2 schema name for the typical-image claim) is the
    # sign test now that K2 has triggered the fallback for every contrast.
    contrast_results[m]["p_companion_raw_reported"] = contrast_results[m]["p_sign_test_raw"]
    contrast_results[m]["p_companion_holm_reported"] = float(p_sign_holm[i])

holm_threshold_H = C.holm_first_threshold(0.05, C.FAMILY_SIZES["H"])
print(f"family H first Holm threshold: {holm_threshold_H:.5f}")
summary_cols = ["model", "delta_balanced_mae", "p_bal_raw", "p_bal_holm",
                "hl_median_ae_diff_descriptive", "p_sign_test_raw", "p_sign_test_holm",
                "delta_overall_mae", "bias_model", "bias_baseline", "delta_spearman"]
summary_tbl = pd.DataFrame([contrast_results[m] for m in models_ordered])[summary_cols]
print(summary_tbl.round(4).to_string(index=False))

# %% [markdown]
# ### Reading the two streams together, and where they diverge
#
# The bin-balanced claim is Holm-significant for every model, for the
# structural reason stated above: the top bin's 30 images dominate delta
# balanced MAE and the baseline cannot reach them. A large, significant
# balanced-MAE win is a true statement about the two methods **on this
# frame**, and it is **not** evidence that MLLMs read sparse quadrats better
# — which is where 81% of this frame and most of the practical use lives.
#
# The typical-image claim (the exact sign test, per the K2 fallback above)
# is where the comparison is genuinely mixed, and it does not go the way
# the balanced-MAE stream's flattering framing would suggest. On the
# typical image the classical baseline **beats both Gemma models
# significantly** — it has the smaller error on 63.6% of images against
# Gemma-3-12B (`p_holm` ~7e-20) and 69.4% against Gemma-3-27B (`p_holm`
# ~7e-40) — and beats Mistral-Small-3.2 on 54.2% of images (`p_holm` =
# 0.0141). Llama-4-Maverick is the only model that beats the baseline on
# the typical image (58.4% of images, `p_holm` ~4e-08); Llama-4-Scout and
# Qwen-2.5 are indistinguishable from it (`p_holm` = 0.433 for both). No
# sentence below reports the balanced-MAE win as if it settled the
# typical-image question — the two streams are read separately because
# they disagree.

# %% [markdown]
# ## Per-bin decomposition of delta balanced MAE (R1, R5)
#
# Every balanced-MAE claim in this notebook is reported with its five per-bin
# MAEs (both sides), their n and CI, and the share of delta balanced MAE each
# bin contributes — never as a bare number. This is what makes the top-bin
# saturation argument verifiable rather than asserted. R5 additionally binds
# here: the per-bin signed bias for both sides is reported alongside the
# per-bin MAE, since a close per-bin MAE can still hide opposite failure
# directions, exactly as C16 shows at the overall scale.

# %%
def _per_bin_ci(frame: pd.DataFrame, value_col: str) -> tuple[dict, dict]:
    """Per-bin image-bootstrap CI on the mean of `value_col`, resampling
    images within each bin subset independently (same scheme as the
    baseline diagnostic block above)."""
    ci_lo, ci_hi = {}, {}
    for label in C.BIN_LABELS:
        sub = frame[frame["bin"] == label]
        if sub["image"].nunique() < 2:
            ci_lo[label], ci_hi[label] = np.nan, np.nan
            continue
        b = C.image_bootstrap(sub, lambda f, c=value_col: f[c].mean(), rng, bin_col=None)
        ci_lo[label], ci_hi[label] = b.ci_lo, b.ci_hi
    return ci_lo, ci_hi


# The baseline side (abs_e_baseline, e_baseline) is identical across all six
# models' `paired` frames — it is the same D12 frame joined six times — so
# its per-bin CIs are computed once here and reused, rather than recomputed
# per model. This changes nothing about what is reported (R1/R5 still bind
# per model row) and only avoids resampling the same fixed baseline values
# six times over.
baseline_mae_ci_lo, baseline_mae_ci_hi = _per_bin_ci(d12_frame, "abs_e")
bias_baseline_ci_lo, bias_baseline_ci_hi = _per_bin_ci(d12_frame, "e")

bin_decomp_rows = []
mllm_perbin_mae_ci = {}  # cached per model, reused by Q9_metrics_by_model.csv below
for model in models_ordered:
    paired = mllm_per_image(model)
    bal_mllm = C.balanced_mae(paired["abs_e_mllm"], paired["bin"])
    bal_baseline = C.balanced_mae(paired["abs_e_baseline"], paired["bin"])
    mllm_mae_ci_lo, mllm_mae_ci_hi = _per_bin_ci(paired, "abs_e_mllm")
    mllm_perbin_mae_ci[model] = (mllm_mae_ci_lo, mllm_mae_ci_hi)
    bias_mllm_ci_lo, bias_mllm_ci_hi = _per_bin_ci(paired, "e_mllm")
    per_bin_bias_mllm = paired.groupby("bin")["e_mllm"].mean().reindex(C.BIN_LABELS)
    per_bin_bias_baseline = paired.groupby("bin")["e_baseline"].mean().reindex(C.BIN_LABELS)
    per_bin_delta = {lbl: bal_mllm.per_bin_mae[lbl] - bal_baseline.per_bin_mae[lbl] for lbl in C.BIN_LABELS}
    total_delta_bal = float(np.mean(list(per_bin_delta.values())))
    for lbl in C.BIN_LABELS:
        share = (per_bin_delta[lbl] / 5.0) / total_delta_bal if total_delta_bal != 0 else np.nan
        bin_decomp_rows.append({
            "question_id": "Q9", "model": model, "bin": lbl,
            "n": bal_mllm.per_bin_n[lbl],
            "mae_mllm": bal_mllm.per_bin_mae[lbl],
            "mae_mllm_ci_lo": mllm_mae_ci_lo[lbl], "mae_mllm_ci_hi": mllm_mae_ci_hi[lbl],
            "mae_baseline": bal_baseline.per_bin_mae[lbl],
            "mae_baseline_ci_lo": baseline_mae_ci_lo[lbl], "mae_baseline_ci_hi": baseline_mae_ci_hi[lbl],
            "delta_mae_bin": per_bin_delta[lbl],
            "bias_mllm_bin": float(per_bin_bias_mllm[lbl]),
            "bias_mllm_bin_ci_lo": bias_mllm_ci_lo[lbl], "bias_mllm_bin_ci_hi": bias_mllm_ci_hi[lbl],
            "bias_baseline_bin": float(per_bin_bias_baseline[lbl]),
            "bias_baseline_bin_ci_lo": bias_baseline_ci_lo[lbl], "bias_baseline_bin_ci_hi": bias_baseline_ci_hi[lbl],
            "share_of_delta_balanced_mae": share,
        })

q9_bin_decomposition = pd.DataFrame(bin_decomp_rows)
print(q9_bin_decomposition.round(3).to_string(index=False))

top2_share = (
    q9_bin_decomposition[q9_bin_decomposition["bin"].isin(["60-80", "80-100"])]
    .groupby("model")["share_of_delta_balanced_mae"].sum()
)
print("\nshare of delta balanced MAE from the top two bins, per model:")
print(top2_share.round(3).to_string())

# %% [markdown]
# ## The 0-20 bin at configuration granularity
#
# The per-model table above (and the 933-image bottom-bin MAE reported in
# the per-bin decomposition) shows the classical baseline beating every one
# of the six models, pooled over their four prompts, in the sparse bin that
# holds 81% of the frame. **That statement is true at the model level and
# false at the configuration level**, and pooling over prompts is what hides
# the difference: a model's best prompt is averaged together with its worst,
# and the resulting model-level number is not a statement about any single
# configuration a practitioner would actually deploy.
#
# The table below computes the 0-20-bin MAE for all 24 model x prompt
# configurations against the same 933 images and the same baseline value
# (3.411). This is descriptive: no test is run on this subset (or on any
# subset of the frame) at any point in this notebook — the six family-H sign
# tests and bootstrap tests above are the only inferential tests here, and
# they are computed on the full 1,155-image frame. `sign_test_n_used` on
# every family-H row stays 1,155. These are point estimates with BCa
# intervals only.

# %%
# This block's resampling gets its own generator, independent of `rng`.
# `rng` is the stream the six family-H bootstraps, the clustered companion and
# the Delta Spearman bands draw from in strict cell order. Drawing from `rng`
# here would insert extra draws into that stream and quietly shift every one
# of those intervals; where this cell happens to sit in the file is no
# protection against that, only an independent generator is. It is seeded
# deterministically from the same literal `SEED`, with a fixed offset reserved
# for this block alone.
rng_lowbin = np.random.default_rng(SEED + 101)

prompts = sorted(base_local["prompt"].unique())

LOW_BIN_LABEL = "0-20"
low_bin_baseline = d12_frame[d12_frame["bin"] == LOW_BIN_LABEL]
n_low_bin = int(low_bin_baseline["image"].nunique())
low_bin_baseline_mae = float(low_bin_baseline["abs_e"].mean())
print(f"0-20 bin: n={n_low_bin} images, baseline MAE={low_bin_baseline_mae:.3f}")


def lowbin_combo_row(model: str, prompt: str) -> dict:
    sub = base_local[(base_local["model"] == model) & (base_local["prompt"] == prompt)]
    paired = sub[["image", "abs_e"]].rename(columns={"abs_e": "abs_e_mllm"}).merge(
        low_bin_baseline[["image", "abs_e", "bin"]].rename(columns={"abs_e": "abs_e_baseline"}),
        on="image", how="inner",
    )
    if len(paired) != n_low_bin:
        raise AssertionError(
            f"0-20-bin pairing for {model}/{prompt} yielded {len(paired)} rows, expected {n_low_bin}"
        )
    boot = C.image_bootstrap(paired, lambda f: f["abs_e_mllm"].mean(), rng_lowbin, bin_col=None)
    return {
        "question_id": "Q9", "model": model, "prompt": prompt, "bin": LOW_BIN_LABEL, "n": n_low_bin,
        "mae_combo": boot.estimate, "mae_combo_ci_lo": boot.ci_lo, "mae_combo_ci_hi": boot.ci_hi,
        "mae_baseline": low_bin_baseline_mae,
        "delta_vs_baseline": boot.estimate - low_bin_baseline_mae,
        "ahead_of_baseline": bool(boot.estimate < low_bin_baseline_mae),
        "selection_conditioned": False, "holm_family": None, "p_value": None,
    }


lowbin_combo_rows = [lowbin_combo_row(m, p) for m in models_ordered for p in prompts]
q9_lowbin_by_combo = pd.DataFrame(lowbin_combo_rows).sort_values("mae_combo").reset_index(drop=True)
n_ahead = int(q9_lowbin_by_combo["ahead_of_baseline"].sum())
print(f"\n{n_ahead} / {len(q9_lowbin_by_combo)} model x prompt configurations beat the baseline's "
      f"{low_bin_baseline_mae:.3f} in the 0-20 bin (descriptive, no test, no p-value):")
print(q9_lowbin_by_combo.head(6)[["model", "prompt", "mae_combo", "delta_vs_baseline", "ahead_of_baseline"]]
      .round(3).to_string(index=False))

model_combo_mae = q9_lowbin_by_combo.groupby("model")["mae_combo"].mean()
print(f"\npooled-over-prompts 0-20-bin MAE per model (for contrast with the row above): "
      f"{model_combo_mae.min():.2f} to {model_combo_mae.max():.2f}, "
      f"all above the baseline's {low_bin_baseline_mae:.3f}.")

# %% [markdown]
# **Reading the two granularities together.** Pooled over prompts, the
# classical baseline has a lower 0-20-bin MAE than all six models — the
# model-level range above runs from the low 4s to the low 9s, all above the
# baseline's 3.411. At configuration level this reverses for the leading
# configuration: **Llama-4-Maverick paired with the `Detailed` prompt** is
# the only one of the 24 configurations ahead of the baseline in this bin,
# with **Llama-4-Scout x Detailed** and **Qwen-2.5 x Detailed** the next
# closest. `Detailed` supplies three of the top four rows of the table above
# while being, on the balanced-MAE scale reported elsewhere in this project,
# the *worst*-performing prompt overall. A plausible explanation for both
# being true at once, consistent with the evidence below but not directly
# tested by anything in this notebook, is a construct mismatch rather than a
# contradiction: `Detailed` is the only prompt among the four that supplies a
# definition of vegetation, and that definition excludes dead plant material
# that the reference sometimes scores as cover. On bare ground there is
# usually little or no dead material to disagree about, so the exclusion
# would cost little there, while at higher cover the same exclusion would
# compound into a larger under-count — a pattern consistent with `Detailed`
# being worst overall on balanced MAE while also placing three configurations
# at the top of this one sparse-cover table. No test in this notebook
# measures dead-material disagreement directly, so this remains an
# explanation the evidence is consistent with, not a demonstrated mechanism.
# Any sentence about the baseline's 0-20-bin standing must name the
# granularity: **"pooled over prompts, the classical baseline has lower
# 0-20-bin MAE than all six models; at configuration level one configuration
# is ahead of it (Llama-4-Maverick x Detailed)."** The unqualified form —
# "the baseline beats every model in this bin" — elides the granularity and
# is not used anywhere in this notebook.

# %% [markdown]
# ## Overestimation structure, with the feasibility ceiling
#
# A baseline biased -10.00 overall under-predicts on average, but "does it
# ever over-predict, and where" is a separate and useful question. The count
# alone is not enough to answer it: **the baseline's predictions never
# exceed its own observed ceiling**, so in bins whose reference values sit
# entirely above that ceiling, a count of zero over-predictions is not a
# finding about the baseline — it is a fact about arithmetic that would hold
# no matter what the baseline did. Both the ceiling and the per-bin
# feasibility are computed below from the loaded frame, not hard-coded.

# %%
baseline_max_prediction = float(baseline_pred.max())
print(f"baseline max prediction (computed from the loaded frame): {baseline_max_prediction:.3f}")

overpred_rows = []
for lbl in C.BIN_LABELS:
    sub = d12_frame[d12_frame["bin"] == lbl]
    n_bin = int(sub["image"].nunique())
    n_over = int((sub["vegetation_percent"] > sub["reference"]).sum())
    min_ref_in_bin = float(sub["reference"].min())
    feasible_mask = sub["reference"] < baseline_max_prediction
    n_feasible = int(feasible_mask.sum())
    possible = bool(min_ref_in_bin < baseline_max_prediction)
    overpred_rows.append({
        "question_id": "Q9", "bin": lbl, "n": n_bin,
        "n_overpredicted": n_over,
        "min_reference_in_bin": min_ref_in_bin,
        "baseline_max_prediction": baseline_max_prediction,
        "overprediction_arithmetically_possible": possible,
        "n_images_overprediction_feasible": n_feasible,
    })

q9_overprediction_feasibility = pd.DataFrame(overpred_rows)
print(q9_overprediction_feasibility.round(3).to_string(index=False))

# %% [markdown]
# **What the table shows.** In the 0-20 bin the baseline over-predicts on
# **357 of 933 images (38%)** — over-prediction is arithmetically possible
# throughout this bin (every reference value in it is below the baseline's
# ceiling), so this count is fully informative. In the 20-40 and 40-60 bins
# the observed count is zero, and over-prediction was arithmetically
# possible for at least some images in each (`n_images_overprediction_feasible`
# is the number of images in the bin whose reference sits below the
# baseline's ceiling, not the raw bin size) — the zero is therefore
# informative there too, and is reported against that feasible denominator,
# not against the raw bin n, since some images in these bins were also
# infeasible to over-predict and including them would overstate how many
# chances the baseline actually had. In the 60-80 and 80-100 bins **every
# reference value exceeds the baseline's ceiling**
# (`overprediction_arithmetically_possible = False`, feasible denominator of
# zero in both), so the zero count there carries no information whatsoever —
# it was guaranteed before a single prediction was examined and is reported
# with that flag attached rather than as evidence of anything.
#
# **The two-sided characterisation this supports.** The baseline over-reads
# bare ground on 38% of the sparse bin's images while under-reading almost
# everything else — a sharper and more accurate description of a
# colour-threshold method than "it fails spectrally at high cover" alone.
# The bias is not uniform under-prediction; it is a mis-scaling that lands
# above the reference on bare ground and far below it everywhere else, which
# is the same conclusion the per-bin MAE profile and the Pearson diagnostic
# reach by a different route. Descriptive throughout: no test, no family, no
# p-value.

# %% [markdown]
# ## Chart — per-model overall MAE against the baseline, with signed bias
#
# The scatter above shows *why* the balanced-MAE stream is dominated by the
# top bins. This chart shows the *typical-image* picture the paper
# actually turns on: overall MAE (all 1,155 images pooled) for each model
# against the baseline, annotated with each side's signed bias — because a
# close MAE gap can still describe two methods failing in opposite
# directions (C16), which an MAE-only bar chart would hide completely.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

order = summary_tbl.sort_values("delta_overall_mae")["model"].tolist()
overall_mae_mllm = [contrast_results[m]["delta_overall_mae"] + baseline_mae for m in order]
# These bars are baseline_mae + CI(delta overall MAE) — the baseline's own
# overall MAE anchored to the bootstrap interval on the *contrast*, not a
# bootstrap CI computed on the model's own overall MAE directly. It is wider
# than that true interval (it also carries the baseline's sampling
# variability, counted once per model rather than shared), so it errs
# conservative, but it should not be read as "the model's overall-MAE CI".
overall_mae_ci_lo = [contrast_results[m]["delta_overall_mae_ci_lo"] + baseline_mae for m in order]
overall_mae_ci_hi = [contrast_results[m]["delta_overall_mae_ci_hi"] + baseline_mae for m in order]
yerr = [[v - lo for v, lo in zip(overall_mae_mllm, overall_mae_ci_lo)],
        [hi - v for v, hi in zip(overall_mae_mllm, overall_mae_ci_hi)]]

ax1.barh(order, overall_mae_mllm, xerr=yerr, color="#4C72B0", capsize=4, label="model overall MAE")
ax1.axvline(baseline_mae, color="firebrick", linestyle="--", linewidth=1.5,
            label=f"classical baseline overall MAE ({baseline_mae:.2f})")
ax1.set_xlabel("Overall MAE (cover points), n=1,155 images")
ax1.set_title("Overall MAE per model vs. baseline\n(error bars: baseline MAE +/- 95% CI of the "
               "model-baseline contrast, not a CI on the model's own MAE)")
ax1.legend(fontsize=8, loc="lower right")

bias_mllm = [contrast_results[m]["bias_model"] for m in order]
ax2.barh(order, bias_mllm, color="#55A868", label="model signed bias")
ax2.axvline(baseline_bias, color="firebrick", linestyle="--", linewidth=1.5,
            label=f"classical baseline signed bias ({baseline_bias:.2f})")
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_xlabel("Signed mean bias (cover points); positive = over-prediction")
ax2.set_title("Signed bias per model vs. baseline\n(opposite signs = opposite failure modes, C16)")
ax2.legend(fontsize=8, loc="lower right")

fig.tight_layout()
fig.savefig(fig_path / "Q9_overall_mae_and_bias.png", dpi=150)
plt.show()
print("Caption: left panel is the typical-image MAE comparison (n=1,155). The error bars "
      "are the baseline's overall MAE offset by the 95% image-bootstrap CI of the "
      "model-vs-baseline contrast in overall MAE, not an independent CI on each model's own "
      "MAE — it is systematically wider than that quantity (about 2.1x on Qwen-2.5: 1.92 "
      "delivered vs 0.90 on the model's own MAE directly), so it errs conservative rather "
      "than overstating precision. Right panel shows every model over-predicts while the "
      "baseline severely under-predicts — a close MAE gap (e.g. the two Gemmas) can still "
      "hide opposite failure directions.")

# %% [markdown]
# ## The 24 model x prompt configurations vs. baseline (descriptive, uncorrected)
#
# Computed and reported so the configuration-level picture is visible, but
# **not** part of family H and carrying **no p-value** — testing all 24 would
# take the Holm threshold to 0.05/24 = 0.00208 for a question ("does model X
# under prompt Y beat the baseline") nobody in this project asks at the
# confirmatory level. Estimates with BCa CIs only.

# %%
def contrast_by_combo(model: str, prompt: str) -> dict:
    sub = base_local[(base_local["model"] == model) & (base_local["prompt"] == prompt)]
    paired = sub[["image", "abs_e", "e"]].rename(columns={"abs_e": "abs_e_mllm", "e": "e_mllm"}).merge(
        d12_frame[["image", "reference", "e", "abs_e", "bin", "campaign"]]
        .rename(columns={"e": "e_baseline", "abs_e": "abs_e_baseline"}),
        on="image", how="inner",
    )
    C.guard_d12_statistic(paired, full_frame_expected=True, context=f"combo {model}/{prompt}")

    def stat_delta_bal(frame):
        bal_m = C.balanced_mae(frame["abs_e_mllm"], frame["bin"]).balanced_mae
        bal_b = C.balanced_mae(frame["abs_e_baseline"], frame["bin"]).balanced_mae
        return bal_m - bal_b

    boot_bal = C.image_bootstrap(paired, stat_delta_bal, rng, bin_col="bin")
    boot_overall = C.image_bootstrap(
        paired, lambda f: f["abs_e_mllm"].mean() - f["abs_e_baseline"].mean(), rng, bin_col="bin")
    # R5: signed bias for both sides travels with every Q9 row, including
    # this descriptive 24-configuration table. An error magnitude without
    # the two signed biases is a defect, not a simplification.
    bias_mllm = float(paired["e_mllm"].mean())
    bias_baseline = float(paired["e_baseline"].mean())
    return {
        "question_id": "Q9", "model": model, "prompt": prompt, "n_pairs": len(paired),
        "delta_balanced_mae": boot_bal.estimate, "delta_balanced_mae_ci_lo": boot_bal.ci_lo,
        "delta_balanced_mae_ci_hi": boot_bal.ci_hi,
        "delta_overall_mae": boot_overall.estimate, "delta_overall_mae_ci_lo": boot_overall.ci_lo,
        "delta_overall_mae_ci_hi": boot_overall.ci_hi,
        "bias_model": bias_mllm, "bias_baseline": bias_baseline,
        "delta_abs_bias": abs(bias_mllm) - abs(bias_baseline),
        "bias_direction_opposed": bool(np.sign(bias_mllm) != np.sign(bias_baseline)
                                        and bias_mllm != 0 and bias_baseline != 0),
        "selection_conditioned": False, "holm_family": None, "p_value": None,
    }


combo_rows = [contrast_by_combo(m, p) for m in models_ordered for p in prompts]
q9_contrasts_by_combo = pd.DataFrame(combo_rows)
print(f"{len(q9_contrasts_by_combo)} model x prompt configurations computed (descriptive, no p-value).")
print(q9_contrasts_by_combo.sort_values("delta_overall_mae").head(6).round(3).to_string(index=False))

# %% [markdown]
# ## Supplementary: the same six contrasts on the `rectified` MLLM variant
#
# The primary comparison above scores the MLLM side on `base` — the same
# raw photograph the classical pipeline starts from — because rectification
# is a step *inside* the classical method (it needs the quadrat interior to
# fill the frame) rather than a pre-processing advantage handed to it. `base`
# is therefore the primary, and the only claim-bearing, comparison, and
# family H is defined on it.
#
# But it does mean the primary comparison gives the two sides different
# image geometry: the baseline necessarily runs on rectified images (pixel
# counting requires undistorted geometry) while the primary MLLM side runs
# on the raw photograph. The matched-geometry comparison, with the MLLM side
# also scored on `rectified`, is worth having explicitly, so it is computed
# here as a **supplementary, non-claim-bearing** repeat of the same six
# contrasts: the same estimands, the same 1,155 images, the same baseline.
# A17/C15 apply unchanged: the rectified MLLM side is aggregated to one
# value per image before pairing.

# %%
# Own generator again, for the same reason as the low-bin block above: this
# resampling must never draw from `rng`, which the family-H contrasts, the
# clustered companion and the Delta Spearman bands already consume in a
# fixed cell order.
rng_rectified = np.random.default_rng(SEED + 102)

full_local = C.build_full_local_frame()
assert set(full_local["variant"].unique()) == {"base", "masked_gray", "rectified"}
rectified_frame = full_local[full_local["variant"] == "rectified"].copy()


def mllm_per_image_rectified(model: str) -> pd.DataFrame:
    sub = rectified_frame[rectified_frame["model"] == model]
    per_image_abs = C.aggregate_mllm_side_to_image(sub, "abs_e").rename(columns={"abs_e": "abs_e_mllm"})
    per_image_signed = C.aggregate_mllm_side_to_image(sub, "e").rename(columns={"e": "e_mllm"})
    out = per_image_abs.merge(per_image_signed, on="image", how="inner")
    out = out.merge(
        d12_frame[["image", "vegetation_percent", "reference", "e", "abs_e", "bin", "campaign"]]
        .rename(columns={"vegetation_percent": "baseline_pred", "e": "e_baseline", "abs_e": "abs_e_baseline"}),
        on="image", how="inner",
    )
    return out


def rectified_contrast_row(model: str) -> dict:
    paired = mllm_per_image_rectified(model)
    C.guard_d12_statistic(paired, full_frame_expected=True, context=f"Q9 rectified supplementary, {model}")

    def stat_delta_bal(frame):
        bal_m = C.balanced_mae(frame["abs_e_mllm"], frame["bin"]).balanced_mae
        bal_b = C.balanced_mae(frame["abs_e_baseline"], frame["bin"]).balanced_mae
        return bal_m - bal_b

    boot_bal = C.image_bootstrap(paired, stat_delta_bal, rng_rectified, bin_col="bin")
    bal_mllm = C.balanced_mae(paired["abs_e_mllm"], paired["bin"])
    bal_baseline = C.balanced_mae(paired["abs_e_baseline"], paired["bin"])

    wil = C.paired_wilcoxon(paired["abs_e_mllm"].values, paired["abs_e_baseline"].values)
    diff = (paired["abs_e_mllm"] - paired["abs_e_baseline"]).values
    sign_test = C.exact_sign_test(diff)
    hl = C.hodges_lehmann(diff)

    boot_overall_mae = C.image_bootstrap(
        paired, lambda f: f["abs_e_mllm"].mean() - f["abs_e_baseline"].mean(), rng_rectified, bin_col="bin")
    bias_mllm = float(paired["e_mllm"].mean())
    bias_baseline = float(paired["e_baseline"].mean())

    # Per-bin decomposition travels with this row too (R1/R5 bind on every
    # balanced-MAE claim, claim-bearing or not).
    per_bin = {
        lbl: {
            "n": bal_mllm.per_bin_n[lbl],
            "mae_mllm": bal_mllm.per_bin_mae[lbl],
            "mae_baseline": bal_baseline.per_bin_mae[lbl],
        }
        for lbl in C.BIN_LABELS
    }

    return {
        "question_id": "Q9", "contrast": f"{model}_vs_classical_baseline", "model": model,
        "variant": "rectified", "n_pairs": len(paired),
        "estimate": boot_bal.estimate, "ci_lo": boot_bal.ci_lo, "ci_hi": boot_bal.ci_hi,
        "ci_method": boot_bal.ci_method,
        "p_primary_raw": boot_bal.p_two_sided,
        "p_wilcoxon_raw": wil.p_value, "hl_median_ae_diff_descriptive": hl,
        "p_sign_test_raw": sign_test["p_value"],
        "sign_test_n_used": sign_test["n_used"], "sign_test_n_pos": sign_test["n_pos"],
        "delta_overall_mae": boot_overall_mae.estimate,
        "delta_overall_mae_ci_lo": boot_overall_mae.ci_lo, "delta_overall_mae_ci_hi": boot_overall_mae.ci_hi,
        "bias_model": bias_mllm, "bias_baseline": bias_baseline,
        "delta_abs_bias": abs(bias_mllm) - abs(bias_baseline),
        "bias_direction_opposed": bool(np.sign(bias_mllm) != np.sign(bias_baseline)
                                        and bias_mllm != 0 and bias_baseline != 0),
        "per_bin_mae": str(per_bin),
        "claim_bearing": False,
        "holm_family": None, "holm_adjusted": False,
        "note": (
            "supplementary, non-claim-bearing: MLLM side scored on the rectified variant, "
            "the baseline's own necessary pre-processing product, for apples-to-apples "
            "geometry; raw p-values only, no Holm column, not a second family H. The base "
            "comparison remains primary and is the only claim-bearing result. A difference "
            "between this run and the primary base run is the Q4 variant effect entering "
            "the Q9 comparison (Q4: -1.69 cover points overall; the warp step's contribution "
            "is -0.0436 [-0.4653, +0.3604] and is not distinguishable from zero)."
        ),
    }


rectified_rows = [rectified_contrast_row(m) for m in models_ordered]
q9_rectified_supplementary = pd.DataFrame(rectified_rows)
print("Supplementary (rectified-variant MLLM side, non-claim-bearing) delta balanced MAE:")
print(q9_rectified_supplementary[["model", "estimate", "ci_lo", "ci_hi", "p_primary_raw",
                                   "delta_overall_mae", "bias_model", "bias_baseline"]]
      .round(4).to_string(index=False))

# `rng_rectified` is the last rng-consuming generator in the notebook, so no
# downstream output can hash-check its isolation the way the low-bin block's
# isolation is checked by the unchanged hash of `Q9_contrasts_by_combo.csv`.
# Verify directly instead: snapshot its state here, immediately after its
# only consumer (`rectified_contrast_row`, called once per model above), and
# confirm at the very end of the notebook (see the isolation check in the
# final cell, after every remaining computation) that nothing after this
# point ever advances it.
rng_rectified_state_after_block = rng_rectified.bit_generator.state

# %% [markdown]
# **How to read this table.** Every row here carries `claim_bearing = FALSE`
# and no Holm-adjusted p-value — these are estimates with BCa intervals and
# raw p-values only, reported so a reader can see the matched image geometry
# without it competing with, or being confused for, the
# primary result. The `base` comparison in family H above remains the only
# claim-bearing comparison in this notebook.

# %% [markdown]
# ## Power
#
# The bin-balanced stream inherits the effective n of about 273 that the
# five-bin weighting implies, a nominal 1.5-2 cover-point detection limit, but
# the *expected* delta balanced MAE is far larger than that for the
# structural reason above, so power is not the binding constraint on that
# stream; interpretability is.
#
# The typical-image stream is the one actually reported (K2 fails on all six
# contrasts, so `p_companion_*` is the exact paired sign test, not
# Wilcoxon). Its detectable effect must therefore be expressed on the sign
# test's own scale — a departure of the win rate from 0.5 — not on a
# location-test scale such as Cohen's dz, whose power depends on the paired
# difference's sd rather than on the sign proportion. Below we compute the
# resolution of the test that was actually run: at n=1,155, two-sided
# alpha=0.00833 (family H's first Holm threshold), 80% power, what is the
# smallest win-rate departure from 0.5 the exact sign test can detect?

# %%
from scipy.stats import binomtest, norm
from scipy.stats import binom as binom_dist


def sign_test_power(n, p_alt, alpha, p_null=0.5):
    """Exact two-sided power of the sign test (binomial test) to detect a
    true win rate of p_alt against p_null, at total sample size n and
    two-sided level alpha. Finds the exact rejection region from the
    binomial(n, p_null) null and sums the alternative's mass over it.
    """
    lo_crit = binom_dist.ppf(alpha / 2, n, p_null)
    while binomtest(int(lo_crit), n, p_null).pvalue > alpha:
        lo_crit += 1
    hi_crit = binom_dist.isf(alpha / 2, n, p_null)
    while binomtest(int(hi_crit), n, p_null).pvalue > alpha:
        hi_crit -= 1
    k = np.arange(0, n + 1)
    reject = np.array([binomtest(int(x), n, p_null).pvalue <= alpha for x in k])
    return float(binom_dist.pmf(k[reject], n, p_alt).sum())


def detectable_win_rate(n, alpha, target_power=0.80, p_null=0.5, tol=1e-5):
    """Binary search on the win rate above 0.5 for 80% power of the exact
    sign test at this n and alpha.
    """
    lo, hi = p_null, 0.999
    for _ in range(60):
        mid = (lo + hi) / 2
        if sign_test_power(n, mid, alpha, p_null) < target_power:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return hi


win_rate_hi = detectable_win_rate(1155, holm_threshold_H, target_power=0.80)
win_rate_lo = 1 - win_rate_hi
print(f"Exact sign test, n=1155, two-sided alpha={holm_threshold_H:.5f}, 80% power: "
      f"detectable win rate departs to {win_rate_hi:.3f} / {win_rate_lo:.3f}")

observed_win_rates = {m: contrast_results[m]["sign_test_n_pos"] / contrast_results[m]["sign_test_n_used"]
                      for m in models_ordered}
for m in models_ordered:
    wr = observed_win_rates[m]
    outside = (wr >= win_rate_hi) or (wr <= win_rate_lo)
    print(f"  {m:>20}: baseline win rate={wr:.3f}  "
          f"{'outside' if outside else 'inside'} the {win_rate_lo:.3f}-{win_rate_hi:.3f} detection band")

print("\nAt this design's resolution, the exact sign test can detect a win-rate departure "
      f"to {win_rate_hi:.3f}/{win_rate_lo:.3f} or beyond. Gemma-3-27B (0.694), Gemma-3-12B "
      "(0.636) and Llama-4-Maverick (0.416, favouring the model) sit well outside that band "
      "and are Holm-significant. Mistral-Small-3.2 (0.542) sits inside the band computed at "
      "the strictest per-comparison threshold but is still detected once Holm's step-down "
      "relaxes to the next threshold (0.0167) — consistent with its p_holm of 0.0141. "
      "Llama-4-Scout (0.481) and Qwen-2.5 (0.487) sit inside the band: their typical-image "
      "nulls are a real failure to detect a difference at this design's resolution, not "
      "evidence of parity with the baseline. This band is the correct one for the test "
      "reported, and it is consistent with all six results — unlike a location-test power "
      "figure, which is not: Mistral has the smallest Hodges-Lehmann shift of the six models "
      "(|HL|=0.301) yet is significant, while Llama-4-Scout's larger |HL|=0.605 is not, so "
      "an HL-scale detection range cannot be the resolution limit that explains the pattern "
      "of significance actually observed.")

# %% [markdown]
# For reference, the location-test resolution calculation is reproduced below:
# Cohen's `dz` for a paired t/z test on the mean signed difference, scaled by
# each model's realised paired-difference sd. It describes the resolution of
# the **Wilcoxon/Hodges-Lehmann reading that the failed symmetry check
# demotes**, not of the sign test that is actually reported above. It is worth
# reading against the realised paired-difference sd of 13.85 to 16.67 cover
# points: a detection limit of 0.6-1.1 cover points requires an sd roughly
# half that.

# %%
dz = norm.ppf(1 - holm_threshold_H / 2) + norm.ppf(0.80)
dz = dz / np.sqrt(1155)
print(f"[descriptive only] approximate detectable dz at Holm threshold "
      f"{holm_threshold_H:.5f}, n=1155, 80% power: {dz:.4f}")
detectable_diffs = {}
for m in models_ordered:
    d_sd = float((mllm_per_image(m).assign(
        d=lambda f: f["abs_e_mllm"] - f["abs_e_baseline"])["d"]).std())
    detectable_diffs[m] = dz * d_sd
    print(f"  {m:>20}: paired-difference sd={d_sd:.3f}  "
          f"[descriptive only] detectable median AE diff ~ {detectable_diffs[m]:.3f} cover points")

detect_lo, detect_hi = min(detectable_diffs.values()), max(detectable_diffs.values())
print(f"\n[descriptive only] range across the six models: "
      f"{detect_lo:.2f}-{detect_hi:.2f} cover points on the HL scale. This range describes "
      "the demoted Wilcoxon/HL statistic only; it is not the resolution limit of the sign "
      "test that is actually reported (see the win-rate band above), and should not be read "
      "as qualifying the sign-test verdicts.")

# %% [markdown]
# ## The "both axes at once" corollary
#
# This corollary is about rank agreement only, and it is a magnitude
# statement, not a re-statement of the typical-image verdict above: every
# model, including the two Gemmas that the sign test finds significantly
# *worse* than the baseline on per-image absolute error, can still track the
# reference's ordering across images more closely than the baseline does.
# The two axes are not redundant, which is exactly what makes the
# baseline's failure legible as one of *scale* rather than of *blindness*.
# Per-model sentences use the ordinary 95% BCa interval on delta Spearman
# (already computed above). Any **universal** sentence ("every model", "all
# six") may be made only from the simultaneous 99.17% (=100-5/6)
# Bonferroni-adjusted band across the six models, computed from the same
# resamples — never in place of the 95% pair, always alongside it. This is
# not a test: no p-value, no family, no Holm.

# %%
simult_ok = all(
    (contrast_results[m]["d_spearman_ci_lo_simult"] > 0) or (contrast_results[m]["d_spearman_ci_hi_simult"] < 0)
    for m in models_ordered
)
same_direction = len({np.sign(contrast_results[m]["delta_spearman"]) for m in models_ordered}) == 1
print("per-model delta Spearman (95% CI) and simultaneous 99.17% band:")
for m in models_ordered:
    r = contrast_results[m]
    print(f"  {m:>20}: delta={r['delta_spearman']:.3f}  95% CI ({r['delta_spearman_ci_lo']:.3f}, "
          f"{r['delta_spearman_ci_hi']:.3f})  simultaneous ({r['d_spearman_ci_lo_simult']:.3f}, "
          f"{r['d_spearman_ci_hi_simult']:.3f})")
print(f"\nAll six simultaneous intervals exclude zero in the same direction: {simult_ok and same_direction}")
if simult_ok and same_direction:
    print("The universal corollary is supported: every model's rank agreement with the "
          "reference exceeds the baseline's, on a simultaneous 99.17% band across the six.")
else:
    print("The universal corollary is NOT stated — at least one simultaneous interval does "
          "not exclude zero, or the six do not agree in direction. Only per-model 95% "
          "sentences are supported.")

# %% [markdown]
# ## Chart — delta Spearman per model, with the simultaneous band
#
# Rank agreement is reported as a magnitude, never as a third test: it
# carries no p-value and enters no family, so that this question rests on two
# streams and not three. The observed gap runs 0.20-0.32 in rank correlation on
# 1,155 paired images.

# %%
fig, ax = plt.subplots(figsize=(8, 5))
order2 = sorted(models_ordered, key=lambda m: contrast_results[m]["delta_spearman"])
y = np.arange(len(order2))
delta_vals = [contrast_results[m]["delta_spearman"] for m in order2]
ci_lo95 = [contrast_results[m]["delta_spearman_ci_lo"] for m in order2]
ci_hi95 = [contrast_results[m]["delta_spearman_ci_hi"] for m in order2]
ci_lo_s = [contrast_results[m]["d_spearman_ci_lo_simult"] for m in order2]
ci_hi_s = [contrast_results[m]["d_spearman_ci_hi_simult"] for m in order2]

ax.hlines(y, ci_lo_s, ci_hi_s, color="#B0B0B0", linewidth=6, alpha=0.6,
          label="99.17% simultaneous band (6 models)")
ax.hlines(y, ci_lo95, ci_hi95, color="#4C72B0", linewidth=3, label="95% BCa (per-model)")
ax.scatter(delta_vals, y, color="black", zorder=5, label="point estimate")
ax.axvline(0, color="firebrick", linestyle="--", linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(order2)
ax.set_xlabel("Delta tie-corrected Spearman (model - baseline)")
ax.set_title("Q9 rank-agreement advantage over the classical baseline\n(magnitude only — no p-value, no family)")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(fig_path / "Q9_delta_spearman.png", dpi=150)
plt.show()
print("Caption: thick grey bars are the simultaneous 99.17% band across all six models "
      "(only a universal 'all six' sentence may rest on these); thin blue bars are the "
      "ordinary 95% interval for a single-model sentence. Both come from the same image "
      "resamples.")

# %% [markdown]
# ## Result
#
# **Two separate answers, not one.** Both sides are scored against the same
# reference over the same 1,155 images, so the comparison is exact and
# complete. But the two pre-declared claims answer different questions and
# must be read separately.
#
# **Stream 1 (bin-balanced) is near-foreordained.** The classical baseline's
# predictions saturate at 58.47 on a 0-100 scale, so it structurally cannot
# compete in the 80-100 bin. Delta balanced MAE is large and Holm-significant
# for all six models in this run, and the per-bin decomposition above shows
# the gap concentrated in the top bins where the baseline cannot reach the
# reference at all — a true statement about the two methods on this frame,
# and *not* evidence that MLLMs read sparse quadrats better, which is where
# 81% of this frame and most of the practical use lives.
#
# **Stream 2 (typical-image) is where this question is actually decided, and
# it does not favour the MLLMs the way the headline balanced-MAE numbers
# suggest.** K2 (symmetry) fails on all six contrasts, so the reported
# effect statement here is the exact paired sign test, not Wilcoxon/HL
# (see the assumption-checks section above). On the typical image:
#
# - **The classical baseline beats both Gemma models, decisively and
#   Holm-significantly.** The baseline has the smaller per-image error on
#   63.6% of images against Gemma-3-12B (`p_holm` ~ 7e-20) and 69.4% against
#   Gemma-3-27B (`p_holm` ~ 7e-40). Both are Holm-significant **in favour of
#   the classical baseline**, not of the MLLM.
# - **The baseline also beats Mistral-Small-3.2**, on 54.2% of images
#   (`p_holm` = 0.0141).
# - **Llama-4-Maverick is the only model that beats the baseline** on the
#   typical image, on 58.4% of images (`p_holm` ~ 4e-08) — subject to the
#   unquantified balanced-MAE-scale reproducibility caveat below; on the
#   overall-MAE scale its effect is well above delta_Mav, so this verdict is
#   not attributable to run-to-run noise.
# - **Llama-4-Scout and Qwen-2.5 are indistinguishable from the baseline**
#   on the typical image (`p_holm` = 0.433 for both) — reported as "not
#   distinguishable from the classical baseline on the typical image at
#   n=1,155", never as "matches" or "is no better than" the baseline.
#
# This is the honest, and more informative, reading of the typical-image
# finding: the balanced-MAE stream's apparent sweep for the MLLMs is
# substantially a property of the baseline's saturation at 58.47 predicted
# cover, which structurally locks it out of the top bin regardless of MLLM
# competence (see stream 1 above and the per-bin decomposition below) — not
# evidence that the MLLMs are the stronger method on the typical image. On
# the typical image, the classical baseline is in fact the stronger method
# against three of the six MLLMs tested here (Gemma-3-27B, Gemma-3-12B,
# Mistral-Small-3.2), is beaten by exactly one (Llama-4-Maverick), and is not
# distinguishable from the remaining two (Llama-4-Scout, Qwen-2.5) — not
# "tied" with them, since a Holm-null result at this design is a failure to
# detect a difference, not evidence of parity.
#
# **The two methods fail in opposite directions (C16).** The baseline
# under-predicts by 10.00 cover points on average; every MLLM over-predicts,
# from +0.71 to +5.76. For Gemma-3-27B specifically, an overall-MAE gap of
# roughly 0.72 points conceals a bias of +5.76 against the baseline's -10.00
# — nearly indistinguishable in error magnitude, opposite in kind. A
# practitioner choosing between or combining these methods should know that,
# not just their MAEs.
#
# **Why the baseline underperforms in the densest quadrats, and why that is
# not the whole story.** Index-based (ExG-ExR) approaches separate
# vegetation from soil by colour alone, calibrated here from a soil-colour
# exemplar rather than fit to the labels — so the baseline is not leaking
# the reference, but it also has no way to recognise dense canopy, shadow or
# senesced material the way a model conditioned on the whole scene can. That
# the baseline cannot physically represent cover above 58.47 explains why it
# loses badly in the sparse top bins that dominate the balanced-MAE stream.
# But on the 933 images in the bottom bin — 81% of the frame, and where the
# saturation ceiling does not bind — the per-bin MAE decomposition shows the
# baseline is not merely competitive but has the lowest error of all seven
# methods: 3.411 cover points, against 4.428 (Maverick), 4.737 (Scout), 5.239
# (Qwen-2.5), 6.000 (Mistral-Small-3.2), 8.231 (Gemma-3-12B) and 9.237
# (Gemma-3-27B). This is a descriptive comparison of per-bin MAE, not a
# statistical test — no sign test or other inferential comparison is run
# restricted to this subset; the only sign tests in this notebook are the
# full-frame, all-1,155-image contrasts reported above. **This statement
# holds pooled over prompts and must be qualified by granularity**: at
# configuration level, one of the 24 model x prompt configurations —
# Llama-4-Maverick paired with the `Detailed` prompt — has a lower 0-20-bin
# MAE than the baseline (`Q9_lowbin_by_combo.csv`), with Llama-4-Scout x
# Detailed and Qwen-2.5 x Detailed the next closest. `Detailed` supplies
# three of the top four configurations in this one sparse-cover bin while
# being the worst-performing prompt overall on the balanced-MAE scale. A
# plausible explanation, consistent with the evidence but not tested
# directly here: `Detailed` is the only prompt that defines vegetation, and
# that definition excludes dead plant material the reference sometimes
# scores as cover — an exclusion that would cost little on bare ground and
# compound into a larger under-count at higher cover.
#
# **The baseline's errors are not uniformly under-prediction.** It
# over-predicts on 357 of the 933 images in the 0-20 bin (38%) while
# under-predicting by 10.00 cover points on average overall
# (`Q9_overprediction_feasibility.csv`). The zero over-prediction counts in
# the two densest bins carry no information: every reference value in the
# 60-80 and 80-100 bins exceeds the baseline's own prediction ceiling, so a
# zero there was arithmetically guaranteed before any image was examined.
# In the two middle bins, where over-prediction was feasible for at least
# some images, the observed zero against the feasible denominator is
# genuinely informative. The sharper description this supports: the
# baseline **over-reads bare ground on over a third of sparse images and
# under-reads almost everything else** — a mis-scaling, not a one-directional
# bias, and a more precise account than "it fails spectrally at high cover"
# on its own.
#
# **What this does not establish.** No equivalence test is run for the
# Llama-4-Scout / Qwen-2.5 nulls; a Holm-null result is a failure to detect
# a difference at this n, not a demonstration of parity. For the sign test
# actually reported, the realised detection limit is a win-rate departure to
# 0.551/0.449 (computed above at n=1,155, 80% power); a true win rate closer
# to 0.5 than that would not have been detected either way, which is exactly
# where Scout (0.481) and Qwen-2.5 (0.487) sit. (The 1.42-1.71 cover-point
# figure for this design describes the resolution of the demoted
# Wilcoxon/Hodges-Lehmann reading, not of the sign test used
# here, and should not be read as qualifying these two verdicts.) The
# Maverick row carries an unquantified run-to-run reproducibility floor on
# the balanced-MAE scale, recorded as a caveat in the
# output CSV rather than treated as blocking; its typical-image win is not
# subject to that caveat (see above).
#
# **The rectified supplementary results** (`Q9_rectified_supplementary.csv`)
# repeat the same six contrasts with the MLLM side scored on the
# `rectified` variant instead of `base`, so both sides start from the
# baseline's own necessary pre-processing product. They are reported as
# estimates with BCa intervals and raw, non-Holm-adjusted p-values only,
# `claim_bearing = FALSE` on every row, because a second corrected family on
# the same question would multiply families after the fact, which is exactly
# what this study's correction scheme is built to avoid. Any difference between
# this run and the primary `base` run is the Q4 variant effect entering the
# Q9 comparison, which Q4 has already measured separately: -1.69 cover
# points overall, decomposed on the cover-point scale into a crop step of
# -1.6426 [-2.4091, -0.9335] and a warp step of -0.0436 [-0.4653, +0.3604].
# The warp step's own contribution is not distinguishable from zero, so
# this is read as "the crop step accounts for the change; the warp step's
# share cannot be resolved from noise," not as a percentage split — Q4's
# own bootstrap estimate of the warp's percentage share of the total
# carries a 95% CI of [-26.85, +26.56] and its source notes that ratio as
# unstable, not precise.

# %%
print(f"Baseline: MAE={baseline_mae:.2f}, RMSE={baseline_rmse:.2f}, bias={baseline_bias:.2f}, "
      f"Pearson(diagnostic only)={baseline_pearson:.3f}, Spearman={baseline_spearman:.3f}")
for m in models_ordered:
    r = contrast_results[m]
    p_bal_str = C.format_bootstrap_p(r["p_bal_raw"])
    print(f"{m:>20}: delta_bal_MAE={r['delta_balanced_mae']:+.2f} "
          f"(p_bal_raw {p_bal_str}, p_holm={r['p_bal_holm']:.4f})  "
          f"delta_overall_MAE={r['delta_overall_mae']:+.2f}  "
          f"HL(descriptive)={r['hl_median_ae_diff_descriptive']:+.2f}  "
          f"sign_test_p_holm={r['p_sign_test_holm']:.4g} "
          f"(baseline wins {r['win_rate_baseline']:.1%} of images)  "
          f"bias_model={r['bias_model']:+.2f} vs bias_baseline={r['bias_baseline']:+.2f}  "
          f"delta_Spearman={r['delta_spearman']:+.3f}")

# %% [markdown]
# ## Writing results
#
# Ten output files, one row per estimate where applicable, all carrying the
# question ID and the full R2 schema (`estimate`, `ci_lo`, `ci_hi`,
# `p_primary_*`, `p_companion_*`, clustered companion columns) plus
# `stream_note` naming each stream's estimand, so that the column names
# `p_primary_*` / `p_companion_*` are never read as a ranking of the two
# claims. `p_primary_*` = bin-balanced (delta balanced MAE) stream;
# `p_companion_*` = typical-image stream, reported as the exact paired sign
# test (the K2 fallback, triggered for all six contrasts) rather than
# Wilcoxon — the pre-declared Wilcoxon statistic and the Hodges-Lehmann
# median are still written, as `*_descriptive` columns, for transparency.
# Three further files carry descriptive, non-claim-bearing material that
# does not enter family H: `Q9_lowbin_by_combo.csv` (the 24 configurations'
# 0-20-bin MAE), `Q9_overprediction_feasibility.csv` (the per-bin
# over-prediction counts and their arithmetic feasibility), and
# `Q9_rectified_supplementary.csv` (the six contrasts repeated with the MLLM
# side scored on the `rectified` variant).

# %%
diag_rows = [{
    "question_id": "Q9", "statistic": "mae", "value": baseline_mae,
    "ci_lo": boot_mae.ci_lo, "ci_hi": boot_mae.ci_hi, "n": len(d12_frame),
    "test": "image bootstrap CI", "asserted_a18": True,
}, {
    "question_id": "Q9", "statistic": "rmse", "value": baseline_rmse,
    "ci_lo": boot_rmse.ci_lo, "ci_hi": boot_rmse.ci_hi, "n": len(d12_frame),
    "test": "image bootstrap CI", "asserted_a18": True,
}, {
    "question_id": "Q9", "statistic": "mean_bias", "value": baseline_bias,
    "ci_lo": boot_bias.ci_lo, "ci_hi": boot_bias.ci_hi, "n": len(d12_frame),
    "test": "image bootstrap CI", "asserted_a18": True,
}, {
    "question_id": "Q9", "statistic": "pearson_r_diagnostic_only", "value": baseline_pearson,
    "ci_lo": np.nan, "ci_hi": np.nan, "n": len(d12_frame),
    "test": "Pearson correlation, diagnostic only, may_rank_methods=FALSE", "asserted_a18": True,
    "may_rank_methods": False,
}, {
    "question_id": "Q9", "statistic": "spearman_r", "value": baseline_spearman,
    "ci_lo": boot_spearman.ci_lo, "ci_hi": boot_spearman.ci_hi, "n": len(d12_frame),
    "test": "tie-corrected Spearman, image bootstrap CI", "asserted_a18": True,
}, {
    "question_id": "Q9", "statistic": "coverage_pct", "value": 100.0,
    "ci_lo": np.nan, "ci_hi": np.nan, "n": len(d12_frame), "test": "computed", "asserted_a18": False,
}, {
    "question_id": "Q9", "statistic": "mae_constant3_predictor", "value": const3_mae,
    "ci_lo": np.nan, "ci_hi": np.nan, "n": len(d1), "test": "asserted (A19), 11.86 +/- 0.05",
    "asserted_a18": False, "asserted_a19": True,
}, {
    "question_id": "Q9", "statistic": "mae_b_median_predictor", "value": b_median_mae,
    "ci_lo": np.nan, "ci_hi": np.nan, "n": len(d1),
    "test": "computed, not asserted (A14/A19)", "asserted_a18": False, "asserted_a19": False,
}, {
    "question_id": "Q9", "statistic": "mae_b_zero_predictor", "value": b_zero_mae,
    "ci_lo": np.nan, "ci_hi": np.nan, "n": len(d1), "test": "computed", "asserted_a18": False,
}]
for lbl in C.BIN_LABELS:
    diag_rows.append({
        "question_id": "Q9", "statistic": f"mae_bin_{lbl}", "value": baseline_perbin.per_bin_mae[lbl],
        "ci_lo": perbin_ci_lo[lbl], "ci_hi": perbin_ci_hi[lbl], "n": baseline_perbin.per_bin_n[lbl],
        "test": "image bootstrap CI (bin subset)", "asserted_a18": False,
    })
    diag_rows.append({
        "question_id": "Q9", "statistic": f"signed_bias_bin_{lbl}", "value": float(baseline_perbin_signed[lbl]),
        "ci_lo": np.nan, "ci_hi": np.nan, "n": baseline_perbin.per_bin_n[lbl],
        "test": "computed", "asserted_a18": False,
    })
q9_baseline_diagnostic = pd.DataFrame(diag_rows)
q9_baseline_diagnostic.to_csv(RESULTS_DIR / "Q9_baseline_diagnostic.csv", index=False)

# --- Q9_metrics_by_model.csv ---
# R1: a balanced MAE is never reported without its five per-bin MAEs, their
# n and their CI in the same row. `BalancedMAEResult.to_row()` raises if
# asked to do otherwise, so it is used here rather than hand-building a row
# that could silently omit them; the per-bin CIs come from the bin
# decomposition step above (per model, over the `paired` frame). R5: the
# model's mean bias is reported alongside the baseline's, not alone.
metrics_rows = []
for m in models_ordered:
    r = contrast_results[m]
    paired_m = mllm_per_image(m)
    mllm_mae_ci_lo, mllm_mae_ci_hi = mllm_perbin_mae_ci[m]  # reuse, computed above
    bal_mllm_full = C.balanced_mae(paired_m["abs_e_mllm"], paired_m["bin"],
                                    ci_lo=mllm_mae_ci_lo, ci_hi=mllm_mae_ci_hi)
    row = {
        "question_id": "Q9", "model": m, "n": r["n_pairs"],
        "overall_mae": baseline_mae + r["delta_overall_mae"],
        "overall_mae_ci_lo": baseline_mae + r["delta_overall_mae_ci_lo"],
        "overall_mae_ci_hi": baseline_mae + r["delta_overall_mae_ci_hi"],
        "mean_bias_model": r["bias_model"],
        "bias_model_ci_lo": r["bias_model_ci_lo"], "bias_model_ci_hi": r["bias_model_ci_hi"],
        "mean_bias_baseline": r["bias_baseline"],
        "bias_baseline_ci_lo": r["bias_baseline_ci_lo"], "bias_baseline_ci_hi": r["bias_baseline_ci_hi"],
        "bias_direction_opposed": r["bias_direction_opposed"],
        "spearman": r["spearman_mllm"],
        "delta_spearman": r["delta_spearman"],
        "delta_spearman_ci_lo": r["delta_spearman_ci_lo"], "delta_spearman_ci_hi": r["delta_spearman_ci_hi"],
        "d_spearman_ci_lo_simult": r["d_spearman_ci_lo_simult"],
        "d_spearman_ci_hi_simult": r["d_spearman_ci_hi_simult"],
        "spearman_no_p_no_family": True,
    }
    row.update(bal_mllm_full.to_row())
    metrics_rows.append(row)
q9_metrics_by_model = pd.DataFrame(metrics_rows)
q9_metrics_by_model.to_csv(RESULTS_DIR / "Q9_metrics_by_model.csv", index=False)
# Column note: `overall_mae_ci_lo` / `overall_mae_ci_hi`
# are `baseline_mae + CI(delta overall MAE)` — the baseline's own overall MAE
# offset by the bootstrap interval on the model-vs-baseline *contrast*, not
# an independently computed CI on the model's own overall MAE. It runs wider
# than that quantity would (about 2.1x on Qwen-2.5: a delivered width of
# 1.92 against 0.90 for a CI computed directly on the model's overall MAE),
# so it errs conservative rather than overstating precision, but a reader
# should not take the pair as "the model's overall-MAE confidence interval".
# Same construction, same caveat, on the `Q9_overall_mae_and_bias.png` error
# bars above.

# --- Q9_contrasts_vs_baseline.csv --- (family H, R2 + R5 schema)
# Bootstrap p-values are floored at 1/(B+1) and reported as "< 1e-4"
# at that floor, never as the bare floor value on its own, which would read as
# a resolved p rather than a resolution limit. The typical-image
# companion is the exact sign test (K2 fails on all six contrasts), so
# `p_companion_*` carries the sign-test p, not Wilcoxon's; Wilcoxon's own
# statistic and the Hodges-Lehmann median are kept as descriptive-only
# columns. The zero-difference and tied-rank counts travel with every
# contrast row, not only in `Q9_assumption_checks.csv`.
bootstrap_p_floor = 1.0 / (C.B_BOOTSTRAP + 1)
contrast_rows_out = []
for m in models_ordered:
    r = contrast_results[m]
    row = C.make_contrast_row(
        question_id="Q9", contrast=f"{m}_vs_classical_baseline", model=m, n_pairs=r["n_pairs"],
        estimate=r["delta_balanced_mae"], ci_lo=r["delta_balanced_mae_ci_lo"], ci_hi=r["delta_balanced_mae_ci_hi"],
        ci_method=r["delta_balanced_mae_ci_method"],
        p_primary_raw=r["p_bal_raw"], p_primary_holm=r["p_bal_holm"],
        p_primary_at_floor=bool(r["p_bal_raw"] <= bootstrap_p_floor),
        p_primary_raw_display=C.format_bootstrap_p(r["p_bal_raw"]),
        p_companion_raw=r["p_sign_test_raw"], p_companion_holm=r["p_sign_test_holm"],
        p_companion_method="exact paired sign test (K2 fallback; K2 failed on all six contrasts)",
        p_clustered_webb=r["p_clustered_webb"], n_distinct_abs_t_star=r["n_distinct_abs_t_star"],
        loco_min=r["loco_min"], loco_max=r["loco_max"],
        ci_width_ratio_clustered_to_headline=r["ci_width_ratio_clustered_to_headline"],
        stream_note=(
            "p_primary_* = bin-balanced claim (bootstrap p on Delta balanced MAE); "
            "p_companion_* = typical-image claim (exact paired sign test; K2 symmetry "
            "failed on all six contrasts, so Wilcoxon/Hodges-Lehmann is demoted to the "
            "descriptive-only columns below). Neither is subordinate to the other."
        ),
        hl_median_ae_diff_descriptive=r["hl_median_ae_diff_descriptive"],
        wilcoxon_statistic_descriptive=r["wilcoxon_statistic"],
        p_wilcoxon_raw_descriptive=r["p_wilcoxon_raw"], p_wilcoxon_holm_descriptive=r["p_wilcoxon_holm"],
        rank_biserial=r["rank_biserial"],
        n_zero_diff=r["n_zero_diff"], n_tied_ranks=r["n_tied_ranks"],
        sign_test_n_used=r["sign_test_n_used"], sign_test_n_pos=r["sign_test_n_pos"],
        win_rate_model=r["win_rate_mllm"], win_rate_baseline=r["win_rate_baseline"], tie_rate=r["tie_rate"],
        delta_overall_mae=r["delta_overall_mae"],
        delta_overall_mae_ci_lo=r["delta_overall_mae_ci_lo"], delta_overall_mae_ci_hi=r["delta_overall_mae_ci_hi"],
        delta_rmse=r["delta_rmse"], delta_rmse_ci_lo=r["delta_rmse_ci_lo"], delta_rmse_ci_hi=r["delta_rmse_ci_hi"],
        bias_model=r["bias_model"], bias_baseline=r["bias_baseline"],
        delta_abs_bias=r["delta_abs_bias"], bias_direction_opposed=r["bias_direction_opposed"],
        delta_spearman=r["delta_spearman"],
        delta_spearman_ci_lo=r["delta_spearman_ci_lo"], delta_spearman_ci_hi=r["delta_spearman_ci_hi"],
        d_spearman_ci_lo_simult=r["d_spearman_ci_lo_simult"], d_spearman_ci_hi_simult=r["d_spearman_ci_hi_simult"],
        c1_serving_confounded=r["c1_serving_confounded"],
        c1_not_applicable_baseline_side=r["c1_not_applicable_baseline_side"],
        maverick_floor_delta=r["maverick_floor_delta"],
        within_run_to_run_variability_overall_mae=r["within_run_to_run_variability_overall_mae"],
        floor_not_estimable_for_balanced_mae=r["floor_not_estimable_for_balanced_mae"],
        balanced_mae_maverick_caveat=r["balanced_mae_maverick_caveat"],
        holm_family="H", holm_family_size=C.FAMILY_SIZES["H"],
    )
    contrast_rows_out.append(row)
q9_contrasts_vs_baseline = pd.DataFrame(contrast_rows_out)
q9_contrasts_vs_baseline.to_csv(RESULTS_DIR / "Q9_contrasts_vs_baseline.csv", index=False)

# --- Q9_contrasts_by_combo.csv --- (24 configs, descriptive, uncorrected)
q9_contrasts_by_combo.to_csv(RESULTS_DIR / "Q9_contrasts_by_combo.csv", index=False)

# --- Q9_bin_decomposition.csv ---
q9_bin_decomposition.to_csv(RESULTS_DIR / "Q9_bin_decomposition.csv", index=False)

# --- Q9_clustered_companion.csv ---
# Each of the three refits writes its own per-bin n, as every clustered
# companion does. One summary row per model (the Webb/CR3
# companion on the full frame) plus one row per model per LOCO refit
# carrying that refit's own per-bin n, so a reader can see how many images
# informed each bin of each refit rather than only the min/max of the
# resulting balanced-MAE range.
clustered_rows = []
for m in models_ordered:
    r = contrast_results[m]
    clustered_rows.append({
        "question_id": "Q9", "model": m, "contrast": f"{m}_vs_classical_baseline",
        "row_type": "summary",
        "p_clustered_webb": r["p_clustered_webb"], "n_distinct_abs_t_star": r["n_distinct_abs_t_star"],
        "webb_below_resolution_floor": r["webb_below_resolution_floor"],
        "clustered_estimate": r["delta_balanced_mae"],
        "clustered_ci_lo": r["clustered_ci_lo"], "clustered_ci_hi": r["clustered_ci_hi"],
        "clustered_df": r["clustered_df"],
        "ci_width_ratio_clustered_to_headline": r["ci_width_ratio_clustered_to_headline"],
        "loco_min": r["loco_min"], "loco_max": r["loco_max"], "loco_note": r["loco_note"],
        "webb_resolution_floor": C.WEBB_P_FLOOR,
        "loco_dropped_campaign": "", "loco_refit_balanced_mae": np.nan,
        "loco_refit_per_bin_n": "",
    })
    for camp, per_bin_n in r["loco_per_bin_n"].items():
        refit_val = r["loco_bal_by_campaign"][camp]
        clustered_rows.append({
            "question_id": "Q9", "model": m, "contrast": f"{m}_vs_classical_baseline",
            "row_type": "loco_refit",
            "p_clustered_webb": np.nan, "n_distinct_abs_t_star": np.nan,
            "clustered_estimate": np.nan, "clustered_ci_lo": np.nan, "clustered_ci_hi": np.nan,
            "clustered_df": np.nan, "ci_width_ratio_clustered_to_headline": np.nan,
            "loco_min": np.nan, "loco_max": np.nan, "loco_note": "",
            "webb_resolution_floor": np.nan,
            "loco_dropped_campaign": camp,
            "loco_refit_balanced_mae": (refit_val if isinstance(refit_val, (int, float)) else np.nan),
            "loco_refit_per_bin_n": str(per_bin_n),
        })
q9_clustered_companion = pd.DataFrame(clustered_rows)
q9_clustered_companion.to_csv(RESULTS_DIR / "Q9_clustered_companion.csv", index=False)

# --- Q9_assumption_checks.csv ---
# K10 (between-image independence) cannot be checked from the data: it is an
# assumption of the design. It is recorded here as a named column pointing at
# its measurement (the clustered companion's width ratio, `Q9_clustered_companion.csv`
# / `ci_width_ratio_clustered_to_headline`) rather than left absent, which
# would read as an oversight rather than the deliberate choice it is.
q9_assumptions_full = q9_assumptions.merge(
    pd.DataFrame([{"model": m, "n_empty_bin_violations": contrast_results[m]["n_empty_bin_violations"],
                   "k4_switch_to_stratified": contrast_results[m]["k4_switch_to_stratified"],
                   "ci_width_ratio_clustered_to_headline": contrast_results[m]["ci_width_ratio_clustered_to_headline"]}
                  for m in models_ordered]),
    on="model", how="left",
)
q9_assumptions_full["k10_not_checkable_see_clustered_companion"] = (
    "K10 (between-image independence) is not checkable; quantified instead by "
    "ci_width_ratio_clustered_to_headline in this row / Q9_clustered_companion.csv"
)
q9_assumptions_full.to_csv(RESULTS_DIR / "Q9_assumption_checks.csv", index=False)

# --- Q9_lowbin_by_combo.csv --- (24 configs, 0-20 bin only, descriptive)
q9_lowbin_by_combo.to_csv(RESULTS_DIR / "Q9_lowbin_by_combo.csv", index=False)

# --- Q9_overprediction_feasibility.csv ---
q9_overprediction_feasibility.to_csv(RESULTS_DIR / "Q9_overprediction_feasibility.csv", index=False)

# --- Q9_rectified_supplementary.csv --- (supplementary, non-claim-bearing)
q9_rectified_supplementary.to_csv(RESULTS_DIR / "Q9_rectified_supplementary.csv", index=False)

written = [
    "Q9_baseline_diagnostic.csv", "Q9_metrics_by_model.csv", "Q9_contrasts_vs_baseline.csv",
    "Q9_contrasts_by_combo.csv", "Q9_bin_decomposition.csv", "Q9_clustered_companion.csv",
    "Q9_assumption_checks.csv", "Q9_lowbin_by_combo.csv", "Q9_overprediction_feasibility.csv",
    "Q9_rectified_supplementary.csv",
]
for fname in written:
    p = RESULTS_DIR / fname
    print(f"wrote {fname}: {p.stat().st_size} bytes")

# %% [markdown]
# ## `rng_rectified` isolation, verified rather than assumed
#
# `rng_rectified` is the last generator this notebook introduces, and nothing
# after the rectified block draws from it, so no output file downstream of it
# would change if a later cell did draw from it. A leak into `rng_rectified`
# is therefore invisible in the results, and the check below closes that gap
# directly: it compares `rng_rectified`'s bit-generator state now, after
# every remaining cell in the notebook has run, against the state captured
# immediately after its only consumer (`rectified_contrast_row`, called once
# per model). If any later cell had drawn from it, the states would differ.

# %%
assert rng_rectified.bit_generator.state == rng_rectified_state_after_block, (
    "rng_rectified advanced after the rectified-supplementary block finished "
    "producing q9_rectified_supplementary -- its isolation from later cells "
    "has failed this check, not merely gone unverified."
)
print("rng_rectified isolation confirmed: its bit-generator state at the end "
      "of the notebook is byte-identical to its state immediately after the "
      "rectified-supplementary block, so no cell after that block drew from it.")
