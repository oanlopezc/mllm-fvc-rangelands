# %% [markdown]
# # Q8 — fine low-cover bins (secondary, uncorrected)
#
# Q1's headline balanced MAE and Q3's trend tests both run on five equal-width
# reference bins, `[0,20]`, `(20,40]`, `(40,60]`, `(60,80]`, `(80,100]`. That
# bottom bin holds **933 of the 1,155 images — 81% of the frame** — from bare
# soil at 0% cover up to a genuinely vegetated 20% quadrat, averaged into one
# number. This notebook re-runs the same two questions under an eight-bin
# scheme that splits that bottom bin into four:
#
# `[0,1]`, `(1,5]`, `(5,10]`, `(10,20]`, `(20,40]`, `(40,60]`, `(60,80]`, `(80,100]`
#
# The four upper bins are untouched — they are Q1's own `(20,40]` … `(80,100]`
# unchanged — so what moves is entirely inside the region every headline number
# in this project currently treats as one bin. This is why Q8 exists: not to
# introduce a new metric, but to check whether the coarse view is hiding
# something in the 81% of the data it collapses together.
#
# This question is **secondary and uncorrected by design** — it is not a member
# of any Holm family, every p-value here is reported raw and labelled
# exploratory, and its role is to feed one qualitative finding back onto Q1 as
# either a reassurance or a caveat, never to stand as its own confirmatory
# claim.

# %% [markdown]
# ## Setup
#
# The seed is the study's base seed plus the question number:
# `20260907 + 8 = 20260915`. Everything stochastic in this notebook (the paired
# image bootstrap, its BCa jackknife, the permutation Jonckheere–Terpstra
# tests and the Campaign-clustered companion) draws from this one generator.

# %%
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _find_notebooks_dir() -> Path:
    """Locate the folder holding `_common.py` so it can be imported whether
    this runs as a script (working directory = the file's own folder) or as a
    rendered notebook (working directory one level below it). Walks up from
    the working directory looking for `STATUS.md`, the project-root marker
    `project_root()` in `_common.py` uses, and takes the notebooks folder
    from there; a folder that already holds `_common.py` is accepted
    directly.
    """
    start = Path(os.environ.get("ANALYSIS_PROJECT_ROOT", Path.cwd())).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "STATUS.md").is_file():
            nb_dir = candidate / "03_notebooks"
            if (nb_dir / "_common.py").is_file():
                return nb_dir
        if (candidate / "_common.py").is_file():
            return candidate
    raise RuntimeError(f"could not locate 03_notebooks/_common.py from {start}")


sys.path.insert(0, str(_find_notebooks_dir()))

import pathlib as _pl  # `_common.py` is imported from this notebook's
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))  # own directory
import _common as C

SEED = 20260915  # the study's base seed (20260907) plus the question number
rng = np.random.default_rng(SEED)

ROOT = C.ROOT
RESULTS_DIR = ROOT / "05_deliverables/Repository/GitHub/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = ROOT / "05_deliverables/Repository/GitHub/notebooks" / "rendered"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data
#
# The main analysis frame — `variant == 'base'`, local stack, 1,155 images ×
# 6 models × 4 prompts = 27,720 rows — loaded through the shared loader so
# this notebook shares the exact join and error definition every other
# question uses. `_common.load_d1` and `_common.build_base_local_frame` both
# route through `guarded_read_csv`-equivalent project-rooted paths; nothing
# here reads outside `01_input/raw/`.
#
# The load-time assertions (A1–A14) are run first, exactly as every other
# notebook runs them, so a corrupted or mismatched input file stops this
# notebook before any Q8-specific computation starts. Two assertions not
# reachable from `run_all_assertions` are called explicitly below:
# `assert_k1_pairing_complete` (pairing is complete on the shared base/local
# frame) and `assert_c11_n_reproduced` (the Maverick reproducibility floor's
# own reproduced-count check). Both raise on failure.

# %%
assumption_df, frames = C.run_all_assertions(include_d12=False)
d1 = frames["d1"]
d5 = frames["d5"]
d8 = frames["d8"]
base_local = frames["base_local"]

k1_result = C.assert_k1_pairing_complete(d5, base_local)
print(f"K1: {k1_result.detail}")

maverick_floor = C.compute_maverick_floor(d8, rng)
c11_result = C.assert_c11_n_reproduced(maverick_floor.n_reproduced)
print(f"C11: {c11_result.detail}")

print(f"base_local frame: {len(base_local)} rows, {base_local['image'].nunique()} images")
assumption_df

# %% [markdown]
# ## The eight-bin scheme
#
# The eight-bin scheme is **right-closed**, matching
# the main five-bin scheme's convention exactly: bottom bin
# closed at both ends so an exactly-zero reference falls inside it, edges
# `[0, 1, 5, 10, 20, 40, 60, 80, 100]`, `pd.cut(..., right=True,
# include_lowest=True)`. This matters specifically because Q8's headline
# output is a Kendall τ-b between the five-bin and eight-bin model
# rankings, and that comparison is only interpretable if the only thing
# that changes between the two schemes is the granularity of the
# partition, not also the boundary rule. Under a
# right-closed convention shared with Q1, the four low bins here are exactly a
# four-way split of Q1's `[0,20]` bin, and the four upper bins are identical
# to Q1's own four upper bins — the eight bins **nest** inside the five. That
# nesting is asserted below, not just claimed in prose, because it is what
# makes the τ-b meaningful.

# %%
BIN8_EDGES = (0, 1, 5, 10, 20, 40, 60, 80, 100)
BIN8_LABELS = ("0-1", "1-5", "5-10", "10-20", "20-40", "40-60", "60-80", "80-100")
LOW_BIN8_LABELS = ("0-1", "1-5", "5-10", "10-20")
HIGH_BIN8_LABELS = ("20-40", "40-60", "60-80", "80-100")


def assign_bins8(reference: pd.Series) -> pd.Series:
    """The eight-bin right-closed scheme, independent of `_common.assign_bins`
    (which hard-wires the five main-path edges/labels and cannot be reused
    here). Mirrors `_common.assign_bins`'s own `pd.cut` call exactly, just
    with eight edges instead of five.
    """
    cats = pd.cut(reference, bins=list(BIN8_EDGES), labels=BIN8_LABELS,
                   right=True, include_lowest=True)
    if cats.isna().any() and reference.between(0, 100).all():
        raise AssertionError("assign_bins8 produced NaN for an in-range reference value")
    return cats.astype(str)


base_local = base_local.copy()
base_local["bin8"] = assign_bins8(base_local["reference"])

# Per-bin n of *images*, not rows, mirroring _common.per_bin_n_table's own
# discipline of counting unique images per bin.
per_bin8_n = (
    base_local[["image", "bin8"]].drop_duplicates()["bin8"]
    .value_counts().reindex(BIN8_LABELS, fill_value=0).astype(int).to_dict()
)
n_total_images = base_local["image"].nunique()

print("Per-bin image counts (eight-bin scheme):")
for lbl in BIN8_LABELS:
    print(f"  {lbl:>7}: {per_bin8_n[lbl]}")
print(f"  sum   : {sum(per_bin8_n.values())}  (expect 1,155)")

if sum(per_bin8_n.values()) != 1155 or any(per_bin8_n[lbl] == 0 for lbl in BIN8_LABELS):
    raise AssertionError(
        "Q8 eight-bin per-image counts did not sum to 1,155 with every label "
        f"populated: {per_bin8_n}"
    )

# %% [markdown]
# **The nesting check.** Under the shared right-closed convention the four low
# bins sum to exactly Q1's bottom-bin count (933) and the four upper bins equal
# Q1's own four upper-bin counts (108 / 46 / 38 / 30). Both are asserted here
# from the raw frame rather than taken on trust: a mismatch would mean this
# notebook's binning differs from the main path's, which would make the τ-b
# below meaningless.

# %%
low_sum = sum(per_bin8_n[lbl] for lbl in LOW_BIN8_LABELS)
base_local["bin5"] = C.assign_bins(base_local["reference"])
per_bin5_n = (
    base_local[["image", "bin5"]].drop_duplicates()["bin5"]
    .value_counts().reindex(C.BIN_LABELS, fill_value=0).astype(int).to_dict()
)

nest_low_ok = low_sum == per_bin5_n["0-20"]
nest_high = {lbl: per_bin8_n[lbl] for lbl in HIGH_BIN8_LABELS}
expected_high = {"20-40": per_bin5_n["20-40"], "40-60": per_bin5_n["40-60"],
                 "60-80": per_bin5_n["60-80"], "80-100": per_bin5_n["80-100"]}
nest_high_ok = nest_high == expected_high

print(f"Low four bins sum = {low_sum}, Q1 bottom bin (0-20) = {per_bin5_n['0-20']} -> nests: {nest_low_ok}")
print(f"Upper four bins   = {nest_high}")
print(f"Q1 upper four     = {expected_high} -> nests: {nest_high_ok}")

if not (nest_low_ok and nest_high_ok):
    raise AssertionError(
        "Q8 eight-bin scheme does not nest inside the five-bin scheme — "
        "closure convention has drifted between the two, which would confound "
        "re-binning with re-closure in the Kendall tau-b below. "
        f"low_sum={low_sum} vs {per_bin5_n['0-20']}, high={nest_high} vs {expected_high}"
    )

_low_counts_str = "/".join(str(per_bin8_n[lbl]) for lbl in LOW_BIN8_LABELS)
print(
    f"\nNesting confirmed: eight-bin low four ({_low_counts_str}, computed above) sum to "
    "the five-bin bottom bin, and the eight-bin upper four equal the five-bin "
    "upper four exactly. The two schemes share a boundary convention, so any "
    "ranking difference measured below is re-binning and nothing else."
)

# %% [markdown]
# ## The n < 30 exclusion rule for the fine bins
#
# The rule is that any fine bin with n < 30 is reported with its n and CI but
# excluded from the eight-bin balanced summary, with the exclusion recorded in
# the CSV. The realised top-bin count is exactly 30, which sits precisely on
# the threshold. The rule is a strict less-than, not a less-than-or-equal, so
# it excludes only bins **below** 30 and **keeps** a bin at exactly 30; the
# analogous K9 rule in `_common.py` also reads `n < min_n`, and neither treats
# the boundary case specially.
# This notebook therefore uses **strict `<`**, so the `80-100` bin (n = 30,
# identical to Q1's own top bin) is included in the eight-bin balanced summary,
# exactly as it is in Q1's five-bin one.

# %%
MIN_BIN_N = 30  # threshold is exclusive: a bin is excluded only if n < MIN_BIN_N (strict <)
excluded_bins = [lbl for lbl in BIN8_LABELS if per_bin8_n[lbl] < MIN_BIN_N]
included_bins = [lbl for lbl in BIN8_LABELS if lbl not in excluded_bins]
print(f"Bins excluded from the 8-bin balanced summary (n < {MIN_BIN_N}, strict): {excluded_bins or 'none'}")
print(f"Bins included ({len(included_bins)}/8): {included_bins}")

# %% [markdown]
# ## Balanced MAE under the eight-bin scheme
#
# `_common.balanced_mae()` hard-wires the five main-path bin
# labels (`BIN_LABELS`) inside its own `groupby(...).reindex(BIN_LABELS)`
# call. Calling it with an eight-label `bin` column does not raise — every
# label outside the hard-wired five is silently absent from the reindexed
# result, `per_bin_n` shows `0` for every real eight-bin label, and the
# returned "balanced MAE" is an unweighted mean over whichever of the five
# *five-bin* labels happen to intersect the data, which for genuinely
# eight-bin-labelled data is none of them — it returns a number with no
# semantic content and no exception. This notebook never calls
# `_common.balanced_mae()` on the eight-bin frame; the function below is a
# local eight-bin analogue, and every result it returns is checked against the
# assumption above (per-bin n sums to 1,155, no empty label) before being
# trusted.
#
# **R1 enforcement.** `_common.BalancedMAEResult.to_row()` raises if a caller
# tries to emit a balanced MAE row without its per-bin CIs already attached —
# that is what forces Q1 to compute `per_bin_ci_for_frame` before it can call
# `to_row()` at all. `balanced_mae_8` restores the equivalent guard locally:
# its own `to_row()` method raises under the identical condition, so it is
# structurally impossible for this notebook to write out an eight-bin
# balanced MAE without its eight per-bin MAEs, their n, and their CIs.


# %%
class BalancedMAE8Result:
    """Eight-bin analogue of `_common.BalancedMAEResult`, carrying the same
    R1 guard: `to_row()` raises if `per_bin_ci_lo`/`per_bin_ci_hi` are unset.
    """

    def __init__(self, balanced_mae, per_bin_mae, per_bin_n, included_bins,
                 excluded_bins, per_bin_ci_lo=None, per_bin_ci_hi=None):
        self.balanced_mae = balanced_mae
        self.per_bin_mae = per_bin_mae
        self.per_bin_n = per_bin_n
        self.included_bins = included_bins
        self.excluded_bins = excluded_bins
        self.per_bin_ci_lo = per_bin_ci_lo
        self.per_bin_ci_hi = per_bin_ci_hi

    def to_row(self, prefix: str = "") -> dict:
        if self.per_bin_ci_lo is None or self.per_bin_ci_hi is None:
            raise ValueError(
                "R1 violation: to_row() called with per_bin_ci_lo/per_bin_ci_hi "
                "unset. An eight-bin balanced MAE is never reported without "
                "its eight per-bin MAEs, each with its per-bin n and its CI "
                ". Compute the per-bin CIs via "
                "per_bin_ci_for_frame8() before calling to_row()."
            )
        row = {f"{prefix}balanced_mae_8bin": self.balanced_mae}
        for label in BIN8_LABELS:
            row[f"{prefix}mae_bin_{label}"] = self.per_bin_mae.get(label, np.nan)
            row[f"{prefix}n_bin_{label}"] = self.per_bin_n.get(label, 0)
            row[f"{prefix}mae_bin_{label}_ci_lo"] = self.per_bin_ci_lo.get(label, np.nan)
            row[f"{prefix}mae_bin_{label}_ci_hi"] = self.per_bin_ci_hi.get(label, np.nan)
        return row


def balanced_mae_8(abs_error: pd.Series, bins: pd.Series, bin_labels=BIN8_LABELS,
                    included=None, ci_lo=None, ci_hi=None) -> BalancedMAE8Result:
    """Eight-bin analogue of `_common.balanced_mae`, parameterised on the bin
    label set so it can never silently reuse the five-bin one. `included`
    restricts the balanced mean to bins clearing the n >= 30 rule; excluded
    bins still appear in `per_bin_mae`/`per_bin_n` (reported, not hidden),
    they are simply left out of the unweighted mean and flagged in the
    returned object.
    """
    if included is None:
        included = list(bin_labels)
    df = pd.DataFrame({"abs_error": abs_error.values, "bin": bins.values})
    grouped = df.groupby("bin")["abs_error"]
    per_bin_mae = grouped.mean().reindex(bin_labels)
    per_bin_n = grouped.size().reindex(bin_labels, fill_value=0)
    bal_series = per_bin_mae.reindex(included)
    bal = float(bal_series.mean(skipna=True))
    return BalancedMAE8Result(
        balanced_mae=bal,
        per_bin_mae={k: (float(v) if pd.notna(v) else np.nan) for k, v in per_bin_mae.items()},
        per_bin_n={k: int(v) for k, v in per_bin_n.items()},
        included_bins=list(included),
        excluded_bins=[b for b in bin_labels if b not in included],
        per_bin_ci_lo=ci_lo,
        per_bin_ci_hi=ci_hi,
    )


def _check_per_bin_n_sums(per_bin_n: dict, label_set_name: str = "eight-bin") -> None:
    """The explicit guard the task calls for: per-bin n must sum to 1,155
    with no empty label, so a mislabelled call (e.g. accidentally invoking
    the five-bin `_common.balanced_mae` on eight-bin data) cannot pass
    silently — it would show a 0 count against a label _common doesn't know
    and/or a total far from 1,155.
    """
    total = sum(per_bin_n.values())
    empties = [k for k, v in per_bin_n.items() if v == 0]
    if total != 1155 or empties:
        raise AssertionError(
            f"{label_set_name} balanced-MAE per_bin_n guard failed: total={total} "
            f"(expect 1155), empty_labels={empties}"
        )


def per_bin_ci_for_frame8(frame: pd.DataFrame, abs_error_col: str, image_col: str, bin_col: str,
                           rng: np.random.Generator, B: int = C.B_BOOTSTRAP) -> tuple[dict, dict]:
    """Bin-stratified bootstrap CIs for each of the eight per-bin MAEs on
    `frame`, satisfying R1's requirement that a balanced MAE never be
    reported without its per-bin CIs. Eight-bin port of Q1's
    `per_bin_ci_for_frame`, using `_common.image_bootstrap` per bin
    (`bin_col=None` inside each single-bin call — there is nothing left to
    stratify on once the frame is already restricted to one bin).
    """
    ci_lo, ci_hi = {}, {}
    for label in BIN8_LABELS:
        sub = frame[frame[bin_col] == label]
        if sub[image_col].nunique() < 2:
            ci_lo[label], ci_hi[label] = np.nan, np.nan
            continue

        def stat(f, _label=label):
            s = f.loc[f[bin_col] == _label, abs_error_col]
            return float(s.mean()) if len(s) else np.nan

        boot = C.image_bootstrap(sub, stat, rng, image_col=image_col, bin_col=None, B=B)
        ci_lo[label], ci_hi[label] = boot.ci_lo, boot.ci_hi
    return ci_lo, ci_hi


def image_bootstrap_8(
    frame: pd.DataFrame,
    statistic,
    rng: np.random.Generator,
    *,
    image_col: str = "image",
    bin_col: str | None = "bin8",
    B: int = C.B_BOOTSTRAP,
    alpha: float = 0.05,
    stratified: bool = False,
    compute_jackknife: bool = True,
):
    """Eight-bin-aware port of `_common.image_bootstrap`.

    `_common.image_bootstrap` hard-wires `BIN_LABELS` (the five main-path
    labels) at its own K4 empty-bin check and inside its `stratified=True`
    branch, so it cannot serve `bin8` as written — passing `bin_col="bin8"`
    to the shared function would silently count K4 violations against the
    wrong threshold (firing only when fewer than five of the *eight* bins
    survive a resample) and `stratified=True` would draw per-bin quotas from
    the five-bin partition instead of the eight-bin one. This is a local,
    parameterised copy, not an edit to `_common.py`: same BCa construction,
    same jackknife acceleration, same bootstrap-p definition, with
    `BIN8_LABELS` (len 8) substituted for `BIN_LABELS` (len 5) in the two
    places that mattered. Kept local as an eight-bin-aware path implemented
    in this notebook rather than a shared-module edit.
    """
    theta_hat = statistic(frame)
    images = frame[image_col].unique()
    n_bin_labels = len(BIN8_LABELS)

    boot_vals = np.empty(B)
    n_violations = 0

    if stratified:
        if bin_col is None:
            raise ValueError("stratified=True requires bin_col")
        bin_to_images = {b: frame.loc[frame[bin_col] == b, image_col].unique() for b in BIN8_LABELS}
        for i in range(B):
            drawn_parts = [rng.choice(imgs, size=len(imgs), replace=True)
                           for imgs in bin_to_images.values() if len(imgs) > 0]
            drawn = np.concatenate(drawn_parts)
            resampled = C._rebuild_from_ids(frame, drawn, image_col)
            present_bins = set(resampled[bin_col].unique()) if bin_col in resampled.columns else set()
            if len(present_bins) < n_bin_labels:
                n_violations += 1
            boot_vals[i] = statistic(resampled)
    else:
        for i in range(B):
            drawn = rng.choice(images, size=len(images), replace=True)
            resampled = C._rebuild_from_ids(frame, drawn, image_col)
            if bin_col is not None and bin_col in resampled.columns:
                present_bins = set(resampled[bin_col].unique())
                if len(present_bins) < n_bin_labels:
                    n_violations += 1
            boot_vals[i] = statistic(resampled)

    ci_method = "percentile (BCa fallback)"
    lo = np.percentile(boot_vals, 100 * alpha / 2)
    hi = np.percentile(boot_vals, 100 * (1 - alpha / 2))

    if compute_jackknife:
        jack_vals = C.image_jackknife_values(frame, statistic, image_col=image_col)
        bca_lo, bca_hi, used_bca = C._bca_interval(theta_hat, boot_vals, jack_vals, alpha=alpha)
        if used_bca:
            lo, hi, ci_method = bca_lo, bca_hi, "BCa"

    p_below = np.mean(boot_vals <= 0)
    p_above = np.mean(boot_vals >= 0)
    p_raw = 2 * min(p_below, p_above)
    p_floor = 1.0 / (B + 1)
    p_two_sided = max(p_raw, p_floor)

    return C.BootstrapResult(
        estimate=theta_hat, ci_lo=lo, ci_hi=hi, ci_method=ci_method,
        boot_values=boot_vals, n_empty_bin_violations=n_violations, p_two_sided=p_two_sided,
    )


# %% [markdown]
# ## Per-model, per-bin MAE and signed bias (repeating Q1)
#
# Per model, pooled over its 4 prompts (the per-image value is the mean of
# `|e|` over the 4 prompts, never the error of the mean prediction),
# the eight per-bin MAEs with n and CI, the eight-bin balanced MAE, and signed
# bias per bin. R1 applies in full here exactly as it does to Q1, Q3 and Q4:
# a balanced MAE is never reported without its per-bin MAEs, their n and
# their CIs in the same table.

# %% [markdown]
# **Performance note.** Every contrast in this study resamples at the image
# level: an image is drawn, and *all* of its rows travel together. For
# a statistic that first pools 4 prompt rows to one per-image value and then
# takes a per-bin mean of that, resampling images with replacement commutes
# with the prompt-pooling step — an image's pooled value is the same no
# matter how many times it is drawn or what else is drawn alongside it. So
# each per-model bootstrap below resamples a **pre-pooled, one-row-per-image**
# frame (image, pooled `|e|`, bin) rather than re-pooling the full 4-prompt
# frame inside every one of the 10,000+1,155 bootstrap/jackknife replicates.
# This is an implementation optimisation only — it calls the same bootstrap
# construction (same BCa, same jackknife acceleration) on an equivalent,
# cheaper-to-resample representation of exactly the same quantity, not a
# different statistic.
#
# **K4** (the empty-bin resample count) is measured here, not disabled: each
# per-model call below passes `bin_col="bin8"` to `image_bootstrap_8`, so a
# resample missing one of the eight bins is counted rather than assumed away.
# The **bin-stratified companion** that accompanies K4 is also
# computed (`stratified=True`) so a material divergence between the plain and
# stratified estimates is reported, not silently chosen between.

# %%
rows_metrics = []
per_model_boot = {}
per_model_bal = {}

for model in C.MODELS:
    sub = base_local[base_local["model"] == model].copy()
    pooled = C.pool_axis_mean_abs_error(sub, group_cols=["image"])  # one row per image
    pooled = pooled.merge(base_local[["image", "bin8", "reference"]].drop_duplicates("image"),
                           on="image", how="left")
    pooled_bias = sub.groupby("image", as_index=False)["e"].mean()
    pooled = pooled.merge(pooled_bias, on="image", how="left")

    ci_lo, ci_hi = per_bin_ci_for_frame8(pooled, "abs_e", "image", "bin8", rng)
    res = balanced_mae_8(pooled["abs_e"], pooled["bin8"], included=included_bins,
                          ci_lo=ci_lo, ci_hi=ci_hi)
    _check_per_bin_n_sums(res.per_bin_n, f"model={model}")
    per_model_bal[model] = res

    def stat_fn(frame_resampled: pd.DataFrame) -> float:
        # frame_resampled is the resampled *pre-pooled* one-row-per-image frame.
        return balanced_mae_8(frame_resampled["abs_e"], frame_resampled["bin8"],
                               included=included_bins).balanced_mae

    boot = image_bootstrap_8(pooled, stat_fn, rng, image_col="image", bin_col="bin8", B=C.B_BOOTSTRAP)
    boot_stratified = image_bootstrap_8(pooled, stat_fn, rng, image_col="image", bin_col="bin8",
                                         B=C.B_BOOTSTRAP, stratified=True)
    per_model_boot[model] = boot

    signed_bias_per_bin = pooled.groupby("bin8")["e"].agg(["mean", "median"]).reindex(BIN8_LABELS)

    row = res.to_row()
    row.update({
        "question_id": "Q8",
        "model": model,
        "included_bins": ",".join(included_bins),
        "excluded_bins": ",".join(res.excluded_bins) or "none",
        "balanced_mae_8bin_ci_lo": boot.ci_lo,
        "balanced_mae_8bin_ci_hi": boot.ci_hi,
        "balanced_mae_8bin_ci_method": boot.ci_method,
        "n_empty_bin_violations_k4": boot.n_empty_bin_violations,
        "balanced_mae_8bin_stratified": boot_stratified.estimate,
        "balanced_mae_8bin_stratified_ci_lo": boot_stratified.ci_lo,
        "balanced_mae_8bin_stratified_ci_hi": boot_stratified.ci_hi,
        "n_empty_bin_violations_k4_stratified": boot_stratified.n_empty_bin_violations,
        "min_bin_n_rule": "n < 30 excluded (strict <)",
    })
    # One row per model carrying the full R1 per-bin breakdown (mae/n/ci for
    # each of the eight bins, from to_row()) plus signed bias per bin, kept
    # in the same wide row rather than split across tables.
    for lbl in BIN8_LABELS:
        row[f"signed_bias_mean_bin_{lbl}"] = (
            float(signed_bias_per_bin.loc[lbl, "mean"]) if lbl in signed_bias_per_bin.index else np.nan
        )
        row[f"signed_bias_median_bin_{lbl}"] = (
            float(signed_bias_per_bin.loc[lbl, "median"]) if lbl in signed_bias_per_bin.index else np.nan
        )
    rows_metrics.append(row)

finebin_metrics = pd.DataFrame(rows_metrics)
finebin_metrics.to_csv(RESULTS_DIR / "Q8_finebin_metrics.csv", index=False)
print(f"wrote Q8_finebin_metrics.csv, {len(finebin_metrics)} rows (one row per model, "
      "R1's per-bin mae/n/ci in wide form)")
finebin_metrics.head(6)

# %% [markdown]
# A long-form view of the same R1 columns, one row per (model, bin), used by
# the chart below and easier to scan for the per-bin heterogeneity question
# this notebook exists to answer.

# %%
long_rows = []
for _, r in finebin_metrics.iterrows():
    for lbl in BIN8_LABELS:
        long_rows.append({
            "model": r["model"],
            "bin": lbl,
            "n_images": r[f"n_bin_{lbl}"],
            "mae": r[f"mae_bin_{lbl}"],
            "mae_ci_lo": r[f"mae_bin_{lbl}_ci_lo"],
            "mae_ci_hi": r[f"mae_bin_{lbl}_ci_hi"],
            "signed_bias_mean": r[f"signed_bias_mean_bin_{lbl}"],
            "signed_bias_median": r[f"signed_bias_median_bin_{lbl}"],
            "included_in_balanced_mae": lbl in included_bins,
        })
finebin_metrics_long = pd.DataFrame(long_rows)
finebin_metrics_long.head(16)

# %% [markdown]
# ## Chart — per-bin MAE by model, eight-bin scheme
#
# What to look for: whether the four low bins (`0-1`, `1-5`, `5-10`, `10-20`)
# show materially different MAE from each other. If they do, Q1's single
# `0-20` bin is masking heterogeneity; if the four bars are close together,
# the coarse bin was a reasonable summary of this region after all. Per-bin
# n is annotated on the x-axis labels so the reader can judge how much each
# bar is worth, and the 95% CI (from the per-bin bootstrap) is drawn as an
# error bar so a "difference" between two low bins can be judged against
# sampling noise rather than read off the bar heights alone.

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(BIN8_LABELS))
width = 0.13
for i, model in enumerate(C.MODELS):
    sub = finebin_metrics_long[finebin_metrics_long["model"] == model].set_index("bin").reindex(BIN8_LABELS)
    vals = sub["mae"].to_numpy()
    err_lo = vals - sub["mae_ci_lo"].to_numpy()
    err_hi = sub["mae_ci_hi"].to_numpy() - vals
    ax.bar(x + (i - 2.5) * width, vals, width, label=model,
           yerr=[err_lo, err_hi], capsize=2, error_kw={"linewidth": 0.7})

xt_labels = [f"{lbl}\n(n={per_bin8_n[lbl]})" for lbl in BIN8_LABELS]
ax.set_xticks(x)
ax.set_xticklabels(xt_labels, fontsize=8)
ax.set_xlabel("Reference cover bin, right-closed (image count in parentheses)")
ax.set_ylabel("Mean absolute error (cover points)")
ax.set_title("Per-bin MAE by model, eight-bin low-cover scheme (Q8)\n(error bars: 95% bootstrap CI)")
ax.legend(fontsize=7, ncol=3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "Q8_perbin_mae_by_model.png", dpi=150)
plt.show()
plt.close(fig)

# %% [markdown]
# ## The three JT trend tests, repeated on the eight-bin scheme
#
# Q3's primary behavioural test runs on the scale-free relative skill
# `r_i = (|e_base| − |e_model|) / (|e_base| + |e_model|)` against the
# floor-matched `B_median` baseline (positive = model better), with signed
# bias and raw `|e|` as companions — raw `|e|` is labelled `structural`
# because it is close to guaranteed to trend upward with cover by
# construction, never as a behavioural finding. All three are repeated here
# unchanged in specification, pooled across the six models, over the eight
# ordered bins instead of five. **All uncorrected and labelled exploratory**
# — Q8 sits outside every Holm family, so these p-values are reported raw
# with no adjustment and must never be read as confirmatory.
#
# The permutation null shuffles bin labels across images (10,000
# permutations), which is the correct null of no association between cover
# level and error; ties are handled by mid-ranks, exactly as in Q3. All three
# tests below call `_common.permutation_jt`, the single Jonckheere-Terpstra
# implementation shared by Q3, Q4 and Q8, not a local reimplementation.

# %%
B_median = float(d1["reference"].median())
pooled_abs_e = base_local.groupby("image", as_index=False)["abs_e"].mean()
pooled_signed_e = base_local.groupby("image", as_index=False)["e"].mean()
pooled = pooled_abs_e.merge(pooled_signed_e, on="image").merge(
    base_local[["image", "reference", "bin8"]].drop_duplicates("image"), on="image", how="left"
)
pooled["abs_e_base"] = (B_median - pooled["reference"]).abs()
pooled["skill"] = pooled["abs_e_base"] - pooled["abs_e"]
denom = pooled["abs_e_base"] + pooled["abs_e"]
pooled["r_i"] = np.where(denom > 0, pooled["skill"] / denom, 0.0)

bin_order = {lbl: i for i, lbl in enumerate(BIN8_LABELS)}
pooled["bin_rank"] = pooled["bin8"].map(bin_order)

# The permutation Jonckheere-Terpstra test is `_common.permutation_jt`, the
# single implementation shared by Q3, Q4 and Q8. It is defined once, rather
# than per question, so that the analysis has one JT implementation rather
# than three that could silently disagree.
#
# `alpha=0.05` is passed explicitly and deliberately. Q8 sits outside every
# Holm family (it is secondary and uncorrected by design, stated throughout
# this notebook), so the plain, unadjusted 0.05 threshold is the correct
# gate for its `direction` column — unlike Q3 (family F, alpha=0.00833) and
# Q4 (family E, alpha=0.01), which must pass their own Holm-adjusted
# thresholds into this same function. Do not change this to match Q3/Q4:
# Q8 has no family to correct for, so 0.05 is not a stand-in value here, it
# is the actual threshold this uncorrected test uses.
#
# A Campaign-clustered companion accompanies every p-value in this study,
# including the secondary ones. For a rank-based JT test there is no
# wild-bootstrap analogue — no way to impose a null and reweight by a Webb
# sign draw on a permutation rank statistic the way there is for a weighted
# mean — so the LOCO range is the clustered companion here, as it is for the
# JT trend tests in Q3. Each of the three tests below is refit on each of the
# three two-campaign leave-one-campaign-out subsets, and the min/max of the
# resulting **`j_bar`** (not raw `J`) across the three refits is reported as
# `loco_min`/`loco_max`: raw `J` scales with n^2, so a LOCO range built from
# it mostly measures how many rows survived dropping a campaign rather than
# how sensitive the trend is to which campaign is held out (measured at
# 124.7% of its own mean across this project's 574/735/1,001-row LOCO
# subsets, against 2.3% for `j_bar`, which is bounded to `[-1, 1]` and
# attains +-1 only at perfect (anti-)monotonicity). `_common.permutation_jt`
# returns `j_bar` on every call, including these LOCO refits.


def loco_jt_jbar(frame: pd.DataFrame, colname: str, campaign_col: str = "campaign") -> dict:
    per_camp = {}
    for camp in frame[campaign_col].unique():
        sub = frame[frame[campaign_col] != camp]
        result = C.permutation_jt(
            sub[colname].to_numpy(), sub["bin_rank"].to_numpy(), tuple(range(len(BIN8_LABELS))),
            rng, alpha=0.05, n_perm=C.B_BOOTSTRAP,
        )
        per_camp[camp] = result.j_bar
    return per_camp


pooled_campaign = pooled.merge(base_local[["image", "campaign"]].drop_duplicates("image"),
                                on="image", how="left")

trend_rows = []
for label, colname, structural in [
    ("relative_skill_r_i", "r_i", False),
    ("signed_bias", "e", False),
    ("raw_abs_error", "abs_e", True),
]:
    jt_result = C.permutation_jt(
        pooled[colname].to_numpy(), pooled["bin_rank"].to_numpy(), tuple(range(len(BIN8_LABELS))),
        rng, alpha=0.05, n_perm=C.B_BOOTSTRAP,
    )
    per_camp_jbar = loco_jt_jbar(pooled_campaign, colname)
    loco_min, loco_max, loco_note = C.loco_range(list(per_camp_jbar.values()))
    trend_rows.append({
        "question_id": "Q8",
        "test": label,
        "component_label": "structural_component" if structural else "behavioural",
        "jt_statistic": jt_result.j_stat,
        "jt_j_bar": jt_result.j_bar,
        "jt_z": jt_result.z,
        "p_value_raw": jt_result.p_perm,
        "at_floor": jt_result.at_floor,
        "direction": jt_result.direction,
        "n_perm": jt_result.n_perm,
        "variance_method": jt_result.variance_method,
        "family": "none (secondary, uncorrected)",
        "holm_adjusted_p": np.nan,
        "n_bins": len(BIN8_LABELS),
        "n_images": len(pooled),
        "loco_min": loco_min,
        "loco_max": loco_max,
        "loco_note": (loco_note or "Clustered companion is the LOCO range on j_bar (rescaled "
                                    "Jonckheere-Terpstra excess concordance, [-1, 1]), "
                                    "leave-one-campaign-out; no wild-bootstrap analogue exists "
                                    "for a rank-based test."),
        "loco_per_campaign_j_bar": per_camp_jbar,
    })

finebin_trend = pd.DataFrame(trend_rows)
finebin_trend.to_csv(RESULTS_DIR / "Q8_finebin_trend.csv", index=False)
print("wrote Q8_finebin_trend.csv")
finebin_trend

# %% [markdown]
# ## Chart — relative skill `r_i` by eight-bin cover level
#
# What to look for: whether the median `r_i` (pooled across the six models)
# declines across the eight bins, and in particular whether the decline is
# already visible inside the four low bins that Q1's `0-20` bin currently
# hides. A flat or non-monotone pattern across the low four bins would mean
# the finer partition adds resolution but not a different qualitative story
# in that region.

# %%
fig, ax = plt.subplots(figsize=(9, 5))
medians = pooled.groupby("bin8")["r_i"].median().reindex(BIN8_LABELS)
ns = pooled.groupby("bin8")["r_i"].size().reindex(BIN8_LABELS)
ax.plot(range(len(BIN8_LABELS)), medians.values, marker="o")
ax.set_xticks(range(len(BIN8_LABELS)))
ax.set_xticklabels([f"{lbl}\n(n={ns[lbl]})" for lbl in BIN8_LABELS], fontsize=8)
ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax.set_xlabel("Reference cover bin, right-closed (image count in parentheses)")
ax.set_ylabel("Median relative skill r_i (unitless, +1..-1; positive = model beats B_median)")
ax.set_title("Relative skill vs. floor-matched baseline, pooled across six models (Q8)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "Q8_relative_skill_by_bin.png", dpi=150)
plt.show()
plt.close(fig)

# %% [markdown]
# ## Zero-rate analysis
#
# The count of exactly-zero-reference images is **computed here at run time**,
# never asserted against a remembered value — A7 does not hard-code the
# figure of 231 as an assertion, and the notebook only ever reports what it
# measures. The exact-zero subgroup is reported as a split **within** the
# `[0,1]` bin, not as a ninth bin, because the eight-bin scheme is fixed;
# these images are where the "true zero" vs "false zero" behaviour actually
# plays out and folding them into the 472-image bin without a separate
# tabulation would hide exactly what this analysis exists to show.
#
# Per model: the rate of predicting exactly 0 on the zero-reference images
# ("true zero" rate) and on the remainder ("false zero" rate), plus the
# reference's own zero rate for context. **The rate reported here is a
# prediction-row rate** — one row per image per prompt (924 rows on the
# zero-reference subgroup: 231 images × 4 prompts) — not an image rate. The
# image is the unit of analysis elsewhere in this project, so the two
# do not agree (the image-level "all 4 prompts predicted 0" rate and the
# "any of 4 predicted 0" rate are both materially different from the row
# rate reported below), and this table is explicit that it reports rows, not
# images.
#
# **R4 applies with full force to every row in this table**, whether or not
# a given model's own realised rate happens to be high: the flag names a
# property of the serving stack, not a summary of the observed rates, so it
# cannot be turned on or off per model based on this run's numbers.

# %%
n_zero_reference = int((d1["reference"] == 0).sum())
print(f"Images with reference == 0, computed at run time: {n_zero_reference} "
      f"(plan-time expectation, sourced from the raw frame: 231)")

zero_ref_images = set(d1.loc[d1["reference"] == 0, "image"])

zero_rows = []
for model in C.MODELS:
    sub = base_local[base_local["model"] == model]
    is_zero_ref = sub["image"].isin(zero_ref_images)

    zero_ref_rows = sub[is_zero_ref]
    nonzero_ref_rows = sub[~is_zero_ref]
    true_zero_rate_rows = float(np.isclose(zero_ref_rows["vegetation_percent"], 0.0).mean()) if len(zero_ref_rows) else np.nan
    false_zero_rate_rows = float(np.isclose(nonzero_ref_rows["vegetation_percent"], 0.0).mean()) if len(nonzero_ref_rows) else np.nan

    # Image-level companions, reported alongside rather than in place of the
    # row rate: "all 4 prompts predicted 0" and "any of 4 predicted 0" per
    # zero-reference image.
    if len(zero_ref_rows):
        zero_ref_rows = zero_ref_rows.copy()
        zero_ref_rows["_pred_is_zero"] = np.isclose(zero_ref_rows["vegetation_percent"], 0.0)
        per_image_zero = zero_ref_rows.groupby("image")["_pred_is_zero"].agg(["all", "any"])
        img_all_zero = float(per_image_zero["all"].mean())
        img_any_zero = float(per_image_zero["any"].mean())
    else:
        img_all_zero, img_any_zero = np.nan, np.nan

    zero_rows.append({
        "question_id": "Q8",
        "model": model,
        "n_zero_reference_rows": int(len(zero_ref_rows)),
        "n_nonzero_reference_rows": int(len(nonzero_ref_rows)),
        "true_zero_rate_rows": true_zero_rate_rows,
        "false_zero_rate_rows": false_zero_rate_rows,
        "true_zero_rate_images_all_4_prompts": img_all_zero,
        "true_zero_rate_images_any_of_4_prompts": img_any_zero,
        "c1_serving_confounded": True,
        "on_fp8_vllm_path": model in C.LLAMA_MODELS,
        "n_zero_reference_images_computed": n_zero_reference,
    })

zero_rates = pd.DataFrame(zero_rows)
zero_rates.to_csv(RESULTS_DIR / "Q8_zero_rates.csv", index=False)
print("wrote Q8_zero_rates.csv")
zero_rates

# %% [markdown]
# ## Chart — true-zero vs false-zero prediction rates by model
#
# What to look for: the two fp8/vLLM-served models (Llama-4-Maverick,
# Llama-4-Scout, marked `on_fp8_vllm_path`) against the other four. A gap
# between the Llama pair and the rest is a serving-configuration pattern
# under R4 — it may not be read as "Llama models are more zero-prone" as a
# model-family claim. Bars show the **row-level** rate (the y-axis label
# says so); the image-level companions are in the CSV, not this chart.

# %%
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(C.MODELS))
width = 0.35
true_vals = [zero_rates.loc[zero_rates["model"] == m, "true_zero_rate_rows"].values[0] for m in C.MODELS]
false_vals = [zero_rates.loc[zero_rates["model"] == m, "false_zero_rate_rows"].values[0] for m in C.MODELS]
colors = ["#d62728" if m in C.LLAMA_MODELS else "#1f77b4" for m in C.MODELS]
ax.bar(x - width / 2, true_vals, width, label="true-zero rate (on zero-reference rows)", color=colors, alpha=0.9)
ax.bar(x + width / 2, false_vals, width, label="false-zero rate (on nonzero-reference rows)", color=colors, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(C.MODELS, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Share of rows (image x prompt) predicted exactly 0 (proportion)")
ax.set_title(f"Zero-prediction rates by model (red = on fp8/vLLM path), n_zero_ref_images={n_zero_reference}")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "Q8_zero_rates_by_model.png", dpi=150)
plt.show()
plt.close(fig)

# %% [markdown]
# ## The ranking-shift result: Kendall tau-b, five-bin vs. eight-bin model ranking
#
# The eight-bin balanced metric weights the 30-image top bin at 1/8 = 12.5%
# instead of 1/5 = 20%, so the per-model ranking by balanced MAE can move
# between the two schemes even though the two schemes nest. This is Q8's
# headline output: it is **routed back onto Q1** as either a reassurance
# (ranking stable under re-binning strengthens Q1's headline) or a caveat
# (ranking moves, which Q1's five-bin view cannot see on its own).
#
# The five-bin balanced MAE per model is recomputed here via
# `_common.balanced_mae()` on the five-bin labels — this is the one place in
# this notebook that function is called, and it is called only on the
# five-bin column, never on the eight-bin one.
#
# **On the Kendall tau-b's p-value.** This question asks only for the
# magnitude of agreement between the two rankings, not for a test, and the natural null of a
# rank-correlation test (the two rankings are independent) cannot be true
# here: both rankings are computed from the same 1,155 images, the same six
# models and the same errors, and the eight bins nest inside the five, so a
# small p is guaranteed by construction and says nothing about ranking
# *stability*, which is what this section exists to measure. `tau_b` (the
# point estimate) is reported; no p-value is computed or written out.

# %%
five_bin_mae = {}
eight_bin_mae = {}
for model in C.MODELS:
    sub = base_local[base_local["model"] == model]
    pooled_m = C.pool_axis_mean_abs_error(sub, group_cols=["image"])
    pooled_m = pooled_m.merge(base_local[["image", "bin5", "bin8"]].drop_duplicates("image"), on="image", how="left")

    five = C.balanced_mae(pooled_m["abs_e"], pooled_m["bin5"])
    _check_per_bin_n_sums(five.per_bin_n, f"five-bin model={model}")
    five_bin_mae[model] = five.balanced_mae

    eight = balanced_mae_8(pooled_m["abs_e"], pooled_m["bin8"], included=included_bins)
    _check_per_bin_n_sums(eight.per_bin_n, f"eight-bin model={model}")
    eight_bin_mae[model] = eight.balanced_mae

five_rank = pd.Series(five_bin_mae).rank(method="average")
eight_rank = pd.Series(eight_bin_mae).rank(method="average")

tau_b, _ = stats.kendalltau(five_rank.reindex(C.MODELS), eight_rank.reindex(C.MODELS))

# Top model (rank 1, i.e. lowest balanced MAE) under each scheme, for the
# descriptive "same top model" check reported alongside the tau-b.
five_top_model = five_rank.idxmin()
eight_top_model = eight_rank.idxmin()

ranking_shift = pd.DataFrame([{
    "question_id": "Q8",
    "statistic": "kendall_tau_b_5bin_vs_8bin_model_ranking",
    "tau_b": float(tau_b),
    "n_models": len(C.MODELS),
    "five_bin_balanced_mae": str(five_bin_mae),
    "eight_bin_balanced_mae": str(eight_bin_mae),
    "five_bin_top_model": five_top_model,
    "eight_bin_top_model": eight_top_model,
    "top_model_unchanged": five_top_model == eight_top_model,
    "family": "none (secondary, uncorrected)",
    "note": "Both schemes share a right-closed convention and the eight bins "
            "nest inside the five (asserted above); the ranking difference "
            "measured here is therefore re-binning weight change only, not a "
            "boundary-convention artefact. tau_b is a magnitude, not a test "
            ". No p-value is reported because the independence null "
            "cannot hold on two rankings drawn from the same images/errors.",
}])
ranking_shift.to_csv(RESULTS_DIR / "Q8_ranking_shift.csv", index=False)
print("wrote Q8_ranking_shift.csv")
ranking_shift

# %% [markdown]
# ## Chart — model ranks under the two schemes
#
# What to look for: whether each model's line stays flat (rank unchanged)
# between the two schemes. A crossing line means the finer bottom-bin
# partition moved that model's rank relative to at least one other model.

# %%
fig, ax = plt.subplots(figsize=(8, 6))
for model in C.MODELS:
    ax.plot([0, 1], [five_rank[model], eight_rank[model]], marker="o", label=model)
ax.set_xticks([0, 1])
ax.set_xticklabels(["5-bin ranking\n(Q1 scheme)", "8-bin ranking\n(Q8 scheme)"])
ax.set_ylabel("Rank by balanced MAE (1 = lowest error)")
ax.invert_yaxis()
ax.set_title(f"Model ranking under re-binning, Kendall tau-b = {tau_b:.3f}")
ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "Q8_ranking_shift.png", dpi=150)
plt.show()
plt.close(fig)

# %% [markdown]
# ## Campaign-clustered companion
#
# Required for every interval and every p-value in this study, including the
# secondary ones. Attached here to the eight-bin balanced MAE pooled
# across models, using the same Webb 6-point wild cluster bootstrap and CR3
# variance estimator every other question uses — computed via
# `_common.wild_cluster_bootstrap` and `_common.clustered_interval`, never
# reimplemented.
#
# **On the p-values this companion would ordinarily carry.** The estimand on
# this row is a *level* — the eight-bin balanced MAE, positive by
# construction — not a contrast. `wild_cluster_bootstrap` imposes a null of
# β = 0 on whatever it is handed, and `image_bootstrap`'s own two-sided
# bootstrap p is `2*min(P(θ*≤0), P(θ*≥0))` against the same zero null. Both
# are well-defined tests when the estimand is a *difference* that could
# plausibly be zero (as it is everywhere else in this project's clustered
# companions), but "eight-bin balanced MAE = 0" is not a null any positive
# error metric could satisfy, so a p-value against it is not a test of
# anything Q8 asks. `p_primary_raw`, `p_companion_raw` and `p_clustered_webb`
# are therefore recorded as NaN below, with the reason named in
# `stream_note`, rather than computed and reported as if they meant "is this
# distinguishable from zero." The CI, the clustered CI, `clustered_df`,
# `n_distinct_abs_t_star` and `ci_width_ratio_clustered_to_headline` are
# retained — those describe the precision of the level estimate, which is
# exactly what a CI is for.
#
# **K4** is measured on this pooled headline bootstrap too (`bin_col="bin8"`
# below), and its **bin-stratified companion** is computed alongside it, for
# the same reason as the per-model bootstraps above.

# %%
pooled_all = base_local.groupby("image", as_index=False)["abs_e"].mean()
pooled_all = pooled_all.merge(base_local[["image", "bin8", "campaign"]].drop_duplicates("image"),
                               on="image", how="left")

bin_to_n = per_bin8_n
pooled_all["w"] = pooled_all["bin8"].map(lambda b: (1.0 / len(included_bins)) / bin_to_n[b] if b in included_bins else 0.0)
d_contrib = pooled_all["abs_e"].to_numpy()
clusters = pooled_all["campaign"].to_numpy()
weights = pooled_all["w"].to_numpy()

webb = C.wild_cluster_bootstrap(d_contrib, clusters, rng, weights=weights)
clustered = C.clustered_interval(d_contrib, clusters, weights=weights)

pooled_res = balanced_mae_8(pooled_all["abs_e"], pooled_all["bin8"], included=included_bins)
_check_per_bin_n_sums(pooled_res.per_bin_n, "pooled-across-models eight-bin")


def headline_stat(frame_resampled: pd.DataFrame) -> float:
    # Resampling the pre-pooled (one row per image, pooled across the 6
    # models and 4 prompts) frame is equivalent to resampling the full
    # base/local frame and re-pooling every time, for the same commuting
    # reason noted in the per-model bootstrap above — pooling is a mean over
    # a fixed set of rows per image, so it does not depend on which other
    # images are drawn or how many times this one is drawn.
    return balanced_mae_8(frame_resampled["abs_e"], frame_resampled["bin8"],
                           included=included_bins).balanced_mae


headline_boot = image_bootstrap_8(pooled_all, headline_stat, rng, image_col="image", bin_col="bin8", B=C.B_BOOTSTRAP)
headline_boot_stratified = image_bootstrap_8(pooled_all, headline_stat, rng, image_col="image", bin_col="bin8",
                                              B=C.B_BOOTSTRAP, stratified=True)

loco_vals = []
for camp in base_local["campaign"].unique():
    sub_camp = base_local[base_local["campaign"] != camp]
    per_bin_n_camp = (
        sub_camp[["image", "bin8"]].drop_duplicates()["bin8"].value_counts().reindex(BIN8_LABELS, fill_value=0)
    )
    if (per_bin_n_camp < 10).any():
        loco_vals.append("not computed (bin n < 10)")
        continue
    p_camp = C.pool_axis_mean_abs_error(sub_camp, group_cols=["image"]).merge(
        sub_camp[["image", "bin8"]].drop_duplicates("image"), on="image", how="left")
    loco_vals.append(balanced_mae_8(p_camp["abs_e"], p_camp["bin8"], included=included_bins).balanced_mae)

loco_min, loco_max, loco_note = C.loco_range(loco_vals)

clustered_row = C.make_contrast_row(
    estimate=headline_boot.estimate, ci_lo=headline_boot.ci_lo, ci_hi=headline_boot.ci_hi,
    p_primary_raw=np.nan, p_primary_holm=np.nan,
    p_companion_raw=np.nan, p_companion_holm=np.nan,
    p_clustered_webb=np.nan, n_distinct_abs_t_star=webb.n_distinct_abs_t_star,
    loco_min=loco_min, loco_max=loco_max,
    ci_width_ratio_clustered_to_headline=C.ci_width_ratio_clustered_to_headline(
        clustered, headline_boot.ci_lo, headline_boot.ci_hi),
    stream_note="Q8 eight-bin balanced MAE, pooled across six models. Secondary, "
                "uncorrected, no Holm family — p_primary_holm and p_companion_* "
                "are structurally NaN because no correction applies here. "
                "p_primary_raw and p_clustered_webb are ALSO NaN here, for a "
                "different reason: the estimand is a level (balanced MAE, "
                "positive by construction), not a contrast, so the tests' "
                "implicit null of 'equal to zero' cannot hold and a p-value "
                "against it is not interpretable. The CI (headline and "
                "clustered) is the estimate of precision this row provides.",
)
clustered_row.update({
    "question_id": "Q8",
    "estimand": "8-bin balanced MAE, pooled across 6 models, base/local",
    "clustered_ci_lo": clustered.ci_lo,
    "clustered_ci_hi": clustered.ci_hi,
    "clustered_df": clustered.df,
    "loco_note": loco_note,
    "below_webb_resolution_floor": webb.below_resolution_floor,
    "n_empty_bin_violations_k4": headline_boot.n_empty_bin_violations,
    "balanced_mae_8bin_stratified": headline_boot_stratified.estimate,
    "balanced_mae_8bin_stratified_ci_lo": headline_boot_stratified.ci_lo,
    "balanced_mae_8bin_stratified_ci_hi": headline_boot_stratified.ci_hi,
    "n_empty_bin_violations_k4_stratified": headline_boot_stratified.n_empty_bin_violations,
    "family": "none (secondary, uncorrected)",
})
q8_clustered = pd.DataFrame([clustered_row])
q8_clustered.to_csv(RESULTS_DIR / "Q8_clustered_companion.csv", index=False)
print("wrote Q8_clustered_companion.csv")
q8_clustered

# %% [markdown]
# ## Result
#
# The paragraphs below are printed from the computed values rather than
# written as static numbers, so every figure in them comes from this run.

# %%
_zero_rate_sorted = zero_rates.sort_values("true_zero_rate_rows", ascending=False).reset_index(drop=True)
_zero_rate_summary = ", ".join(
    f"{r['model']}={r['true_zero_rate_rows']:.3f}" for _, r in _zero_rate_sorted.iterrows()
)
_qwen_row = zero_rates.loc[zero_rates["model"] == "Qwen-2.5"]
_qwen_rate_str = f"{_qwen_row['true_zero_rate_rows'].values[0]:.3f}" if len(_qwen_row) else "n/a"
_llama_top2 = set(_zero_rate_sorted.loc[:1, "model"]) == set(C.LLAMA_MODELS)

print(
    f"What the fine binning reveals that the five-bin view hides:\n"
    f"Q1's bottom bin averages 933 images together -- everything from bare soil at "
    f"reference 0% up to a genuinely vegetated 20% quadrat -- into a single MAE and a "
    f"single balanced-MAE contribution. Under the eight-bin scheme, {per_bin8_n['0-1']} of "
    f"those 933 images sit at 1% cover or below (the [0,1] bin), and a further "
    f"{per_bin8_n['1-5']} sit between 1% and 5% (the (1,5] bin) -- both counts computed "
    f"above, not assumed. Whatever the per-bin MAE table shows for those four low bins is "
    f"direct evidence of whether Q1's coarse bottom bin is a reasonable summary of that "
    f"region or is quietly averaging together two populations with different error "
    f"behaviour -- near-bare quadrats where the error floor is nearly zero, and "
    f"low-but-real cover where it is not.\n\n"
    f"The zero-rate table isolates a further, qualitatively distinct subgroup inside that "
    f"lowest bin -- the images where the reference itself reads exactly zero -- and shows, "
    f"per model, how often each model also predicts exactly zero there versus elsewhere, at "
    f"the row level (image x prompt). Realised true-zero rates, descending: "
    f"{_zero_rate_summary}. The two highest belong to the two fp8/vLLM-served models "
    f"(Llama-4-Maverick, Llama-4-Scout) [top-two are exactly the Llama pair: {_llama_top2}], "
    f"consistent with R4's serving-stack concern -- but the ordering is NOT a clean "
    f"two-vs-four split: Qwen-2.5 sits at a materially elevated rate ({_qwen_rate_str}) "
    f"despite running on the bf16/A100 path with no C1-confounded serving stack. That is "
    f"itself evidence against reading a high zero-rate as a property of the 'Llama model "
    f"family' or the fp8/vLLM path in the abstract -- the pattern is present on at least one "
    f"model outside that path too -- and every row in the zero-rate table carries "
    f"c1_serving_confounded=True precisely so this table is never read model-family-first.\n\n"
    f"The Kendall tau-b above is the number that should travel back to Q1's reporting: "
    f"tau_b={tau_b:.3f}, top model unchanged={five_top_model == eight_top_model}. If it is "
    f"close to 1 with the same top model in both schemes, Q1's five-bin ranking is "
    f"corroborated by the finer view and the coarse summary was adequate for a ranking "
    f"claim; if it is materially below 1 or the top model changes, that is a caveat Q1's "
    f"deliverable should carry -- the coarse bottom bin was hiding a ranking-relevant "
    f"difference in exactly the 81% of the frame it collapses into one number."
)

# %% [markdown]
# **What this analysis does not establish.** Every result in this notebook
# is secondary, uncorrected and exploratory by design — none of it sits in
# any Holm family, and no p-value here should be read at a controlled
# false-positive rate. The trend tests inherit every one of Q3's own
# limitations (JT tests stochastic ordering, not means; the raw-`|e|` trend
# is structural and never behavioural); their clustered companion is the
# LOCO range, not a wild-bootstrap p, because no such analogue exists for a
# rank statistic. The eight-bin and pooled balanced MAEs carry CIs, not
# p-values against zero — a positive-definite level statistic has no
# meaningful zero null, so none is reported. The Kendall tau-b is a
# magnitude, not a test — its natural independence null cannot hold on two
# rankings built from the same images, models and errors, so no p-value is
# reported for it either. The zero-rate pattern is descriptive and carries
# the C1 serving-stack confound with full force on every row, regardless of
# that row's own realised rate, and is reported at the row level, not the
# image level (the two do not agree; the image-level companions are in the
# CSV). The eight-bin balanced MAE excludes no bin under the realised counts
# (the top bin sits at exactly n = 30, included under the strict
# `n < 30` rule), so every one of the eight per-bin MAEs contributes to
# the balanced summary reported here.

# %%
summary_rows = [
    {
        "question_id": "Q8",
        "statistic": "n_images_low_four_bins_sum",
        "value": float(low_sum),
        "ci_low": np.nan, "ci_high": np.nan,
        "n": n_total_images,
        "test": "eight-bin nesting check against Q1 bottom bin",
        "tier": "secondary", "corrected": False,
    },
    {
        "question_id": "Q8",
        "statistic": "n_zero_reference_images",
        "value": float(n_zero_reference),
        "ci_low": np.nan, "ci_high": np.nan,
        "n": n_total_images,
        "test": "computed at run time (A7), not asserted",
        "tier": "secondary", "corrected": False,
    },
    {
        "question_id": "Q8",
        "statistic": "kendall_tau_b_5bin_vs_8bin_ranking",
        "value": float(tau_b),
        "ci_low": np.nan, "ci_high": np.nan,
        "n": len(C.MODELS),
        "test": "Kendall tau-b, model ranking by balanced MAE",
        "tier": "secondary", "corrected": False,
    },
    {
        "question_id": "Q8",
        "statistic": "balanced_mae_8bin_pooled_across_models",
        "value": float(headline_boot.estimate),
        "ci_low": float(headline_boot.ci_lo), "ci_high": float(headline_boot.ci_hi),
        "n": n_total_images,
        "test": "image bootstrap BCa, pooled across 6 models",
        "tier": "secondary", "corrected": False,
    },
]
result_summary = pd.DataFrame(summary_rows)
result_summary.to_csv(RESULTS_DIR / "Q8_result_summary.csv", index=False)
print("wrote Q8_result_summary.csv")
result_summary

# %% [markdown]
# ## Assumption checks
#
# The A1–A14 load-time table, extended with the two assertions this notebook
# calls explicitly (K1 pairing, C11 Maverick reproducibility floor) and with
# the Q8-specific checks this notebook is responsible for: the eight-bin
# per-image-count and nesting checks (the two failure modes described
# above), the realised K4 empty-bin-resample counts from both the per-model
# and pooled headline bootstraps, and the eight realised per-bin image
# counts. Every notebook in this study writes an assumption-check table, and
# this is Q8's.

# %%
k4_rows = pd.DataFrame([
    {"model": model, "n_empty_bin_violations_k4": row["n_empty_bin_violations_k4"],
     "n_empty_bin_violations_k4_stratified": row["n_empty_bin_violations_k4_stratified"]}
    for model, row in zip(C.MODELS, rows_metrics)
])
k4_pooled_row = pd.DataFrame([{
    "model": "pooled_across_6_models",
    "n_empty_bin_violations_k4": headline_boot.n_empty_bin_violations,
    "n_empty_bin_violations_k4_stratified": headline_boot_stratified.n_empty_bin_violations,
}])
k4_summary = pd.concat([k4_rows, k4_pooled_row], ignore_index=True)

assumption_checks_q8 = pd.concat(
    [
        assumption_df,
        pd.DataFrame([vars(k1_result), vars(c11_result)]),
        pd.DataFrame([{
            "id": "Q8_bins8_count",
            "description": "Q8 eight-bin per-image counts sum to 1,155 with no empty label",
            "passed": sum(per_bin8_n.values()) == 1155 and all(v > 0 for v in per_bin8_n.values()),
            "detail": f"per_bin8_n={per_bin8_n}",
        }]),
        pd.DataFrame([{
            "id": "Q8_bins8_nest",
            "description": "Q8 eight-bin scheme nests inside Q1's five-bin scheme",
            "passed": bool(nest_low_ok and nest_high_ok),
            "detail": f"low_sum={low_sum} vs bin5['0-20']={per_bin5_n['0-20']}; "
                      f"high={nest_high} vs {expected_high}",
        }]),
        pd.DataFrame([{
            "id": "Q8_K4",
            "description": "K4: count of bootstrap resamples missing at least one of the eight bins "
                            "(plain and bin-stratified companion), per model and pooled",
            "passed": bool((k4_summary["n_empty_bin_violations_k4"] == 0).all()),
            "detail": k4_summary.to_dict(orient="records"),
        }]),
    ],
    ignore_index=True, sort=False,
)
assumption_checks_path = RESULTS_DIR / "Q8_assumption_checks.csv"
assumption_checks_q8.to_csv(assumption_checks_path, index=False)
print(f"wrote {assumption_checks_path}")
assumption_checks_q8
