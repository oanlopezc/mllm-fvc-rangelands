# %% [markdown]
# # Q5 — Does self-reported confidence usefully flag bad estimates?
#
# Every model in this study was asked to report a confidence value alongside
# its vegetation-cover estimate. If that number is worth anything operationally,
# it should be higher on images the model gets badly wrong less often, i.e. low
# confidence should rank images by error better than chance. This notebook asks
# that question directly, model by model, using each model's self-reported
# confidence (`D5.v5`) as a classifier for a large-error label and reading its
# discrimination as the area under the ROC curve (AUC).
#
# The answer is not simply "yes" or "no" here — it can run in either
# direction, and it does. An AUC above 0.5 means confidence is informative in
# the direction expected (low confidence flags bad estimates); an AUC below
# 0.5 means confidence is **anti-informative** — systematically *higher* on
# the images the model gets worst, which is an operationally important and
# actively misleading signal, not merely a weaker version of the positive
# result. Both directions, plus "no detectable signal", are reported below by
# name for every model.
#
# Two things make this a harder question than "compute AUC and look at it":
#
# 1. **Confidence was returned on inconsistent scales and partly unrecoverable.**
#    Some values could not be resolved to a common 0–1 scale at all and were
#    deliberately left missing (`confidence_parse_method == 'ambiguous_unresolved'`).
#    This is missing *not at random* — a model that declines to report an
#    interpretable confidence exactly when it is unsure about its own estimate
#    would look artificially well-calibrated if those rows were simply dropped.
#    The complete-case AUC is reported, but it is bracketed by explicit MNAR
#    bounds so a reader can see how much the missing rows could move the answer.
# 2. **Confidence is coarsely quantised** — a model reports 5–14 distinct
#    values across the whole frame, but the unit this notebook actually
#    computes on is the model x prompt **cell**, and within a cell there are
#    only **3–10** distinct usable values (verified below, not assumed).
#    A conventional ROC sweep and a tie-aware rank statistic are not the same
#    computation once ties are this common, so AUC here is computed as the
#    tie-aware Mann–Whitney U statistic, not by sweeping thresholds.
#
# **What this analysis does not claim.** Confidence is treated throughout as an
# *ordinal* signal only. No calibration metric that assumes confidence is a
# probability (ECE, Brier score, log-loss, any mean or Pearson correlation of
# confidence) is computed anywhere in this notebook — nothing in the data
# establishes that these self-reported numbers are probabilities, or that they
# mean the same thing from one model to the next.

# %% [markdown]
# ## Setup
#
# The seed is fixed so every bootstrap and resampling step in this notebook
# reproduces identically on every run. `_common.py` is the shared module used by
# every question notebook in this project; nothing here re-implements the
# bootstrap, the bin assignment, the Wilcoxon companion or the Holm helper.

# %%
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; this notebook only saves figures
import matplotlib.pyplot as plt

# `_common.py` sits in the notebooks folder, alongside this file. A rendered
# notebook runs with its working directory one level below that folder, so what
# must be added to `sys.path` is this file's own location rather than cwd.
# `__file__` is unavailable in a rendered notebook, so this falls back to the
# ANALYSIS_PROJECT_ROOT environment variable, then to the parent of cwd (the
# layout when the rendered notebook is opened by hand in Jupyter).
import os as _os

_notebook_dir_candidates = [
    Path(_os.environ["ANALYSIS_PROJECT_ROOT"]) / "05_deliverables/Repository/GitHub/notebooks"
    if "ANALYSIS_PROJECT_ROOT" in _os.environ else None,
    Path.cwd().parent,   # rendered notebook: one level below the notebooks folder
    Path.cwd(),          # already in the notebooks folder
]
for _cand in _notebook_dir_candidates:
    if _cand is not None and (_cand / "_common.py").is_file():
        sys.path.insert(0, str(_cand))
        break
else:
    raise RuntimeError("could not locate _common.py from any candidate notebook directory")

import pathlib as _pl  # `_common.py` is imported from this notebook's
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))  # own directory
import _common as c

SEED = 20260912  # the study's base seed (20260907) plus the question number.
# Written as a plain integer rather than an expression, so the seed that
# governs this run can be read straight off the source.
rng = np.random.default_rng(SEED)

# The confidence-threshold filtering sweep and the AUC top-set each draw from
# their own generator, seeded off `SEED` with a fixed, distinct offset, rather
# than from the module-level `rng` above. `rng` is shared by the family-G,
# Spearman, missing-not-at-random, label-robustness and clustered-companion
# computations, every one of which is a reported result. A block drawing from
# `rng` would shift every draw made after it in program order, so reordering
# the cells would quietly move numbers that are already reported. One
# generator per block keeps each block's random stream independent of where
# any other code sits in this file, in either direction.
rng_filtering = np.random.default_rng(SEED + 1001)  # confidence-filtering sweep + per-bin change table
rng_topset = np.random.default_rng(SEED + 1002)      # AUC top-set (MCB / MCS)

ROOT = c.project_root()
RESULTS_DIR = ROOT / "05_deliverables/Repository/GitHub/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FAMILY = "G"
FAMILY_SIZE = c.FAMILY_SIZES[FAMILY]
ALPHA = 0.05
OUTLIER_THRESHOLD = 20.0  # |e| > 20 cover points, fixed before any test was run

print(f"seed={SEED}, family={FAMILY}, family_size={FAMILY_SIZE}")

# %% [markdown]
# ## Data
#
# The main-path frame is `variant == 'base'`, local stack: 1,155 images x 6
# models x 4 prompts = 27,720 rows. `_common.run_all_assertions()` runs the
# load-time assertions A1–A14 that gate every notebook in this project and
# returns the assembled frames. Two additional gates are not reachable from
# that one-call path and must be invoked explicitly here:
#
# - **K1** (`assert_k1_pairing_complete`) — every image must have exactly 24
#   base/local rows (6 models x 4 prompts) on the combined frame this notebook
#   actually uses. A2–A4 check related but different predicates and would not
#   by themselves catch a frame with some images at 23 rows and others at 25.
# - **C11** (`assert_c11_n_reproduced`) — the Llama-4-Maverick reproducibility
#   floor's own precondition (92/100 images reproduced exactly). Q5 does not
#   use the floor directly (it bounds an error-magnitude comparison, not a
#   ranking statistic), but it is the module's convention that any notebook
#   whose family touches Maverick confirms C11 before proceeding.

# %%
assumption_df, frames = c.run_all_assertions(include_d12=False)
base_local = frames["base_local"]
d5_full = frames["d5"]
d8 = frames["d8"]

k1_result = c.assert_k1_pairing_complete(d5_full, base_local)
print(k1_result)

maverick_floor = c.compute_maverick_floor(d8, rng)
c11_result = c.assert_c11_n_reproduced(maverick_floor.n_reproduced)
print(c11_result)

print(f"base_local: {len(base_local)} rows, {base_local['image'].nunique()} images, "
      f"{base_local['model'].nunique()} models, {base_local['prompt'].nunique()} prompts")

# %% [markdown]
# ## The outlier label, and why the threshold is 20
#
# `outlier = |e| > 20` cover points, a threshold fixed on a fixed cover-point
# scale because prediction error is strongly non-normal for
# every model (D'Agostino–Pearson p < 1e-130, skew −0.2 to −1.5, kurtosis 5–13
# across the six models). A standard-deviation-based rule (e.g. "outlier if
# |e| exceeds 2 SD") was considered and rejected on exactly that evidence: with
# skew and kurtosis varying this much across models, a fixed SD multiple marks
# a different absolute error threshold for each one, so "outlier" would not be
# a comparable definition from model to model. A fixed cover-point threshold is
# comparable by construction, and 20 was chosen because it sits close to the
# 90th percentile of `|e|` for five of the six models — a magnitude a field
# ecologist would recognise as a bad enough miss to act on, and one that yields
# a comparable worst-decile across models rather than a different quantile for
# each. The realised outlier rate and the p90 of `|e|` are reported per model
# below so the "close to p90" claim is checked, not assumed.

# %%
base_local["outlier"] = (base_local["abs_e"] > OUTLIER_THRESHOLD).astype(int)

label_rate_table = (
    base_local.groupby("model")
    .agg(outlier_rate=("outlier", "mean"), n=("outlier", "size"),
         p90_abs_e=("abs_e", lambda s: float(s.quantile(0.90))))
    .reindex(list(c.MODELS))
)
print(label_rate_table)

# %% [markdown]
# ## Confidence usability, computed on the base/local frame only (A13)
#
# Counted over a model's whole response file, unusable confidence appears on
# 286 / 183 / 11 / 3 rows (Gemma-3-12B / Scout / Mistral / Maverick) and on no
# rows at all for Gemma-3-27B and Qwen. Those are **counts over all 13,860 rows
# per model, spanning all three variants** (base, masked_gray, rectified). Q5
# runs on `variant == 'base'` only, where each file's slice is a third of that
# total, so the count that actually governs Q5's missingness bounds has to be
# computed on the base slice. Assertion A13 does exactly that, and its result
# is used directly rather than any whole-file figure.

# %%
a13_result = c.assert_a13(base_local, d5_full)
print(a13_result)

base_local["confidence_unusable"] = (
    base_local["confidence_parse_method"].eq("ambiguous_unresolved") | base_local["confidence"].isna()
)
unusable_by_model = (
    base_local.groupby("model")["confidence_unusable"].agg(n_unusable="sum", n_total="size")
    .reindex(list(c.MODELS))
)
unusable_by_model["share_unusable"] = unusable_by_model["n_unusable"] / unusable_by_model["n_total"]
print(unusable_by_model)

# %% [markdown]
# These are the realised, base/local-only counts. They are roughly a third of
# the whole-file (all-variant) figures, exactly as A13 anticipates, and they
# are what the MNAR bounds below are built from.

# %% [markdown]
# ## Degeneracy check (K8)
#
# Confidence is coarsely quantised. K8 flags any model x prompt **cell** with
# two or fewer effective confidence levels, or more than 90% of its mass on a
# single level; a flagged cell's AUC is reported but not read as a ranking
# claim. The check is run at the unit the AUC is actually computed on — model
# x prompt, 24 cells — because pooling prompts first is not conservative here:
# pooling adds distinct levels and dilutes any single level's mass, so a cell
# that is genuinely near-degenerate at the prompt level can look fine once
# three other prompts are folded in. A per-model pooled view is also kept
# alongside for reference, but the flag that governs interpretation below is
# the per-cell one.

# %%
k8_combo_rows = []
for (model, prompt), g in base_local.groupby(["model", "prompt"]):
    usable = g.loc[~g["confidence_unusable"], "confidence"]
    res = c.check_k8_confidence_degeneracy(usable)
    res["model"] = model
    res["prompt"] = prompt
    k8_combo_rows.append(res)
k8_combo_df = pd.DataFrame(k8_combo_rows)
print(k8_combo_df)

k8_rows = []
for model, g in base_local.groupby("model"):
    usable = g.loc[~g["confidence_unusable"], "confidence"]
    res = c.check_k8_confidence_degeneracy(usable)
    res["model"] = model
    k8_rows.append(res)
k8_df = pd.DataFrame(k8_rows).set_index("model").reindex(list(c.MODELS))
print(k8_df)

# %% [markdown]
# One open item, noted but not resolved here: on the **API** stack (`D6.v4`),
# Llama-4-Maverick gives a self-reported confidence of exactly 1.0 on 42% of
# rows. Whether that is genuine over-confidence is not established here.
# Q5's main path is the **local** stack only, so this does not
# block anything below; API confidence is never read, computed or interpreted
# anywhere in this notebook, and any Maverick AUC below is the local-stack
# number only.

# %% [markdown]
# ## Estimator: tie-aware Mann–Whitney U as AUC
#
# **Why not a threshold sweep.** With only 3–10 distinct confidence levels per
# model x prompt cell, an ROC curve built by sweeping thresholds is a coarse
# step function, and a naive implementation that does not account for ties at
# each level
# reports a different (biased) number than the standard tie-aware statistic.
# The Mann–Whitney U statistic, computed with mid-ranks for tied values, is
# exactly equivalent to the area under the ROC curve when ties are broken by
# splitting credit evenly — this is the standard identity `AUC = U / (n_pos *
# n_neg)`, and it is the version reported here throughout, including in the
# 24-cell model x prompt sub-analysis. Full operating-point tables (the
# distinct confidence levels actually observed, and the TPR/FPR at each) are
# written out alongside so a reader can see how few points the curve has.

# %%
# **Orientation convention (read this before reading any AUC below).** The
# question this notebook asks is whether *low* confidence flags a bad
# estimate, so the classifier direction that matters is "predict outlier for
# low confidence" — exactly the direction used in the operating-point table
# below ("predict outlier for confidence <= level"). AUC is therefore defined
# as `P(confidence on a random non-outlier > confidence on a random outlier)`
# (ties split 0.5), equivalently `1 - P(conf_outlier > conf_non-outlier)`, so
# that **AUC > 0.5 means confidence is usefully LOW on the images the model
# gets badly wrong**, and AUC < 0.5 means confidence runs the *wrong* way —
# it is systematically higher on the images the model gets worst, i.e.
# anti-informative. This is the opposite of the naive "positive class first"
# convention (`outlier==1` as the first Mann-Whitney sample), which would
# instead score high confidence on bad images as AUC > 0.5 and invert every
# reading below. `1 - U/(n_pos*n_neg)` implements the stated convention while
# still using the identical tie-aware `mannwhitneyu` computation.
def tie_aware_auc(confidence: np.ndarray, outlier: np.ndarray) -> dict:
    """AUC via the tie-aware Mann-Whitney U statistic (mid-rank ties).

    Orientation: `outlier == 1` (a badly-wrong image, |e| > 20) is scored as
    the class confidence should rank LOW. Returns
    `auc = P(conf_non-outlier > conf_outlier) + 0.5*P(tie)`, so AUC > 0.5
    means low confidence flags outliers (informative, in the direction the
    plan states); AUC < 0.5 means confidence runs the wrong way (higher on
    the images the model gets worst — anti-informative). Also returns
    n_pos, n_neg and the count of tied confidence values contributing to the
    ranking, so tie prevalence is visible alongside every AUC this function
    produces.
    """
    conf = np.asarray(confidence, dtype=float)
    lab = np.asarray(outlier, dtype=int)
    pos = conf[lab == 1]
    neg = conf[lab == 0]
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return {"auc": np.nan, "n_pos": n_pos, "n_neg": n_neg, "n_tied_values": 0, "u_stat": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # `pos` (outlier) passed first to mannwhitneyu, giving
        # U = #{(i,j): pos_i > neg_j} + 0.5*#ties -- then the sign is
        # flipped (`1 - U/(n_pos*n_neg)`) so the reported AUC scores
        # "confidence lower on outliers" as the positive direction, per the
        # convention stated above.
        res = stats.mannwhitneyu(pos, neg, alternative="two-sided", method="asymptotic")
    u_pos_gt_neg = float(res.statistic)
    auc = 1.0 - (u_pos_gt_neg / (n_pos * n_neg))
    ranks = stats.rankdata(conf)
    _, counts = np.unique(ranks, return_counts=True)
    n_tied = int(np.sum(counts[counts > 1]))
    return {"auc": auc, "n_pos": n_pos, "n_neg": n_neg, "n_tied_values": n_tied, "u_stat": u_pos_gt_neg}


def complete_case_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Complete-case rows: usable confidence only, for the primary AUC."""
    return frame.loc[~frame["confidence_unusable"]].copy()

# %% [markdown]
# ## Family G — 6 tests, AUC = 0.5 per model
#
# **Unit and pooling.** AUC is computed per model x prompt (24 cells, each
# n = 1,155 independent images before the complete-case filter — one row per
# image, no within-cell dependence). The per-model headline is the **mean of
# that model's 4 prompt-level AUCs**. Its confidence interval comes from the
# image-level paired bootstrap: each resample draws image IDs with
# replacement and carries every one of that image's rows — all 4 prompts, one
# model at a time here — together, so the resampling never breaks the pairing
# a repeated-measures design imposes.
#
# **Family G's test statistic.** H0: AUC = 0.5, one test per model. The
# verdict is the two-sided image-bootstrap p-value on `(mean AUC - 0.5)`,
# `2*min(P(theta* <= 0), P(theta* >= 0))`, floored at `1/(B+1)`, computed from
# the same 10,000 resamples that produce the CI — Holm-adjusted within family
# G (size 6, first threshold 0.05/6 = 0.00833). The 95% BCa interval on
# `(mean AUC - 0.5)` is reported beside it as the unadjusted companion. The 24
# model x prompt AUCs below are the pre-declared sub-analysis and are reported
# with CIs and no p-values — they inform the headline without inflating the
# family. The 15 between-model AUC comparisons are not planned and are not
# computed; overlapping CIs in the per-model table are not a test.

# %%
def model_mean_auc_statistic(frame_cc: pd.DataFrame) -> float:
    """Mean of the 4 within-model prompt-level AUCs, on a complete-case frame
    for one model. Used both for the point estimate and inside the bootstrap.
    """
    per_prompt = []
    for prompt, g in frame_cc.groupby("prompt"):
        per_prompt.append(tie_aware_auc(g["confidence"].values, g["outlier"].values)["auc"])
    return float(np.nanmean(per_prompt))


auc_by_combo_rows = []
auc_by_model_rows = []
family_g_pvals = []
family_g_models = []

for model in c.MODELS:
    model_frame = base_local[base_local["model"] == model]
    model_cc = complete_case_frame(model_frame)

    # 24-cell sub-analysis: one row per (model, prompt).
    for prompt, g in model_cc.groupby("prompt"):
        res = tie_aware_auc(g["confidence"].values, g["outlier"].values)
        # image-bootstrap CI for this single model x prompt cell
        cell_full = model_frame[model_frame["prompt"] == prompt]  # for image set; refiltered inside stat

        def combo_stat(fr, _prompt=prompt):
            fr_cc = complete_case_frame(fr[fr["prompt"] == _prompt])
            return tie_aware_auc(fr_cc["confidence"].values, fr_cc["outlier"].values)["auc"]

        boot = c.image_bootstrap(model_frame, combo_stat, rng, bin_col="bin", B=c.B_BOOTSTRAP)
        unstable = res["n_pos"] < 20
        k8_cell = k8_combo_df[(k8_combo_df["model"] == model) & (k8_combo_df["prompt"] == prompt)].iloc[0]
        auc_by_combo_rows.append({
            "model": model, "prompt": prompt,
            "auc": res["auc"], "ci_lo": boot.ci_lo, "ci_hi": boot.ci_hi, "ci_method": boot.ci_method,
            "n_pos": res["n_pos"], "n_neg": res["n_neg"], "n_tied_values": res["n_tied_values"],
            "n_used": res["n_pos"] + res["n_neg"],
            "n_excluded_unusable_confidence": int(model_frame[model_frame["prompt"] == prompt]["confidence_unusable"].sum()),
            "unstable_low_positives": unstable,
            "k8_degenerate": bool(k8_cell["degenerate"]),
            "k8_n_levels": int(k8_cell["n_levels"]),
            "k8_max_single_level_mass": float(k8_cell["max_single_level_mass"]),
            "k8_degenerate_pooled_by_model": bool(k8_df.loc[model, "degenerate"]),
        })

    # Per-model headline: mean of the 4 prompt AUCs, with its own bootstrap CI.
    point_estimate = model_mean_auc_statistic(model_cc)

    # `stat_fn` returns (mean AUC - 0.5), not the raw AUC. `_common.image_bootstrap`
    # always tests its statistic against a null of exactly zero
    # (`p = 2*min(P(theta*<=0), P(theta*>=0))`); family G's null is AUC = 0.5, so
    # the statistic handed to the bootstrap must already be centred on that null,
    # or every replicate of a bounded, strictly-positive quantity like AUC lands
    # on the same side of zero and the test cannot fail to reject. Centring here
    # also means `boot.estimate`/`boot.ci_lo`/`boot.ci_hi` come back already on
    # the delta scale the output columns are named for (`ci_lo_delta` etc.), so
    # no further shift is applied anywhere downstream.
    def stat_fn(fr, _model=model):
        fr_model = fr[fr["model"] == _model] if "model" in fr.columns else fr
        return model_mean_auc_statistic(complete_case_frame(fr_model)) - 0.5

    boot = c.image_bootstrap(model_frame, stat_fn, rng, bin_col="bin", B=c.B_BOOTSTRAP)
    k4 = c.check_k4_bootstrap_bin_coverage(boot.n_empty_bin_violations, c.B_BOOTSTRAP)

    n_excluded_model = int(model_frame["confidence_unusable"].sum())
    n_total_model = len(model_frame)
    n_pos_model = int((model_cc["outlier"] == 1).sum())
    n_neg_model = int((model_cc["outlier"] == 0).sum())

    family_g_pvals.append(boot.p_two_sided)
    family_g_models.append(model)

    p_floor = 1.0 / (c.B_BOOTSTRAP + 1)
    auc_by_model_rows.append({
        "model": model,
        "mean_auc": point_estimate,
        "delta_vs_0.5": point_estimate - 0.5,
        "ci_lo_delta": boot.ci_lo, "ci_hi_delta": boot.ci_hi, "ci_method": boot.ci_method,
        "p_bootstrap_raw": boot.p_two_sided,
        "p_at_floor": bool(boot.p_two_sided <= p_floor),
        "n_empty_bin_violations": boot.n_empty_bin_violations,
        "switch_to_stratified": k4["switch_to_stratified"],
        "n_excluded_unusable_confidence": n_excluded_model,
        "n_total_rows": n_total_model,
        "share_excluded": n_excluded_model / n_total_model,
        "n_pos": n_pos_model, "n_neg": n_neg_model,
        "k8_degenerate": bool(
            k8_combo_df.loc[k8_combo_df["model"] == model, "degenerate"].any()
        ),  # True if ANY of this model's 4 prompt cells is K8-degenerate
        "k8_degenerate_pooled_by_model": bool(k8_df.loc[model, "degenerate"]),
    })

auc_by_combo_df = pd.DataFrame(auc_by_combo_rows)
auc_by_model_df = pd.DataFrame(auc_by_model_rows).set_index("model").reindex(list(c.MODELS)).reset_index()

holm_p = c.holm_adjust(auc_by_model_df["p_bootstrap_raw"].values, family=FAMILY, family_size=FAMILY_SIZE)
auc_by_model_df["p_bootstrap_holm"] = holm_p
auc_by_model_df["holm_first_threshold"] = c.holm_first_threshold(ALPHA, FAMILY_SIZE)
auc_by_model_df["family"] = FAMILY
auc_by_model_df["family_size"] = FAMILY_SIZE

print(auc_by_model_df[["model", "mean_auc", "delta_vs_0.5", "ci_lo_delta", "ci_hi_delta",
                        "p_bootstrap_raw", "p_bootstrap_holm"]])

# %% [markdown]
# ## Cut-free companion: tie-corrected Spearman(confidence, |e|)
#
# The 20-point dichotomy could throw away a monotone relationship that never
# crosses the outlier threshold cleanly. Tie-corrected Spearman rho between
# confidence and `|e|` is computed per model x prompt as a check against that:
# it needs only that `confidence` be ordinal, which it is, and it carries no
# p-value into family G — it exists so a null AUC can be read against whether
# a monotone (non-dichotomised) association exists at all.

# %%
spearman_rows = []
for model in c.MODELS:
    model_frame = base_local[base_local["model"] == model]
    for prompt, g in model_frame.groupby("prompt"):
        g_cc = complete_case_frame(g)
        rho, p_unadjusted = c.spearman_tie_corrected(g_cc["confidence"].values, g_cc["abs_e"].values)

        def rho_stat(fr, _prompt=prompt):
            fr_cc = complete_case_frame(fr[fr["prompt"] == _prompt])
            r, _ = c.spearman_tie_corrected(fr_cc["confidence"].values, fr_cc["abs_e"].values)
            return r

        boot = c.image_bootstrap(model_frame, rho_stat, rng, bin_col="bin", B=c.B_BOOTSTRAP)
        spearman_rows.append({
            "model": model, "prompt": prompt, "spearman_rho": rho,
            "ci_lo": boot.ci_lo, "ci_hi": boot.ci_hi, "ci_method": boot.ci_method,
            "n_used": len(g_cc),
        })
spearman_df = pd.DataFrame(spearman_rows)
print(spearman_df.to_string(index=False))
spearman_model_means = spearman_df.groupby("model")["spearman_rho"].mean().reindex(list(c.MODELS))
print()
print("Per-model mean Spearman rho (confidence, |e|), averaged over 4 prompts:")
print(spearman_model_means.to_string())

# %% [markdown]
# ## Full operating-point tables
#
# For every model x prompt cell, every distinct confidence level actually
# observed, with the true-positive rate and false-positive rate at that level
# treated as a decision threshold (predict outlier for confidence <= that
# level, since low confidence is the direction that should flag error), and
# the n of images at that exact level. This is the evidence for "the ROC is a
# coarse step function" — a reader can count the points on it directly.

# %%
op_point_rows = []
for model in c.MODELS:
    model_frame = base_local[base_local["model"] == model]
    for prompt, g in model_frame.groupby("prompt"):
        g_cc = complete_case_frame(g)
        levels = np.sort(g_cc["confidence"].unique())
        n_pos_total = int((g_cc["outlier"] == 1).sum())
        n_neg_total = int((g_cc["outlier"] == 0).sum())
        for lev in levels:
            below_or_eq = g_cc["confidence"] <= lev
            tp = int(((g_cc["outlier"] == 1) & below_or_eq).sum())
            fp = int(((g_cc["outlier"] == 0) & below_or_eq).sum())
            n_at_level = int((g_cc["confidence"] == lev).sum())
            tpr = tp / n_pos_total if n_pos_total else np.nan
            fpr = fp / n_neg_total if n_neg_total else np.nan
            op_point_rows.append({
                "model": model, "prompt": prompt, "confidence_level": lev,
                "n_at_level": n_at_level, "tpr": tpr, "fpr": fpr,
                "n_pos_total": n_pos_total, "n_neg_total": n_neg_total,
            })
op_points_df = pd.DataFrame(op_point_rows)
print(f"{len(op_points_df)} operating points across 24 model x prompt cells "
      f"({op_points_df.groupby(['model','prompt']).size().mean():.1f} points/cell on average)")

# %% [markdown]
# ## MNAR bounds — a required companion, not optional
#
# Confidence is missing deliberately, not at random: a model that declines to
# report an interpretable confidence exactly when it is confused about its own
# estimate would look artificially well-calibrated if those rows were simply
# dropped from the complete-case AUC above. Three pre-declared analyses:
#
# 1. **Complete-case AUC** (already computed above, primary) — the excluded n
#    and its outlier rate are stated alongside it.
# 2. **MNAR bounds — genuine worst case, not a common fill.** An unresolved
#    row's true confidence is unknown, and it could be either an outlier or a
#    non-outlier — a single shared fill value applied to both labels lets an
#    unusable outlier and an unusable non-outlier partially cancel each other's
#    pull on the AUC, which understates how far the missing rows could move
#    the answer. The genuine worst case in each direction instead sends every
#    unresolved row **the way its own label would push hardest**: for the
#    upper bound, unresolved outliers are assigned a confidence *below* every
#    observed value (maximally informative-looking) and unresolved
#    non-outliers a confidence *above* every observed value; the lower bound
#    reverses both assignments. The two extremes are placed strictly outside
#    the observed range (`observed_min - 1`, `observed_max + 1`) rather than
#    tied with it, so a filled row never shares a rank with a real observation
#    it is supposed to dominate. If the bounds straddle 0.5, that model's AUC
#    is reported as **not determined by this data** rather than as a point
#    estimate.
# 3. **Is unusability itself a signal?** Fisher's exact test on the 2x2 table
#    of {unusable vs usable confidence} x {outlier vs not}. The unit here must
#    still be the image: `model_frame` has up to four rows per image (one per
#    prompt), so a single pooled 2x2 table over all rows overstates the
#    evidence by roughly the replication factor. The test is run **per
#    prompt** (4 independent 1,155-image tables per model), and all four
#    p-values are reported rather than a single pooled number.

# %%
mnar_rows = []
for model in c.MODELS:
    model_frame = base_local[base_local["model"] == model]
    unusable_mask = model_frame["confidence_unusable"]
    n_unusable = int(unusable_mask.sum())
    n_total = len(model_frame)
    outlier_rate_unusable = float(model_frame.loc[unusable_mask, "outlier"].mean()) if n_unusable else np.nan
    outlier_rate_usable = float(model_frame.loc[~unusable_mask, "outlier"].mean())

    obs_conf = model_frame.loc[~unusable_mask, "confidence"]
    lo_extreme = float(obs_conf.min()) - 1.0  # strictly below every observed value
    hi_extreme = float(obs_conf.max()) + 1.0  # strictly above every observed value

    def bound_stat_extreme(frame_model, *, positives_get_low: bool):
        """Genuine worst-case fill: unresolved rows are sent the direction
        their own label would push hardest, strictly outside the observed
        range, never tied with it. `positives_get_low=True` gives the AUC's
        upper bound under this orientation (low confidence on outliers is the
        informative direction); `positives_get_low=False` gives the lower
        bound.
        """
        filled = frame_model.copy()
        mask = filled["confidence_unusable"]
        pos_fill = lo_extreme if positives_get_low else hi_extreme
        neg_fill = hi_extreme if positives_get_low else lo_extreme
        filled.loc[mask & (filled["outlier"] == 1), "confidence"] = pos_fill
        filled.loc[mask & (filled["outlier"] == 0), "confidence"] = neg_fill
        per_prompt = []
        for prompt, g in filled.groupby("prompt"):
            per_prompt.append(tie_aware_auc(g["confidence"].values, g["outlier"].values)["auc"])
        return float(np.nanmean(per_prompt))

    auc_upper_extreme = bound_stat_extreme(model_frame, positives_get_low=True)
    auc_lower_extreme = bound_stat_extreme(model_frame, positives_get_low=False)
    bound_min = min(auc_lower_extreme, auc_upper_extreme)
    bound_max = max(auc_lower_extreme, auc_upper_extreme)
    straddles_half = bound_min <= 0.5 <= bound_max

    # Fisher's exact test: unusable/usable x outlier/not, per prompt — the
    # image is the unit, and each image contributes one row per prompt, so a
    # table pooled over all 4 prompts would count every image up to 4 times.
    fisher_p_by_prompt = {}
    fisher_or_by_prompt = {}
    for prompt, g in model_frame.groupby("prompt"):
        ct = pd.crosstab(g["confidence_unusable"], g["outlier"])
        ct = ct.reindex(index=[False, True], columns=[0, 1], fill_value=0)
        odds_ratio_p, fisher_p_p = stats.fisher_exact(ct.values)
        fisher_p_by_prompt[prompt] = fisher_p_p
        fisher_or_by_prompt[prompt] = odds_ratio_p

    mnar_rows.append({
        "model": model,
        "n_total_rows": n_total,
        "n_unusable": n_unusable,
        "share_unusable": n_unusable / n_total,
        "outlier_rate_unusable": outlier_rate_unusable,
        "outlier_rate_usable": outlier_rate_usable,
        "auc_bound_lowfill": auc_lower_extreme,
        "auc_bound_highfill": auc_upper_extreme,
        "auc_bound_min": bound_min,
        "auc_bound_max": bound_max,
        "auc_not_determined_by_data": straddles_half,
        "fisher_p_detailed": fisher_p_by_prompt.get("Detailed", np.nan),
        "fisher_p_short": fisher_p_by_prompt.get("Short", np.nan),
        "fisher_p_grid_overlay": fisher_p_by_prompt.get("Grid-Overlay", np.nan),
        "fisher_p_point_hint": fisher_p_by_prompt.get("Point-Hint", np.nan),
        "fisher_or_detailed": fisher_or_by_prompt.get("Detailed", np.nan),
        "fisher_or_short": fisher_or_by_prompt.get("Short", np.nan),
        "fisher_or_grid_overlay": fisher_or_by_prompt.get("Grid-Overlay", np.nan),
        "fisher_or_point_hint": fisher_or_by_prompt.get("Point-Hint", np.nan),
        "fisher_note": (
            "Run per prompt (4 independent 1,155-image tables), never pooled across prompts "
            "within a model: model_frame carries up to 4 rows per image, so a single pooled "
            "2x2 table would count each image up to 4 times and overstate the evidence."
        ),
        "complete_case_mean_auc": float(
            auc_by_model_df.loc[auc_by_model_df["model"] == model, "mean_auc"].iloc[0]
        ),
    })

missingness_df = pd.DataFrame(mnar_rows).set_index("model").reindex(list(c.MODELS)).reset_index()
print(missingness_df[["model", "n_unusable", "share_unusable", "auc_bound_min", "auc_bound_max",
                       "auc_not_determined_by_data", "fisher_p_detailed", "fisher_p_short",
                       "fisher_p_grid_overlay", "fisher_p_point_hint"]])

# %% [markdown]
# The realised base/local unusable counts are small relative to n = 1,155 for
# every model, so the MNAR bounds below are expected to be narrow — but they
# are computed from A13's realised count, not assumed to be narrow in advance.

# %% [markdown]
# ## Label robustness — decoupling the label from the reference floor
#
# `|e| > 20` is only reachable when the reference or the prediction exceeds 20,
# so the outlier label is correlated with reference bin, and a positive AUC
# could partly mean "confidence tracks cover level" rather than "confidence
# tracks error". Two pre-declared checks:
#
# - **Per-bin quantile label**: outlier redefined as `|e|` above the 90th
#   percentile of `|e|` *within its own reference bin* — this keeps the
#   worst-decile framing but removes the shared, cover-independent threshold.
# - **0–20 reference bin only** (n = 933 images): the confound is removed
#   entirely by conditioning on the bin the fixed threshold is least likely to
#   distort.
#
# If AUC collapses toward 0.5 under either check, the headline result was
# substantially detecting bin structure rather than a within-bin error signal,
# and is reported that way.

# %%
MIN_POSITIVES = 20  # same stability threshold as the headline cells


def per_bin_quantile_label(frame: pd.DataFrame) -> pd.Series:
    q90_by_bin = frame.groupby("bin")["abs_e"].transform(lambda s: s.quantile(0.90))
    return (frame["abs_e"] > q90_by_bin).astype(int)


label_robustness_rows = []
label_robustness_cell_rows = []  # per (model, prompt, robustness-label) n_pos/n_neg detail
for model in c.MODELS:
    model_frame = base_local[base_local["model"] == model].copy()
    model_cc = complete_case_frame(model_frame)

    headline_auc = float(auc_by_model_df.loc[auc_by_model_df["model"] == model, "mean_auc"].iloc[0])

    # Per-bin quantile label. Same stability rule as the headline cells: a
    # prompt cell with fewer than MIN_POSITIVES positives cannot support an
    # AUC and is excluded from the mean rather than silently folded in.
    model_cc = model_cc.copy()
    model_cc["outlier_per_bin_q"] = per_bin_quantile_label(model_cc)
    per_prompt_q = []
    for prompt, g in model_cc.groupby("prompt"):
        res_q = tie_aware_auc(g["confidence"].values, g["outlier_per_bin_q"].values)
        unstable_q = res_q["n_pos"] < MIN_POSITIVES
        label_robustness_cell_rows.append({
            "model": model, "prompt": prompt, "robustness_label": "per_bin_quantile",
            "auc": res_q["auc"], "n_pos": res_q["n_pos"], "n_neg": res_q["n_neg"],
            "unstable_low_positives": unstable_q,
        })
        if not unstable_q:
            per_prompt_q.append(res_q["auc"])
    n_cells_excluded_q = 4 - len(per_prompt_q)
    auc_per_bin_q = float(np.nanmean(per_prompt_q)) if per_prompt_q else np.nan
    base_rate_per_bin_q = (
        model_cc.groupby("bin")["outlier_per_bin_q"].mean().reindex(c.BIN_LABELS).to_dict()
    )

    # 0-20 bin only.
    sub_020 = model_cc[model_cc["bin"] == "0-20"]
    per_prompt_020 = []
    for prompt, g in sub_020.groupby("prompt"):
        res_020 = tie_aware_auc(g["confidence"].values, g["outlier"].values)
        unstable_020 = res_020["n_pos"] < MIN_POSITIVES
        label_robustness_cell_rows.append({
            "model": model, "prompt": prompt, "robustness_label": "0_20_bin_only",
            "auc": res_020["auc"], "n_pos": res_020["n_pos"], "n_neg": res_020["n_neg"],
            "unstable_low_positives": unstable_020,
        })
        if not unstable_020:
            per_prompt_020.append(res_020["auc"])
    n_cells_excluded_020 = 4 - len(per_prompt_020)
    auc_020 = float(np.nanmean(per_prompt_020)) if per_prompt_020 else np.nan
    base_rate_020 = float(sub_020["outlier"].mean()) if len(sub_020) else np.nan
    n_020_images = int(sub_020["image"].nunique())

    delta_per_bin_quantile = auc_per_bin_q - headline_auc if np.isfinite(auc_per_bin_q) else np.nan
    delta_0_20_bin_only = auc_020 - headline_auc if np.isfinite(auc_020) else np.nan

    label_robustness_rows.append({
        "model": model,
        "headline_auc_complete_case": headline_auc,
        "auc_per_bin_quantile_label": auc_per_bin_q,
        "delta_per_bin_quantile": delta_per_bin_quantile,
        "n_prompt_cells_excluded_per_bin_quantile": n_cells_excluded_q,
        "base_rate_per_bin_quantile_0-20": base_rate_per_bin_q.get("0-20", np.nan),
        "base_rate_per_bin_quantile_20-40": base_rate_per_bin_q.get("20-40", np.nan),
        "base_rate_per_bin_quantile_40-60": base_rate_per_bin_q.get("40-60", np.nan),
        "base_rate_per_bin_quantile_60-80": base_rate_per_bin_q.get("60-80", np.nan),
        "base_rate_per_bin_quantile_80-100": base_rate_per_bin_q.get("80-100", np.nan),
        "auc_0_20_bin_only": auc_020,
        "delta_0_20_bin_only": delta_0_20_bin_only,
        "n_prompt_cells_excluded_0_20_bin_only": n_cells_excluded_020,
        "base_rate_0_20_bin_only": base_rate_020,
        "n_images_0_20_bin": n_020_images,
        # A robustness check "collapses" the headline result if the relabelled
        # departure from chance is less than half the headline departure, OR
        # if the relabelled AUC sits on the opposite side of 0.5 from the
        # headline (a sign reversal is a collapse regardless of magnitude —
        # scoring it as "strengthening" because |departure| grew would let a
        # direction flip hide behind a purely magnitude-based rule). Using a
        # fixed -0.05 margin against |headline - 0.5| is unusable once the
        # headline departure itself is under 0.05 (the margin goes negative
        # and no departure can ever be small enough to trip it), so the rule
        # is relative to the headline's own departure instead of an absolute
        # offset from it.
        "collapsed_toward_chance": (
            np.isfinite(delta_per_bin_quantile)
            and (
                np.sign(auc_per_bin_q - 0.5) != np.sign(headline_auc - 0.5)
                or abs(auc_per_bin_q - 0.5) < 0.5 * abs(headline_auc - 0.5)
            )
        ) or (
            np.isfinite(delta_0_20_bin_only)
            and (
                np.sign(auc_020 - 0.5) != np.sign(headline_auc - 0.5)
                or abs(auc_020 - 0.5) < 0.5 * abs(headline_auc - 0.5)
            )
        ),
    })

label_robustness_df = pd.DataFrame(label_robustness_rows).set_index("model").reindex(list(c.MODELS)).reset_index()
label_robustness_cells_df = pd.DataFrame(label_robustness_cell_rows)
print(label_robustness_df[["model", "headline_auc_complete_case", "auc_per_bin_quantile_label",
                            "n_prompt_cells_excluded_per_bin_quantile", "auc_0_20_bin_only",
                            "n_prompt_cells_excluded_0_20_bin_only", "collapsed_toward_chance"]])
print(label_robustness_cells_df[label_robustness_cells_df["unstable_low_positives"]])

# %% [markdown]
# ## Scout: with and without chain-of-thought-recovered rows
#
# Llama-4-Scout has 5 base/local rows whose confidence was recovered from
# chain-of-thought text (`confidence_parse_method == 'cot_resolved'`) rather
# than parsed directly — a different provenance from the rest of the
# complete-case set. Scout's headline AUC is reported both including and
# excluding these rows, so the CoT-recovered rows' influence is visible rather
# than silently folded in.

# %%
scout_frame = base_local[base_local["model"] == "Llama-4-Scout"]
scout_cc_with_cot = complete_case_frame(scout_frame)
scout_cc_without_cot = scout_cc_with_cot[scout_cc_with_cot["confidence_parse_method"] != "cot_resolved"]

auc_with_cot = model_mean_auc_statistic(scout_cc_with_cot)
auc_without_cot = model_mean_auc_statistic(scout_cc_without_cot)
n_cot_rows = int((scout_cc_with_cot["confidence_parse_method"] == "cot_resolved").sum())

scout_cot_sensitivity = pd.DataFrame([{
    "model": "Llama-4-Scout",
    "n_cot_resolved_rows": n_cot_rows,
    "mean_auc_with_cot_resolved": auc_with_cot,
    "mean_auc_without_cot_resolved": auc_without_cot,
    "delta": auc_with_cot - auc_without_cot,
}])
print(scout_cot_sensitivity)

# %% [markdown]
# ## Campaign-clustered companion
#
# The headline bootstrap treats images as independent. The campaign-clustered
# companion measures the cost of that assumption rather than asserting it is
# small. Every interval and every p-value in this study carries such a
# companion, so it is attached here not
# only to the six per-model headlines but also to the 24 model x prompt AUC
# cells and the 24 Spearman rho cells, none of which carry a p-value but all of
# which carry a CI. Per-image contributions to a rank statistic like AUC or
# Spearman rho have no wild-cluster-bootstrap analogue (there is no per-image
# `d_i` a Webb reweighting could act on the way it does for a weighted mean),
# so for rank-based statistics the leave-one-campaign-out (LOCO) range is
# itself the clustered companion. That range is reported for all 54
# estimates: recompute the same statistic three times, once with each
# campaign dropped, and report the range across the three two-campaign refits
# alongside the full-frame estimate. LOCO is three refits of an
# already-computed statistic, not a bootstrap, so it adds no meaningful cost
# even at this count.

# %%
def loco_refits_for_stat(frame_all_prompts: pd.DataFrame, stat_of_cc_frame) -> dict:
    """Three LOCO refits of `stat_of_cc_frame`, one per campaign dropped.
    `stat_of_cc_frame` receives the complete-case-filtered, campaign-dropped
    frame and returns a scalar.
    """
    out = {}
    for camp in sorted(frame_all_prompts["campaign"].unique()):
        sub = frame_all_prompts[frame_all_prompts["campaign"] != camp]
        out[camp] = stat_of_cc_frame(complete_case_frame(sub))
    return out


clustered_rows = []
for model in c.MODELS:
    model_frame = base_local[base_local["model"] == model]

    # Per-model headline LOCO (mean of 4 prompt AUCs).
    loco_estimates = loco_refits_for_stat(model_frame, model_mean_auc_statistic)
    loco_min, loco_max, loco_note = c.loco_range(list(loco_estimates.values()))

    headline = auc_by_model_df.loc[auc_by_model_df["model"] == model].iloc[0]
    clustered_rows.append({
        "model": model, "prompt": np.nan, "row_type": "headline",
        "estimate": headline["mean_auc"],
        "loco_min": loco_min, "loco_max": loco_max, "loco_note": loco_note,
        "loco_estimate_campaign_a": loco_estimates.get(sorted(model_frame["campaign"].unique())[0], np.nan),
        "loco_estimate_campaign_b": loco_estimates.get(sorted(model_frame["campaign"].unique())[1], np.nan)
        if len(model_frame["campaign"].unique()) > 1 else np.nan,
        "loco_estimate_campaign_c": loco_estimates.get(sorted(model_frame["campaign"].unique())[2], np.nan)
        if len(model_frame["campaign"].unique()) > 2 else np.nan,
        "p_clustered_webb": np.nan,
        "n_distinct_abs_t_star": np.nan,
        "clustered_method_note": (
            "No wild-cluster-bootstrap analogue exists for a rank statistic (AUC); "
            "the leave-one-campaign-out range is reported as the clustered companion directly."
        ),
    })

    # 24-cell sub-analysis: per model x prompt AUC.
    for prompt in sorted(model_frame["prompt"].unique()):
        def prompt_auc_stat(fr_cc, _prompt=prompt):
            g = fr_cc[fr_cc["prompt"] == _prompt]
            return tie_aware_auc(g["confidence"].values, g["outlier"].values)["auc"]

        loco_p = loco_refits_for_stat(model_frame, prompt_auc_stat)
        loco_p_min, loco_p_max, loco_p_note = c.loco_range(list(loco_p.values()))
        camps_sorted = sorted(model_frame["campaign"].unique())
        clustered_rows.append({
            "model": model, "prompt": prompt, "row_type": "auc_by_model_x_prompt",
            "estimate": auc_by_combo_df.loc[
                (auc_by_combo_df["model"] == model) & (auc_by_combo_df["prompt"] == prompt), "auc"
            ].iloc[0],
            "loco_min": loco_p_min, "loco_max": loco_p_max, "loco_note": loco_p_note,
            "loco_estimate_campaign_a": loco_p.get(camps_sorted[0], np.nan) if len(camps_sorted) > 0 else np.nan,
            "loco_estimate_campaign_b": loco_p.get(camps_sorted[1], np.nan) if len(camps_sorted) > 1 else np.nan,
            "loco_estimate_campaign_c": loco_p.get(camps_sorted[2], np.nan) if len(camps_sorted) > 2 else np.nan,
            "p_clustered_webb": np.nan,
            "n_distinct_abs_t_star": np.nan,
            "clustered_method_note": (
                "No wild-cluster-bootstrap analogue exists for a rank statistic (AUC); "
                "the leave-one-campaign-out range is reported as the clustered companion directly."
            ),
        })

    # 24-cell sub-analysis: per model x prompt Spearman rho.
    for prompt in sorted(model_frame["prompt"].unique()):
        def prompt_rho_stat(fr_cc, _prompt=prompt):
            g = fr_cc[fr_cc["prompt"] == _prompt]
            r, _ = c.spearman_tie_corrected(g["confidence"].values, g["abs_e"].values)
            return r

        loco_r = loco_refits_for_stat(model_frame, prompt_rho_stat)
        loco_r_min, loco_r_max, loco_r_note = c.loco_range(list(loco_r.values()))
        camps_sorted = sorted(model_frame["campaign"].unique())
        clustered_rows.append({
            "model": model, "prompt": prompt, "row_type": "spearman_confidence_vs_abs_e",
            "estimate": spearman_df.loc[
                (spearman_df["model"] == model) & (spearman_df["prompt"] == prompt), "spearman_rho"
            ].iloc[0],
            "loco_min": loco_r_min, "loco_max": loco_r_max, "loco_note": loco_r_note,
            "loco_estimate_campaign_a": loco_r.get(camps_sorted[0], np.nan) if len(camps_sorted) > 0 else np.nan,
            "loco_estimate_campaign_b": loco_r.get(camps_sorted[1], np.nan) if len(camps_sorted) > 1 else np.nan,
            "loco_estimate_campaign_c": loco_r.get(camps_sorted[2], np.nan) if len(camps_sorted) > 2 else np.nan,
            "p_clustered_webb": np.nan,
            "n_distinct_abs_t_star": np.nan,
            "clustered_method_note": (
                "No wild-cluster-bootstrap analogue exists for a rank statistic (Spearman rho); "
                "the leave-one-campaign-out range is reported as the clustered companion directly."
            ),
        })

clustered_df = pd.DataFrame(clustered_rows)
print(clustered_df[clustered_df["row_type"] == "headline"][
    ["model", "estimate", "loco_min", "loco_max", "loco_note"]
])
print(f"{len(clustered_df)} clustered-companion rows "
      f"(6 headline + 24 combo AUC + 24 Spearman rho = 54, matching every CI/p-value this notebook reports)")

# %% [markdown]
# ## Charts

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
for ax, model in zip(axes.ravel(), c.MODELS):
    model_frame = base_local[base_local["model"] == model]
    model_cc = complete_case_frame(model_frame)
    n_excl = int(model_frame["confidence_unusable"].sum())
    n_total = len(model_frame)

    # Pooled ROC curve (all 4 prompts pooled) purely for display; the AUC
    # annotated is the headline mean-of-4-prompts statistic, not this pooled
    # curve's own AUC, and the two are typically close but not identical.
    # Traversal: the decision rule is "predict outlier for confidence <= level",
    # so the curve must be swept from the *lowest* level to the highest — at the
    # lowest level nothing is predicted positive (TPR=FPR=0) and at the highest
    # every row is (TPR=FPR=1). Sweeping descending, as a naive "score" sweep
    # would, starts the walk at (1,1) and traces the curve backwards.
    conf = model_cc["confidence"].values
    lab = model_cc["outlier"].values
    levels = np.sort(np.unique(conf))  # ascending: lowest confidence first
    n_pos_total = int((lab == 1).sum())
    n_neg_total = int((lab == 0).sum())
    tprs, fprs = [0.0], [0.0]
    for lev in levels:
        pred_positive = conf <= lev  # low confidence -> predicted outlier
        tp = int(((lab == 1) & pred_positive).sum())
        fp = int(((lab == 0) & pred_positive).sum())
        tprs.append(tp / n_pos_total if n_pos_total else np.nan)
        fprs.append(fp / n_neg_total if n_neg_total else np.nan)
    # The last swept level already predicts every row positive (TPR=FPR=1), so
    # no further point is appended here.

    headline = auc_by_model_df.loc[auc_by_model_df["model"] == model].iloc[0]
    ax.plot(fprs, tprs, marker="o", markersize=4, color="steelblue")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title(
        f"{model}\nmean AUC={headline['mean_auc']:.3f} "
        f"[{headline['ci_lo_delta']+0.5:.3f}, {headline['ci_hi_delta']+0.5:.3f}]\n"
        f"n={n_total-n_excl} usable, {n_excl} unusable excluded",
        fontsize=9,
    )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

fig.suptitle(
    "Confidence-as-classifier ROC per model, pooled across 4 prompts\n"
    "(low confidence predicts |e| > 20; dashed line = chance; "
    "reported AUC is the mean of the 4 prompt-level AUCs, not this pooled curve's own)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(ROOT / "05_deliverables/Repository/GitHub/notebooks" / "rendered" / "Q5_roc_curves.png", dpi=150)
plt.show()

# %% [markdown]
# **What to look for.** A curve that hugs the diagonal means confidence does
# not discriminate outliers from non-outliers for that model. A curve that
# bows toward the top-left corner (high TPR at low FPR) means low confidence
# is disproportionately found on the images the model got badly wrong —
# confidence is informative in the direction this notebook asks about. A
# curve that bows toward the **bottom-right** (below the diagonal) means the
# opposite: confidence runs the wrong way, and is disproportionately *high*
# on the images the model got badly wrong. The annotated AUC and its 95%
# interval are the headline per-model statistic (mean of 4 prompt-level
# AUCs), and the excluded-row count in each panel's title is the same A13
# figure reported in `Q5_missingness_bounds.csv`.

# %%
fig2, ax2 = plt.subplots(figsize=(8, 5))
y_pos = np.arange(len(c.MODELS))
means = auc_by_model_df["mean_auc"].values
lo = auc_by_model_df["ci_lo_delta"].values + 0.5
hi = auc_by_model_df["ci_hi_delta"].values + 0.5
# `ci_lo_delta`/`ci_hi_delta` are already on the (mean AUC - 0.5) scale that
# `stat_fn` centred the bootstrap on, so adding 0.5 here returns them to the
# AUC scale directly — no other transformation is applied. An interval that
# fails to contain its own point estimate, or that leaves [0, 1], must be
# visible rather than clipped into looking plausible, so the half-widths
# below are asserted non-negative rather than clamped.
xerr_lo = means - lo
xerr_hi = hi - means
assert np.all(xerr_lo >= -1e-9) and np.all(xerr_hi >= -1e-9), (
    "a headline BCa interval does not contain its own point estimate — "
    f"means-lo={xerr_lo}, hi-means={xerr_hi}"
)
xerr_lo = np.clip(xerr_lo, 0, None)  # guards only floating-point dust at 1e-9, not a real asymmetry
xerr_hi = np.clip(xerr_hi, 0, None)
ax2.errorbar(means, y_pos, xerr=[xerr_lo, xerr_hi], fmt="o", color="darkorange",
             capsize=4, label="Headline mean AUC (95% CI)")
ax2.axvline(0.5, linestyle="--", color="grey", label="Chance (AUC = 0.5)")
ax2.set_xlim(0, 1)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(auc_by_model_df["model"])
ax2.set_xlabel("AUC (confidence as a classifier for |e| > 20)")
ax2.set_title("Per-model headline AUC with 95% bootstrap CI\n(no between-model comparison is tested — overlapping CIs are not a test)")
ax2.legend(loc="lower right", fontsize=8)
fig2.tight_layout()
fig2.savefig(ROOT / "05_deliverables/Repository/GitHub/notebooks" / "rendered" / "Q5_auc_by_model.png", dpi=150)
plt.show()

# %% [markdown]
# **What to look for.** Whether each model's interval sits clear of 0.5 (a
# discriminating signal) or straddles it (no detectable signal at this
# study's power — recall the pre-declared power calculation: an AUC of 0.60
# is detectable at alpha = 0.05/6, an AUC of 0.55 is not). The dashed vertical
# line marks chance.

# %% [markdown]
# ## Assumption checks written out

# %%
k5_rows = []
for model in c.MODELS:
    model_frame = base_local[base_local["model"] == model]
    k5 = c.check_k5_normality(model_frame["e"].values)
    k5["model"] = model
    k5_rows.append(k5)
k5_df = pd.DataFrame(k5_rows)

q5_assumption_rows = []
for _, row in assumption_df.iterrows():
    q5_assumption_rows.append({"check": row["id"], "description": row["description"],
                                "passed": row["passed"], "detail": row["detail"]})
q5_assumption_rows.append({"check": "K1", "description": k1_result.description,
                            "passed": k1_result.passed, "detail": k1_result.detail})
q5_assumption_rows.append({"check": "C11", "description": c11_result.description,
                            "passed": c11_result.passed, "detail": c11_result.detail})
for model in c.MODELS:
    q5_assumption_rows.append({
        "check": "K8_pooled", "description": f"confidence non-degeneracy, prompts pooled ({model}) — reference only",
        "passed": not bool(k8_df.loc[model, "degenerate"]),
        "detail": f"n_levels={k8_df.loc[model,'n_levels']}, max_mass={k8_df.loc[model,'max_single_level_mass']:.3f}",
    })
for _, k8row in k8_combo_df.iterrows():
    q5_assumption_rows.append({
        "check": "K8", "description": f"confidence non-degeneracy ({k8row['model']} x {k8row['prompt']})",
        "passed": not bool(k8row["degenerate"]),
        "detail": f"n_levels={k8row['n_levels']}, max_mass={k8row['max_single_level_mass']:.3f}",
    })
q5_assumption_checks_df = pd.DataFrame(q5_assumption_rows)
q5_assumption_checks_df.to_csv(ROOT / "05_deliverables/Repository/GitHub/results" / "Q5_assumption_checks.csv", index=False)
print(q5_assumption_checks_df)

# %% [markdown]
# ## Result
#
# **Family G verdict.** Six pre-declared tests of H0: AUC = 0.5, one per
# model, Holm-adjusted within family G (size 6, first threshold 0.05/6 =
# 0.00833). The table below is the full result; read the `p_bootstrap_holm`
# column against 0.05 for the corrected verdict, and the `ci_lo_delta` /
# `ci_hi_delta` columns (added to 0.5) for the unadjusted interval on the AUC
# itself.
#
# **There are three possible outcomes here, not two, because AUC is a
# two-sided departure from 0.5 and this notebook's orientation convention
# gives each side a distinct, opposite operational meaning.** Recall the
# convention fixed above: AUC is `P(confidence on a random non-outlier >
# confidence on a random outlier)`, so AUC > 0.5 means confidence is usefully
# *low* on the images the model gets badly wrong, and AUC < 0.5 means
# confidence runs the *opposite* way — it is systematically *higher* on the
# images the model gets worst. A Holm-adjusted verdict therefore resolves to
# one of:
#
# 1. **Informative** — Holm-adjusted p below 0.05 and the interval on
#    `(mean AUC - 0.5)` excludes zero **above** it. Read as: within that
#    model, higher self-reported confidence ranks images with smaller
#    absolute error better than chance. A ranking claim only — it does not
#    imply confidence is a calibrated probability, and it licenses no
#    comparison across models.
# 2. **Anti-informative** — Holm-adjusted p below 0.05 and the interval
#    excludes zero **below** it. Read as: within that model, confidence runs
#    the wrong way — it is systematically *higher*, not lower, on the images
#    the model gets badly wrong. This is not "less informative" or "no
#    signal"; it is a detected, opposite-direction relationship, and
#    operationally it means the model's own confidence value should not be
#    trusted at face value as a flag for bad estimates — if anything, treating
#    high confidence as a warning sign would do better than chance for this
#    model.
# 3. **Not detected** — Holm-adjusted p at or above 0.05, or the interval
#    spans zero. Read as: confidence does not usefully flag bad estimates for
#    that model, in either direction, at a magnitude this study's power could
#    detect (a detectable departure from 0.5 is about 0.10 at this sample size
#    and correction; an AUC of 0.55 is not detectable and this outcome must be
#    read as "no strong signal", not "no signal at all").
#
# The MNAR bounds and the label-robustness checks are read alongside every
# verdict, not after it: a model whose bounds straddle 0.5 has an AUC this
# data does not determine, regardless of what the complete-case point estimate
# says, and a model whose AUC collapses toward chance under the per-bin or
# 0-20-bin-only label was substantially detecting reference-bin structure
# rather than a genuine confidence-error relationship.
#
# A fourth outcome is possible and is reported separately from the three
# above: **label-dependent / not robustly classified**. This applies when the
# headline clears the Holm threshold, but the direction is not corroborated —
# the cut-free Spearman companion disagrees with the headline's sign, or a
# relabelling check collapses the departure from chance by at least half or
# reverses its sign, or the cells feeding the headline are K8-degenerate. A
# model in this category gets a direction-free verdict, not an
# informative/anti-informative label it cannot support: the point estimate is
# reported as `not robustly classified`, and the disagreement across companions
# is stated as the finding, rather than resolved by picking the headline over
# its own checks.

# %%
def classify_family_g_result(row: pd.Series) -> str:
    """Three-way Holm-adjusted verdict, before the robustness override below.

    This function alone decides only whether the headline clears the Holm
    threshold and on which side of 0.5 it sits. It does not look at the
    Spearman companion, the relabelling checks, or K8 degeneracy — those are
    applied afterward, and can downgrade (never invent) a directional verdict
    to `label-dependent / not robustly classified`.
    """
    if row["p_bootstrap_holm"] >= ALPHA:
        return "not detected"
    return "informative" if row["mean_auc"] > 0.5 else "anti-informative"


auc_by_model_df["result_direction"] = auc_by_model_df.apply(classify_family_g_result, axis=1)

# Robustness override: a directional verdict (informative / anti-informative)
# is downgraded to "label-dependent / not robustly classified" if any of the
# model's own pre-declared companions disagree with the headline's sign, or if
# the cells feeding it are K8-degenerate. This is what the Spearman companion
# and the relabelling checks exist for: a headline that
# clears the Holm threshold is not, by itself, sufficient evidence for a
# directional operational claim if the model's own companions point the
# other way.
headline_sign = np.sign(auc_by_model_df["mean_auc"].values - 0.5)
# AUC and Spearman rho use opposite sign conventions for the same
# "informative" direction: AUC > 0.5 means confidence on non-outliers
# exceeds confidence on outliers, i.e. confidence and |e| are NEGATIVELY
# related, so the informative pairing is AUC > 0.5 with rho < 0 (and
# anti-informative is AUC < 0.5 with rho > 0). Comparing sign(rho) directly
# to sign(auc - 0.5) would flag every model as disagreeing; the correct
# comparison is against sign(-rho).
spearman_sign = np.sign(
    -auc_by_model_df["model"].map(spearman_model_means).values
)
disagrees_with_spearman = spearman_sign != headline_sign
collapsed = auc_by_model_df["model"].map(
    label_robustness_df.set_index("model")["collapsed_toward_chance"]
).fillna(False).values
k8_flag = auc_by_model_df["k8_degenerate"].values

not_robust = (
    (auc_by_model_df["result_direction"] != "not detected").values
    & (disagrees_with_spearman | collapsed | k8_flag)
)
auc_by_model_df["not_robustly_classified"] = not_robust
auc_by_model_df["result_direction_robust"] = np.where(
    not_robust, "label-dependent / not robustly classified",
    auc_by_model_df["result_direction"],
)

result_summary = auc_by_model_df[[
    "model", "mean_auc", "ci_lo_delta", "ci_hi_delta", "p_bootstrap_raw",
    "p_bootstrap_holm", "p_at_floor", "result_direction", "result_direction_robust",
    "not_robustly_classified", "k8_degenerate",
]].copy()
result_summary["ci_lo_auc"] = result_summary["ci_lo_delta"] + 0.5
result_summary["ci_hi_auc"] = result_summary["ci_hi_delta"] + 0.5
result_summary["spearman_rho_mean"] = result_summary["model"].map(spearman_model_means)
print(result_summary[["model", "mean_auc", "ci_lo_auc", "ci_hi_auc", "p_bootstrap_holm",
                       "p_at_floor", "spearman_rho_mean", "result_direction_robust"]].to_string(index=False))
print()
for _, r in result_summary.iterrows():
    # `p_at_floor` is a property of the RAW bootstrap p (floored at
    # 1/(B+1)), not of its Holm image — the two are different numbers
    # (raw floor 9.999e-05 vs its Holm image 6x that, 5.9994e-04 for the five
    # floored models here). The line below prints the Holm-adjusted p and
    # labels it `Holm p=`, so the floor annotation is keyed to whichever
    # quantity is actually printed: the Holm value itself, formatted to full
    # precision, with a parenthetical noting that the underlying raw p was at
    # its Monte-Carlo floor (not that the Holm value itself is floored).
    holm_val = r["p_bootstrap_holm"]
    if r["p_at_floor"]:
        p_str = f"{holm_val:.4g} (raw p at bootstrap floor)"
    else:
        p_str = f"{holm_val:.4f}"
    print(f"  {r['model']}: {r['result_direction_robust']} "
          f"(AUC={r['mean_auc']:.4f}, 95% CI [{r['ci_lo_auc']:.4f}, {r['ci_hi_auc']:.4f}], "
          f"Holm p={p_str}, mean Spearman rho={r['spearman_rho_mean']:.4f})")
print()
print(missingness_df[["model", "auc_bound_min", "auc_bound_max", "auc_not_determined_by_data"]])
print()
print(label_robustness_df[["model", "headline_auc_complete_case", "auc_per_bin_quantile_label",
                            "auc_0_20_bin_only", "collapsed_toward_chance"]])
print()
print("Companion disagreement flags feeding the robustness override:")
print(result_summary[["model", "k8_degenerate"]].assign(
    spearman_disagrees_with_headline=disagrees_with_spearman,
    relabelling_collapsed=collapsed,
).to_string(index=False))

# %% [markdown]
# ## Confidence-threshold filtering — what a practitioner actually does with a
# ## confidence score, and what that filtering actually buys
#
# The AUC above answers "does confidence rank images by error at all?" The
# question a practitioner asks next is operational: *if I discard the
# predictions the model is least confident in, does the remaining set get
# more accurate?* Filtering on a threshold nearly always looks like it helps,
# and the mechanism behind that appearance is usually not a reduction in
# error — it is a **change in what gets scored**. Confidence correlates with
# reference cover (sparse 0–20% quadrats are visually easy and get called with
# more confidence than cluttered ones), so tightening a cutoff preferentially
# retains easy, sparse scenes. Reporting the resulting MAE without saying so
# lets a composition shift pass as an accuracy gain.
#
# **This block is descriptive only — no test, no p-value, no family.** Family
# G's six tests of AUC = 0.5 are unaffected and unchanged.
#
# Three quantities are reported at every cutoff, and the third is the reason
# this analysis exists:
#
# 1. **Fraction of predictions retained** — the coverage cost of filtering.
# 2. **Pooled MAE of the retained set**, with an image-bootstrap BCa interval.
# 3. **Share of the retained set in the 0–20 cover bin.** Unfiltered, that
#    share is computed below and comes out close to 80.8%. If this
#    share rises as the cutoff tightens, part of any apparent MAE improvement
#    was bought by scoring an easier, sparser sample — not by the model
#    becoming more accurate on the same population of images.
#
# **No sentence anywhere in this notebook or the paper may report a
# filtered MAE without stating the retained fraction and the retained 0–20
# share alongside it** — a filtered MAE reported alone is a composition change
# dressed as an accuracy gain.
#
# **The cutoff grid is the realisable one, verified here rather than assumed.**
# On the *model x prompt* cell, the unit this notebook actually filters on,
# `D5.v5` (self-reported confidence) takes only **3–10 distinct values per
# cell**. The grid used below is therefore the distinct
# usable confidence values actually observed **in each cell**, plus a
# retain-all row, never a fixed step (e.g. 0.1) grid, which would invent
# cutpoints no prediction sits on and silently repeat the same retained set
# under several different labels.
#
# **Confidence is not comparable across models as a numeric cutoff.**
# `confidence` is ordinal, and a cut on an ordered scale within one model is a
# legitimate operation — but the same numeric value means different things in
# different models' ranges.
# The sweep below is therefore always run and reported **within one
# model x prompt cell at a time**; no cutoff is ever compared at the same
# numeric value across cells, and no cross-model ranking is drawn from this
# block.

# %%
cell_grids = {}
for (model, prompt), g in base_local.groupby(["model", "prompt"]):
    usable_conf = g.loc[~g["confidence_unusable"], "confidence"]
    n_levels = int(usable_conf.nunique())
    cell_grids[(model, prompt)] = n_levels

n_levels_series = pd.Series(cell_grids)
print(f"Distinct usable confidence levels per model x prompt cell: "
      f"min={n_levels_series.min()}, max={n_levels_series.max()} "
      f"(this is the per-cell range; the per-model range is wider, because a "
      f"model pooled over its four prompts uses more distinct values than it "
      f"does under any one of them)")
assert n_levels_series.min() >= 2, "a cell has fewer than 2 usable confidence levels; the sweep needs at least one real cutpoint"
assert 3 <= n_levels_series.min() and n_levels_series.max() <= 10, (
    f"distinct usable confidence levels per cell fell outside the 3-10 range "
    f"this notebook's prose claims (observed min={n_levels_series.min()}, "
    f"max={n_levels_series.max()}); the prose above must be updated to match "
    f"before proceeding"
)

unfiltered_020_share = float((base_local["bin"] == "0-20").mean())
print(f"Unfiltered share of images in the 0-20 cover bin: {unfiltered_020_share:.4f}")

IMAGES_Q5_UNIVERSE = base_local["image"].unique()
IMG_INDEX_Q5_UNIVERSE = pd.Index(IMAGES_Q5_UNIVERSE)

# At the tightest realisable cutoffs a cell's retained set can fall to one or
# two images. The image bootstrap resamples 1,155 images with replacement, so
# most replicates then draw none of the retained images and contribute no
# defined value: for n_retained = 1 the fraction of replicates with zero
# draws of that one image is (1 - 1/1155)^1155 = 36.8%; for n_retained = 2 it
# is 13.5%. A percentile interval computed on whatever replicates survive is
# not the same object as a percentile interval on a healthy retained set, and
# it must not carry the same label. `N_RETAINED_CI_FLOOR` is the retained-n
# below which a row's interval is flagged as resting on a depleted bootstrap
# rather than suppressed outright — the tail is kept and reported, because it
# is the informative end of the sweep, but marked so a reader cannot mistake
# it for a normal estimate. 30 is a sample-size guard, not a threshold derived
# from the undefined-replicate rate: that rate is negligible at the boundary
# ((1 - 30/1155)^1155 is of order 1e-14) and becomes material only in the last
# few rows, where a single retained image leaves roughly a third of resamples
# empty. What makes a row below 30 untrustworthy is the retained set itself,
# not the resampling — a percentile interval over one or two images describes
# those images and nothing wider.
N_RETAINED_CI_FLOOR = 30

# %%
# The quantity being estimated throughout this block is the plain pooled
# (unbalanced) MAE of the retained set — not the five-bin balanced MAE this
# plan uses for ranking configurations elsewhere. The filtering sweep is
# about what a practitioner sees after thresholding, which is the pooled
# number a reader would compute directly on the retained rows.
def filtering_sweep_for_group(g: pd.DataFrame, images_universe: np.ndarray, rng: np.random.Generator,
                               row_type: str, model_label, prompt_label) -> list[dict]:
    """The filtering sweep (retained fraction, pooled MAE with BCa CI,
    retained 0-20 share) for one group (a model x prompt cell, or a
    per-model pool of all 4 prompts), evaluated at every realisable cutoff.

    Runs the image resampling **once** per group and reduces every replicate
    with `np.add.at` (an image-indexed scatter-sum), the same
    precomputed-array strategy Q1 uses for its 24-combo balanced-MAE
    bootstrap and this notebook uses above for the AUC top-set. A naive
    per-cutoff call to `c.image_bootstrap` would redraw 10,000 fresh image
    resamples and rebuild a pandas frame for every one of a cell's up-to-11
    cutoffs; here the B=10,000 draws are generated once per group and reused,
    with per-cutoff cost reduced to two `np.add.at` scatter-sums over the
    masked rows.
    """
    g_cc = complete_case_frame(g)
    n_excluded_unusable = int(g["confidence_unusable"].sum())
    usable_levels = np.sort(g_cc["confidence"].unique())
    cutoffs = [-np.inf] + list(usable_levels)

    n_images = len(images_universe)
    img_index = pd.Index(images_universe)
    row_img_pos = img_index.get_indexer(g["image"].values)
    conf_arr = np.where(g["confidence_unusable"].values, np.nan, g["confidence"].values)
    abs_e_arr = g["abs_e"].values
    is_020_arr = (g["bin"].values == "0-20")

    B = c.B_BOOTSTRAP
    # One set of B image-index draws per group, reused across every cutoff.
    draws = rng.integers(0, n_images, size=(B, n_images))  # (B, n_images), values are image positions
    counts_mat = np.zeros((B, n_images), dtype=np.int32)
    for b in range(B):
        np.add.at(counts_mat[b], draws[b], 1)
    # Per-row draw multiplicity for every replicate: (B, n_rows).
    row_counts_mat = counts_mat[:, row_img_pos]

    rows_out = []
    for cutoff in cutoffs:
        cutoff_mask = (conf_arr >= cutoff) if np.isfinite(cutoff) else np.ones(len(g), dtype=bool)
        cutoff_mask = cutoff_mask & ~np.isnan(conf_arr)
        n_retained_obs = int(cutoff_mask.sum())
        if n_retained_obs == 0:
            continue
        retained_fraction_obs = n_retained_obs / len(g_cc)
        share_020_obs = float(is_020_arr[cutoff_mask].mean())
        theta_hat = float(abs_e_arr[cutoff_mask].mean())

        w = row_counts_mat[:, cutoff_mask]                 # (B, n_retained)
        num = w @ abs_e_arr[cutoff_mask]                   # (B,)
        den = w.sum(axis=1)                                 # (B,)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot_vals = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)

        # Leave-one-image-out jackknife on the same masked-mean statistic,
        # vectorised over all n_images at once via one scatter-sum per
        # retained row rather than a per-image Python loop.
        retained_img_pos = row_img_pos[cutoff_mask]
        retained_abs_e = abs_e_arr[cutoff_mask]
        sum_by_image = np.zeros(n_images)
        cnt_by_image = np.zeros(n_images)
        np.add.at(sum_by_image, retained_img_pos, retained_abs_e)
        np.add.at(cnt_by_image, retained_img_pos, 1)
        total_sum, total_cnt = retained_abs_e.sum(), retained_abs_e.size
        jack_sum = total_sum - sum_by_image
        jack_cnt = total_cnt - cnt_by_image
        with np.errstate(invalid="ignore", divide="ignore"):
            jack_vals = np.where(jack_cnt > 0, jack_sum / np.where(jack_cnt > 0, jack_cnt, 1), np.nan)

        finite_boot = boot_vals[np.isfinite(boot_vals)]
        finite_jack = jack_vals[np.isfinite(jack_vals)]
        n_boot_replicates_defined = int(len(finite_boot))
        boot_replicates_undefined_fraction = 1.0 - n_boot_replicates_defined / B
        degenerate_ci = n_retained_obs < N_RETAINED_CI_FLOOR

        if len(finite_boot) < B or len(finite_jack) < n_images:
            ci_lo = float(np.percentile(finite_boot, 2.5)) if len(finite_boot) else np.nan
            ci_hi = float(np.percentile(finite_boot, 97.5)) if len(finite_boot) else np.nan
            if degenerate_ci:
                ci_method = (
                    f"percentile (BCa fallback, degenerate: n_retained={n_retained_obs} "
                    f"< {N_RETAINED_CI_FLOOR}, {boot_replicates_undefined_fraction:.1%} "
                    f"of replicates undefined)"
                )
            else:
                ci_method = "percentile (BCa fallback)"
        else:
            bca_lo, bca_hi, used_bca = c._bca_interval(theta_hat, boot_vals, jack_vals)
            if used_bca:
                ci_lo, ci_hi, ci_method = bca_lo, bca_hi, "BCa"
            else:
                ci_lo = float(np.percentile(boot_vals, 2.5))
                ci_hi = float(np.percentile(boot_vals, 97.5))
                ci_method = "percentile (BCa fallback)"
            # len(finite_boot) == B here, so degenerate_ci (n_retained < floor)
            # cannot co-occur with this branch in practice for the floor chosen
            # above, but the label is still driven off the same flag for safety
            # if the floor is ever lowered.
            if degenerate_ci:
                ci_method = f"{ci_method}, degenerate: n_retained={n_retained_obs} < {N_RETAINED_CI_FLOOR}"

        rows_out.append({
            "question_id": "Q5", "row_type": row_type,
            "model": model_label, "prompt": prompt_label,
            "cutoff": cutoff if np.isfinite(cutoff) else np.nan,
            "cutoff_label": "retain_all" if not np.isfinite(cutoff) else f"{cutoff:.4g}",
            "n_retained": n_retained_obs, "n_cell_usable": len(g_cc),
            "n_excluded_unusable_confidence": n_excluded_unusable,
            "retained_fraction": retained_fraction_obs,
            "pooled_mae": theta_hat, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "ci_method": ci_method,
            "n_boot_replicates_defined": n_boot_replicates_defined,
            "boot_replicates_undefined_fraction": boot_replicates_undefined_fraction,
            "ci_degenerate": degenerate_ci,
            "share_retained_0_20_bin": share_020_obs,
            "unfiltered_share_0_20_bin": unfiltered_020_share,
        })
    return rows_out


filtering_rows = []
for (model, prompt), g in base_local.groupby(["model", "prompt"]):
    filtering_rows.extend(
        filtering_sweep_for_group(g, IMAGES_Q5_UNIVERSE, rng_filtering, "single_configuration", model, prompt)
    )

filtering_df = pd.DataFrame(filtering_rows)
print(f"{len(filtering_df)} sweep rows across {filtering_df.groupby(['model','prompt']).ngroups} "
      f"model x prompt cells (single-configuration arm)")
print(filtering_df[filtering_df["model"] == "Gemma-3-27B"][
    ["prompt", "cutoff_label", "retained_fraction", "pooled_mae", "share_retained_0_20_bin"]
].to_string(index=False))

# %% [markdown]
# ### Per-model aggregate rows
#
# The 6 per-model aggregates (pooling all 4 prompts' retained rows at each
# realisable cutoff) give one sweep per model, alongside the 24 per-cell
# sweeps above, matching the unit the AUC headline already uses.

# %%
model_filtering_rows = []
for model, g_model in base_local.groupby("model"):
    model_filtering_rows.extend(
        filtering_sweep_for_group(g_model, IMAGES_Q5_UNIVERSE, rng_filtering, "per_model_aggregate", model, np.nan)
    )

model_filtering_df = pd.DataFrame(model_filtering_rows)
print(f"{len(model_filtering_df)} sweep rows across the 6 per-model aggregates")

# %% [markdown]
# ### Per-bin MAE change at a pre-declared representative cutoff
#
# The representative cutoff is fixed by a rule stated before the sweep is
# inspected, not chosen after seeing which cutoff looks best: **the
# realisable cutoff whose retained fraction is closest to 0.50**, ties broken
# to the higher cutoff. At that cutoff, per-bin MAE is reported before and
# after filtering, with n and a CI in both states, so a reader can see
# whether an apparent gain is a within-bin improvement or a between-bin
# composition shift. A gain that shows up only in the marginal pooled MAE and
# not within any individual bin is a composition effect, not an accuracy
# effect, and is reported as one.

# %%
def representative_cutoff_row(cell_df: pd.DataFrame) -> pd.Series:
    """The realisable cutoff (excluding retain-all) whose retained_fraction
    is closest to 0.50, ties broken toward the higher cutoff."""
    candidates = cell_df[cell_df["cutoff_label"] != "retain_all"].copy()
    candidates["dist_to_half"] = (candidates["retained_fraction"] - 0.5).abs()
    min_dist = candidates["dist_to_half"].min()
    tied = candidates[np.isclose(candidates["dist_to_half"], min_dist)]
    return tied.sort_values("cutoff", ascending=False).iloc[0]


def masked_mean_with_bca(abs_e_arr: np.ndarray, row_img_pos: np.ndarray, mask: np.ndarray,
                          images_universe: np.ndarray, rng: np.random.Generator) -> tuple:
    """Pooled mean of `abs_e_arr[mask]` with an image-bootstrap BCa interval,
    computed with one fresh set of B image draws — the same scatter-sum
    reduction used in `filtering_sweep_for_group` above, specialised to a
    single fixed mask (a bin subset) rather than a sweep across cutoffs.

    Returns `(theta_hat, ci_lo, ci_hi, ci_method, n_boot_replicates_defined,
    boot_replicates_undefined_fraction, ci_degenerate)`. `ci_degenerate` uses
    the same `N_RETAINED_CI_FLOOR` (images retained by the mask, not rows) as
    the sweep above, so a bin subset thinned by filtering to a handful of
    images gets the same degenerate labelling rather than a bare
    "BCa"/"percentile" string indistinguishable from a healthy interval.
    """
    n_images = len(images_universe)
    n_mask = int(mask.sum())
    if n_mask == 0:
        return np.nan, np.nan, np.nan, "n/a (empty subset)", 0, 1.0, True
    theta_hat = float(abs_e_arr[mask].mean())
    B = c.B_BOOTSTRAP
    draws = rng.integers(0, n_images, size=(B, n_images))
    counts_mat = np.zeros((B, n_images), dtype=np.int32)
    for b in range(B):
        np.add.at(counts_mat[b], draws[b], 1)
    w = counts_mat[:, row_img_pos[mask]]
    num = w @ abs_e_arr[mask]
    den = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot_vals = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)

    retained_img_pos = row_img_pos[mask]
    retained_abs_e = abs_e_arr[mask]
    sum_by_image = np.zeros(n_images)
    cnt_by_image = np.zeros(n_images)
    np.add.at(sum_by_image, retained_img_pos, retained_abs_e)
    np.add.at(cnt_by_image, retained_img_pos, 1)
    total_sum, total_cnt = retained_abs_e.sum(), retained_abs_e.size
    jack_sum = total_sum - sum_by_image
    jack_cnt = total_cnt - cnt_by_image
    with np.errstate(invalid="ignore", divide="ignore"):
        jack_vals = np.where(jack_cnt > 0, jack_sum / np.where(jack_cnt > 0, jack_cnt, 1), np.nan)

    finite_boot = boot_vals[np.isfinite(boot_vals)]
    finite_jack = jack_vals[np.isfinite(jack_vals)]
    n_boot_replicates_defined = int(len(finite_boot))
    boot_replicates_undefined_fraction = 1.0 - n_boot_replicates_defined / B
    degenerate_ci = n_mask < N_RETAINED_CI_FLOOR

    if len(finite_boot) < B or len(finite_jack) < n_images:
        ci_lo = float(np.percentile(finite_boot, 2.5)) if len(finite_boot) else np.nan
        ci_hi = float(np.percentile(finite_boot, 97.5)) if len(finite_boot) else np.nan
        if degenerate_ci:
            ci_method = (
                f"percentile (BCa fallback, degenerate: n={n_mask} < {N_RETAINED_CI_FLOOR}, "
                f"{boot_replicates_undefined_fraction:.1%} of replicates undefined)"
            )
        else:
            ci_method = "percentile (BCa fallback)"
        return theta_hat, ci_lo, ci_hi, ci_method, n_boot_replicates_defined, boot_replicates_undefined_fraction, degenerate_ci
    bca_lo, bca_hi, used_bca = c._bca_interval(theta_hat, boot_vals, jack_vals)
    if used_bca:
        ci_lo, ci_hi, ci_method = bca_lo, bca_hi, "BCa"
    else:
        ci_lo, ci_hi = float(np.percentile(boot_vals, 2.5)), float(np.percentile(boot_vals, 97.5))
        ci_method = "percentile (BCa fallback)"
    if degenerate_ci:
        ci_method = f"{ci_method}, degenerate: n={n_mask} < {N_RETAINED_CI_FLOOR}"
    return theta_hat, ci_lo, ci_hi, ci_method, n_boot_replicates_defined, boot_replicates_undefined_fraction, degenerate_ci


perbin_change_rows = []
for (model, prompt), cell_df in filtering_df.groupby(["model", "prompt"]):
    rep_row = representative_cutoff_row(cell_df)
    rep_cutoff = rep_row["cutoff"]

    full_cell = base_local[(base_local["model"] == model) & (base_local["prompt"] == prompt)]
    full_cc = complete_case_frame(full_cell)
    row_img_pos_cell = IMG_INDEX_Q5_UNIVERSE.get_indexer(full_cc["image"].values)
    abs_e_cell = full_cc["abs_e"].values
    bin_cell = full_cc["bin"].values
    conf_cell = full_cc["confidence"].values

    for bin_label in c.BIN_LABELS:
        before_mask = bin_cell == bin_label
        after_mask = before_mask & (conf_cell >= rep_cutoff)
        n_before = int(before_mask.sum())
        n_after = int(after_mask.sum())

        mae_before, ci_lo_before, ci_hi_before, ci_method_before, n_boot_defined_before, boot_undefined_frac_before, ci_degenerate_before = masked_mean_with_bca(
            abs_e_cell, row_img_pos_cell, before_mask, IMAGES_Q5_UNIVERSE, rng_filtering
        )
        mae_after, ci_lo_after, ci_hi_after, ci_method_after, n_boot_defined_after, boot_undefined_frac_after, ci_degenerate_after = masked_mean_with_bca(
            abs_e_cell, row_img_pos_cell, after_mask, IMAGES_Q5_UNIVERSE, rng_filtering
        )

        perbin_change_rows.append({
            "question_id": "Q5", "row_type": "perbin_mae_change_at_representative_cutoff",
            "model": model, "prompt": prompt,
            "representative_cutoff": rep_cutoff,
            "representative_cutoff_retained_fraction": rep_row["retained_fraction"],
            "bin": bin_label,
            "n_before": n_before, "mae_before": mae_before,
            "ci_lo_before": ci_lo_before, "ci_hi_before": ci_hi_before,
            "ci_method_before": ci_method_before,
            "n_boot_replicates_defined_before": n_boot_defined_before,
            "ci_degenerate_before": ci_degenerate_before,
            "n_after": n_after, "mae_after": mae_after,
            "ci_lo_after": ci_lo_after, "ci_hi_after": ci_hi_after,
            "ci_method_after": ci_method_after,
            "n_boot_replicates_defined_after": n_boot_defined_after,
            "ci_degenerate_after": ci_degenerate_after,
        })

perbin_change_df = pd.DataFrame(perbin_change_rows)
print(f"{len(perbin_change_df)} per-bin change rows (24 cells x 5 bins)")
print(perbin_change_df[perbin_change_df["model"] == "Gemma-3-27B"][
    ["prompt", "bin", "n_before", "mae_before", "n_after", "mae_after"]
].to_string(index=False))

# %% [markdown]
# ### What the representative-cutoff table actually shows
#
# The pooled MAE and the per-bin breakdown above are computed on the same
# cutoff, so the two can be read against each other directly rather than
# left as two separate tables. For Gemma-3-27B x Detailed (the AUC top-set's
# observed best configuration), at its pre-declared representative cutoff:

# %%
_g27_detailed_cell = filtering_df[(filtering_df["model"] == "Gemma-3-27B") & (filtering_df["prompt"] == "Detailed")]
_g27_detailed_all = _g27_detailed_cell[_g27_detailed_cell["cutoff_label"] == "retain_all"].iloc[0]
_g27_detailed_rep = representative_cutoff_row(_g27_detailed_cell)
_g27_detailed_bins = perbin_change_df[(perbin_change_df["model"] == "Gemma-3-27B") & (perbin_change_df["prompt"] == "Detailed")]
_n_bins_worse = int((_g27_detailed_bins["mae_after"] > _g27_detailed_bins["mae_before"]).sum())
_share_before = _g27_detailed_all["share_retained_0_20_bin"]
_share_after = _g27_detailed_rep["share_retained_0_20_bin"]

print(
    f"Gemma-3-27B x Detailed: pooled MAE falls {_g27_detailed_all['pooled_mae']:.2f} "
    f"(unfiltered) -> {_g27_detailed_rep['pooled_mae']:.2f} (retained_fraction="
    f"{_g27_detailed_rep['retained_fraction']:.2f}), while within-bin MAE gets WORSE "
    f"in {_n_bins_worse} of 5 bins (see the per-bin table above/below), and the "
    f"0-20 cover-bin share of the retained set rises {_share_before:.3f} -> {_share_after:.3f}."
)

# %% [markdown]
# **This is the composition artefact the block was built to demonstrate, not
# a caveat on top of it.** The pooled-MAE improvement is real, but most of it
# is not an accuracy gain on a fixed population — it is a shift toward
# scoring an easier, sparser sample. Filtering makes three of the five cover
# bins individually *harder* to predict within (their own MAE rises once the
# low-confidence rows are removed from them), and the images that remain skew
# further toward the 0–20% bin, which was already the easiest and the
# majority class before any filtering. A reader who sees only the pooled
# number — 10.89 down to 8.63 in this configuration — would conclude
# filtering makes the model more accurate; the per-bin table shows that for
# most of the cover range, it does the opposite, and the pooled figure moves
# because the population being scored changed.

# %% [markdown]
# ### The ensemble arm lives in `Q2_ensembles.py`, not here
#
# This identical sweep — same three quantities, same realisable-grid rule,
# same representative-cutoff rule — is also run for the six
# ensemble estimators (A1-A5, B1) on their out-of-fold predictions, written to
# `Q2_confidence_filtering.csv`. An ensemble's confidence is the **median** of
# its members' `D5.v5` values (median, not mean, because confidence is
# ordinal). That computation is placed in `Q2_ensembles.py` because that is
# the notebook holding the out-of-fold predictions and the fold-level member
# sets the sweep needs, and re-deriving an out-of-fold split here would either
# duplicate that machinery or silently diverge from it. `Q5_confidence_filtering.csv`
# below carries the single-configuration arm only; the two tables are read
# side by side, and neither substitutes for the other.

# %% [markdown]
# ## Chart — retained fraction, pooled MAE and the 0-20 share, together
#
# **What to look for.** For the three configurations named in the AUC
# top-set below (the observed best and its two closest competitors), pooled
# MAE (left axis) against the retained fraction (x axis, decreasing left to
# right as the cutoff tightens), with the retained 0-20 share (right axis)
# drawn on the same panel. If the 0-20 share rises as MAE falls, the two
# lines moving in step is the composition-shift mechanism this block exists
# to show — not an independent confirmation that filtering "works".
#
# **Points at the tightest cutoffs are marked, not deleted.** Once
# `n_retained` for a cutoff falls below `N_RETAINED_CI_FLOOR` (30 images),
# most bootstrap replicates draw none of the retained images and the interval
# is built on a heavily depleted replicate set (`ci_degenerate = True` in
# `Q5_confidence_filtering.csv`, see the block above). Those points are drawn
# as hollow markers with no CI band: the point estimate is still the correct
# sample mean of one or a few images, but a tight-looking hollow point at the
# right edge of a panel is a measurement on n = 1 or n = 2 images, not a
# precisely estimated population MAE, and must not be read as the strongest
# evidence in the panel.

# %%
highlight_combos = [("Gemma-3-27B", "Detailed"), ("Llama-4-Maverick", "Grid-Overlay"), ("Qwen-2.5", "Short")]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
for ax, (model, prompt) in zip(axes, highlight_combos):
    cell_df = filtering_df[(filtering_df["model"] == model) & (filtering_df["prompt"] == prompt)].sort_values("retained_fraction")
    healthy = cell_df[~cell_df["ci_degenerate"]]
    degenerate = cell_df[cell_df["ci_degenerate"]]
    ax2 = ax.twinx()
    # Solid line through every point so the trend is visible, then the
    # degenerate tail is re-drawn with a hollow marker and no CI shading.
    ax.plot(cell_df["retained_fraction"], cell_df["pooled_mae"], color="steelblue", linewidth=1, zorder=1)
    ax.fill_between(healthy["retained_fraction"], healthy["ci_lo"], healthy["ci_hi"], color="steelblue", alpha=0.15)
    ax.plot(healthy["retained_fraction"], healthy["pooled_mae"], marker="o", linestyle="none",
            color="steelblue", label="pooled MAE (n_retained >= 30)", zorder=3)
    if len(degenerate):
        ax.plot(degenerate["retained_fraction"], degenerate["pooled_mae"], marker="o", linestyle="none",
                markerfacecolor="none", markeredgecolor="steelblue", markeredgewidth=1.5, markersize=8,
                label=f"pooled MAE (n_retained < {N_RETAINED_CI_FLOOR}, degenerate CI)", zorder=4)
    ax2.plot(cell_df["retained_fraction"], cell_df["share_retained_0_20_bin"], marker="s", color="firebrick", label="0-20 share")
    ax2.axhline(unfiltered_020_share, linestyle="--", color="firebrick", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("fraction of predictions retained")
    ax.set_ylabel("pooled MAE (cover points)", color="steelblue")
    ax2.set_ylabel("share of retained set in 0-20 bin", color="firebrick")
    ax.set_title(f"{model}\n{prompt}", fontsize=10)
    ax.invert_xaxis()
    ax.legend(loc="upper left", fontsize=6.5, framealpha=0.85)
fig.suptitle(
    "Confidence-threshold filtering: pooled MAE and the 0-20 cover-bin share move together\n"
    "Dashed red line = unfiltered 0-20 share. A falling blue line beside a rising red line is a\n"
    f"composition shift, not a pure accuracy gain. Hollow markers: n_retained < {N_RETAINED_CI_FLOOR} "
    "images, CI degenerate (not a precise estimate)",
    fontsize=9,
)
fig.tight_layout(rect=[0, 0, 1, 0.85])
fig.savefig(ROOT / "05_deliverables/Repository/GitHub/notebooks" / "rendered" / "Q5_confidence_filtering.png", dpi=150)
plt.show()

# %% [markdown]
# ## AUC top-set over the 24 configurations — replacing a single-winner claim
#
# The observed maximum AUC is **Gemma-3-27B x Detailed at 0.798**, with
# **Llama-4-Maverick x Grid-Overlay at 0.737** and **Qwen-2.5 x Short at
# 0.720** close behind — confirmed directly from `auc_by_combo_df` above, not
# assumed. Any statement of the form "configuration X is the single
# highest-discriminating configuration" is withdrawn: the maximum of 24
# correlated-by-construction statistics is upward-biased (the same
# winner's-curse problem the balanced-MAE top-set handles), and three configurations sit
# within 0.078 of each other against per-cell standard errors of roughly
# 0.02-0.03 on ~115 positives per cell — the maximum is not well identified.
#
# The construction mirrors Q1's top-set exactly, with the sign flipped
# because **higher** AUC is better here (Q1's balanced MAE is a
# lower-is-better statistic): for each configuration *j* of the 24,
# `D_j = max_{k!=j} theta_k - theta_j` (theta = AUC), so *j* is best iff
# `D_j <= 0`, and *j* may be excluded only when `D_j` is confidently above
# zero. The `max_{k!=j}` is **re-selected inside every bootstrap replicate**
# — never held at the observed winner — over the same 10,000 paired image
# resamples used for the family-G intervals above. Both the direct MCB bound
# and a Hansen MCS elimination are run on the same resamples; as in Q1, any
# difference between the two is reported as their
# **union**, with `topset_implementations_disagree = TRUE` set on every row
# and flagged as a defect rather than silently resolved.

# %%
ALL_COMBOS_Q5 = sorted(auc_by_combo_df.set_index(["model", "prompt"]).index.tolist())
N_COMBO_Q5 = len(ALL_COMBOS_Q5)
combo_labels_q5 = [f"{m}||{p}" for (m, p) in ALL_COMBOS_Q5]

# Every combo's (confidence, outlier, usable) triple is precomputed once as a
# plain numpy array aligned to a common image index (1,155 images), the same
# strategy Q1 uses for its balanced-MAE top-set: a per-replicate pandas
# groupby across 24 combos x 10,000 replicates is far too slow at that call
# volume, so the hot loop below never touches pandas.
IMAGES_Q5 = base_local["image"].unique()
_IMG_INDEX_Q5 = pd.Index(IMAGES_Q5)
_N_IMAGES_Q5 = len(_IMG_INDEX_Q5)

_CONF_MAT_Q5 = np.full((_N_IMAGES_Q5, N_COMBO_Q5), np.nan)
_OUTLIER_MAT_Q5 = np.zeros((_N_IMAGES_Q5, N_COMBO_Q5), dtype=int)
_USABLE_MAT_Q5 = np.zeros((_N_IMAGES_Q5, N_COMBO_Q5), dtype=bool)
for j, (m, p) in enumerate(ALL_COMBOS_Q5):
    cell = base_local[(base_local["model"] == m) & (base_local["prompt"] == p)].set_index("image")
    cell = cell.reindex(_IMG_INDEX_Q5)
    _CONF_MAT_Q5[:, j] = cell["confidence"].values
    _OUTLIER_MAT_Q5[:, j] = cell["outlier"].fillna(0).astype(int).values
    _USABLE_MAT_Q5[:, j] = ~cell["confidence_unusable"].fillna(True).values


def tie_aware_auc_from_arrays(conf: np.ndarray, lab: np.ndarray) -> float:
    """Same statistic and orientation as `tie_aware_auc`, computed directly
    from already-filtered numpy arrays (no scipy.mannwhitneyu call) so it can
    be run 24 x 10,000 times without per-call overhead dominating the cost.
    Ties are broken by mid-rank exactly as `stats.rankdata` does.
    """
    pos = conf[lab == 1]
    neg = conf[lab == 0]
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = stats.rankdata(conf)
    sum_ranks_pos = ranks[lab == 1].sum()
    # Standard Mann-Whitney identity: U_pos = sum_ranks_pos - n_pos*(n_pos+1)/2
    # is #{(i,j): pos_i > neg_j} + 0.5*#ties, i.e. exactly `u_pos_gt_neg` in
    # `tie_aware_auc` above. The same orientation flip is applied.
    u_pos_gt_neg = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return 1.0 - (u_pos_gt_neg / (n_pos * n_neg))


def auc_vector_for_images(drawn_images: np.ndarray) -> np.ndarray:
    """AUC for all 24 model x prompt combos at once, on a resampled
    (with-replacement) set of image IDs, operating on the precomputed arrays
    above. A drawn image contributes as many repeated rows as it was drawn;
    rows where that combo's confidence is unusable are excluded, exactly
    mirroring the complete-case AUC computed on the observed frame.
    """
    positions = _IMG_INDEX_Q5.get_indexer(drawn_images)
    conf_drawn = _CONF_MAT_Q5[positions]        # (n_drawn, N_COMBO_Q5)
    outlier_drawn = _OUTLIER_MAT_Q5[positions]
    usable_drawn = _USABLE_MAT_Q5[positions]
    out = np.empty(N_COMBO_Q5)
    for j in range(N_COMBO_Q5):
        mask = usable_drawn[:, j]
        out[j] = tie_aware_auc_from_arrays(conf_drawn[mask, j], outlier_drawn[mask, j])
    return out


theta_hat_q5 = auc_vector_for_images(IMAGES_Q5)
# Sanity check: the vectorised statistic above must reproduce the same
# per-combo AUCs already computed by `tie_aware_auc` (via `mannwhitneyu`) in
# `auc_by_combo_df`, on the observed (unresampled) frame.
_check_theta = auc_by_combo_df.set_index(["model", "prompt"]).loc[ALL_COMBOS_Q5, "auc"].values
assert np.allclose(theta_hat_q5, _check_theta, atol=1e-9), (
    "vectorised top-set AUC does not match the tie_aware_auc/mannwhitneyu observed statistic"
)

B_MCB_Q5 = c.B_BOOTSTRAP
boot_theta_q5 = np.empty((B_MCB_Q5, N_COMBO_Q5))
for b in range(B_MCB_Q5):
    drawn = rng_topset.choice(IMAGES_Q5, size=len(IMAGES_Q5), replace=True)
    boot_theta_q5[b] = auc_vector_for_images(drawn)

best_counts_q5 = np.zeros(N_COMBO_Q5)
for b in range(B_MCB_Q5):
    best_counts_q5[np.argmax(boot_theta_q5[b])] += 1
prob_best_q5 = best_counts_q5 / B_MCB_Q5


def d_vector_max_better(theta: np.ndarray) -> np.ndarray:
    """D_j = max_{k!=j} theta_k - theta_j, the higher-is-better mirror of
    Q1's d_vector (which uses min because lower balanced MAE is better).
    j is best iff D_j <= 0."""
    n = len(theta)
    d = np.empty(n)
    for j in range(n):
        others = np.delete(theta, j)
        d[j] = others.max() - theta[j]
    return d


D_hat_q5 = d_vector_max_better(theta_hat_q5)
D_star_q5 = np.apply_along_axis(d_vector_max_better, 1, boot_theta_q5)
se_hat_q5 = D_star_q5.std(axis=0, ddof=1)
se_hat_q5_safe = np.where(se_hat_q5 > 0, se_hat_q5, np.nan)

tail_stats_q5 = (D_star_q5 - D_hat_q5[None, :]) / se_hat_q5_safe[None, :]
max_tail_q5 = np.nanmax(tail_stats_q5, axis=1)
c_crit_q5 = float(np.percentile(max_tail_q5, 95))

mcb_lower_bound_q5 = D_hat_q5 - c_crit_q5 * se_hat_q5_safe
mcb_topset_mask_q5 = mcb_lower_bound_q5 <= 0
print(f"MCB critical value c = {c_crit_q5:.4f}")
print(f"MCB top-set size: {int(mcb_topset_mask_q5.sum())} / {N_COMBO_Q5}")


def mcs_elimination_q5(D_star_full: np.ndarray, labels: list[str], alpha: float = 0.05) -> set[str]:
    remaining = list(range(len(labels)))
    while len(remaining) > 1:
        sub_theta_hat = theta_hat_q5[remaining]
        sub_theta_star = boot_theta_q5[:, remaining]
        d_hat_sub = d_vector_max_better(sub_theta_hat)
        d_star_sub = np.apply_along_axis(d_vector_max_better, 1, sub_theta_star)
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


mcs_topset_q5 = mcs_elimination_q5(D_star_q5, combo_labels_q5)
mcb_topset_q5 = {combo_labels_q5[j] for j in range(N_COMBO_Q5) if mcb_topset_mask_q5[j]}
union_topset_q5 = mcb_topset_q5 | mcs_topset_q5
symmetric_diff_q5 = mcb_topset_q5.symmetric_difference(mcs_topset_q5)
disagree_q5 = len(symmetric_diff_q5) > 0

print(f"MCS top-set size: {len(mcs_topset_q5)} / {N_COMBO_Q5}")
print(f"MCB top-set size: {len(mcb_topset_q5)} / {N_COMBO_Q5}")
print(f"Union top-set size: {len(union_topset_q5)} / {N_COMBO_Q5}")
print(f"topset_implementations_disagree = {disagree_q5} (symmetric difference: {len(symmetric_diff_q5)} combos)")
if disagree_q5:
    print("Disagreeing combos:", sorted(symmetric_diff_q5))

observed_best_idx_q5 = int(np.argmax(theta_hat_q5))
observed_best_key_q5 = ALL_COMBOS_Q5[observed_best_idx_q5]

topset_rows_q5 = []
for j, label in enumerate(combo_labels_q5):
    model_j, prompt_j = ALL_COMBOS_Q5[j]
    k8_cell = k8_combo_df[(k8_combo_df["model"] == model_j) & (k8_combo_df["prompt"] == prompt_j)].iloc[0]
    topset_rows_q5.append({
        "question_id": "Q5",
        "model": model_j, "prompt": prompt_j,
        "auc": theta_hat_q5[j],
        "prob_best": prob_best_q5[j],
        "D_hat": D_hat_q5[j], "se_hat": se_hat_q5[j],
        "mcb_lower_bound": mcb_lower_bound_q5[j],
        "in_mcb_topset": label in mcb_topset_q5,
        "in_mcs_topset": label in mcs_topset_q5,
        "in_union_topset": label in union_topset_q5,
        "topset_implementations_disagree": disagree_q5,
        "mcb_critical_value": c_crit_q5,
        "is_observed_best": (model_j, prompt_j) == observed_best_key_q5,
        "k8_degenerate": bool(k8_cell["degenerate"]),
        "selection_conditioned": True,
    })
topset_bootstrap_q5 = pd.DataFrame(topset_rows_q5).sort_values("auc", ascending=False).reset_index(drop=True)
print(topset_bootstrap_q5[["model", "prompt", "auc", "in_union_topset", "is_observed_best", "k8_degenerate"]].to_string(index=False))

# %% [markdown]
# **What this licenses, and nothing more.** Membership in or exclusion from
# the 95% top-set above — never a directional pairwise claim ("Gemma-3-27B x
# Detailed discriminates better than Qwen-2.5 x Short" is not licensed at any
# p-value, because the reference is data-selected). No p-value is attached and
# this construction enters no family; family G's six tests of AUC = 0.5 are
# untouched. A configuration flagged K8-degenerate may still appear in the
# top-set, and the flag travels with its row — it still does not support a
# ranking interpretation on its own.

# %%
fig, ax = plt.subplots(figsize=(11, 6))
plot_df_q5 = topset_bootstrap_q5.copy()
plot_df_q5["label"] = plot_df_q5["model"] + " / " + plot_df_q5["prompt"]
colors_q5 = plot_df_q5["in_union_topset"].map({True: "seagreen", False: "lightgray"})
ax.scatter(range(len(plot_df_q5)), plot_df_q5["auc"], c=colors_q5, s=40, zorder=3)
ax.plot(range(len(plot_df_q5)), plot_df_q5["auc"], color="gray", linewidth=0.5, zorder=2)
ax.axhline(0.5, linestyle="--", color="grey", linewidth=1)
ax.set_xticks(range(len(plot_df_q5)))
ax.set_xticklabels(plot_df_q5["label"], rotation=90, fontsize=7)
ax.set_ylabel("AUC (confidence as classifier for |e| > 20)")
ax.set_title(
    f"24 model x prompt configurations, sorted by AUC (higher is better)\n"
    f"green = the {int(plot_df_q5['in_union_topset'].sum())} configurations in the 95% top-set "
    f"(union of MCB and MCS), indistinguishable from the best; gray = excluded"
)
fig.tight_layout()
fig.savefig(ROOT / "05_deliverables/Repository/GitHub/notebooks" / "rendered" / "Q5_topset_ranking.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# **What to look for.** The observed best (Gemma-3-27B x Detailed) sits at
# the top, but if the green cluster contains several configurations,
# read the finding as "these configurations cannot be distinguished from the
# best at 95%" — not as a ranking of 24. A wide top-set here is the expected,
# honest answer given the per-cell standard errors (roughly 0.02-0.03) and the
# count of positives per cell, not a sign the construction failed.

# %% [markdown]
# ## Write results
#
# Every contrast-shaped row below carries the full result schema so that
# `stream_note`, the clustered-companion columns and the Holm columns are
# never silently dropped from an output CSV, even where a column is not
# applicable to this estimand (e.g. Q5 has no `p_companion_*` Wilcoxon stream
# in the family-G sense — AUC is a ranking statistic, not a paired-difference
# one — those columns are recorded as `NaN` with the reason in `stream_note`
# rather than omitted).

# %%
STREAM_NOTE_Q5 = (
    "p_primary_* = family-G bootstrap p on (mean AUC - 0.5), Holm-adjusted within family G "
    "(size 6). No paired-difference companion stream exists for a ranking statistic (AUC), "
    "so p_companion_* is NaN by construction, not by omission. Cut-free companion is "
    "tie-corrected Spearman(confidence, |e|), reported separately with no p-value in this family. "
    "AUC orientation: AUC = P(confidence on a random non-outlier > confidence on a random outlier), "
    "so AUC > 0.5 means low confidence usefully flags outliers ('informative') and AUC < 0.5 means "
    "confidence is systematically higher on the images the model gets worst ('anti-informative') — "
    "see result_direction for the Holm-adjusted verdict per model. A directional verdict is downgraded "
    "to 'label-dependent / not robustly classified' if the cut-free Spearman companion disagrees in "
    "sign, a relabelling check collapses the departure from chance by half or more or reverses its "
    "sign, or the feeding cells are K8-degenerate; see not_robustly_classified."
)

r2_rows = []
for _, row in auc_by_model_df.iterrows():
    model = row["model"]
    clustered_row = clustered_df.loc[
        (clustered_df["model"] == model) & (clustered_df["row_type"] == "headline")
    ].iloc[0]
    contrast_row = c.make_contrast_row(
        estimate=row["mean_auc"],
        ci_lo=row["ci_lo_delta"] + 0.5,
        ci_hi=row["ci_hi_delta"] + 0.5,
        p_primary_raw=row["p_bootstrap_raw"],
        p_primary_holm=row["p_bootstrap_holm"],
        p_companion_raw=np.nan,
        p_companion_holm=np.nan,
        p_clustered_webb=clustered_row["p_clustered_webb"],
        n_distinct_abs_t_star=clustered_row["n_distinct_abs_t_star"],
        loco_min=clustered_row["loco_min"],
        loco_max=clustered_row["loco_max"],
        ci_width_ratio_clustered_to_headline=np.nan,
        stream_note=STREAM_NOTE_Q5,
        question_id="Q5",
        model=model,
        family=FAMILY,
        family_size=FAMILY_SIZE,
        holm_first_threshold=row["holm_first_threshold"],
        n_excluded_unusable_confidence=row["n_excluded_unusable_confidence"],
        n_total_rows=row["n_total_rows"],
        k8_degenerate=row["k8_degenerate"],
        n_empty_bin_violations=row["n_empty_bin_violations"],
        loco_note=clustered_row["loco_note"],
        p_primary_at_floor=bool(row["p_at_floor"]),
        n_pos=row["n_pos"], n_neg=row["n_neg"],
        result_direction=row["result_direction_robust"],
        result_direction_unadjusted_for_robustness=row["result_direction"],
        not_robustly_classified=bool(row["not_robustly_classified"]),
    )
    r2_rows.append(contrast_row)

q5_auc_by_model_df = pd.DataFrame(r2_rows)
q5_auc_by_model_df.to_csv(RESULTS_DIR / "Q5_auc_by_model.csv", index=False)

auc_by_combo_out = auc_by_combo_df.copy()
auc_by_combo_out["question_id"] = "Q5"
auc_by_combo_out["stream_note"] = (
    "24-cell model x prompt sub-analysis, pre-declared; reported with CI and no "
    "p-value by design, so it informs the family-G headline without inflating it. The 15 "
    "between-model comparisons are not planned and are not computed here."
)

op_points_out = op_points_df.copy()
op_points_out["question_id"] = "Q5"
op_points_out.to_csv(RESULTS_DIR / "Q5_roc_operating_points.csv", index=False)

missingness_out = missingness_df.copy()
missingness_out["question_id"] = "Q5"
missingness_out["outlier_threshold"] = OUTLIER_THRESHOLD

label_robustness_out = label_robustness_df.copy()
label_robustness_out["question_id"] = "Q5"
label_robustness_out["row_type"] = "per_model_summary"

label_robustness_cells_out = label_robustness_cells_df.copy()
label_robustness_cells_out["question_id"] = "Q5"
label_robustness_cells_out["row_type"] = "per_prompt_cell_detail"

label_robustness_full = pd.concat(
    [label_robustness_out, label_robustness_cells_out], ignore_index=True, sort=False
)
label_robustness_full.to_csv(RESULTS_DIR / "Q5_label_robustness.csv", index=False)

clustered_out = clustered_df.copy()
clustered_out["question_id"] = "Q5"
clustered_out["family"] = FAMILY
clustered_out.to_csv(RESULTS_DIR / "Q5_clustered_companion.csv", index=False)

# Q5 writes six output CSVs. The tie-corrected Spearman companion belongs with
# the AUC-by-combo sub-analysis, since both are reported per model x prompt
# with a CI and no p-value, and the Scout CoT-resolved sensitivity check is a
# robustness note on the missingness bounds (a differently sourced subset of
# the same unusable-versus-usable confidence question A13 answers). Both are
# therefore appended to the file they belong with rather than written out as a
# seventh and eighth CSV.
spearman_out = spearman_df.copy()
spearman_out["question_id"] = "Q5"
spearman_out["row_type"] = "spearman_confidence_vs_abs_e"
auc_by_combo_out["row_type"] = "auc_by_model_x_prompt"
combo_and_spearman = pd.concat([auc_by_combo_out, spearman_out], ignore_index=True, sort=False)
combo_and_spearman.to_csv(RESULTS_DIR / "Q5_auc_by_combo.csv", index=False)

scout_cot_sensitivity_out = scout_cot_sensitivity.copy()
scout_cot_sensitivity_out["question_id"] = "Q5"
scout_cot_sensitivity_out["row_type"] = "scout_cot_resolved_sensitivity"
missingness_out["row_type"] = "mnar_bounds"
missingness_and_scout = pd.concat([missingness_out, scout_cot_sensitivity_out], ignore_index=True, sort=False)
missingness_and_scout.to_csv(RESULTS_DIR / "Q5_missingness_bounds.csv", index=False)

# Confidence-threshold filtering sweep (single-configuration arm): the 24
# model x prompt cells and the 6 per-model aggregates, plus the per-bin
# MAE-change table at the pre-declared representative cutoff — all
# descriptive, no p-value, no family, per the block above.
filtering_out = pd.concat([filtering_df, model_filtering_df], ignore_index=True, sort=False)
filtering_out["row_type_perbin_change"] = np.nan
perbin_change_out = perbin_change_df.copy()
filtering_full = pd.concat([filtering_out, perbin_change_out], ignore_index=True, sort=False)
filtering_full.to_csv(RESULTS_DIR / "Q5_confidence_filtering.csv", index=False)

# AUC top-set over the 24 configurations, replacing any single-winner claim.
topset_bootstrap_q5.to_csv(RESULTS_DIR / "Q5_auc_topset.csv", index=False)

print("Written:")
for fname in ["Q5_auc_by_model.csv", "Q5_auc_by_combo.csv", "Q5_roc_operating_points.csv",
              "Q5_missingness_bounds.csv", "Q5_label_robustness.csv", "Q5_clustered_companion.csv",
              "Q5_confidence_filtering.csv", "Q5_auc_topset.csv", "Q5_assumption_checks.csv"]:
    print(" -", fname)

q5_auc_by_model_df
