# %% [markdown]
# # Q6 — Does the capture device affect estimates?
#
# This notebook does not run a statistical test. It demonstrates, structurally,
# that the question cannot be answered from this data — and shows exactly why,
# so a reader does not have to take that on faith.
#
# Two phones were used to collect the 1,155 photographs in this frame: a Samsung
# Galaxy S25 Ultra and a Samsung SM-S908E. Every Galaxy image was taken in one
# field campaign (`campaign_3`) and every SM-S908E image was taken in the
# other two (`campaign_2`, `campaign_1`). No image exists where the two
# devices photographed the same campaign. That is not a sampling limitation to
# be modelled around — it is a **rank deficiency** in the design: `device` is a
# deterministic function of `campaign`, so a linear model cannot assign a
# coefficient to one without it absorbing (or being absorbed by) the other.
#
# The honest output of this question is therefore a demonstration, not a
# p-value. Four things are shown, in order: the crosstab that makes the
# confounding visible cell by cell; the design-matrix rank that makes it
# algebraically exact; a numerical check that the "device effect" and the
# "campaign_3 vs. rest" campaign contrast are the same number to machine
# precision; and a look at why even that campaign contrast is not clean on its
# own. No test is run because none would be valid — reporting a p-value here
# would let a reader conclude "we tested for a device effect and found none",
# which is a different and false claim next to the true one: **this data cannot
# distinguish a device effect from a campaign effect, in either direction, at
# any sample size.**

# %% [markdown]
# ## Setup
#
# The seed is fixed for reproducibility even though Q6 runs no test and
# reports no interval — `_common.compute_maverick_floor` is called below
# purely to check assertion C11, and, being shared machinery, it accepts an
# `rng` regardless. Nothing here tests a hypothesis, so no Holm family, no
# correction and no family size apply — see the "Multiple comparisons"
# section below for why.

# %%
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 20260913  # _common.BASE_SEED (20260907) + question number 6, written as a
# plain integer rather than an expression so the seed that governs this run can
# be read straight off the source.
rng = np.random.default_rng(SEED)

# `_common.py` lives beside this notebook. The working directory is not
# guaranteed to be the notebooks folder, and `__file__` is not defined inside
# an executed notebook cell, so that folder is located the same way
# `_common.project_root()` locates the project root: via the
# ANALYSIS_PROJECT_ROOT environment variable when it is set, falling back to
# walking up from the current working directory (an interactive session) to
# find STATUS.md, then descending into the notebooks folder.
_p = Path(os.environ.get("ANALYSIS_PROJECT_ROOT", Path.cwd())).resolve()
for _candidate in [_p, *_p.parents]:
    if (_candidate / "STATUS.md").is_file():
        sys.path.insert(0, str(_candidate / "05_deliverables/Repository/GitHub/notebooks"))
        break
else:
    raise RuntimeError(f"project root not found from {_p}")

import pathlib as _pl  # `_common.py` is imported from this notebook's
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))  # own directory
import _common as C

ROOT = C.project_root()

# %% [markdown]
# ## Data
#
# `D1` (the 1,155-image reference frame, which carries `campaign`) and `D3`
# (capture device per photograph) are loaded through `_common`'s guarded
# readers, and every one of `_common`'s load-time assertions is run —
# including `A9`, the assertion that checks the device x campaign crosstab
# cell by cell (not on collapsed totals), and `K1`/`C11`, which are not
# reachable from `run_all_assertions()` and so must be called explicitly
# here.

# %%
assumption_df, frames = C.run_all_assertions(include_d12=False)
d1 = frames["d1"]
d3 = frames["d3"]
d5 = frames["d5"]
base_local = frames["base_local"]
d8 = frames["d8"]

print(f"D1: {len(d1)} images, D3: {len(d3)} device records, base/local frame: {len(base_local)} rows")

# A9 is already included in run_all_assertions(); shown explicitly here because
# it is the load-bearing check for this notebook specifically.
a9 = C.assert_a9(d1, d3)
print(f"A9 passed: {a9.passed} — {a9.detail}")

# K1 and C11 are not reachable from run_all_assertions() and are called
# explicitly so their guarantees are actually checked, not merely assumed
# available.
k1 = C.assert_k1_pairing_complete(d5, base_local)
print(f"K1 passed: {k1.passed} — {k1.detail}")

# C11 only needs `n_reproduced`, which `compute_maverick_floor` computes
# before it ever touches the bootstrap — `n_reproduced` does not depend on
# `B` at all. Q6 makes no Maverick claim and has no use for the B=10,000
# reproducibility-floor interval, so B=1 is passed to avoid spending 10,000
# resamples (and advancing `rng`) on a bootstrap this notebook never reads.
maverick_floor = C.compute_maverick_floor(d8, rng, B=1)
c11 = C.assert_c11_n_reproduced(maverick_floor.n_reproduced)
print(f"C11 passed: {c11.passed} — {c11.detail}")

assumption_rows = [assumption_df, pd.DataFrame([vars(k1), vars(c11)])]
assumption_checks = pd.concat(assumption_rows, ignore_index=True)

if not (a9.passed and k1.passed and c11.passed):
    raise C.AssertionFailed(
        "A9, K1 or C11 failed — stopping per '(a failed assertion stops the "
        "notebook)'. If A9 specifically failed, device would have become "
        "partially estimable and this question must go back to the user "
        "before anything is estimated."
    )

# %% [markdown]
# ## Assumption checks
#
# Q6 runs no test, so there are no test assumptions (normality, independence
# of residuals, etc.) to check in the usual sense. The one assumption that
# matters here is structural rather than distributional: **is the crosstab
# really 2x3 with three empty cells?** That is exactly what A9 checks, cell by
# cell, and it has already been confirmed above. If A9 ever failed on a future
# data refresh, device would no longer be perfectly confounded with campaign,
# the design matrix below would regain full rank, and this question would
# become partially estimable — at which point it needs a stated test and a
# stated correction family, rather than being quietly re-analysed here.

# %% [markdown]
# ## 1. The crosstab: three non-zero cells, three structural zeros
#
# **Why this comes first.** A reader grasps three zero cells in a
# 2x3 table faster than a rank statement, and the table is the same fact the
# design-matrix argument formalises below. `D3.v2` (`device`) has two levels;
# `D1.v2` (`campaign`) has three. The full 2x3 table is built and reported,
# never collapsed to device totals. A device total, such as the 1,001 images
# the SM-S908E contributes across two campaigns, is not a cell: totals alone
# are consistent with the two devices sharing a campaign, so they cannot show
# the three empty cells that carry the whole argument. A9 therefore checks the
# table cell by cell, and its docstring states the six counts it expects.

# %%
device_campaign = d3.merge(d1[["image", "campaign"]], on="image", how="inner", validate="one_to_one")
crosstab = pd.crosstab(device_campaign["device"], device_campaign["campaign"])
# fixed column/row order for a stable, readable table
campaign_order = [C.CAMPAIGN_2, C.CAMPAIGN_1, C.CAMPAIGN_3]
device_order = [C.DEVICE_GALAXY, C.DEVICE_SM]
crosstab = crosstab.reindex(index=device_order, columns=campaign_order, fill_value=0)
print(crosstab)

n_nonzero_cells = int((crosstab.values > 0).sum())
n_zero_cells = int((crosstab.values == 0).sum())
print(f"non-zero cells: {n_nonzero_cells} of 6, structural zeros: {n_zero_cells} of 6")

crosstab_long = crosstab.reset_index().melt(id_vars="device", var_name="campaign", value_name="n_images")
crosstab_long["cell_type"] = np.where(crosstab_long["n_images"] > 0, "observed", "structural_zero")
crosstab_long

# %% [markdown]
# ### Chart — the crosstab as a heatmap
#
# **What to look for:** a checkerboard pattern, not a gradient. Each device
# occupies entirely different columns (campaigns) from the other. If device
# had any images in common with the other device's campaigns, this figure
# would show a filled cell off the diagonal-like pattern; it does not, on any
# of the three campaigns.

# %%
fig, ax = plt.subplots(figsize=(7, 3.2))
mat = crosstab.values.astype(float)
im = ax.imshow(mat, cmap="Blues", vmin=0)
ax.set_xticks(range(len(campaign_order)))
ax.set_xticklabels(campaign_order, rotation=15, ha="right")
ax.set_yticks(range(len(device_order)))
ax.set_yticklabels(device_order)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        val = int(mat[i, j])
        label = str(val) if val > 0 else "0\n(structural)"
        ax.text(j, i, label, ha="center", va="center",
                color="white" if val > 300 else "black", fontsize=10)
ax.set_xlabel("Campaign (D1.v2)")
ax.set_ylabel("Capture device (D3.v2)")
ax.set_title("Device x campaign: 3 non-zero cells, 3 structural zeros (A9)")
fig.colorbar(im, ax=ax, label="n images")
fig.tight_layout()
fig.savefig(ROOT / "05_deliverables/Repository/GitHub/notebooks" / "rendered" / "Q6_crosstab.png", dpi=150)
plt.show()

# %% [markdown]
# ## 2. The algebra: the design matrix is rank-deficient
#
# **Why this step matters.** The crosstab shows the confound descriptively;
# this step shows it is not merely a small-sample coincidence but an exact,
# unavoidable linear-algebraic fact. Build one row per image with three
# campaign indicator (one-hot) columns and one device indicator column —
# **four columns in total** (the three campaign dummies already form a
# complete one-hot basis over the 3 campaign levels, so no separate intercept
# column is added — adding one would only introduce the ordinary
# dummy-variable-trap redundancy on top of the one this step is isolating,
# which is not the redundancy Q6 is about) — and compute the rank of that
# matrix. If device carried any information not already contained in
# campaign, the four columns would be linearly independent and the matrix
# would have rank 4. It does not.

# %%
design = d1[["image", "campaign"]].merge(d3[["image", "device"]], on="image", how="inner", validate="one_to_one")
assert len(design) == 1155

campaign_dummies = pd.get_dummies(design["campaign"], prefix="campaign").astype(float)
device_dummy = (design["device"] == C.DEVICE_GALAXY).astype(float).rename("device_is_galaxy")

X = pd.concat([campaign_dummies, device_dummy], axis=1)
n_columns = X.shape[1]
rank = int(np.linalg.matrix_rank(X.to_numpy()))

print(f"design matrix: {X.shape[0]} rows x {n_columns} columns")
print(f"columns: {list(X.columns)}")
print(f"rank: {rank}")

# The device column is a deterministic function of the campaign columns:
# device_is_galaxy == campaign_campaign_3 on every row, exactly.
device_equals_campaign_indicator = bool(
    (device_dummy.to_numpy() == campaign_dummies["campaign_" + C.CAMPAIGN_3].to_numpy()).all()
)
print(f"device_is_galaxy == campaign_campaign_3 on every row: {device_equals_campaign_indicator}")

rank_deficiency = n_columns - rank
print(f"rank deficiency: {n_columns} columns, rank {rank} -> {rank_deficiency} redundant dimension(s)")

# %% [markdown]
# The three campaign dummies alone are already full column rank (3 of 3) — a
# complete one-hot basis over the 3 campaign levels has no dummy-variable-trap
# redundancy to begin with, since no intercept column competes with it here.
# So the single rank deficiency found above (4 columns, rank 3) is entirely
# the device column adding no new direction: it is identical to the
# `campaign_campaign_3` column, so **no estimator — not OLS, not a
# robust variant, not a Bayesian model with any prior that respects the data
# exactly as observed — can separate a device coefficient from a campaign
# coefficient in a model that contains both.** Dropping campaign from the
# model does not rescue device: it does not create missing information, it
# only relabels the campaign contrast as if it were a device contrast. This is
# why Q6 is not "estimated with a caveat" — there is no model specification,
# however chosen, under which the two effects separate.

# %% [markdown]
# ## 3. The numerical demonstration: the "device effect" IS the campaign contrast
#
# **Why this step, given that steps 1 and 2 already prove the point
# algebraically.** A rank argument can feel abstract to a reader who wants to
# see it in the actual numbers this project reports. So: compute the
# "device effect" (Galaxy images vs. SM-S908E images) and, separately, the
# campaign contrast (campaign_3 vs. the other two campaigns combined),
# from **two independently derived partitions of the same 1,155 images** —
# one read off `D3.v2` (device), one read off `D1.v2` (campaign) — and assert
# they agree to machine precision. They must, because the two partitions are
# identical row for row (step 2 above), but the point of doing it twice is
# that the assertion can actually fail if the two source columns ever
# disagree, which a check that compares a number to itself cannot.
#
# **Metric, and why it is overall MAE rather than balanced MAE.** Balanced MAE
# is reported per campaign (and so, here, per device) only where every one of
# the five cover bins holds at least 10 images of that campaign, because a bin
# thinner than that gives its weight in the average to a handful of
# photographs. The Galaxy arm of this contrast, which is exactly the
# `campaign_3` campaign, does not clear that bar: its per-bin image
# counts are 117 / 17 / 7 / 8 / **5**, three bins under the threshold. Overall
# MAE is reported instead, with no bin weighting and no minimum-n rule, and it
# demonstrates the identity in step 2 exactly as well, since it is still
# computed from the same partition.
#
# **No confidence interval and no p-value, deliberately.**
# This number is not a test of anything Q6 asks — its only possible null
# ("the campaign_3-vs-rest contrast is zero") is a hypothesis about
# *campaign*, which Q6 does not pose and which is in no Holm family. The
# stronger, already-computed demonstration is the bit-for-bit column equality
# in step 2 (`device_column_equals_campaign_indicator`); an interval on this
# number would invite exactly the "no device effect was detected" misreading
# this notebook's opening cell says it exists to prevent. `ci_low`/`ci_high`
# are therefore left blank for this row, deliberately, not because they were
# not computed.

# %%
frame = base_local.copy()
frame["is_galaxy_by_campaign"] = frame["campaign"] == C.CAMPAIGN_3  # partition read off D1.v2

# Independent second partition, read off D3.v2 (device) rather than D1.v2
# (campaign), so the equality asserted below is a real check on two source
# columns rather than a comparison of one number to itself.
device_frame = frame.merge(d3[["image", "device"]], on="image", how="left", validate="many_to_one")
frame["is_galaxy_by_device"] = (device_frame["device"] == C.DEVICE_GALAXY).to_numpy()

device_split_matches_campaign_split = bool(
    (frame["is_galaxy_by_device"] == frame["is_galaxy_by_campaign"]).all()
)
print(f"device-column partition identical to campaign-column partition, row for row: {device_split_matches_campaign_split}")


def overall_mae_diff(f: pd.DataFrame, partition_col: str) -> float:
    """Galaxy-arm overall MAE minus rest-arm overall MAE, unweighted: no
    per-bin weighting and no minimum-n rule, the fallback used when a
    campaign has fewer than 10 images in some cover bin."""
    galaxy = f[f[partition_col]]
    rest = f[~f[partition_col]]
    if galaxy["image"].nunique() == 0 or rest["image"].nunique() == 0:
        return np.nan
    return float(galaxy["abs_e"].mean() - rest["abs_e"].mean())


device_diff = overall_mae_diff(frame, "is_galaxy_by_device")
campaign_diff = overall_mae_diff(frame, "is_galaxy_by_campaign")
print(f"'device' contrast (Galaxy - SM-S908E), overall MAE, partition from D3.v2: {device_diff:.10f}")
print(f"'campaign' contrast (campaign_3 - rest), overall MAE, partition from D1.v2: {campaign_diff:.10f}")

# The two independently derived contrasts must agree to machine precision.
# This assertion can fail, unlike a value compared to itself,
# if the device and campaign columns were ever not in exact row-for-row
# agreement, which step 2 showed they are.
assert np.isclose(device_diff, campaign_diff, atol=1e-12, rtol=0), (
    "device contrast and campaign contrast disagree beyond machine precision "
    "— step 2's column-equality claim would be false"
)
machine_precision_equal = bool(np.isclose(device_diff, campaign_diff, atol=1e-12, rtol=0))
print(f"equal to machine precision, two independently-derived partitions: {machine_precision_equal}")
observed_diff = device_diff  # the single number carried into the output row

galaxy_per_bin_n = C.per_bin_n_table(frame[frame["is_galaxy_by_device"]])
print(f"Galaxy-arm (= campaign_3) per-bin image n: {galaxy_per_bin_n} — "
      f"three bins below S0.5's n>=10 rule, hence overall MAE rather than balanced MAE above")

# %% [markdown]
# ## 4. Why it is not even a clean campaign contrast
#
# **Why this matters beyond the identifiability point.** Even setting aside
# that "device" cannot be separated from "campaign", the campaign contrast
# itself is not a single clean confound — it is several confounds stacked.
# `campaign_3` differs from the other two campaigns in phone optics,
# site, season, date and observer (C3), and, concretely, in what it
# photographed: its per-bin composition is not the same as the other two
# campaigns'. This is reported here because a reviewer will ask about the two
# phones, and pre-empting that with the crosstab, the rank finding and this
# compositional check is stronger than waiting to be asked.

# %%
per_campaign_bin_counts = (
    d1.assign(bin=C.assign_bins(d1["reference"]))
    .groupby(["campaign", "bin"])["image"]
    .nunique()
    .reindex(pd.MultiIndex.from_product([campaign_order, C.BIN_LABELS], names=["campaign", "bin"]), fill_value=0)
    .reset_index(name="n_images")
)
per_campaign_bin_wide = per_campaign_bin_counts.pivot(index="campaign", columns="bin", values="n_images").reindex(
    index=campaign_order, columns=C.BIN_LABELS
)
per_campaign_bin_pct = per_campaign_bin_wide.div(per_campaign_bin_wide.sum(axis=1), axis=0) * 100
print(per_campaign_bin_wide)
print()
print(per_campaign_bin_pct.round(1))

campaign_3_top_bin_n = int(per_campaign_bin_wide.loc[C.CAMPAIGN_3, "80-100"])
print(f"\ncampaign_3 contributes {campaign_3_top_bin_n} images to the 80-100 bin.")

# %% [markdown]
# ### Chart — per-campaign bin composition
#
# **What to look for:** whether the bars for `campaign_3` (the Galaxy
# campaign) have a different shape from the other two, not just a different
# height. A different shape means the "device" contrast in step 3 is also
# partly a scene-composition contrast, on top of everything else C3 already
# lists.

# %%
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(C.BIN_LABELS))
width = 0.25
for i, camp in enumerate(campaign_order):
    ax.bar(x + (i - 1) * width, per_campaign_bin_pct.loc[camp].to_numpy(), width, label=camp)
ax.set_xticks(x)
ax.set_xticklabels(C.BIN_LABELS)
ax.set_xlabel("Reference cover bin (%, right-closed)")
ax.set_ylabel("Share of that campaign's images (%)")
ax.set_title("Per-campaign bin composition — device is confounded with scene composition too")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "05_deliverables/Repository/GitHub/notebooks" / "rendered" / "Q6_campaign_composition.png", dpi=150)
plt.show()

# %% [markdown]
# ## Multiple comparisons
#
# **Q6 contributes zero tests to any Holm family.** This question is a
# statement that device is not identifiable, so it holds no test and no
# p-value, and it sits in none of the correction families the other questions
# share. No Holm correction applies here, no
# threshold is spent, and no p-value is computed anywhere in this notebook:
# the one quantity that could have carried one (the overall-MAE contrast in
# step 3) is reported as a point value only, by design, because its only
# possible null is a hypothesis about campaign, not about device.

# %% [markdown]
# ## Result
#
# **Device cannot be separated from campaign in this data, at any sample
# size.** The crosstab has three non-zero cells and three structural zeros;
# the 4-column design matrix built from campaign and device has rank 3, one
# short of full column rank, because the device column is an exact linear
# function of the campaign columns; and the "device effect" computed on
# overall MAE is numerically identical to the campaign_3-vs-rest campaign
# contrast, because they are the same partition of images under two names —
# verified from two independently derived partitions, not a value compared to
# itself.
# Dropping campaign from a model does not recover device — it only relabels
# the campaign effect as if it were a device effect, and even that relabelled
# contrast is not clean, because `campaign_3` differs from the other two
# campaigns in scene composition (5 images in the 80-100 bin) as well as in
# phone optics, site, season, date and observer.
#
# **What this establishes, and what it does not.** It establishes that no
# amount of additional data collected under the current design — more images
# from these same three campaigns — would change the answer, because the
# non-identifiability is structural, not a power problem. It does not
# establish that the two phones produce similar or different estimates; that
# question is simply not reachable from these data, in either direction.
#
# **What would have been needed instead.** Answering this question would
# require the two phones photographing overlapping quadrats within the same
# campaign — the same scenes, the same site, the same season, ideally the
# same day, captured on both devices — so that a device contrast could be
# formed while holding campaign fixed. No such overlap exists anywhere in
# this frame. This is the single most useful sentence Q6 produces for anyone
# designing the next data-collection campaign, and it is worth more than any
# number this question could otherwise have reported.

# %%
crosstab_out = crosstab_long.copy()
crosstab_out.insert(0, "question_id", "Q6")
(ROOT / "05_deliverables/Repository/GitHub/results").mkdir(parents=True, exist_ok=True)
crosstab_out.to_csv(ROOT / "05_deliverables/Repository/GitHub/results" / "Q6_device_campaign_crosstab.csv", index=False)
crosstab_out

# %%
identifiability_rows = [
    {
        "question_id": "Q6",
        "statistic": "design_matrix_shape",
        "value": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n": len(design),
        "test": "none (structural demonstration)",
        "detail": f"{n_columns} columns (3 campaign dummies + 1 device dummy, no separate intercept)",
    },
    {
        "question_id": "Q6",
        "statistic": "design_matrix_rank",
        "value": float(rank),
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n": len(design),
        "test": "none (structural demonstration)",
        "detail": f"rank {rank} of {n_columns} columns -> device column lies in the span of the campaign columns",
    },
    {
        "question_id": "Q6",
        "statistic": "device_column_equals_campaign_indicator",
        "value": float(device_equals_campaign_indicator),
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n": len(design),
        "test": "none (structural demonstration)",
        "detail": "device_is_galaxy == campaign_campaign_3 on every row, exactly",
    },
    {
        "question_id": "Q6",
        "statistic": "NOT_A_DEVICE_EFFECT_overall_mae_diff",
        "value": float(observed_diff),
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n": int(frame["image"].nunique()),
        "test": "none (descriptive; identical partition to the device split)",
        "detail": (
            "overall MAE, Galaxy-arm (= campaign_3) minus rest, cover points; "
            "labelled NOT_A_DEVICE_EFFECT because it is numerically identical to the "
            "campaign_3-vs-rest campaign contrast, verified above from two independently "
            "derived partitions (D3.v2 and D1.v2); overall MAE used rather than balanced MAE "
            "because the Galaxy arm's per-bin image n (117/17/7/8/5) fails S0.5's n>=10 rule "
            "in three of five bins; no CI or p-value is reported because its only possible "
            "null is a hypothesis about campaign, which Q6 does not pose and which is in no "
            "Holm family — an interval here would invite the false 'no device effect detected' "
            "reading this notebook exists to prevent"
        ),
    },
    {
        "question_id": "Q6",
        "statistic": "campaign_3_top_bin_n",
        "value": float(campaign_3_top_bin_n),
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n": int(per_campaign_bin_wide.loc[C.CAMPAIGN_3].sum()),
        "test": "none (descriptive composition check)",
        "detail": "campaign_3's 80-100 bin count; the campaign contrast also confounds scene composition",
    },
    {
        "question_id": "Q6",
        "statistic": "holm_family_tests_contributed",
        "value": 0.0,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n": len(design),
        "test": "n/a",
        "detail": "Q6 contributes zero tests to any Holm family; no correction applies (see 'Multiple comparisons' section table row for Q6: 0 tests)",
    },
]
identifiability_out = pd.DataFrame(identifiability_rows)
identifiability_out.to_csv(ROOT / "05_deliverables/Repository/GitHub/results" / "Q6_identifiability_demonstration.csv", index=False)
identifiability_out

# %%
assumption_checks_path = ROOT / "05_deliverables/Repository/GitHub/results" / "Q6_assumption_checks.csv"
assumption_checks.to_csv(assumption_checks_path, index=False)
print(f"wrote {assumption_checks_path}")
assumption_checks
