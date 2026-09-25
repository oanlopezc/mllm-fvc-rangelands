# %% [markdown]
# # Q0 — the classical vegetation-index baseline, computed from the rectified photographs
#
# Every baseline number reported elsewhere in this study — the per-bin MAE, the
# signed bias, the over-prediction count, the two illustrative exemplar images —
# comes from one file, `traditional_baseline.csv`: one predicted cover value per
# image, 1,155 rows. This notebook is what produces that file. It reads the
# rectified quadrat photographs directly and computes the prediction from pixel
# colour alone; nothing downstream is taken on faith.
#
# **The method, in one paragraph.** Each rectified photograph (already
# perspective-corrected to 1536x1536 pixels around the quadrat frame) is
# converted to chromatic coordinates — each pixel's R, G, B divided by that
# pixel's own R+G+B sum, which removes overall illumination level so the index
# below responds to hue rather than brightness. On those chromatic values the
# excess-green-minus-excess-red index (Meyer & Neto, 2008), ExG-ExR =
# (2G-R-B) - (1.4R-G) = 3G-2.4R-B, is computed for every pixel. A 40-pixel
# border is excluded on every side before anything is measured, because the
# rectification warps the quadrat's physical frame to sit right at the crop's
# edge, and frame-coloured pixels are not vegetation or soil. A single
# soil/vegetation threshold is then calibrated once, from 200 images that are
# themselves near-bare ground by the human reference (<=1% cover) — pixel
# colour evidence only, the FVC labels of the 1,155 target images are never
# consulted — and every image's prediction is the percentage of its interior
# pixels whose index exceeds that threshold.
#
# **Why a calibrated threshold rather than a fixed constant.** ExG-ExR has no
# universal zero point that separates "soil" from "vegetation" across
# different soils, cameras and lighting; a threshold fixed by convention would
# be calibrated to nothing observed in this dataset. Drawing the threshold
# from this study's own bare-soil exemplars ties it to the actual soil colour
# in these photographs, at the cost of depending on which images the reference
# labels put at or below 1% cover and which pixels happen to be drawn from
# them — both draws are seeded below so the dependency is at least
# reproducible.
#
# **This notebook computes predictions only.** No statistical test, interval
# or model comparison is run here — that is the classical-baseline notebook's
# job, downstream of this file. This notebook's only responsibility is to
# reproduce, exactly, the method by which the 1,155 predictions were derived.

# %% [markdown]
# ## Setup
#
# Two independent random draws happen during threshold calibration — which
# 200 bare-soil images are drawn, and which pixels are sampled from each —
# and both are seeded so this notebook reproduces the same threshold and the
# same 1,155 predictions on every run. The two draws use separate generators
# (Python's own `random` for the image draw, NumPy for the pixel draw)
# because that is what the reference implementation this notebook reproduces
# used, and matching it means matching its random-number sources as well as
# its seed values.

# %%
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


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

SEED = 20260913  # Written as a plain integer, not an expression, so the seed
# that governs this run can be read straight off the source. This
# module-level seed governs nothing in the computation below directly. The two
# draws that matter are `IMAGE_SEED` and `PIXEL_SEED`, fixed separately
# immediately below, because both must match the reference method's own seed
# values rather than a value derived from SEED.
rng = np.random.default_rng(SEED)

ROOT = co.project_root()
RAW = ROOT / "01_input" / "raw"
RECTIFIED_DIR = RAW / "rectified"

# --- pipeline constants, fixed by the method this notebook reproduces -------
BORDER_PX = 40                   # excluded on every side of the 1536x1536 rectified crop
SOIL_EXEMPLAR_MAX_FVC = 1.0      # an image qualifies as a bare-soil exemplar at <= 1% reference cover
N_SOIL_EXEMPLARS = 200           # how many eligible exemplars are drawn
MAX_PIXELS_PER_EXEMPLAR = 3000   # cap on interior pixels sampled from each exemplar
THRESHOLD_PERCENTILE = 99        # the pooled soil distribution's percentile used as the threshold
IMAGE_SEED = 13                  # Python `random`: which 200 exemplars are drawn
PIXEL_SEED = 13                  # NumPy: which pixels within each exemplar are drawn

print(f"numpy {np.__version__}, pandas {pd.__version__}")
print(f"seed={SEED} (image_seed={IMAGE_SEED}, pixel_seed={PIXEL_SEED})")
print(f"project root: {ROOT}")
print(f"rectified images: {RECTIFIED_DIR}")

# %% [markdown]
# ## Data
#
# The reference cover value for every image comes from `All_frame1155.csv`
# (the study's authoritative reference frame), read the same way every other
# notebook in this project reads it. The rectified photographs themselves are
# read directly from disk, one at a time, since the pipeline below needs
# pixel data rather than a table.

# %%
d1 = co.load_d1()  # columns: image, reference, campaign (plus others not used here)
refs = d1[["image", "reference"]].copy()
refs["reference"] = pd.to_numeric(refs["reference"], errors="coerce")
n_before_dropna = len(refs)
refs = refs.dropna(subset=["reference"]).reset_index(drop=True)
print(f"{len(refs)}/{n_before_dropna} images with a usable reference value")

all_image_paths = sorted(RECTIFIED_DIR.glob("*.jpg"))
print(f"{len(all_image_paths)} rectified image files found on disk")

# %% [markdown]
# ## Assumption checks
#
# - **Every image on disk has a reference value, and vice versa.** The
#   classical index needs no held-out data and no model fitting against
#   cover, so completeness of the image/reference join is the only thing
#   that could silently shrink the 1,155-image scope this baseline is
#   supposed to cover.
# - **The rectified crop is exactly 1536x1536.** The 40-pixel border and the
#   soil-exemplar pixel cap are both calibrated to that geometry; a different
#   size would mean the interior region excludes a different fraction of the
#   frame than intended.
# - **The threshold-calibration draw is reproducible from the stated seeds.**
#   Checked by recomputing it below and reporting the resulting threshold
#   next to the reference run's own value — not by tuning anything to match.

# %%
assumption_rows = []

merged_check = refs[["image"]].merge(
    pd.DataFrame({"image": [p.name for p in all_image_paths]}), on="image", how="outer", indicator=True,
)
n_unmatched = int((merged_check["_merge"] != "both").sum())
assumption_rows.append({
    "id": "image_reference_join_complete",
    "description": "every rectified image file has a reference value and every referenced image has a file on disk",
    "passed": n_unmatched == 0 and len(all_image_paths) == 1155 and len(refs) == 1155,
    "detail": f"n_images_on_disk={len(all_image_paths)}, n_with_reference={len(refs)}, n_unmatched={n_unmatched}",
})

_sample_img = np.array(Image.open(all_image_paths[0]).convert("RGB"))
assumption_rows.append({
    "id": "rectified_geometry",
    "description": "rectified crop is 1536x1536 pixels, matching the geometry the 40px border and pixel cap assume",
    "passed": _sample_img.shape[:2] == (1536, 1536),
    "detail": f"sampled shape (checked on {all_image_paths[0].name}) = {_sample_img.shape}",
})

for row in assumption_rows:
    print(f"[{'PASS' if row['passed'] else 'FAIL'}] {row['id']}: {row['detail']}")
if not all(r["passed"] for r in assumption_rows):
    raise AssertionError("a load-time assumption check failed — stopping before the (slow) pixel pipeline runs")

# %% [markdown]
# ## Pipeline functions
#
# The vegetation index and its interior-region extraction. Everything below
# calls only these two functions plus plain NumPy/pandas.

# %%
def normalize_rgb(rgb_uint8: np.ndarray) -> np.ndarray:
    """Chromatic normalisation: each pixel's R, G, B divided by that pixel's
    own R+G+B sum, with a zero sum (a pure-black pixel) replaced by 1 to
    avoid dividing by zero. This removes overall illumination level from the
    index below, so two patches of the same hue under different brightness
    score the same.
    """
    rgb = rgb_uint8.astype(np.float64)
    total = rgb.sum(axis=-1, keepdims=True)
    total = np.where(total == 0, 1.0, total)
    return rgb / total


def compute_exg_exr(rgb_norm: np.ndarray) -> np.ndarray:
    """ExG - ExR = (2G - R - B) - (1.4R - G) = 3G - 2.4R - B (Meyer & Neto, 2008)."""
    r, g, b = rgb_norm[..., 0], rgb_norm[..., 1], rgb_norm[..., 2]
    exg = 2 * g - r - b
    exr = 1.4 * r - g
    return exg - exr


def interior_index_values(path: Path, border: int = BORDER_PX) -> np.ndarray:
    """ExG-ExR values for one image's interior pixels only. A `border`-pixel
    margin is excluded on every side because the rectification warps the
    quadrat's physical frame to sit right at the crop's edge; without this
    exclusion, frame-coloured pixels would be measured as if they were soil
    or vegetation, in both the threshold calibration and the final
    prediction.
    """
    rgb = np.array(Image.open(path).convert("RGB"))
    idx = compute_exg_exr(normalize_rgb(rgb))
    return idx[border:-border, border:-border].ravel()


print("Pipeline functions defined.")

# %% [markdown]
# ## Threshold calibration
#
# 200 images at or below 1% reference cover are drawn as bare-soil exemplars,
# sorted by filename first so the draw is reproducible from the seed alone —
# no external ordering (such as the row order of a reference spreadsheet) is
# available or used here. Up to 3,000 interior pixels are sampled from each
# exemplar, the sampled pixels from all 200 images are pooled, and the 99th
# percentile of that pool is the threshold. This is pixel-colour evidence
# only: no image outside this pool of 200, and no cover value of any of the
# 1,155 target images, enters the calibration.

# %%
eligible_names = sorted(refs.loc[refs["reference"] <= SOIL_EXEMPLAR_MAX_FVC, "image"].tolist())
eligible_paths = [RECTIFIED_DIR / n for n in eligible_names if (RECTIFIED_DIR / n).exists()]
n_eligible_missing_file = len(eligible_names) - len(eligible_paths)

random.seed(IMAGE_SEED)
exemplars = random.sample(eligible_paths, min(N_SOIL_EXEMPLARS, len(eligible_paths)))
print(f"{len(eligible_paths)} images eligible (reference <= {SOIL_EXEMPLAR_MAX_FVC}%), "
      f"{len(exemplars)} drawn (image_seed={IMAGE_SEED}); "
      f"{n_eligible_missing_file} eligible images had no file on disk")

# %%
np.random.seed(PIXEL_SEED)
pooled_soil = []
t0 = time.time()
for i, path in enumerate(exemplars, 1):
    vals = interior_index_values(path)
    if len(vals) > MAX_PIXELS_PER_EXEMPLAR:
        vals = np.random.choice(vals, MAX_PIXELS_PER_EXEMPLAR, replace=False)
    pooled_soil.append(vals)
    if i % 50 == 0:
        print(f"  calibration {i}/{len(exemplars)} images  ({time.time() - t0:.0f}s)")
pooled_soil = np.concatenate(pooled_soil)

THRESHOLD = float(np.percentile(pooled_soil, THRESHOLD_PERCENTILE))
print(f"\nsoil-exemplar pixel pool: n={pooled_soil.size:,}, mean={pooled_soil.mean():.4f}, "
      f"std={pooled_soil.std():.4f}")
print(f"threshold (soil p{THRESHOLD_PERCENTILE}) = {THRESHOLD:.10f}")

# %% [markdown]
# ## Predict cover for every image
#
# For each of the 1,155 rectified images, the prediction is the percentage of
# interior pixels whose ExG-ExR value exceeds the calibrated threshold. This
# is the slowest step in this notebook (1,155 images at 1536x1536), so
# progress is printed every 200 images.

# %%
rows = []
t0 = time.time()
for i, path in enumerate(all_image_paths, 1):
    vals = interior_index_values(path)
    rows.append({
        "image": path.name,
        "n_interior_px": int(vals.size),
        "vegetation_percent": 100.0 * float((vals > THRESHOLD).sum()) / vals.size,
    })
    if i % 200 == 0:
        print(f"  prediction {i}/{len(all_image_paths)} images  ({time.time() - t0:.0f}s)")

pred_df = pd.DataFrame(rows).merge(refs, on="image", how="left")
n_before_join = len(pred_df)
pred_df = pred_df.dropna(subset=["reference"]).reset_index(drop=True)
print(f"\n{len(pred_df)}/{n_before_join} images matched to a reference value "
      f"({time.time() - t0:.0f}s total)")

if len(pred_df) != 1155:
    raise AssertionError(f"expected 1,155 predicted images joined to a reference, got {len(pred_df)}")

# %% [markdown]
# ## Cross-check against the reference implementation
#
# A reference run of this exact method, made outside this project against the
# same rectified images, recorded a threshold, a pooled MAE and bias, five
# per-bin MAE values, and two illustrative exemplar images. Because the exact
# ordering that reference run drew its 200 exemplars from does not survive
# into this project (see the note on the sorted exemplar list above), this
# run's draw is not guaranteed to be identical, pixel for pixel, to that
# run's draw — so it is compared here, not assumed to match, and any gap is
# reported rather than absorbed. Nothing below is tuned to close a gap; the
# threshold and every downstream number are taken exactly as computed above.

# %%
REFERENCE_THRESHOLD = -0.10284167025464387
REFERENCE_POOLED_MAE = 10.967321608271373
REFERENCE_POOLED_BIAS = -10.13097486174213
REFERENCE_PER_BIN_MAE = {"0-20": 3.4082, "20-40": 25.9529, "40-60": 45.9339, "60-80": 58.2462, "80-100": 78.6048}
REFERENCE_BEST_IMAGE = {"image": "IMG_20250112_151917.jpg", "vegetation_percent": 45.6555}
REFERENCE_WORST_IMAGE = {"image": "IMG_20250115_143343.jpg", "vegetation_percent": 0.4716}
PER_BIN_MAE_TOLERANCE = 0.5  # cover points; a larger gap means the exemplar draw diverged more than expected

pred_df["e"] = pred_df["vegetation_percent"] - pred_df["reference"]
pred_df["abs_e"] = pred_df["e"].abs()
pred_df["bin"] = co.assign_bins(pred_df["reference"])

pooled_mae = float(pred_df["abs_e"].mean())
pooled_bias = float(pred_df["e"].mean())
per_bin_mae = pred_df.groupby("bin")["abs_e"].mean().reindex(co.BIN_LABELS)

threshold_diff = THRESHOLD - REFERENCE_THRESHOLD
pooled_mae_diff = pooled_mae - REFERENCE_POOLED_MAE
pooled_bias_diff = pooled_bias - REFERENCE_POOLED_BIAS

print(f"threshold:    this run={THRESHOLD:.10f}  reference run={REFERENCE_THRESHOLD:.10f}  diff={threshold_diff:+.10f}")
print(f"pooled MAE:   this run={pooled_mae:.6f}  reference run={REFERENCE_POOLED_MAE:.6f}  diff={pooled_mae_diff:+.6f}")
print(f"pooled bias:  this run={pooled_bias:.6f}  reference run={REFERENCE_POOLED_BIAS:.6f}  diff={pooled_bias_diff:+.6f}")

per_bin_check_rows = []
for label in co.BIN_LABELS:
    this_val = float(per_bin_mae.loc[label])
    ref_val = REFERENCE_PER_BIN_MAE[label]
    diff = this_val - ref_val
    per_bin_check_rows.append({
        "bin": label, "this_run_mae": this_val, "reference_run_mae": ref_val,
        "diff": diff, "within_tolerance": abs(diff) <= PER_BIN_MAE_TOLERANCE,
    })
per_bin_check_df = pd.DataFrame(per_bin_check_rows)
print("\nPer-bin MAE cross-check (tolerance = 0.5 cover points):")
print(per_bin_check_df.to_string(index=False))

per_bin_all_within_tolerance = bool(per_bin_check_df["within_tolerance"].all())
if not per_bin_all_within_tolerance:
    print("\nAt least one bin's MAE differs from the reference run by more than 0.5 cover points. "
          "This is reported as a finding — the exemplar draw diverged more than expected — and is "
          "not corrected or tuned away.")
else:
    print("\nEvery bin's MAE agrees with the reference run to within 0.5 cover points.")

# %%
best_row_check = pred_df.loc[pred_df["image"] == REFERENCE_BEST_IMAGE["image"]]
worst_row_check = pred_df.loc[pred_df["image"] == REFERENCE_WORST_IMAGE["image"]]

if len(best_row_check) == 1:
    v = float(best_row_check["vegetation_percent"].iloc[0])
    print(f"best dense example {REFERENCE_BEST_IMAGE['image']}: this run predicted {v:.4f}, "
          f"reference run predicted {REFERENCE_BEST_IMAGE['vegetation_percent']}, "
          f"diff={v - REFERENCE_BEST_IMAGE['vegetation_percent']:+.4f}")
else:
    print(f"best dense example {REFERENCE_BEST_IMAGE['image']} not found in this run's predictions")

if len(worst_row_check) == 1:
    v = float(worst_row_check["vegetation_percent"].iloc[0])
    print(f"worst dense example {REFERENCE_WORST_IMAGE['image']}: this run predicted {v:.4f}, "
          f"reference run predicted {REFERENCE_WORST_IMAGE['vegetation_percent']}, "
          f"diff={v - REFERENCE_WORST_IMAGE['vegetation_percent']:+.4f}")
else:
    print(f"worst dense example {REFERENCE_WORST_IMAGE['image']} not found in this run's predictions")

# %% [markdown]
# ## Every image processed, every image joined, the reference copy checked
#
# Two completeness checks are run here: all 1,155 images processed, and every
# one of them joined to a reference FVC value. A
# third check compares the `reference` column this notebook is about to write
# against an **independently re-read copy of `All_frame1155.csv`** — read
# again from disk here, not the `d1` frame already held in memory — so the
# comparison is not against the exact same values this column was sourced
# from in the first place. That independent re-read is what makes this check
# capable of catching a real divergence, rather than comparing `d1` to
# itself, which cannot fail regardless of what the pipeline computed.

# %%
n_processed = len(pred_df)
n_joined = int(pred_df["reference"].notna().sum())

_d1_independent = pd.read_csv(RAW / "All_frame1155.csv").rename(
    columns={"Filename": "image", "Veg %": "reference"}
)[["image", "reference"]]
ref_check = pred_df[["image", "reference"]].merge(
    _d1_independent, on="image", suffixes=("_out", "_d1_reread"),
)
n_reference_mismatch = int((~np.isclose(ref_check["reference_out"], ref_check["reference_d1_reread"])).sum())

assumption_rows.append({
    "id": "all_images_processed",
    "description": "all 1,155 rectified images produced a prediction",
    "passed": n_processed == 1155,
    "detail": f"n_processed={n_processed}",
})
assumption_rows.append({
    "id": "all_images_joined_to_reference",
    "description": "every processed image joined to exactly one reference value",
    "passed": n_joined == 1155,
    "detail": f"n_joined={n_joined}",
})
assumption_rows.append({
    "id": "reference_matches_independent_reread",
    "description": "the reference column about to be written agrees with an independently re-read copy of "
                    "All_frame1155.csv (read again from disk, not the in-memory d1 frame it was sourced from)",
    "passed": n_reference_mismatch == 0,
    "detail": f"n_mismatch={n_reference_mismatch} of {len(ref_check)} compared",
})
assumption_rows.append({
    "id": "threshold_cross_check",
    "description": "calibrated threshold compared against the reference implementation's own run (not tuned to match)",
    "passed": True,  # informational: a differing threshold is a finding, not a failure
    "detail": f"this_run={THRESHOLD:.10f}, reference_run={REFERENCE_THRESHOLD:.10f}, diff={threshold_diff:+.10f}",
})
assumption_rows.append({
    "id": "per_bin_mae_cross_check",
    "description": "per-bin MAE agrees with the reference run to within 0.5 cover points, per bin",
    "passed": per_bin_all_within_tolerance,
    "detail": per_bin_check_df.to_dict(orient="records"),
})

for row in assumption_rows[2:]:
    print(f"[{'PASS' if row['passed'] else 'FAIL'}] {row['id']}: {row['detail']}")

# %% [markdown]
# ## Chart — this run's threshold against the reference run's soil-pixel evidence
#
# **What to look for.** The distribution of pooled soil-exemplar pixel values
# this run drew, with this run's threshold and the reference run's threshold
# both marked. The two vertical lines sitting close together is what
# "reproduces the reference method" looks like in this figure; a visible gap
# between them is the visible form of whatever the cross-check table above
# already reports numerically.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(pooled_soil, bins=80, color="saddlebrown", alpha=0.75)
ax.axvline(THRESHOLD, color="black", linestyle="-", linewidth=1.5,
           label=f"this run's threshold ({THRESHOLD:.4f})")
ax.axvline(REFERENCE_THRESHOLD, color="firebrick", linestyle="--", linewidth=1.5,
           label=f"reference run's threshold ({REFERENCE_THRESHOLD:.4f})")
ax.set_xlabel("ExG-ExR index value (chromatic-normalised interior pixels, bare-soil exemplars)")
ax.set_ylabel("pixel count")
ax.set_title(f"Pooled bare-soil pixel distribution used for threshold calibration\n"
             f"n={pooled_soil.size:,} pixels from {len(exemplars)} images at or below "
             f"{SOIL_EXEMPLAR_MAX_FVC}% reference cover")
ax.legend(fontsize=8)
fig.tight_layout()
RENDERED_DIR = ROOT / "05_deliverables/Repository/GitHub/notebooks" / "rendered"
RENDERED_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(RENDERED_DIR / "Q0_threshold_calibration.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Chart — predicted against reference, all 1,155 images
#
# **What to look for.** Every image's prediction from this run against its
# reference cover, with the 1:1 line for context. This is the same
# relationship the classical-baseline notebook analyses in full; here it
# serves only as a visual check that this run's output looks like the
# baseline it is supposed to reproduce, not a substitute for that notebook's
# own analysis.

# %%
fig, ax = plt.subplots(figsize=(6.5, 6.5))
colors = np.where(pred_df["bin"] == "0-20", "steelblue", "firebrick")
ax.scatter(pred_df["reference"], pred_df["vegetation_percent"], c=colors, s=12, alpha=0.6)
lims = [0, max(pred_df["reference"].max(), pred_df["vegetation_percent"].max()) * 1.02]
ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="1:1 line")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("reference cover (%)")
ax.set_ylabel("predicted cover, this run (%)")
ax.set_title(f"This run's predictions vs. reference, all {len(pred_df)} images")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(RENDERED_DIR / "Q0_prediction_vs_reference.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Keeping the copy of `traditional_baseline.csv` as supplied
#
# This notebook writes `traditional_baseline.csv`, and a copy of that file may
# already sit in `01_input/raw/` when it runs. The cell below copies whatever
# is on disk to `00_context/traditional_baseline_supplied.csv`, with a short
# note on where it came from, so the copy as supplied is kept alongside the
# one this notebook computes. The copy happens only when no preserved file is
# there yet, so running this notebook again leaves the preserved copy
# untouched.

# %%
_preserved_path = ROOT / "00_context" / "traditional_baseline_supplied.csv"
_preserved_note_path = ROOT / "00_context" / "traditional_baseline_supplied.README.md"

if _preserved_path.exists():
    print(f"{_preserved_path} already exists — leaving it untouched (guard against re-run clobbering it).")
else:
    import shutil
    shutil.copy2(RAW / "traditional_baseline.csv", _preserved_path)
    _preserved_note_path.write_text(
        "# traditional_baseline_supplied.csv\n\n"
        "This is the `traditional_baseline.csv` that was present in `01_input/raw/` before "
        "`Q0_baseline_index.py` ran for the first time, preserved here so it is not lost when "
        "that notebook overwrites the working copy with its own recomputed predictions.\n\n"
        "## Provenance\n\n"
        "Named in `00_context/traditional_baseline.README.md` as coming from "
        "`TraditionalBaseline/notebooks/Rectified_Image_Run.ipynb`. It is not the seeded "
        "reference implementation kept in `00_context/paper-baseline-reference/`: measured "
        "directly against that implementation's own per-image output "
        "(`paper_baseline_per_image.csv`), 1,154 of this file's 1,155 predictions differ "
        "from the seeded run's, so this file cannot be the seeded run's draw.\n\n"
        "## Replacement\n\n"
        "Replaced on 2026-09-14 by `Q0_baseline_index.py`, which recomputes the same method "
        "with both random draws seeded and reproducible.\n"
    )
    print(f"Preserved the supplied file to {_preserved_path}")
    print(f"Wrote provenance note to {_preserved_note_path}")

# %% [markdown]
# ## Result
#
# This notebook computes the classical ExG-ExR baseline from the 1,155
# rectified photographs, using pixel-colour evidence only for threshold
# calibration and never consulting the target images' own reference cover
# values. The calibrated threshold and the pooled and per-bin error against
# the reference implementation's own run are reported in the cross-check
# section above; any bin whose MAE differs from that run by more than 0.5
# cover points is flagged there rather than adjusted to close the gap, since
# the two runs cannot be guaranteed to draw the identical 200 exemplar images
# (the reference run's own draw order is not available in this project — see
# the note on the sorted exemplar list above).
#
# The output is one row per image — `image`, `vegetation_percent`, `reference`
# — written in that column order so it can be read downstream exactly as the
# existing baseline file was.

# %%
out_df = pred_df[["image", "vegetation_percent", "reference"]].copy()
out_path = RAW / "traditional_baseline.csv"
out_df.to_csv(out_path, index=False)
print(f"Wrote {len(out_df)} rows to {out_path}")

# %%
q0_assumption_checks = pd.DataFrame([
    {"question_id": "Q0", "id": r["id"], "description": r["description"], "passed": r["passed"],
     "detail": r["detail"]}
    for r in assumption_rows
])

q0_summary = pd.DataFrame([{
    "question_id": "Q0",
    "statistic": "threshold",
    "value": THRESHOLD,
    "ci_low": np.nan,
    "ci_high": np.nan,
    "n": len(pred_df),
    "test": "99th percentile of pooled bare-soil exemplar pixels (no statistical test)",
}, {
    "question_id": "Q0",
    "statistic": "pooled_mae",
    "value": pooled_mae,
    "ci_low": np.nan,
    "ci_high": np.nan,
    "n": len(pred_df),
    "test": "descriptive (no interval; this notebook computes predictions only)",
}, {
    "question_id": "Q0",
    "statistic": "pooled_mean_bias",
    "value": pooled_bias,
    "ci_low": np.nan,
    "ci_high": np.nan,
    "n": len(pred_df),
    "test": "descriptive (no interval; this notebook computes predictions only)",
}])

RESULTS_DIR = ROOT / "05_deliverables/Repository/GitHub/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
q0_assumption_checks.to_csv(RESULTS_DIR / "Q0_assumption_checks.csv", index=False)
q0_summary.to_csv(RESULTS_DIR / "Q0_baseline_index_summary.csv", index=False)

print("\nWritten:")
print(f"  01_input/raw/traditional_baseline.csv  ({len(out_df)} rows)")
print(f"  results/Q0_assumption_checks.csv")
print(f"  results/Q0_baseline_index_summary.csv")
q0_summary
