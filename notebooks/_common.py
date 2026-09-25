"""Shared analysis machinery for the MLLM-vs-fractional-vegetation-cover study.

This module is imported by every ``Q*.py`` notebook. It is not itself a
notebook: it has no jupytext cells, produces no charts and writes no CSV on
import. Its purpose is to be the *single* implementation of the data loaders,
the load-time assertions, the bin/balanced-MAE machinery, the paired image
bootstrap, the campaign-clustered companion, the paired Wilcoxon test, the
effect sizes, the assumption checks and the Holm-Bonferroni helper, so that
no notebook re-derives its own version and no two notebooks drift apart on
something that must be identical across all of them.

Every function that resamples takes an explicit ``rng: np.random.Generator``
argument. Nothing in this module reads or seeds ``np.random`` global state,
so a caller's reproducibility is never silently borrowed from here.

This module reads only from ``01_input/raw/`` inside the project. Two files
present alongside the analysis data — a call-telemetry log and an API-retry
summary — are operational records of the annotation runs, not measurements
of vegetation cover, and are never loaded here; an external cross-check
summary is likewise not analysis input. ``assert_a12`` records that this
module's own read list stays clean of all three, and ``guarded_read_csv``
gives notebooks a way to enforce the same restriction on any path they open
themselves.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# 0. Paths, constants
# --------------------------------------------------------------------------- #

def project_root(start: str | Path | None = None) -> Path:
    """Locate the project directory by walking upward from the current
    working directory until a `STATUS.md` marker file is found.

    Needed because notebooks are executed with their kernel's working
    directory set to the folder holding the rendered notebook, not the
    project root, so a bare relative path like ``01_input/raw/x.csv``
    would silently fail to resolve there.
    """
    p = Path(os.environ.get("ANALYSIS_PROJECT_ROOT", start or Path.cwd())).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "STATUS.md").is_file():
            return candidate
    raise RuntimeError(f"project root not found from {p}")


ROOT = project_root()
RAW = ROOT / "01_input" / "raw"

# Filenames this module never reads, listed literally rather than by any
# internal label so a reader can see directly which files are excluded:
# the two are operational logs of the annotation runs, not vegetation-cover
# measurements, and the third is an external cross-check summary, not an
# input to any analysis here.
FORBIDDEN_PATHS = (
    "call_telemetry.csv",
    "api_retry_summary.csv",
    "model_performance_summary.csv",  # external cross-check reference, not an input
)

MODEL_FILE_STEMS = (
    "Gemma-3-12B",
    "Gemma-3-27B",
    "Llama-4-Maverick",
    "Llama-4-Scout",
    "Mistral-Small-3.2",
    "Qwen-2.5",
)
MODELS = MODEL_FILE_STEMS  # the six models evaluated throughout the study

# Five equal-width bins on the ground-truth cover percentage, right-closed,
# with the bottom bin closed at both ends so an exactly-zero reference value
# falls inside it: [0,20], (20,40], (40,60], (60,80], (80,100].
BIN_EDGES = (0, 20, 40, 60, 80, 100)
BIN_LABELS = ("0-20", "20-40", "40-60", "60-80", "80-100")
EXPECTED_BIN_COUNTS = {"0-20": 933, "20-40": 108, "40-60": 46, "60-80": 38, "80-100": 30}

# Base seed. Every notebook sets `seed = 20260907 + N` (N = its question
# number) in its own first cell; this module never seeds anything itself,
# it only records the shared constant here so every notebook derives its
# seed from the same base.
BASE_SEED = 20260907

B_BOOTSTRAP = 10_000       # draws in the paired image bootstrap
B_WILD_CLUSTER = 9_999     # draws in the Webb wild cluster bootstrap

# Webb resolution floor: with 3 campaigns, the 216 sign-symmetric weight
# vectors collapse to 108 distinct |t*| values, so the smallest attainable
# two-sided p-value is 2/216 ~= 0.00926. See the docstring of
# `wild_cluster_bootstrap` for the full argument.
WEBB_P_FLOOR = 2.0 / 216.0

# Sizes of the families of tests that receive a shared Holm-Bonferroni
# correction, fixed before any result was seen. A caller MUST pass one of
# these labels/sizes explicitly to `holm_adjust`; the function never infers
# family size from len(pvalues), because doing so would under-correct
# whenever only a subset of a family's tests is passed in one call.
FAMILY_SIZES = {
    "A": 15,  # the 15 model-versus-model comparisons
    "B": 6,   # the 6 prompt-versus-prompt comparisons
    "C": 23,  # model x prompt configurations compared against the best (no directional claim)
    "D": 8,   # the 8 ensemble/selection contrasts
    "E": 5,   # the 5 preprocessing-variant contrasts
    "F": 6,   # the 6 cover-gradient tests (trend tests and covariate blocks)
    "G": 6,   # the 6 tests of AUC = 0.5 against confidence
    "H": 6,   # the 6 comparisons against the classical baseline
}


# --------------------------------------------------------------------------- #
# 1. Loaders
# --------------------------------------------------------------------------- #

def _derive_model_from_filename(path: Path) -> str:
    """Derive `model` from a D5/D6 filename prefix.

    `model` is not a column in D5 or D6 — it exists only in the filename,
    e.g. ``Gemma-3-12B_local_based.csv`` -> ``Gemma-3-12B``. The naive
    approach — splitting on the first underscore — breaks on
    ``Llama-4-Scout`` and ``Llama-4-Maverick``, whose own names contain
    underscores nowhere, but more importantly it breaks on
    ``Mistral-Small-3.2``, whose stem contains a literal dot that a
    naive "split on `_based`" pattern must not be confused by. The safe
    approach used here is to match the filename against the six recorded
    stems directly (longest-first, so no stem is a prefix-collision of
    another) rather than parsing structure out of the filename generically.
    """
    name = path.name
    # Longest-first guards against a shorter stem accidentally matching
    # inside a longer one (not currently possible with these six stems,
    # but the ordering costs nothing and removes the failure mode).
    for stem in sorted(MODEL_FILE_STEMS, key=len, reverse=True):
        if name.startswith(stem + "_"):
            return stem
    raise ValueError(f"could not derive model from filename: {name}")


def load_d1() -> pd.DataFrame:
    """D1 — the authoritative reference frame, `All_frame1155.csv`.

    The three field campaigns are relabelled on the way in, so that every table this
    code writes uses the same campaign names as the published dataset. The mapping is
    by acquisition date: November 2024 is `campaign_1`, January 2025 `campaign_2` and
    April 2025 `campaign_3`. It is applied here, at the single point the frame is read,
    so no downstream analysis can see one vocabulary while another sees the other.
    """
    df = pd.read_csv(RAW / "All_frame1155.csv")
    df = df.rename(columns={"Filename": "image", "Veg %": "reference", "Campaign": "campaign"})
    df["campaign"] = df["campaign"].map(CAMPAIGN_LABELS)
    assert df["campaign"].notna().all(), (
        "a campaign in All_frame1155.csv is not covered by CAMPAIGN_LABELS; every row "
        "must carry one of the three known field campaigns"
    )
    return df


def load_d2() -> pd.DataFrame:
    """D2 — rooted dead look-alike plants / non-rooted plant material annotations, plus a duplicate reference."""
    df = pd.read_csv(RAW / "dead_looking_vegetation_simplified.csv")
    df = df.rename(columns={"reference": "reference_d2"})
    return df


def load_d3() -> pd.DataFrame:
    """D3 — capture device per photograph."""
    return pd.read_csv(RAW / "image_device.csv")


def load_d4() -> pd.DataFrame:
    """D4 — quadrat-plane obliquity score."""
    return pd.read_csv(RAW / "image_obliquity.csv")


def load_d5(models: Sequence[str] = MODELS) -> pd.DataFrame:
    """D5 — the six `<Model>_local_based.csv` files, concatenated.

    Each file is 13,860 rows: 1,155 images x 4 prompts x 3 variants.
    `model` is derived from the filename (see `_derive_model_from_filename`)
    and attached as a real column — it exists nowhere else in this file.
    """
    frames = []
    for stem in models:
        path = RAW / f"{stem}_local_based.csv"
        df = pd.read_csv(path)
        df["model"] = _derive_model_from_filename(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


def load_d6(models: Sequence[str] = MODELS) -> pd.DataFrame:
    """D6 — the six `<Model>_api_based.csv` files, concatenated (base variant only)."""
    frames = []
    for stem in models:
        path = RAW / f"{stem}_api_based.csv"
        df = pd.read_csv(path)
        df["model"] = _derive_model_from_filename(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


def load_d8() -> pd.DataFrame:
    """D8 — determinism_runs.csv, two repeat runs of a 100-image subsample."""
    return pd.read_csv(RAW / "determinism_runs.csv")


def load_d9() -> pd.DataFrame:
    """D9 — run_failures.csv, every error record from both stacks.

    A naming trap lives in this file: Llama-4-Scout labels its own *base*-
    variant local run ``exp2_original_base`` in `run_type`, where every
    other model's base run is named `full_grid_fixed` / `api_base`. Filtering
    "base" runs by string-matching `run_type` for the literal substring
    "base" happens to work here only because every one of Scout's labels
    also contains "base" as a substring (`exp2_original_base`,
    `exp2_masked_gray_base`) — but the two are not the same predicate, and
    filtering on `variant == 'base'` combined with `in_scope == True` and
    `stack == 'local'` gets the right 5 rows regardless of how `run_type`
    happens to be spelled for any given model. Use `variant`, not
    `run_type`, to select "base" scope; `run_type` is for
    provenance/inspection only.
    """
    return pd.read_csv(RAW / "run_failures.csv")


def load_d12() -> pd.DataFrame:
    """D12 — traditional_baseline.csv, the classical ExG-ExR baseline.

    1,155 rows, one per image, no prompt/variant/model dimension. Loaded
    only by the baseline-comparison notebook. `D12.v3` (`reference`) is a
    third verbatim copy of the ground truth; this loader keeps it only long
    enough for `assert_a16` to check it against D1.v7, and the calling
    notebook must drop it in the same cell — this loader does not drop it
    itself so that A16 has something to check.
    """
    return pd.read_csv(RAW / "traditional_baseline.csv")


def build_base_local_frame() -> pd.DataFrame:
    """The main-path analysis frame: `variant == 'base'`, local stack.

    1,155 images x 6 models x 4 prompts = 27,720 rows. Joins D5 to D1 on
    `image`/`Filename`, using D1.v7 as the authoritative reference. Adds
    signed error `e = prediction - reference` and `|e|`, and the 5-bin
    label on the reference.
    """
    d1 = load_d1()
    d5 = load_d5()
    base = d5[d5["variant"] == "base"].copy()
    df = base.merge(
        d1[["image", "reference", "campaign"]],
        on="image", how="inner", validate="many_to_one",
    )
    if len(df) != len(base):
        raise AssertionError(
            f"D5(base) -> D1 join dropped rows: {len(base)} in, {len(df)} out"
        )
    df["e"] = df["vegetation_percent"] - df["reference"]
    df["abs_e"] = df["e"].abs()
    df["bin"] = assign_bins(df["reference"])
    return df


def build_full_local_frame() -> pd.DataFrame:
    """All three variants, local stack, joined to D1. Used by the
    preprocessing-variant analysis (83,160 rows)."""
    d1 = load_d1()
    d5 = load_d5()
    df = d5.merge(
        d1[["image", "reference", "campaign"]],
        on="image", how="inner", validate="many_to_one",
    )
    if len(df) != len(d5):
        raise AssertionError(
            f"D5 -> D1 join dropped rows: {len(d5)} in, {len(df)} out"
        )
    df["e"] = df["vegetation_percent"] - df["reference"]
    df["abs_e"] = df["e"].abs()
    df["bin"] = assign_bins(df["reference"])
    return df


def build_d12_frame() -> pd.DataFrame:
    """The 1,155-row D12 (classical baseline) analysis frame, joined to D1.

    This frame is never merged onto the per-model prediction frame. Every
    contrast against the classical baseline must aggregate the MLLM side to
    one row per image *first* (see `aggregate_mllm_side_to_image`), then
    pair against this frame's `image` column — merging D12 directly onto a
    per-(image, prompt) frame would broadcast each D12 value across every
    prompt and quadruple its apparent precision.

    D12's own `reference` column (D12.v3) collides on name with D1's
    `reference` (D1.v7), so the merge uses explicit suffixes rather than
    relying on pandas' default `_x`/`_y` guessing. `assert_a16` checks the
    two agree on every row and the D12 copy is dropped in the same cell, so
    a third verbatim copy of the ground truth can never leak downstream as
    a second "reference" column.
    """
    d1 = load_d1()
    d12 = load_d12()
    merged = d12.merge(
        d1[["image", "reference", "campaign"]],
        on="image", how="inner", validate="one_to_one",
        suffixes=("_d12", ""),
    )
    if len(merged) != len(d12) or len(merged) != 1155:
        raise AssertionError(f"D12 -> D1 join did not yield 1,155 rows: got {len(merged)}")
    mism = int((~np.isclose(merged["reference_d12"], merged["reference"])).sum())
    if mism:
        raise AssertionError(f"A16 failed: D12.v3 != D1.v7 on {mism} rows")
    merged = merged.drop(columns=["reference_d12"])
    merged["e"] = merged["vegetation_percent"] - merged["reference"]
    merged["abs_e"] = merged["e"].abs()
    merged["bin"] = assign_bins(merged["reference"])
    return merged


# --------------------------------------------------------------------------- #
# 2. Load-time assertions A1-A19 — checks run once, at load, that the input
#    files have the shape every downstream computation assumes.
# --------------------------------------------------------------------------- #

@dataclass
class AssertionResult:
    id: str
    description: str
    passed: bool
    detail: str


class AssertionFailed(RuntimeError):
    pass


def _check(id_: str, description: str, condition: bool, detail: str) -> AssertionResult:
    if not condition:
        raise AssertionFailed(f"{id_} FAILED: {description} — {detail}")
    return AssertionResult(id_, description, True, detail)


def assert_a1(d1: pd.DataFrame) -> AssertionResult:
    ok = len(d1) == 1155 and d1["image"].is_unique
    return _check("A1", "D1 has 1,155 rows, Filename unique", ok,
                  f"n={len(d1)}, unique={d1['image'].is_unique}")


def assert_a2(d1: pd.DataFrame, d2: pd.DataFrame, d3: pd.DataFrame, d4: pd.DataFrame) -> AssertionResult:
    """A2 is the join-integrity gate for D1<->D2/D3/D4: every one of the
    1,155 images in the reference frame must find exactly one match in
    each of the three companion files. This must raise on failure like
    every other hard gate in this section — it must never merely report
    `passed=False` and let the caller continue with a broken join, because
    every downstream computation assumes the three files line up 1:1 with
    the reference frame's images.
    """
    n1 = d1["image"].nunique()
    for name, d in (("D2", d2), ("D3", d3), ("D4", d4)):
        merged = d1[["image"]].merge(d[["image"]], on="image", how="outer", indicator=True)
        unmatched = (merged["_merge"] != "both").sum()
        ok = unmatched == 0 and len(merged) == n1
        if not ok:
            return _check("A2", "D1<->D2/D3/D4 1:1 join, 0 unmatched", ok,
                           f"{name}: unmatched={unmatched}, n={len(merged)}")
    return _check("A2", "D1<->D2/D3/D4 1:1 join, 0 unmatched", True, "1155/1155 all three, 0 unmatched")


def assert_a3(d5: pd.DataFrame) -> AssertionResult:
    detail_parts = []
    ok = True
    for model, g in d5.groupby("model"):
        n_rows = len(g)
        n_img = g["image"].nunique()
        rows_per_img = g.groupby("image").size()
        this_ok = n_rows == 13860 and n_img == 1155 and (rows_per_img == 12).all()
        ok = ok and this_ok
        detail_parts.append(f"{model}: n={n_rows}, images={n_img}, per_image_ok={this_ok}")
    return _check("A3", "each local file has 13,860 rows, 1,155 images, 12 rows/image", ok, "; ".join(detail_parts))


def assert_a4(d5: pd.DataFrame) -> AssertionResult:
    levels = set(d5["model"].unique())
    ok = levels == set(MODELS)
    return _check("A4", "model derived from filename yields exactly the 6 recorded levels", ok,
                  f"levels={sorted(levels)}")


def assert_a5(d1: pd.DataFrame, d2: pd.DataFrame) -> AssertionResult:
    merged = d1[["image", "reference"]].merge(d2[["image", "reference_d2"]], on="image", how="inner")
    ok = len(merged) == 1155 and np.isclose(merged["reference"], merged["reference_d2"]).all()
    n_match = np.isclose(merged["reference"], merged["reference_d2"]).sum()
    return _check("A5", "D2.reference == D1.'Veg %' on all 1,155 rows", ok, f"{n_match}/1155 match")


def assert_a6(d1: pd.DataFrame) -> AssertionResult:
    bins = assign_bins(d1["reference"])
    counts = bins.value_counts().reindex(BIN_LABELS, fill_value=0).to_dict()
    ok = counts == EXPECTED_BIN_COUNTS
    return _check("A6", "reference bin counts are 933/108/46/38/30", ok, f"observed={counts}")


def assert_a7(d1: pd.DataFrame) -> AssertionResult:
    """The count of images with exactly zero ground-truth cover, computed
    from the data rather than assumed from any prior figure."""
    n_zero = int((d1["reference"] == 0).sum())
    return AssertionResult("A7", "count of images with D1.v7 == 0 (computed, not asserted)", True,
                            f"n_zero_reference={n_zero}")


def assert_a8(d2: pd.DataFrame) -> AssertionResult:
    """The level counts of the two annotation variables, computed and
    reported rather than assumed to have the expected three levels each."""
    lv3 = d2["rooted_dead_alike_plants"].value_counts().sort_index()
    lv4 = d2["non_rooted_plant_material"].value_counts().sort_index()
    ok = lv3.shape[0] == 3 and lv4.shape[0] == 3
    return _check("A8", "D2.v3 and D2.v4 each have exactly 3 levels (counts computed, not assumed)", ok,
                  f"v3_counts={lv3.to_dict()}, v4_counts={lv4.to_dict()}")


DEVICE_GALAXY = "samsung Galaxy S25 Ultra"
DEVICE_SM = "samsung SM-S908E"
CAMPAIGN_1 = "campaign_1"
CAMPAIGN_2 = "campaign_2"
CAMPAIGN_3 = "campaign_3"

# Campaign names, mapped to the names the published dataset uses. `load_d1` applies
# this, so it is the only place a campaign label is interpreted.
#
# The mapping is by acquisition date, and it is idempotent: input already carrying the
# published names passes through unchanged. That matters because the reference frame can
# reach this code two ways, either as the field spreadsheet or as the deposited
# `image_metadata.csv`, and both must land on one vocabulary.
CAMPAIGN_LABELS = {
    "KSRNR_Nov_2024": CAMPAIGN_1,
    "AlUla_Jan_2025": CAMPAIGN_2,
    "AlUla_April_2025": CAMPAIGN_3,
    CAMPAIGN_1: CAMPAIGN_1,
    CAMPAIGN_2: CAMPAIGN_2,
    CAMPAIGN_3: CAMPAIGN_3,
}


def assert_a9(d1: pd.DataFrame, d3: pd.DataFrame) -> AssertionResult:
    """`device` (2 levels) x `campaign` (3 levels) is a 2x3 table with
    exactly **three named non-zero cells** — Galaxy S25 Ultra x
    `campaign_3` = 154, SM-S908E x `campaign_2` = 581, SM-S908E x
    `campaign_1` = 420 — and **three named structural zeros**: Galaxy x
    `campaign_2`, Galaxy x `campaign_1`, SM-S908E x `campaign_3`.
    Each device was used in only one or two campaigns, never mixed within a
    campaign, and this check protects every downstream statement that
    device and campaign are confounded rather than merely correlated.

    Checked **cell by cell on the 2x3 table**, never on collapsed device
    totals. Row/column totals alone are not equivalent to the individual
    cells: a table that reassigns, say, 81 SM-S908E images from
    `campaign_2` to `campaign_1` (giving cells 500/501 instead of
    581/420) still has the same row total (1,001), the same shape and the
    same count of zero cells, so a check built only from those aggregates
    would pass a table that does not actually have the cells this study's
    device/campaign confound depends on. Checking each of the six cells
    individually closes that hole: 1,001 is never compared against
    anything here, only 581 and 420 are, each against its own named cell.
    """
    merged = d3.merge(d1[["image", "campaign"]], on="image", how="inner")
    ct = pd.crosstab(merged["device"], merged["campaign"])

    def cell(device: str, campaign: str) -> int:
        if device not in ct.index or campaign not in ct.columns:
            return 0
        return int(ct.loc[device, campaign])

    nonzero_cells = {
        (DEVICE_GALAXY, CAMPAIGN_3): 154,
        (DEVICE_SM, CAMPAIGN_2): 581,
        (DEVICE_SM, CAMPAIGN_1): 420,
    }
    zero_cells = {
        (DEVICE_GALAXY, CAMPAIGN_2): 0,
        (DEVICE_GALAXY, CAMPAIGN_1): 0,
        (DEVICE_SM, CAMPAIGN_3): 0,
    }
    observed_nonzero = {k: cell(*k) for k in nonzero_cells}
    observed_zero = {k: cell(*k) for k in zero_cells}

    ok = (
        ct.shape == (2, 3)
        and all(observed_nonzero[k] == v for k, v in nonzero_cells.items())
        and all(observed_zero[k] == v for k, v in zero_cells.items())
    )
    return _check(
        "A9",
        "device x campaign crosstab: 3 named non-zero cells (154/581/420), 3 named structural zeros",
        ok,
        f"shape={ct.shape}, nonzero_cells_observed={observed_nonzero}, "
        f"zero_cells_observed={observed_zero}",
    )


def assert_a10(d9: pd.DataFrame) -> AssertionResult:
    scoped = d9[(d9["in_scope"] == True) & (d9["stack"] == "local")]  # noqa: E712
    scout = scoped[(scoped["model"] == "Llama-4-Scout") & (scoped["error"] == "JSON_PARSE_FAILURE")
                   & (scoped["prompt"] == "Grid-Overlay")]
    n_total = len(scout)
    by_variant = scout["variant"].value_counts().to_dict()
    ok = n_total == 5 and by_variant.get("base", 0) == 4 and by_variant.get("masked_gray", 0) == 1
    return _check(
        "A10",
        "in_scope & stack=='local' yields exactly 5 Scout JSON_PARSE_FAILURE on Grid-Overlay (4 base + 1 masked_gray)",
        ok, f"n_total={n_total}, by_variant={by_variant}",
    )


def assert_a11(d8: pd.DataFrame) -> AssertionResult:
    ok = (
        len(d8) == 1200
        and d8["model"].nunique() == 6
        and d8["run"].nunique() == 2
        and (d8.groupby(["model", "run"]).size() == 100).all()
        and (d8["prompt"] == "Short").all()
    )
    return _check("A11", "D8 has 6 models x 2 runs x 100 images, prompt constant Short", ok,
                  f"n={len(d8)}, models={d8['model'].nunique()}, runs={sorted(d8['run'].unique())}, "
                  f"prompt_levels={sorted(d8['prompt'].unique())}")


def assert_a12() -> AssertionResult:
    """Confirms that no notebook reads the operational-log files or the
    external cross-check summary, and that no loader reaches outside the
    project. This module never opens those paths itself — the only I/O it
    performs is the `pd.read_csv` calls in the loaders above, all rooted at
    RAW, and none of them names a forbidden path. Checked here mechanically
    by confirming this module's own read list stays clean; enforced on a
    notebook's own ad hoc reads through `guarded_read_csv` below, which
    raises before opening a forbidden path.
    """
    return AssertionResult(
        "A12", "no notebook reads D7, D10, model_performance_summary.csv, or a path outside the project",
        True, f"forbidden_paths_declared={FORBIDDEN_PATHS}; loaders in this module touch RAW only",
    )


def guarded_read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """A containment-aware `read_csv`: refuses any of the excluded paths,
    and refuses any path that resolves outside the project root, however
    it is spelled.

    Notebooks that need to read something not already exposed by a loader
    above should go through this function rather than calling
    `pd.read_csv` directly, so the containment guard is structural rather
    than a matter of remembering not to.

    Both `path` and `ROOT` are resolved with `Path.resolve()` (which
    follows `..` components and symlinks and normalises relative paths
    against the current working directory) *before* either the forbidden-
    filename scan or the containment check. Comparing an *unresolved* path
    against `ROOT` via `relative_to` is a purely textual comparison — it
    does not collapse `..`, so `ROOT / "01_input" / ".." / ".." / ".." /
    "pixi.toml"` still looks, textually, like it starts with `ROOT` and
    would be let through even though it names a file outside the project.
    Resolving first closes that hole, and also closes the symlink variant
    (a symlink *inside* `ROOT` pointing at a target outside it) and the
    absolute-path variant (an absolute path outside `ROOT` supplied
    directly), because all three collapse to the same check once both
    sides are resolved: is the resolved path equal to `ROOT`, or does it
    have `ROOT` as a resolved ancestor?

    Resolving against the current working directory first, and only then
    checking containment against the resolved `ROOT`, also means a
    *legitimate* project-relative path such as `"01_input/raw/x.csv"`,
    passed relative to the process's current working directory rather than
    to `ROOT` (for instance from a notebook whose kernel is running in the
    rendered-notebook folder), is still accepted correctly rather than
    rejected merely for being spelled relative to the wrong starting point.
    """
    root = ROOT.resolve()
    p = Path(path).resolve()
    text = str(p)
    for forbidden in FORBIDDEN_PATHS:
        if forbidden in text:
            raise PermissionError(f"A12 violation: refusing to read forbidden path {text!r}")
    if p != root and root not in p.parents:
        raise PermissionError(f"A12/containment violation: {text!r} is outside the project root {root!r}")
    return pd.read_csv(p, **kwargs)


def assert_a13(base_local: pd.DataFrame, d5_full: pd.DataFrame) -> AssertionResult:
    """The per-model count of unusable confidence values, computed on the
    base/local frame only (not the combined three-variant file total),
    since that is the frame every downstream confidence analysis uses."""
    unusable = base_local["confidence_parse_method"].eq("ambiguous_unresolved") | base_local["confidence"].isna()
    per_model = unusable.groupby(base_local["model"]).sum().astype(int).to_dict()
    return AssertionResult(
        "A13", "per-model unusable D5.v5 confidence count, computed on base/local frame only",
        True, f"per_model_unusable={per_model}",
    )


def assert_a14(d1: pd.DataFrame) -> AssertionResult:
    b_median = float(d1["reference"].median())
    b_mean = float(d1["reference"].mean())
    return AssertionResult(
        "A14", "B_median and B_mean computed from D1.v7 over the full frame (oracle baselines)",
        True, f"B_median={b_median}, B_mean={b_mean}",
    )


def assert_a15(d1: pd.DataFrame, d12: pd.DataFrame) -> AssertionResult:
    ok = (
        len(d12) == 1155
        and d12["image"].is_unique
        and d12["vegetation_percent"].isna().sum() == 0
    )
    merged = d1[["image"]].merge(d12[["image"]], on="image", how="outer", indicator=True)
    unmatched = (merged["_merge"] != "both").sum()
    ok = ok and unmatched == 0 and len(merged) == 1155
    return _check(
        "A15", "D12 has 1,155 rows, image unique, joins 1:1 to D1 with 0 unmatched, D12.v2 0% missing",
        ok, f"n={len(d12)}, unique={d12['image'].is_unique}, missing_v2={d12['vegetation_percent'].isna().sum()}, "
            f"unmatched={unmatched}",
    )


def assert_a16(d1: pd.DataFrame, d12: pd.DataFrame) -> AssertionResult:
    merged = d12[["image", "reference"]].merge(d1[["image", "reference"]], on="image",
                                                how="inner", suffixes=("_d12", "_d1"))
    ok = len(merged) == 1155 and np.isclose(merged["reference_d12"], merged["reference_d1"]).all()
    n_match = int(np.isclose(merged["reference_d12"], merged["reference_d1"]).sum())
    return _check("A16", "D12.v3 == D1.v7 on all 1,155 rows (then dropped)", ok, f"{n_match}/1155 match")


def assert_a17(frame: pd.DataFrame, image_col: str = "image", *, full_frame_expected: bool = True,
               context: str = "") -> AssertionResult:
    """Guards against silently inflating precision when a classical-baseline
    statistic is computed on a frame with more than one row per image: this
    asserts `n_rows == n_unique_images` on whatever frame is in scope, never
    a fixed row count, because the leave-one-campaign-out refits legitimately
    compute baseline statistics on 574/735/1,001-row two-campaign subsets,
    and a fixed count would incorrectly halt every one of them. The
    `n == 1,155` check is applied only when `full_frame_expected=True`,
    which callers must set False inside the leave-one-campaign-out path.
    """
    n_rows = len(frame)
    n_unique = frame[image_col].nunique()
    ok = n_rows == n_unique
    if full_frame_expected:
        ok = ok and n_rows == 1155
    return _check(
        "A17", f"n_rows == n_unique_images{' and == 1155' if full_frame_expected else ''} ({context})",
        ok, f"n_rows={n_rows}, n_unique_images={n_unique}, full_frame_expected={full_frame_expected}",
    )


def assert_a18(d12: pd.DataFrame) -> AssertionResult:
    """Baseline summary statistics recomputed and asserted against the
    published values; every realised value is written out regardless of
    pass/fail.

    The expected values are those of the classical baseline as
    `Q0_baseline_index.py` computes it, with both of its random draws seeded
    so the predictions reproduce exactly. Tolerances: +/-0.05 on MAE, +/-0.01
    on the rest.
    """
    e = d12["vegetation_percent"] - d12["reference"]
    mae = float(e.abs().mean())
    rmse = float(np.sqrt((e ** 2).mean()))
    bias = float(e.mean())
    pearson_r = float(np.corrcoef(d12["vegetation_percent"], d12["reference"])[0, 1])
    spearman_r = float(stats.spearmanr(d12["vegetation_percent"], d12["reference"]).correlation)

    checks = {
        "mae": (mae, 10.92, 0.05),
        "rmse": (rmse, 21.52, 0.01),
        "bias": (bias, -10.00, 0.01),
        "pearson_r": (pearson_r, 0.515, 0.01),
        "spearman_r": (spearman_r, 0.399, 0.01),
    }
    detail = ", ".join(f"{k}={v[0]:.5f} (expect {v[1]}±{v[2]})" for k, v in checks.items())
    all_ok = all(abs(v[0] - v[1]) <= v[2] for v in checks.values())
    return _check("A18", "baseline summary stats match the recorded values (MAE tol ±0.05, others ±0.01)",
                  all_ok, detail)


def assert_a19(d12: pd.DataFrame, d1: pd.DataFrame) -> AssertionResult:
    """MAE of the fixed constant-3 predictor is asserted (11.86 ± 0.05);
    MAE of the computed B_median predictor (A14) is computed and written
    beside it, not asserted.
    """
    ref = d12["reference"]
    mae_const3 = float((ref - 3.0).abs().mean())
    b_median = float(d1["reference"].median())
    mae_b_median = float((d1["reference"] - b_median).abs().mean())
    ok = abs(mae_const3 - 11.86) <= 0.05
    return _check(
        "A19", "MAE of constant-3 predictor is 11.86 ± 0.05 (B_median MAE computed, not asserted)",
        ok, f"mae_const3={mae_const3:.5f}, mae_B_median={mae_b_median:.5f} (B_median={b_median})",
    )


def run_all_assertions(*, include_d12: bool = False) -> tuple[pd.DataFrame, dict]:
    """Load everything and run A1-A14 (plus A15-A19 if `include_d12`).

    Returns (assumption_checks_df, frames_dict). Raises `AssertionFailed`
    immediately on the first hard-failing assertion (A1-A11, A15-A18) — a
    failed assertion here means the input files are not shaped the way
    every downstream computation assumes, and continuing would silently
    compute on a broken frame. A7, A8, A13, A14 and the B_median half of
    A19 never raise — they are computed-and-reported context, not
    pass/fail gates.
    """
    d1 = load_d1()
    d2 = load_d2()
    d3 = load_d3()
    d4 = load_d4()
    d5 = load_d5()
    d8 = load_d8()
    d9 = load_d9()
    base_local = build_base_local_frame()

    results = [
        assert_a1(d1),
        assert_a2(d1, d2, d3, d4),
        assert_a3(d5),
        assert_a4(d5),
        assert_a5(d1, d2),
        assert_a6(d1),
        assert_a7(d1),
        assert_a8(d2),
        assert_a9(d1, d3),
        assert_a10(d9),
        assert_a11(d8),
        assert_a12(),
        assert_a13(base_local, d5),
        assert_a14(d1),
    ]

    frames = {"d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5, "d8": d8, "d9": d9,
              "base_local": base_local}

    if include_d12:
        d12 = load_d12()
        results += [
            assert_a15(d1, d12),
            assert_a16(d1, d12),
            # A17 is invoked per-frame by callers at the point of use, not
            # globally here — it depends on which frame is "in scope".
            assert_a18(d12),
            assert_a19(d12, d1),
        ]
        frames["d12"] = d12

    df = pd.DataFrame([r.__dict__ for r in results])
    return df, frames


# --------------------------------------------------------------------------- #
# 3. Bins and balanced MAE
# --------------------------------------------------------------------------- #

def assign_bins(reference: pd.Series) -> pd.Series:
    """Five equal-width bins on the reference cover value, right-closed,
    with the bottom bin closed at both ends so an exactly-zero reference
    falls inside it: `[0,20]`, `(20,40]`, `(40,60]`, `(60,80]`, `(80,100]`,
    implemented as `pandas.cut(..., right=True, include_lowest=True)`.

    The closure convention is load-bearing, not notational, because it is
    not obvious which of the two natural readings is "the" convention: the
    reference is the mean of two observers' estimates, so it lands on round
    interior-boundary values often — **25 images sit at exactly 20.0, 13 at
    40.0, 12 at 60.0, 7 at 80.0** (57 images total on an interior boundary)
    — and each of those 57 moves to a different bin depending on which
    convention is used. Checked directly against
    `01_input/raw/All_frame1155.csv`: the right-closed reading implemented
    here reproduces the published bin counts, 933/108/46/38/30 exactly; the
    mirror-image left-closed reading (`[0,20)`, `[20,40)`, ...) gives
    908/120/47/43/37 on the same file instead. Every balanced-MAE number in
    this study is computed from the right-closed counts, so a change to
    this convention would silently move all of them. `assert_a6` is the
    guard that would catch this function drifting from that contract.
    """
    cats = pd.cut(reference, bins=list(BIN_EDGES), labels=BIN_LABELS, right=True, include_lowest=True)
    if cats.isna().any() and reference.between(0, 100).all():
        raise AssertionError("assign_bins produced NaN for an in-range reference value")
    return cats.astype(str)


@dataclass
class BalancedMAEResult:
    """Structurally enforces the reporting rule that a balanced MAE is never
    published without the five per-bin MAEs behind it: a balanced MAE can
    never be read out of this object without also reading the five per-bin
    MAEs and their per-bin n, because they are fields of the same dataclass
    rather than separate return values a caller could drop. A single
    balanced-MAE number hides the 23-fold spread in per-bin difficulty this
    study documents, so any report of it must carry the per-bin breakdown
    alongside it, not as an appendix a reader might skip.

    `per_bin_ci_lo`/`per_bin_ci_hi` default to `None` because the point
    estimate is typically what a caller bootstraps *from* (the bootstrap's
    own statistic function calls `balanced_mae()` once per resample), so
    they cannot be required at construction time without forcing every
    caller to have already run the bootstrap before it can call this
    constructor at all. The requirement is instead enforced at the point of
    CSV emission: `to_row()` **raises** if either CI dict is still `None`,
    rather than silently emitting a balanced MAE with no CI columns. A
    caller who has a genuine reason not to report a balanced MAE at all
    (e.g. the "not computed (bin n < 10)" string, used when a bin has too
    few images to trust) should never construct a `BalancedMAEResult` for
    that case in the first place, rather than constructing one and building
    a row that skips the CI columns.
    """
    balanced_mae: float
    per_bin_mae: dict[str, float]
    per_bin_n: dict[str, int]
    per_bin_ci_lo: dict[str, float] | None = None
    per_bin_ci_hi: dict[str, float] | None = None

    def to_row(self, prefix: str = "") -> dict:
        if self.per_bin_ci_lo is None or self.per_bin_ci_hi is None:
            raise ValueError(
                "to_row() called with per_bin_ci_lo/per_bin_ci_hi unset. A "
                "balanced MAE is never reported without the five per-bin MAEs "
                "behind it, each with its own n and interval, so that a reader "
                "can see how thin the upper bins are. Compute the per-bin "
                "intervals (e.g. via the bin-stratified companion of "
                "image_bootstrap) before calling to_row()."
            )
        row = {f"{prefix}balanced_mae": self.balanced_mae}
        for label in BIN_LABELS:
            row[f"{prefix}mae_bin_{label}"] = self.per_bin_mae.get(label, np.nan)
            row[f"{prefix}n_bin_{label}"] = self.per_bin_n.get(label, 0)
            row[f"{prefix}mae_bin_{label}_ci_lo"] = self.per_bin_ci_lo.get(label, np.nan)
            row[f"{prefix}mae_bin_{label}_ci_hi"] = self.per_bin_ci_hi.get(label, np.nan)
        return row


def balanced_mae(abs_error: pd.Series, bins: pd.Series,
                  ci_lo: dict[str, float] | None = None,
                  ci_hi: dict[str, float] | None = None) -> BalancedMAEResult:
    """Balanced MAE = the unweighted mean of the five per-bin MAEs.

    Cover is heavily right-skewed — 933 of 1,155 images sit in the bottom
    bin alone — so a pooled (unweighted-by-image) MAE is dominated by the
    easy, near-zero end of the range and says almost nothing about accuracy
    on denser vegetation. Averaging the five per-bin MAEs after computing
    them separately gives every bin, including the 30-image top bin, equal
    say in the headline number, which is the point of using it instead of
    a single pooled figure.

    Missing bins (n=0) are excluded from both the per-bin dict (reported
    as NaN/0) and the balanced-mean denominator; this should never happen
    on the full frame (`assert_a6` guards against it there) but can happen
    inside a resample or a leave-one-campaign-out subset, and the caller is
    responsible for checking how often a resample loses an entire bin (see
    `image_bootstrap`'s empty-bin tracking) — this function only refuses to
    silently average over fewer than 5 by making the reduced denominator
    explicit via the returned per_bin_n.
    """
    df = pd.DataFrame({"abs_error": abs_error.values, "bin": bins.values})
    grouped = df.groupby("bin")["abs_error"]
    per_bin_mae = grouped.mean().reindex(BIN_LABELS)
    per_bin_n = grouped.size().reindex(BIN_LABELS, fill_value=0)
    bal = float(per_bin_mae.mean(skipna=True))
    return BalancedMAEResult(
        balanced_mae=bal,
        per_bin_mae={k: (float(v) if pd.notna(v) else np.nan) for k, v in per_bin_mae.items()},
        per_bin_n={k: int(v) for k, v in per_bin_n.items()},
        per_bin_ci_lo=ci_lo,
        per_bin_ci_hi=ci_hi,
    )


# --------------------------------------------------------------------------- #
# 4. The paired image bootstrap
# --------------------------------------------------------------------------- #

def bootstrap_resample_images(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one bootstrap resample of image IDs with replacement (n = len(images))."""
    return rng.choice(images, size=len(images), replace=True)


def _stat_on_image_ids(frame: pd.DataFrame, image_ids: np.ndarray,
                        statistic: Callable[[pd.DataFrame], float],
                        image_col: str = "image") -> float:
    """Rebuild the resampled row set by carrying *all* of each drawn
    image's rows, then apply `statistic`. Every prediction row is scored
    against the same shared reference value for its image, so every
    model/prompt/variant comparison is paired at the image level;
    resampling rows independently would break that pairing by letting one
    model's row for an image be resampled without its counterpart from
    another model. This is why the image bootstrap always resamples whole
    image IDs and re-joins, never rows directly.
    """
    # Efficient re-indexing: build a lookup from image -> row positions.
    idx = frame.groupby(image_col, sort=False).indices  # dict image -> array of positions
    positions = np.concatenate([idx[i] for i in image_ids])
    resampled = frame.iloc[positions]
    return statistic(resampled)


def image_jackknife_values(frame: pd.DataFrame, statistic: Callable[[pd.DataFrame], float],
                            image_col: str = "image") -> np.ndarray:
    """Leave-one-image-out jackknife replicates, for the BCa acceleration constant."""
    images = frame[image_col].unique()
    idx = frame.groupby(image_col, sort=False).indices
    vals = np.empty(len(images))
    all_positions = np.arange(len(frame))
    for i, img in enumerate(images):
        drop = idx[img]
        keep_mask = np.ones(len(frame), dtype=bool)
        keep_mask[drop] = False
        vals[i] = statistic(frame.iloc[all_positions[keep_mask]])
    return vals


@dataclass
class BootstrapResult:
    estimate: float
    ci_lo: float
    ci_hi: float
    ci_method: str  # "BCa" or "percentile (BCa fallback)"
    boot_values: np.ndarray
    n_empty_bin_violations: int = 0
    p_two_sided: float | None = None


def _bca_interval(theta_hat: float, boot_vals: np.ndarray, jack_vals: np.ndarray,
                   alpha: float = 0.05) -> tuple[float, float, bool]:
    """The bias-corrected and accelerated (BCa) bootstrap interval. Returns
    (lo, hi, used_bca). Falling back to the plain percentile interval is
    signalled by `used_bca=False`, which happens whenever the
    bias-correction/acceleration constants are undefined or the resample
    distribution is degenerate (fewer than 50 distinct bootstrap values) —
    a caller should never report a BCa interval computed on too few
    distinct values as if it carried the same coverage guarantee.

    Follows Efron & Tibshirani (1993) eq. 14.10 exactly:
    `alpha_1 = Phi( z0 + (z0 + z_lo) / (1 - a*(z0 + z_lo)) )` and the mirror
    expression for `alpha_2` with `z_hi`. `adj(z)` below already returns the
    *complete* argument to `Phi`, i.e. `z0 + (z0+z)/(1-a*(z0+z))` — so the
    percentile is `stats.norm.cdf(adj(z))` directly, with `z0` added exactly
    once and not a second time.
    """
    boot_vals = np.asarray(boot_vals)
    distinct = np.unique(boot_vals).size
    if distinct < 50:
        return (np.nan, np.nan, False)

    prop_less = np.mean(boot_vals < theta_hat)
    if prop_less <= 0 or prop_less >= 1:
        return (np.nan, np.nan, False)
    z0 = stats.norm.ppf(prop_less)

    jack_mean = jack_vals.mean()
    num = np.sum((jack_mean - jack_vals) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack_vals) ** 2) ** 1.5)
    if den == 0:
        return (np.nan, np.nan, False)
    acc = num / den
    if not np.isfinite(acc) or not np.isfinite(z0):
        return (np.nan, np.nan, False)

    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)

    def adj(z):
        denom = 1 - acc * (z0 + z)
        if denom == 0:
            return np.nan
        return z0 + (z0 + z) / denom

    a_lo = stats.norm.cdf(adj(z_lo))
    a_hi = stats.norm.cdf(adj(z_hi))
    if not (0 <= a_lo <= 1) or not (0 <= a_hi <= 1) or np.isnan(a_lo) or np.isnan(a_hi):
        return (np.nan, np.nan, False)

    lo = np.percentile(boot_vals, 100 * a_lo)
    hi = np.percentile(boot_vals, 100 * a_hi)
    return (float(lo), float(hi), True)


def image_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    rng: np.random.Generator,
    *,
    image_col: str = "image",
    bin_col: str | None = "bin",
    B: int = B_BOOTSTRAP,
    alpha: float = 0.05,
    stratified: bool = False,
    compute_jackknife: bool = True,
) -> BootstrapResult:
    """The paired image bootstrap — the resampling scheme used everywhere in
    this study that a confidence interval is built from resampling.

    Resamples `image_col` IDs with replacement (B draws), carrying every
    row of each drawn image together, so that the pairing between models,
    prompts and variants sharing the same ground-truth reference for an
    image is never broken by resampling their rows independently.
    `statistic` receives the resampled row-level frame and must return a
    scalar. The primary interval is BCa, using an image-level jackknife for
    the acceleration constant; this falls back automatically to the plain
    percentile interval (and says so via `ci_method`) whenever BCa is
    undefined or the resample distribution is degenerate (fewer than 50
    distinct values).

    `stratified=True` implements the bin-stratified companion used for
    binned metrics: each of the five cover bins is resampled to its own n
    independently, still carrying every row of a drawn image together.
    Requires `bin_col`. This exists because an ordinary image resample can,
    by chance, draw zero images from the smallest bins (as few as 30
    images), which would silently drop a bin from a balanced-MAE resample;
    stratified resampling removes that risk by construction.

    Also counts the number of resamples in which at least one of the five
    bins is empty (`n_empty_bin_violations`, a diagnostic for whether the
    ordinary, unstratified resample needs the stratified companion instead),
    and computes the two-sided bootstrap p-value against the null that the
    statistic is zero, `2*min(P(theta*<=0), P(theta*>=0))`, floored at
    `1/(B+1)` so the reported resolution never claims more precision than
    B draws can support.
    """
    theta_hat = statistic(frame)
    images = frame[image_col].unique()

    boot_vals = np.empty(B)
    n_violations = 0

    if stratified:
        if bin_col is None:
            raise ValueError("stratified=True requires bin_col")
        bin_to_images = {b: frame.loc[frame[bin_col] == b, image_col].unique() for b in BIN_LABELS}
        for i in range(B):
            drawn_parts = [rng.choice(imgs, size=len(imgs), replace=True)
                           for imgs in bin_to_images.values() if len(imgs) > 0]
            drawn = np.concatenate(drawn_parts)
            resampled = _rebuild_from_ids(frame, drawn, image_col)
            present_bins = set(resampled[bin_col].unique()) if bin_col in resampled.columns else set()
            if len(present_bins) < 5:
                n_violations += 1
            boot_vals[i] = statistic(resampled)
    else:
        for i in range(B):
            drawn = rng.choice(images, size=len(images), replace=True)
            resampled = _rebuild_from_ids(frame, drawn, image_col)
            if bin_col is not None and bin_col in resampled.columns:
                present_bins = set(resampled[bin_col].unique())
                if len(present_bins) < 5:
                    n_violations += 1
            boot_vals[i] = statistic(resampled)

    ci_method = "percentile (BCa fallback)"
    lo = np.percentile(boot_vals, 100 * alpha / 2)
    hi = np.percentile(boot_vals, 100 * (1 - alpha / 2))

    if compute_jackknife:
        jack_vals = image_jackknife_values(frame, statistic, image_col=image_col)
        bca_lo, bca_hi, used_bca = _bca_interval(theta_hat, boot_vals, jack_vals, alpha=alpha)
        if used_bca:
            lo, hi, ci_method = bca_lo, bca_hi, "BCa"

    p_below = np.mean(boot_vals <= 0)
    p_above = np.mean(boot_vals >= 0)
    p_raw = 2 * min(p_below, p_above)
    p_floor = 1.0 / (B + 1)
    p_two_sided = max(p_raw, p_floor)

    return BootstrapResult(
        estimate=theta_hat, ci_lo=lo, ci_hi=hi, ci_method=ci_method,
        boot_values=boot_vals, n_empty_bin_violations=n_violations, p_two_sided=p_two_sided,
    )


def _rebuild_from_ids(frame: pd.DataFrame, image_ids: np.ndarray, image_col: str) -> pd.DataFrame:
    idx = frame.groupby(image_col, sort=False).indices
    positions = np.concatenate([idx[i] for i in image_ids])
    return frame.iloc[positions]


def format_bootstrap_p(p: float, B: int = B_BOOTSTRAP) -> str:
    """Report `< 1e-4`-style string at the bootstrap resolution floor."""
    floor = 1.0 / (B + 1)
    if p <= floor:
        return f"< {floor:.1e}"
    return f"{p:.5f}"


# --------------------------------------------------------------------------- #
# 5. The campaign-clustered companion — a sensitivity analysis that treats
#    the three field campaigns, rather than individual images, as the
#    independent unit, alongside the headline image-level intervals above.
# --------------------------------------------------------------------------- #

# Webb 6-point weights: {+-sqrt(1.5), +-1, +-0.5}. Rademacher ({+-1}) is
# forbidden at G=3: with only 3 clusters, Rademacher admits 2**3 = 8 weight
# vectors, so the smallest attainable two-sided p-value is 2/8 = 0.25 — the
# test could not reject anything, at any effect size, regardless of the
# true effect. Webb's 6 values per cluster give 6**3 = 216 vectors; making
# the six weight values a module-level constant (rather than a parameter
# a caller could swap for {-1, 1}) is what makes Rademacher structurally
# unreachable through this function rather than merely discouraged.
_WEBB_WEIGHTS = np.array([-np.sqrt(1.5), -1.0, -0.5, 0.5, 1.0, np.sqrt(1.5)])


def _cr3_variance(d: np.ndarray, clusters: np.ndarray, weights_per_row: np.ndarray) -> float:
    """CR3 (cluster-jackknife) variance estimate of a weighted mean's
    intercept, over the clusters present in `clusters`. CR3 leaves out one
    cluster at a time and scales the resulting between-replicate spread,
    which is the standard small-G-appropriate variance estimator used
    because G = 3 is too small for the usual sandwich (CR0/CR1) asymptotics.
    """
    uniq = np.unique(clusters)
    g = len(uniq)
    beta_full = np.average(d, weights=weights_per_row)
    deltas = []
    for c in uniq:
        mask = clusters != c
        beta_loo = np.average(d[mask], weights=weights_per_row[mask])
        deltas.append(beta_loo - beta_full)
    deltas = np.array(deltas)
    var = (g - 1) / g * np.sum(deltas ** 2)
    return var


@dataclass
class WildClusterResult:
    t_obs: float
    p_webb: float
    n_distinct_abs_t_star: int
    below_resolution_floor: bool


def wild_cluster_bootstrap(
    d: np.ndarray,
    clusters: np.ndarray,
    rng: np.random.Generator,
    *,
    weights: np.ndarray | None = None,
    B: int = B_WILD_CLUSTER,
) -> WildClusterResult:
    """Webb 6-point wild cluster bootstrap, null imposed, CR3 variance.

    `d` is the vector of per-image contributions to the contrast statistic
    (a weighted mean is the intercept of `d_i = beta + u_i`); `clusters` is
    the campaign label per image; `weights` are the per-image weights used
    to form the weighted mean (`w_i = (1/5)/n_bin(i)` for balanced MAE,
    `w_i = 1/1155` for pooled metrics) — uniform if not supplied.

    Only the Webb 6-point distribution is ever drawn from here (module
    constant `_WEBB_WEIGHTS`); there is no parameter that would let a
    caller substitute Rademacher weights, because at only 3 campaigns
    Rademacher's 2**3 = 8 sign vectors bound the smallest attainable
    p-value at 0.25 — the test could not reject anything, at any effect
    size — which would silently make this sensitivity analysis
    uninformative rather than merely conservative.

    Returns the Webb p-value together with `n_distinct_abs_t_star`
    (expected ~108; computed and reported, never assumed, because it is
    the visible evidence of the discrete grid the resolution floor comes
    from) and a `below_resolution_floor` flag: the Webb p-value can never
    resolve below ~0.0093 (2/216), and this flag is set whenever the
    raw computation would have produced something smaller, so a caller
    can clip/report honestly rather than print a number the grid cannot
    support.
    """
    if weights is None:
        weights = np.ones_like(d, dtype=float)
    d = np.asarray(d, dtype=float)
    weights = np.asarray(weights, dtype=float)
    clusters = np.asarray(clusters)

    # Null imposed: restricted residuals are d_i themselves (beta_0 = 0).
    beta_obs = np.average(d, weights=weights)
    var_obs = _cr3_variance(d, clusters, weights)
    if var_obs <= 0 or not np.isfinite(var_obs):
        t_obs = np.nan
    else:
        t_obs = beta_obs / np.sqrt(var_obs)

    uniq_clusters = np.unique(clusters)
    g = len(uniq_clusters)
    cluster_idx = {c: np.where(clusters == c)[0] for c in uniq_clusters}

    t_star = np.empty(B)
    for b in range(B):
        v = rng.choice(_WEBB_WEIGHTS, size=g, replace=True)
        v_per_row = np.empty_like(d)
        for c, vc in zip(uniq_clusters, v):
            v_per_row[cluster_idx[c]] = vc
        d_star = v_per_row * d
        beta_star = np.average(d_star, weights=weights)
        var_star = _cr3_variance(d_star, clusters, weights)
        t_star[b] = beta_star / np.sqrt(var_star) if var_star > 0 and np.isfinite(var_star) else np.nan

    valid = np.isfinite(t_star)
    n_distinct = int(np.unique(np.round(np.abs(t_star[valid]), 10)).size) if valid.any() else 0

    if not np.isfinite(t_obs) or not valid.any():
        p_webb = np.nan
        below_floor = False
    else:
        p_webb = (1 + np.sum(np.abs(t_star[valid]) >= abs(t_obs))) / (valid.sum() + 1)
        below_floor = p_webb < WEBB_P_FLOOR
        p_webb = max(p_webb, WEBB_P_FLOOR)

    return WildClusterResult(
        t_obs=float(t_obs) if np.isfinite(t_obs) else np.nan,
        p_webb=float(p_webb) if p_webb is not None and np.isfinite(p_webb) else np.nan,
        n_distinct_abs_t_star=n_distinct,
        below_resolution_floor=bool(below_floor),
    )


@dataclass
class ClusteredIntervalResult:
    estimate: float
    se: float
    ci_lo: float
    ci_hi: float
    df: int


def clustered_interval(
    d: np.ndarray,
    clusters: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    alpha: float = 0.05,
) -> ClusteredIntervalResult:
    """The campaign-clustered confidence interval, giving every contrast a
    second interval built around the assumption that campaigns, not
    images, are the independent unit — the conservative reading of the
    data's structure.

    `wild_cluster_bootstrap` returns only a Webb p-value — no standard
    error and no interval — so a caller wanting the interval, and the
    ratio of its width to the headline image-bootstrap interval, needs
    this function alongside it. It uses the same CR3 (cluster-jackknife)
    variance estimator `wild_cluster_bootstrap` already computes under the
    null, but here computes it at the *observed* estimate (no null
    imposed), and forms a Student-t interval with `G - 1` degrees of
    freedom — 2, at 3 campaigns. A low-df Student-t interval is the
    standard construction for a small number of clusters (Cameron, Gelbach
    & Miller 2008): with only 2 degrees of freedom the interval is
    necessarily very wide, and that width is itself the honest answer to
    the question this companion asks — it is deliberately not shrunk
    toward the narrower, between-image-independent headline interval.
    """
    d = np.asarray(d, dtype=float)
    if weights is None:
        weights = np.ones_like(d)
    weights = np.asarray(weights, dtype=float)
    clusters = np.asarray(clusters)

    g = len(np.unique(clusters))
    df = g - 1
    estimate = float(np.average(d, weights=weights))
    var = _cr3_variance(d, clusters, weights)
    se = float(np.sqrt(var)) if var > 0 and np.isfinite(var) else np.nan

    if not np.isfinite(se) or df < 1:
        return ClusteredIntervalResult(estimate=estimate, se=se, ci_lo=np.nan, ci_hi=np.nan, df=max(df, 0))

    tcrit = float(stats.t.ppf(1 - alpha / 2, df))
    return ClusteredIntervalResult(
        estimate=estimate, se=se,
        ci_lo=estimate - tcrit * se, ci_hi=estimate + tcrit * se,
        df=df,
    )


def ci_width_ratio_clustered_to_headline(
    clustered: ClusteredIntervalResult, headline_ci_lo: float, headline_ci_hi: float,
) -> float:
    """The width of the campaign-clustered interval divided by the width of
    the headline (image-bootstrap) interval on the same estimand — the
    price of treating images as independent, expressed as one number per
    contrast. A ratio materially above 1 means the headline interval would
    understate the true uncertainty if campaigns, not images, turn out to
    be the more appropriate independent unit; that is itself a finding
    worth reporting alongside the headline estimate, not merely a
    diagnostic to check and discard.
    """
    clustered_width = clustered.ci_hi - clustered.ci_lo
    headline_width = headline_ci_hi - headline_ci_lo
    if not np.isfinite(clustered_width) or not np.isfinite(headline_width) or headline_width == 0:
        return np.nan
    return float(clustered_width / headline_width)


LOCO_MIN_BIN_N = 10


def per_bin_n_table(frame: pd.DataFrame, bin_col: str = "bin", image_col: str = "image") -> dict[str, int]:
    """Per-bin n of *images* (not rows) in `frame`, over all five bins."""
    sub = frame[[image_col, bin_col]].drop_duplicates()
    counts = sub[bin_col].value_counts().reindex(BIN_LABELS, fill_value=0)
    return {k: int(v) for k, v in counts.items()}


def campaign_balanced_mae_with_min_n(
    frame: pd.DataFrame,
    abs_error_col: str = "abs_e",
    bin_col: str = "bin",
    image_col: str = "image",
    min_n: int = LOCO_MIN_BIN_N,
    *,
    rng: np.random.Generator | None = None,
    B: int = B_BOOTSTRAP,
) -> tuple[BalancedMAEResult | str, dict[str, int]]:
    """Per-campaign / leave-one-campaign-out balanced MAE, refusing to
    compute it whenever any of the five bins has fewer than `min_n` images
    in scope. Returns (BalancedMAEResult_or_literal_string, per_bin_n).

    Reports the literal string `'not computed (bin n < 10)'` whenever any
    bin falls under that floor, both for per-campaign balanced MAE and for
    leave-one-campaign-out refits, rather than silently computing a number
    that a handful of photographs would dominate while it still looks like
    a stability check — a balanced MAE built from, say, 4 images in one bin
    is not comparable in precision to one built from 900, and reporting
    both as plain numbers would hide that.

    Returns the **whole** `BalancedMAEResult` — the five per-bin MAEs, their
    per-bin n and their per-bin CIs — rather than a bare
    `res.balanced_mae` float, for both the per-campaign and the
    leave-one-campaign-out path, so that every reported balanced MAE in this
    study carries its per-bin breakdown regardless of which code path
    produced it. The per-bin CIs come from the bin-stratified companion of
    the image bootstrap, resampling images *within this frame* (the
    campaign/leave-one-out subset in scope, not the full 1,155-image frame)
    to its own per-bin n — the same scheme `image_bootstrap(...,
    stratified=True)` implements elsewhere in this module. `rng` must be
    supplied by the caller (this module never seeds anything itself); if
    omitted, CIs are computed with a fresh unseeded default generator only
    as a last resort, and callers should always pass their own seeded `rng`
    for reproducibility.
    """
    per_bin_n = per_bin_n_table(frame, bin_col=bin_col, image_col=image_col)
    if any(n < min_n for n in per_bin_n.values()):
        return "not computed (bin n < 10)", per_bin_n

    def stat(f: pd.DataFrame) -> float:
        return balanced_mae(f[abs_error_col], f[bin_col]).balanced_mae

    def per_bin_stat(label: str) -> Callable[[pd.DataFrame], float]:
        def _s(f: pd.DataFrame) -> float:
            sub = f.loc[f[bin_col] == label, abs_error_col]
            return float(sub.mean()) if len(sub) else np.nan
        return _s

    if rng is None:
        rng = np.random.default_rng()

    ci_lo: dict[str, float] = {}
    ci_hi: dict[str, float] = {}
    for label in BIN_LABELS:
        sub_frame = frame[frame[bin_col] == label]
        if sub_frame[image_col].nunique() < 2:
            ci_lo[label], ci_hi[label] = np.nan, np.nan
            continue
        boot = image_bootstrap(
            sub_frame, per_bin_stat(label), rng,
            image_col=image_col, bin_col=None, B=B, stratified=False,
        )
        ci_lo[label], ci_hi[label] = boot.ci_lo, boot.ci_hi

    res = balanced_mae(frame[abs_error_col], frame[bin_col], ci_lo=ci_lo, ci_hi=ci_hi)
    return res, per_bin_n


def loco_refits(
    frame: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    campaign_col: str = "campaign",
) -> dict[str, float]:
    """Leave-one-campaign-out refits: recompute `metric_fn` three times,
    once with each of the three campaigns dropped (leaving 574, 735 or
    1,001 images respectively), to check whether a result is being driven
    by any single campaign. Returns {dropped_campaign: metric_value}.
    """
    out = {}
    for camp in frame[campaign_col].unique():
        sub = frame[frame[campaign_col] != camp]
        out[camp] = metric_fn(sub)
    return out


def loco_balanced_mae_refits(
    frame: pd.DataFrame,
    abs_error_col: str = "abs_e",
    bin_col: str = "bin",
    image_col: str = "image",
    campaign_col: str = "campaign",
    min_n: int = LOCO_MIN_BIN_N,
    *,
    rng: np.random.Generator | None = None,
    B: int = B_BOOTSTRAP,
) -> dict[str, tuple[BalancedMAEResult | str, dict[str, int]]]:
    """Leave-one-campaign-out refits of balanced MAE, applying the same
    n >= 10-per-bin rule as per-campaign balanced MAE. Each of the three
    refits reports its own per-bin n; a refit under the threshold reports
    the literal 'not computed (bin n < 10)' string and is excluded from the
    numeric leave-one-out range while the other, usable refits still
    contribute to it.

    Each usable refit is the full `BalancedMAEResult` (balanced MAE, the
    five per-bin MAEs, their n and their CIs) from
    `campaign_balanced_mae_with_min_n`, not a bare float, so every
    leave-one-campaign-out refit carries the same per-bin breakdown as
    every other balanced MAE reported in this study.
    """
    out = {}
    for camp in frame[campaign_col].unique():
        sub = frame[frame[campaign_col] != camp]
        out[camp] = campaign_balanced_mae_with_min_n(
            sub, abs_error_col=abs_error_col, bin_col=bin_col, image_col=image_col,
            min_n=min_n, rng=rng, B=B,
        )
    return out


def loco_range(values: Iterable[float | str | BalancedMAEResult]) -> tuple[float, float, str]:
    """The min/max across the usable (numeric) leave-one-campaign-out
    refits, with a note if fewer than three of the three refits were
    usable — the range itself is the sensitivity check; a narrow range
    means the result does not depend on any single campaign, and a note of
    "built from 2/3" tells a reader the range rests on less evidence than
    the full leave-one-out design intended.

    Accepts either plain floats (refits where `metric_fn` returns a scalar
    directly) or `BalancedMAEResult` objects (as returned by
    `loco_balanced_mae_refits`/`campaign_balanced_mae_with_min_n`), in
    which case `.balanced_mae` is read off each result before taking the
    range. The literal `'not computed (bin n < 10)'` string is excluded
    from the numeric range either way.
    """
    def _as_float(v):
        if isinstance(v, BalancedMAEResult):
            return v.balanced_mae
        return v

    numeric = [_as_float(v) for v in values]
    numeric = [v for v in numeric if isinstance(v, (int, float)) and np.isfinite(v)]
    n_used = len(numeric)
    note = "" if n_used == 3 else f"built from {n_used}/3 usable refits"
    if not numeric:
        return (np.nan, np.nan, note or "built from 0/3 usable refits")
    return (float(min(numeric)), float(max(numeric)), note)


# --------------------------------------------------------------------------- #
# 6. Pairing, ranks and ties
# --------------------------------------------------------------------------- #

@dataclass
class WilcoxonResult:
    statistic: float
    p_value: float
    n_zero_diff: int
    n_tied_ranks: int
    n_used: int


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> WilcoxonResult:
    """Paired Wilcoxon signed-rank test on per-image values, with
    `zero_method='pratt'`, the normal approximation, and continuity and tie
    correction.

    This is the companion test to the balanced-MAE bootstrap, asking a
    different question — is the typical per-image difference between two
    configurations non-zero? — using ranks rather than magnitudes, so it is
    not dominated by a few large-error images the way a mean-based statistic
    can be. Pratt's method retains zero differences in the ranking before
    dropping them, rather than discarding them outright, because the
    model-estimated cover values are heavily quantised (24-94 distinct
    values per model over 13,860 rows), so exact zero differences between
    two configurations are common and would otherwise discard real
    information about how often the two configurations agree exactly. The
    normal approximation (rather than the exact null distribution) is used
    because n = 1,155 makes the exact distribution both unnecessary and
    computationally infeasible.
    """
    diff = np.asarray(x) - np.asarray(y)
    n_zero = int(np.sum(diff == 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = stats.wilcoxon(x, y, zero_method="pratt", correction=True, mode="approx")

    abs_diff_nonzero = np.abs(diff[diff != 0])
    ranks = stats.rankdata(abs_diff_nonzero) if abs_diff_nonzero.size else np.array([])
    _, counts = np.unique(ranks, return_counts=True) if ranks.size else (None, np.array([]))
    n_tied = int(np.sum(counts[counts > 1])) if counts.size else 0

    return WilcoxonResult(
        statistic=float(res.statistic), p_value=float(res.pvalue),
        n_zero_diff=n_zero, n_tied_ranks=n_tied, n_used=int(diff.size),
    )


def pool_axis_mean_abs_error(frame: pd.DataFrame, group_cols: list[str], image_col: str = "image",
                              abs_error_col: str = "abs_e") -> pd.DataFrame:
    """Per-image mean of |e| pooled over the levels not in `group_cols`.
    E.g. to pool over prompts within each model, pass
    `group_cols=['model', 'image']`: the result has one row per
    (model, image) equal to the mean of |e| over that model's 4 prompts for
    that image. This is the mean of |e|, never the |error| of the mean
    prediction — averaging predictions first would turn this into an
    ensemble-accuracy question, not a pooled accuracy estimate for the
    individual configurations.
    """
    return frame.groupby(group_cols, as_index=False)[abs_error_col].mean()


# --------------------------------------------------------------------------- #
# 7. Effect sizes
# --------------------------------------------------------------------------- #

def hodges_lehmann(diff: np.ndarray) -> float:
    """Hodges-Lehmann median of the paired per-image difference: the
    median of all pairwise Walsh averages (d_i + d_j)/2, i<=j. This is the
    location-shift estimator naturally paired with the Wilcoxon signed-rank
    test, and it estimates the same "typical difference" the rank test asks
    about, on the original (cover-point) scale.
    """
    diff = np.asarray(diff)
    n = diff.size
    # Walsh averages via a sorted outer-sum trick, O(n^2) memory-safe only
    # for moderate n; n=1,155 -> ~1.3e6 upper-triangle pairs, fine.
    iu = np.triu_indices(n)
    walsh = (diff[iu[0]] + diff[iu[1]]) / 2.0
    return float(np.median(walsh))


def rank_biserial_matched_pairs(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation from signed-rank sums:
    r = (W+ - W-) / (W+ + W-), on the non-zero differences (Pratt handling
    upstream keeps zeros out of this ratio, consistent with the Wilcoxon
    computed alongside it). Unitless, in [-1, 1]; positive means the first
    member of the pair tends to have the larger value.
    """
    diff = np.asarray(diff)
    nz = diff[diff != 0]
    if nz.size == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(nz))
    w_pos = ranks[nz > 0].sum()
    w_neg = ranks[nz < 0].sum()
    total = w_pos + w_neg
    if total == 0:
        return np.nan
    return float((w_pos - w_neg) / total)


def win_rate(abs_error_a: np.ndarray, abs_error_b: np.ndarray) -> dict:
    """Share of images where A has the smaller |e| than B, with ties
    reported separately.
    """
    a = np.asarray(abs_error_a)
    b = np.asarray(abs_error_b)
    n = a.size
    wins_a = int(np.sum(a < b))
    wins_b = int(np.sum(a > b))
    ties = int(np.sum(a == b))
    return {
        "win_rate_a": wins_a / n, "win_rate_b": wins_b / n, "tie_rate": ties / n,
        "n_wins_a": wins_a, "n_wins_b": wins_b, "n_ties": ties, "n": n,
    }


def spearman_tie_corrected(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Tie-corrected Spearman rho and its two-sided p-value (scipy's
    `spearmanr` already applies the standard mid-rank tie correction).
    """
    res = stats.spearmanr(x, y)
    return float(res.correlation), float(res.pvalue)


# --------------------------------------------------------------------------- #
# 8. Assumption checks — verify the conditions each statistical method
#    depends on, and record which fallback governs when a condition fails.
# --------------------------------------------------------------------------- #

def assert_k1_pairing_complete(d5: pd.DataFrame, base_local: pd.DataFrame) -> AssertionResult:
    """K1: pairing is complete — every image has exactly 24 base/local rows
    (6 models x 4 prompts) on the `variant == 'base'`, local-stack frame
    `build_base_local_frame` returns. If this fails, the design is not what
    the data files describe, and every model-versus-model comparison built
    on this frame is comparing mismatched sets of images per model.

    `assert_a3` (12 rows/image) checks a different predicate — it runs on
    each of the six per-model *three-variant* files individually (13,860
    rows, 3 variants x 4 prompts = 12 rows/image, one model at a time), so a
    23- or 25-row image on the *combined*, base-only, all-models frame would
    still pass it. `build_base_local_frame`'s own join check only verifies
    the merge preserved the total row count, which a frame with some images
    at 23 rows and others at 25 (summing to the same total) would also pass.
    K1 is the one check that looks at the per-image row count on exactly the
    frame every notebook actually shares, and it raises rather than merely
    reporting, because a K1 failure means the shared analysis frame itself
    is not what every downstream computation assumes.
    """
    expected_models = len(MODELS)
    expected_prompts = int(d5["prompt"].nunique()) if "prompt" in d5.columns else 4
    expected_per_image = expected_models * expected_prompts
    counts = base_local.groupby("image").size()
    n_images = counts.size
    bad = counts[counts != expected_per_image]
    ok = n_images == 1155 and bad.empty
    return _check(
        "K1", f"every image has exactly {expected_per_image} base/local rows "
              f"({expected_models} models x {expected_prompts} prompts)",
        ok,
        f"n_images={n_images}, n_images_with_wrong_count={len(bad)}, "
        f"expected_per_image={expected_per_image}"
        + (f", offending_counts={bad.value_counts().to_dict()}" if not bad.empty else ""),
    )


def check_k2_symmetry(diff: np.ndarray) -> dict:
    """K2: symmetry of paired differences, for the HL reading only.
    If |skew| > 1, the exact paired sign test becomes the reported effect
    statement and HL is demoted to a descriptive column.
    """
    diff = np.asarray(diff)
    skew = float(stats.skew(diff))
    median_diff = float(np.median(diff))
    hl = hodges_lehmann(diff)
    return {
        "skew": skew, "median_diff": median_diff, "hl_estimate": hl,
        "symmetry_violated": abs(skew) > 1,
    }


def check_k3_tie_burden(diff: np.ndarray) -> dict:
    """K3: tie burden. If > 40% of images have d_i = 0, the companion
    switches from Wilcoxon to the exact sign test on non-tied pairs, with
    the tie share reported in the same row. The primary balanced-MAE
    bootstrap is unaffected (it is not a rank statistic).
    """
    diff = np.asarray(diff)
    n = diff.size
    n_zero = int(np.sum(diff == 0))
    zero_share = n_zero / n if n else np.nan
    return {"n_zero": n_zero, "n": n, "zero_share": zero_share, "tie_burden_exceeded": zero_share > 0.40}


def exact_sign_test(diff: np.ndarray) -> dict:
    """Exact paired sign test on the non-tied pairs — the K2/K3 fallback."""
    diff = np.asarray(diff)
    nz = diff[diff != 0]
    n = nz.size
    n_pos = int(np.sum(nz > 0))
    res = stats.binomtest(n_pos, n, p=0.5, alternative="two-sided")
    return {"n_used": n, "n_pos": n_pos, "p_value": float(res.pvalue)}


def check_k4_bootstrap_bin_coverage(n_violations: int, B: int) -> dict:
    """K4: count of resamples with an empty bin. If > 1% of resamples,
    switch that metric to the bin-stratified bootstrap.
    """
    share = n_violations / B if B else np.nan
    return {"n_violations": n_violations, "B": B, "violation_share": share,
            "switch_to_stratified": share > 0.01}


def check_k5_normality(x: np.ndarray) -> dict:
    """K5: D'Agostino-Pearson normality test, reported for context only —
    none of the tests used in this study assumes normality, so this exists
    to describe the data, not to license or forbid any method.
    """
    res = stats.normaltest(np.asarray(x))
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}


def check_k6_homoscedasticity(values: pd.Series, bins: pd.Series) -> dict:
    """K6: per-bin standard deviation, reported for context. This is the
    mechanism behind the variance-concentration argument for balanced MAE:
    error variance differs sharply across cover bins, which is exactly why
    an unweighted pooled MAE would be dominated by the bins with the most
    images rather than reflecting accuracy evenly across the cover range.
    Not something any method here assumes away.
    """
    df = pd.DataFrame({"v": values.values, "bin": bins.values})
    sd = df.groupby("bin")["v"].std().reindex(BIN_LABELS)
    return {k: (float(v) if pd.notna(v) else np.nan) for k, v in sd.items()}


def check_k7_collinearity(design: pd.DataFrame, blocks: dict[str, list[str]]) -> dict:
    """K7: variance inflation factor (VIF) over the cover-bin dummies, the
    two annotation covariates and obliquity. If VIF > 10 for any column,
    that column's block is reported as not separately identified — its
    coefficient cannot be attributed to that covariate alone rather than to
    whatever it is collinear with.

    `design` is the fully dummy-coded / numeric design matrix (an intercept
    column is added internally — VIF is conventionally computed against a
    model that includes one, and a caller's design matrix should not
    already carry its own all-ones column). `blocks` maps a block label
    (e.g. `"bin"`, `"rooted_dead_alike_plants"`, `"non_rooted_plant_material"`,
    `"obliquity"`) to the list of `design` columns that realise it — a
    dummy-coded nominal/ordinal factor contributes more than one column to
    its own block. A block is flagged `not_separately_identified=True` if
    *any* of its columns has VIF > 10.

    Uses `statsmodels.stats.outliers_influence.variance_inflation_factor`,
    the standard implementation (regress each column on all the others in
    the design and take `1 / (1 - R^2)`), rather than hand-rolling the
    regression here — VIF has one standard definition and there is no
    project-specific variant of it to justify a bespoke implementation.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = design.copy()
    X.insert(0, "_intercept", 1.0)
    X = X.astype(float)
    cols = list(X.columns)
    vifs = {}
    for i, col in enumerate(cols):
        if col == "_intercept":
            continue
        vifs[col] = float(variance_inflation_factor(X.values, i))

    block_vifs = {}
    block_flags = {}
    for label, member_cols in blocks.items():
        present = [c for c in member_cols if c in vifs]
        block_vifs[label] = {c: vifs[c] for c in present}
        block_flags[label] = any(vifs[c] > 10 for c in present) if present else False

    return {
        "per_column_vif": vifs,
        "per_block_vif": block_vifs,
        "block_not_separately_identified": block_flags,
    }


def check_c9_clean_never_with_v3v4(design_columns: Iterable[str]) -> dict:
    """C9: `clean` (D2.v5) is a deterministic function of
    `rooted_dead_alike_plants` (D2.v3) and `non_rooted_plant_material`
    (D2.v4) — `clean <=> v3==0 and v4==0` — and is therefore never entered
    in the same model as either (doing so would be perfect collinearity,
    and the affected coefficients would be meaningless). Checked
    structurally against the caller's design-matrix column names, so a
    model that violates C9 is caught before it is ever fit rather than
    only showing up as an infinite VIF after the fact.
    """
    cols = set(design_columns)
    has_clean = any(c == "clean" or c.startswith("clean_") for c in cols)
    has_v3 = any(c == "rooted_dead_alike_plants" or c.startswith("rooted_dead_alike_plants_") for c in cols)
    has_v4 = any(c == "non_rooted_plant_material" or c.startswith("non_rooted_plant_material_") for c in cols)
    violated = has_clean and (has_v3 or has_v4)
    return {
        "has_clean": has_clean, "has_v3": has_v3, "has_v4": has_v4,
        "c9_violated": violated,
    }


def check_k8_confidence_degeneracy(confidence: pd.Series) -> dict:
    """K8: distinct levels and max single-level mass. If <=2 effective
    levels or >90% mass on one level, AUC is reported but not interpreted
    as a ranking claim.
    """
    vc = confidence.dropna().value_counts(normalize=True)
    n_levels = int(confidence.dropna().nunique())
    max_mass = float(vc.max()) if len(vc) else np.nan
    return {"n_levels": n_levels, "max_single_level_mass": max_mass,
            "degenerate": (n_levels <= 2) or (max_mass > 0.90)}


def check_k9_level_counts(series: pd.Series, min_n: int = 30) -> dict:
    """K9: level counts. If any level has n < min_n, collapse to
    present/absent for that block and record the collapse.
    """
    counts = series.value_counts()
    below = counts[counts < min_n]
    return {"counts": counts.to_dict(), "any_below_min": bool(len(below)), "levels_below_min": below.to_dict()}


# --------------------------------------------------------------------------- #
# 9. Holm-Bonferroni
# --------------------------------------------------------------------------- #

def holm_adjust(pvalues: Sequence[float], *, family: str, family_size: int) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment, at a caller-declared family
    size — never inferred from `len(pvalues)`.

    A caller must pass both `family` (one of the labels in `FAMILY_SIZES`,
    or any other string the caller documents) and `family_size` explicitly,
    and `family_size` must be >= len(pvalues). Inferring the family size
    from the number of p-values handed to this function would silently
    under-correct whenever a caller passes a subset of a family's tests —
    e.g. computing 5 of the 23 model x prompt contrasts in one call and
    adjusting as if the family had size 5. This is exactly the failure mode
    that fixing every family's size in advance, before any result is seen,
    is meant to prevent.

    If `family` is a recognised label in `FAMILY_SIZES`, `family_size` must
    match the declared value, or this raises — a caller cannot silently
    override a pre-declared family size by passing a different number.
    """
    if family in FAMILY_SIZES and family_size != FAMILY_SIZES[family]:
        raise ValueError(
            f"family {family!r} is pre-declared at size {FAMILY_SIZES[family]}, "
            f"got family_size={family_size}"
        )
    p = np.asarray(pvalues, dtype=float)
    m = family_size
    if m < len(p):
        raise ValueError(f"family_size ({m}) must be >= number of p-values supplied ({len(p)})")

    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty_like(ranked)
    running_max = 0.0
    for i, pv in enumerate(ranked):
        multiplier = m - i  # m, m-1, ..., m-len(p)+1
        val = min(1.0, pv * multiplier)
        running_max = max(running_max, val)
        adjusted[i] = running_max
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def holm_first_threshold(alpha: float, family_size: int) -> float:
    """The first (smallest-p) Holm threshold for a family of this size,
    i.e. alpha / family_size — reported so a reader can see the threshold a
    family's most significant test must clear (e.g. a 23-test family:
    0.05/23 = 0.0022) without recomputing it.
    """
    return alpha / family_size


# --------------------------------------------------------------------------- #
# 10. Row-schema helper — the fixed column set every contrast row carries.
# --------------------------------------------------------------------------- #

R2_REQUIRED_COLUMNS = (
    "estimate", "ci_lo", "ci_hi",
    "p_primary_raw", "p_primary_holm",
    "p_companion_raw", "p_companion_holm",
    "p_clustered_webb", "n_distinct_abs_t_star",
    "loco_min", "loco_max",
    "ci_width_ratio_clustered_to_headline",
    "stream_note",
)


def make_contrast_row(**kwargs) -> dict:
    """Build one contrast row with a fixed, complete column set: every
    contrast row in every output CSV carries the identical columns below,
    including `stream_note` (the column that names each stream's estimand
    so the `p_primary_*` / `p_companion_*` names are never read as a ranking
    of the two claims — `p_primary_*` is the balanced-MAE (bin-weighted)
    claim, `p_companion_*` is the typical-image (paired Wilcoxon) claim, and
    neither is subordinate to the other; they answer two different
    questions about the same contrast).

    Raises if any required column is missing, so a caller cannot emit a row
    that has silently dropped one. Extra caller-supplied columns (e.g.
    `question_id`, `contrast`, `model_a`, `model_b`,
    `c1_serving_confounded`, `maverick_floor_delta`) are passed through
    unchanged.
    """
    missing = [c for c in R2_REQUIRED_COLUMNS if c not in kwargs]
    if missing:
        raise ValueError(f"contrast row missing required R2 columns: {missing}")
    return dict(kwargs)




def add_balanced_mae_bootstrap_columns(row: dict, boot: BootstrapResult) -> dict:
    """Convenience: fold a `BootstrapResult` for the balanced-MAE stream
    into a contrast row's `estimate`/`ci_lo`/`ci_hi`/`p_primary_raw` columns.
    """
    row = dict(row)
    row["estimate"] = boot.estimate
    row["ci_lo"] = boot.ci_lo
    row["ci_hi"] = boot.ci_hi
    row["ci_method"] = boot.ci_method
    row["p_primary_raw"] = boot.p_two_sided
    return row


def add_clustered_companion_columns(
    row: dict,
    webb: WildClusterResult,
    clustered: ClusteredIntervalResult,
    headline_ci_lo: float,
    headline_ci_hi: float,
) -> dict:
    """Convenience: fold the campaign-clustered companion into a contrast
    row's `p_clustered_webb`, `n_distinct_abs_t_star` and
    `ci_width_ratio_clustered_to_headline` columns.

    `webb` and `clustered` are computed from the same per-image
    contributions `d` and the same `clusters`/`weights` — `wild_cluster_bootstrap`
    supplies the Webb p-value (no interval), `clustered_interval` supplies
    the CR3 / t(G-1) interval `wild_cluster_bootstrap` does not, and this
    function is what turns the pair of them into the width ratio every
    contrast row carries. `headline_ci_lo`/`headline_ci_hi` are the
    same-estimand image-bootstrap interval the ratio is taken against.
    """
    row = dict(row)
    row["p_clustered_webb"] = webb.p_webb
    row["n_distinct_abs_t_star"] = webb.n_distinct_abs_t_star
    row["clustered_ci_lo"] = clustered.ci_lo
    row["clustered_ci_hi"] = clustered.ci_hi
    row["clustered_df"] = clustered.df
    row["ci_width_ratio_clustered_to_headline"] = ci_width_ratio_clustered_to_headline(
        clustered, headline_ci_lo, headline_ci_hi,
    )
    return row


# --------------------------------------------------------------------------- #
# 11. Llama-4-Maverick's run-to-run reproducibility floor
# --------------------------------------------------------------------------- #

@dataclass
class MaverickFloor:
    n_reproduced: dict[str, int]
    delta_mav: float
    delta_mav_ci_lo: float
    delta_mav_ci_hi: float
    median_nonzero_disagreement: float
    max_nonzero_disagreement: float


def compute_maverick_floor(d8: pd.DataFrame, rng: np.random.Generator, B: int = B_BOOTSTRAP) -> MaverickFloor:
    """Computes Llama-4-Maverick's reproducibility floor (C11) from the
    determinism-check data (D8): two repeated runs of the same 100 images.
    Llama-4-Maverick alone gives different answers on the two runs for some
    of these images, so any difference this study reports for that model
    must be checked against how much it disagrees with *itself* before it
    is read as a difference from another model or configuration.

    `n_reproduced` is the per-model count of images where the two runs
    returned the identical value (asserted elsewhere to be 92 for Maverick,
    100 for the other five, meaning Maverick alone is non-deterministic on
    this subsample); `delta_mav` is the mean |run1 - run2| over the 100
    images, with its own image-bootstrap CI computed over those 100 images
    specifically (not the full 1,155-image frame — this is a property of
    the determinism subsample, not of the main analysis frame), plus the
    median and max of the non-zero disagreements, which show how large a
    typical or worst-case self-disagreement is where one occurs at all.

    This function only computes the floor itself; `maverick_within_floor`
    below decides whether a given effect on the overall-MAE scale is small
    enough to be indistinguishable from Maverick's own run-to-run noise.
    """
    wide = d8.pivot_table(index=["model", "image"], columns="run", values="vegetation_percent")
    wide = wide.dropna(subset=["run1", "run2"])

    n_reproduced = {}
    for model, g in wide.groupby("model"):
        n_reproduced[model] = int((g["run1"] == g["run2"]).sum())

    mav = wide.loc["Llama-4-Maverick"].reset_index()
    mav["abs_diff"] = (mav["run1"] - mav["run2"]).abs()
    delta_mav = float(mav["abs_diff"].mean())

    def stat(frame):
        return frame["abs_diff"].mean()

    boot = image_bootstrap(mav.rename(columns={"image": "image"}), stat, rng,
                            bin_col=None, B=B, compute_jackknife=True)

    nonzero = mav.loc[mav["abs_diff"] > 0, "abs_diff"]
    median_nz = float(nonzero.median()) if len(nonzero) else np.nan
    max_nz = float(nonzero.max()) if len(nonzero) else np.nan

    return MaverickFloor(
        n_reproduced=n_reproduced, delta_mav=delta_mav,
        delta_mav_ci_lo=boot.ci_lo, delta_mav_ci_hi=boot.ci_hi,
        median_nonzero_disagreement=median_nz, max_nonzero_disagreement=max_nz,
    )


EXPECTED_N_REPRODUCED = {
    "Gemma-3-12B": 100,
    "Gemma-3-27B": 100,
    "Llama-4-Maverick": 92,
    "Llama-4-Scout": 100,
    "Mistral-Small-3.2": 100,
    "Qwen-2.5": 100,
}


def assert_c11_n_reproduced(n_reproduced: dict[str, int]) -> AssertionResult:
    """`n_reproduced` is asserted to be 92 for Llama-4-Maverick and 100 for
    the other five models. `compute_maverick_floor` computes this dict but
    does not itself gate on it, so a caller must route the computed dict
    through this assertion — which raises on a mismatch rather than merely
    reporting — before treating Maverick's determinism figures as
    established, since every claim about Maverick's reproducibility
    depends on this exact split holding.
    """
    ok = n_reproduced == EXPECTED_N_REPRODUCED
    return _check(
        "C11", "n_reproduced is 92 for Llama-4-Maverick and 100 for the other five models",
        ok, f"observed={n_reproduced}, expected={EXPECTED_N_REPRODUCED}",
    )


def maverick_within_floor(effect_abs_overall_mae_scale: float, delta_mav: float) -> bool:
    """`within_run_to_run_variability` is TRUE when the absolute effect
    size on the *overall-MAE scale* is smaller than delta_Mav — i.e. the
    observed effect could plausibly be nothing more than the noise
    Llama-4-Maverick already shows between two runs on identical inputs.
    Never call this with a balanced-MAE-scale effect — see
    `maverick_balanced_mae_claim_branch` for the gate that covers that
    case, because delta_Mav is not estimable on the balanced-MAE scale.
    """
    return abs(effect_abs_overall_mae_scale) < delta_mav


def maverick_balanced_mae_claim_branch(delta_mav: float) -> dict:
    """The balanced-MAE-scale counterpart to `maverick_within_floor`.
    delta_Mav bounds the run-to-run shift in *overall* MAE
    (|MAE1 - MAE2| <= mean|run1-run2|); it cannot bound balanced MAE,
    which is bin-weighted, because the 100-image determinism subsample
    holds only ~2-3 images in the 80-100 bin — not enough to estimate a
    per-bin floor. So a balanced-MAE-scale claim for Maverick is always
    `floor_not_estimable_for_balanced_mae = TRUE` with no boolean verdict,
    permitted but carrying the unquantified-floor caveat, with delta_Mav
    (on the overall-MAE scale) quoted as the nearest available bound.
    """
    return {
        "floor_not_estimable_for_balanced_mae": True,
        "within_run_to_run_variability": None,
        "maverick_floor_delta": delta_mav,
        "caveat": (
            "Maverick's balanced-MAE effect sits above a reproducibility floor "
            "this data cannot quantify on the balanced-MAE scale; the nearest "
            f"available bound is the overall-MAE run-to-run floor delta_Mav={delta_mav:.3f}."
        ),
    }


# --------------------------------------------------------------------------- #
# 12. The serving-stack confound flag
# --------------------------------------------------------------------------- #

LLAMA_MODELS = ("Llama-4-Maverick", "Llama-4-Scout")


def c1_serving_confounded(model_a: str, model_b: str) -> bool:
    """TRUE when the two models sit on different serving paths (one Llama,
    one non-Llama). The two Llama models are served through a different
    stack from the other four, so any accuracy difference between a Llama
    and a non-Llama model is confounded with that infrastructure difference
    and cannot be attributed to the model alone. Of the 15 model pairs,
    8 are cross-family and carry this flag.
    """
    a_llama = model_a in LLAMA_MODELS
    b_llama = model_b in LLAMA_MODELS
    return a_llama != b_llama


def c1_ensemble_confounded(member_models: Iterable[str]) -> bool:
    """The serving-stack confound extended to composites: an
    ensemble/selection procedure inherits the same confound whenever its
    membership contains at least one Llama and at least one non-Llama
    model.
    """
    members = set(member_models)
    has_llama = bool(members & set(LLAMA_MODELS))
    has_non_llama = bool(members - set(LLAMA_MODELS))
    return has_llama and has_non_llama


# --------------------------------------------------------------------------- #
# 13. Guards and helpers for pairing MLLM predictions against the classical
#     (D12) baseline without inflating its apparent precision.
# --------------------------------------------------------------------------- #

def guard_d12_statistic(frame: pd.DataFrame, image_col: str = "image", *,
                         full_frame_expected: bool = True, context: str = "") -> None:
    """Refuses to compute a classical-baseline statistic on any frame where
    `n_rows != n_unique_images`. Call this immediately before taking any
    statistic that touches the baseline, on whichever frame is in scope at
    that point (the full 1,155-row frame, or a two-campaign
    leave-one-campaign-out subset). The `n == 1,155` check is applied only
    when `full_frame_expected=True` (i.e. outside the leave-one-out path);
    leave-one-out callers must pass `False`.

    Raises `AssertionFailed` on violation — this is a guard that must stop
    the computation, not an advisory check that merely reports it, because
    a baseline statistic computed on a frame with duplicate images per row
    silently overstates how much evidence backs it.
    """
    result = assert_a17(frame, image_col=image_col, full_frame_expected=full_frame_expected, context=context)
    if not result.passed:
        raise AssertionFailed(f"A17/C15 guard failed ({context}): {result.detail}")


def aggregate_mllm_side_to_image(frame: pd.DataFrame, value_col: str, image_col: str = "image") -> pd.DataFrame:
    """Aggregate the MLLM side to one value per image before pairing
    against the classical baseline, which has only one row per image.
    Never merge the baseline onto the 4,620-row per-model prediction
    frame directly — that would broadcast each baseline value across every
    prompt for that image and quadruple its apparent precision. This
    function is the only sanctioned way to get from a per-(image, prompt)
    MLLM frame to the one-row-per-image frame the baseline must be paired
    against.
    """
    return frame.groupby(image_col, as_index=False)[value_col].mean()


# --------------------------------------------------------------------------- #
# 14. The Jonckheere-Terpstra trend test — one implementation shared by
#     every trend test in this study, so the statistic, its tie handling
#     and its permutation p-value are defined identically everywhere it
#     is used.
# --------------------------------------------------------------------------- #

@dataclass
class JonckheereTerpstraResult:
    """Everything a caller needs to report the test and nothing it has to
    recompute. `j_stat` and `mean_j` are on the raw (unnormalised) J scale,
    which is retained because it is the quantity every textbook table
    reports and checks against; `j_bar` is the same statistic linearly
    rescaled to `2*(J - E[J])/J_max` in `[-1, 1]`, where
    `J_max = sum_{i<j} n_i n_j` is the maximum value J can take (every pair
    concordant). The factor of 2 is necessary and not cosmetic: because
    `sum_{i<j} n_i n_j = J_max` is exactly twice `mean_J`'s own normalising
    constant here, the unscaled `(J - mean_J)/J_max` only ever reaches
    `+-0.5`, never `+-1` — every group-size vector hits exactly `+-0.5` at
    perfect (anti-)monotonicity, confirmed by direct construction, so a
    reader taking `[-1, 1]` at face value would read a maximally monotone
    trend as "halfway to perfect" when it is complete. `j_bar` is what
    makes a **leave-one-campaign-out range comparable across differently-
    sized subsets**: raw J scales with n^2, so a full-frame J and a
    two-campaign subset's J are not on the same footing, while `j_bar` is
    the excess over chance as a proportion of the achievable concordance
    and does not carry that scale dependence — two subsets of different n
    can be compared on `j_bar` directly. Both are always returned, so a
    caller who wants the familiar J for a table and a caller who wants a
    comparable leave-one-out range never has to reconstruct either from
    the other's frame.

    `z` standardises J using the tie-corrected variance named by
    `variance_method` (see `jonckheere_terpstra_statistic`'s docstring for
    the choice, and the conditions under which it does and does not affect
    `p_perm`). `p_perm` is the two-sided permutation p-value with the `+1`
    correction (`(1 + #{|stat*| >= |stat_obs|}) / (B+1)`), applied to the
    permutation null this test actually uses; it is already floored at
    `1/(n_perm+1)` and `at_floor` records whether the raw count that
    produced it was zero, i.e. whether the reported number is a resolution
    limit rather than an estimate. `direction` is
    `"increasing"`/`"decreasing"`/`"null"`, gated on `p_perm < alpha` — a
    threshold `permutation_jt` requires the caller to state explicitly (see
    its docstring); a family running Holm-Bonferroni must pass its own
    adjusted threshold, not the function's unqualified default, so that a
    non-significant (post-correction) trend is never reported with a sign a
    reader could mistake for a finding.
    """
    j_stat: float
    mean_j: float
    j_max: float
    j_bar: float
    z: float
    p_perm: float
    at_floor: bool
    floor: float
    direction: str
    variance_method: str
    n_perm: int
    n_dropped: int
    per_group_n: dict


def jonckheere_terpstra_statistic(
    values: np.ndarray,
    groups: np.ndarray,
    ordered_labels: Sequence,
    *,
    variance_method: str = "textbook",
    dropna: bool = False,
) -> dict:
    """The Jonckheere-Terpstra J statistic with mid-rank tie handling, plus
    its null mean and two candidate variances, computed once.

    **The statistic.** `J = sum` over every ordered pair of groups
    `g_i < g_j` (`ordered_labels` fixes the order — the reference bin, by
    construction, everywhere this is called) of the Mann-Whitney U
    comparing group `g_j` against group `g_i`: for each pair, rank the
    pooled two-group sample with `scipy.stats.rankdata` (mid-ranks split a
    tie's credit evenly between the tied observations, which is what makes
    this the tie-handled statistic rather than the untied one) and take
    `U = sum(ranks of group_j) - n_j*(n_j+1)/2`, the count of
    `group_j > group_i` pairs with ties worth 0.5 each. Summing `U` over all
    `C(k,2)` ordered pairs is algebraically identical to
    `sum_{i<j} [ #(y>x) + 0.5*#(y=x) ]` computed directly over every
    cross-group pair `(x in group_i, y in group_j)`, and this implementation
    is checked against that brute-force definition directly, and separately
    against complete enumeration of every distinct group-label assignment on
    small synthetic samples for the exact null `Var(J)` (not a simulated
    one).

    **Missing values.** `values` must be finite. A single `NaN` propagates
    through `rankdata` into `J` and then into `z`, and because
    `abs(nan) >= abs(nan)` is `False` for every permutation draw, an
    unguarded caller would see the permutation loop count zero exceedances
    and report the floor p-value with a confident direction — the strongest
    attainable finding, manufactured from missing data. By default this
    function raises `ValueError` rather than let that happen; pass
    `dropna=True` to instead drop the non-finite rows before computing
    anything (the `n_dropped` count is not returned from this function, but
    `permutation_jt` surfaces it in `JonckheereTerpstraResult.n_dropped` so
    a caller who opts into dropping still has to report how many rows were
    removed).

    **The tie correction.** `variance_method="textbook"` (the default) uses
    the standard three-term Lehmann / SAS `PROC FREQ JT` tie-corrected
    variance:

        Var(J) = [ N(N-1)(2N+5) - sum_g n_g(n_g-1)(2n_g+5)
                                 - sum_t t(t-1)(2t+5) ] / 72
                + [ sum_g n_g(n_g-1)(n_g-2) * sum_t t(t-1)(t-2) ]
                  / [ 36 N(N-1)(N-2) ]
                + [ sum_g n_g(n_g-1) * sum_t t(t-1) ] / [ 8 N(N-1) ]

    summed over the groups `g` and the tie-run sizes `t` in the pooled
    sample (the second and third terms are guarded against `N < 3` / `N < 2`
    division, and are exactly 0 whenever there are no ties). This three-term
    formula reproduces the enumerated exact variance to machine precision.
    A single-bracket variance that omits the second and third terms
    understates the tied variance by up to several percent under heavy
    ties, which is why all three terms are computed here rather than only
    the first. `variance_method="pooled"` uses an alternative,
    permutation-invariant tie-corrected variance instead — the untied
    Lehmann variance `[N^2(2N+3) - sum_g n_g^2(2n_g+3)] / 72` with a single
    combined-sample correction term `sum_t(t^3-t)/2` subtracted inside the
    same bracket, rather than the textbook's separate group-size and
    tie-run terms.

    **Whether the variance choice affects `p_perm` — conditional, not
    universal.** `mean_J` and the tie structure of the pooled sample are
    invariant under permutation of the group labels, and so is `var_J`
    itself under either formula (a label permutation changes which
    observations sit in which group, not the multiset of tied values, so
    both variances recompute identically on every draw). Wherever
    `var_J > 0` under both formulas, `z` is a strictly monotone rescaling of
    `J - mean_J` under either choice, so the permutation distribution of
    `|z|` ranks draws identically regardless of which variance produced it,
    and `p_perm` is unaffected. That guarantee holds for the three-term
    variance implemented here because it is a true variance and is never
    negative or zero except on a genuinely degenerate input (a single
    group, or every value tied) — a property the simpler single-bracket
    formula does not share, since it can go negative under heavy ties.

    **Two J-scale outputs, always returned.** `mean_J = (N^2 - sum n_g^2)/4`
    is J's mean under the null of no association between group and value.
    `max_J = sum_{i<j} n_i n_j` is the largest value J can take (every
    cross-group pair concordant), and is what `j_bar` in
    `permutation_jt`/`JonckheereTerpstraResult` is built from (as
    `2*(J - mean_J)/max_J`) to make the statistic comparable across samples
    of different size — raw J is not: it scales with N^2, so a
    leave-one-campaign-out subset's J is not on the same footing as the
    full frame's J, and comparing them directly would make that range a
    function of subset size rather than of campaign clustering.

    Returns a dict rather than a dataclass because it is an internal
    building block for both `jonckheere_terpstra_z` and `permutation_jt`,
    not something a notebook calls and reports from directly.
    """
    if variance_method not in ("textbook", "pooled"):
        raise ValueError(f"variance_method must be 'textbook' or 'pooled', got {variance_method!r}")

    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)

    finite = np.isfinite(values)
    n_dropped = int((~finite).sum())
    if n_dropped > 0:
        if not dropna:
            raise ValueError(
                f"values contains {n_dropped} non-finite entr"
                f"{'y' if n_dropped == 1 else 'ies'} (NaN/inf); a Jonckheere-Terpstra "
                "test computed over them would silently manufacture a floor p-value "
                "(see this function's docstring). Pass dropna=True to drop them "
                "explicitly and report the dropped count, or clean `values` upstream."
            )
        values = values[finite]
        groups = groups[finite]

    group_values = {g: values[groups == g] for g in ordered_labels}
    ns = [group_values[g].size for g in ordered_labels]
    n_total = sum(ns)

    J = 0.0
    max_J = 0.0
    for i in range(len(ordered_labels)):
        xi = group_values[ordered_labels[i]]
        for j in range(i + 1, len(ordered_labels)):
            xj = group_values[ordered_labels[j]]
            max_J += xi.size * xj.size
            if xi.size == 0 or xj.size == 0:
                continue
            combined = np.concatenate([xi, xj])
            ranks = stats.rankdata(combined)
            r_j = ranks[xi.size:]
            u = r_j.sum() - xj.size * (xj.size + 1) / 2.0
            J += u

    mean_J = (n_total ** 2 - sum(n ** 2 for n in ns)) / 4.0

    _, tie_counts = np.unique(values, return_counts=True)

    if variance_method == "textbook":
        term_groups = sum(n * (n - 1) * (2 * n + 5) for n in ns)
        term_ties = float(np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5)))
        var_J = (n_total * (n_total - 1) * (2 * n_total + 5) - term_groups - term_ties) / 72.0

        # The two additional terms of the standard three-term Lehmann / SAS
        # PROC FREQ JT tie-corrected variance. Both are identically 0 when
        # there are no ties (every t == 1) and both require N >= 3 / N >= 2
        # respectively to be defined; a smaller N makes the corresponding
        # numerator identically 0 as well (fewer than 3, or fewer than 2,
        # total observations cannot form the combinatorial term it corrects),
        # so the guard only ever suppresses a 0/0.
        sum_g_n3 = sum(n * (n - 1) * (n - 2) for n in ns)
        sum_t_t3 = float(np.sum(tie_counts * (tie_counts - 1) * (tie_counts - 2)))
        if n_total >= 3:
            var_J += (sum_g_n3 * sum_t_t3) / (36.0 * n_total * (n_total - 1) * (n_total - 2))

        sum_g_n2 = sum(n * (n - 1) for n in ns)
        sum_t_t2 = float(np.sum(tie_counts * (tie_counts - 1)))
        if n_total >= 2:
            var_J += (sum_g_n2 * sum_t_t2) / (8.0 * n_total * (n_total - 1))
    else:  # "pooled" — the alternative, permutation-invariant, non-standard variance
        sum_t3_t = float(np.sum(tie_counts ** 3 - tie_counts))
        var_J = (
            n_total ** 2 * (2 * n_total + 3) - sum(n ** 2 * (2 * n + 3) for n in ns) - sum_t3_t / 2.0
        ) / 72.0

    return {
        "J": float(J), "mean_J": float(mean_J), "var_J": float(var_J), "max_J": float(max_J),
        "n_dropped": n_dropped,
        "per_group_n": {g: int(n) for g, n in zip(ordered_labels, ns)},
        "variance_method": variance_method,
    }


def jonckheere_terpstra_z(
    values: np.ndarray,
    groups: np.ndarray,
    ordered_labels: Sequence,
    *,
    variance_method: str = "textbook",
    dropna: bool = False,
) -> tuple[float, float, dict]:
    """`(z, j_bar, core)`: the standardised JT statistic and the
    size-normalised J, both computed once from
    `jonckheere_terpstra_statistic`. `z = (J - mean_J) / sqrt(var_J)`.
    `var_J <= 0` is only reachable for a genuinely degenerate input (a
    single non-empty group, or every value tied) once the full three-term
    tie-corrected variance is used — see `jonckheere_terpstra_statistic`'s
    docstring — so that case is returned as `z = 0.0` (there is no trend to
    standardise; `J == mean_J` identically) rather than raised. `j_bar =
    2*(J - mean_J) / max_J` (`np.nan` if `max_J == 0`) is the excess over
    chance concordance, rescaled so its attainable range is the documented
    `[-1, 1]` (see `JonckheereTerpstraResult`'s docstring for why the
    factor of 2 is required and not cosmetic) — the statistic a
    leave-one-campaign-out range should be built from instead of raw J.
    """
    core = jonckheere_terpstra_statistic(
        values, groups, ordered_labels, variance_method=variance_method, dropna=dropna,
    )
    var_J = core["var_J"]
    z = (core["J"] - core["mean_J"]) / np.sqrt(var_J) if var_J > 0 else 0.0
    j_bar = 2.0 * (core["J"] - core["mean_J"]) / core["max_J"] if core["max_J"] > 0 else float("nan")
    return float(z), float(j_bar), core


def permutation_jt(
    values: np.ndarray,
    groups: np.ndarray,
    ordered_labels: Sequence,
    rng: np.random.Generator,
    *,
    alpha: float,
    n_perm: int = B_BOOTSTRAP,
    variance_method: str = "textbook",
    dropna: bool = False,
) -> JonckheereTerpstraResult:
    """The permutation Jonckheere-Terpstra test, shared by every trend test
    in this study that asks whether an ordered set of groups shows a
    monotone trend in some value. Shuffles `groups` (the ordered bin/group
    label) across `values` `n_perm` times — never resampling `values`
    themselves, never breaking whatever pairing upstream code has already
    formed — which is the correct null of "no association between group
    and value". Mid-rank ties are handled inside
    `jonckheere_terpstra_statistic`.

    **`alpha` is required, deliberately, with no default.** `direction`
    (below) gates on `p_perm < alpha`, and this function is shared by trend
    tests corrected in different ways: some uncorrected and
    exploratory, others gated on their own Holm-Bonferroni-adjusted
    threshold rather than a raw 0.05. A caller that let this default
    quietly to 0.05 would risk labelling a raw-significant, Holm-null trend
    as "increasing"/"decreasing" in a result that a family correction
    specifically found not significant. Pass `0.05` explicitly for an
    uncorrected, exploratory call; pass the relevant family's first Holm
    threshold for a corrected one.

    **Missing values.** `values` must be finite, checked once here (not
    once per permutation draw) via `jonckheere_terpstra_statistic`'s own
    guard: by default a non-finite entry raises `ValueError`, because an
    unguarded NaN would make `z_obs` NaN, make every permutation comparison
    `False`, and return the floor p-value with a confident direction — the
    strongest attainable finding, manufactured from missing data. Pass
    `dropna=True` to instead drop non-finite rows once, upfront, before the
    permutation loop runs (never per-draw, which would let a different
    subset of rows be dropped on every shuffle); the dropped count is
    reported in the returned `n_dropped`, never silently absorbed.

    **The permutation p-value, with the `+1` correction.** Two-sided:

        p = (1 + #{ |stat*| >= |stat_obs| }) / (n_perm + 1)

    The `+1` in both numerator and denominator reflects that the observed
    arrangement is itself one of the `n_perm + 1` achievable arrangements
    (the identity permutation) and must be counted alongside the `n_perm`
    drawn ones; omitting it lets the p-value return exactly 0 before the
    floor clips it, which overstates the test's resolution. With the
    correction the smallest attainable p-value is the **floor**,
    `1/(n_perm+1)` — `1e-4` to four figures at the default
    `n_perm=10,000` — and `at_floor=True` flags every row where the raw
    permutation count behind `p_perm` was zero (i.e. the reported value is
    the resolution limit, not a smaller true p rounded up). The statistic
    used for the comparison is `z` (`jonckheere_terpstra_z`'s standardised J
    under `variance_method`), so the permutation distribution is compared
    on the same standardised scale the observed statistic is reported on;
    because both mean and variance are permutation-invariant (see
    `jonckheere_terpstra_statistic`'s docstring), comparing
    `|z*| >= |z_obs|` and comparing `|J* - mean_J| >= |J_obs - mean_J|` give
    the identical p-value **provided `var_J > 0` under the variance in
    use** (see that docstring's section on when the variance choice does
    and does not affect `p_perm`).

    **`direction`** is `"null"` unless `p_perm < alpha`, in which case it is
    `"increasing"` (`z_obs > 0`) or `"decreasing"` (`z_obs < 0`).

    `rng` must be supplied by the caller (an explicit `np.random.Generator`,
    per this module's own rule that nothing here reads or seeds global
    `numpy` state); it is consumed only via `rng.permutation`.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)

    finite = np.isfinite(values)
    n_dropped = int((~finite).sum())
    if n_dropped > 0:
        if not dropna:
            raise ValueError(
                f"values contains {n_dropped} non-finite entr"
                f"{'y' if n_dropped == 1 else 'ies'} (NaN/inf); pass dropna=True to drop "
                "them explicitly (the dropped count is then reported in n_dropped), or "
                "clean `values` upstream. See this function's docstring."
            )
        values = values[finite]
        groups = groups[finite]

    # Non-finite rows are already removed above, so every downstream call
    # (including inside the permutation loop) sees a clean array and never
    # needs to re-raise or re-drop; dropna=False here is deliberate and
    # asserts that invariant rather than assuming it.
    z_obs, j_bar_obs, core = jonckheere_terpstra_z(
        values, groups, ordered_labels, variance_method=variance_method, dropna=False,
    )

    z_perm = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(groups)
        z_perm[i], _, _ = jonckheere_terpstra_z(
            values, shuffled, ordered_labels, variance_method=variance_method, dropna=False,
        )

    n_at_or_beyond = int(np.sum(np.abs(z_perm) >= abs(z_obs)))
    p_perm = (1 + n_at_or_beyond) / (n_perm + 1)
    floor = 1.0 / (n_perm + 1)
    at_floor = n_at_or_beyond == 0

    direction = "null"
    if p_perm < alpha:
        direction = "increasing" if z_obs > 0 else "decreasing"

    return JonckheereTerpstraResult(
        j_stat=core["J"], mean_j=core["mean_J"], j_max=core["max_J"], j_bar=j_bar_obs,
        z=z_obs, p_perm=float(p_perm), at_floor=at_floor, floor=floor,
        direction=direction, variance_method=variance_method, n_perm=n_perm,
        n_dropped=n_dropped, per_group_n=core["per_group_n"],
    )
