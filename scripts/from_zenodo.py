"""Build the input layout the notebooks read from the published dataset.

The deposited tables and the tables these notebooks read hold the same data in a
different shape. The deposit is normalised for a reader: one row per photograph,
one file per kind of annotation, and the model responses concatenated with a
`model` column. The notebooks read the shape the analysis was written against:
the reference frame as the field spreadsheet, the device and the camera tilt in
their own files, and the responses split into one file per model with the model
named by the filename.

This script performs that conversion, so nothing in `notebooks/` has to change
depending on where its input came from.

    python scripts/from_zenodo.py --dataset path/to/unzipped/dataset --out data/raw

Then point the notebooks at the result, and run `Q0_baseline_index.py` first: it
computes `traditional_baseline.csv`, which the Q9 notebooks read and which no
deposited file carries, because it is derived from the rectified photographs
rather than measured.

Two properties of the result are worth stating plainly:

**Assertion A5 checks the join, not two sources.** The dataset holds one
reference cover per photograph, in `image_metadata.csv`. This script copies that
column into the annotation file, because the analysis expects to find it in
both, so A5 compares a column against its own source. It confirms that every
annotation row matches a photograph, which is what it can establish here.

**`run_failures.csv` is not written.** No notebook reads it, which is itself one
of the checks (A12).
"""

import argparse
import pathlib
import shutil
import sys

import pandas as pd

# The six models, spelled as the analysis expects them in a filename. The `model`
# column of the deposited prediction tables uses these same names.
MODELS = [
    "Gemma-3-12B",
    "Gemma-3-27B",
    "Llama-4-Maverick",
    "Llama-4-Scout",
    "Mistral-Small-3.2",
    "Qwen-2.5",
]

N_IMAGES = 1155
N_LOCAL = 83_160          # 1,155 photographs x 4 prompts x 3 variants x 6 models
N_API = 27_720            # 1,155 x 4 prompts x 6 models, base variant only
N_DETERMINISM = 1_200     # 100 photographs x 6 models x 2 runs, Short prompt
BIN_COUNTS = {"0-20": 933, "20-40": 108, "40-60": 46, "60-80": 38, "80-100": 30}


def read(dataset: pathlib.Path, name: str, expect_rows: int) -> pd.DataFrame:
    path = dataset / name
    if not path.is_file():
        sys.exit(f"missing from the dataset: {name}")
    df = pd.read_csv(path)
    if len(df) != expect_rows:
        sys.exit(f"{name} has {len(df)} rows, expected {expect_rows}")
    print(f"  read  {name:<34} {len(df):>7,} rows")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=pathlib.Path,
                    help="directory holding the unzipped dataset CSVs")
    ap.add_argument("--out", required=True, type=pathlib.Path,
                    help="directory to write the notebook input layout into")
    args = ap.parse_args()

    print(f"reading {args.dataset}")
    meta = read(args.dataset, "image_metadata.csv", N_IMAGES)
    scene = read(args.dataset, "scene_annotations.csv", N_IMAGES)
    local = read(args.dataset, "predictions_local.csv", N_LOCAL)
    api = read(args.dataset, "predictions_api.csv", N_API)
    determinism = read(args.dataset, "determinism_runs.csv", N_DETERMINISM)

    # Fail before writing anything if the dataset is not the one this expects.
    got_models = sorted(local["model"].unique())
    if got_models != sorted(MODELS):
        sys.exit(f"predictions_local.csv holds models {got_models}, expected {sorted(MODELS)}")
    observed_bins = meta["cover_bin"].value_counts().to_dict()
    if {k: observed_bins.get(k, 0) for k in BIN_COUNTS} != BIN_COUNTS:
        sys.exit(f"cover bin counts are {observed_bins}, expected {BIN_COUNTS}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting {args.out}")

    def write(df, name):
        df.to_csv(args.out / name, index=False)
        print(f"  wrote {name:<34} {len(df):>7,} rows")

    # The reference frame. Only Campaign, Filename and Veg % are read; the dates
    # are carried through because they cost nothing and make the file legible.
    write(meta.rename(columns={
        "campaign": "Campaign", "image": "Filename", "reference_fvc": "Veg %",
        "date": "Date", "time": "Time",
    })[["Campaign", "Date", "Time", "Filename", "Veg %"]], "All_frame1155.csv")

    # Scene annotations, with the reference attached. See the note in this
    # script's docstring about what A5 can and cannot check from deposited data.
    write(scene.merge(meta[["image", "reference_fvc"]], on="image", how="left")
               .rename(columns={"reference_fvc": "reference"})
               [["image", "reference", "rooted_dead_alike_plants",
                 "non_rooted_plant_material"]],
          "dead_looking_vegetation_simplified.csv")

    write(meta[["image", "device"]], "image_device.csv")
    write(meta[["image", "obliquity"]], "image_obliquity.csv")

    # Copied rather than read and rewritten. Its `latency_ms` values carry more
    # decimal places than a round trip through a CSV writer preserves, and two
    # of the 1,200 rows change in the last digit if it is rewritten.
    shutil.copyfile(args.dataset / "determinism_runs.csv", args.out / "determinism_runs.csv")
    print(f"  copied determinism_runs.csv            {len(determinism):>7,} rows")

    # One file per model, with `model` dropped: the analysis derives it from the
    # filename, and a column of the same name would be duplicated on load.
    #
    # Row order is set deliberately, not inherited. A bootstrap draws from the
    # array of distinct images in the order they appear in the frame, so two
    # orderings of the same rows give the same point estimates and different
    # interval bounds under the same seed. These are the orders the deposited
    # values were analysed in; the deposit itself stores prompt before variant.
    local_cols = ["image", "prompt", "variant", "vegetation_percent",
                  "confidence", "confidence_parse_method"]
    api_cols = [c for c in local_cols if c != "variant"]
    print()
    for model in MODELS:
        rows = local[local["model"] == model]
        write(rows.sort_values(["image", "variant", "prompt"], kind="stable")[local_cols],
              f"{model}_local_based.csv")
    print()
    for model in MODELS:
        rows = api[api["model"] == model]
        write(rows.sort_values(["image", "prompt"], kind="stable")[api_cols],
              f"{model}_api_based.csv")

    print("\nStill needed, and not part of the deposited tables:")
    print("  rectified/                    from images_rectified.zip, for Q0")
    print("  traditional_baseline.csv      written by Q0_baseline_index.py; run it first")


if __name__ == "__main__":
    main()
