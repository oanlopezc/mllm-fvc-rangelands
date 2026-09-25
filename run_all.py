"""Run every notebook once, in order, and report what each one wrote.

Each notebook is a standalone script that reads the input tables and writes its
own files into `results/`. They are independent of one another with one
exception: `Q0_baseline_index.py` writes the classical baseline that the two Q9
notebooks read, so it runs first.

    pixi run python run_all.py
    pixi run python run_all.py --only Q6_device Q8_low_cover_bins

Notebooks run one at a time. The two longest take over an hour each, and running
them together contends for memory and disk without shortening the total by much.

Figures are written with the Agg backend, so this works over a terminal session
with no display attached.
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
NOTEBOOKS = HERE / "notebooks"
RESULTS = HERE / "results"

# Q0 first: the Q9 notebooks read the baseline it writes. The rest are ordered
# cheapest first, so a run that is going to fail tends to fail early.
ORDER = [
    "Q0_baseline_index",
    "Q2_ensembles",
    "Q6_device",
    "Q9_baseline_part1_pattern",
    "Q7_serving_stack",
    "Q9_baseline_part2_contrasts",
    "Q8_low_cover_bins",
    "Q5_confidence_part1_ranking",
    "Q5_confidence_part2_discrimination",
    "Q3_bins",
    "Q4_preprocessing",
    "Q1_accuracy",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="run just these notebooks, by stem")
    args = ap.parse_args()

    if args.only:
        unknown = [n for n in args.only if n not in ORDER]
        if unknown:
            sys.exit(f"no such notebook: {', '.join(unknown)}\nknown: {', '.join(ORDER)}")
    todo = [n for n in ORDER if not args.only or n in set(args.only)]
    missing = [n for n in todo if not (NOTEBOOKS / f"{n}.py").is_file()]
    if missing:
        sys.exit(f"notebook file not found: {', '.join(missing)}")

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"

    RESULTS.mkdir(exist_ok=True)
    failed = []
    for i, name in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {name}", flush=True)
        started = time.time()
        before = {p.name for p in RESULTS.glob("*.csv")}
        proc = subprocess.run([sys.executable, str(NOTEBOOKS / f"{name}.py")],
                              cwd=HERE, env=env)
        elapsed = time.time() - started
        wrote = len({p.name for p in RESULTS.glob("*.csv")} - before)
        if proc.returncode == 0:
            print(f"        ok, {elapsed / 60:.1f} min, {wrote} new file(s)\n", flush=True)
        else:
            failed.append(name)
            print(f"        FAILED (exit {proc.returncode}) after {elapsed / 60:.1f} min\n",
                  flush=True)
            if name == "Q0_baseline_index":
                sys.exit("Q0 failed and the Q9 notebooks read what it writes; stopping")

    print(f"{len(todo) - len(failed)} of {len(todo)} notebooks ran, "
          f"{len(list(RESULTS.glob('*.csv')))} files in results/")
    for name in failed:
        print(f"  failed: {name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
