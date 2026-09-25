#!/bin/bash
#SBATCH --job-name=fvc-mllm-fullgrid-1gpu
#SBATCH --account=<your-slurm-account>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:45:00
#SBATCH --output=logs/fullgrid_1gpu_%j.out
#SBATCH --error=logs/fullgrid_1gpu_%j.err
#
# One model, one GPU, the full 1,155-photograph grid (run_type=full_grid_fixed). This is the
# script that collected the grid the paper reports. Each of the four single-GPU models is an
# independent worker writing to its own log, and each fits one card, so every model is submitted
# as a separate job.
#
# Qwen runs at the resolution its own checkpoint defaults to (config.IMAGE_MAX_PIXELS = None),
# uncapped. A hosted provider typically caps that budget well below the default to control cost,
# so capping it here would measure a guess at someone else's serving configuration rather than the
# local deployment. Qwen's mean absolute error therefore sits about 1.5 percentage points away
# from its API counterpart, which is a consequence of the resolution difference and not a fault.
#
# --time=03:45:00 stays under the four-hour cap on the GPU partition a single-card request lands
# in, and is generous for one model: Qwen, the slowest of the four, runs the grid's 4,620
# responses in roughly 130 minutes on one card at batch_size=2.
#
# Resumable: resubmit with the same model as often as needed and it continues from wherever it
# stopped.
# Progress check, no allocation needed: python -m experiment1.check_progress
#
# Usage: sbatch scripts/run_fullgrid_single_model.sbatch "Qwen2.5-VL-32B-Instruct"

set -euo pipefail
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
cd "$PROJECT_ROOT"
source scripts/env.sh

mkdir -p results/raw logs

MODEL="${1:?Usage: sbatch scripts/run_fullgrid_single_model.sbatch <model-name>}"

echo "=== Validating image/reference join (must be exactly ${EXPECTED_JOIN_COUNT:-1155}) ==="
python -m experiment1.build_join

echo "=== Full grid: $MODEL (single GPU) ==="
python -m experiment1.run_worker --model "$MODEL" --gpu 0 --full-grid-fixed

echo "=== Done (or ran out of walltime -- resubmit to continue) ==="
python -m experiment1.check_progress
