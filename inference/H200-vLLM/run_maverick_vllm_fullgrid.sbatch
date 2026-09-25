#!/bin/bash -l
#SBATCH --job-name=maverick-vllm-fullgrid
#SBATCH --account=<your-slurm-account>
#SBATCH --partition=<your-gpu-partition>
#SBATCH --qos=<your-qos>
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/MLLMs-Local/logs/maverick_vllm_fullgrid_%j.out
#SBATCH --error=/path/to/MLLMs-Local/logs/maverick_vllm_fullgrid_%j.err
#
# Maverick's full 1,155-photograph grid via vLLM, at batch_size=1. That is the batch size the
# determinism and validation stages ran at, so the grid is collected under the configuration those
# checks describe rather than an untested one.
#
# 3h budget: once the model is loaded, generation runs at roughly one response per second across
# all four prompts, which puts the grid's 4,620 responses at roughly 80 to 90 minutes. The rest is
# headroom, in case latency per response differs at the full 1,155-photograph scale.

set -euo pipefail

cd $HOME/MLLMs-Local
mkdir -p logs

VENV_PY="${ASSETS_DIR:?set ASSETS_DIR to the directory holding vllm-venv-clean}"/vllm-venv-clean/bin/python
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_RPC_TIMEOUT=1800000
MODEL=Llama-4-Maverick-17B-128E-Instruct-FP8

echo "=== vLLM full grid started: $(date -u --iso-8601=seconds) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"${VENV_PY}" worker_vllm.py --model "$MODEL" --tensor-parallel-size 4 --full-grid-fixed --batch-size 1

echo "=== Progress ==="
$HOME/.pixi/bin/pixi run --manifest-path $HOME/MLLMs-Local/pixi.toml python -m experiment1.check_progress

echo "=== vLLM full grid ended: $(date -u --iso-8601=seconds) ==="
