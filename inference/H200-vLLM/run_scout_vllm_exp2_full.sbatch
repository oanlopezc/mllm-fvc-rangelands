#!/bin/bash -l
#SBATCH --job-name=scout-vllm-exp2-full
#SBATCH --account=<your-slurm-account>
#SBATCH --partition=<your-gpu-partition>
#SBATCH --qos=<your-qos>
#SBATCH --gres=gpu:nvidia_h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=00:55:00
#SBATCH --output=/path/to/MLLMs-Local/logs/scout_vllm_exp2_full_%j.out
#SBATCH --error=/path/to/MLLMs-Local/logs/scout_vllm_exp2_full_%j.err
#
# One slice of Scout's full 1,155-photograph grid over the image variants, split by image variant,
# prompt group and image range so that each job finishes inside an hour.
#
# batch_size=32. Raising the batch size and the memory fraction beyond this lowers generation-phase
# GPU utilisation rather than raising it, so a larger batch buys nothing here.
#
# Why the split has to go two ways. Latency per photograph at batch_size=32 is uneven across
# prompts: v1_point_hint 2,490 ms (Scout's slowest), v2_short 251 ms, v3_detailed_ecology 1,068 ms,
# v4_grid_overlay 1,085 ms. The most even two-way split by cost is v1 alone against v2, v3 and v4
# together (2,490 against 2,404 ms per photograph), but either group alone is about 48 minutes of
# generation for all 1,155 photographs, and Scout's vLLM load takes about 13 minutes on top of
# that, which reaches the hour with no margin. Each prompt group is therefore halved by image range
# through --exp2-offset as well, giving roughly 23 to 24 minutes of generation plus the load.
#
# Usage: sbatch scripts/run_scout_vllm_exp2_full.sbatch <masked_gray|rectified> <prompts> <offset> <limit>

set -euo pipefail

cd $HOME/MLLMs-Local
mkdir -p logs

SOURCE="${1:?usage: sbatch run_scout_vllm_exp2_full.sbatch <masked_gray|rectified> <prompts> <offset> <limit>}"
PROMPTS="${2:?usage: sbatch run_scout_vllm_exp2_full.sbatch <masked_gray|rectified> <prompts> <offset> <limit>}"
OFFSET="${3:?usage: sbatch run_scout_vllm_exp2_full.sbatch <masked_gray|rectified> <prompts> <offset> <limit>}"
LIMIT="${4:?usage: sbatch run_scout_vllm_exp2_full.sbatch <masked_gray|rectified> <prompts> <offset> <limit>}"

VENV_PY="${ASSETS_DIR:?set ASSETS_DIR to the directory holding vllm-venv-clean}"/vllm-venv-clean/bin/python
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_RPC_TIMEOUT=1800000
MODEL=Llama-4-Scout-17B-16E-Instruct

echo "=== Scout vLLM exp2 full-grid slice started: $(date -u --iso-8601=seconds) ==="
echo "source=${SOURCE} prompts=${PROMPTS} offset=${OFFSET} limit=${LIMIT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

UTIL_LOG="logs/gpu_util_${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used --format=csv -l 20 > "$UTIL_LOG" &
UTIL_PID=$!
trap 'kill "$UTIL_PID" 2>/dev/null || true' EXIT

"${VENV_PY}" worker_vllm.py --model "$MODEL" --tensor-parallel-size 2 \
    --exp2-source "$SOURCE" --exp2-prompts "$PROMPTS" --exp2-offset "$OFFSET" --exp2-limit-images "$LIMIT" \
    --batch-size 32

kill "$UTIL_PID" 2>/dev/null || true
echo "=== GPU utilization summary (${UTIL_LOG}) ==="
awk -F', ' 'NR>1 {sum+=$3; n++} END {if (n>0) printf "mean utilization.gpu across %d samples: %.1f%%\n", n, sum/n}' "$UTIL_LOG" || true

echo "=== Progress ==="
$HOME/.pixi/bin/pixi run --manifest-path $HOME/MLLMs-Local/pixi.toml python -m experiment1.check_progress

echo "=== Scout vLLM exp2 full-grid slice ended: $(date -u --iso-8601=seconds) ==="
