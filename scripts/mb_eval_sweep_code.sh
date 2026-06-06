#!/bin/sh
#SBATCH --job-name=t2_sweep_code
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output job%A_%a.%N.out
#SBATCH --error  job%A_%a.%N.err
#SBATCH -p gpu-v100-32gb
#SBATCH --time=06:00:00
#SBATCH --array=0-23%6

# =============================================================================
# Gate-eval the HTCL sweep variants on coding (humanevalplus, mbppplus). One
# variant per array task from the sweep manifest; indices past the manifest exit
# cleanly. N_SAMPLES defaults to 5 (vs the protocol 10) for a faster ranking
# gate; the winning variant gets the full n_samples=10 eval via mb_eval_code_tier2.sh.
#
# Usage:
#   sbatch -p gpu-v100-32gb --array=0-11%6 scripts/mb_eval_sweep_code.sh
# =============================================================================

set -e
hostname; date

BASE_NAME="${BASE_NAME:-Llama-3.1-8B}"
BIGCODE_ENV="${BIGCODE_ENV:-/work/pnag/envs/bigcode}"
BIGCODE_DIR="${BIGCODE_DIR:-/work/pnag/bigcode-evaluation-harness}"
REPO="${REPO:-/work/pnag/Expert-Model-Merging}"
MANIFEST="${MANIFEST:-mb_merged/${BASE_NAME}/sweep_manifest.txt}"
N_SAMPLES="${N_SAMPLES:-5}"
BATCH="${BATCH:-5}"
TASKS="${TASKS:-humanevalplus,mbppplus}"

module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${BIGCODE_ENV}"

export HF_TOKEN="$(tr -d '[:space:]' < ${REPO}/hf_token.txt 2>/dev/null)"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${REPO}/${MANIFEST}" 2>/dev/null)
if [ -z "$ROW" ]; then
  echo "[sweep-code] no variant at index ${SLURM_ARRAY_TASK_ID}; exiting."
  exit 0
fi
TAG=$(echo "$ROW" | awk '{print $1}')
# Manifest dirs are repo-relative; make absolute for the bigcode harness cwd.
MODEL="${REPO}/$(echo "$ROW" | awk '{print $2}')"

OUT="${REPO}/results/mb_eval/${BASE_NAME}/${TAG}"
mkdir -p "$OUT"

echo "[sweep-code] idx=${SLURM_ARRAY_TASK_ID} tag=${TAG} model=${MODEL} n_samples=${N_SAMPLES}"
cd "${BIGCODE_DIR}"
accelerate launch --num_processes 1 main.py \
  --model "${MODEL}" \
  --tasks "${TASKS}" \
  --max_length_generation 512 --precision bf16 \
  --temperature 0.2 --n_samples "${N_SAMPLES}" --batch_size "${BATCH}" \
  --allow_code_execution \
  --metric_output_path "${OUT}/code_eval.json"

echo "[sweep-code] done tag=${TAG}"
date
