#!/bin/sh
#SBATCH --job-name=whc_eval_code
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output job%A_%a.%N.out
#SBATCH --error  job%A_%a.%N.err
#SBATCH -p AI_Center_L40S
#SBATCH --time=04:00:00
#SBATCH --array=0-9%2

# =============================================================================
# Coding eval (humanevalplus, mbppplus) of base + 7 merges + 2 experts, one
# model per array task, %2 concurrent. Targets the 48GB L40S so we can run
# MergeBench's full protocol (n_samples 10, batch 10) without OOM.
#
# Prereqs (already done in this project):
#   - bigcode env at /work/pnag/envs/bigcode, harness at
#     /work/pnag/bigcode-evaluation-harness.
#   - Harness patched for eager attention (scripts/mb_patch_bigcode.py) — gemma2
#     SDPA hits cutlassF on these GPUs.
#   - The 5 MergeBench baseline merge dirs had their unreadable tokenizer.json
#     replaced with the base tokenizer (bigcode has no tokenizer override flag).
#
# Usage:
#   sbatch scripts/mb_eval_code.sh                         # full, all 10 models
#   N_SAMPLES=1 BATCH=1 LIMIT=2 sbatch scripts/mb_eval_code.sh   # smoke
# =============================================================================

set -e
hostname; date

BASE_NAME="${BASE_NAME:-gemma-2-2b}"
BIGCODE_ENV="${BIGCODE_ENV:-/work/pnag/envs/bigcode}"
BIGCODE_DIR="${BIGCODE_DIR:-/work/pnag/bigcode-evaluation-harness}"
REPO="${REPO:-/work/pnag/Expert-Model-Merging}"
N_SAMPLES="${N_SAMPLES:-10}"
BATCH="${BATCH:-10}"
TASKS="${TASKS:-humanevalplus,mbppplus}"
LIMIT="${LIMIT:-}"

module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${BIGCODE_ENV}"

export HF_TOKEN="$(tr -d '[:space:]' < ${REPO}/hf_token.txt 2>/dev/null)"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# index -> "tag absolute_model_path" (base, 7 merges, 2 experts).
MODELS="
base ${REPO}/mb_ckpts/google__${BASE_NAME}
task_arith ${REPO}/mb_merged/${BASE_NAME}/task_arith
whc_diag ${REPO}/mb_merged/${BASE_NAME}/whc_diag
TaskArithmetic ${REPO}/mb_merged/${BASE_NAME}/TaskArithmetic
TIES ${REPO}/mb_merged/${BASE_NAME}/TIES
DARE ${REPO}/mb_merged/${BASE_NAME}/DARE
Consensus ${REPO}/mb_merged/${BASE_NAME}/Consensus
LocalizeAndStitch ${REPO}/mb_merged/${BASE_NAME}/LocalizeAndStitch
math_expert ${REPO}/mb_ckpts/MergeBench__${BASE_NAME}_math
coding_expert ${REPO}/mb_ckpts/MergeBench__${BASE_NAME}_coding
fisher_merge ${REPO}/mb_merged/${BASE_NAME}/fisher_merge
whc_diag_fisher ${REPO}/mb_merged/${BASE_NAME}/whc_diag_fisher
"
ROW=$(echo "$MODELS" | grep -v '^$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
TAG=$(echo "$ROW" | awk '{print $1}')
MODEL=$(echo "$ROW" | awk '{print $2}')

OUT="${REPO}/results/mb_eval/${BASE_NAME}/${TAG}"
mkdir -p "$OUT"

LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

echo "[eval-code] task_id=${SLURM_ARRAY_TASK_ID} tag=${TAG} model=${MODEL} tasks=${TASKS} n_samples=${N_SAMPLES} batch=${BATCH} limit=${LIMIT:-full}"

cd "${BIGCODE_DIR}"
accelerate launch --num_processes 1 main.py \
  --model "${MODEL}" \
  --tasks "${TASKS}" \
  --max_length_generation 512 --precision bf16 \
  --temperature 0.2 --n_samples "${N_SAMPLES}" --batch_size "${BATCH}" \
  --allow_code_execution ${LIMIT_ARG} \
  --metric_output_path "${OUT}/code_eval.json"

echo "[eval-code] done tag=${TAG}"
date
