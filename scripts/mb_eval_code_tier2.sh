#!/bin/sh
#SBATCH --job-name=t2_eval_code
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output job%A_%a.%N.out
#SBATCH --error  job%A_%a.%N.err
#SBATCH -p AI_Center_L40S
#SBATCH --time=06:00:00
#SBATCH --array=0-12%6

# =============================================================================
# Tier 2 coding eval (humanevalplus, mbppplus) for Llama-3.1-8B, all five
# domains merged. base + 7 merges + 5 experts = 13 models, one per array task.
# MergeBench's full protocol: n_samples 10, batch 10, temperature 0.2, bf16.
#
# Reuses the bigcode env + harness from Tier 1 (/work/pnag/envs/bigcode,
# /work/pnag/bigcode-evaluation-harness). The eager-attention patch applied for
# gemma2 is harmless for Llama (correct, just slightly slower); no re-patch needed.
#
# Tokenizer: the 5 MergeBench baseline merge dirs must have the base tokenizer
# in place (bigcode has no tokenizer-override flag). Run scripts/mb_fix_tokenizers.py
# once after merging — see TIER2_RUNBOOK.md.
#
# Usage:
#   sbatch scripts/mb_eval_code_tier2.sh                          # full, 13 models
#   N_SAMPLES=1 BATCH=1 LIMIT=2 sbatch scripts/mb_eval_code_tier2.sh   # smoke
# =============================================================================

set -e
hostname; date

BASE_NAME="${BASE_NAME:-Llama-3.1-8B}"
BASE_REPO_DIR="${BASE_REPO_DIR:-NousResearch__Meta-Llama-3.1-8B}"
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

# index -> "tag absolute_model_path": base, 7 merges, 5 experts (idx 0-12).
MODELS="
base ${REPO}/mb_ckpts/${BASE_REPO_DIR}
task_arith ${REPO}/mb_merged/${BASE_NAME}/task_arith
whc_diag ${REPO}/mb_merged/${BASE_NAME}/whc_diag
TaskArithmetic ${REPO}/mb_merged/${BASE_NAME}/TaskArithmetic
TIES ${REPO}/mb_merged/${BASE_NAME}/TIES
DARE ${REPO}/mb_merged/${BASE_NAME}/DARE
Consensus ${REPO}/mb_merged/${BASE_NAME}/Consensus
LocalizeAndStitch ${REPO}/mb_merged/${BASE_NAME}/LocalizeAndStitch
instruction_expert ${REPO}/mb_ckpts/MergeBench__${BASE_NAME}_instruction
math_expert ${REPO}/mb_ckpts/MergeBench__${BASE_NAME}_math
coding_expert ${REPO}/mb_ckpts/MergeBench__${BASE_NAME}_coding
safety_expert ${REPO}/mb_ckpts/MergeBench__${BASE_NAME}_safety
multilingual_expert ${REPO}/mb_ckpts/MergeBench__${BASE_NAME}_multilingual
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
