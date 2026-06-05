#!/bin/sh
#SBATCH --job-name=t2_eval_lm
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output job%A_%a.%N.out
#SBATCH --error  job%A_%a.%N.err
#SBATCH -p AI_Center_L40S
#SBATCH --time=12:00:00
#SBATCH --array=0-12%2

# =============================================================================
# Tier 2 lm-eval driver: instruction (ifeval), math (gsm8k_cot), and
# multilingual (m_mmlu/arc/hellaswag x {fr,es,de,ru}) for Llama-3.1-8B, all
# five domains merged. One model per array task; %2 caps concurrency.
#
# Task names + batch sizes mirror MergeBench's scripts/evaluate.sh verbatim.
# Llama-3.1-8B (unlike gemma2) needs NO eager-attention workaround, so SDPA is
# left on. 8B in bf16 ~16GB weights -> target the 48GB L40S (or gpu-v100-32gb).
#
# Tokenizer: pass the base tokenizer for every checkpoint. MergeBench's baseline
# Merger.save re-serializes tokenizer.json with a tokenizers lib the lmeval env
# cannot parse; the base tokenizer is identical and loads cleanly.
#
# Usage:
#   sbatch scripts/mb_eval_lm_tier2.sh                    # all 3 groups, 13 models
#   GROUP=math sbatch scripts/mb_eval_lm_tier2.sh          # only gsm8k_cot
#   GROUP=multilingual sbatch scripts/mb_eval_lm_tier2.sh  # only the 12 ML tasks
#   LIMIT=200 GROUP=math sbatch scripts/mb_eval_lm_tier2.sh # fast gate
# =============================================================================

set -e
hostname; date

BASE_NAME="${BASE_NAME:-Llama-3.1-8B}"
BASE_REPO_DIR="${BASE_REPO_DIR:-NousResearch__Meta-Llama-3.1-8B}"
LMEVAL_ENV="${LMEVAL_ENV:-/work/pnag/envs/lmeval}"
BATCH="${BATCH:-8}"
LIMIT="${LIMIT:-}"
GROUP="${GROUP:-all}"     # all | instruction | math | multilingual

cd /work/pnag/Expert-Model-Merging

module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${LMEVAL_ENV}"

export HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt 2>/dev/null)"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# index -> "tag local_path": base, 7 merges, 5 experts (= 13 rows, idx 0-12).
MODELS="
base mb_ckpts/${BASE_REPO_DIR}
task_arith mb_merged/${BASE_NAME}/task_arith
whc_diag mb_merged/${BASE_NAME}/whc_diag
TaskArithmetic mb_merged/${BASE_NAME}/TaskArithmetic
TIES mb_merged/${BASE_NAME}/TIES
DARE mb_merged/${BASE_NAME}/DARE
Consensus mb_merged/${BASE_NAME}/Consensus
LocalizeAndStitch mb_merged/${BASE_NAME}/LocalizeAndStitch
instruction_expert mb_ckpts/MergeBench__${BASE_NAME}_instruction
math_expert mb_ckpts/MergeBench__${BASE_NAME}_math
coding_expert mb_ckpts/MergeBench__${BASE_NAME}_coding
safety_expert mb_ckpts/MergeBench__${BASE_NAME}_safety
multilingual_expert mb_ckpts/MergeBench__${BASE_NAME}_multilingual
"
ROW=$(echo "$MODELS" | grep -v '^$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
TAG=$(echo "$ROW" | awk '{print $1}')
MODEL=$(echo "$ROW" | awk '{print $2}')

TOKENIZER_DIR="mb_ckpts/${BASE_REPO_DIR}"
OUT="results/mb_eval/${BASE_NAME}/${TAG}"
mkdir -p "$OUT"

LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

ML_TASKS="m_mmlu_fr,arc_fr,hellaswag_fr,m_mmlu_es,arc_es,hellaswag_es,m_mmlu_de,arc_de,hellaswag_de,m_mmlu_ru,arc_ru,hellaswag_ru"

run_group () {
  TASKS="$1"
  echo "[eval-lm] task_id=${SLURM_ARRAY_TASK_ID} tag=${TAG} tasks=${TASKS} batch=${BATCH} limit=${LIMIT:-full}"
  lm_eval --model hf \
    --model_args "pretrained=${MODEL},tokenizer=${TOKENIZER_DIR},dtype=bfloat16" \
    --tasks "${TASKS}" --device cuda:0 --batch_size "${BATCH}" ${LIMIT_ARG} \
    --output_path "${OUT}"
}

case "$GROUP" in
  instruction)  run_group "ifeval" ;;
  math)         run_group "gsm8k_cot" ;;
  multilingual) run_group "$ML_TASKS" ;;
  all)
    run_group "gsm8k_cot"
    run_group "ifeval"
    run_group "$ML_TASKS"
    ;;
  *) echo "unknown GROUP=$GROUP"; exit 2 ;;
esac

echo "[eval-lm] done tag=${TAG} group=${GROUP}"
date
