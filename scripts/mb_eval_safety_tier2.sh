#!/bin/sh
#SBATCH --job-name=t2_eval_safety
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output job%A_%a.%N.out
#SBATCH --error  job%A_%a.%N.err
#SBATCH -p AI_Center_L40S
#SBATCH --time=08:00:00
#SBATCH --array=0-12%6

# =============================================================================
# Tier 2 safety eval (wildguardtest, harmbench, xstest, do_anything_now) for
# Llama-3.1-8B, all five domains merged. base + 7 merges + 5 experts = 13 models.
#
# This is the FOURTH eval environment (MergeBench uses three external harnesses
# plus this). It needs:
#   - the safety-eval-fork repo cloned (MergeBench's fork of allenai/safety-eval)
#   - a `safety-eval` conda env with vLLM + the fork's requirements
#   - the WildGuard classifier weights (downloaded on first run; gated -> HF_TOKEN)
# See TIER2_RUNBOOK.md for one-time setup. Metric: RTA (refuse-to-answer) for
# wildguardtest/harmbench/do_anything_now; accuracy for xstest.
#
# vLLM loads the model from disk, so the 5 baseline merge dirs must already have
# the base tokenizer in place (scripts/mb_fix_tokenizers.py). 8B + vLLM KV cache
# fits the 48GB L40S comfortably.
#
# Usage:
#   sbatch scripts/mb_eval_safety_tier2.sh                       # full, 13 models
#   TASKS=xstest LIMIT=8 sbatch scripts/mb_eval_safety_tier2.sh   # smoke
# =============================================================================

set -e
hostname; date

BASE_NAME="${BASE_NAME:-Llama-3.1-8B}"
BASE_REPO_DIR="${BASE_REPO_DIR:-NousResearch__Meta-Llama-3.1-8B}"
SAFETY_ENV="${SAFETY_ENV:-/work/pnag/envs/safety-eval}"
SAFETY_DIR="${SAFETY_DIR:-/work/pnag/safety-eval-fork}"
REPO="${REPO:-/work/pnag/Expert-Model-Merging}"
TASKS="${TASKS:-wildguardtest,harmbench,xstest,do_anything_now}"
TEMPLATE="${TEMPLATE:-llama3}"
BATCH="${BATCH:-8}"

module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SAFETY_ENV}"

export HF_TOKEN="$(tr -d '[:space:]' < ${REPO}/hf_token.txt 2>/dev/null)"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"
export OPENAI_API_KEY="EMPTY"   # NON-empty placeholder: fork builds AsyncOpenAI() at import & newer openai lib rejects ""; our 4 tasks use LOCAL (WildGuard) classifiers
export TOKENIZERS_PARALLELISM=false

# index -> "tag absolute_model_path": base, 7 merges, 5 experts, champion (idx 0-13).
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
ta_pe_inst0.8_codi0.4 ${REPO}/mb_merged/${BASE_NAME}/ta_pe_inst0.8_codi0.4
ta_pe_inst0.8_codi0.49_safe0.64_mult0.74 ${REPO}/mb_merged/${BASE_NAME}/ta_pe_inst0.8_codi0.49_safe0.64_mult0.74
ta_pe_mix_pool_kl ${REPO}/mb_merged/${BASE_NAME}/ta_pe_mix_pool_kl
ta_pe_inst0.8_mult0.73 ${REPO}/mb_merged/${BASE_NAME}/ta_pe_inst0.8_mult0.73
ta_pl_b8 ${REPO}/mb_merged/${BASE_NAME}/ta_pl_b8
ta_pl_b8_frzgen ${REPO}/mb_merged/${BASE_NAME}/ta_pl_b8_frzgen
"
ROW=$(echo "$MODELS" | grep -v '^$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
TAG=$(echo "$ROW" | awk '{print $1}')
MODEL=$(echo "$ROW" | awk '{print $2}')

OUT="${REPO}/results/mb_eval/${BASE_NAME}/${TAG}"
mkdir -p "$OUT"

LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

echo "[eval-safety] task_id=${SLURM_ARRAY_TASK_ID} tag=${TAG} model=${MODEL} tasks=${TASKS} template=${TEMPLATE}"

cd "${SAFETY_DIR}"
python evaluation/eval.py generators \
  --model_name_or_path "${MODEL}" \
  --use_vllm \
  --model_input_template_path_or_name "${TEMPLATE}" \
  --tasks "${TASKS}" \
  --report_output_path "${OUT}/safety_eval.json" \
  --save_individual_results_path "${OUT}/safety_generation.json" \
  --batch_size "${BATCH}" ${LIMIT_ARG}

echo "[eval-safety] done tag=${TAG}"
date
