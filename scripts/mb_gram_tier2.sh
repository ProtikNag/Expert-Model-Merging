#!/bin/sh
#SBATCH --job-name=t2_gram
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output job%A_%a.%N.out
#SBATCH --error  job%A_%a.%N.err
#SBATCH -p gpu-v100-32gb
#SBATCH --time=06:00:00
#SBATCH --array=0-4%5

# =============================================================================
# Estimate per-Linear input Grams for each of the five Llama-3.1-8B experts on
# its own validation data, for the DATA-using whc_gram merge (HTCL-data) vs
# RegMean / RegMean++. One domain per array task. Writes mb_grams/Llama-3.1-8B/
# <domain>/model.safetensors keyed by the Linear weight names.
#
# MEMORY: the default excludes mlp.down_proj (the [14336,14336] Gram), keeping
# the per-expert CPU accumulator ~13 GB so it fits 96 GB. To include down_proj
# for the faithful all-Linear RegMean comparison, set EXCLUDE="" and bump --mem
# to ~200G (and expect a high-RAM node; the V100 partition may not grant it, in
# which case run those on AI_Center_L40S).
#
# Llama needs eager attention here too (the estimator already sets it). 8B bf16
# forward fits the 32 GB V100; Grams accumulate on CPU.
#
# CAVEAT: the per-domain validation dataset ids below follow MergeBench's
# "<domain>_val" convention. VERIFY each exists on HF before a long run; if a
# domain lacks a _val set, point it at a slice of the training set via the
# DATASETS row + --text-field. These MUST match the ids used for Fisher so the
# two data-using statistics see identical inputs.
#
# Usage:
#   sbatch scripts/mb_gram_tier2.sh                          # 5 domains, no down_proj
#   EXCLUDE="" N_SAMPLES=512 sbatch scripts/mb_gram_tier2.sh # all Linears, more samples
# =============================================================================

set -e
hostname; date

BASE_NAME="${BASE_NAME:-Llama-3.1-8B}"
BASE_REPO_DIR="${BASE_REPO_DIR:-NousResearch__Meta-Llama-3.1-8B}"
MERGE_ENV="${MERGE_ENV:-/work/pnag/envs/merging}"
REPO="${REPO:-/work/pnag/Expert-Model-Merging}"
N_SAMPLES="${N_SAMPLES:-256}"
MAX_LEN="${MAX_LEN:-1024}"
EXCLUDE="${EXCLUDE:-down_proj}"        # set EXCLUDE="" to include all Linears
GRAM_DTYPE="${GRAM_DTYPE:-fp16}"

cd "${REPO}"
module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MERGE_ENV}"

export HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt 2>/dev/null)"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"

# index -> "domain dataset". Keep in lockstep with mb_fisher_tier2.sh.
ROWS="
instruction MergeBench/instruction_val
math MergeBench/math_val
coding MergeBench/coding_val
safety MergeBench/safety_val
multilingual MergeBench/multilingual_val
"
ROW=$(echo "$ROWS" | grep -v '^$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
DOMAIN=$(echo "$ROW" | awk '{print $1}')
DATASET=$(echo "$ROW" | awk '{print $2}')

# Round 0: hook each domain's own expert. Iterative catch-up (K>=1): set
# LINEARIZE_AT to the round-(k-1) merged checkpoint to hook IT on every domain's
# data instead (the linearization point moves to the merge; the experts stay
# fixed). OUT_ROOT redirects the output (e.g. mb_grams/<base>_k1).
LINEARIZE_AT="${LINEARIZE_AT:-}"
OUT_ROOT="${OUT_ROOT:-mb_grams/${BASE_NAME}}"
if [ -n "${LINEARIZE_AT}" ]; then
  EXPERT="${LINEARIZE_AT}"
else
  EXPERT="mb_ckpts/MergeBench__${BASE_NAME}_${DOMAIN}"
fi
TOKENIZER="mb_ckpts/${BASE_REPO_DIR}"
OUT="${OUT_ROOT}/${DOMAIN}"

echo "[gram] domain=${DOMAIN} expert=${EXPERT} dataset=${DATASET} n=${N_SAMPLES} exclude='${EXCLUDE}'"
python -u scripts/mb_gram_estimate.py \
  --expert "${EXPERT}" --tokenizer "${TOKENIZER}" \
  --dataset "${DATASET}" --out "${OUT}" \
  --n-samples "${N_SAMPLES}" --max-len "${MAX_LEN}" \
  --exclude-modules "${EXCLUDE}" --gram-dtype "${GRAM_DTYPE}"

echo "[gram] done domain=${DOMAIN}"
date
