#!/bin/sh
#SBATCH --job-name=t2_fisher
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output job%A_%a.%N.out
#SBATCH --error  job%A_%a.%N.err
#SBATCH -p gpu-v100-32gb
#SBATCH --time=04:00:00
#SBATCH --array=0-4%5

# =============================================================================
# Estimate the diagonal empirical Fisher for each of the five Llama-3.1-8B
# experts on its own validation data, for the DATA (Fisher) curvature in the HTCL
# sweep. One domain per array task. Writes mb_fisher/Llama-3.1-8B/<domain>/.
#
# CAVEAT: the per-domain validation dataset ids below are MergeBench's "<domain>_val"
# convention (the script's docstring example uses MergeBench/math_val). VERIFY each
# exists on HF before a long run; override any with the DATASETS map. If a domain
# lacks a _val set, point it at a slice of the training set (Tulu3IF/DartMath/
# MagiCoder/WildguardMix/Aya) via --text-field as needed.
#
# Llama needs eager attention here too (the estimator already sets it). 8B + grads
# fit the 32GB V100; Fisher accumulates on CPU.
#
# Usage:
#   sbatch scripts/mb_fisher_tier2.sh                       # all 5 domains
#   N_SAMPLES=512 sbatch scripts/mb_fisher_tier2.sh         # more samples
# =============================================================================

set -e
hostname; date

BASE_NAME="${BASE_NAME:-Llama-3.1-8B}"
BASE_REPO_DIR="${BASE_REPO_DIR:-NousResearch__Meta-Llama-3.1-8B}"
MERGE_ENV="${MERGE_ENV:-/work/pnag/envs/merging}"
REPO="${REPO:-/work/pnag/Expert-Model-Merging}"
N_SAMPLES="${N_SAMPLES:-256}"
MAX_LEN="${MAX_LEN:-1024}"

cd "${REPO}"
module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MERGE_ENV}"

export HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt 2>/dev/null)"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"

# index -> "domain dataset". Verify these dataset ids exist on HF first.
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

EXPERT="mb_ckpts/MergeBench__${BASE_NAME}_${DOMAIN}"
TOKENIZER="mb_ckpts/${BASE_REPO_DIR}"
OUT="mb_fisher/${BASE_NAME}/${DOMAIN}"

echo "[fisher] domain=${DOMAIN} expert=${EXPERT} dataset=${DATASET} n=${N_SAMPLES}"
python -u scripts/mb_fisher_estimate.py \
  --expert "${EXPERT}" --tokenizer "${TOKENIZER}" \
  --dataset "${DATASET}" --out "${OUT}" \
  --n-samples "${N_SAMPLES}" --max-len "${MAX_LEN}"

echo "[fisher] done domain=${DOMAIN}"
date
