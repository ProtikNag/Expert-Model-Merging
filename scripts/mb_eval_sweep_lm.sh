#!/bin/sh
#SBATCH --job-name=t2_sweep_lm
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
# Gate-eval the HTCL sweep variants on math (gsm8k_cot) + instruction (ifeval)
# only -- multilingual is the slow/OOM domain and is NOT needed to rank lam/alpha.
# One variant per array task, read from the sweep manifest (tag dir per line).
# Indices past the manifest length exit cleanly (no-op).
#
# LIMIT gates gsm8k/ifeval for speed (default 500 -> ~0.02 stderr, enough to
# rank). The winning variant gets a full eval later via mb_eval_lm_tier2.sh.
# attn_implementation=eager is required on V100 (cutlassF), base tokenizer for all.
#
# Usage:
#   sbatch -p gpu-v100-32gb --array=0-11%6 scripts/mb_eval_sweep_lm.sh   # 12-variant grid
#   LIMIT=0 sbatch ... scripts/mb_eval_sweep_lm.sh                       # full eval (LIMIT=0 -> no cap)
# =============================================================================

set -e
hostname; date

BASE_NAME="${BASE_NAME:-Llama-3.1-8B}"
BASE_REPO_DIR="${BASE_REPO_DIR:-NousResearch__Meta-Llama-3.1-8B}"
LMEVAL_ENV="${LMEVAL_ENV:-/work/pnag/envs/lmeval}"
REPO="${REPO:-/work/pnag/Expert-Model-Merging}"
MANIFEST="${MANIFEST:-mb_merged/${BASE_NAME}/sweep_manifest.txt}"
BATCH="${BATCH:-8}"
LIMIT="${LIMIT:-500}"            # 0 = no cap (full)
TASKS="${TASKS:-gsm8k_cot,ifeval}"

cd "${REPO}"
module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${LMEVAL_ENV}"

export HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt 2>/dev/null)"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}" 2>/dev/null)
if [ -z "$ROW" ]; then
  echo "[sweep-lm] no variant at index ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}; exiting."
  exit 0
fi
TAG=$(echo "$ROW" | awk '{print $1}')
MODEL=$(echo "$ROW" | awk '{print $2}')

TOKENIZER_DIR="mb_ckpts/${BASE_REPO_DIR}"
OUT="results/mb_eval/${BASE_NAME}/${TAG}"
mkdir -p "$OUT"

LIMIT_ARG=""
[ "$LIMIT" != "0" ] && LIMIT_ARG="--limit $LIMIT"

echo "[sweep-lm] idx=${SLURM_ARRAY_TASK_ID} tag=${TAG} model=${MODEL} tasks=${TASKS} limit=${LIMIT}"
lm_eval --model hf \
  --model_args "pretrained=${MODEL},tokenizer=${TOKENIZER_DIR},dtype=bfloat16,attn_implementation=eager" \
  --tasks "${TASKS}" --device cuda:0 --batch_size "${BATCH}" ${LIMIT_ARG} \
  --output_path "${OUT}"

echo "[sweep-lm] done tag=${TAG}"
date
