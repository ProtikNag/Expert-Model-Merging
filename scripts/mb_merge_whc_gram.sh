#!/bin/sh
#SBATCH --job-name=t2_whcgram
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=300G
#SBATCH --output job%A.%N.out
#SBATCH --error  job%A.%N.err
#SBATCH -p BigMem-64core
#SBATCH --time=08:00:00

# =============================================================================
# Data-using whc_gram merge (HTCL-data) as a high-RAM CPU batch job. Requires
# mb_grams/Llama-3.1-8B/<domain>/model.safetensors from scripts/mb_gram_tier2.sh.
#
# Per-key peak: N copies of the input Gram + the [in,in] solve. Without
# down_proj that is ~5x[4096,4096] (tiny). WITH down_proj it is ~5x[14336,14336]
# fp32 (~16.5 GB) plus the solve, and the 32 down_proj solves dominate runtime
# (~tens of minutes). BigMem (2 TB) clears either case; --mem=300G is generous
# for the no-down_proj default and enough with it. CPU-only (no --gres).
#
# Runs FIRST as single-pass (K=0). The iterative catch-up (K>=1) is a separate
# loop documented in TIER2_RUNBOOK.md (re-estimate Grams on the merged model,
# then re-run this). Tokenizers are fixed after so the merged dirs load + eval.
#
# Usage:
#   sbatch scripts/mb_merge_whc_gram.sh                          # lams 0,1e-3,1e-2
#   LAMS=0,1e-3,1e-2,1e-1 GAMMAS=0 sbatch scripts/mb_merge_whc_gram.sh
#   LAMS=1e-3 GAMMAS=0.1 FISHER_ROOT=mb_fisher/Llama-3.1-8B sbatch scripts/mb_merge_whc_gram.sh
# =============================================================================

set -e
hostname; date

REPO="${REPO:-/work/pnag/Expert-Model-Merging}"
MERGE_ENV="${MERGE_ENV:-/work/pnag/envs/merging}"
CONFIG="${CONFIG:-configs/mergebench_tier2.yaml}"
GRAM_ROOT="${GRAM_ROOT:-mb_grams/Llama-3.1-8B}"
LAMS="${LAMS:-1e-2}"
ALPHAS="${ALPHAS:-1}"
FALLBACKS="${FALLBACKS:-mean}"
SCALE="${SCALE:-0.4}"
GAMMAS="${GAMMAS:-0}"
FISHER_ROOT="${FISHER_ROOT:-}"

cd "${REPO}"
module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MERGE_ENV}"

export HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt 2>/dev/null)"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"

FISHER_ARG=""
if [ -n "${FISHER_ROOT}" ]; then
  FISHER_ARG="--fisher-root ${FISHER_ROOT}"
fi

echo "[whc-gram] config=${CONFIG} gram_root=${GRAM_ROOT} lams=${LAMS} alphas=${ALPHAS} fallbacks=${FALLBACKS} gammas=${GAMMAS}"
python -u scripts/mb_merge_whc_gram.py --config "${CONFIG}" \
  --gram-root "${GRAM_ROOT}" --lams "${LAMS}" --alphas "${ALPHAS}" \
  --fallbacks "${FALLBACKS}" --scale "${SCALE}" --gammas "${GAMMAS}" ${FISHER_ARG}
python    scripts/mb_fix_tokenizers.py --config "${CONFIG}"

echo "[whc-gram] done"
date
