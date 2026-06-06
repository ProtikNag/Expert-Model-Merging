#!/bin/sh
#SBATCH --job-name=t2_sweep_merge
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=600G
#SBATCH --output job%A.%N.out
#SBATCH --error  job%A.%N.err
#SBATCH -p BigMem-64core
#SBATCH --time=10:00:00

# =============================================================================
# Merge the HTCL (lam, alpha) sweep on BigMem-64core (~2 TB). whc_diag streams
# per-key (~30 GB peak) so this is comfortable; the long pole is wall time across
# the grid (~20 min/variant). CPU-only.
#
# LAMS / ALPHAS / FISHER_ROOT override the grid. Default dataless grid = 3x4 = 12
# variants (~4 h). Add FISHER_ROOT to also sweep the data Fisher curvature.
#
# Usage:
#   sbatch scripts/mb_sweep_merge.sh
#   LAMS=1e-4,1e-5 ALPHAS=1,2,3,4 sbatch scripts/mb_sweep_merge.sh
#   FISHER_ROOT=mb_fisher/Llama-3.1-8B sbatch scripts/mb_sweep_merge.sh
# =============================================================================

set -e
hostname; date

REPO="${REPO:-/work/pnag/Expert-Model-Merging}"
MERGE_ENV="${MERGE_ENV:-/work/pnag/envs/merging}"
CONFIG="${CONFIG:-configs/mergebench_tier2.yaml}"
LAMS="${LAMS:-1e-3,1e-4,1e-5}"
ALPHAS="${ALPHAS:-1,2,3,4}"

cd "${REPO}"
module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MERGE_ENV}"

export HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt 2>/dev/null)"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-16}"

FISHER_ARG=""
[ -n "$FISHER_ROOT" ] && FISHER_ARG="--fisher-root $FISHER_ROOT"

echo "[sweep-merge] lams=${LAMS} alphas=${ALPHAS} fisher=${FISHER_ROOT:-none}"
python -u scripts/mb_sweep_whc.py --config "${CONFIG}" \
  --lams "${LAMS}" --alphas "${ALPHAS}" ${FISHER_ARG}

echo "[sweep-merge] done"
date
