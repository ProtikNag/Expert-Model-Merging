#!/bin/sh
#SBATCH --job-name=t2_pscale_merge
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=600G
#SBATCH --output job%A.%N.out
#SBATCH --error  job%A.%N.err
#SBATCH -p BigMem-64core
#SBATCH --time=08:00:00

# =============================================================================
# Merge the HTCL per-parameter update-scale sweep on BigMem-64core (~2 TB).
# whc_diag streams per-key (~30 GB peak) so this is comfortable; per variant the
# per-key task-vector math is a little heavier than the global-alpha sweep but
# still ~20-30 min/variant. CPU-only.
#
# LAM / CONS_AMAXES / COH_AMAXES / COH_BETAS override the grid. Default =
# 3 consensus (amax 3,5,8) + 2 coherence (amax 5, beta 1,2) = 5 variants.
#
# Usage:
#   sbatch scripts/mb_sweep_pscale.sh
#   LAM=1e-3 CONS_AMAXES=3,5,8 COH_AMAXES=5 COH_BETAS=1,2 sbatch scripts/mb_sweep_pscale.sh
# =============================================================================

set -e
hostname; date

REPO="${REPO:-/work/pnag/Expert-Model-Merging}"
MERGE_ENV="${MERGE_ENV:-/work/pnag/envs/merging}"
CONFIG="${CONFIG:-configs/mergebench_tier2.yaml}"
LAM="${LAM:-1e-3}"
CONS_AMAXES="${CONS_AMAXES:-3,5,8}"
COH_AMAXES="${COH_AMAXES:-5}"
COH_BETAS="${COH_BETAS:-1,2}"
MANIFEST="${MANIFEST:-}"   # optional: route this run's variants to a separate manifest

cd "${REPO}"
module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MERGE_ENV}"

export HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt 2>/dev/null)"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-16}"

echo "[pscale-merge] lam=${LAM} cons_amaxes=${CONS_AMAXES} coh_amaxes=${COH_AMAXES} coh_betas=${COH_BETAS} manifest=${MANIFEST:-<default>}"
python -u scripts/mb_sweep_pscale.py --config "${CONFIG}" \
  --lam "${LAM}" --cons-amaxes "${CONS_AMAXES}" \
  --coh-amaxes "${COH_AMAXES}" --coh-betas "${COH_BETAS}" \
  ${MANIFEST:+--manifest "${MANIFEST}"}

echo "[pscale-merge] done"
date
