#!/bin/sh
#SBATCH --job-name=t2_route_merge
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=300G
#SBATCH --output job%A.%N.out
#SBATCH --error  job%A.%N.err
#SBATCH -p BigMem-64core
#SBATCH --time=03:00:00

# =============================================================================
# Dominance-ROUTED per-parameter scaling sweep (single pass, CPU-only).
# Breaks the instruction-vs-coding alpha coupling: coding-owned params capped
# at LOWS, everything else at HIGHS. See scripts/mb_sweep_routed.py.
#
#   LAM=1e-3 HIGHS=3,3.5,4,5 LOWS=1 LOW_DOMAINS=coding \
#   MANIFEST=mb_merged/Llama-3.1-8B/routed_r0.txt sbatch scripts/mb_sweep_routed.sh
# =============================================================================
set -e
hostname; date

REPO="${REPO:-/work/pnag/Expert-Model-Merging}"
MERGE_ENV="${MERGE_ENV:-/work/pnag/envs/merging}"
CONFIG="${CONFIG:-configs/mergebench_tier2.yaml}"
LAM="${LAM:-1e-3}"
HIGHS="${HIGHS:-3,3.5,4,5}"
LOWS="${LOWS:-1}"
LOW_DOMAINS="${LOW_DOMAINS:-coding}"
MANIFEST="${MANIFEST:-}"

cd "${REPO}"
module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MERGE_ENV}"

export HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt 2>/dev/null)"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"

echo "[routed-merge] lam=${LAM} highs=${HIGHS} lows=${LOWS} low_domains=${LOW_DOMAINS} manifest=${MANIFEST:-<default>}"
python -u scripts/mb_sweep_routed.py --config "${CONFIG}" \
  --lam "${LAM}" --highs "${HIGHS}" --lows "${LOWS}" \
  --low-domains "${LOW_DOMAINS}" \
  ${MANIFEST:+--manifest "${MANIFEST}"}

echo "[routed-merge] done"
date
