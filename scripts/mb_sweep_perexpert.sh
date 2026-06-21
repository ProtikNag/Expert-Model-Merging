#!/bin/sh
#SBATCH --job-name=t2_pe_merge
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=300G
#SBATCH --output job%A.%N.out
#SBATCH --error  job%A.%N.err
#SBATCH -p BigMem-64core
#SBATCH --time=03:00:00

# Per-expert scaled task arithmetic sweep (single pass). See mb_sweep_perexpert.py.
#   BASE=0.4 SETS="instruction=0.6,0.8,1.0;coding=0.3,0.4" \
#   MANIFEST=mb_merged/Llama-3.1-8B/perexpert_r0.txt sbatch scripts/mb_sweep_perexpert.sh
set -e
hostname; date
REPO="${REPO:-/work/pnag/Expert-Model-Merging}"
MERGE_ENV="${MERGE_ENV:-/work/pnag/envs/merging}"
CONFIG="${CONFIG:-configs/mergebench_tier2.yaml}"
BASE="${BASE:-0.4}"
SETS="${SETS:-instruction=0.6,0.8,1.0;coding=0.3,0.4}"
MANIFEST="${MANIFEST:-}"
cd "${REPO}"
module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MERGE_ENV}"
export HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt 2>/dev/null)"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"

# Expand SETS (';'-separated DOMAIN=vals) into repeated --set flags.
SET_ARGS=""
OLDIFS="$IFS"; IFS=';'
for kv in $SETS; do
  [ -n "$kv" ] && SET_ARGS="$SET_ARGS --set $kv"
done
IFS="$OLDIFS"

echo "[pe-merge] base=${BASE} sets=${SETS} manifest=${MANIFEST:-<default>}"
python -u scripts/mb_sweep_perexpert.py --config "${CONFIG}" \
  --base "${BASE}" ${SET_ARGS} \
  ${MANIFEST:+--manifest "${MANIFEST}"}
echo "[pe-merge] done"; date
