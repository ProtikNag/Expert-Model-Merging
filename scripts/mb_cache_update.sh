#!/bin/sh
#SBATCH --job-name=t2_cache_upd
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=400G
#SBATCH --output job%A.%N.out
#SBATCH --error  job%A.%N.err
#SBATCH -p BigMem-64core
#SBATCH --time=03:00:00

# Single-pass cache of the whc_diag consensus update (u + ratio) + per-layer
# task-vector energy diagnostic. See scripts/mb_cache_update.py.
#   LAM=1e-3 OUT=mb_cache/Llama-3.1-8B_l1e-3 sbatch scripts/mb_cache_update.sh
set -e
hostname; date
REPO="${REPO:-/work/pnag/Expert-Model-Merging}"
MERGE_ENV="${MERGE_ENV:-/work/pnag/envs/merging}"
CONFIG="${CONFIG:-configs/mergebench_tier2.yaml}"
LAM="${LAM:-1e-3}"
OUT="${OUT:-mb_cache/Llama-3.1-8B_l1e-3}"
cd "${REPO}"
module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MERGE_ENV}"
export HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt 2>/dev/null)"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"
echo "[cache] lam=${LAM} out=${OUT}"
python -u scripts/mb_cache_update.py --config "${CONFIG}" --lam "${LAM}" --out "${OUT}"
echo "[cache] done"; date
