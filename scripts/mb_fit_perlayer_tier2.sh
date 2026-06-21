#!/bin/sh
#SBATCH --job-name=t2_perlayer
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --output job%A.%N.out
#SBATCH --error  job%A.%N.err
#SBATCH -p AI_Center_L40S
#SBATCH --time=10:00:00

# =============================================================================
# Per-LAYER (per-block) curvature merge for Tier 2 Llama-3.1-8B. Breaks the
# scalar interference wall: each (expert, block) gets its own coefficient, the
# empirical-Fisher curvature solves all N*n_blocks at once. See
# scripts/mb_fit_perlayer_surrogate.py for the full rationale.
#
#   INIT=0.8,0.4,0.4,0.4,0.4 N_BLOCKS=8 STEPS=4 OUT_TAG=ta_pl_b8 \
#       sbatch --nodelist=node493 scripts/mb_fit_perlayer_tier2.sh
# =============================================================================
set -e
hostname; date
REPO="${REPO:-/work/pnag/Expert-Model-Merging}"
MERGE_ENV="${MERGE_ENV:-/work/pnag/envs/merging}"
CONFIG="${CONFIG:-configs/mergebench_tier2.yaml}"
INIT="${INIT:-0.8,0.4,0.4,0.4,0.4}"
N_BLOCKS="${N_BLOCKS:-8}"
NPD="${NPD:-32}"
MAXLEN="${MAXLEN:-512}"
STEPS="${STEPS:-4}"
RIDGE="${RIDGE:-0.1}"
STEPCLIP="${STEPCLIP:-0.15}"
OBJ="${OBJ:-teacher_mix}"
ATTR="${ATTR:-pooled}"
SOFTDISC="${SOFTDISC:-kl}"
KLMIX="${KLMIX:-}"
FREEZE="${FREEZE:-}"
OUT_TAG="${OUT_TAG:-ta_pl_b8}"

cd "${REPO}"
module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MERGE_ENV}"
export HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt 2>/dev/null)"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"

EXTRA=""
[ "$OBJ" = "teacher_mix" ] && EXTRA="$EXTRA --soft-discrepancy $SOFTDISC"
[ -n "$KLMIX" ] && EXTRA="$EXTRA --kl-mix $KLMIX"
[ -n "$FREEZE" ] && EXTRA="$EXTRA --freeze $FREEZE"

echo "[perlayer] init=${INIT} n_blocks=${N_BLOCKS} steps=${STEPS} ridge=${RIDGE} clip=${STEPCLIP} obj=${OBJ} attr=${ATTR} tag=${OUT_TAG}"
python -u scripts/mb_fit_perlayer_surrogate.py \
  --config "${CONFIG}" \
  --init-from "${INIT}" --n-blocks "${N_BLOCKS}" \
  --objective "${OBJ}" --attribution "${ATTR}" \
  --n-per-domain "${NPD}" --max-len "${MAXLEN}" \
  --steps "${STEPS}" --ridge "${RIDGE}" --step-clip "${STEPCLIP}" \
  --out-tag "${OUT_TAG}" \
  --manifest "mb_merged/Llama-3.1-8B/${OUT_TAG}.txt" \
  --merge ${EXTRA}
echo "[perlayer] done"; date
