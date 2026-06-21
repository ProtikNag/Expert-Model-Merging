#!/bin/sh
#SBATCH --job-name=t2_surrogate
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --output job%A.%N.out
#SBATCH --error  job%A.%N.err
#SBATCH -p AI_Center_L40S
#SBATCH --time=06:00:00

# =============================================================================
# DERIVE the champion's per-expert coefficients from the Gauss-Newton / Taylor
# surrogate (scripts/mb_fit_perexpert_surrogate.py) instead of hand-sweeping
# them. Replaces the 2-D grid that produced (0.8, 0.4, 0.4, 0.4, 0.4) with a
# principled N-coefficient solve on a tiny per-domain val buffer (data-light).
#
# Runs in the `merging` env (cuda torch + transformers + datasets), eager attn.
# 8B + grads fit the L40S 48 GB GPU (the Fisher estimator proves this at 8B on a
# 32 GB V100); the 5 task vectors are cached on CPU bf16 (~80 GB) + w_pre (~16 GB),
# so --mem=200G and a high-RAM node (node493/L40S has 257 GB). A100 (dgx_aic) also
# works and lets you raise --n-per-domain.
#
# PREREQ: the MergeBench/<domain>_val datasets must exist on HF (same caveat as
# mb_fisher_tier2.sh -- instruction/safety may need a training-set slice + the
# --datasets override). This is independent of the safety eval; it does NOT need
# the merged baselines, only base + the 5 experts in mb_ckpts.
#
# Usage:
#   sbatch scripts/mb_fit_surrogate_tier2.sh                       # single-pass from 0.4
#   INIT=0.8,0.4,0.4,0.4,0.4 STEPS=3 sbatch scripts/mb_fit_surrogate_tier2.sh  # refine from champion
#   NPD=64 STEPS=2 sbatch scripts/mb_fit_surrogate_tier2.sh        # bigger buffer
# After it finishes, eval the written tag through the gate/forgetting drivers and
# compare s* against the hand-swept champion (instruction=0.8, coding=0.4).
# =============================================================================

set -e
hostname; date

REPO="${REPO:-/work/pnag/Expert-Model-Merging}"
MERGE_ENV="${MERGE_ENV:-/work/pnag/envs/merging}"
CONFIG="${CONFIG:-configs/mergebench_tier2.yaml}"
INIT="${INIT:-0.4,0.4,0.4,0.4,0.4}"
NPD="${NPD:-32}"
MAXLEN="${MAXLEN:-512}"
STEPS="${STEPS:-1}"
RIDGE="${RIDGE:-1e-2}"
STEPCLIP="${STEPCLIP:-0.3}"   # trust region: max |delta_i| per coord per step
OUT_TAG="${OUT_TAG:-ta_pe_surrogate}"
OBJ="${OBJ:-teacher_kl}"    # teacher_kl | teacher_hard | teacher_mix | ce
ATTR="${ATTR:-own_domain}"  # own_domain (default, per-expert decoupled -> pushes s up) | pooled (ablation)
SOFTDISC="${SOFTDISC:-kl}"  # teacher_mix soft term: kl | js | logit_mse | feature_mse
KLMIX="${KLMIX:-}"          # teacher_mix per-domain lambda (default in py: 0,0,0,1,1)
DUMP="${DUMP:-}"            # optional npz path: dump per-example derivatives at step 0 (offline mu sweep)
DATASETS="${DATASETS:-}"   # optional: "instruction=...;safety=..." override for missing _val sets
FREEZE="${FREEZE:-}"       # comma domain names held at init (metric-aware: freeze argmax-generative)

cd "${REPO}"
module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MERGE_ENV}"

export HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt 2>/dev/null)"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"

DS_ARG=""
[ -n "$DATASETS" ] && DS_ARG="--datasets $DATASETS"
[ -n "$DUMP" ] && DS_ARG="$DS_ARG --dump-derivatives $DUMP"
[ "$OBJ" = "teacher_mix" ] && DS_ARG="$DS_ARG --soft-discrepancy $SOFTDISC"
[ -n "$KLMIX" ] && DS_ARG="$DS_ARG --kl-mix $KLMIX"
[ -n "$FREEZE" ] && DS_ARG="$DS_ARG --freeze $FREEZE"

echo "[surrogate] init=${INIT} steps=${STEPS} n_per_domain=${NPD} ridge=${RIDGE} obj=${OBJ} attr=${ATTR} softdisc=${SOFTDISC} klmix=${KLMIX:-default} tag=${OUT_TAG}"
python -u scripts/mb_fit_perexpert_surrogate.py \
  --config "${CONFIG}" \
  --init-from "${INIT}" \
  --objective "${OBJ}" --attribution "${ATTR}" \
  --n-per-domain "${NPD}" --max-len "${MAXLEN}" \
  --steps "${STEPS}" --ridge "${RIDGE}" --step-clip "${STEPCLIP}" \
  --out-tag "${OUT_TAG}" \
  --manifest "mb_merged/Llama-3.1-8B/${OUT_TAG}.txt" \
  --merge ${DS_ARG}

echo "[surrogate] done"
date
