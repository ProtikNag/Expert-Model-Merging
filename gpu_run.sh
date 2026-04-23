#!/bin/sh
#SBATCH --job-name=whc_glue
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output job%j.%N.out
#SBATCH --error  job%j.%N.err
#SBATCH -p gpu
#SBATCH --time=20:00:00

# =============================================================================
# WHC vs. recent model-merging baselines on GLUE + RoBERTa-base.
# One-seed run. Full pipeline:
#   1. Fine-tune one RoBERTa expert per GLUE task.
#   2. Collect diagonal empirical Fisher and per-Linear activation Grams.
#   3. Merge with every method (Simple, TA, TIES, Fisher, Fisher+epsI,
#      RegMean, RegMean++, WHC-A/B/C/D in dataless and Fisher flavors).
#   4. Generate figures (PNG + SVG).
#
# Usage:
#   sbatch gpu_run.sh
# =============================================================================

hostname
date

# -- GPU / Environment --------------------------------------------------------

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1

module load cuda/12.3 2>/dev/null || true
module load python3/anaconda/2023.9 2>/dev/null || true

if [ -d "/work/pnag/envs/ml_env" ]; then
    source activate /work/pnag/envs/ml_env/
elif [ -d "venv" ]; then
    . venv/bin/activate
fi

python --version
python -c "import torch; print('  torch=' + torch.__version__ + ', CUDA=' + str(torch.cuda.is_available()) + ', device_count=' + str(torch.cuda.device_count()))"

if [ -d "/work/pnag/Expert-Model-Merging" ]; then
    cd /work/pnag/Expert-Model-Merging/
fi

set -e

CONFIG="${CONFIG:-configs/glue_roberta.yaml}"

mkdir -p results/logs results/glue/figures/png results/glue/figures/svg

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/logs/whc_glue_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run.log"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log_msg "============================================================"
log_msg " WHC-GLUE Merging Experiment"
log_msg "============================================================"
log_msg " Config:   ${CONFIG}"
log_msg " GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
log_msg " Log:      ${LOG_FILE}"
log_msg "============================================================"

# -----------------------------------------------------------------------------
# STEP 1: Fine-tune experts and collect statistics
# -----------------------------------------------------------------------------

log_msg ""
log_msg "------------------------------------------------------------"
log_msg " Step 1: Fine-tune RoBERTa experts + collect Fisher/Grams"
log_msg "------------------------------------------------------------"

python scripts/lm_train_experts.py \
    --config "${CONFIG}" \
    --device cuda 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]:-$?}

if [ ${EXIT_CODE} -ne 0 ]; then
    log_msg "FAILED: expert training (exit code ${EXIT_CODE})"
    exit ${EXIT_CODE}
fi

log_msg "Expert training complete."

# -----------------------------------------------------------------------------
# STEP 2: Run every merging method, tune on val, report on test
# -----------------------------------------------------------------------------

log_msg ""
log_msg "------------------------------------------------------------"
log_msg " Step 2: Run merging methods"
log_msg "------------------------------------------------------------"

python scripts/lm_run_merging.py \
    --config "${CONFIG}" \
    --device cuda 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]:-$?}

if [ ${EXIT_CODE} -ne 0 ]; then
    log_msg "FAILED: merging (exit code ${EXIT_CODE})"
    exit ${EXIT_CODE}
fi

log_msg "Merging complete."

# -----------------------------------------------------------------------------
# STEP 3: Figures
# -----------------------------------------------------------------------------

log_msg ""
log_msg "------------------------------------------------------------"
log_msg " Step 3: Generate figures (PNG + SVG)"
log_msg "------------------------------------------------------------"

python scripts/lm_make_figures.py \
    --config "${CONFIG}" 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]:-$?}

if [ ${EXIT_CODE} -ne 0 ]; then
    log_msg "WARNING: figure generation failed (exit code ${EXIT_CODE})"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

log_msg ""
log_msg "============================================================"
log_msg " Completed at $(date)"
log_msg "============================================================"
log_msg " checkpoints: $(du -sh checkpoints/glue 2>/dev/null | cut -f1)"
log_msg " results:     $(du -sh results/glue 2>/dev/null | cut -f1)"
log_msg " log:         ${LOG_FILE}"
log_msg "============================================================"

date
