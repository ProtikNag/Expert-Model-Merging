#!/bin/sh
#SBATCH --job-name=whc_mergebench
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output job%j.%N.out
#SBATCH --error  job%j.%N.err
#SBATCH -p gpu
#SBATCH --time=12:00:00

# =============================================================================
# WHC vs. MergeBench baselines on a 2-task subset (math + coding), Gemma-2-2B.
#
# Stages (control with STAGE=tier0|merge|eval|all, default all):
#   tier0 : CPU divergence diagnostic + figures.
#   merge : produce merged checkpoints (WHC + dataless baselines).
#   eval  : evaluate each merged checkpoint on gsm8k (math) and
#           humanevalplus/mbppplus (coding) via MergeBench's harnesses.
#
# Prerequisites on the HPC node:
#   - conda env `merging` with this repo's requirements.txt installed.
#   - MergeBench cloned at $MERGEBENCH_DIR (default ./MergeBench) and, for the
#     eval stage, its `lmeval` and `bigcode` conda envs (see MergeBench README).
#   - `huggingface-cli login` done (Gemma base is gated).
#
# Usage:
#   sbatch scripts/mb_gpu_run.sh
#   STAGE=merge sbatch scripts/mb_gpu_run.sh
# =============================================================================

hostname; date
set -e

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cuda/12.3 2>/dev/null || true
module load python3/anaconda/2023.9 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true

if [ -d "/work/pnag/Expert-Model-Merging" ]; then
    cd /work/pnag/Expert-Model-Merging/
fi

CONFIG="${CONFIG:-configs/mergebench.yaml}"
STAGE="${STAGE:-all}"
DOMAINS="${DOMAINS:-math,coding}"
ONLY="${ONLY:-}"          # comma-separated method subset for fail-fast merges
MERGEBENCH_DIR="${MERGEBENCH_DIR:-./MergeBench}"
MERGE_ENV="${MERGE_ENV:-merging}"
BASE_NAME="${BASE_NAME:-gemma-2-2b}"
MERGED_ROOT="mb_merged/${BASE_NAME}"
EVAL_OUT="results/mb_eval/${BASE_NAME}"

mkdir -p results/logs "${EVAL_OUT}"
TS=$(date +"%Y%m%d_%H%M%S")
LOG="results/logs/mb_gpu_${TS}.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG}"; }

log "============================================================"
log " WHC vs MergeBench baselines | base=${BASE_NAME} domains=${DOMAINS}"
log " stage=${STAGE}  config=${CONFIG}  mergebench=${MERGEBENCH_DIR}"
log "============================================================"

# HF auth: prefer an exported HF_TOKEN, else a gitignored hf_token.txt created
# directly on this node (never committed). Falls back to cached login.
if [ -z "${HF_TOKEN:-}" ] && [ -f hf_token.txt ]; then
    export HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt)"
    log "HF_TOKEN loaded from hf_token.txt"
fi

conda activate "${MERGE_ENV}" 2>/dev/null || true
PY="$(command -v python)"
log "python: ${PY}"
${PY} -c "import torch,transformers,safetensors; print('  torch',torch.__version__,'cuda?',torch.cuda.is_available())" 2>&1 | tee -a "${LOG}"

# Clone MergeBench if missing (needed for baselines + eval harness scripts).
if [ ! -d "${MERGEBENCH_DIR}" ]; then
    log "cloning MergeBench -> ${MERGEBENCH_DIR}"
    git clone --depth 1 https://github.com/uiuctml/MergeBench "${MERGEBENCH_DIR}" 2>&1 | tee -a "${LOG}"
fi

# Apply our hardware-specific patches to the clone (idempotent): memory-bounded
# TIES trim + eager attention for LocalizeAndStitch. Without these, TIES OOMs
# above 128G and L&S fails on the missing (unbuildable) flash-attn.
log "--- patching MergeBench clone for this node ---"
${PY} -u scripts/mb_patch_mergebench.py --mergebench "${MERGEBENCH_DIR}" 2>&1 | tee -a "${LOG}"

# -----------------------------------------------------------------------------
# Download base + experts (idempotent; snapshot_download skips existing files).
# -----------------------------------------------------------------------------
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "merge" ]; then
    log "--- downloading base + experts (${DOMAINS}) ---"
    ${PY} -u scripts/mb_download.py --config "${CONFIG}" --domains "${DOMAINS}" 2>&1 | tee -a "${LOG}"
fi

# -----------------------------------------------------------------------------
# STAGE: tier0 (CPU divergence + figures)
# -----------------------------------------------------------------------------
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "tier0" ]; then
    log "--- Tier 0: divergence diagnostic ---"
    ${PY} -u scripts/mb_tier0_divergence.py --config "${CONFIG}" --domains "${DOMAINS}" 2>&1 | tee -a "${LOG}"
    ${PY} -u scripts/mb_make_figures.py --config "${CONFIG}" 2>&1 | tee -a "${LOG}"
fi

# -----------------------------------------------------------------------------
# STAGE: merge (WHC + dataless baselines; all weights-only)
# -----------------------------------------------------------------------------
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "merge" ]; then
    log "--- Tier 1: merging (ours + dataless baselines) ---"
    ONLY_ARG=""
    [ -n "${ONLY}" ] && ONLY_ARG="--only ${ONLY}"
    ${PY} -u scripts/mb_tier1_merge.py --config "${CONFIG}" --domains "${DOMAINS}" --tier all ${ONLY_ARG} 2>&1 | tee -a "${LOG}"
fi

# -----------------------------------------------------------------------------
# STAGE: eval (gsm8k for math, humanevalplus/mbppplus for coding)
# Runs MergeBench's eval harnesses in their dedicated conda envs. We eval the
# base model once (reference) plus every merged checkpoint.
# -----------------------------------------------------------------------------
eval_one() {
    MODEL_DIR="$1"; TAG="$2"
    OUT="${EVAL_OUT}/${TAG}"
    mkdir -p "${OUT}"
    log "  [eval:${TAG}] math (gsm8k_cot)"
    conda activate lmeval 2>/dev/null || { log "  WARN: no lmeval env, skipping math"; return; }
    lm_eval --model hf --model_args "pretrained=${MODEL_DIR},dtype=bfloat16" \
        --tasks gsm8k_cot --batch_size 16 --output_path "${OUT}" 2>&1 | tee -a "${LOG}" || log "  WARN: gsm8k failed for ${TAG}"
    conda deactivate 2>/dev/null || true

    log "  [eval:${TAG}] coding (humanevalplus, mbppplus)"
    conda activate bigcode 2>/dev/null || { log "  WARN: no bigcode env, skipping coding"; return; }
    ( cd "${MERGEBENCH_DIR}/../bigcode-evaluation-harness" 2>/dev/null || cd bigcode-evaluation-harness 2>/dev/null || { log "  WARN: bigcode harness dir not found"; exit 0; }
      accelerate launch main.py --model "${MODEL_DIR}" --max_length_generation 512 \
        --precision bf16 --tasks humanevalplus,mbppplus --temperature 0.2 \
        --n_samples 10 --batch_size 10 --allow_code_execution \
        --metric_output_path "${OUT}/code_eval.json" --use_auth_token ) 2>&1 | tee -a "${LOG}" || log "  WARN: code eval failed for ${TAG}"
    conda deactivate 2>/dev/null || true
}

if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "eval" ]; then
    log "--- Eval: base reference + each merged checkpoint ---"
    # Base local dir mirrors scripts/mb_download.py's layout: repo '/' -> '__'.
    BASE_LOCAL="mb_ckpts/google__${BASE_NAME}"
    eval_one "${BASE_LOCAL}" "base"
    for d in "${MERGED_ROOT}"/*/ ; do
        [ -d "$d" ] || continue
        TAG="$(basename "$d")"
        eval_one "${d%/}" "${TAG}"
    done
fi

log "============================================================"
log " done. merged: ${MERGED_ROOT}  eval: ${EVAL_OUT}  log: ${LOG}"
log "============================================================"
date
