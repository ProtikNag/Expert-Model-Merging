#!/bin/sh
#SBATCH --job-name=whc_glue
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output logs/whc_glue_%A_%a.out
#SBATCH --error  logs/whc_glue_%A_%a.err
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --time=20:00:00
#SBATCH --array=0

# =============================================================================
# WHC vs. recent model-merging baselines on GLUE + RoBERTa-base.
#
# One-seed run (per user request). Full pipeline:
#   1. Fine-tune one RoBERTa expert per GLUE task.
#   2. Collect diagonal empirical Fisher and per-Linear activation Grams.
#   3. Merge with every method (Simple, TA, TIES, Fisher, Fisher+epsI,
#      RegMean, RegMean++, WHC-A/B/C/D in dataless and Fisher flavors).
#   4. Generate figures (PNG + SVG in separate folders).
#
# Usage:
#   cd <REPO_ROOT>
#   sbatch gpu_run.sh
# Override the repo root / python env with REPO_ROOT= and PY_ENV= below.
# =============================================================================

hostname
date

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1

# Load cluster modules if available (ignore errors on non-SLURM hosts).
module load cuda/12.3      2>/dev/null || true
module load python3/anaconda/2023.9 2>/dev/null || true

# ── Environment (override via env vars as needed) ────────────────────────
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
PY_ENV="${PY_ENV:-/work/pnag/envs/ml_env}"
CONFIG="${CONFIG:-configs/glue_roberta.yaml}"

if [ -d "$PY_ENV" ]; then
  source activate "$PY_ENV" 2>/dev/null || conda activate "$PY_ENV" 2>/dev/null || true
fi

cd "$REPO_ROOT"
mkdir -p logs results/logs results/glue/figures/png results/glue/figures/svg

# ── Env preflight: fail loudly with a useful message if env is broken ─────
set -e
PY="$(which python 2>/dev/null || true)"
if [ -z "$PY" ]; then
  echo "ERR: no python on PATH after 'source activate $PY_ENV'"
  echo "     HINT: recreate the env and verify 'which python' points inside it."
  exit 1
fi
python - <<'PYCHECK' || { echo "ERR: env is missing required packages"; exit 1; }
import sys, torch, transformers, datasets, sklearn, sentencepiece, yaml
print("  python :", sys.executable)
print("  torch  :", torch.__version__, "cuda?", torch.cuda.is_available())
print("  trf    :", transformers.__version__)
print("  ds     :", datasets.__version__)
PYCHECK
set +e

echo "============================================"
echo "WHC-GLUE Merging Experiment"
echo "  Array task:  ${SLURM_ARRAY_TASK_ID:-standalone}"
echo "  Repo:        $REPO_ROOT"
echo "  Config:      $CONFIG"
echo "  GPU:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Started:     $(date)"
echo "============================================"

# ── Stage 1: fine-tune experts and collect statistics ──────────────────────
python scripts/lm_train_experts.py \
    --config "$CONFIG" \
    --device cuda

# ── Stage 2: run every merging method, tune on val, report on test ─────────
python scripts/lm_run_merging.py \
    --config "$CONFIG" \
    --device cuda

# ── Stage 3: generate figures ──────────────────────────────────────────────
python scripts/lm_make_figures.py \
    --config "$CONFIG"

echo "============================================"
echo "Completed at $(date)"
echo "Artifacts:"
echo "  checkpoints:  $(du -sh checkpoints/glue 2>/dev/null | cut -f1)"
echo "  results:      $(du -sh results/glue 2>/dev/null | cut -f1)"
echo "  logs:         $(ls results/logs/ 2>/dev/null | wc -l) files"
echo "============================================"

date
