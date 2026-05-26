#!/bin/sh
#SBATCH --job-name=whc_eval_math
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output job%A_%a.%N.out
#SBATCH --error  job%A_%a.%N.err
#SBATCH -p gpu
#SBATCH --time=06:00:00
#SBATCH --array=0-7%2

# =============================================================================
# Full gsm8k_cot evaluation of the base model + all 7 merged checkpoints.
#
# One model per array task; %2 caps concurrency at 2 so the node's two GPUs run
# two models at a time and per-model failures stay isolated. Config locked in
# from the smoke run: batch_size 4 + eager attention fit gemma-2-2b on the 16GB
# GPU (SDPA hits "cutlassF: no kernel" on gemma2 here; batch 16 OOMs the 256k-
# vocab logits).
#
# Usage:
#   sbatch scripts/mb_eval_math.sh                 # full gsm8k_cot, all 8 models
#   LIMIT=200 sbatch scripts/mb_eval_math.sh       # fast gate on 200 problems
#   BATCH=8   sbatch scripts/mb_eval_math.sh        # try a bigger batch
# =============================================================================

set -e
hostname; date

BASE_NAME="${BASE_NAME:-gemma-2-2b}"
LMEVAL_ENV="${LMEVAL_ENV:-/work/pnag/envs/lmeval}"
BATCH="${BATCH:-4}"
LIMIT="${LIMIT:-}"            # empty = full test set; set e.g. 200 for a gate
TASK="${TASK:-gsm8k_cot}"

cd /work/pnag/Expert-Model-Merging

module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${LMEVAL_ENV}"

export HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt 2>/dev/null)"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"   # default /tmp is only 2G here
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# index -> "tag local_path" (base first, then the 7 merges).
MODELS="
base mb_ckpts/google__${BASE_NAME}
task_arith mb_merged/${BASE_NAME}/task_arith
whc_diag mb_merged/${BASE_NAME}/whc_diag
TaskArithmetic mb_merged/${BASE_NAME}/TaskArithmetic
TIES mb_merged/${BASE_NAME}/TIES
DARE mb_merged/${BASE_NAME}/DARE
Consensus mb_merged/${BASE_NAME}/Consensus
LocalizeAndStitch mb_merged/${BASE_NAME}/LocalizeAndStitch
"
ROW=$(echo "$MODELS" | grep -v '^$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
TAG=$(echo "$ROW" | awk '{print $1}')
MODEL=$(echo "$ROW" | awk '{print $2}')

# Use the base model's tokenizer for every checkpoint. Merging never changes the
# vocab, but MergeBench's Merger.save re-serializes tokenizer.json with a newer
# tokenizers lib than the lmeval env can parse ("data did not match any variant
# of untagged enum ModelWrapper"). The original base tokenizer loads cleanly and
# is identical, so we point all models at it.
TOKENIZER_DIR="mb_ckpts/google__${BASE_NAME}"

OUT="results/mb_eval/${BASE_NAME}/${TAG}"
mkdir -p "$OUT"

LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

echo "[eval-math] task_id=${SLURM_ARRAY_TASK_ID} tag=${TAG} model=${MODEL} batch=${BATCH} limit=${LIMIT:-full} task=${TASK}"

lm_eval --model hf \
  --model_args "pretrained=${MODEL},tokenizer=${TOKENIZER_DIR},dtype=bfloat16,attn_implementation=eager" \
  --tasks "${TASK}" --device cuda:0 --batch_size "${BATCH}" ${LIMIT_ARG} \
  --output_path "${OUT}"

echo "[eval-math] done tag=${TAG}"
date
