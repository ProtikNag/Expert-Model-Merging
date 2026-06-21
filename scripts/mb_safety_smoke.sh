#!/bin/sh
#SBATCH --job-name=t2_safety_smoke
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output job%A.%N.out
#SBATCH --error  job%A.%N.err
#SBATCH -p AI_Center_L40S
#SBATCH --time=02:00:00

# Validate the freshly-built safety harness end-to-end on ONE model + ONE task at
# a tiny limit BEFORE the full 14-model run (which also downloads the WildGuard
# classifier). Also pins transformers to the vllm-0.6.3-compatible 4.46.3 (the
# fork left it unpinned -> pip pulled 5.x, which breaks vllm at model-load).
set -e
hostname; date
REPO="${REPO:-/work/pnag/Expert-Model-Merging}"
ENV="${ENV:-/work/pnag/envs/safety-eval}"
FORK="${FORK:-/work/pnag/safety-eval-fork}"
MODEL="${MODEL:-$REPO/mb_merged/Llama-3.1-8B/ta_pe_inst0.8_codi0.4}"
TASKS="${TASKS:-xstest}"
LIMIT="${LIMIT:-8}"
TEMPLATE="${TEMPLATE:-llama3}"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"
export HF_TOKEN="$(tr -d '[:space:]' < ${REPO}/hf_token.txt 2>/dev/null)"
# NON-empty placeholder: the fork builds AsyncOpenAI() at import, and the newer
# openai lib rejects "" as missing creds. Our 4 tasks use LOCAL (WildGuard)
# classifiers, so the key is never actually used for a call.
export OPENAI_API_KEY="EMPTY"
export TOKENIZERS_PARALLELISM=false

module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

echo "===== pin transformers 4.46.3 (vllm 0.6.3 compat) ====="
pip install "transformers==4.46.3" 2>&1 | tail -4
python -c "import transformers,vllm,torch;print('transformers',transformers.__version__,'vllm',vllm.__version__,'torch',torch.__version__)"

OUT="$REPO/results/mb_eval/Llama-3.1-8B/_smoke_safety"
mkdir -p "$OUT"
echo "===== smoke: $TASKS limit=$LIMIT on $(basename "$MODEL") ====="
cd "$FORK"
python evaluation/eval.py generators \
  --model_name_or_path "$MODEL" \
  --use_vllm \
  --model_input_template_path_or_name "$TEMPLATE" \
  --tasks "$TASKS" \
  --report_output_path "$OUT/safety_eval.json" \
  --save_individual_results_path "$OUT/safety_generation.json" \
  --batch_size 8 --limit "$LIMIT"

echo "===== smoke result ====="
if [ -f "$OUT/safety_eval.json" ]; then
  echo "SMOKE_OK — safety_eval.json written:"; cat "$OUT/safety_eval.json"
else
  echo "SMOKE_FAIL — no safety_eval.json produced"
fi
date
