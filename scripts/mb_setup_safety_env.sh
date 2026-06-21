#!/bin/sh
#SBATCH --job-name=t2_safety_setup
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --output job%A.%N.out
#SBATCH --error  job%A.%N.err
#SBATCH -p AI_Center_L40S
#SBATCH --time=04:00:00

# =============================================================================
# One-time setup of the 4th eval harness (MergeBench safety) on an L40S node,
# which (unlike the login/sandbox shell) has outbound network for git+pip+HF.
# Recipe: the README's `vllm==0.4.2` is STALE — the fork's own requirements.txt
# requires torch>=2.4.0 and vllm>=0.6.2 ("for llama 3 compat"). Pinning 0.4.2
# forced torch 2.3 + an old dep closure that source-builds sentencepiece/pandas
# and FAILS on RHEL7 (no compiler toolchain). So we follow the FORK's pins:
#   - sentencepiece from conda-forge (binary; the one dep that won't pip-build here)
#   - vllm==0.6.3.post1 (pulls torch 2.4.0 + transformers/tokenizers as cu121 wheels)
#   - pip install -e . + -r requirements.txt (now satisfied by wheels)
# RHEL7: numpy<2 (project_eval_envs_hpc). WildGuard classifier weights are gated
# -> HF_TOKEN; they download on first eval run, not here.
# Idempotent: skips clone/env-create if present; pip re-runs are cheap when satisfied.
# =============================================================================
set -e
hostname; date
REPO="${REPO:-/work/pnag/Expert-Model-Merging}"
FORK="${FORK:-/work/pnag/safety-eval-fork}"
ENV="${ENV:-/work/pnag/envs/safety-eval}"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"
export PIP_CACHE_DIR=/work/pnag/tmp/pipcache; mkdir -p "$PIP_CACHE_DIR"
export HF_TOKEN="$(tr -d '[:space:]' < ${REPO}/hf_token.txt 2>/dev/null)"

module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "===== [1/5] clone fork ====="
if [ -d "$FORK/.git" ]; then
  echo "fork already present at $FORK"
else
  git clone https://github.com/nouhadziri/safety-eval-fork "$FORK"
fi

echo "===== [2/5] create env ====="
if [ -d "$ENV" ]; then
  echo "env already present at $ENV"
else
  conda create -p "$ENV" python=3.10 -y
fi
conda activate "$ENV"
python --version

echo "===== [3/5] the 2 deps with NO glibc-2.17 pip wheel, from conda-forge (libmamba) ====="
# RHEL7 glibc 2.17: a C compiler IS present (pyzmq/pyairports build fine) but RUST
# is NOT, so Rust-based pkgs whose newest wheel targets glibc 2.28 (libcst, tiktoken)
# source-build and FAIL. sentencepiece (C++) needs cmake/protobuf it also lacks.
# Route those to conda-forge binaries. pyarrow/pandas/numpy/tokenizers DO have
# glibc-2.17 pip wheels at the lmeval-proven versions -> pip constraints file below
# (avoids a huge conda solve that OOM'd the classic solver). libmamba = fast/low-mem.
conda install -p "$ENV" -c conda-forge --solver=libmamba -y sentencepiece libcst tiktoken

echo "===== [4/5] constraints file pins ONLY the glibc-2.28-prone natives ====="
CONSTRAINTS=/work/pnag/tmp/safety_constraints.txt
cat > "$CONSTRAINTS" <<'CON'
pyarrow==14.0.2
pandas==2.2.3
numpy==1.26.4
datasets==2.19.0
CON
echo "constraints:"; cat "$CONSTRAINTS"

echo "===== [4b/5] torch 2.4 (cu121 wheel, glibc-2.17 ok) + vllm + fork ====="
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -c "$CONSTRAINTS" vllm==0.6.3.post1
cd "$FORK"
pip install -e . --no-deps   # don't let the fork's >= pins re-pull glibc-2.28 natives
# Remaining fork deps install as wheels; constraints cap the native ones to known-good.
[ -f requirements.txt ] && pip install -c "$CONSTRAINTS" --upgrade-strategy only-if-needed -r requirements.txt || echo "(no requirements.txt)"

# The fork leaves transformers unpinned -> pip pulls 5.x, but vllm 0.6.3.post1 was
# written for transformers 4.45-4.46 and breaks at model-load on 5.x. Pin it back.
pip install "transformers==4.46.3"

echo "===== [5/5] import smoke test ====="
python - <<'PY'
ok=True
for m in ("vllm","torch","transformers"):
    try:
        mod=__import__(m); print(m, getattr(mod,"__version__","?"))
    except Exception as e:
        ok=False; print("IMPORT FAIL", m, repr(e))
import torch; print("cuda available:", torch.cuda.is_available())
print("SETUP_OK" if ok else "SETUP_INCOMPLETE")
PY
echo "[safety-setup] done"; date
