#!/bin/sh
#SBATCH --job-name=t2_merge
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=600G
#SBATCH --output job%A.%N.out
#SBATCH --error  job%A.%N.err
#SBATCH -p BigMem-64core
#SBATCH --time=08:00:00

# =============================================================================
# Tier 2 merge as a high-RAM CPU batch job on BigMem-64core (~2 TB/node). Do NOT
# run the merge on the login node: whc_diag briefly holds ~12 copies of the
# 128k-vocab embedding (~25-30 GB peak) and the login cgroup OOM-kills it
# ("Killed", no traceback).
#
# The MergeBench baseline mergers are not streamed: TIES stacks all 5 task
# vectors into one matrix (~160 GB single alloc), DARE/L&S allocate tens of GB in
# torch.topk over the full flattened task vector. A 180 GB V100 node OOM-kills
# TIES/DARE/L&S; the 2 TB BigMem node clears them with room to spare. The merge
# math is CPU-only, so no --gres (BigMem has no GPUs). Override with
# `sbatch -p <p> --mem=<m>` only if BigMem is saturated.
#
# METHODS controls which methods run (comma-separated --only list). Default is
# the six NOT already done (task_arith completed on login). Run whc_diag FIRST
# and alone so an OOM in a baseline (uncatchable SIGKILL) cannot take it down.
#
# Usage:
#   METHODS=whc_diag sbatch scripts/mb_merge_tier2.sh                      # ours first
#   METHODS=TaskArithmetic,TIES,DARE,Consensus,LocalizeAndStitch sbatch scripts/mb_merge_tier2.sh
#   sbatch scripts/mb_merge_tier2.sh                                       # all six remaining
# =============================================================================

set -e
hostname; date

REPO="${REPO:-/work/pnag/Expert-Model-Merging}"
MERGE_ENV="${MERGE_ENV:-/work/pnag/envs/merging}"
CONFIG="${CONFIG:-configs/mergebench_tier2.yaml}"
METHODS="${METHODS:-whc_diag,TaskArithmetic,TIES,DARE,Consensus,LocalizeAndStitch}"

cd "${REPO}"

module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MERGE_ENV}"

export HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt 2>/dev/null)"
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"
# Cap torch CPU threads to the allocation so we do not oversubscribe the node.
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:-8}"

echo "[merge] config=${CONFIG} methods=${METHODS}"
python -u scripts/mb_tier1_merge.py --config "${CONFIG}" --tier all --only "${METHODS}"
python    scripts/mb_fix_tokenizers.py --config "${CONFIG}"

echo "[merge] done"
date
