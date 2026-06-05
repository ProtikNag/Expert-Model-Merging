#!/bin/sh
#SBATCH --job-name=t2_merge
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=180G
#SBATCH --output job%A.%N.out
#SBATCH --error  job%A.%N.err
#SBATCH -p gpu-v100-32gb
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# =============================================================================
# Tier 2 merge as a high-RAM batch job. Do NOT run the merge on the login node:
# whc_diag briefly holds ~12 copies of the 128k-vocab embedding (~25-30 GB peak)
# and the login cgroup OOM-kills it ("Killed", no traceback).
#
# 8B x 5 experts + base, all in fp32 at the widest key -> request lots of RAM.
# The MergeBench baseline mergers additionally load full models (not streamed),
# so 180 GB fits a 192 GB V100 node. No GPU is used by the merge math; the --gres line
# only satisfies partitions that require it (gpu-v100-32gb nodes have the RAM and
# are usually free). Override the partition with `sbatch -p <p>` if you have a
# dedicated high-mem CPU partition.
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
