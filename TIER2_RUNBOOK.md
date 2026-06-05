# Tier 2 Runbook — five domains on Llama-3.1-8B

Widen WHC/HTCL from the Tier 1 two-domain gemma run to all five MergeBench
domains on the largest base, `meta-llama/Llama-3.1-8B`. Dataless tier only;
RegMean/RegMean++ deferred. Workflow is the usual git push (Mac) → pull (HPC).

Config: [`configs/mergebench_tier2.yaml`](configs/mergebench_tier2.yaml). Read the
project memory env notes first (`project_merging_env_hpc`, `project_eval_envs_hpc`,
`feedback_env_setup`). Reuse the standing envs; do not build new ones except the
safety env (Step 5).

The 13 evaluated models = base + 7 merges (task_arith, whc_diag, TaskArithmetic,
TIES, DARE, Consensus, LocalizeAndStitch) + 5 specialists (one per domain).

## What is different from Tier 1 (gemma-2-2b)

- **No eager-attention workaround.** Llama-3.1-8B runs fine on SDPA. The gemma2
  `cutlassF` patch is irrelevant here (the bigcode harness keeps its global eager
  patch, which is harmless for Llama, just marginally slower).
- **8B in bf16 ≈ 16 GB weights.** The 16 GB `gpu` node will not fit generation.
  Target the **48 GB L40S** (`AI_Center_L40S`) or `gpu-v100-32gb`. All Tier 2
  SLURM scripts default to `-p AI_Center_L40S`.
- **Five domains, four benchmark families.** Math + multilingual + instruction go
  through lm-eval; coding through bigcode; safety through a **new fourth env**.
- **Merging six 8B models** is memory-heavier than gemma. The reader streams one
  tensor at a time, so peak RAM stays bounded, but give the merge job ≥ 64 GB RAM.

## Step 0 — sync + license

```sh
# Mac
git add -A && git commit -m "Tier 2 scaffold: Llama-3.1-8B five-domain config + drivers"
git push
# HPC (login001)
cd /work/pnag/Expert-Model-Merging && git pull
```

`meta-llama/Llama-3.1-8B` is gated. Accept the license on HF and confirm
`hf_token.txt` (repo root, gitignored) holds a token with access.

## Step 1 — download base + 5 experts

```sh
module load python3/anaconda/2023.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/pnag/envs/merging
export TMPDIR=/work/pnag/tmp; mkdir -p "$TMPDIR"

python scripts/mb_download.py --config configs/mergebench_tier2.yaml
```

Lands under `mb_ckpts/meta-llama__Llama-3.1-8B` and
`mb_ckpts/MergeBench__Llama-3.1-8B_<domain>` for all five domains (~6 × 16 GB).

## Step 2 — Tier 0 divergence (optional sanity check)

```sh
python scripts/mb_tier0_divergence.py --config configs/mergebench_tier2.yaml
```

Confirms the five experts are near-orthogonal before merging (as at Tier 1).

## Step 3 — merge (7 methods, N=5)

Run on a high-RAM node (≥ 64 GB). The merge is CPU-bound and needs no GPU.

```sh
# our methods + dataless baselines (skips RegMean/RegMean++)
python -u scripts/mb_tier1_merge.py \
  --config configs/mergebench_tier2.yaml --tier all
```

Writes `mb_merged/Llama-3.1-8B/{task_arith,whc_diag,TaskArithmetic,TIES,DARE,Consensus,LocalizeAndStitch}`.
Per-method failures are isolated; the summary lands in
`results/mergebench/tier1_merges_Llama-3.1-8B.json`.

Then fix the baseline tokenizers (our two dirs are already clean):

```sh
python scripts/mb_fix_tokenizers.py --config configs/mergebench_tier2.yaml
```

## Step 4 — lm-eval + bigcode evals

Reuse the existing `lmeval` and `bigcode` envs.

**Env gap to check first.** The Tier 1 `lmeval` env only ever ran `gsm8k_cot`.
Tier 2 adds two task families with extra needs:
- `ifeval` requires `langdetect`, `immutabledict`, `nltk` (`pip install` into the
  lmeval env; wheel-only, fine on RHEL7).
- the okapi multilingual tasks (`m_mmlu_*`, `arc_*`, `hellaswag_*`) must be in the
  task registry and download their own HF datasets. Verify before submitting:
  `lm_eval --tasks list 2>/dev/null | grep -E 'ifeval|m_mmlu_fr|arc_fr|hellaswag_fr'`.
  If any are missing, the env's lm-eval is too old — upgrade it in place (keep the
  numpy<2 pin).

```sh
# lm-eval: math + instruction + multilingual, all 13 models, 2 at a time
sbatch scripts/mb_eval_lm_tier2.sh
# bigcode: humanevalplus + mbppplus
sbatch scripts/mb_eval_code_tier2.sh
```

The lm-eval job is the long pole (gsm8k 8-shot generation + ~60 K multilingual
MC items per model). If it crowds the 12 h limit, split it and fan out:

```sh
GROUP=math         sbatch scripts/mb_eval_lm_tier2.sh
GROUP=instruction  sbatch scripts/mb_eval_lm_tier2.sh
GROUP=multilingual sbatch scripts/mb_eval_lm_tier2.sh
```

Smoke first if unsure: `LIMIT=8 GROUP=math sbatch scripts/mb_eval_lm_tier2.sh`
and `N_SAMPLES=1 BATCH=1 LIMIT=2 sbatch scripts/mb_eval_code_tier2.sh`.

## Step 5 — safety eval (new fourth environment)

MergeBench evaluates safety with a fork of `allenai/safety-eval` driven by vLLM,
not lm-eval. One-time setup on an L40S node:

```sh
cd /work/pnag
git clone https://github.com/uiuctml/safety-eval-fork.git   # confirm exact URL in MergeBench README
conda create -p /work/pnag/envs/safety-eval python=3.10 -y
conda activate /work/pnag/envs/safety-eval
export TMPDIR=/work/pnag/tmp
cd safety-eval-fork && pip install -e .                       # pulls vllm + classifiers
```

RHEL7 caveats from `project_eval_envs_hpc` apply (pin numpy<2; wheel-only). vLLM
needs a recent CUDA/torch wheel; if the L40S CUDA toolchain balks, that env is the
one place you may have to deviate from the standing pins. The WildGuard classifier
weights download on first run (gated → HF_TOKEN).

```sh
# smoke: one task, tiny limit, on the base model
TASKS=xstest LIMIT=8 sbatch scripts/mb_eval_safety_tier2.sh
# full: 4 safety tasks, all 13 models
sbatch scripts/mb_eval_safety_tier2.sh
```

Safety is the least load-bearing column for the WHC claim and the most setup, so
it is fine to land it last. The go/no-go (Step 6) reads on the cross-domain Avg.

## Step 6 — assemble the table

```sh
python scripts/mb_make_tier2_table.py --config configs/mergebench_tier2.yaml
```

Prints per-model rows and writes `results/mergebench/tier2_table_Llama-3.1-8B.json`.
It also emits HTML `<tr>` rows; paste them into the Tier 2 `<tbody>` in
[`results/mergebench/TIER1_TABLE.md`](results/mergebench/TIER1_TABLE.md), replacing
the empty cells. Bold the column maxima by hand.

**Reader locations** (the assembler handles these, listed for debugging):
- lm-eval: `results/mb_eval/Llama-3.1-8B/<tag>/**/results_*.json`, newest by mtime
  per group; `gsm8k_cot["exact_match,flexible-extract"]`,
  `ifeval["prompt_level_strict_acc,none"]`, multilingual per-task `acc`/`acc_norm`.
- bigcode: `results/mb_eval/Llama-3.1-8B/<tag>/code_eval.json`, `<task>["pass@1"]`.
- safety: `results/mb_eval/Llama-3.1-8B/<tag>/safety_eval.json`. Schema varies by
  safety-eval version, so the parser is best-effort — eyeball the safety column
  against the raw JSON before trusting it.

## Go / no-go

Tier 2 passes if HTCL is the dataless-tier column max (or tied) on the
cross-domain **Avg**, mirroring Tier 1. The five specialist rows give per-domain
ceilings for normalized-performance context. If HTCL holds, proceed to the parked
Tier 3 / data-tier items (`whc_tree_iter` port, RegMean/RegMean++ data tier, the
RL/KL variant) per [`HANDOFF.md`](HANDOFF.md) and `project_whc_variants_roadmap`.

## Resource cheat-sheet

| Stage | Node | GPU | Why |
|---|---|---|---|
| download / merge | any high-RAM (≥64 GB) | none | CPU + RAM, streams tensors |
| lm-eval | `AI_Center_L40S` / `gpu-v100-32gb` | 1 × 48/32 GB | 8B weights + gen |
| bigcode | `AI_Center_L40S` | 1 × 48 GB | n_samples 10, batch 10 |
| safety (vLLM) | `AI_Center_L40S` | 1 × 48 GB | vLLM KV cache |
