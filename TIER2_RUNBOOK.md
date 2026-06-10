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

**Base model.** The official `meta-llama/Llama-3.1-8B` is gated and access was
pending, so the config points at the ungated mirror `NousResearch/Meta-Llama-3.1-8B`
(byte-identical re-upload). The experts and merged-checkpoint paths still use
`base_name: Llama-3.1-8B`. Once official access is granted, download the canonical
repo and run `scripts/mb_verify_base.py` (Step 6.5) to confirm the merge anchor was
bit-identical. `hf_token.txt` (repo root, gitignored) needs a valid token; the
mirror and the five experts download cleanly with it (nothing here is gated).

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

## Step 7 — make HTCL competitive (the sweep)

The plain closed form (`lam=1e-4`, `alpha=1`) underperforms the dataless baselines
at N=5 because it returns a curvature-weighted *mean* of the experts, diluting each
update by ~1/N versus a *sum* of task vectors. The fix is the new `alpha`
task-vector scale (`w_M = w_pre + alpha*(w_M^HTCL - w_pre)`, `alpha ~ N` undoes the
dilution) plus a tuned `lam`, evaluated fairly the same way the baselines are tuned.

**7a. Dataless (lam, alpha) sweep — do first.** Merge the grid on BigMem, then
gate-eval on math + instruction + coding only (skip the slow/OOM multilingual):

```sh
# merge the grid (default 3 lam x 4 alpha = 12 variants, ~4 h on BigMem)
sbatch scripts/mb_sweep_merge.sh
# fixup baseline-style tokenizers not needed (whc variants save base tokenizer)

# gate-eval all variants (LIMIT=500 gsm8k/ifeval; coding n_samples=5). Set the
# array to the variant count (12 -> 0-11).
sbatch -p gpu-v100-32gb --array=0-11%6 scripts/mb_eval_sweep_lm.sh
sbatch -p gpu-v100-32gb --array=0-11%6 scripts/mb_eval_sweep_code.sh

# rank
python scripts/mb_sweep_table.py --config configs/mergebench_tier2.yaml
```

The ranker prints each variant's gate score and whether the best HTCL clears the
top dataless baseline. Promote the winner to the full four-domain eval by adding
its tag to the main drivers (or just eval it full: `LIMIT=0` on the sweep lm
driver, `N_SAMPLES=10` on the sweep code driver, restricted to its index).

**7b. Data (Fisher) sweep — optional second.** Estimate per-expert diagonal Fisher,
then add it to the sweep:

```sh
# VERIFY the <domain>_val dataset ids in the script exist on HF first
sbatch scripts/mb_fisher_tier2.sh
# add the Fisher curvature to the grid (doubles the variant count)
FISHER_ROOT=mb_fisher/Llama-3.1-8B sbatch scripts/mb_sweep_merge.sh
# re-run the sweep evals with the manifest now listing the fisher variants too
```

**7c. Data tier (`whc_gram` = HTCL-data) — now implemented, see Step 9.** The
GLUE winner that beat RegMean is the Gram-based merge. It is ported to the 8B
scale as the `whc_gram` method (single-pass N-expert RegMean with a
ridge-toward-mean and an optional Fisher ridge) plus the iterative catch-up loop.
This is the real shot
at clearing the dataless tie, since 7a/7b show one global alpha cannot serve
instruction and coding at once. Run it per Step 9.

## Step 8 — finish the main table (infra fixes)

**Multilingual on L40S.** It OOM'd / was infeasibly slow on V100 under eager
(378k loglikelihood requests). Re-run just that group on the L40S with SDPA and
auto batch:

```sh
ATTN=sdpa BATCH=auto GROUP=multilingual sbatch -p AI_Center_L40S --array=0-12%2 scripts/mb_eval_lm_tier2.sh
```

(`ATTN=`/`GROUP=`/`BATCH=` are env vars and must all come **before** `sbatch`.)

**Expert ceilings.** The pure-expert rows scored below base because the base
tokenizer override broke their chat format. Re-run the expert indices with their
own tokenizer:

```sh
TOK=self sbatch -p gpu-v100-32gb --array=8-12%5 scripts/mb_eval_lm_tier2.sh
```

## Step 9 — data tier (`whc_gram` = HTCL-data vs RegMean / Fisher)

The data-using merge. Three jobs: estimate per-expert input Grams (GPU), run the
`whc_gram` merge (high-RAM CPU), gate-eval via the existing sweep drivers.

**9a. Estimate Grams (GPU array, one domain per task).** Verify the `<domain>_val`
dataset ids first (same caveat as Fisher). The default excludes `mlp.down_proj`
so the per-expert accumulator fits 96 GB; that is a fast, tractable first pass.

```sh
# VERIFY MergeBench/<domain>_val ids exist on HF (must match the Fisher ids)
sbatch scripts/mb_gram_tier2.sh                          # 5 domains, no down_proj
# faithful all-Linear version (needs a high-RAM node; V100 may not grant it):
EXCLUDE="" sbatch -p AI_Center_L40S --mem=200G scripts/mb_gram_tier2.sh
```

**9b. Merge (high-RAM CPU on BigMem).** Writes `whc_gram_l<lam>_g<gamma>` dirs and
a manifest the sweep evals read.

```sh
LAMS=0,1e-3,1e-2,1e-1 GAMMAS=0 sbatch scripts/mb_merge_whc_gram.sh
# lam=0 is plain RegMean (the ablation point); lam>0 is the ridge toward the mean.
# Add the Fisher ridge once mb_fisher/ exists:
LAMS=1e-3 GAMMAS=0.1,1.0 FISHER_ROOT=mb_fisher/Llama-3.1-8B sbatch scripts/mb_merge_whc_gram.sh
```

**9c. Gate-eval (reuse the sweep drivers via the whc_gram manifest).** Set the
array to the variant count (4 lams -> `0-3`).

```sh
MANIFEST=mb_merged/Llama-3.1-8B/whc_gram_manifest.txt \
  sbatch -p gpu-v100-32gb --array=0-3%4 scripts/mb_eval_sweep_lm.sh
MANIFEST=mb_merged/Llama-3.1-8B/whc_gram_manifest.txt \
  sbatch -p AI_Center_L40S --array=0-3%4 scripts/mb_eval_sweep_code.sh
python scripts/mb_sweep_table.py --config configs/mergebench_tier2.yaml \
  --manifest mb_merged/Llama-3.1-8B/whc_gram_manifest.txt
```

**9d. Iterative catch-up (`K>=1`, the GLUE winner's edge).** Re-estimate each
domain's Grams *on the round-(k-1) merged model*, then re-merge the original
experts with the refreshed Grams. Reuses the same two scripts, only `--expert`
(now the merged dir) and `--out` (a round-`k` Gram dir) change:

```sh
BEST=mb_merged/Llama-3.1-8B/whc_gram_l1e-3_g0          # pick the 9c winner
for D in instruction math coding safety multilingual; do
  python -u scripts/mb_gram_estimate.py --expert "$BEST" \
    --tokenizer mb_ckpts/NousResearch__Meta-Llama-3.1-8B \
    --dataset MergeBench/${D}_val --out mb_grams/Llama-3.1-8B_k1/${D} \
    --exclude-modules down_proj
done
GRAM_ROOT=mb_grams/Llama-3.1-8B_k1 LAMS=1e-3 GAMMAS=0 sbatch scripts/mb_merge_whc_gram.sh
# eval the K=1 variant the same way as 9c; repeat for K=2 if it still improves.
```

**Baselines for the head-to-head.** Fisher (`fisher_merge`, `whc_diag_fisher`)
is already wired — `scripts/mb_merge_fisher.py` after Step 7b's `mb_fisher/`.
MergeBench's own `RegMean`/`RegMeanPlusPlus` are data-using and need MergeBench's
data pipeline deps installed (not stubbed) plus the task datasets; run them via
`mb_tier1_merge.py --tier all --only RegMean,RegMeanPlusPlus` once that env is
built. `whc_gram` at `lam=0` is our internal RegMean reference if theirs is
blocked.

## Step 6.5 — verify the base mirror (once official access lands)

The merge anchored task vectors at the ungated `NousResearch/Meta-Llama-3.1-8B`
mirror. When `meta-llama/Llama-3.1-8B` access is granted, download it and confirm
the mirror was bit-identical:

```sh
python scripts/mb_download.py --config configs/mergebench_tier2.yaml \
  --domains ""   # base only, or temporarily point base_model at meta-llama/...
python scripts/mb_verify_base.py \
  --mirror mb_ckpts/NousResearch__Meta-Llama-3.1-8B \
  --official mb_ckpts/meta-llama__Llama-3.1-8B
```

A clean run (0 mismatches, max diff 0) means no redo. If it differs non-trivially,
re-merge from the official base and re-run the evals.

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
