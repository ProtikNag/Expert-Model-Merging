# Session Handoff — WHC on MergeBench (gemma-2-2b, math+coding)

Paste this to a new Claude session to resume. Project memory already holds the
HPC env details ([[project_merging_env_hpc]], [[project_eval_envs_hpc]],
[[project_ml_env_transformers_pin]], [[feedback_env_setup]]) — read those first.

## What this is
Validating WHC (diagonal dataless curvature-aware merging, `whc_diag`) against
MergeBench baselines on gemma-2-2b, two domains (math, coding). Workflow is
git push from Mac → pull on HPC (login001), RHEL7. Repo:
github.com/ProtikNag/Expert-Model-Merging (branch main).

## Done
- All 7 merges complete on disk: `mb_merged/gemma-2-2b/{task_arith, whc_diag,
  TaskArithmetic, TIES, DARE, Consensus, LocalizeAndStitch}` + base in `mb_ckpts/`.
- Tier 0 divergence diagnostic done (experts near-orthogonal, mean cosine 0.094).
- Both eval envs built + harnesses patched (eager attn): lmeval + bigcode.
- Config locked: attn_implementation=eager, base tokenizer for all models,
  TMPDIR=/work/pnag/tmp. batch_size 4 on 16GB GPU / up to 16 on 48GB L40S.
- **MATH COLUMN COMPLETE (gsm8k_cot, full 1319, stderr ~0.014):**

  | method | gsm8k_cot |
  |---|---|
  | base | 0.2813 |
  | Consensus | 0.3306 |
  | TaskArithmetic (MergeBench) | 0.4215 |
  | TIES | 0.4230 |
  | LocalizeAndStitch | 0.4223 |
  | task_arith (ours, cross-check) | 0.4238 |
  | DARE | 0.4306 |
  | **whc_diag (ours)** | **0.4723** |

  WHC wins the dataless tier: +4.2 pts over best baseline (DARE), ~2σ.
  Cross-check holds (task_arith 0.4238 ≈ TaskArithmetic 0.4215), so the pipeline
  is validated. THIS IS A TIER-1 MATH PASS.

## TIER 1 COMPLETE — full table in results/mergebench/TIER1_TABLE.md
Both math + coding columns done, all 10 models + specialist brackets. whc_diag
wins all 3 benchmarks among dataless methods and has the best cross-domain avg
(38.2), beating both specialists' averages. Coding: whc_diag humaneval+ 0.2689 /
mbpp+ 0.4040 (#1 dataless, ties coding specialist 0.398 on mbpp+); coding_expert
ceiling 0.321/0.398; math_expert floor 0.181/0.323. Cross-check holds on all 3.

## NOW: adding Fisher to the table (data tier, our own clean impl)
Decided NOT to run MergeBench's Fisher (DeepSpeed+trl two-stage, infeasible on our
env). Instead compute diagonal Fisher ourselves and merge via our pipeline for a
controlled ablation: fisher_merge (plain Fisher avg = Matena&Raffel precursor),
whc_diag_fisher (our anchor + true Fisher), vs whc_diag (anchor + dataless
taskvec). Code: scripts/mb_fisher_estimate.py + fisher_merge in llm_merge.py.

## MATH also has specialist brackets (full 1319, complete)
math_expert 0.5603 (ceiling), coding_expert 0.3184. WHC recovers 68.5% of the
base->expert headroom vs DARE 53.5%, TA 50.3%; retains 84.3% of the math
specialist while also being a code model. Math column is DONE.

## In flight (as of session close)
- FULL CODING column launched: `sbatch scripts/mb_eval_code.sh` (humanevalplus +
  mbppplus, n_samples 10 / batch 10, all 10 models, %2) on L40S node493.
  ~4-6 hr. Driver, eager patch, and base-tokenizer fix for the 5 baselines all
  done. Coding smoke on base passed (evalplus present, code execution works,
  model load ~18s on L40S vs 8min on node242).

## Next steps (in order)
1. When `squeue -u pnag` shows no `whc_eval_code` tasks, read the coding table.
   NOTE coding results are in `<tag>/code_eval.json` (NOT results_*.json):
   parse `humanevalplus["pass@1"]` and `mbppplus["pass@1"]`. Reader is in this
   repo's git history / the assistant can regenerate it.
2. Assemble the full Tier-1 table: math (gsm8k_cot) + coding (humanevalplus,
   mbppplus), base + 7 merges + 2 experts. Go/no-go: does whc_diag match/beat
   the dataless tier on coding too (it already won math at +4.2pts, ~2sigma).
3. If WHC holds on coding => Tier-1 PASS, proceed to Tier 2.

## Math reader (lm-eval accumulates timestamped results_*.json, read NEWEST by
mtime, parse results.gsm8k_cot["exact_match,flexible-extract"]). Coding reader:
parse <tag>/code_eval.json pass@1 per task.

## Then (Tier 2, only if Tier 1 coding also passes)
Widen to all 5 domains (N=5); add data-using baselines RegMean/RegMean++
(`baselines_data` in configs/mergebench.yaml, needs data env + GPU); WHC fisher
ablation (whc_diag curvature=fisher). Tier 3 = larger bases + WHC tree/iterative
variants.

## PARKED — pick up later (do NOT forget)
Current focus is Option B = whc_diag (dataless) for ALL Tier experiments; Fisher
merging baseline NOT needed now. See memory [[project_whc_variants_roadmap]] for
detail. Three parked items:
1. WHC is actually TWO methods: whc_diag (dataless, diagonal taskvec ≈ Fisher
   merging; MergeBench) vs whc_tree (data, Gram+iterative ≈ RegMean+iteration;
   GLUE winner 0.667 > RegMean 0.609). After the Tier experiments, PORT the
   data-using whc_tree_iter to MergeBench and compare in the DATA tier vs
   RegMean/RegMean++/Fisher — that's the variant that beats the strong baseline.
2. Both whc_diag and whc_tree derive from ONE 2nd-order Taylor consolidation
   objective => paper framing is one framework, two instances (dataless + data).
3. ADD a new RL/KL WHC variant (docs/HTCL_Extension_Proposal (2).pdf): static
   closed form breaks on RL experts (state-visitation depends on w). Fix = HTCL-KL
   (Eq 10) with policy-drift KL + occupancy drift + advantage-weighted KL + kappa
   damping. A more compact unfinished version exists (see its figures). Do LAST.

## GPU / cluster notes (IMPORTANT — avoids hours of queue waiting)
- The `gpu` partition is ONE 16GB node (node242) and is heavily contended.
  `dgx_aic` is saturated with multi-day jobs. Do NOT default to these.
- Real capacity lives elsewhere. Survey with:
  `sinfo -o "%P %.6t %.6D %N %G"` and look for `idle`/`mix` GPU partitions.
- **L40S nodes (node493/494, partitions AI_Center_L40S / aic_L40S_short): 48GB
  Ada GPUs, often idle.** Best choice — big memory enables large batch (BATCH=16+)
  and the full coding protocol. Submit with `-p AI_Center_L40S`.
- `gpu-v100-32gb` (node363-382) and `gpu-v100-16gb` also have many free GPUs.
- GPU types differ (V100 16/32GB Volta, L40S 48GB Ada). LLM generation is
  memory-bandwidth-bound (V100 ~900GB/s ≈ L40S ~864GB/s per token), so L40S wins
  mainly via larger batch, not per-token speed.
- Override the script's `#SBATCH -p gpu` with `sbatch -p <partition>`. Lower
  `--time` to help backfill.

## Hard-won gotchas (don't relearn)
- Activate envs by FULL PREFIX PATH: `module load python3/anaconda/2023.7` then
  `source "$(conda info --base)/etc/profile.d/conda.sh"` then
  `conda activate /work/pnag/envs/<env>` (merging | lmeval | bigcode).
- RHEL7 gcc 4.8.5: any non-wheel package source-builds and dies. Pin numpy<2,
  pandas<2.3, pyarrow<15, datasets==2.18.0, transformers==4.44.2 (eval envs).
- /tmp is only 2G → always `export TMPDIR=/work/pnag/tmp` before pip/large jobs.
- gemma2 needs eager attention (SDPA cutlassF failure); batch 4 fits 16GB,
  256k-vocab logits OOM at batch 16 on 16GB (fine on 48GB).
- eval in bf16 + base tokenizer + 8-shot gsm8k_cot matches MergeBench's protocol.
