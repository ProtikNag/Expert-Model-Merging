# Session Handoff — HTCL on MergeBench (Tier 2, Llama-3.1-8B, five domains)

Paste this to a new Claude session to resume. Project memory holds the HPC env
details ([[project_merging_env_hpc]], [[project_eval_envs_hpc]],
[[project_tier2_scaffold]], [[feedback_env_setup]]) — read those first. The full
operational guide is [`TIER2_RUNBOOK.md`](TIER2_RUNBOOK.md).

Branch: `tier2-llama-scaffold` (NOT merged to main). Repo:
github.com/ProtikNag/Expert-Model-Merging. Workflow: push from Mac, pull on HPC.

## What this is
The merging method is named **HTCL** (the dataless diagonal variant; `whc_diag`
in code). Dataless, one-shot, curvature-aware merging from one 2nd-order Taylor
consolidation objective. We are validating it on MergeBench.

## Tier 1 — DONE, PASS (gemma-2-2b, math+coding, N=2)
HTCL wins the dataless tier on all three benchmarks and has the best cross-domain
average of all 12 models; the dataless task-vector proxy matches true Fisher. Full
table in [`results/mergebench/TIER1_TABLE.md`](results/mergebench/TIER1_TABLE.md).

## Tier 2 — IN PROGRESS (Llama-3.1-8B, all five domains, N=5)
Base = ungated mirror `NousResearch/Meta-Llama-3.1-8B` (official meta-llama is
gated, access pending; `scripts/mb_verify_base.py` confirms bit-identity once
granted). All five experts downloaded; all 7 merges built (BigMem; TIES stacks a
~160 GB matrix so it needs the 2 TB node).

### THE KEY FINDING — HTCL needs a task-vector scale at N>2
Plain `whc_diag` (lam=1e-4, alpha=1) **underperformed every dataless baseline** at
N=5 (gate = math+instr+coding): instr 15.3, math 73.2, vs baselines ~25-27 / ~78.
Cause: the closed form returns a curvature-weighted **mean** of experts, which
dilutes each update by ~1/N versus task arithmetic's **sum**. That is why it won at
N=2 but lost at N=5. Fix (in `mergebench/llm_merge.py`): an **update scale alpha**,
`w_M = w_pre + alpha*(w_M^HTCL - w_pre)`, with `alpha ~ N` undoing the dilution.
Distinct from the per-expert weights (still uniform 1/N). See NOTES.md §11.

### Sweep result — COMPLETE, and the verdict is a TIE (not a win)
`scripts/mb_sweep_whc.py` merged a 3 lam x 4 alpha dataless grid; gate-eval via
`scripts/mb_eval_sweep_{lm,code}.sh` (math+instr+coding, LIMIT=500, n_samples=5);
rank with `scripts/mb_sweep_table.py`. Best variant **`whc_tv_l1e-3_a2` (alpha=2)
gates at 51.3**, just UNDER Consensus 51.8 / TaskArithmetic 51.6. **No (lam, alpha)
clears the baseline cluster.** Why: alpha raises math+instruction sharply (instr
14->27->31 as alpha 1->2->3) but **trades away coding** (heval+/mbpp+ fall as alpha
rises); a single global alpha cannot satisfy instruction (wants high alpha) and
coding (wants low alpha) at once, so the net is a wash. The early "alpha rescued
HTCL" read was from math+instr ONLY; with coding in, it is a statistical tie.

Full four-domain table (multilingual now in; safety pending) in
[`results/mergebench/TIER1_TABLE.md`](results/mergebench/TIER1_TABLE.md): tuned HTCL
~ baselines (~52), default HTCL (alpha=1) last among merges (47.9). Multilingual is
non-discriminative (~52-54 for all, base included).

## CONCLUSION — the dataless contribution is a tie, not SOTA
At N=5, tuned HTCL is competitive with but does not beat the dataless tier. Polishing
alpha will not change this (the coding/instruction trade-off is structural). The
leverage for a stronger result is NOT the dataless sweep. See the strategy notes in
the conversation: target TMLR/CoLLAs/workshop, not a dataless-SOTA claim.

## Next steps (in order)
1. **Port `whc_tree` (data, iterative variant) to MergeBench and test vs
   RegMean/RegMean++.** Highest-leverage experiment — the real shot at a
   "beats the strong baseline" headline. GLUE precedent: whc_tree 0.667 > RegMean
   0.609. Needs RegMean-style activation statistics on a data slice + the iterative
   tree merge (new code). This decides whether the paper is a tie-study or stronger.
2. **Complete the table honestly:** build the safety env (safety-eval-fork + vLLM,
   TIER2_RUNBOOK Step 5), full-eval the best dataless variant `l1e-3_a2`
   (LIMIT=0, n_samples=10, + multilingual on L40S ATTN=sdpa), run safety for the
   merges + baselines. Safety is the one domain where the tie could become a loss
   (aggressive merge erodes refusal).
3. **Fix the math expert ceiling** — gsm8k still 36.8 < base; the `TOK=self` rerun
   did not take. Quick check (did 21561683 run? is its results file newer?).
4. **Add multi-seed** before any submission — single seed is the biggest reviewer
   objection.
5. Lock framing around the unification + the N-scaling dilution analysis (+ data
   win if #1 lands).

## Eval state (as of handoff)
- Merges math+instr+coding: done (cross-check holds: our task_arith ~ MergeBench
  TaskArithmetic). Multilingual: running on L40S, nearly complete.
- Experts: math+instr done but ceilings were WRONG (math_expert gsm8k 36.8 < base)
  because the base-tokenizer override breaks the experts' own format. Rerun with
  `TOK=self` (the lm driver now supports it); expect math_expert ~78.
- Safety: NOT started (needs the 4th env).

## Hard-won infra gotchas (don't relearn)
- **V100 (Volta) needs eager attention** for Llama too — SDPA hits "cutlassF: no
  kernel found to launch!" (GPU-specific, NOT gemma-only). The lm driver forces
  `attn_implementation=eager` (knob `ATTN`).
- **Multilingual MUST run on L40S (Ada) with SDPA**: 378,872 okapi loglikelihood
  requests OOM on the 32 GB V100 under eager and ETA was 151 h. On L40S with
  `ATTN=sdpa BATCH=auto GROUP=multilingual` it does ~1 model/hour.
- **Merges need BigMem-64core (~2 TB)**: MergeBench's TIES/DARE/L&S allocate tens-
  to-hundreds of GB (not streamed); our `whc_diag` streams per-key (~30 GB peak).
- **ifeval is the slow eval tail** (merged models ramble to max-length; ~30 s/it).
- **Never run merges/eval on the login node** (cgroup OOM = "Killed").
- Activate envs by FULL PREFIX PATH (merging | lmeval | bigcode | safety-eval).
  If `python` throws a SyntaxError on an f-string, the conda env dropped -> you are
  on system Python 2; re-activate.

## PARKED — pick up later
- **whc_tree (Gram, iterative / "catch-up")** — the GLUE winner that beat RegMean;
  the data-using variant for the strong-baseline regime. Port to LLMs (needs
  RegMean-style activation statistics) is the next phase if the dataless alpha
  sweep + Fisher do not fully clear the baselines. See [[project_whc_variants_roadmap]].
- **Data Fisher sweep** — `scripts/mb_fisher_tier2.sh` then
  `FISHER_ROOT=... sbatch scripts/mb_sweep_merge.sh` (verify the `<domain>_val`
  dataset ids on HF first).
- **RL/KL HTCL variant** (docs/HTCL_Extension_Proposal) — do last.
