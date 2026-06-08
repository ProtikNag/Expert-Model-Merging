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

### Sweep result so far (POSITIVE)
`scripts/mb_sweep_whc.py` merged a 3 lam x 4 alpha dataless grid; gate-eval via
`scripts/mb_eval_sweep_{lm,code}.sh` (math+instr+coding only, LIMIT=500, n_samples=5).
At lam=1e-3: instr **14 -> 27 -> 31** as alpha 1->2->3 (alpha=3 beats every
baseline's 27.0); math **74 -> 80 -> 77**. Partial gate (math+instr): a3=54.0,
a2=53.3 > best baseline 52.75. **alpha rescued HTCL on the measured domains.**
Coding gate was still running at handoff; other lam blocks pending. Rank with
`python scripts/mb_sweep_table.py --config configs/mergebench_tier2.yaml`.

## OPEN QUESTION — does it hold for all five?
We have tuned alpha on math+instr+coding only. Two unknowns:
1. **Coding** — HTCL's weak spot (heval+ 41.5). Data landing; could go either way.
2. **Safety** — UNTESTED (the 4th eval env is not built) and the real risk: a more
   aggressive merge (larger alpha) is exactly what erodes refusal behavior. A
   five-task win is NOT established until the promoted variant is evaluated on all
   five domains incl. safety and the five-domain average still beats baselines.

## Next steps (in order)
1. Finish the gate; `mb_sweep_table.py` to pick the best (lam, alpha) — likely
   `whc_tv_l1e-3_a2` or `_a3`.
2. Promote the winner to a FULL eval: math+instr+coding at full size/n_samples=10,
   plus multilingual on L40S (ATTN=sdpa). Add its row to the Tier 2 table.
3. **Build the safety env** (safety-eval-fork + vLLM, TIER2_RUNBOOK Step 5) and run
   safety for the winner + baselines. Confirm the five-domain average holds.
4. If safety drops under the winning alpha: try a smaller alpha or a safety-aware
   anchor — do NOT abandon alpha; let data decide.

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
