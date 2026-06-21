# Session Handoff — HTCL on MergeBench (Tier 2, Llama-3.1-8B, five domains)

Paste this to a new Claude session to resume. Project memory holds the HPC env
details ([[project_merging_env_hpc]], [[project_eval_envs_hpc]],
[[project_tier2_scaffold]], [[feedback_env_setup]]) — read those first. The full
operational guide is [`TIER2_RUNBOOK.md`](TIER2_RUNBOOK.md).

Branch: `tier2-llama-scaffold` (NOT merged to main). Repo:
github.com/ProtikNag/Expert-Model-Merging. Workflow: push from Mac, pull on HPC.

## ✅ CURVATURE CAMPAIGN — CLOSED 2026-06-21 (conclusive negative; the diagnosis is the contribution)
2026-06-18→21. The user asked for a *genuine* curvature/Taylor surrogate to numerically BEAT the
hand-tuned champion (Bet D, avg 55.43). It cannot, and we now know exactly why. Full ledger:
**[`results/mergebench/EXPERIMENTS_curvature.md`](results/mergebench/EXPERIMENTS_curvature.md)**.
Code: `scripts/mb_fit_perexpert_surrogate.py` (scalar, 5 DOF) + `scripts/mb_fit_perlayer_surrogate.py`
(per-block, 40 DOF) + wrappers `mb_fit_surrogate_tier2.sh` / `mb_fit_perlayer_tier2.sh` (env `merging`).

**Outcome:** every curvature-derived candidate loses to the champion.
- **Phase 1 (per-expert scalar) — EXHAUSTED.** C 52.58; cold pooled `ta_pe_mix_pool_kl` ~53.0; clean
  ml-boost `ta_pe_inst0.8_mult0.73` 53.96. The champion `[0.8,0.4,0.4,0.4,0.4]` is the Pareto-optimal
  scalar point.
- **Phase 2 (per-block, 40 DOF) — FAILS.** `ta_pl_b8` cold (gsm8k 0.594 @ math mean 0.19). Final
  `ta_pl_b8_frzgen` (freeze generative at champion, per-block solve ONLY safety+ml): **ifeval 23.66
  (−14.1), gsm8k@500 64.80 (−12.7), ml 48.84 (−2.9)** — generative collapsed even though FROZEN, and ml
  (the target) fell. Loses by ~5. Numbers in `results/mb_eval/Llama-3.1-8B/ta_pl_b8_frzgen/`.

**The contribution (three findings, write these up):** (1) **the interference wall** — generative
domains want HIGH own-coeff, discriminative ones want LOW interference; no shared coefficient crosses it
because at N=5 the experts are entangled in weight space (share the same tensors); (2) **metric-
faithfulness asymmetry** — a distillation surrogate is faithful to distribution-matching metrics
(ml acc_norm, safety RTA) but ANTI-faithful to argmax-generative (ifeval/gsm8k/pass@1); (3) **per-block
depth-sensitivity** — placing a domain in some blocks breaks depth-sequential generative computation
and perturbing late/lm_head blocks collaterally destroys generative decoding. **Framing:** Bet D (55.43,
dataless) is the positive result; this campaign is the analysis of why curvature coefficient-derivation
does not improve on a benchmark-tuned per-expert merge at N=5. NOT a curvature-SOTA claim.

**Still finishing (does NOT change the verdict):** coding + safety gates for `ta_pl_b8_frzgen` (idx18)
running on node493; numbers will be appended to the ledger table when they land.

## ✅ WIN SECURED — per-EXPERT coefficient decoupling (Bet D)
2026-06-16. We HAVE the dataless win. After weight-space routing failed/tied (Bets A/B below),
the entanglement finding said the separable axis is the EXPERT, not the weight. Bet D acts on
that axis: `w = w_pre + Σ_i s_i·τ_i`, each expert its own scalar. Result on the T1 gate:
**`ta_pe_inst0.8_codi0.4` = GATE 54.1 vs Consensus 51.8 (+2.3)** AND **best forgetting of all
methods, MEANF −5.4 vs −3.0** — wins both axes, dataless. instr 25.1→38.0, coding held. 5/6
variants beat top baseline. Full tracking in **`results/mergebench/EXPERIMENTS_routed.md`**.
Code: `merge_task_arith_perexpert_multi` (`mergebench/llm_merge.py`); driver
`scripts/mb_sweep_perexpert.{py,sh}`.

**Round 1 refinement — DONE.** Champion `ta_pe_inst0.8_codi0.4` stays on top; whole neighborhood
53.1–53.8 gate / −4.3…−5.0 forgetting = flat-topped optimum (not overfit). **Full protocol
(limit=None, n_samples=10) CONFIRMS:** gate 53.2 (+1.4), 4-domain avg 53.9 vs 51.9 = **+2.0**
(instr +12.6, coding −3.1, math/multiling flat). NOT a gate-limit artifact.

## ⏸ PAUSED 2026-06-17 — only the SAFETY column remains (user is running another project)
4 of 5 domains done + win secured. SAFETY (5th domain) is the last piece. ALL prereqs cleared,
**but nothing is launched** (user paused). Full detail in `EXPERIMENTS_routed.md` SAFETY section.
- Env `/work/pnag/envs/safety-eval` built & proven (smoke loaded champion in vLLM, generated all
  prompts). Recipe in memory [[hpc_safety_eval_env_rhel7]].
- WildGuard gating **✅ accepted & verified** (ProtikNag token gets HTTP 200 on the gated repo).
- Baseline re-merges **in flight** (jobs 21584950 light / 21584951 heavy; slow GPFS I/O; should
  self-complete). Champion weights already on disk.
- **RESUME WITH:** `sbatch --array=0-13%4 scripts/mb_eval_safety_tier2.sh` (idx 13=champion ready
  now). Then assemble final 5-domain table: `scripts/mb_make_tier2_table.py`.
  Caveat: fork's eval.py IGNORES `--limit`, so runs are full-size; --time=08:00:00.

**OPTIONAL later (not blocking the win):** buffer-LEARN {s_i} for a data-light variant; fold into
curvature framework (per-expert importance × per-param curvature); multi-seed. Clean mb_merged
weight dirs between rounds (eval results live in `results/mb_eval/`, weight dirs disposable).

### Weight-space routing (Bets A/B) — FAILED/KILLED (kept for the paper)
Bet A dominance-ROUTED per-param α (`pscale="consensus_routed"`) best 49.1 — cross-talk: pinning
coding-owned params at α=1 doesn't protect coding. Bet B per-layer α killed by the entanglement
diagnostic (coding task-vector energy FLAT ~22% across all 32 layers). KEY PAPER FINDING: at N=5
the domains are entangled in WEIGHT space; no per-param/per-layer partition separates them — only
the per-EXPERT axis does. Cache retained at `mb_cache/Llama-3.1-8B_l1e-3`.

---

## ⏩ PRIOR WORK — dataless win attempt via per-parameter scaling
Started 2026-06-16. The dataless tie (global-alpha HTCL 51.3 vs Consensus 51.8)
is capped by a SINGLE global `alpha` (instruction wants high, coding low). The
fix being tested: **per-parameter update scaling** in `whc_diag`, derived
dataless-ly from inter-expert agreement. Two modes (`mergebench/llm_merge.py`
`pscale=`):
- **consensus** (primary bet): rescale the curvature-weighted-mean update up to
  the *agreeing* experts' additive task-vector sum, capped at `alpha_max`. Leaves
  single-expert params at ~1 (no over-extrapolation — the flaw that makes pure
  coherence reproduce the coding loss) and bounds conflicted params. Unifies the
  §11 dilution analysis with TIES-style sign consensus.
- **coherence** (the AAAI-plan form, run for comparison): `a_p = 1 + (amax-1)*coh^beta`.

Code: `mergebench/llm_merge.py` (`pscale`,`alpha_max`,`beta`; + single-pass
`merge_whc_diag_pscale_multi` that reads the 6 models ONCE for all variants),
driver `scripts/mb_sweep_pscale.{py,sh}`, tests `tests/test_pscale.py` (all pass),
ledger [`results/mergebench/EXPERIMENTS_pscale.md`]. Commits through `a65ba8d`.

**Round 0 — DONE (verdict in).** 5 variants (`whc_cons_l1e-3_am{3,5,8}`,
`whc_coh_l1e-3_am5_b{1,2}`). Best `cons_am3` gate **50.7 < 51.0 kill gate** (below the
51.3 tie and Consensus 51.8). Mechanism validated though: consensus > coherence at every
α_max; **instruction 32.4 = best of any method**. Full table + analysis in
`results/mergebench/EXPERIMENTS_pscale.md`.

**Forgetting table — NEW (user-requested).** `scripts/mb_forgetting_table.py`,
forgetting_d = specialist_d − merged_d under the identical T1 protocol. Headline:
**`cons_am3` has the LOWEST instruction forgetting (+14.6) of every method** (Consensus
+21.8, TIES +30.3). NB: the math specialist scores only 36.8 on gsm8k_cot completion
format (chat-SFT format mismatch) so all merges "beat" it — math forgetting is negative
for everyone, not a bug; instruction is the meaningful axis. Coding-axis cells pending
`coding_expert` code eval (job 21584012; staged at `mb_ckpts/_stage_coding_expert` =
expert weights + base tokenizer, to dodge the newer-`tokenizer.json` parse error).

**Round 1 — DONE (verdict in, 2026-06-16). NO AGGREGATE DATALESS WIN → fold to analysis.**
All Round-1 + coding_expert evals drained; both tables run on `pscale_r1_all.txt`. The α_max
peak is now fully bracketed (am1=46.4, am2=51.3, am2.5=51.0–51.3, am3=50.3–50.7, am3.5=47.4,
am4=45.3, am5=35.2): one broad plateau at **51.0–51.3**. Best variant
`whc_cons_l3e-4_am2.5 = 51.3` **clears the 51.0 kill gate but only TIES the tuned global-alpha
HTCL (51.3) and stays below baseline Consensus (51.8)** — instruction gains are exactly offset
by mbpp+/heval+ coding loss along α_max. Lowering lam (3e-4) did not move the ceiling.
**Decision: honor the design, no promotion to full eval.** Surviving contributions (the paper):
(1) consensus > coherence at every α_max; (2) **lowest instruction forgetting of any method**
(+14.6 at am3 vs Consensus +21.8 / TIES +30.3); (3) a dataless closed-form per-parameter
dilution-undo with a fully-characterized accuracy/forgetting trade-off in α_max. Full Round-1
gate + FINAL forgetting tables in `results/mergebench/EXPERIMENTS_pscale.md`.

**If the pipeline died / needs re-running:**
```sh
# re-merge (single pass; whc_diag does NOT need BigMem — it streams per-key, but
# BigMem 2TB avoids GPFS page-cache eviction; any idle node works, AI_Center_L40S
# node493 starts fastest). ~15-30 min depending on GPFS contention.
sbatch -p BigMem-64core -n 8 --mem=300G --time=02:00:00 scripts/mb_sweep_pscale.sh
# re-eval (array 0-4 = the 5 variants):
MAN=mb_merged/Llama-3.1-8B/pscale_manifest.txt
MANIFEST=$MAN sbatch -p gpu-v100-32gb  --array=0-4%5 scripts/mb_eval_sweep_lm.sh
MANIFEST=$MAN sbatch -p AI_Center_L40S --array=0-4%5 scripts/mb_eval_sweep_code.sh
```

**HPC gotcha discovered this session (add to env memory):** the per-key merge is
**GPFS-mmap-I/O-bound** (safetensors reads via mmap; under cluster I/O contention
the page faults stall — `cxiWaitEventWait`/`gpfs_filemap_fault`, ~73% CPU). It is
NOT compute-bound and NOT fixed by more RAM. Symptom: merge sits at "<50 keys" for
many minutes. It still completes (~6 keys/min worst case). `defq-*` general
partitions can pend for a LONG time on `Priority` even with idle cores; the
`AI_Center_L40S`/`BigMem-64core` queues start in <1 min. A CPU-only merge can be
sent to a GPU partition (no `--gres`). To repoint a dependent eval after moving a
merge: `scontrol update jobid=<eval> Dependency=afterok:<newmerge>` then scancel
the old merge.

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
0. **DATA TIER IS DONE — it's a loss.** Full campaign in
   [`results/mergebench/EXPERIMENTS_whc_gram.md`]: `whc_gram` caps at ~49 gate
   (< dataless tie 51.3 < baselines 51.8). alpha refuted (catastrophic on the
   full-covariance solve), iterative K=1 flat (the GLUE edge does not transfer).
   Stop pushing the data tier.
1. **AAAI win attempt — IN PROGRESS, see the ⏩ CURRENT WORK section at top +
   [`results/mergebench/EXPERIMENTS_pscale.md`] + [`docs/AAAI_PLAN.md`].** Target:
   a robust >1 pt N=5 win on the DATALESS side via **per-parameter update scaling**
   of `whc_diag`. Refined the AAAI-plan bet: pure coherence-gating over-boosts
   single-expert params (the coding-loss mechanism), so the primary variant is
   **consensus** (rescale to the agreeing experts' additive sum, cap alpha_max);
   coherence is run alongside for comparison. Phase 1 (T1 gate) RUNNING now; kill
   gate best gate < 51.0 -> fold to analysis. The whc_gram infra (merge/eval
   drivers, T1/T2 proxy) is reused as-is.
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
- **whc_tree (Gram, iterative / "catch-up")** — PORTED as `whc_gram` (Next-steps
  #1, TIER2_RUNBOOK Step 9). The single-pass merge + Gram estimator + iterative
  loop are coded and unit-tested; what remains is the HPC run and, if it clears
  the baselines, folding the result into the table. See [[project_whc_variants_roadmap]].
- **Data Fisher sweep** — `scripts/mb_fisher_tier2.sh` then
  `FISHER_ROOT=... sbatch scripts/mb_sweep_merge.sh` (verify the `<domain>_val`
  dataset ids on HF first).
- **RL/KL HTCL variant** (docs/HTCL_Extension_Proposal) — do last.
