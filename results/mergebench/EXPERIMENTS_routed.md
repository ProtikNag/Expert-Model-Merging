# Win campaign v2 — break the instruction↔coding α coupling

Started 2026-06-16, after the pscale campaign (`EXPERIMENTS_pscale.md`) confirmed
**no aggregate dataless win**: every per-parameter α scheme plateaus at gate 51.0–51.3,
under baseline Consensus (51.8). User directive: **get a win from ANY axis** (less/no/some
data, fewer epochs, better accuracy, lower forgetting) — explore novel ideas, do not fold
until something wins. Track everything here.

## The lever we proved exists

Rounds 0–1 measured a clean, monotone, *opposite* α-response per domain:
- **instruction:** ifeval 15.3 → 32.4 as α 1 → 3 (wants HIGH α).
- **coding:** mbpp+ 55.7 → 39.8, heval+ 41.5 → 37.7 as α 1 → 4 (wants LOW α).
- math rises mildly (73 → 77); safety/multilingual not in the T1 gate.

A single α — global or per-parameter consensus — can only sit at the crossover (~51).
**The win is to decouple:** high α where instruction lives, α≈1 where coding lives.

## Why "replay buffer" the obvious way is OUT

The repo's data tier already IS a replay buffer: `scripts/mb_gram_estimate.py` /
`mb_fisher_estimate.py` collect 256 val examples/domain → activation Grams / Fisher →
`whc_gram` RegMean merge. It **capped at 49.1** (`EXPERIMENTS_whc_gram.md`), below dataless.
Structural: the Gram solve extrapolates and α can't rescue it. So a buffer→*statistics*
path is a known dead end. A buffer's remaining value is to **fit a few scaling
coefficients** (Bet B), not to estimate covariance.

## Attack plan (ranked)

| Bet | Idea | Win axis | Data | Status |
|---|---|---|---|---|
| **A** | dominance-**routed** per-param α_max: param owned (argmax\|τ_i\|) by coding → cap α_low; else α_high. Dataless, closed-form, one pass. | accuracy, dataless | none | **RUNNING (R0)** |
| **B** | replay-buffer-calibrated per-layer/group α: cache `u`+consensus ratio once, fit ~2–32 caps on 32–64 ex/domain (CLM loss / gate proxy). | accuracy, data-light | tiny | queued (if A is close) |
| **C** | merge-as-init + few-epoch finetune on buffer → beat from-scratch multitask at a fraction of the epochs. | efficiency | tiny | safety net |

Mechanism note (A): consensus already leaves *single-expert* params ≈1, yet coding still
fell as α rose — so the coding loss is partly **cross-talk** (boosting other params shifts
activations into coding layers). Routing tests whether explicitly pinning coding-OWNED
params at α_low recovers coding while instruction keeps its high-α gain. If coding recovers
toward α=1 levels (mbpp+ ~55, heval+ ~41) with instruction near 30+, gate → ~51.7–52.x.

## Code

`mergebench/llm_merge.py`: new `pscale="consensus_routed"` (+ helpers `_consensus_ratio`,
`_dominant_expert`); takes `low_idx` (expert indices to protect) and `alpha_low`. Reuses the
single-pass `merge_whc_diag_pscale_multi`. Sanity-checked: instr-owned param boosted to
α_high, coding-owned pinned at α_low; `alpha_low==alpha_high` recovers plain consensus.
Driver `scripts/mb_sweep_routed.{py,sh}`. Domain/expert order (fixes `low_idx`):
`[instruction, math, coding, safety, multilingual]` → **coding = index 2**.

## Round 0 (routed) — RUNNING

lam=1e-3, low_domains=coding, α_low=1, α_high ∈ {3, 3.5, 4, 5}. Merge `21584168` (BigMem,
single-pass) → manifest `mb_merged/Llama-3.1-8B/routed_r0.txt`. Evals (afterok merge):
lm `21584169` (dgx_aic A100), code `21584170` (L40S), arrays 0-3%2.

**Verdict:**
```sh
python scripts/mb_sweep_table.py --config configs/mergebench_tier2.yaml \
  --manifest mb_merged/Llama-3.1-8B/routed_r0.txt
python scripts/mb_forgetting_table.py --config configs/mergebench_tier2.yaml \
  --manifest mb_merged/Llama-3.1-8B/pscale_manifest.txt \
  --manifest mb_merged/Llama-3.1-8B/pscale_r1_all.txt \
  --manifest mb_merged/Llama-3.1-8B/routed_r0.txt
```
**Bar:** gate > 51.8 → dataless win (promote to full eval + seeds). 51.3–51.8 → improved but
still tie; escalate to Bet B (buffer-calibrated, finer routing). < 51.3 → routing doesn't
help; reconsider grouping (per-layer vs per-param) or jump to Bet C.

| variant | α_high | α_low | low | math | instr | heval+ | mbpp+ | GATE |
|---|---|---|---|---|---|---|---|---|
| whc_route_l1e-3_h3_l1_c | 3 | 1 | coding | 79.2 | 27.4 | 39.5 | 50.2 | **49.1** |
| whc_route_l1e-3_h3.5_l1_c | 3.5 | 1 | coding | 77.8 | 27.2 | 36.7 | 46.5 | 47.1 |
| whc_route_l1e-3_h4_l1_c | 4 | 1 | coding | 75.0 | 30.0 | 37.4 | 42.3 | 46.2 |
| whc_route_l1e-3_h5_l1_c | 5 | 1 | coding | 70.0 | 27.6 | 31.5 | 32.4 | 40.4 |
| _baseline Consensus_ | – | – | – | 78.2 | 25.1 | 49.8 | 54.1 | _51.8_ |
| _best pscale (cons am2.5/am2)_ | – | – | – | – | – | – | – | _51.3_ |
| _no-scale am1_ | 1 | 1 | – | 73.2 | 15.3 | 41.5 | 55.7 | _46.4_ |

**FINAL READ — Bet A FAILED (best 49.1, below every dataless point).** Routing hurt BOTH axes:
coding collapsed (heval+ 39.5/mbpp+ 50.2, worse than no-scale 41.5/55.7) AND instruction
under-recovered (27.4 vs plain consensus am3's 32.4). **→ CROSS-TALK confirmed from both sides:**
pinning coding-OWNED params at α=1 (a) doesn't protect coding — boosting the *rest* still
corrupts the activations feeding coding layers — and (b) starves instruction, which also rides
on params the coding expert happens to own. **Per-PARAMETER ownership is the wrong unit** (and
per-layer is too — see entanglement diagnostic below). The capabilities are not separable in
weight space at all.

## Bet B — per-LAYER α scaling — KILLED by the entanglement diagnostic

Cache job `21584214` (`scripts/mb_cache_update.py`, also dumps per-layer task-vector energy).
**Coding energy is FLAT across all 32 layers: frac 0.208–0.250, mean 0.225, stdev 0.013.**
Coding is ~22% of EVERY layer's task-vector energy — not concentrated anywhere. So per-layer
α routing cannot decouple coding (no "coding layers" to hold low); it degenerates to global α.
B1 dead on arrival; B2 (buffer-learned per-layer α) also unpromising — entangled weights mean
the *same numbers* carry both capabilities, so scaling a shared update can't separate them.

**KEY FINDING (paper-worthy).** At N=5 / Llama-3.1-8B the domains are **entangled in parameter
space**: neither per-parameter ownership (Bet A) nor per-layer energy (Bet B) partitions coding
from instruction. This is *why* one global α sits at the crossover and all weight-space routing
ties. → The separable axis is not *which weights* but *which expert*.
(Cache retained at `mb_cache/Llama-3.1-8B_l1e-3` — u + ratio + meta — for any later reuse.)

## Bet D — per-EXPERT coefficient decoupling (the axis the data supports)

Each expert keeps its OWN scaled task vector: `w = w_pre + Σ_i s_i · τ_i`. This decouples by
expert and is immune to weight entanglement (each τ_i is that expert's direction). Diagnosis
says instruction is DILUTED (needs high s) and coding OVER-shoots (needs moderate s). Sweep,
others (math/safety/multilingual) at 0.4 baseline:
- s_instr ∈ {0.6, 0.8, 1.0}, s_coding ∈ {0.3, 0.4}.
Single-pass multi-variant additive merge (`merge_task_arith_perexpert_multi`). Cheap; baseline
TaskArithmetic (global 0.4) already = 51.6, just 0.2 under Consensus 51.8, so a per-expert boost
of the diluted instruction expert is the most direct shot at clearing 51.8.
**Bar:** gate > 51.8 = dataless win. Then optionally buffer-LEARN {s_i} (data-light) and/or fold
into the curvature framework (per-expert importance × per-param curvature).

### Bet D Round 0 — RUNNING
base=0.4 (math/safety/multilingual); instruction ∈ {0.6,0.8,1.0} × coding ∈ {0.3,0.4} = 6
variants. (First attempt `21584453` died on **disk-quota-exceeded** while saving — `mb_merged`
had grown to 768G of evaluated weight dirs. Freed 704G via
`find mb_merged/<base> -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +` — the gate/forgetting
tables read `results/mb_eval/`, NOT the weight dirs, so evaluated merges are disposable; keep the
tiny `*.txt` manifests + `mb_cache/`.) Resubmitted: merge `21584488` (BigMem) → manifest
`mb_merged/Llama-3.1-8B/perexpert_r0.txt`; evals afterok lm `21584489` (dgx_aic), code `21584490`
(L40S), arrays 0-5%2. Verdict:
`python scripts/mb_sweep_table.py --config configs/mergebench_tier2.yaml --manifest
mb_merged/Llama-3.1-8B/perexpert_r0.txt`. (Reference: TaskArithmetic global-0.4 = 51.6,
instr 25.7; we boost the diluted instruction expert and trim coding.)

| variant | s_instr | s_coding | math | instr | heval+ | mbpp+ | GATE | MEANF |
|---|---|---|---|---|---|---|---|---|
| **ta_pe_inst0.8_codi0.4** | 0.8 | 0.4 | 79.2 | 38.0 | 45.5 | 53.9 | **54.1** | **−5.4** |
| ta_pe_inst1_codi0.3 | 1.0 | 0.3 | 77.6 | 41.2 | 42.1 | 52.8 | 53.4 | −4.7 |
| ta_pe_inst1_codi0.4 | 1.0 | 0.4 | 76.6 | 38.4 | 44.6 | 52.4 | 53.0 | −4.2 |
| ta_pe_inst0.8_codi0.3 | 0.8 | 0.3 | 78.4 | 36.8 | 43.3 | 53.1 | 52.9 | −4.1 |
| ta_pe_inst0.6_codi0.4 | 0.6 | 0.4 | 78.2 | 31.4 | 46.2 | 52.8 | 52.1 | −3.4 |
| ta_pe_inst0.6_codi0.3 | 0.6 | 0.3 | 79.4 | 29.8 | 43.8 | 53.4 | 51.6 | −2.8 |
| _baseline Consensus_ | – | – | 78.2 | 25.1 | 49.8 | 54.1 | _51.8_ | _−3.0_ |
| _TaskArithmetic g0.4_ | – | – | 78.9 | 25.7 | 47.9 | 53.9 | _51.6_ | _−2.8_ |

### ✅ WIN — Bet D clears the bar on BOTH axes (dataless)
`ta_pe_inst0.8_codi0.4`: **GATE 54.1 vs Consensus 51.8 (+2.3)** AND **best forgetting of any
method, MEANF −5.4 vs Consensus −3.0**. Mechanism confirmed exactly as diagnosed: boosting the
DILUTED instruction expert (s=0.8) recovers instruction 25.1→38.0 (forgetting +21.8→+9.0) while
trimming the OVER-shooting coding expert (s=0.4) holds coding (heval 45.5/mbpp 53.9, only mildly
below Consensus 49.8/54.1). 5 of 6 variants beat the top baseline. Per-EXPERT decoupling is the
separable axis (vs weight-space routing Bets A/B which tied/failed) — the entanglement finding
predicted this and it held. Optimum is INTERIOR (inst 0.8 > 1.0 on gate; codi 0.4 > 0.3) → Round 1
refines around it.

### Bet D Round 1 — refine the optimum — DONE (champion confirmed robust)
base=0.4; instruction ∈ {0.7,0.8,0.9} × coding ∈ {0.4,0.5}. Merge `21584719` → lm `21584720` /
code `21584721`. The R0 champion **`ta_pe_inst0.8_codi0.4` stays on top (gate 54.1, MEANF −5.3)**;
the whole neighborhood lands 53.1–53.8 / −4.3…−5.0 — a FLAT-TOPPED optimum (no knife-edge, no
overfit). Confirms inst=0.8 / codi=0.4 / others=0.4.

| variant | s_in | s_co | math | instr | heval+ | mbpp+ | GATE | MEANF |
|---|---|---|---|---|---|---|---|---|
| **ta_pe_inst0.8_codi0.4** | 0.8 | 0.4 | 79.2 | 37.8 | 45.5 | 53.9 | **54.1** | **−5.3** |
| ta_pe_inst0.7_codi0.5 | 0.7 | 0.5 | 77.6 | 36.2 | 47.7 | 53.6 | 53.8 | −5.0 |
| ta_pe_inst0.9_codi0.5 | 0.9 | 0.5 | 77.2 | 38.4 | 46.0 | 53.0 | 53.6 | −4.9 |
| ta_pe_inst0.8_codi0.5 | 0.8 | 0.5 | 76.8 | 36.8 | 47.2 | 53.7 | 53.6 | −4.9 |
| ta_pe_inst0.9_codi0.4 | 0.9 | 0.4 | 77.2 | 37.6 | 45.2 | 53.6 | 53.4 | −4.7 |
| ta_pe_inst0.7_codi0.4 | 0.7 | 0.4 | 78.6 | 33.6 | 46.7 | 53.4 | 53.1 | −4.3 |

### ✅ FULL-PROTOCOL CONFIRMATION (limit=None, n_samples=10) — win holds
Champion `ta_pe_inst0.8_codi0.4` vs Consensus at FULL protocol (not the gate limit=500/n=5):
gsm8k 77.5 / ifeval **37.7** / heval+ 44.3 / mbpp+ 53.3 → **gate 53.2** vs Consensus
(78.2/25.1/49.8/54.1 → 51.8). **+1.4 at full protocol** (gate estimate was +2.3; full eval is
slightly harsher, expected). Driver = instruction **+12.6** (37.7 vs 25.1). NOT a gate-limit artifact.

**Multilingual landed (full, 12 tasks):** champion 51.7 vs Consensus 52.5 (−0.8, near-wash — boosting
the instruction expert did NOT collateral-damage the untouched domains). Full-protocol per-domain
(coding = heval+/mbpp+ avg): math 77.5/78.2, instr **37.7/25.1 (+12.6)**, coding 48.8/52.0 (−3.1),
multiling 51.7/52.5 (−0.8) → **4-domain avg 53.9 vs 51.9 = +2.0**. Clean trade: huge instruction gain
for a modest coding give-back; math/multilingual flat. Only SAFETY (5th domain) remains to complete
the table — env build in progress (see below).

### PROMOTION — full protocol on the champion — RUNNING
Champion = `ta_pe_inst0.8_codi0.4`. Baselines (Consensus etc.) ALREADY have full-protocol results
(lm gsm8k_cot/ifeval/multilingual at limit=None; code n_samples=10 — e.g. Consensus heval+ 49.8 /
mbpp+ 54.1). Only the champion needed promoting from gate→full. Added champion as row idx 13 to
`scripts/mb_eval_lm_tier2.sh` + `mb_eval_code_tier2.sh`. Jobs: full lm `21584861` (L40S, ATTN=sdpa,
GROUP=all, LIMIT unset → gsm8k_cot+ifeval+12 multilingual tasks), full code `21584862` (n_samples=10).
Champion's old gate (limit=500) lm jsons archived to `…/_gate_limit500/`.
Verdict: `mb_sweep_table.py` / `mb_forgetting_table.py` (they now read the full jsons).

### SAFETY (5th domain) — STANDING UP THE 4TH HARNESS (user chose "set up now", champion + all baselines)
Never run for ANY tier-2 model before. Three things needed, all in flight:
1. **Env build** (`scripts/mb_setup_safety_env.sh`, job `21584869`, L40S): clone
   `github.com/nouhadziri/safety-eval-fork` (real URL per MergeBench/README.md:61 — runbook's
   `uiuctml` URL was a placeholder, now FIXED), conda env `/work/pnag/envs/safety-eval` py3.10,
   `pip install -e . && -r requirements.txt && vllm==0.4.2`, numpy<2 (RHEL7). vLLM build is the risk
   (deviates from standing torch pins; existing envs prove torch+cu121 wheels work on this box).
   Network: compute nodes have it (my login/sandbox shell does NOT — that's why setup is a batch job).
2. **Baseline weights re-merge** (job `21584867`, BigMem): vLLM loads from disk, but I'd cleaned
   `mb_merged` — re-merging whc_diag,task_arith,TaskArithmetic,TIES,DARE,Consensus,LocalizeAndStitch
   + fix tokenizers. base + 5 experts persist in `mb_ckpts`. Champion already on disk.
3. **Safety eval** (`scripts/mb_eval_safety_tier2.sh`, champion added as idx 13 → 14 rows): submit
   `sbatch --array=0-13%4 scripts/mb_eval_safety_tier2.sh` ONLY after (1) reports SETUP_OK and (2)
   the re-merge drains. Tasks wildguardtest,harmbench,xstest,do_anything_now; metric RTA/acc; writes
   `results/mb_eval/<tag>/safety_eval.json`. WildGuard classifier downloads on first run (gated→HF_TOKEN).

Win is ALREADY secured on the 4-domain gate + forgetting; safety completes the full 5-domain table.

### SAFETY — STATUS AS OF 2026-06-17 (PAUSED by user; nothing running on the safety axis)
All prerequisites cleared; the eval is NOT launched (user paused to run another project).
1. **Env build** — DONE. `/work/pnag/envs/safety-eval` works end-to-end (smoke job 21584924
   loaded the champion in vLLM 0.6.3.post1 and generated all 450 xstest prompts). Recipe saved
   to memory [[hpc_safety_eval_env_rhel7]]. Key pins: torch 2.4.0+cu121, vllm 0.6.3.post1,
   transformers 4.46.3; OPENAI_API_KEY="EMPTY" (non-empty placeholder); conda-forge for
   sentencepiece/libcst/tiktoken.
2. **WildGuard gating** — ✅ ACCEPTED & VERIFIED 2026-06-17. The `ProtikNag` token (hf_token.txt)
   now gets HTTP 200 on `allenai/wildguard/resolve/main/config.json` (was 403). The classifier all
   4 tasks use is reachable.
3. **Baseline weights re-merge** — IN FLIGHT (started ~2026-06-17 morning; should finish on their
   own, slow due to GPFS mmap I/O [[hpc_gpfs_mmap_merge_io]]):
   - LIGHT job **21584950** (node493, L40S): task_arith done, whc_diag done, TaskArithmetic (in
     prog), Consensus (pending).
   - HEAVY job **21584951** (node463, BigMem): TIES done, DARE (in prog), LocalizeAndStitch (pending).
   On disk now: ta_pe_inst0.8_codi0.4 (champion, ready), task_arith, whc_diag, TIES. Plus all the
   Round-0/1 sweep variants (ta_pe_inst0.7-0.9_codi0.4-0.5).

**UPDATE 2026-06-17 (resume session).** Re-merge state: HEAVY 21584951 COMPLETED (TIES/DARE/L&S on
disk); LIGHT 21584950 was OOM-KILLED (132G>120G cap I forced on L40S) at TaskArithmetic, so
TaskArithmetic + Consensus were MISSING. Re-ran them on the default BigMem 600G path as job
**21586553** (RUNNING). Checked node availability first: L40S node493 idle (2 GPUs), node494 drain;
BigMem mix nodes free; dgx-1 8 GPUs but in use by others.

Launched safety eval **21586554** (`--array=0,1,2,4,5,7,8,9,10,11,12,13%2` on the 2 idle L40S GPUs)
+ dependent **21586555** (`--array=3,6` afterok 21586553). **BOTH HIT A SECOND GATE → all FAILED in
~1-2 min:** `allenai/wildguardmix` (the DATASET supplying `wildguardtest` prompts) is gated
*separately* from the classifier model `allenai/wildguard`. Accepting the model did NOT cover the
dataset. Cancelled 21586555. The other 3 tasks (harmbench/xstest/do_anything_now) read bundled files
— no gate. See [[hpc_safety_eval_env_rhel7]].
- wildguard (model) classifier: token → HTTP 200 ✅
- wildguardmix (dataset): token → HTTP 403 ❌ → **USER must accept
  https://huggingface.co/datasets/allenai/wildguardmix** (logged in as ProtikNag).

**TO RESUME SAFETY (after wildguardmix accepted + merge 21586553 done):**
`sbatch --array=0-13%4 scripts/mb_eval_safety_tier2.sh`  (idx 13 = champion; idx 0-12 = base/7
merges/5 experts). Writes `results/mb_eval/<tag>/safety_eval.json`. `--limit` is IGNORED by the
fork's eval.py (full-size runs); --time=08:00:00. Then `scripts/mb_make_tier2_table.py`.

### NEXT METHOD WORK — DERIVE the per-expert coeffs from a curvature/Taylor surrogate (form ②)
The champion's s=(0.8,0.4,0.4,0.4,0.4) is hand-swept on the gate (only instruction+coding tuned;
others = 0.4 baseline). To remove the "benchmark-tuned HP" critique, derive all N coeffs from a
Gauss-Newton (empirical-Fisher) quadratic surrogate of the pooled multitask loss in the N-dim
task-vector subspace: per-example directional deriv `d_{n,i}=<g_n, tau_i>` → `grad_s,i=mean d_{n,i}`,
`A_{ij}=mean d_{n,i} d_{n,j}`, Newton step `delta=-(A+eps I)^{-1} grad_s` (ridge + trust-region clip),
re-linearise K times. Off-diagonal A_{ij} models the instruction↔coding cross-talk the scalar grid
can't. Data-LIGHT (tiny MergeBench/<domain>_val buffer, not fully dataless). This is the per-expert
collapse of WHC's per-param Hessian weighting; form ① (per-expert scalar × per-param curvature) is the
later "full" variant. Code: `scripts/mb_fit_perexpert_surrogate.py` + wrapper
`scripts/mb_fit_surrogate_tier2.sh` (merging env, eager attn, L40S 48G + ~200G RAM for the 5 CPU-cached
tau). Run AFTER safety table lands: `INIT=0.8,0.4,0.4,0.4,0.4 STEPS=3 sbatch scripts/mb_fit_surrogate_tier2.sh`
(or single-pass from 0.4), then eval the written `ta_pe_surrogate` tag and compare s* to the champion.
CAVEAT: instruction/safety may lack a `_val` set (same as Fisher) → use the `DATASETS=` override.
RISK: dataless per-param curvature already LOST (whc_diag/whc_gram ≤51) — keep s_i a free per-EXPERT
scalar, normalise any per-param curvature WITHIN each expert, never re-blend globally.

### → CURVATURE CAMPAIGN MOVED to its own ledger: [`EXPERIMENTS_curvature.md`](EXPERIMENTS_curvature.md)
The full Gauss-Newton/Taylor-surrogate campaign (Phase 1 per-expert scalar, Phase 2 per-block) lives
there. Headline so far: **scalar space exhausted (interference wall), per-block placement breaks
depth-sensitive generative (gsm8k 0.594 @ math mean 0.19); frozen-generative per-block `ta_pl_b8_frzgen`
eval in flight.** The champion `ta_pe_inst0.8_codi0.4` (avg 55.43) still stands.
