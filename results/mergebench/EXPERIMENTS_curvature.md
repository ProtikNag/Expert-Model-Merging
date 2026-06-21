# Curvature/Taylor-surrogate campaign — make a genuine curvature-aware merge WIN

Started 2026-06-18, branch `tier2-llama-scaffold`. Follows the per-EXPERT win
(`EXPERIMENTS_routed.md` Bet D). **Goal:** make a *genuine* curvature-aware /
Gauss-Newton / Taylor-series surrogate (NOT relabeled task arithmetic) NUMERICALLY
beat the hand-tuned champion. User directive: *"find a way to make the curvature-aware
merge better … figure out the problem that is hindering it and find a way to make it
better … I want a win."* Get a win on a fast (moderate-size) gate first, then scale.

## The bar — champion `ta_pe_inst0.8_codi0.4` (Bet D)

Per-EXPERT scaled task arithmetic `w = w_pre + Σ_i s_i·τ_i`, hand-swept coeffs
`s = [inst 0.8, math 0.4, coding 0.4, safety 0.4, multilingual 0.4]`. Full protocol
(limit=None, n_samples=10):

| domain | metric | champion |
|---|---|---|
| instruction | ifeval prompt_level_strict_acc | 37.8 |
| math | gsm8k_cot exact_match (flexible) | 79.2 (77.5 @limit=500) |
| coding | avg(humanevalplus, mbppplus pass@1) | 48.8 |
| safety | same-formula avg(1−wg, 1−hb, xstest, 1−dan) | 59.7 official / **57.61** same-formula |
| multilingual | avg of 12 (m_mmlu acc + arc/hellaswag acc_norm) | 51.7 |
| **avg** | | **55.43** |

NO curvature is used by the champion — it is the hand-tuned scalar optimum. The task is
to derive coefficients from a curvature surrogate that **beats** this number.

## The surrogate (code: `scripts/mb_fit_perexpert_surrogate.py`, env `merging`)

Quadratic Gauss-Newton model of the pooled multitask loss in the N-dim task-vector
subspace. Per validation example `n`, expert `i`:
- directional derivative `d_{n,i} = <g_n, τ_i>` (g_n = per-example loss gradient)
- gradient `grad_s,i = mean_n d_{n,i}`
- **empirical-Fisher curvature** `A_{ij} = mean_n d_{n,i} d_{n,j}`
- damped Newton step `δ = −(A + ridge·I)⁻¹ grad_s`, trust-region clipped, re-linearized K steps.

**Attribution modes** (which curvature the solve sees):
- `own_domain` — block-diagonal A (each example attributed only to its own domain's
  coordinate). Blind to cross-domain interference → pushes every `s_i → 1`.
- `pooled` — full off-diagonal `A_{ij}`. SEES the instruction↔coding↔safety cross-talk.

**`teacher_mix` objective** (the behavioral-mimicry fix): per-domain
`loss_i = (1−λ_i)·argmax-CE + λ_i·SOFT`, default `λ = [0,0,0,1,1]` — HARD argmax-CE for
the generative domains (inst/math/coding, whose metrics reward argmax alignment), SOFT
distribution-matching for safety+multilingual (whose RTA/acc_norm metrics reward matching
the expert's *distribution*). Soft-discrepancy menu `--soft-discrepancy {kl, js, logit_mse,
feature_mse}`. Flags: `--freeze <domains>` (hold named coords at init), `--objective`,
`--kl-mix`. Wrapper `scripts/mb_fit_surrogate_tier2.sh` (`OBJ/SOFTDISC/KLMIX/STEPCLIP/FREEZE`).

---

## Phase 1 — per-EXPERT scalar curvature (5 DOF)

### Candidate C (hard-label distillation) — LOSES by 2.85
`s* = [inst 0.8, math 0.4, coding 0.49, safety 0.64, multilingual 0.74]`. Full 5-domain:

| | inst | math | coding | safety | ml | avg |
|---|---|---|---|---|---|---|
| C | 38.2 (+0.4) | 76.0 (−3.2) | 47.5 (−1.3) | **50.9 (−8.8)** | 50.3 (−1.4) | **52.58** |
| champion | 37.8 | 79.2 | 48.8 | 59.7 | 51.7 | 55.43 |

Loses on 4/5 domains. **Safety COLLAPSED**: raising the safety coeff 0.4→0.64 made the
model refuse LESS (wildguard harm 37.7→31, harmbench ASR 51.2→35.6). Root cause: hard-label
CE is ~flat in over-merging directions, so the solve over-scales coords whose benchmark metric
the proxy can't see.

### own_domain + KL soft — OSCILLATED (unusable)
mean_loss ROSE 1.33→1.75; every coord slammed the clip; `s*` over-merged `[0.7,1.3,0.7,0.7,1.06]`.
own_domain is blind to interference and drives every coord toward its own expert (s→1).

### pooled + KL soft, COLD start — `ta_pe_mix_pool_kl` — LOSES (generative collapse)
PIVOT to `pooled` attribution + tighter step (clip 0.15, ridge 0.05, 4 steps) CONVERGED
(mean_loss 1.334→1.299) and derived a conservative, safety-preserving
`s* = [inst 0.60, math 0.10, coding 0.30, safety 0.40, ml 0.73]`. The KL term recovered the
champion's safety 0.40 DATALESS — the exact value C got wrong. **But the full eval loses:**

| | inst | math@500 | coding | safety | ml | ~avg |
|---|---|---|---|---|---|---|
| ta_pe_mix_pool_kl | 23.8 (−14.0) | 67.0 (−10.5) | 47.7 | **70.8** | 54.3 (+2.6) | ~53.0 |

The pooled solve UNDER-merges the two generative domains the champion leans into (it drives
math→0.10 to cut interference, even though gsm8k NEEDS the math expert). safety same-formula
70.83 and ml +2.6 are REAL gains the surrogate found dataless — but they come from LOW *other*
coefficients (less interference), not from the safety/ml coeffs themselves (proven below).

### Metric-faithfulness — the central diagnosis
The distillation surrogate is **faithful ONLY to distribution-matching metrics** (ml acc_norm,
safety RTA) and **ANTI-faithful to argmax-generative metrics** (ifeval / gsm8k / pass@1). Token
argmax-CE is ~flat in the math direction (the merged model still argmax-matches the teacher on
easy tokens at low coeff), so lowering the surrogate loss does NOT track the generative benchmark.

### warm + freeze generative — `ta_pe_mawc` — dead (oscillated, not eval'd)
Warm-start at champion, FREEZE inst/math/coding, let pooled+KL refine ONLY ml+safety. Predicted
the win `[0.8,0.4,0.4,0.4,~0.7] → ~55.96`. Instead ml/safety OSCILLATED to ~0.37 (below champion)
and never converged — pooled walks the free coords back toward the bad interference equilibrium.

### clean ml-boost control — `ta_pe_inst0.8_mult0.73` — LOSES by ~1.1
Champion with ONLY ml 0.4→0.73 (isolate the cold candidate's ml gain):
inst 35.1 / math 76.2@500 / coding 48.0 / safety 56.9 / **ml 51.85** → 53.96. **The ml boost gained
only +0.15 ml** (not the cold candidate's +2.6) and BLED every other domain. This PROVES the cold
candidate's ml/safety gains came from its LOW other coefficients, not from ml=0.73.

### ✅ Verdict: SCALAR SPACE IS EXHAUSTED — the interference wall
Three independent candidates all lose, one clean mechanism. **Two domain families pull OPPOSITE
on every shared scalar coefficient:**
- **instruction / math / coding** (generative) want HIGH own-coefficient — they need their expert present.
- **safety / multilingual** (discriminative / refusal) want LOW interference from ALL other experts —
  merging degrades RTA/acc_norm.

The champion `[0.8,0.4,0.4,0.4,0.4]` is the **Pareto-optimal point** in 5-scalar space. No scalar
point beats it; curvature cannot find a better scalar because none exists.

---

## Phase 2 — per-LAYER / per-block curvature DOF (40 DOF)

The forced next step: give each (expert, block) its own coefficient so the merge can be aggressive
where a domain helps and recessive where it interferes — breaking the scalar wall. Code:
`scripts/mb_fit_perlayer_surrogate.py` + `scripts/mb_fit_perlayer_tier2.sh`. Coefficient matrix
`S` shape `(n_experts, n_blocks)`; Llama 32 layers → 8 blocks (keys/block [37,36,36,36,36,36,36,38]).
Saves the merged model directly from the in-memory GPU model (no external scalar merge).

### `ta_pl_b8` (cold per-block, all 40 DOF free) — NEGATIVE RESULT
Converged (mean_loss 1.358→1.289). Derived `S*` (per-block means):
inst 0.634 / **math 0.188** / coding 0.218 / safety 0.263 / ml 0.504, with strong per-block
variation (e.g. ml block7=0.801, math block5=0.346 vs block0=0.075).

**Math-gate kill-check: gsm8k flexible @500 = 0.594** — WORSE than the scalar collapse (0.670 at
math mean 0.10). **Per-block placement does NOT preserve a depth-sensitive generative domain at low
mean coefficient.** Math reasoning is deep *sequential* computation; concentrating the math expert in
some blocks and near-absent in others breaks the reasoning chain MORE than uniform weakening. → per-block
freedom helps only the DISCRIMINATIVE domains; depth-sensitive generative must be held at champion.

### `ta_pl_b8_frzgen` (frozen-generative per-block, 16 DOF) — IN FLIGHT
Sharpened recipe: FREEZE inst/math/coding at champion `[0.8,0.4,0.4]`, per-block pooled solve ONLY
safety + multilingual. Converged cleanly (mean_loss 1.358→1.335, monotone). Derived `S*`:

| expert | mean | per-block S* |
|---|---|---|
| instruction | 0.800 (frozen) | flat 0.8 |
| math | 0.400 (frozen) | flat 0.4 |
| coding | 0.400 (frozen) | flat 0.4 |
| safety | **0.211** (↓ from 0.4) | early-weighted [0.32,0.13,0.21,0.25,0.17,0.24,0.23,0.16] |
| multilingual | **0.459** (↑ from 0.4) | late-weighted [0.44,0.37,0.28,0.29,0.45,0.50,0.57,**0.79**] |

The interference wall again: the pooled solve pushed safety DOWN and ml UP-and-late. Since generative
is pinned at champion, this candidate can only GAIN via safety/ml and only LOSE via generative drift
(the safety/ml deltas still perturb shared weights). **Eval running on node493** (idx18 in all three
eval scripts): safety `21591978`, multilingual `21591979`, math@500 `21591980`, ifeval `21591981`,
coding `21591982`. Decisive gates = safety + ml. Predicted outcome (given the metric-faithfulness
diagnosis): the surrogate lowering safety to 0.211 likely REDUCES the safety expert's refusal
contribution → safety RTA falls; a clear win is unlikely, but the data decides.

## Standing conclusion (pending the frzgen gates)

Coefficient placement — scalar (Phase 1) OR per-block (Phase 2) — has not beaten the hand-tuned
champion, with a single mechanistic explanation: the surrogate's distillation loss is faithful to
distribution-matching metrics but anti-faithful to argmax-generative ones, and the two domain families
impose an interference wall no shared coefficient can cross. If `ta_pl_b8_frzgen` also only ties, the
honest contribution becomes **the diagnosis itself** (the interference wall + the metric-faithfulness
asymmetry + the per-block depth-sensitivity negative result) rather than a curvature SOTA claim.
