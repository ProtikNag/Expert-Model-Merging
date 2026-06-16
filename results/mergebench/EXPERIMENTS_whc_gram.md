# whc_gram (HTCL-data) experiment ledger

Campaign to make the data-using merge beat the dataless tie. Base Llama-3.1-8B,
N=5 experts, MergeBench. Every promising result here MUST be reconfirmed at full
scale (tier T2) before it goes in a table or the paper.

## Central hypothesis

`whc_gram` is RegMean + a ridge toward the mean. RegMean's solve
`(sum_i W_i G_i)(sum_i G_i)^-1` is a *weighted average* of the experts, so the
method inherits the **same ~1/N update dilution** we diagnosed for `whc_diag`
(NOTES Sec 11): at N=5 each expert's update is averaged down rather than summed,
which is why math/instruction collapse vs the task-arithmetic baselines. The
dilution hits two key-groups:

1. **Gram-solved Linears** (q/k/v/o/gate/up_proj) — diluted by the averaging solve.
2. **Non-Gram keys** (down_proj, embeddings, lm_head, norms; ~99 keys) — diluted
   by the literal ensemble mean.

Predicted fixes (the same medicine that lifted `whc_diag` instr 15->31):
- `alpha` (global update-scale, ~N) attacks group 1+2 uniformly.
- `gram_fallback=task_arith` attacks group 2 (sum, not mean).
- iterative catch-up (K>=1) refines the Gram linearization point (the GLUE edge).

If alpha+fallback recover math/instruction without losing the coding strength
(mbpp+ is already table-best), the data tier clears the bar and the dilution
analysis becomes a single unified thread across both HTCL tiers.

## Proxy protocol (cheap -> full)

| Tier | Eval | Cost/variant | Use |
|---|---|---|---|
| **T0 micro** | `LIMIT=150` lm + `N_SAMPLES=1` code | ~20-30 min | triage many variants to a shortlist; coarse ranking only |
| **T1 gate** | `LIMIT=500` lm + `N_SAMPLES=5` code | ~1.5 h | the trusted decision proxy (it cleanly separates the baselines: Consensus 51.8 vs TIES 48.6) |
| **T2 full** | `LIMIT=0` lm + `N_SAMPLES=10` code + multilingual + safety | ~half day | final confirmation of the chosen model ONLY |

**Rules.** Decisions are made at **T1**. T0 only avoids wasting T1 compute on
obvious losers. **No result is reported until reconfirmed at T2.** Gate =
mean(math gsm8k_cot, instr ifeval, humaneval+, mbpp+). Baseline bar to beat:
**Consensus 51.8** (dataless), and the tuned dataless HTCL **51.3**.

Commands (T1; T0 = prepend `LIMIT=150 N_SAMPLES=1`):
```sh
MAN=mb_merged/Llama-3.1-8B/whc_gram_manifest.txt
N=$(wc -l < $MAN)
MANIFEST=$MAN sbatch -p gpu-v100-32gb   --array=0-$((N-1))%6 scripts/mb_eval_sweep_lm.sh
MANIFEST=$MAN sbatch -p AI_Center_L40S  --array=0-$((N-1))%6 scripts/mb_eval_sweep_code.sh
python scripts/mb_sweep_table.py --config configs/mergebench_tier2.yaml --manifest $MAN
```

## Results

Gate columns are math / instr / heval+ / mbpp+ (%). Tier in the last column.

### Round 0 — single-pass lam sweep (DONE, T1)

| variant | lam | alpha | fallback | math | instr | heval+ | mbpp+ | GATE | tier |
|---|---|---|---|---|---|---|---|---|---|
| whc_gram_l0_g0 | 0 | 1 | mean | 0.0 | 1.4 | 0.0 | 0.0 | **0.4** | T1 |
| whc_gram_l0.001 | 1e-3 | 1 | mean | 75.2 | 19.2 | 43.7 | 55.5 | 48.4 | T1 |
| whc_gram_l0.01 | 1e-2 | 1 | mean | 75.0 | 19.0 | 44.6 | 56.0 | **48.7** | T1 |
| whc_gram_l0.1 | 1e-1 | 1 | mean | 76.0 | 17.6 | 44.1 | 56.6 | 48.6 | T1 |
| _baseline Consensus_ | | | | 78.2 | 25.1 | 49.8 | 54.1 | _51.8_ | T1 |
| _dataless HTCL (best)_ | | | | | | | | _51.3_ | T1 |

**Read.** Single-pass loses (best 48.7 < 51.8). lam=0 (plain RegMean) is
numerically degenerate at 8B (ill-conditioned Gram solve) -> ablation-only. lam
barely matters (48.4-48.7); the ridge mostly just stabilizes. Wins **mbpp+ only**
(best in table); loses math -3, instr -6, heval+ -5. Loss is on the diluted
metrics, consistent with the hypothesis.

### Round 1 — dilution fixes: alpha x fallback (PENDING)

Fix lam=1e-2 (round-0 best). Sweep alpha in {1,2,3}, fallback in {mean,task_arith}.
`a1_m` reproduces the round-0 control (48.7).

Merge (BigMem):
```sh
LAMS=1e-2 ALPHAS=1,2,3 FALLBACKS=mean,task_arith \
  sbatch scripts/mb_merge_whc_gram.sh
```
Then T1 gate per the protocol above (6 variants -> `--array=0-5%6`).

| variant | lam | alpha | fallback | math | instr | heval+ | mbpp+ | GATE | tier |
|---|---|---|---|---|---|---|---|---|---|
| whc_gram_l0.01_a1_m (control) | 1e-2 | 1 | mean | 75.0 | 19.0 | 44.6 | 56.0 | 48.7 | T1 |
| **whc_gram_l0.01_a1_ta** | 1e-2 | 1 | task_arith | 76.2 | 23.0 | 43.2 | 54.0 | **49.1** | T1 |
| whc_gram_l0.01_a2_m | 1e-2 | 2 | mean | 40.4 | 33.4 | 39.9 | 48.0 | 40.4 | T1 |
| whc_gram_l0.01_a2_ta | 1e-2 | 2 | task_arith | 48.2 | 29.0 | 37.1 | 40.8 | 38.8 | T1 |
| whc_gram_l0.01_a3_m | 1e-2 | 3 | mean | 8.8 | 25.0 | 11.5 | 16.3 | 15.4 | T1 |
| whc_gram_l0.01_a3_ta | 1e-2 | 3 | task_arith | abandoned (alpha=3 catastrophic) | | | | | — |

**Verdict.** Control reproduces round 0 (48.7) -> code path sound.
- **task_arith fallback: confirmed, mild.** a1_ta lifts instr 19->23 and math
  75->76 (the diluted non-Gram keys), +0.4 gate. Keep it.
- **alpha: REFUTED, catastrophic.** alpha=2 craters math 75->40; alpha=3 collapses
  everything (math 8.8). Opposite of whc_diag. Reason: whc_diag is a per-coordinate
  MEAN (bounded inside the experts' values) so scaling its deviation extrapolates
  gently; the Gram least-squares solve ALREADY extrapolates beyond the experts (not
  a convex combination), so alpha amplifies that into out-of-distribution garbage.
  The diagonal-method dilution fix does NOT transfer to the full-covariance method.
  Drop alpha from the data-tier search (keep alpha=1).
- Best data config so far: **a1_ta = 49.1**, still 2.7 below Consensus 51.8. The
  residual gap is spread across math/instr/heval (heval -4.7 the largest), i.e. the
  Gram (averaging) solve is structurally weaker than additive task-arith on these
  tasks. The remaining real lever is improving the SOLVE quality, not rescaling it.

### Round 2 — iterative catch-up K>=1 (PENDING)

Re-linearize the Gram at the merged weights (the GLUE differentiator; it improves
the solve without rescaling, so it sidesteps the alpha failure). Base config =
Round-1 winner `whc_gram_l0.01_a1_ta` (lam=1e-2, task_arith fallback, alpha=1).

K=1 procedure (reuses mb_gram_estimate.py with --expert pointed at the merged
model; mb_gram_tier2.sh now takes LINEARIZE_AT + OUT_ROOT):
```sh
# 1. re-estimate each domain's Gram ON the round-1 winner
LINEARIZE_AT=mb_merged/Llama-3.1-8B/whc_gram_l0.01_a1_ta \
  OUT_ROOT=mb_grams/Llama-3.1-8B_k1 sbatch scripts/mb_gram_tier2.sh
# 2. re-merge the ORIGINAL experts with the refreshed Grams
GRAM_ROOT=mb_grams/Llama-3.1-8B_k1 LAMS=1e-2 ALPHAS=1 FALLBACKS=task_arith \
  MANIFEST=mb_merged/Llama-3.1-8B/whc_gram_k1_manifest.txt \
  sbatch scripts/mb_merge_whc_gram.sh
# 3. T1 gate (1 variant); repeat K=2 from the K=1 model if it improves
```

| variant | round | math | instr | heval+ | mbpp+ | GATE | tier |
|---|---|---|---|---|---|---|---|
| whc_gram_l0.01_a1_ta | K=0 | 76.2 | 23.0 | 43.2 | 54.0 | 49.1 | T1 |
| whc_gram_k1 | K=1 | 75.2 | 21.8 | 43.4 | 53.6 | **48.5** | T1 |

**Verdict: FLAT (slightly worse). Iterative catch-up does NOT transfer.** K=1 is
48.5 vs K=0 49.1 (instr 23->21.8, math 76->75, mbpp 54->53.6, heval flat) -- a -0.6
move, no recovery. Re-linearising the Gram at the merged point, the exact lever
that beat RegMean on GLUE (whc_tree_iter 0.667 > 0.609), buys nothing at 8B / N=5.
This is itself a reportable result: the GLUE-scale iterative data merge does not
scale to billion-parameter LLMs. Per the pre-registered rule, do NOT run K=2.

## Campaign verdict (data tier)

Three rounds, two main levers tested and both failed:
- **Round 0** single-pass: 48.7, loses (lam=0 degenerate; ridge needed).
- **Round 1** alpha: REFUTED (catastrophic, extrapolation); task_arith fallback:
  mild +0.4 -> 49.1 (the data-tier best).
- **Round 2** iterative K=1: FLAT (48.5), doesn't transfer.

**The data tier (whc_gram) caps at ~49**, below the dataless tie (51.3) and the
dataless baselines (51.8). The residual gap is structural -- the Gram (averaging)
solve is weaker than additive task arithmetic on math/instr/heval, and neither
rescaling (alpha) nor re-linearising (K) closes it. Spending data to measure each
expert's activation geometry does not beat the dataless method at this scale.

**Remaining backlog levers are low-upside** (best case a tie, not a win):
all-Linear Grams (down_proj coverage; targets the heval -6 deficit, ~+1-1.5 gate
at most), RegMean off-diagonal reduction, per-expert Gram normalisation. None
changes the average-vs-sum structure that costs math/instr.

**Recommendation: fold to the analysis framing.** The contribution is the unified
curvature-anchored merging objective (dataless whc_diag <-> data whc_gram), the
N-scaling dilution analysis + the alpha fix (diagonal), and three clean structural
findings: (1) plain RegMean is numerically degenerate at 8B, the ridge-toward-mean
makes Gram-merging usable; (2) the alpha update-scale that rescues diagonal merging
is catastrophic for full-covariance merging (the solve extrapolates); (3) the
GLUE-winning iterative catch-up does not transfer to LLM scale. HTCL ties the
dataless tier; the data tier is an honest negative. Target TMLR / workshop.

## Backlog (ideas to integrate if Round 1 is promising)

- **K>=1 iterative catch-up** — re-estimate Grams on the merged model, re-merge
  the original experts; repeat until gate stops rising. The GLUE differentiator.
- **All-Linear Grams** — include down_proj (needs a high-RAM Gram estimate);
  removes that block from the fallback group entirely.
- **Per-domain / per-key adaptive alpha** — instruction wants high alpha, coding
  wants low; a single global alpha cannot serve both (the open whc_diag lead).
- **Condition-aware ridge** — scale lam per layer by the Gram's spectrum instead
  of a flat lam (the lam=0 degeneracy says conditioning varies by layer).
- **Hybrid by layer-type** — Gram-solve attention, task_arith the MLP, etc.
