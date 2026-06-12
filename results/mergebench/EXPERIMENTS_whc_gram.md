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
| whc_gram_l0.01_a1_m | 1e-2 | 1 | mean | | | | | | — |
| whc_gram_l0.01_a2_m | 1e-2 | 2 | mean | | | | | | — |
| whc_gram_l0.01_a3_m | 1e-2 | 3 | mean | | | | | | — |
| whc_gram_l0.01_a1_ta | 1e-2 | 1 | task_arith | | | | | | — |
| whc_gram_l0.01_a2_ta | 1e-2 | 2 | task_arith | | | | | | — |
| whc_gram_l0.01_a3_ta | 1e-2 | 3 | task_arith | | | | | | — |

**Decision criteria.** If any variant's instr recovers toward ~25 and GATE clears
~51 -> dilution hypothesis confirmed, proceed to Round 2 (iterative K>=1) and lam
re-tune at the winning (alpha,fallback). If alpha helps instr but kills coding
(the whc_diag trade-off reappears), that is the structural ceiling and the data
tier likely caps at a tie -> consider folding. Record the trade-off either way.

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
