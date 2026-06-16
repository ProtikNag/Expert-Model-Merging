# Per-parameter update-scaling (pscale) experiment ledger

The dataless win attempt. Base Llama-3.1-8B, N=5 experts, MergeBench. Every
promising result here MUST be reconfirmed at full scale (tier T2) before it goes
in a table or the paper. Decisions are made at **T1** (LIMIT=500 lm, n_samples=5
code). Baseline bar: **Consensus 51.8** (top dataless), tuned global-alpha HTCL
**51.3** (the tie we must beat).

## Central hypothesis

The global-alpha sweep (`EXPERIMENTS` in NOTES §11 / `whc_tv_*` variants) capped
HTCL at 51.3 because a **single global `alpha` cannot serve every domain**:
instruction wants a large scale (instr 14→31 as alpha 1→3), coding wants a small
one (mbpp+ 55→48). A global scalar is a wash.

The fix is to make the scale **per parameter**, derived dataless-ly from the
experts' own task vectors `tau_i = w_i - w_pre`. Two mechanisms:

1. **consensus** (primary bet): `a_p = clip(|c_p| / (|u_p| + eps), 1, alpha_max)`,
   where `u_p` is the whc_diag update and `c_p = sum_{i: sign(tau_i)==sign(u_p)} tau_i`
   is the additive task-vector sum of the experts that agree with the merged
   update's direction. This rescales the curvature-weighted **mean** up to the
   additive **sum** *only where experts agree*, directly undoing the ~1/N
   dilution where it actually occurs.
2. **coherence** (the AAAI-plan form, for comparison):
   `a_p = 1 + (alpha_max - 1) * coherence_p^beta`,
   `coherence_p = |sum_i tau_i| / (sum_i |tau_i| + eps)`.

**Why consensus should beat coherence.** Both boost params where experts agree.
But on a **single-expert-dominated** param (only the coding expert moved it),
coherence = 1, so coherence boosts it to `alpha_max` — pushing the weight *beyond
the expert itself* (over-extrapolation). That is precisely the mechanism that
degrades coding under global alpha. Consensus, on a single-expert param, has
`c_p = tau_k` and `u_p ≈ tau_k`, so `a_p ≈ 1`: no over-boost. Consensus boosts
only genuine **multi-expert agreement** (where summing is the right thing) and
leaves single-expert and conflicted params alone. It is the principled
dilution-undo and unifies the §11 dilution analysis with TIES-style sign
consensus, from one curvature objective.

Dataless, closed-form, O(params), one merge. Composes with the curvature
weighting (`whc_diag`). Implemented in `mergebench/llm_merge.py` (`pscale`,
`alpha_max`, `beta`); unit-tested limits in `tests/test_pscale.py`; swept by
`scripts/mb_sweep_pscale.py` / `.sh`.

## Proxy protocol

Same as the whc_gram campaign. T1 gate = mean(gsm8k_cot, ifeval, humaneval+,
mbpp+). Kill gate (AAAI plan Phase 1): best T1 gate **< 51.0 → no dataless win,
fold to analysis**.

Commands (T1):
```sh
MAN=mb_merged/Llama-3.1-8B/pscale_manifest.txt
sbatch scripts/mb_sweep_pscale.sh                                   # merge (BigMem)
MANIFEST=$MAN sbatch -p gpu-v100-32gb  --array=0-4%5 scripts/mb_eval_sweep_lm.sh
MANIFEST=$MAN sbatch -p AI_Center_L40S --array=0-4%5 scripts/mb_eval_sweep_code.sh
python scripts/mb_sweep_table.py --config configs/mergebench_tier2.yaml --manifest $MAN
```

## Results

Gate columns: math gsm8k_cot / instr ifeval / heval+ / mbpp+ (%). T1.

### Round 0 — consensus vs coherence at lam=1e-3 (RUNNING)

Grid: consensus alpha_max ∈ {3,5,8}; coherence alpha_max=5, beta ∈ {1,2}.
Jobs: merge 21583069 (BigMem) → eval 21583071 (lm, v100) + 21583072 (code, L40S).

| variant | mode | alpha_max | beta | math | instr | heval+ | mbpp+ | GATE | tier |
|---|---|---|---|---|---|---|---|---|---|
| whc_cons_l1e-3_am3 | consensus | 3 | – | | | | | | T1 |
| whc_cons_l1e-3_am5 | consensus | 5 | – | | | | | | T1 |
| whc_cons_l1e-3_am8 | consensus | 8 | – | | | | | | T1 |
| whc_coh_l1e-3_am5_b1 | coherence | 5 | 1 | | | | | | T1 |
| whc_coh_l1e-3_am5_b2 | coherence | 5 | 2 | | | | | | T1 |
| _global-alpha best (whc_tv_l1e-3_a2)_ | global | 2 | – | | | | | _51.3_ | T1 |
| _baseline Consensus_ | – | – | – | 78.2 | 25.1 | 49.8 | 54.1 | _51.8_ | T1 |

**Read.** (pending)
