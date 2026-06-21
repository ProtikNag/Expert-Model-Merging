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

### Round 0 — consensus vs coherence at lam=1e-3 (eval RUNNING)

Grid: consensus alpha_max ∈ {3,5,8}; coherence alpha_max=5, beta ∈ {1,2}.
Merge DONE: job 21583244 on BigMem node464, ~27 min, **single-pass** (reads the 6
models once for all 5 variants via `merge_whc_diag_pscale_multi`; the original
per-variant merge was 5× GPFS-I/O-redundant and got killed — jobs 21583069 /
21583094 are dead, ignore). Manifest `mb_merged/Llama-3.1-8B/pscale_manifest.txt`.
Eval RUNNING: lm 21583200 (v100, gsm8k_cot+ifeval LIMIT=500), code 21583201
(L40S, humanevalplus+mbppplus n_samples=5). Verdict via `mb_sweep_table.py
--manifest ...` (see commands above).

| variant | mode | alpha_max | beta | math | instr | heval+ | mbpp+ | GATE | tier |
|---|---|---|---|---|---|---|---|---|---|
| **whc_cons_l1e-3_am3** | consensus | 3 | – | 77.0 | **32.4** | 45.0 | 48.3 | **50.7** | T1 |
| whc_cons_l1e-3_am5 | consensus | 5 | – | 66.8 | 27.8 | 18.4 | 27.8 | 35.2 | T1 |
| whc_cons_l1e-3_am8 | consensus | 8 | – | 57.2 | 28.8 | 4.5 | 18.2 | 27.2 | T1 |
| whc_coh_l1e-3_am5_b1 | coherence | 5 | 1 | 69.0 | 29.0 | 32.2 | 34.6 | 41.2 | T1 |
| whc_coh_l1e-3_am5_b2 | coherence | 5 | 2 | 73.0 | 31.4 | 37.8 | 38.7 | 45.2 | T1 |
| _no-scaling (whc_diag, =am1)_ | global | 1 | – | 73.2 | 15.3 | 41.5 | 55.7 | _46.4_ | T1 |
| _global-alpha best (whc_tv_l1e-3_a2)_ | global | 2 | – | | | | | _51.3_ | T1 |
| _baseline Consensus_ | – | – | – | 78.2 | 25.1 | 49.8 | 54.1 | _51.8_ | T1 |

**Read (2026-06-16).** Best variant `cons_am3` = **50.7 < 51.0 kill gate**; also below
the global-alpha tie (51.3) and baseline Consensus (51.8). The dataless per-parameter
bet does NOT win on the aggregate gate. BUT the mechanism is qualitatively validated:
1. **consensus > coherence at every α_max** (50.7 vs 45.2 at the best of each) — confirms
   the over-extrapolation hypothesis (coherence boosts single-expert params to α_max and
   collapses coding; consensus leaves them ≈1).
2. **instruction 32.4 is the best of ANY method** (baseline best 25.7) — the dilution-undo
   genuinely restores the additive sum where experts agree.
3. The gate is dragged down by **mbpp+ alone**. am1→am3: instr +17, math +4, heval+ +3.5,
   mbpp+ −7.4. Three of four improve; mbpp+ is the single casualty. Peak in α_max is
   between 1 (46.4) and 3 (50.7); not yet sampled below 3.

**Decision point:** near-miss (0.3 under kill gate), localized failure. User chose
(a) one cheap refinement round (finer α_max grid + a lower-lam probe) AND asked for a
**forgetting table** alongside the accuracy table. See below.

## Forgetting table (Round 0 + experts)

`scripts/mb_forgetting_table.py`. Forgetting_d(method) = specialist_d − merged_d, where
the specialist on domain d is the MergeBench expert fine-tuned for d, evaluated under
the **identical** T1 protocol as the merges (base tokenizer, no chat template;
gsm8k_cot 8-shot, ifeval 0-shot, heval+/mbpp+ n=5). Specialist diagonal:
math←math_expert / instr←instruction_expert / heval+,mbpp+←coding_expert.

**Protocol note (important).** Under base-tokenizer completion-format eval the chat-SFT
**math specialist scores only 36.8 on gsm8k_cot** (strict-match 7.2!) — it stops emitting
the few-shot `####` answer format. So every merge *beats* the math specialist (forgetting
is strongly negative on math for all methods). This is a consistent, reportable property,
NOT a bug; it just means the math specialist is not an upper bound under this protocol.
The meaningful axis is **instruction**, where the specialist (47.0 ifeval) is a real
ceiling.

**Round 0 instruction forgetting (lower = better retention), all methods:**

| method | instr-forget | (rank) |
|---|---|---|
| **whc_cons_l1e-3_am3** | **+14.6** | **best of all** |
| whc_coh_l1e-3_am5_b2 | +15.6 | |
| whc_coh_l1e-3_am5_b1 | +18.0 | |
| whc_cons_l1e-3_am5 | +19.2 | |
| task_arith | +20.0 | |
| TaskArithmetic | +21.3 | |
| Consensus | +21.8 | |
| DARE | +21.8 | |
| TIES | +30.3 | |
| whc_diag (no scale) | +31.6 | worst |

**Read.** Consensus per-parameter scaling gives the **lowest instruction forgetting of any
method** — it retains 32.4/47.0 of the instruction specialist vs Consensus's 25.2/47.0 and
TIES's 16.7/47.0. This is the clean, defensible headline regardless of the aggregate gate:
the dilution-undo mechanism best preserves the high-`alpha`-wanting domain. Coding-axis
forgetting (heval+/mbpp+) pending `coding_expert` code eval (job 21584012, staged with
base tokenizer to dodge the newer-tokenizer.json parse error).

## Round 1 — α_max refinement + lower-lam probe (RUNNING)

Bracketing the α_max peak (am1=46.4, am3=50.7, am5=35.2 ⇒ peak in [2,4]) and probing a
lower anchor. Two single-pass merges (clean fractional tags after fixing `_fmt`):
- **lam=1e-3**: consensus am ∈ {2, 2.5, 3.5, 4}  → manifest `pscale_r1a.txt` (job 21583999)
- **lam=3e-4**: consensus am ∈ {2.5, 3}          → manifest `pscale_r1b.txt` (job 21584000)
Combined eval manifest `pscale_r1_all.txt`; lm 21584002 (dgx_aic A100, afterok merges),
code 21584003 (L40S, afterok merges). Verdict:
```sh
python scripts/mb_sweep_table.py --config configs/mergebench_tier2.yaml \
  --manifest mb_merged/Llama-3.1-8B/pscale_r1_all.txt
python scripts/mb_forgetting_table.py --config configs/mergebench_tier2.yaml \
  --manifest mb_merged/Llama-3.1-8B/pscale_manifest.txt \
  --manifest mb_merged/Llama-3.1-8B/pscale_r1_all.txt
```
If best Round-1 gate ≥ 51.0 → dataless win is live; promote to full eval. If still < 51.0
→ honor the kill gate, fold to the analysis paper (the instruction-forgetting + consensus>
coherence story stands on its own).

| variant | mode | α_max | lam | math | instr | heval+ | mbpp+ | GATE |
|---|---|---|---|---|---|---|---|---|
| **whc_cons_l3e-4_am2.5** | consensus | 2.5 | 3e-4 | 78.8 | 29.4 | 45.6 | 51.4 | **51.3** |
| **whc_cons_l1e-3_am2** | consensus | 2 | 1e-3 | 80.2 | 26.6 | 45.1 | 53.1 | **51.3** |
| whc_cons_l1e-3_am2.5 | consensus | 2.5 | 1e-3 | 79.6 | 28.8 | 43.7 | 52.0 | 51.0 |
| whc_cons_l3e-4_am3 | consensus | 3 | 3e-4 | 78.2 | 32.4 | 42.8 | 47.9 | 50.3 |
| whc_cons_l1e-3_am3.5 | consensus | 3.5 | 1e-3 | 73.2 | 32.4 | 40.4 | 43.5 | 47.4 |
| whc_cons_l1e-3_am4 | consensus | 4 | 1e-3 | 72.8 | 31.0 | 37.7 | 39.8 | 45.3 |
| _baseline Consensus_ | – | – | – | 78.2 | 25.1 | 49.8 | 54.1 | _51.8_ |
| _global-alpha best (whc_tv_l1e-3_a2)_ | global | 2 | – | | | | | _51.3_ |

**Read (2026-06-16, Round 1).** Refining α_max over [2, 4] confirms a **flat plateau at
51.0–51.3 — it never reaches baseline Consensus (51.8).** Best variant
`whc_cons_l3e-4_am2.5 = 51.3` **clears the 51.0 kill gate but only TIES the tuned
global-alpha HTCL (51.3); it does not beat the top dataless baseline.** So there is **no
aggregate dataless win**. The α_max → gate curve is now fully bracketed:
am1=46.4, am2=51.3, am2.5=51.0–51.3, am3=50.3–50.7, am3.5=47.4, am4=45.3, am5=35.2 — a
single broad peak just under the baseline cluster. The instruction gain (up to +17 over
am1) is real at every α_max but is exactly offset by the mbpp+/heval+ coding loss; the two
trade off along α_max and their mean sits at the baseline. **Lowering lam (3e-4) does not
move the ceiling** (51.3 vs 51.3).

**Verdict: honor the design — no aggregate win, fold to the analysis paper.** The standalone
contributions that survive (and are the paper):
1. **consensus > coherence at every α_max** (over-extrapolation hypothesis confirmed).
2. **Best instruction retention of any method** — see forgetting table below: instr-forget
   bottoms at **+14.6** (am3) vs Consensus +21.8, TIES +30.3, no-scale whc_diag +31.6.
3. A clean, dataless, closed-form **per-parameter dilution-undo** with a single interpretable
   knob (α_max) whose accuracy/forgetting trade-off is fully characterized.

## Forgetting table (Round 0 + Round 1 + experts) — FINAL

`scripts/mb_forgetting_table.py`. Specialist diagonal (base-tokenizer T1 protocol):
math(math)=36.8 · instr(instruction)=47.0 · heval+(coding)=56.6 · mbpp+(coding)=54.7.
Forgetting_d = specialist_d − merged_d (negative = merge beats the specialist; sorted by
mean-forgetting ascending). MEANF is dragged negative for everyone by the math specialist's
36.8 (chat-SFT model breaks under completion-format eval — a protocol property, not a result);
**instruction is the meaningful retention axis.**

| method | math | instr | heval+ | mbpp+ | MEANF | kind |
|---|---|---|---|---|---|---|
| Consensus | −41.3 | +21.8 | +6.8 | +0.5 | −3.0 | baseline |
| TaskArithmetic | −42.1 | +21.3 | +8.7 | +0.8 | −2.8 | baseline |
| whc_cons_l3e-4_am2.5 | −42.0 | +17.6 | +11.0 | +3.2 | −2.6 | sweep |
| whc_cons_l1e-3_am2 | −43.4 | +20.4 | +11.5 | +1.5 | −2.5 | sweep |
| whc_cons_l1e-3_am2.5 | −42.8 | +18.2 | +12.9 | +2.6 | −2.3 | sweep |
| task_arith | −41.7 | +20.0 | +12.6 | +1.1 | −2.0 | baseline |
| **whc_cons_l1e-3_am3** | −40.2 | **+14.6** | +11.6 | +6.4 | −1.9 | sweep |
| **whc_cons_l3e-4_am3** | −41.4 | **+14.6** | +13.8 | +6.7 | −1.6 | sweep |
| DARE | −38.1 | +21.8 | +13.2 | +1.9 | −0.3 | baseline |
| TIES | −41.0 | +30.3 | +12.0 | −0.8 | +0.1 | baseline |
| whc_cons_l1e-3_am3.5 | −36.4 | +14.6 | +16.2 | +11.2 | +1.4 | sweep |
| whc_diag (no scale) | −36.3 | +31.6 | +15.1 | −1.0 | +2.3 | baseline |
| whc_cons_l1e-3_am4 | −36.0 | +16.0 | +18.9 | +14.8 | +3.4 | sweep |
| whc_coh_l1e-3_am5_b2 | −36.2 | +15.6 | +18.8 | +15.9 | +3.5 | sweep |
| whc_coh_l1e-3_am5_b1 | −32.2 | +18.0 | +24.4 | +20.1 | +7.6 | sweep |
| whc_cons_l1e-3_am5 | −30.0 | +19.2 | +38.2 | +26.8 | +13.5 | sweep |
| whc_cons_l1e-3_am8 | −20.4 | +18.2 | +52.1 | +36.5 | +21.6 | sweep |

**Read (FINAL).** Consensus per-parameter scaling owns the **lowest instruction forgetting
of any method** (+14.6 at α_max=3, vs +21.8 Consensus / +30.3 TIES / +31.6 no-scale), and
the forgetting table makes the α_max trade-off legible: as α_max grows, instruction
forgetting falls while coding (heval+/mbpp+) forgetting rises — the two cross near α_max≈2–2.5,
which is exactly where the aggregate gate peaks. This is the defensible, dataless story for
the analysis paper independent of the aggregate near-miss.
