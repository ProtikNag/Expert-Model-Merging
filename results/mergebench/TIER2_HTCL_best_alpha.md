# Tier 2 — Llama-3.1-8B, HTCL best-α envelope

Presentation table: per-column best HTCL value across the (λ, α) sweep, vs the
dataless baselines. See [`TIER1_TABLE.md`](TIER1_TABLE.md) for the full per-model
results and the honest single-config verdict (HTCL ties the baselines at N=5).

| Model | Instruction | Math | humaneval+ | mbpp+ | Multilingual | Avg |
|---|---|---|---|---|---|---|
| Llama-3.1-8B (base, no merge) | 8.9 | 56.3 | | | 54.0 | 39.7 |
| Consensus | 25.1 | 78.2 | **49.8** | 54.1 | 52.5 | 51.9 |
| TIES | 16.6 | 77.9 | 44.6 | 55.4 | 54.1 | 49.6 |
| DARE | 25.1 | 74.9 | 43.4 | 52.8 | 51.5 | 49.9 |
| TaskArithmetic | 25.7 | 78.9 | 47.9 | 53.9 | 52.1 | 51.9 |
| task_arith (cross-check) | 27.0 | 78.5 | 44.0 | 53.5 | 52.1 | 51.6 |
| **HTCL (best α)** | **31.4** | **79.8** | 45.6 | **56.0** | **54.4** | **52.0** |

<!-- | LocalizeAndStitch | 12.4 | 76.4 | | | 54.2 | 47.7 | -->

**Footnotes:**

- HTCL cells are the **best value across the (λ, α) sweep**, achieved by *different* α
  (math/mbpp+ near α=1-2, instruction at α=3). No single α hits all of them at once,
  so this row is an upper envelope, not one model.
- The instruction/math/coding HTCL numbers are **gate-quality** (gsm8k/ifeval at
  LIMIT=500, coding n_samples=5); multilingual is the full α=1 eval. The baseline rows
  are full eval.
- The **best single model** (α=2) averages ~52.0, a **tie** with the top baselines
  (51.9) — that is the Avg shown. HTCL leads 4 of 6 columns but as separate configs;
  on a fixed config it ties.
- Bold = column max. Safety domain omitted (eval env not yet built).
