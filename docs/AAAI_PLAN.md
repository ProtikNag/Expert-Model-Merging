# Plan: toward an AAAI win (model merging, HTCL framework)

Status: PROPOSED (pending Protik's approval of the method bet + kill thresholds).
Owner: Protik. Drafted 2026-06-16.

## Premise (the honest constraint)

MergeBench at N=5 is **saturated**: every method, ours and all baselines, lands in
46.4-51.8 gate. The win margin available over the top baseline (Consensus 51.8) is
~3 pts, smaller than the spread among the baselines themselves. So an AAAI win must
be: a **robust margin (>1 pt, multi-seed)** on the cross-domain average / normalized
score, a **principled mechanism**, and **supporting analysis**. The data tier is
spent (see [`../results/mergebench/EXPERIMENTS_whc_gram.md`]) -- it caps at ~49. The
win path is the **dataless** side, via the one unexploited mechanism below.

## The bet: coherence-gated per-parameter scaling

We measured why dataless capped at the 51.3 tie: a single global update-scale alpha
cannot serve all domains (instruction wants high alpha, coding wants low). The fix
is to stop using one alpha. For the merged update `u = w_M^HTCL - w_pre`, scale it
per parameter by inter-expert agreement:

```
coherence_p = |sum_i tau_i,p| / (sum_i |tau_i,p| + eps)      in [0,1]
alpha_p     = 1 + (alpha_max - 1) * coherence_p^beta
w_M         = w_pre + alpha_p (elementwise*) u
```

Coherent params (experts agree) get boosted toward alpha_max (undo the 1/N
dilution); conflicted params stay near 1 (avoid the out-of-distribution blow-up that
killed global alpha in the data tier). This directly resolves the documented tension
and composes with the curvature weighting (`whc_diag`), so the method is two parts
from one objective -- harder to dismiss as "just TIES".

Dataless, closed-form, O(params), one merge. **Honest odds: ~40-50%** of a robust
>1 pt win at full scale + multi-seed, given saturation. Phase 1 tells us cheaply.

## Phases (cheap -> full, with kill gates)

| Phase | What | Cost | Kill gate |
|---|---|---|---|
| 0. Build | coherence-gated alpha in `mergebench/llm_merge.py`; unit-test limits (coherence=1 -> global alpha; =0 -> alpha=1) | ~0.5 d | -- |
| 1. Gate sweep | sweep (alpha_max, beta) at T1 (LIMIT=500) | ~1-2 d | best gate < 51.0 -> no dataless win; fold to analysis paper |
| 2. Lock the win | full eval (LIMIT=0, n=10, +multilingual+safety) + 3 seeds + fix math_expert ceiling | ~2-3 d | win gone at full/multi-seed -> it was noise; fold |
| 3. Breadth | re-run full pipeline on a 2nd base (gemma-2-2b 5-domain) | ~3-4 d | no win on base 2 -> scope the claim |
| 4. Write | framework + method + analysis | ~1-2 wk | -- |

T1/T2 proxy protocol and the merge/eval infra are the same ones used in the
whc_gram campaign (see EXPERIMENTS_whc_gram.md "Proxy protocol").

## What makes it an AAAI paper (not just a number)

1. Unified second-order merging framework spanning dataless <-> data tiers.
2. Method: curvature-weighted merge + coherence-gated adaptive scaling, one objective.
3. Wins: Tier 1 (N=2, already a win) + Tier 2 (N=5, new method) across 2 bases, multi-seed.
4. Analysis: the N-scaling dilution derivation + alpha fix; the alpha asymmetry
   (diagonal merge is rescued by alpha, full-covariance merge is destroyed by it
   because the least-squares solve extrapolates); plain RegMean is numerically
   degenerate at 8B (the ridge-toward-mean makes Gram-merging usable); the
   GLUE-winning iterative catch-up does NOT transfer to LLM scale.

## Risks

- Saturation: coherence-gating may still only tie. Phase 1 is make-or-break, ~2 days, cheap.
- Novelty positioning: separate cleanly from TIES (discrete sign-trim), Consensus
  (TALL masks), AdaMerging (learned TEST-TIME coefficients). Our angle: continuous,
  curvature-grounded, closed-form, no test data.
- Deadline: AAAI full-paper deadlines are typically ~mid-August (VERIFY the exact
  AAAI-2027 date). From mid-June that is ~2 months -- feasible if Phase 1 hits.

## Open decisions (need Protik)

1. Approve coherence-gated alpha as the primary win bet (vs a data-informed additive merge)?
2. Phase-1 kill threshold = 51.0 gate? (stop-and-fold line)
3. 2nd base for breadth: gemma-2-2b or another MergeBench base?

## Fallback if Phase 1 fails

Fold to an analysis / negative-results paper (TMLR or workshop): the unified
objective + dilution analysis + the three structural findings stand on their own,
independent of a SOTA number. Lower venue, but a real and defensible contribution.
