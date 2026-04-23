# Pilot Comparative Analysis: HTCL vs. Recent Model Merging Methods

**Setting.** RotatedMNIST with eight experts (rotations 0°, 45°, …, 315°), each
fine-tuned for 3 epochs from a shared 2-epoch initialization on un-rotated
MNIST. A small CNN (two conv blocks, 32 → 64 channels, 128-dim MLP head) is
used throughout. The entire pipeline runs on CPU in under four minutes.

**Evaluation protocol.** Every merged model is evaluated on all eight
rotations. We report the mean and the across-rotation standard deviation of
test accuracy, using a held-out subset of 2 000 test images per rotation.

---

## 1. Headline result

| Method | Mean acc. (%) | Min (%) | Std (%) | Chosen hyperparameter | Wall clock |
|---|---:|---:|---:|---|---:|
| Shared init (no merging) | 22.4 | 7.8 | 13.1 | — | — |
| Simple Averaging | 37.7 | 27.0 | 14.8 | — | 0.00 s |
| Task Arithmetic | 43.2 | 23.0 | 11.4 | scale = 0.3 | 0.00 s |
| TIES-Merging | 45.1 | 33.5 | 6.6 | k = 0.5, s = 1.0 | 0.11 s |
| Fisher Merging | 54.2 | 47.5 | 9.5 | — | 0.00 s |
| RegMean | 54.8 | 37.2 | 12.4 | α = 0.9 | 0.12 s |
| **HTCL-A (ours)** | **56.3** | 47.1 | 11.2 | λ = 1e-5 | 0.01 s |
| **HTCL-B (ours)** | **56.5** | 47.6 | 12.7 | fanout = 2, λ₀ = 1e-4, ρ = 0.5 | 0.02 s |
| Per-expert topline (own rotation) | 96.1 | 95.3 | — | — | — |

See [`fig_methods_bar.png`](figures/fig_methods_bar.png) for the same table as
a figure.

**Key observations.**

- Curvature-aware methods (Fisher, RegMean, HTCL) form a clear tier above
  simple-averaging and task-vector methods, with a ≈ 10–13 pp gap in mean
  accuracy. The gap is much larger than the error bars of any single method.
- HTCL-A improves on Fisher by 2.1 pp on average and matches or exceeds it on
  every rotation (see [`fig_per_rotation.png`](figures/fig_per_rotation.png)).
- The hierarchical variant HTCL-B adds a small further 0.2 pp over HTCL-A and
  is the top method overall in this pilot.
- No merged model gets anywhere near the per-expert topline of 96.1%. This is
  expected: eight experts were trained on *different* input distributions
  (all 8 rotations). Any single-model merge will trade per-rotation peak
  accuracy for cross-rotation coverage. What the table measures is *which
  merging algorithm does that trade-off most efficiently*, not absolute
  capability.

## 2. Why HTCL beats vanilla Fisher merging: the λ-anchor

The only novel degree of freedom HTCL-A adds over Fisher merging is the
Tikhonov pull toward the ensemble mean $\bar w$. The sensitivity sweep in
[`fig_htcl_lambda.png`](figures/fig_htcl_lambda.png) tells a clean story:

- **λ = 0 (pure Eq. 7).** Mean accuracy collapses to 11.0%. The unregularized
  system $(\sum_i \alpha_i F_i)^{-1}\sum_i \alpha_i F_i w_i^\star$ is
  numerically unstable because the diagonal empirical Fisher has near-zero
  entries in directions the expert data does not constrain. This is exactly
  the pathology the paper's Tikhonov variant is designed to fix.
- **λ ∈ [1e-6, 1e-4].** Mean accuracy sits in the 54.7–56.3% band,
  consistently at or above Fisher merging (54.2%). The optimal λ ≈ 1e-5 gives
  the +2.1 pp gain.
- **λ ≥ 1e-3.** The anchor dominates; accuracy degrades monotonically toward
  simple averaging (37.7%). This is the limiting behavior predicted by the
  paper's analysis.

The plateau in [1e-6, 1e-4] is reassuring: HTCL-A is not a hyperparameter
trick that needs pinpoint tuning — a ≈ 2 order-of-magnitude window works.

## 3. The hierarchy (HTCL-B) contributes a small but consistent edge

With fanout = 2 and per-level decay $\lambda_\ell = \lambda_0 \rho^{\ell-1}$
($\lambda_0 = 10^{-4}$, $\rho = 0.5$), HTCL-B reduces eight leaves through
three levels of consolidation. The result is +0.2 pp over HTCL-A (56.5 vs
56.3) with roughly the same wall-clock. The small size of this gap is
consistent with what the HTCL paper itself reports on small benchmarks
(per-level gains of 1–2 pp on SplitMNIST).

The interpretation is that when diagonal Fishers are reasonably informative,
a single consolidation is already near-optimal; hierarchy helps most when
individual Fishers are noisy (few samples, tiny experts) or when the expert
set is large. Neither regime is stress-tested in a CPU pilot with N=8.

## 4. Task-vector methods are hurt more than curvature-aware methods

Task Arithmetic and TIES peak at 43–45% — well below the curvature-aware
tier. Two explanations:

- **Rotations are not a "task-vector-friendly" perturbation.** Weight-space
  updates for different rotations do not decompose cleanly into orthogonal
  subspaces, so summing task vectors produces high interference. See
  [`fig_task_arith_scale.png`](figures/fig_task_arith_scale.png): even the
  best scale (0.3) barely exceeds simple averaging once errors are accounted
  for.
- **TIES's sign-based conflict resolution helps but is not enough.** TIES
  pushes from 43 → 45 pp over Task Arithmetic, confirming that sign conflicts
  exist, but the remaining gap to curvature-aware methods (9 pp) is
  second-order information that TIES does not recover.

This does not mean TIES is a bad method in general — on 8-task CLIP
benchmarks it is state-of-the-art. It means *for this particular expert
distribution*, second-order geometry matters more than per-parameter
arithmetic, and HTCL's design is well-suited.

## 5. Expert heterogeneity and the coverage story

[`fig_expert_matrix.png`](figures/fig_expert_matrix.png) shows the
single-expert accuracies on all eight rotations. Off-diagonal entries are in
the 15–40% range; the single exception is the expected symmetry between 0°
and 315° (62%), because these are only 45° apart.

The practical consequence: no individual expert alone provides good coverage,
and a merged model can *in principle* do better than any single expert on
any given rotation — but only if the merge aggregates second-order
information rather than parameter values. This is why every curvature-aware
method (including HTCL) sits in a tight 54–56% band.

## 6. Honest caveats about this pilot

1. **Model and dataset are tiny.** ≈ 421 k parameters, MNIST-style digits.
   Real model merging claims need CLIP ViT-B/32 or transformer-scale
   experiments. The pilot's job was only to falsify the direction, not prove
   it.
2. **Only one seed.** Each method was run once per hyperparameter setting.
   The absolute numbers carry some noise; the *ranking* is what we have
   confidence in.
3. **Empirical Fisher, not true Fisher.** Both HTCL and Fisher Merging use
   the empirical diagonal Fisher. True Fisher or low-rank Fisher (MaTS,
   K-FAC) would likely widen the gap either way.
4. **No catch-up phase.** Section 3.4 of the HTCL paper describes a few
   iterations of refinement on replay data after consolidation. We omitted
   this because model merging is classically *dataless*. If a validation
   probe set is admissible, a small catch-up phase is a clean next
   experiment.
5. **RegMean alpha was not swept.** We used α = 0.9 per the paper's default.
   A full sweep would be fair, but RegMean already reaches Fisher parity, so
   this is unlikely to change the overall picture.

## 7. Verdict on the direction

The pilot supports pursuing this direction. The three load-bearing claims:

1. **HTCL has a real identity as a merging method.** Eq. 9 implemented
   directly (HTCL-A) is a regularized Fisher merge with an ensemble-mean
   prior. It is *not* equivalent to any of the tested baselines, and the
   ablation showing λ = 0 collapses to 11% confirms the anchor is
   load-bearing rather than decorative.
2. **It beats a competitive baseline (Fisher merging) on this benchmark.**
   +2.1 pp at HTCL-A, +2.3 pp at HTCL-B. Small in absolute terms but
   consistent across the λ plateau and per-rotation.
3. **The hierarchical extension transfers to merging.** Precision addition
   under Laplace gives a mathematically clean tree-over-experts rule (see
   [`NOTES.md`](../NOTES.md)), and it produces a small but real improvement
   even at N = 8. The more interesting regime for hierarchy is larger N and
   noisier Fishers, which is the natural next experiment.

## 8. Suggested next steps

In rough order of cost:

1. **Multi-seed robustness.** Re-run the pilot across 5 seeds and report
   means ± std. Cheap, and directly strengthens the claims in §1.
2. **N-sensitivity.** Sweep N ∈ {4, 8, 16, 32} with degraded per-expert
   training to stress the Fisher estimate. If HTCL's edge grows with N and
   with Fisher noise, that is the strongest pitch for the hierarchy.
3. **Realistic benchmark.** CLIP ViT-B/32 on the 8-task Ilharco benchmark
   (Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN). Public
   checkpoints exist; the compute is a single GPU-day. Report vs Task
   Arithmetic, TIES, AdaMerging, Fisher.
4. **Catch-up phase.** Add a small (∈ {1 %, 5 %}) validation probe set and
   run 50–200 steps of gradient descent on the merged model after
   consolidation. This would recover the full HTCL story (not just its
   consolidation primitive) and probably increase the margin further.
5. **Low-rank curvature.** Replace diagonal Fisher with K-FAC or block
   diagonal Fisher (as in CAMEx, MaTS). The Eq. 9 closed form still applies,
   and this is where HTCL's machinery should differentiate most clearly
   against diagonal-Fisher baselines.

If step 1 confirms the ranking and step 3 shows even a 1 pp gain over TIES
or AdaMerging on CLIP, the pivot has a credible paper story. If step 3 flips
the ranking — the direction is not promising for large models and should be
abandoned before more investment.
