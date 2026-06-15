# Notes on HTCL → WHC (Model-Merging) Adaptation

Canonical record of every modification made to the HTCL / CRL papers when
porting their machinery to the model-merging problem. Each item names the
source equation, states the original assumption, notes why it needs changing
for merging, and gives the change we make.

Source papers:

- HTCL: [`docs/Hierarchical_continual_learning___arxiv_copy.pdf`](docs/Hierarchical_continual_learning___arxiv_copy.pdf)
- WHC derivation: [`docs/CRL.pdf`](docs/CRL.pdf)

---

## 1. Setting: from sequential CL to parallel merging

**Original.** HTCL assumes tasks arrive sequentially under a permutation
$\pi(t)$ and its distinguishing contributions target *task-order
sensitivity* via intra-group permutation search.

**Merging.** All $N$ experts are available at once, each trained
independently from a shared pretrained init $w_\text{pre}$. There is no
permutation to search.

**Modification.** Drop the permutation search. Retain only the consolidation
rules (Eq. 6 and Eq. 9) as merging primitives.

---

## 2. Two update rules, both ported

### Eq. 9 (CRL) — N-expert one-shot closed form

$$
\hat w_\lambda = \Big(\textstyle\sum_i \alpha_i H_i + \lambda I\Big)^{-1}
                \Big(\textstyle\sum_i \alpha_i H_i w_i^\star + \lambda \bar w\Big)
$$

This is the minimizer of the regularized weighted-Hessian surrogate. Applied
directly with $\alpha_i = 1/N$ it gives **WHC-A** (flat) and is the atomic
operation inside **WHC-B** (hierarchical tree over experts).

### Eq. 6 (HTCL) — two-model incremental update

$$
w_1^{(t)} = w_1^{(t-1)} + \big(H + \lambda I\big)^{-1}
           \big(\lambda \Delta d - g\big),\qquad
\Delta d = w_\ell^{(t)} - w_1^{(t-1)}
$$

This is the minimizer of the surrogate "past-expert loss + $\tfrac{\lambda}{2}
\|w - w_\ell\|^2$" at the current anchor point. $H$ and $g$ are the curvature
and gradient of the past-expert loss at $w_1^{(t-1)}$.

Applied pairwise it gives **WHC-pair** (symmetrized: apply in both directions
and average). Embedded in a hierarchical tree it gives **WHC-C**. Applied in
a chain over $N$ experts it gives **WHC-D** (sequential absorption).

---

## 3. The Hessian proxy: dataless task-vector curvature

**Original.** Both papers use the Hessian $H_i = \nabla^2 L_i(w_i^\star)$ and
in practice approximate it by the diagonal empirical Fisher.

**Merging constraint.** Fisher requires labeled per-expert data at merge
time. To put our method in the **dataless** tier (alongside Simple
Averaging, Task Arithmetic, TIES), we need a curvature proxy that uses only
expert weights and the pretrained init.

**Modification.** Use the squared task vector as the per-parameter curvature:
$$
\hat F_i^{(k)} = \big(w_i^{(k)} - w_\text{pre}^{(k)}\big)^2.
$$

**Theoretical justification.** Under gradient descent with small steps,
$w_T - w_0 = -\eta \sum_t g_t$, so $(w_T - w_0)^2 = \eta^2 (\sum_t g_t)^2 \le
\eta^2 T \sum_t g_t^2 \propto \eta^2 T \hat F$. The squared task vector is
proportional to the path-integrated gradient (up to a coarsening factor) and
has been used as an importance proxy since Synaptic Intelligence (Zenke et
al. 2017) and, implicitly, in TIES (Yadav et al. 2023).

**Implementation.** [`src/merging/whc.py`](src/merging/whc.py) exposes
`taskvec_curvature(states, pretrained)` alongside `select_curvature(...,
source=...)` with ``"taskvec"`` (default, dataless) and ``"fisher"``
(ablation, uses provided empirical Fisher).

---

## 4. Hierarchical tree over experts

**Original.** HTCL's hierarchy is temporal — levels correspond to
progressively slower timescales.

**Merging.** No time. We reinterpret the hierarchy as a tree over experts:
leaves are the $N$ experts, each internal node fuses its two children using
either Eq. 9 (in WHC-B) or Eq. 6 (in WHC-C).

**Curvature propagation.** Under the Laplace approximation each child
posterior is Gaussian with precision $F_{c_j}$; the product is Gaussian with
precision $\sum_j F_{c_j}$. So the propagated curvature of an internal
node is the sum of its children's curvatures.

**Per-level $\lambda_\ell$.** Following HTCL's intuition that deeper levels
should be more conservative, we use $\lambda_\ell = \lambda_0 \cdot
\rho^{\ell-1}$ with $\rho \in (0, 1]$.

---

## 5. Option X vs. Option Y for $g$ in Eq. 6

**Original.** HTCL computes $g^{(t-1)}$ empirically on replay data at the
current hierarchical state.

**Merging.** Access to per-expert data at merge time would make the method
no longer strictly dataless. We support two options:

- **Option X** (default): assume $g = 0$ (small-step approximation and/or
  the fact that $w_1^{(0)} = w_i$ starts at an expert's own optimum). Keeps
  WHC-C/D fully dataless.
- **Option Y** (ablation): compute $g$ empirically on the expert's training
  data (the data we already use for the Fisher ablation). Faithful to the
  paper but not dataless.

In WHC-D's sequential chain the anchor aggregates multiple experts; when
Option Y is enabled we approximate the aggregate gradient as the mean of
per-expert gradients at the current anchor.

**Implementation.** The merge functions accept an optional `grad_fn`
callable. When `None` (default), Option X. Otherwise Option Y.

---

## 6. Expert weighting $\alpha_i$

HTCL supports general $\alpha_i$ with $\sum_i \alpha_i = 1$. In merging we
default to uniform $\alpha_i = 1/N$. The constant factor is absorbed into
$\lambda$ during tuning, so only $\lambda$ is exposed as a hyperparameter.

---

## 7. What is *not* imported from HTCL

- **Intra-group permutation search.** There is no arrival order to search.
- **Catch-up phase.** The post-consolidation gradient iterations on replay
  data would require per-expert data at merge time and move the method out
  of the dataless tier. Noted as future work.
- **Multi-level temporal semantics.** Replaced by the tree-over-experts
  interpretation in §4.

---

## 8. Bug fixes and faithfulness notes for baselines

- **TIES global trim** ([`src/merging/ties.py`](src/merging/ties.py)): the
  pilot implementation trimmed per-tensor; the paper and official code trim
  globally across the flattened task vector. Fixed to use
  `_global_trim(..., keep_frac)`.
- **RegMean shape math**
  ([`src/merging/regmean.py`](src/merging/regmean.py)): verified
  `torch.linalg.solve(G.T, sumGW.T).T == sumGW @ G^{-1}` for symmetric Gram.
- **RegMean++ simple variant**
  ([`src/merging/regmean_plus.py`](src/merging/regmean_plus.py)): we provide
  both the faithful per-layer-refresh version and a lighter "simple" drop-in
  that uses merged-so-far averaged weights for Gram collection. The light
  version is used when probe-forward compute is a constraint.

---

## 9. Fairness of the comparison

- **Val-based hparam selection.** Every method is tuned on a held-out
  validation split of each GLUE task's standard validation set (halved into
  val/test). Final metric is reported on test.
- **Matched sweep budget.** Config grids are sized comparably per method
  (roughly 1–8 configs each). See [`configs/glue_roberta.yaml`](configs/glue_roberta.yaml).
- **Stratified comparison.** Methods are split into the dataless tier (no
  data at merge time) and the statistics-using tier (needs Fisher / Grams /
  gradients). WHC-*-dataless sits in the dataless tier; WHC-A-fisher is
  included only as an ablation for comparing the dataless proxy against the
  true Fisher.
- **Per-task primary metric.** We use the GLUE-standard per-task metric
  (Matthews for CoLA, F1/Acc average for MRPC/QQP, accuracy otherwise).
- **Single seed** per user request; multi-seed is a straightforward follow-up.

---

## 10. Expected advantages of WHC to validate in the GPU run

1. **Dataless.** WHC-A/B/C/D-dataless use weights only — no data at merge
   time — matching Simple/TA/TIES in access requirements.
2. **One-shot / closed-form.** No training, no gradient descent. Runtime
   dominated by a handful of elementwise operations.
3. **Curvature-aware.** Unlike Simple/TA/TIES, WHC uses per-parameter
   importance (even if proxied). Expected to beat the dataless tier.
4. **Ensemble-mean anchor.** Prevents the rank-deficient-inverse collapse
   that plagues vanilla Fisher merging at strong regularization.

If these hold on RoBERTa-base + 7 GLUE tasks, WHC is a defensible
contribution as **dataless, one-shot, curvature-aware merging**.

---

## 11. N-scaling dilution and the update-scale $\alpha$ (MergeBench Tier 2)

**Naming.** The merging method is branded **HTCL** in the writeup; the dataless
diagonal instance (Eq. 9 with squared-task-vector curvature) is `whc_diag` in code.

**Observation.** On MergeBench Tier 1 (gemma-2-2b, $N=2$) HTCL won the dataless
tier. On Tier 2 (Llama-3.1-8B, $N=5$) the *same* closed form (`lam=1e-4`)
**underperformed every dataless baseline** on the math+instruction+coding gate
(instr 15.3, math 73.2 vs baselines $\sim$25–27 / $\sim$78).

**Diagnosis.** Substituting $w_i = w_\text{pre} + \Delta_i$ into the diagonal
closed form, the net update is
$$
w_M - w_\text{pre} \;=\;
\frac{\sum_i \Delta_i^{3} + (\lambda/N)\sum_i \Delta_i}
     {\sum_i \Delta_i^{2} + \lambda}.
$$
For small $\lambda$ this is $\approx \sum_i \Delta_i^3 / \sum_i \Delta_i^2$, a
curvature-weighted **mean** of the per-expert task vectors. A mean shrinks each
expert's contribution by $\sim 1/N$ relative to task arithmetic's **sum**
$\,\text{scale}\cdot\sum_i \Delta_i$. At $N=2$ the dilution is mild; at $N=5$ it
is severe, so HTCL under-applies every expert and trails the baselines. The
$\sum_i \Delta_i^3$ numerator also partially cancels under sign conflict, which is
worse the more experts disagree.

**Modification — update scale $\alpha$.** Rescale the net deviation from base:
$$
w_M \;=\; w_\text{pre} \;+\; \alpha\,\big(w_M^{\text{HTCL}} - w_\text{pre}\big),
$$
with $\alpha \approx N$ compensating the averaging dilution. This is the analogue
of task arithmetic's scaling coefficient and is **distinct from the per-expert
weights $\alpha_i$ of §6** (those stay uniform $1/N$; this is a single global
post-hoc scale). $\alpha=1$ recovers the plain closed form, so the change is
backward compatible.

**Implementation.** `merge_checkpoints(..., alpha=...)` in
[`mergebench/llm_merge.py`](mergebench/llm_merge.py); swept by
[`scripts/mb_sweep_whc.py`](scripts/mb_sweep_whc.py) over a $(\lambda,\alpha)$ grid.

**Evidence (gate, dataless, $\lambda=10^{-3}$; math+instr+coding, LIMIT=500).**
$\alpha$ lifts math and instruction sharply (instr $14\!\to\!27\!\to\!31$, math
$74\!\to\!80\!\to\!77$ as $\alpha: 1\!\to\!2\!\to\!3$) but **trades away coding**
(heval+/mbpp+ fall monotonically as $\alpha$ rises: mbpp+ $55\!\to\!53\!\to\!48$).
Instruction wants high $\alpha$, coding wants low $\alpha$, and a single global
$\alpha$ cannot serve both.

**Verdict.** No $(\lambda,\alpha)$ clears the dataless baseline cluster. The best
variant `whc_tv_l1e-3_a2` ($\alpha=2$) gates at 51.3 vs Consensus 51.8 / Task
Arithmetic 51.6 — a **statistical tie, not a win** (gate noise $\sim$2 pts). So
$\alpha$ makes HTCL *competitive* with the dataless tier at $N=5$ but does not beat
it; the trade-off is structural, not a tuning artifact. The averaging dilution
diagnosis is correct and useful as analysis, but the dataless result is a tie.

**Implication for the contribution.** The leverage for a stronger result is the
*data*-using iterative variant (`whc_tree`, §7's "catch-up") against
RegMean/RegMean++, not further dataless tuning. The dataless tie + the N-scaling
dilution analysis + (if it lands) a data-tier win is the realistic story. A
domain-adaptive scale (per-expert or per-parameter $\alpha$ in place of one global
scale) is an open lead suggested directly by the coding/instruction tension. Safety
(untested) is the one domain where the tie could become a loss, since larger
$\alpha$ is a more aggressive merge.

## 12. Data tier: the `whc_gram` port (HTCL-data vs RegMean)

**The merge.** For each Linear weight $W_i \in \mathbb{R}^{o\times d}$ with input
Gram $G_i = \tfrac{1}{T}\sum_t x_t x_t^\top \in \mathbb{R}^{d\times d}$ (the
token-averaged second moment of the layer input, collected on expert $i$'s own
domain data), the $N$-expert single-pass closed form is
$$
W_M \;=\; \Big(\textstyle\sum_i W_i G_i + \lambda \bar W\Big)
          \Big(\textstyle\sum_i G_i + \lambda I + \gamma\,\mathrm{diag}(\bar F_{\text{in}})\Big)^{-1},
$$
with $\bar W = \tfrac1N\sum_i W_i$ and $\bar F_{\text{in}}$ the input-dim projection
of the diagonal Fisher (average the $[o,d]$ diagonal over the output axis, $\to[d]$).
This is the $N$-way generalization of the GLUE `whc_tree` pairwise node (§4), lifted
from the binary tree to one solve per layer for the billion-param regime.

**Limits (the identity of the method).** As $\lambda\to0$ this is exactly RegMean,
$W_M=(\sum_i W_iG_i)(\sum_i G_i)^{-1}$ (Jin et al. 2023). As $\lambda\to\infty$ it
collapses to the simple mean $\bar W$. The $\lambda\bar W$ term is the
ridge-toward-mean anchor that is HTCL's signature (the full-Gram analogue of the
diagonal $\lambda\bar w$ anchor of §2, Eq. 9); the $\gamma$ term injects curvature
that pure-Gram RegMean discards. So $\lambda$ is the single knob separating
HTCL-data from RegMean, and $\lambda=0$ is a built-in RegMean ablation point.

**Why no $\alpha$ here.** The §11 dilution fix does not apply: RegMean-type solves
are not curvature-weighted *means* of the experts, so there is no $1/N$ shrink to
undo. The data tier trades the global scale $\alpha$ for the data-driven Gram
geometry, which is the point.

**Iterative catch-up ($K\ge1$).** Re-collect each domain's Gram *on the merged
model* (the linearization point moves to the merge, the experts stay fixed), then
re-solve. This is §7.2 Fix 3 — the part that, on GLUE, pushed `whc_tree_iter` past
RegMean. At 8B scale it is an orchestration loop (re-estimate Grams on the merged
checkpoint, re-merge), not new merge math.

**Implementation.** `merge_checkpoints(method="whc_gram", grams_dirs=..., lam=...,
gamma=..., fisher_dirs=...)` in [`mergebench/llm_merge.py`](mergebench/llm_merge.py);
Grams by [`scripts/mb_gram_estimate.py`](scripts/mb_gram_estimate.py) (forward hooks
on the target Linears, $X^\top X$ accumulated on CPU); driver
[`scripts/mb_merge_whc_gram.py`](scripts/mb_merge_whc_gram.py). Unit test
[`tests/test_whc_gram.py`](tests/test_whc_gram.py) pins the closed form, the
RegMean and mean limits, and the no-Gram/non-Linear fallbacks.

**Scale caveat.** Llama-3.1-8B's `mlp.down_proj` has a $14336\times14336$ Gram
($\sim$3.3 GB fp32 each, $\sim$105 GB across 32 layers per expert). The default
estimate excludes it (fits 96 GB) for a fast first pass; the faithful all-Linear
RegMean comparison needs a high-RAM node. Match the included-layer set across
`whc_gram` and the RegMean baseline or the comparison is apples-to-oranges.

**Empirical results (Tier 2, T1 gate; ledger
[`EXPERIMENTS_whc_gram.md`](results/mergebench/EXPERIMENTS_whc_gram.md)).**
Single-pass `whc_gram` loses: best gate 48.7 vs Consensus 51.8, below even the
tuned dataless HTCL (51.3). $\lambda=0$ (plain RegMean) is numerically degenerate
at 8B (the summed activation Gram is near low-rank, so the unregularised solve
explodes); the ridge-toward-mean is what makes Gram-merging *usable* here, a point
for the method even though the gate still trails. `whc_gram` wins only `mbpp+`
(table-best) and loses math/instr/heval+, i.e. the averaging solve is structurally
weaker than additive task arithmetic on those domains.

**The $\alpha$ asymmetry (diagonal vs full covariance).** The update-scale $\alpha$
that rescues `whc_diag` (instr $15\to31$) is *catastrophic* for `whc_gram`:
$\alpha=2$ craters math $75\to40$, $\alpha=3$ collapses the model (math $8.8$). The
reason is a real structural difference. `whc_diag` returns a per-coordinate convex
*mean* of the experts' weights, bounded inside their hull, so scaling the deviation
from base extrapolates gently. The full-Gram least-squares solve
$(\sum W_iG_i)(\sum G_i)^{-1}$ is **not** a convex combination — it already
extrapolates beyond the experts — so multiplying that deviation by $\alpha$ throws
the weights out of distribution into garbage. Implication: the dilution fix is
diagonal-specific; for the data tier the lever is improving the *solve* (iterative
re-linearisation, the GLUE `whc_tree_iter` edge), not rescaling it. The
`task_arith` fallback on the non-Gram keys is the one transferable piece (+0.4 gate,
instr $19\to23$), since those keys really were a diluted mean.
