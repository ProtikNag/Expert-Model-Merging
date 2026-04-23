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
