"""Weighted Hessian Consolidation (WHC) — merging primitives ported from HTCL.

Two update rules are ported from the HTCL / CRL derivations:

Eq. 9 (CRL.pdf) — N-expert one-shot closed form:

    .. math::
        \\hat w_\\lambda = \\Big(\\sum_i \\alpha_i H_i + \\lambda I\\Big)^{-1}
                          \\Big(\\sum_i \\alpha_i H_i w_i + \\lambda \\bar w\\Big)

Eq. 6 (HTCL paper) — two-model incremental update, pulling ``w_anchor``
toward ``w_new``:

    .. math::
        w_\\text{next} = w_\\text{anchor} + (H + \\lambda I)^{-1}
                        (\\lambda (w_\\text{new} - w_\\text{anchor}) - g)

where ``H`` and ``g`` are the curvature and gradient of the past-expert loss
at ``w_anchor``. When ``g = 0`` (i.e. ``w_anchor`` is exactly at a past
expert's optimum, or we invoke the small-step approximation), Eq. 6 reduces
to a Bayesian-update-shaped interpolation.

Curvature source
----------------
Both rules accept an arbitrary per-expert diagonal curvature dict. We support
two choices:

- ``"fisher"``: true empirical diagonal Fisher (requires labeled data).
- ``"taskvec"``: :math:`|w_i - w_\\text{pre}|^2`, the squared task vector. A
  path-integrated-gradient proxy for Fisher that is fully **dataless**. This
  is the default and is what makes WHC a dataless merging method.

Hierarchy, sequential, pairwise
-------------------------------
We expose five user-facing functions:

- :func:`whc_a`       — Eq. 9 with N experts at once.
- :func:`whc_b`       — Eq. 9 applied recursively in a tree.
- :func:`whc_pair`    — Eq. 6 on exactly two models (symmetrized).
- :func:`whc_c`       — hierarchical tree, every internal node merges its
  two children via :func:`whc_pair`.
- :func:`whc_d`       — sequential: chain Eq. 6 to absorb experts one at a
  time.

All rules operate on floating-point parameters only; non-float buffers (e.g.
BatchNorm ``num_batches_tracked``) are kept from the first expert.

A shared numerical floor ``_EPS`` protects the division when curvatures have
exactly-zero entries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch

from ..utils import mergeable_keys

_EPS = 1e-12

GradFn = Callable[[Dict[str, torch.Tensor], int], Dict[str, torch.Tensor]]
"""Signature of an Option-Y gradient function.

``grad_fn(w_now, i)`` returns ``\\nabla L_i(w_now)`` -- the gradient of
expert-``i``'s loss at the current merged point ``w_now``. When ``None``, we
use Option X (``g = 0``), keeping the update fully dataless.
"""


# ---------------------------------------------------------------------------
# Curvature proxies
# ---------------------------------------------------------------------------

def taskvec_curvature(expert_states: List[Dict[str, torch.Tensor]],
                      pretrained: Dict[str, torch.Tensor],
                      ) -> List[Dict[str, torch.Tensor]]:
    """Return per-expert diagonal curvature dicts using the task-vector
    proxy :math:`|w_i - w_\\text{pre}|^2`.

    This uses **only** expert weights and the pretrained init -- no data at
    merge time. Under gradient descent with small steps, this quantity is
    proportional to the path-integrated gradient squared and is a standard
    dataless surrogate for the Fisher (Zenke et al. 2017, Yadav et al. 2023).
    """
    keys = mergeable_keys(pretrained)
    out: List[Dict[str, torch.Tensor]] = []
    for state in expert_states:
        out.append({k: (state[k] - pretrained[k]) ** 2 for k in keys})
    return out


def select_curvature(expert_states: List[Dict[str, torch.Tensor]],
                     pretrained: Dict[str, torch.Tensor],
                     source: str,
                     fishers: Optional[List[Dict[str, torch.Tensor]]] = None,
                     ) -> List[Dict[str, torch.Tensor]]:
    """Dispatch to the requested curvature source."""
    if source == "taskvec":
        return taskvec_curvature(expert_states, pretrained)
    if source == "fisher":
        if fishers is None:
            raise ValueError("source='fisher' but no fishers provided")
        return fishers
    raise ValueError(f"Unknown curvature source: {source!r}")


# ---------------------------------------------------------------------------
# Internal primitives
# ---------------------------------------------------------------------------

def _ensemble_mean(expert_states: List[Dict[str, torch.Tensor]],
                   keys: Sequence[str]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([s[k] for s in expert_states], 0).mean(0)
            for k in keys}


def _uniform_weights(n: int) -> List[float]:
    return [1.0 / n] * n


def _eq9_merge(expert_states: List[Dict[str, torch.Tensor]],
               curvatures: List[Dict[str, torch.Tensor]],
               lam: float,
               weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    """Eq. 9 applied to the given experts with the given curvatures.

    Non-float buffers are copied from the first expert (they are outside the
    merging calculus entirely).
    """
    template = expert_states[0]
    keys = mergeable_keys(template)
    w_bar = _ensemble_mean(expert_states, keys)
    merged: Dict[str, torch.Tensor] = {}
    for k, v in template.items():
        if not v.dtype.is_floating_point:
            merged[k] = v.clone()
            continue
        if any(k not in F for F in curvatures):
            merged[k] = w_bar[k]
            continue
        num = torch.zeros_like(v)
        den = torch.zeros_like(v)
        for alpha, s, F in zip(weights, expert_states, curvatures):
            aF = alpha * F[k]
            num.add_(aF * s[k])
            den.add_(aF)
        merged[k] = (num + lam * w_bar[k]) / (den + lam + _EPS)
    return merged


def _eq6_step(w_anchor: Dict[str, torch.Tensor],
              H_anchor: Dict[str, torch.Tensor],
              w_new: Dict[str, torch.Tensor],
              lam: float,
              g_anchor: Optional[Dict[str, torch.Tensor]] = None,
              ) -> Dict[str, torch.Tensor]:
    """One Eq. 6 step: update ``w_anchor`` toward ``w_new`` using curvature
    ``H_anchor`` and optional gradient ``g_anchor`` (Option Y).

    When ``g_anchor`` is None we use Option X (``g = 0``). Non-float buffers
    are copied from ``w_anchor`` unchanged.
    """
    keys = mergeable_keys(w_anchor)
    out: Dict[str, torch.Tensor] = {}
    for k, v in w_anchor.items():
        if not v.dtype.is_floating_point or k not in H_anchor or k not in w_new:
            out[k] = v.clone()
            continue
        delta_d = w_new[k] - v
        g = g_anchor[k] if (g_anchor is not None and k in g_anchor) else 0.0
        inv = 1.0 / (H_anchor[k] + lam + _EPS)
        delta_w = inv * (lam * delta_d - g)
        out[k] = v + delta_w
    return out


def _combine_curvatures(H_list: List[Dict[str, torch.Tensor]],
                        weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    """Weighted sum of diagonal curvature dicts.

    When we merge two nodes, the precision of the merged node is the
    precision sum of the children (Laplace posterior product). For Eq. 9
    nodes the weights are expert-uniform inside that node. For Eq. 6
    sequential absorption, the effective weights differ -- callers set them.
    """
    keys = list(H_list[0].keys())
    return {k: sum(w * H[k] for w, H in zip(weights, H_list))
            for k in keys}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def whc_a(expert_states: List[Dict[str, torch.Tensor]],
          curvatures: List[Dict[str, torch.Tensor]],
          lam: float,
          weights: Optional[List[float]] = None,
          ) -> Dict[str, torch.Tensor]:
    """N-expert one-shot Eq. 9 with ensemble-mean Tikhonov anchor."""
    if weights is None:
        weights = _uniform_weights(len(expert_states))
    return _eq9_merge(expert_states, curvatures, lam=lam, weights=weights)


def whc_b(expert_states: List[Dict[str, torch.Tensor]],
          curvatures: List[Dict[str, torch.Tensor]],
          fanout: int = 2,
          lam0: float = 1e-4,
          rho: float = 0.5) -> Dict[str, torch.Tensor]:
    """Hierarchical tree using Eq. 9 at each internal node.

    Level-:math:`\\ell` regularizer is :math:`\\lambda_\\ell = \\lambda_0
    \\rho^{\\ell-1}` so deeper levels trust their aggregate curvature more.
    Propagated curvature of a node is the sum of its children's curvatures.
    """
    @dataclass
    class _Node:
        state: Dict[str, torch.Tensor]
        H: Dict[str, torch.Tensor]

    level = 1
    current = [_Node(s, H) for s, H in zip(expert_states, curvatures)]
    while len(current) > 1:
        lam = lam0 * (rho ** (level - 1))
        nxt: List[_Node] = []
        for i in range(0, len(current), fanout):
            group = current[i:i + fanout]
            if len(group) == 1:
                nxt.append(group[0]); continue
            w = _uniform_weights(len(group))
            merged_state = _eq9_merge([g.state for g in group],
                                      [g.H for g in group],
                                      lam=lam, weights=w)
            merged_H = _combine_curvatures([g.H for g in group],
                                           weights=[1.0] * len(group))
            nxt.append(_Node(merged_state, merged_H))
        current = nxt
        level += 1
    return current[0].state


def whc_pair(w_a: Dict[str, torch.Tensor], H_a: Dict[str, torch.Tensor],
             w_b: Dict[str, torch.Tensor], H_b: Dict[str, torch.Tensor],
             lam: float,
             g_a: Optional[Dict[str, torch.Tensor]] = None,
             g_b: Optional[Dict[str, torch.Tensor]] = None,
             ) -> Dict[str, torch.Tensor]:
    """Symmetric two-model Eq. 6 merge.

    Apply Eq. 6 in both directions (A-as-anchor and B-as-anchor) and average
    the results. Option-Y gradients ``g_a`` / ``g_b`` are optional.

    Returns the merged state dict.
    """
    w_ab = _eq6_step(w_a, H_a, w_b, lam, g_anchor=g_a)
    w_ba = _eq6_step(w_b, H_b, w_a, lam, g_anchor=g_b)
    keys = mergeable_keys(w_ab)
    merged: Dict[str, torch.Tensor] = {}
    for k, v in w_ab.items():
        if not v.dtype.is_floating_point:
            merged[k] = v.clone()
            continue
        merged[k] = 0.5 * (w_ab[k] + w_ba[k])
    return merged


def whc_c(expert_states: List[Dict[str, torch.Tensor]],
          curvatures: List[Dict[str, torch.Tensor]],
          fanout: int = 2,
          lam0: float = 1e-4,
          rho: float = 0.5,
          grad_fn: Optional[GradFn] = None,
          ) -> Dict[str, torch.Tensor]:
    """Hierarchical tree using symmetric Eq. 6 (:func:`whc_pair`) at every
    internal node.

    Each internal node has exactly two children (we only support
    ``fanout=2`` here because :func:`whc_pair` is two-model by construction;
    a lone child is promoted unchanged). Curvature propagation and per-level
    :math:`\\lambda_\\ell` match :func:`whc_b`.

    If ``grad_fn`` is provided it is called at the leaves only (each leaf
    corresponds to a real expert). Internal nodes use Option X since the
    anchor there no longer corresponds to a specific expert.
    """
    @dataclass
    class _Node:
        state: Dict[str, torch.Tensor]
        H: Dict[str, torch.Tensor]
        leaf_idx: Optional[int] = None   # None => internal (synthetic) node

    nodes = [_Node(s, H, leaf_idx=i)
             for i, (s, H) in enumerate(zip(expert_states, curvatures))]
    level = 1
    while len(nodes) > 1:
        lam = lam0 * (rho ** (level - 1))
        nxt: List[_Node] = []
        for i in range(0, len(nodes), 2):
            group = nodes[i:i + 2]
            if len(group) == 1:
                nxt.append(group[0]); continue
            a, b = group
            g_a = grad_fn(a.state, a.leaf_idx) if (grad_fn and a.leaf_idx is not None) else None
            g_b = grad_fn(b.state, b.leaf_idx) if (grad_fn and b.leaf_idx is not None) else None
            merged_state = whc_pair(a.state, a.H, b.state, b.H,
                                    lam=lam, g_a=g_a, g_b=g_b)
            merged_H = _combine_curvatures([a.H, b.H], weights=[1.0, 1.0])
            nxt.append(_Node(merged_state, merged_H, leaf_idx=None))
        nodes = nxt
        level += 1
    return nodes[0].state


def whc_d(expert_states: List[Dict[str, torch.Tensor]],
          curvatures: List[Dict[str, torch.Tensor]],
          order: Optional[List[int]] = None,
          lam: float = 1e-4,
          grad_fn: Optional[GradFn] = None,
          ) -> Dict[str, torch.Tensor]:
    """Sequential Eq. 6 absorption.

    Start from expert ``order[0]``; then for each subsequent expert in
    ``order``, apply Eq. 6 with the current aggregate as anchor and the new
    expert as the target. Curvature is accumulated as the sum of absorbed
    experts' curvatures.

    When ``grad_fn`` is provided (Option Y), the anchor gradient is the
    empirical gradient of the *current anchor*'s loss, which is a weighted
    mixture of past-expert losses. We approximate it as the mean of the
    absorbed experts' gradients at the current point (a cheap proxy).

    When ``grad_fn`` is None (Option X), we assume ``g = 0``. This keeps
    the method fully dataless.
    """
    n = len(expert_states)
    if order is None:
        order = list(range(n))
    if len(order) != n or set(order) != set(range(n)):
        raise ValueError("order must be a permutation of range(n)")

    # Initialize with the first expert.
    w = {k: v.clone() for k, v in expert_states[order[0]].items()}
    H = {k: v.clone() for k, v in curvatures[order[0]].items()}
    absorbed = [order[0]]

    for idx in order[1:]:
        if grad_fn is None:
            g = None
        else:
            # Mean of per-absorbed-expert gradients at the current anchor.
            per_g = [grad_fn(w, j) for j in absorbed]
            g = {k: torch.stack([pg[k] for pg in per_g], 0).mean(0)
                 for k in per_g[0]}
        w = _eq6_step(w, H, expert_states[idx], lam=lam, g_anchor=g)
        H = _combine_curvatures([H, curvatures[idx]], weights=[1.0, 1.0])
        absorbed.append(idx)
    return w


# ---------------------------------------------------------------------------
# Fisher-only ablations (for isolating the anchor vs. stabilization story)
# ---------------------------------------------------------------------------

def fisher_ridge(expert_states: List[Dict[str, torch.Tensor]],
                 curvatures: List[Dict[str, torch.Tensor]],
                 eps: float,
                 weights: Optional[List[float]] = None,
                 ) -> Dict[str, torch.Tensor]:
    """Tikhonov-regularized Fisher merge anchored at the origin.

    Ablation for isolating the numerical-stabilization effect from the
    ensemble-mean anchor in WHC.
    """
    if weights is None:
        weights = _uniform_weights(len(expert_states))
    merged: Dict[str, torch.Tensor] = {}
    for k, v in expert_states[0].items():
        if not v.dtype.is_floating_point:
            merged[k] = v.clone()
            continue
        if any(k not in F for F in curvatures):
            merged[k] = torch.stack([s[k] for s in expert_states], 0).mean(0)
            continue
        num = torch.zeros_like(v)
        den = torch.zeros_like(v)
        for alpha, s, F in zip(weights, expert_states, curvatures):
            aF = alpha * F[k]
            num.add_(aF * s[k])
            den.add_(aF)
        merged[k] = num / (den + eps + _EPS)
    return merged
