"""Weighted Hessian Consolidation (WHC) — binary-tree pairwise merging.

Implements the function-matching closed-form merge from docs/whc.pdf §6
Eq. (26),

    w_M = (sum_i α_i G_i + λ I)^{-1} (sum_i α_i G_i w_i + λ w̄),

specialized to N = 2 at every internal node of a binary tree built over
the expert models. Per-linear-layer input Grams G_i = X_i^T X_i are the
curvature metric (Eq. (23) Kronecker shortcut), so the d×d
parameter-space Gauss-Newton kernel is carried implicitly by the
m×m input Gram.

For non-linear parameters (biases, LayerNorm γ/β, position embeddings,
1-d affine weights) Eq. (26) does not apply layer-locally, so we fall
back to a uniform pairwise average.

Internal-node Grams accumulate as the α-weighted sum of their two
children (sum-of-precisions / additive Gauss-Newton update), so deeper
levels of the tree carry larger curvature mass.

Hyperparameter λ is the Tikhonov ridge that doubles as an
ensemble-mean anchor: λ → 0 reduces to RegMean-style functional
matching (no anchor), λ → ∞ reduces to plain pairwise averaging.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Pairwise primitive (one internal node)
# ---------------------------------------------------------------------------

def _pair_merge_layer(W_a: torch.Tensor, G_a: torch.Tensor,
                      W_b: torch.Tensor, G_b: torch.Tensor,
                      lam: float,
                      alpha_a: float = 0.5,
                      alpha_b: float = 0.5) -> torch.Tensor:
    """Apply Eq. (26) to one linear-layer weight matrix.

    Args:
        W_a, W_b: ``[out, in]`` expert weight matrices.
        G_a, G_b: ``[in, in]`` symmetric input-activation Grams.
        lam: Tikhonov ridge / ensemble-anchor strength.
        alpha_a, alpha_b: nonnegative mixing weights, default uniform.

    Returns:
        ``[out, in]`` merged weight matrix.

    With ensemble anchor :math:`\\bar W = \\alpha_a W_a + \\alpha_b W_b`,

    .. math::
        W_M = \\big(\\alpha_a W_a G_a + \\alpha_b W_b G_b + \\lambda \\bar W\\big)
              \\big(\\alpha_a G_a + \\alpha_b G_b + \\lambda I\\big)^{-1}.
    """
    in_dim = W_a.shape[1]
    G_sum = alpha_a * G_a + alpha_b * G_b
    W_bar = alpha_a * W_a + alpha_b * W_b
    rhs = alpha_a * (W_a @ G_a) + alpha_b * (W_b @ G_b) + lam * W_bar
    lhs = G_sum + (lam + _EPS) * torch.eye(
        in_dim, dtype=G_sum.dtype, device=G_sum.device,
    )
    # G_sum is symmetric → lhs.T == lhs. Solve lhs X^T = rhs^T, then transpose.
    return torch.linalg.solve(lhs, rhs.T).T


def _pair_merge_state(state_a: Dict[str, torch.Tensor],
                      grams_a: Dict[str, torch.Tensor],
                      state_b: Dict[str, torch.Tensor],
                      grams_b: Dict[str, torch.Tensor],
                      lam: float,
                      alpha_a: float = 0.5,
                      alpha_b: float = 0.5,
                      ) -> Tuple[Dict[str, torch.Tensor],
                                 Dict[str, torch.Tensor]]:
    """Merge two nodes ``(state, grams)``.

    Linear-weight parameters keyed by ``grams_*`` are merged via Eq. (26);
    every other floating-point parameter is averaged with the same
    ``alpha`` weights. Non-float buffers are copied from node ``a``.

    Returns:
        ``(merged_state, merged_grams)``. ``merged_grams`` is the
        α-weighted sum of the two child Grams, used as the curvature for
        the next level of the tree.
    """
    merged_state: Dict[str, torch.Tensor] = {}
    merged_grams: Dict[str, torch.Tensor] = {}
    linear_keys = set(grams_a.keys()) & set(grams_b.keys())

    for k, v in state_a.items():
        if not v.dtype.is_floating_point:
            merged_state[k] = v.clone()
            continue
        if k in linear_keys and v.dim() == 2:
            merged_state[k] = _pair_merge_layer(
                v, grams_a[k], state_b[k], grams_b[k],
                lam=lam, alpha_a=alpha_a, alpha_b=alpha_b,
            )
        else:
            merged_state[k] = alpha_a * v + alpha_b * state_b[k]

    for k in linear_keys:
        merged_grams[k] = alpha_a * grams_a[k] + alpha_b * grams_b[k]

    return merged_state, merged_grams


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def whc_tree(expert_states: List[Dict[str, torch.Tensor]],
             grams_list: List[Dict[str, torch.Tensor]],
             lam: float,
             order: Optional[List[int]] = None,
             ) -> Dict[str, torch.Tensor]:
    """Pairwise binary-tree merge of N expert models.

    The tree is built bottom-up. At each level, adjacent nodes are paired
    and merged via :func:`_pair_merge_state`; if the level has odd count,
    the last node is carried up to the next level unchanged. The process
    repeats until a single root remains.

    Args:
        expert_states: per-expert backbone state dicts, length ``N``.
        grams_list: per-expert input-Gram dicts, length ``N``. Keys are
            linear-weight parameter names (matching the keys produced by
            :func:`src.merging.regmean.collect_linear_grams`).
        lam: Tikhonov ridge / ensemble-anchor strength. The single
            hyperparameter to sweep.
        order: optional permutation of ``range(N)``. Pairing follows this
            ordering left-to-right at every level. Default is the
            natural index order from ``expert_states``.

    Returns:
        The merged backbone state dict (root of the tree), ready to load
        into the model via ``load_backbone_state_dict``.
    """
    n = len(expert_states)
    if n != len(grams_list):
        raise ValueError("expert_states and grams_list must have equal length.")
    if n == 0:
        raise ValueError("Need at least one expert.")
    if order is None:
        order = list(range(n))
    if sorted(order) != list(range(n)):
        raise ValueError("`order` must be a permutation of range(len(experts)).")

    @dataclass
    class _Node:
        state: Dict[str, torch.Tensor]
        grams: Dict[str, torch.Tensor]

    nodes: List[_Node] = [
        _Node(expert_states[i], grams_list[i]) for i in order
    ]

    while len(nodes) > 1:
        nxt: List[_Node] = []
        for i in range(0, len(nodes), 2):
            if i + 1 == len(nodes):
                # Odd tail: promote unchanged to the next level.
                nxt.append(nodes[i])
                continue
            a, b = nodes[i], nodes[i + 1]
            merged_state, merged_grams = _pair_merge_state(
                a.state, a.grams, b.state, b.grams, lam=lam,
            )
            nxt.append(_Node(merged_state, merged_grams))
        nodes = nxt

    return nodes[0].state
