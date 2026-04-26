"""Weighted Hessian Consolidation (WHC) — binary-tree pairwise merging.

Implements the function-matching closed-form merge from docs/whc.pdf §6
Eq. (27) / §8 Eq. (29),

    w_M = (sum_i α_i M_i + λI)^{-1}
          (sum_i α_i M_i w_i  -  β sum_i α_i g_i  +  λ w̄),

with M_i = β H_i + (1 - β) G_i and G_i the per-linear-layer activation
Gram (Eq. 23, the Kronecker shortcut). The implementation specializes
the formula to N = 2 at every internal node of a balanced binary tree
built over the expert models.

Three knobs implement the three "fixes" from §7.2:

- ``gamma`` (Fisher-anchored ridge): replaces λI by λI + γ diag(F̄_in),
  where F̄_in is the input-dimension projection of the diagonal Fisher.
  This injects curvature information that pure G-based merging
  (RegMean) discards.
- ``beta`` (gradient correction): retains the -β α_i g_i term from
  Eq. (27). Honest about the fact that experts are not at the optimum.
- ``K`` + ``recollect_grams_fn`` (iterative re-linearization): redoes
  the tree merge K extra times, refreshing G_i at the *current merged
  weight* between rounds. This is whc.pdf §7.2 Fix 3 in practice.

For non-linear parameters (biases, LayerNorm γ/β, position embeddings)
the Eq. (26) layer-wise solve does not apply, so we fall back to an
α-weighted average and apply the gradient correction directly:
``w_merge = α_a(w_a - β g_a) + α_b(w_b - β g_b)``.

Internal-node curvature accumulates as the α-weighted sum of children
(sum-of-precisions / additive Gauss-Newton update).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

_EPS = 1e-6

# A callback that takes a merged backbone state dict and returns the per-expert
# Gram dicts re-collected on that state. Used for iterative re-linearization.
RecollectGramsFn = Callable[
    [Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]],
]


# ---------------------------------------------------------------------------
# Node container
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    """One node in the merge tree.

    ``state``    : full backbone state dict at this node.
    ``grams``    : per-linear-layer input Gram (key -> [in, in]).
    ``fish_in``  : optional per-linear-layer input-dim Fisher (key -> [in]).
    ``grad``     : optional gradient at expert weights (key -> same shape
                   as state[key]). ``None`` for internal nodes when no
                   children carried gradient information.
    """
    state: Dict[str, torch.Tensor]
    grams: Dict[str, torch.Tensor]
    fish_in: Optional[Dict[str, torch.Tensor]] = None
    grad: Optional[Dict[str, torch.Tensor]] = None


# ---------------------------------------------------------------------------
# Pairwise primitive
# ---------------------------------------------------------------------------

def _pair_merge_layer(W_a: torch.Tensor, G_a: torch.Tensor,
                      W_b: torch.Tensor, G_b: torch.Tensor,
                      lam: float,
                      *,
                      alpha_a: float = 0.5,
                      alpha_b: float = 0.5,
                      F_a_in: Optional[torch.Tensor] = None,
                      F_b_in: Optional[torch.Tensor] = None,
                      gamma: float = 0.0,
                      g_a: Optional[torch.Tensor] = None,
                      g_b: Optional[torch.Tensor] = None,
                      beta: float = 0.0,
                      ) -> torch.Tensor:
    """Apply Eq. (27) at one linear-layer weight matrix.

    Closed form:

    .. math::
        W_M = \\big(α_a W_a G_a + α_b W_b G_b
                    - β(α_a g_a + α_b g_b) + λ \\bar W\\big)
              \\big(α_a G_a + α_b G_b + λI + γ \\mathrm{diag}(\\bar F_{in})\\big)^{-1}.
    """
    in_dim = W_a.shape[1]
    G_sum = alpha_a * G_a + alpha_b * G_b
    W_bar = alpha_a * W_a + alpha_b * W_b
    rhs = alpha_a * (W_a @ G_a) + alpha_b * (W_b @ G_b) + lam * W_bar
    if beta != 0.0 and g_a is not None and g_b is not None:
        rhs = rhs - beta * (alpha_a * g_a + alpha_b * g_b)

    ridge_scalar = (lam + _EPS)
    eye = torch.eye(in_dim, dtype=G_sum.dtype, device=G_sum.device)
    lhs = G_sum + ridge_scalar * eye
    if gamma != 0.0 and F_a_in is not None and F_b_in is not None:
        F_avg = alpha_a * F_a_in + alpha_b * F_b_in
        lhs = lhs + gamma * torch.diag(F_avg)

    # G_sum is symmetric → lhs is symmetric. Solve lhs X^T = rhs^T, transpose.
    return torch.linalg.solve(lhs, rhs.T).T


def _pair_merge_node(a: _Node, b: _Node, lam: float,
                     *,
                     alpha_a: float = 0.5, alpha_b: float = 0.5,
                     gamma: float = 0.0, beta: float = 0.0,
                     ) -> _Node:
    """Merge two nodes via Eq. (27), returning the merged node.

    Linear weights matched by ``grams`` keys go through the closed-form
    solve. Other floating-point parameters are α-blended with the
    optional gradient correction. Curvature, Fisher, and gradient all
    propagate as α-weighted sums for the next level.
    """
    merged_state: Dict[str, torch.Tensor] = {}
    merged_grams: Dict[str, torch.Tensor] = {}
    merged_fish: Optional[Dict[str, torch.Tensor]] = None
    merged_grad: Optional[Dict[str, torch.Tensor]] = None
    linear_keys = set(a.grams.keys()) & set(b.grams.keys())

    use_fish = (a.fish_in is not None and b.fish_in is not None and gamma != 0.0)
    use_grad = (a.grad is not None and b.grad is not None and beta != 0.0)

    if a.fish_in is not None and b.fish_in is not None:
        merged_fish = {}
    if a.grad is not None and b.grad is not None:
        merged_grad = {}

    for k, v in a.state.items():
        if not v.dtype.is_floating_point:
            merged_state[k] = v.clone()
            continue
        g_a = a.grad.get(k) if (use_grad and k in a.grad) else None
        g_b = b.grad.get(k) if (use_grad and k in b.grad) else None
        if k in linear_keys and v.dim() == 2:
            F_a_in = a.fish_in.get(k) if (use_fish and k in a.fish_in) else None
            F_b_in = b.fish_in.get(k) if (use_fish and k in b.fish_in) else None
            merged_state[k] = _pair_merge_layer(
                v, a.grams[k], b.state[k], b.grams[k], lam=lam,
                alpha_a=alpha_a, alpha_b=alpha_b,
                F_a_in=F_a_in, F_b_in=F_b_in, gamma=gamma,
                g_a=g_a, g_b=g_b, beta=beta,
            )
        else:
            wa = v - beta * g_a if (g_a is not None) else v
            wb = b.state[k] - beta * g_b if (g_b is not None) else b.state[k]
            merged_state[k] = alpha_a * wa + alpha_b * wb

    for k in linear_keys:
        merged_grams[k] = alpha_a * a.grams[k] + alpha_b * b.grams[k]

    if merged_fish is not None:
        common = set(a.fish_in.keys()) & set(b.fish_in.keys())
        for k in common:
            merged_fish[k] = alpha_a * a.fish_in[k] + alpha_b * b.fish_in[k]

    if merged_grad is not None:
        common = set(a.grad.keys()) & set(b.grad.keys())
        for k in common:
            merged_grad[k] = alpha_a * a.grad[k] + alpha_b * b.grad[k]

    return _Node(merged_state, merged_grams, merged_fish, merged_grad)


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------

def _project_fisher_to_in_dim(fishers: List[Dict[str, torch.Tensor]],
                              grams_list: List[Dict[str, torch.Tensor]],
                              ) -> List[Dict[str, torch.Tensor]]:
    """For each linear-weight key in ``grams_list``, take the matching
    diagonal-Fisher entry (shape ``[out, in]``) and project to ``[in]``
    by averaging over the output dimension.

    Fisher diagonals are per-parameter, but the layer-wise ridge in Eq. (27)
    lives in input-dim space (we solve an [in, in] linear system per layer).
    Projecting to the input dimension gives a per-input-dimension prior
    strength that adds cleanly to the Tikhonov term.
    """
    out: List[Dict[str, torch.Tensor]] = []
    for fisher, grams in zip(fishers, grams_list):
        proj: Dict[str, torch.Tensor] = {}
        for k in grams:
            if k in fisher and fisher[k].dim() == 2:
                proj[k] = fisher[k].mean(dim=0)   # [out, in] -> [in]
        out.append(proj)
    return out


def _whc_tree_once(expert_states: List[Dict[str, torch.Tensor]],
                   grams_list: List[Dict[str, torch.Tensor]],
                   lam: float,
                   *,
                   fishers_in: Optional[List[Dict[str, torch.Tensor]]] = None,
                   gradients: Optional[List[Dict[str, torch.Tensor]]] = None,
                   gamma: float = 0.0,
                   beta: float = 0.0,
                   order: Optional[List[int]] = None,
                   ) -> Dict[str, torch.Tensor]:
    """Single-pass binary-tree merge."""
    n = len(expert_states)
    if order is None:
        order = list(range(n))

    nodes: List[_Node] = []
    for i in order:
        nodes.append(_Node(
            state=expert_states[i],
            grams=grams_list[i],
            fish_in=fishers_in[i] if fishers_in is not None else None,
            grad=gradients[i] if gradients is not None else None,
        ))

    while len(nodes) > 1:
        nxt: List[_Node] = []
        for i in range(0, len(nodes), 2):
            if i + 1 == len(nodes):
                nxt.append(nodes[i])
                continue
            nxt.append(_pair_merge_node(
                nodes[i], nodes[i + 1], lam=lam,
                gamma=gamma, beta=beta,
            ))
        nodes = nxt
    return nodes[0].state


def whc_tree(expert_states: List[Dict[str, torch.Tensor]],
             grams_list: List[Dict[str, torch.Tensor]],
             lam: float,
             *,
             fishers: Optional[List[Dict[str, torch.Tensor]]] = None,
             gradients: Optional[List[Dict[str, torch.Tensor]]] = None,
             gamma: float = 0.0,
             beta: float = 0.0,
             K: int = 0,
             recollect_grams_fn: Optional[RecollectGramsFn] = None,
             order: Optional[List[int]] = None,
             ) -> Dict[str, torch.Tensor]:
    """Pairwise binary-tree merge of ``N`` expert backbones.

    Args:
        expert_states: per-expert backbone state dicts, length ``N``.
        grams_list: per-expert input Gram dicts, length ``N``. Keys are
            linear-weight parameter names.
        lam: Tikhonov ridge coefficient λ.
        fishers: optional per-expert diagonal Fisher dicts (per-parameter
            shape, matching the state dicts). When provided together with
            ``gamma > 0``, injects ``γ diag(F̄_in)`` into the layer-wise
            ridge (whc.pdf Eq. 19/20 ablation).
        gradients: optional per-expert gradient-at-expert-weights dicts
            (shape matching the state dicts). When provided together with
            ``beta > 0``, applies the ``-β α_i g_i`` correction term from
            Eq. (27).
        gamma: weight on the Fisher-anchored ridge term.
        beta: weight on the gradient correction term.
        K: number of *additional* re-linearization rounds. K=0 returns
            after one tree merge. K>0 requires ``recollect_grams_fn``.
        recollect_grams_fn: callable that takes the current merged state
            and returns a fresh ``grams_list`` collected on that state.
        order: optional permutation of ``range(N)`` for the leaf order.

    Returns:
        Merged backbone state dict (root of the tree).
    """
    n = len(expert_states)
    if n != len(grams_list):
        raise ValueError("expert_states and grams_list must have equal length.")
    if n == 0:
        raise ValueError("Need at least one expert.")
    if K > 0 and recollect_grams_fn is None:
        raise ValueError("K>0 requires recollect_grams_fn.")

    fishers_in = (_project_fisher_to_in_dim(fishers, grams_list)
                  if fishers is not None else None)

    state = _whc_tree_once(
        expert_states, grams_list, lam=lam,
        fishers_in=fishers_in, gradients=gradients,
        gamma=gamma, beta=beta, order=order,
    )

    for _ in range(K):
        new_grams_list = recollect_grams_fn(state)
        # Re-project Fisher onto the new Grams' linear-weight keys
        # (key set should be identical, but re-project to be safe).
        new_fishers_in = (_project_fisher_to_in_dim(fishers, new_grams_list)
                          if fishers is not None else None)
        state = _whc_tree_once(
            expert_states, new_grams_list, lam=lam,
            fishers_in=new_fishers_in, gradients=gradients,
            gamma=gamma, beta=beta, order=order,
        )

    return state
