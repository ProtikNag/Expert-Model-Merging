"""Fisher-weighted Averaging (Matena & Raffel, 2022).

Per-parameter elementwise update using the diagonal empirical Fisher:

.. math::
    \\theta^\\star_k = \\frac{\\sum_i F_{i,k}\\, \\theta_{i,k}}
                              {\\sum_i F_{i,k}}
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

_EPS = 1e-12


def fisher_merge(expert_states: List[Dict[str, torch.Tensor]],
                 fishers: List[Dict[str, torch.Tensor]],
                 weights: Optional[List[float]] = None,
                 ) -> Dict[str, torch.Tensor]:
    """Fisher-weighted parameter merge.

    Parameters
    ----------
    expert_states:
        Fine-tuned expert state dicts.
    fishers:
        One diagonal Fisher per expert, keyed by parameter name. Non-parameter
        buffers (e.g. BatchNorm ``running_mean``) are allowed to be absent.
    weights:
        Optional per-expert scalars :math:`\\alpha_i`. Default: uniform.
    """
    n = len(expert_states)
    if weights is None:
        weights = [1.0 / n] * n
    merged: Dict[str, torch.Tensor] = {}
    for k, v in expert_states[0].items():
        if not v.dtype.is_floating_point:
            merged[k] = v.clone()
            continue
        if any(k not in f for f in fishers):
            # Non-grad buffer (e.g. BN running stats): fall back to mean.
            merged[k] = torch.stack([s[k] for s in expert_states],
                                    dim=0).mean(dim=0)
            continue
        num = torch.zeros_like(v)
        den = torch.zeros_like(v)
        for alpha, state, fish in zip(weights, expert_states, fishers):
            w = alpha * fish[k]
            num.add_(w * state[k])
            den.add_(w)
        merged[k] = num / (den + _EPS)
    return merged
