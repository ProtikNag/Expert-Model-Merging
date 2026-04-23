"""TIES-Merging (Yadav et al., NeurIPS 2023).

Three steps on each task vector :math:`\\tau_i = \\theta_i - \\theta_\\text{pre}`:

1. **Trim.** Per task vector, keep the top-``keep_frac`` fraction of entries
   by magnitude **across the entire flattened task vector** (this is the
   paper's convention; per-tensor trimming would bias toward small tensors
   like biases). Zero out the rest.
2. **Elect sign.** For each entry, choose the sign with the larger sum of
   magnitudes across tasks.
3. **Disjoint merge.** Average only the entries whose sign agrees with the
   elected sign; normalize by the count of contributors.

Finally apply a global scale.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from ..utils import mergeable_keys


def _global_trim(tensors: List[torch.Tensor],
                 keep_frac: float) -> List[torch.Tensor]:
    """Trim ``tensors`` jointly, keeping the top ``keep_frac`` fraction by
    magnitude across the concatenated flat vector.

    Returns a new list of tensors (same shapes) with below-threshold entries
    zeroed out. This matches the official TIES-Merging implementation, where
    trimming is applied to the task vector as one blob, not per-parameter.
    """
    flat = torch.cat([t.detach().reshape(-1) for t in tensors])
    n_total = flat.numel()
    n_keep = max(1, int(keep_frac * n_total))
    if n_keep >= n_total:
        return [t.clone() for t in tensors]
    thresh = torch.topk(flat.abs(), n_keep, largest=True).values.min()
    out: List[torch.Tensor] = []
    for t in tensors:
        mask = t.abs() >= thresh
        out.append(t * mask)
    return out


def ties_merging(expert_states: List[Dict[str, torch.Tensor]],
                 pretrained_state: Dict[str, torch.Tensor],
                 keep_frac: float,
                 scale: float) -> Dict[str, torch.Tensor]:
    """Merge experts with TIES: global trim -> elect sign -> disjoint mean."""
    keys = mergeable_keys(pretrained_state)

    # Build and trim each task vector globally across parameters.
    trimmed_per_expert: List[Dict[str, torch.Tensor]] = []
    for state in expert_states:
        taus = [state[k] - pretrained_state[k] for k in keys]
        taus_trimmed = _global_trim(taus, keep_frac)
        trimmed_per_expert.append({k: t for k, t in zip(keys, taus_trimmed)})

    merged: Dict[str, torch.Tensor] = {}
    for k, theta_pre in pretrained_state.items():
        if k not in keys:
            merged[k] = theta_pre.clone()
            continue
        stacked = torch.stack([d[k] for d in trimmed_per_expert], dim=0)
        elected_sign = torch.sign(stacked.sum(dim=0))
        agree = (torch.sign(stacked) == elected_sign) & (stacked != 0)
        agree_f = agree.float()
        contrib = (stacked * agree_f).sum(dim=0)
        count = agree_f.sum(dim=0).clamp(min=1.0)
        tv = contrib / count
        merged[k] = theta_pre + scale * tv
    return merged
