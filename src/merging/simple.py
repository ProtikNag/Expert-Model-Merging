"""Simple parameter averaging (a.k.a. Model Soups, Isotropic merging)."""
from __future__ import annotations

from typing import Dict, List

import torch


def simple_average(expert_states: List[Dict[str, torch.Tensor]],
                   ) -> Dict[str, torch.Tensor]:
    """Element-wise mean across expert state dicts over float tensors only."""
    merged: Dict[str, torch.Tensor] = {}
    keys = expert_states[0].keys()
    for k in keys:
        tensors = [s[k] for s in expert_states]
        if tensors[0].dtype.is_floating_point:
            merged[k] = torch.stack(tensors, dim=0).mean(dim=0)
        else:
            # Integer buffers (e.g. BN num_batches_tracked): keep the first.
            merged[k] = tensors[0].clone()
    return merged
