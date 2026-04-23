"""Task Arithmetic (Ilharco et al., ICLR 2023).

theta_merged = theta_pre + scale * sum_i (theta_i - theta_pre).
"""
from __future__ import annotations

from typing import Dict, List

import torch


def task_arithmetic(expert_states: List[Dict[str, torch.Tensor]],
                    pretrained_state: Dict[str, torch.Tensor],
                    scale: float) -> Dict[str, torch.Tensor]:
    """Merge experts via a scaled sum of task vectors.

    Parameters
    ----------
    expert_states:
        Fine-tuned expert state dicts.
    pretrained_state:
        Shared initialization from which each expert was fine-tuned.
    scale:
        Coefficient applied to the sum of task vectors (commonly 0.3-0.5).
    """
    merged: Dict[str, torch.Tensor] = {}
    for k, theta_pre in pretrained_state.items():
        if not theta_pre.dtype.is_floating_point:
            merged[k] = theta_pre.clone()
            continue
        tv_sum = torch.zeros_like(theta_pre)
        for s in expert_states:
            tv_sum.add_(s[k] - theta_pre)
        merged[k] = theta_pre + scale * tv_sum
    return merged
