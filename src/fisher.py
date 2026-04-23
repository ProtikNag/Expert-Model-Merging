"""Diagonal empirical Fisher information for neural-network parameters.

We use the *empirical* Fisher (labels taken from the data, not sampled from
the model predictive). This is the standard choice in Fisher Merging
(Matena & Raffel, 2022) and is cheaper than the true Fisher. CAMEx and MaTS
make the same choice.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.enable_grad()
def diagonal_empirical_fisher(model: nn.Module,
                              loader: DataLoader,
                              device: torch.device,
                              n_samples: int) -> Dict[str, torch.Tensor]:
    """Return a dict mapping parameter name to diagonal Fisher tensor.

    The diagonal empirical Fisher at parameter ``w`` is estimated as
    :math:`\\hat F_w = \\frac{1}{N} \\sum_{(x,y)} (\\nabla_w \\log p_w(y|x))^2`.
    Computed one example at a time so the per-sample gradients are clean.

    Parameters
    ----------
    model:
        Network whose parameters' Fisher we estimate.
    loader:
        Source of labelled data.
    device:
        Compute device.
    n_samples:
        Approximate cap on the number of data points consumed.
    """
    model.eval()
    fisher: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(p) for name, p in model.named_parameters()
        if p.requires_grad
    }
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # Process one example at a time so the squared gradient is per-sample.
        for i in range(x.size(0)):
            model.zero_grad(set_to_none=True)
            logits = model(x[i:i + 1])
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -log_probs[0, y[i]]
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                fisher[name].add_(p.grad.detach() ** 2)
            count += 1
            if count >= n_samples:
                break
        if count >= n_samples:
            break
    if count == 0:
        raise RuntimeError("No samples consumed for Fisher estimation.")
    for name in fisher:
        fisher[name].div_(count)
    model.zero_grad(set_to_none=True)
    return fisher
