"""RegMean (Jin et al., ICLR 2023).

For every linear layer with weight :math:`W_i` and activation Gram
:math:`G_i = X_i^\\top X_i`, RegMean solves the closed-form merge

.. math::
    W_M = \\Big(\\sum_i \\hat G_i\\Big)^{-1} \\sum_i \\hat G_i W_i,

where :math:`\\hat G_i = \\alpha G_i + (1 - \\alpha) \\mathrm{diag}(G_i)` is
an off-diagonal-attenuated Gram. Biases and non-linear parameters are averaged.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_EPS = 1e-6


def collect_linear_grams(model: nn.Module,
                         loader: DataLoader,
                         device: torch.device,
                         n_samples: int) -> Dict[str, torch.Tensor]:
    """Collect ``X^T X`` for every ``nn.Linear`` layer, keyed by weight name.

    We register forward hooks on every Linear module, stream inputs through
    the model, and accumulate the Gram matrix of the input features.
    """
    grams: Dict[str, torch.Tensor] = {}
    handles = []
    # Map module id -> canonical parameter key (e.g. "classifier.1.weight").
    name_by_id: Dict[int, str] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            name_by_id[id(module)] = f"{name}.weight"

    def make_hook(mod_id: int):
        def hook(module, inputs, output):
            x = inputs[0].detach()
            if x.dim() > 2:
                x = x.reshape(-1, x.size(-1))
            key = name_by_id[mod_id]
            gram = x.t() @ x
            if key in grams:
                grams[key].add_(gram)
            else:
                grams[key] = gram
        return hook

    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(make_hook(id(module))))

    model.eval()
    count = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            model(x)
            count += x.size(0)
            if count >= n_samples:
                break
    for h in handles:
        h.remove()
    if count == 0:
        raise RuntimeError("No samples consumed for Gram collection.")
    for k in grams:
        grams[k].div_(count)
    return grams


def regmean_merge(expert_states: List[Dict[str, torch.Tensor]],
                  grams_list: List[Dict[str, torch.Tensor]],
                  alpha: float) -> Dict[str, torch.Tensor]:
    """Merge experts layer-wise via RegMean on Linear weights.

    Non-Linear parameters (biases, conv weights, etc.) fall back to simple
    averaging. This matches the paper's treatment: only dense matrices
    get the closed-form regression merge.
    """
    merged: Dict[str, torch.Tensor] = {}
    linear_weight_keys = set(grams_list[0].keys())
    for k, v in expert_states[0].items():
        if not v.dtype.is_floating_point:
            merged[k] = v.clone()
            continue
        if k in linear_weight_keys:
            # Shapes: W_i is [out_dim, in_dim], G_i is [in_dim, in_dim].
            in_dim = expert_states[0][k].shape[1]
            sum_G = torch.zeros(in_dim, in_dim,
                                dtype=v.dtype, device=v.device)
            sum_GW = torch.zeros_like(expert_states[0][k])
            for state, grams in zip(expert_states, grams_list):
                G = grams[k]
                # Off-diagonal attenuation.
                if alpha < 1.0:
                    G_hat = alpha * G + (1.0 - alpha) * torch.diag(torch.diag(G))
                else:
                    G_hat = G
                W = state[k]                      # [out, in]
                sum_G.add_(G_hat)
                sum_GW.add_(W @ G_hat)            # [out, in]
            # Solve W_M G_sum^T = sum_GW, i.e. W_M = sum_GW @ G_sum^{-1}.
            # G_sum is symmetric, so G_sum^T == G_sum.
            reg = sum_G + _EPS * torch.eye(in_dim, device=sum_G.device,
                                           dtype=sum_G.dtype)
            merged[k] = torch.linalg.solve(reg.T, sum_GW.T).T
        else:
            merged[k] = torch.stack([s[k] for s in expert_states],
                                    dim=0).mean(dim=0)
    return merged
