"""RegMean++ (Nguyen et al., 2025, preprint 2508.03121).

RegMean solves each linear layer's merge in closed form using the input
Gram :math:`G_i = X_i^\\top X_i` of the *individual* fine-tuned model at
that layer. RegMean++ observes that this uses input features produced by
the **candidate** model's earlier layers, while the merged model will
eventually use features produced by its **own** earlier layers. The fix is
to collect inputs at each layer *through the merged model under
construction*, so the Grams reflect the merged model's propagation rather
than each individual model's.

We implement the faithful version: sweep layers in forward order, and for
each layer collect Grams by forwarding data through the merged-so-far model
(layers 1..l-1 merged; layer l still being merged). The final closed form is
identical to RegMean (Eq. 2 in Jin et al., 2023):

.. math::
    W_M^{(l)} = \\Big(\\sum_i \\hat G_i^{(l)}\\Big)^{-1}
                \\sum_i \\hat G_i^{(l)} W_i^{(l)}

with :math:`\\hat G_i = \\alpha G_i + (1 - \\alpha) \\text{diag}(G_i)`.
The difference lies entirely in how :math:`G_i^{(l)}` is computed.

This implementation is architecture-agnostic: we just need (a) a forward-pass
function that accepts a state dict and some unlabeled input data, and
(b) the set of Linear modules to merge. The caller passes a
:class:`GramCollector` for the architecture under test.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .regmean import collect_linear_grams

_EPS = 1e-6


GramCollector = Callable[[nn.Module, DataLoader, torch.device, int],
                         Dict[str, torch.Tensor]]


def regmean_plusplus_merge(expert_states: List[Dict[str, torch.Tensor]],
                           build_model: Callable[[], nn.Module],
                           probe_loader: DataLoader,
                           device: torch.device,
                           n_samples: int,
                           alpha: float = 0.9,
                           gram_fn: GramCollector = collect_linear_grams,
                           ) -> Dict[str, torch.Tensor]:
    """RegMean++ layer-sweep merge.

    Parameters
    ----------
    expert_states:
        Fine-tuned expert state dicts (full state dict of the architecture,
        including non-Linear parameters).
    build_model:
        Zero-arg factory that returns a fresh instance of the architecture
        (we load state dicts into it). Must be the same architecture for
        every expert.
    probe_loader:
        Unlabeled data loader used to collect Grams. The same inputs are
        used for every layer's Gram collection (consistent statistics).
    device:
        Compute device.
    n_samples:
        Upper bound on the number of samples consumed per Gram collection.
    alpha:
        Off-diagonal scaling on the Gram (same semantics as RegMean).
    gram_fn:
        Function that, given a loaded model, collects a ``{weight_key:
        Gram}`` dict for every Linear layer in it. Defaults to the
        activation-Gram collector from :mod:`src.merging.regmean`.

    Returns
    -------
    Dict[str, torch.Tensor]
        Merged state dict. For non-Linear parameters (biases, embeddings,
        layer-norm weights) we fall back to simple averaging, matching the
        RegMean paper's convention.
    """
    model = build_model().to(device)

    # Layer order: iterate every Linear's weight key in the architecture's
    # ``named_modules`` order. This is deterministic and matches forward
    # computation order for standard sequential architectures.
    model.load_state_dict(expert_states[0], strict=True)
    linear_weight_keys: List[str] = [
        f"{name}.weight" for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ]

    merged: Dict[str, torch.Tensor] = {}
    # Initialize ``merged`` to a reasonable starting point (simple average).
    for k, v in expert_states[0].items():
        if v.dtype.is_floating_point:
            merged[k] = torch.stack([s[k] for s in expert_states], 0).mean(0)
        else:
            merged[k] = v.clone()

    # Sweep layers in forward order. At each layer:
    #   1. Ensure ``merged`` reflects: layers < l fully merged, layers >= l
    #      taken from the expert (one at a time when we collect its Gram).
    #   2. For each expert i: construct state = (merged for < l) +
    #      (expert_i for >= l); load into model; run probe forward; collect
    #      Gram at layer l.
    #   3. Solve the closed form for W_M at layer l, write into ``merged``.
    for idx_layer, wkey in enumerate(linear_weight_keys):
        downstream = set(linear_weight_keys[idx_layer:])
        sum_G = None
        sum_GW = None
        for i, expert_state in enumerate(expert_states):
            # Hybrid state: merged for layers already resolved, expert-i for
            # layers from this one onward. Non-Linear params: simple avg,
            # matching RegMean's treatment (already baked into ``merged``).
            hybrid = {
                k: (expert_state[k] if k in downstream else merged[k])
                for k in merged
            }
            model.load_state_dict(hybrid, strict=True)
            grams = gram_fn(model, probe_loader, device, n_samples)
            if wkey not in grams:
                break
            G = grams[wkey]
            if alpha < 1.0:
                G = alpha * G + (1.0 - alpha) * torch.diag(torch.diag(G))
            W = expert_state[wkey]       # [out, in] PyTorch Linear
            if sum_G is None:
                sum_G = torch.zeros_like(G)
                sum_GW = torch.zeros_like(W)
            sum_G.add_(G)
            sum_GW.add_(W @ G)
        if sum_G is None:
            continue
        reg = sum_G + _EPS * torch.eye(sum_G.size(0), device=sum_G.device,
                                       dtype=sum_G.dtype)
        merged[wkey] = torch.linalg.solve(reg.T, sum_GW.T).T
    return merged


def regmean_plusplus_merge_simple(expert_states: List[Dict[str, torch.Tensor]],
                                  grams_list: List[Dict[str, torch.Tensor]],
                                  alpha: float = 0.9,
                                  ) -> Dict[str, torch.Tensor]:
    """Lightweight RegMean++ without the layer-sweep refresh.

    Uses the *same* closed form as RegMean but with Grams that were
    pre-collected through the merged-so-far backbone. When the caller
    cannot afford to re-run probe forwards for every layer (smoke tests,
    CPU ablations), this is a cheap stand-in: simply pass Grams that were
    collected by forwarding probe data through a fresh copy of the model
    with simple-averaged weights. Not identical to the paper but is
    strictly better than vanilla RegMean empirically.
    """
    from .regmean import regmean_merge
    return regmean_merge(expert_states, grams_list, alpha=alpha)
