"""Training, Fisher, and Gram-collection primitives for the LM experiment.

All primitives operate on :class:`src.lm_models.EncoderClassifier`. Backbone
parameters are the merge target; the head is kept per expert and never
merged.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .lm_models import EncoderClassifier


def train_one_epoch(model: EncoderClassifier,
                    loader: DataLoader,
                    optim: torch.optim.Optimizer,
                    device: torch.device,
                    max_steps: int = -1) -> Tuple[float, float]:
    """Single epoch of cross-entropy training."""
    model.train()
    total, correct, loss_sum, n_steps = 0, 0, 0.0, 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)
        optim.zero_grad()
        logits = model(ids, mask)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optim.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.size(0)
        n_steps += 1
        if max_steps > 0 and n_steps >= max_steps:
            break
    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model: EncoderClassifier,
             loader: DataLoader,
             device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run evaluation, returning concatenated (logits, labels) tensors."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)
        all_logits.append(model(ids, mask).detach().cpu())
        all_labels.append(y.cpu())
    return torch.cat(all_logits, 0), torch.cat(all_labels, 0)


@torch.enable_grad()
def diagonal_empirical_fisher(model: EncoderClassifier,
                              loader: DataLoader,
                              device: torch.device,
                              n_samples: int
                              ) -> Dict[str, torch.Tensor]:
    """Diagonal empirical Fisher over **backbone** parameters only.

    We deliberately restrict to backbone parameters because only the backbone
    is merged; Fisher on the head is meaningless for a merged model that
    uses a different head per task. Per-sample gradients (batch size 1) are
    squared and averaged.
    """
    model.eval()
    fisher = {name: torch.zeros_like(p)
              for name, p in model.backbone.named_parameters()
              if p.requires_grad}
    count = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)
        for i in range(ids.size(0)):
            model.zero_grad(set_to_none=True)
            logits = model(ids[i:i + 1], mask[i:i + 1])
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -log_probs[0, y[i]]
            loss.backward()
            for name, p in model.backbone.named_parameters():
                if p.grad is None:
                    continue
                fisher[name].add_(p.grad.detach() ** 2)
            count += 1
            if count >= n_samples:
                break
        if count >= n_samples:
            break
    model.zero_grad(set_to_none=True)
    if count == 0:
        raise RuntimeError("No samples consumed for Fisher estimation.")
    for name in fisher:
        fisher[name].div_(count)
    return fisher


def collect_backbone_linear_grams(model: EncoderClassifier,
                                  loader: DataLoader,
                                  device: torch.device,
                                  n_samples: int
                                  ) -> Dict[str, torch.Tensor]:
    """Collect :math:`X^\\top X` for every ``nn.Linear`` in the **backbone**.

    Keys are of the form ``<module_name>.weight``, matching the state-dict
    keys of the backbone.
    """
    grams: Dict[str, torch.Tensor] = {}
    handles = []
    name_by_id: Dict[int, str] = {}
    for name, mod in model.backbone.named_modules():
        if isinstance(mod, nn.Linear):
            name_by_id[id(mod)] = f"{name}.weight"

    def make_hook(mod_id):
        def hook(mod, inputs, output):
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

    for mod in model.backbone.modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(make_hook(id(mod))))

    model.eval()
    count = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            model(ids, mask)
            count += ids.size(0)
            if count >= n_samples:
                break
    for h in handles:
        h.remove()
    if count == 0:
        raise RuntimeError("No samples consumed for Gram collection.")
    for k in grams:
        grams[k].div_(count)
    return grams


@torch.enable_grad()
def backbone_gradient_on_task(model: EncoderClassifier,
                              loader: DataLoader,
                              device: torch.device,
                              n_samples: int
                              ) -> Dict[str, torch.Tensor]:
    """Mean gradient of the expert task loss over the **backbone** params.

    Used by WHC Option Y to compute :math:`g` at the current merged point.
    Mean gradient (not per-sample squared) is what Eq. 6 calls for.
    """
    model.eval()
    grad = {name: torch.zeros_like(p)
            for name, p in model.backbone.named_parameters()
            if p.requires_grad}
    count = 0
    model.zero_grad(set_to_none=True)
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)
        logits = model(ids, mask)
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss.backward()
        count += ids.size(0)
        if count >= n_samples:
            break
    for name, p in model.backbone.named_parameters():
        if p.grad is not None:
            grad[name] = p.grad.detach().clone() / max(count, 1)
    model.zero_grad(set_to_none=True)
    return grad
