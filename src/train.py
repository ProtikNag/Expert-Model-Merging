"""Training primitives: shared-init pretraining and per-expert fine-tuning."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optim: torch.optim.Optimizer,
                    device: torch.device) -> Tuple[float, float]:
    """Single pass of cross-entropy training. Returns (loss, accuracy)."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device) -> float:
    """Top-1 accuracy on the provided loader."""
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(-1) == y).sum().item()
        total += x.size(0)
    return correct / total


def fine_tune(model: nn.Module,
              loader: DataLoader,
              epochs: int,
              lr: float,
              weight_decay: float,
              device: torch.device) -> Dict[str, float]:
    """Fine-tune ``model`` in place for ``epochs`` epochs."""
    optim = torch.optim.Adam(model.parameters(), lr=lr,
                             weight_decay=weight_decay)
    history = {"loss": [], "acc": []}
    for _ in range(epochs):
        loss, acc = train_one_epoch(model, loader, optim, device)
        history["loss"].append(loss)
        history["acc"].append(acc)
    return history
