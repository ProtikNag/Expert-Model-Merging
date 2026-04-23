"""Small CNN used for the RotatedMNIST pilot.

Deliberately minimal: two conv blocks plus a two-layer MLP head. On CPU this
trains to competitive MNIST accuracy in seconds, keeping the pilot feasible
without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class CNNConfig:
    channels: List[int]
    hidden: int
    num_classes: int = 10
    image_size: int = 28


class SmallCNN(nn.Module):
    """Two-block CNN with a small MLP classifier head."""

    def __init__(self, cfg: CNNConfig) -> None:
        super().__init__()
        c1, c2 = cfg.channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # 28 -> 14
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # 14 -> 7
        )
        flat = c2 * (cfg.image_size // 4) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, cfg.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(cfg: dict) -> SmallCNN:
    """Instantiate a SmallCNN from a config dict."""
    return SmallCNN(CNNConfig(
        channels=list(cfg["channels"]),
        hidden=int(cfg["hidden"]),
        num_classes=int(cfg.get("num_classes", 10)),
        image_size=int(cfg.get("image_size", 28)),
    ))
