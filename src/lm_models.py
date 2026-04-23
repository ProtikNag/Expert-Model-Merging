"""Encoder-backbone expert model with a per-task classification head.

We deliberately separate the backbone state dict (shared across experts and
the target of merging) from the task-specific head (never merged; kept per
task and reattached at evaluation time).

Compatible with any HuggingFace encoder model that exposes
``AutoModel.from_pretrained``. We default to RoBERTa-base.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


@dataclass
class ExpertConfig:
    model_name: str
    num_labels: int
    head_dropout: float = 0.1


class EncoderClassifier(nn.Module):
    """Transformer encoder + CLS-pooled linear head."""

    def __init__(self, cfg: ExpertConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(cfg.head_dropout)
        self.head = nn.Linear(hidden, cfg.num_labels)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask)
        # Use the first-token ([CLS] / <s>) representation.
        pooled = out.last_hidden_state[:, 0]
        return self.head(self.dropout(pooled))

    # ---- separation of concerns -----------------------------------------

    def backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return the backbone's state dict (what merging operates on)."""
        return {k: v.detach().clone()
                for k, v in self.backbone.state_dict().items()}

    def head_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone()
                for k, v in self.head.state_dict().items()}

    def load_backbone_state_dict(self,
                                 sd: Dict[str, torch.Tensor],
                                 strict: bool = True) -> None:
        self.backbone.load_state_dict(sd, strict=strict)

    def load_head_state_dict(self,
                             sd: Dict[str, torch.Tensor],
                             strict: bool = True) -> None:
        self.head.load_state_dict(sd, strict=strict)


def build_encoder_classifier(model_name: str,
                             num_labels: int,
                             head_dropout: float = 0.1) -> EncoderClassifier:
    return EncoderClassifier(ExpertConfig(
        model_name=model_name,
        num_labels=num_labels,
        head_dropout=head_dropout,
    ))


def pretrained_backbone_state_dict(model_name: str
                                   ) -> Dict[str, torch.Tensor]:
    """Load the pretrained backbone once and return its state dict."""
    backbone = AutoModel.from_pretrained(model_name)
    return {k: v.detach().clone() for k, v in backbone.state_dict().items()}
