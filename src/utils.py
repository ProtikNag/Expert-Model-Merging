"""Shared utilities for the model-merging pilot."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Deterministic RNG seeding across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict:
    """Parse a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create directory and return its Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def state_dict_flat_keys(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    """Stable list of keys for parameter-space operations."""
    return list(state_dict.keys())


def params_to_vector(state_dict: Dict[str, torch.Tensor],
                     keys: Iterable[str]) -> torch.Tensor:
    """Concatenate tensors at the given keys into a single flat vector."""
    return torch.cat([state_dict[k].detach().reshape(-1) for k in keys])


def vector_to_state_dict(vec: torch.Tensor,
                         template: Dict[str, torch.Tensor],
                         keys: Iterable[str]) -> Dict[str, torch.Tensor]:
    """Inverse of ``params_to_vector`` given a template state dict for shapes."""
    out: Dict[str, torch.Tensor] = {}
    offset = 0
    for k in keys:
        numel = template[k].numel()
        out[k] = vec[offset:offset + numel].reshape(template[k].shape).clone()
        offset += numel
    for k, v in template.items():
        if k not in out:
            out[k] = v.detach().clone()
    return out


def mergeable_keys(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    """Keys that participate in parameter-space merging.

    We exclude integer-typed buffers such as ``num_batches_tracked`` that appear
    in BatchNorm layers; those are not real parameters and concatenating them
    into a float vector would silently corrupt the arithmetic.
    """
    keys: List[str] = []
    for k, v in state_dict.items():
        if v.dtype.is_floating_point:
            keys.append(k)
    return keys
