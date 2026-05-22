"""Memory-bounded reading and writing of HuggingFace safetensors checkpoints.

The merge and divergence routines need per-parameter access to several
multi-billion-parameter models at once. Loading every model fully into RAM
does not scale, so we read tensors lazily, one parameter key at a time,
straight from the on-disk safetensors shards.

A checkpoint directory is either:
  - a single ``model.safetensors`` file, or
  - sharded ``model-00001-of-000NN.safetensors`` files plus a
    ``model.safetensors.index.json`` mapping each key to its shard.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# Auxiliary (non-weight) files copied verbatim into a merged checkpoint so
# the result is a loadable HF model directory.
_AUX_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
]


class ShardedStateReader:
    """Lazy, per-key reader over a HuggingFace safetensors checkpoint.

    Open file handles are cached per shard so repeated ``get`` calls do not
    re-open files. Only the requested tensor is materialized in memory.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        single_path = self.model_dir / "model.safetensors"
        self._weight_map: Dict[str, str]
        if index_path.exists():
            with open(index_path, "r") as f:
                self._weight_map = json.load(f)["weight_map"]
        elif single_path.exists():
            with safe_open(single_path, framework="pt", device="cpu") as f:
                self._weight_map = {k: "model.safetensors" for k in f.keys()}
        else:
            raise FileNotFoundError(
                f"No safetensors found in {self.model_dir}. Only safetensors "
                f"checkpoints are supported (found neither model.safetensors "
                f"nor an index)."
            )
        self._handles: Dict[str, object] = {}

    def keys(self) -> List[str]:
        """All parameter keys in the checkpoint."""
        return list(self._weight_map.keys())

    def _handle(self, shard: str):
        if shard not in self._handles:
            self._handles[shard] = safe_open(
                self.model_dir / shard, framework="pt", device="cpu")
        return self._handles[shard]

    def get(self, key: str) -> torch.Tensor:
        """Materialize a single tensor by key (on CPU)."""
        shard = self._weight_map[key]
        return self._handle(shard).get_tensor(key)

    def has(self, key: str) -> bool:
        return key in self._weight_map


def copy_aux_files(src_dir: str | Path, dst_dir: str | Path) -> None:
    """Copy config and tokenizer files from a source checkpoint to a merged
    checkpoint directory so the result loads as a complete HF model."""
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for name in _AUX_FILES:
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)


def save_merged(merged: Dict[str, torch.Tensor],
                save_dir: str | Path,
                aux_src_dir: str | Path) -> None:
    """Write a merged state dict as a single ``model.safetensors`` plus the
    auxiliary files copied from ``aux_src_dir`` (typically the base model)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # safetensors requires contiguous tensors and a string metadata map.
    merged = {k: v.contiguous() for k, v in merged.items()}
    save_file(merged, str(save_dir / "model.safetensors"),
              metadata={"format": "pt"})
    copy_aux_files(aux_src_dir, save_dir)
