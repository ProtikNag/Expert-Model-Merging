"""GLUE data loading and tokenization for the LM merging experiment.

We target seven single-sentence / pair-classification GLUE tasks
(matching Jin et al. 2023 "Dataless Knowledge Fusion"):

    CoLA, SST-2, MRPC, QQP, MNLI, QNLI, RTE

For each task we build:

- ``train_loader``    : for fine-tuning experts and computing Fisher/Grams.
- ``val_loader``      : used for hyperparameter selection across methods.
- ``test_loader``     : used for the final single-number per-task score.

Because GLUE test labels are not public, we split the task's ``validation``
split 50/50 into val and test. This is a standard convention for merging
papers (Jin et al. do the same). The split is seeded so every method sees
exactly the same val/test examples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


TASK_INFO: Dict[str, Dict] = {
    "cola": {"keys": ("sentence", None),            "num_labels": 2,
             "primary_metric": "matthews"},
    "sst2": {"keys": ("sentence", None),            "num_labels": 2,
             "primary_metric": "accuracy"},
    "mrpc": {"keys": ("sentence1", "sentence2"),    "num_labels": 2,
             "primary_metric": "f1_acc_avg"},
    "qqp":  {"keys": ("question1", "question2"),    "num_labels": 2,
             "primary_metric": "f1_acc_avg"},
    "mnli": {"keys": ("premise", "hypothesis"),     "num_labels": 3,
             "primary_metric": "accuracy"},
    "qnli": {"keys": ("question", "sentence"),      "num_labels": 2,
             "primary_metric": "accuracy"},
    "rte":  {"keys": ("sentence1", "sentence2"),    "num_labels": 2,
             "primary_metric": "accuracy"},
}

VALIDATION_SPLIT_NAMES: Dict[str, str] = {
    "mnli": "validation_matched",    # GLUE convention: report _matched
}


@dataclass
class GlueLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    num_labels: int
    task: str


class _TokenizedDataset(Dataset):
    """Wrap a tokenized HuggingFace dataset into a plain PyTorch Dataset."""

    def __init__(self, hf_ds, key_label: str = "label") -> None:
        self.hf_ds = hf_ds
        self.key_label = key_label

    def __len__(self) -> int:
        return len(self.hf_ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.hf_ds[idx]
        item = {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"],
                                           dtype=torch.long),
            "labels": torch.tensor(row[self.key_label], dtype=torch.long),
        }
        return item


def _collate_pad(batch: List[Dict[str, torch.Tensor]],
                 pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad variable-length batches to the longest item."""
    max_len = max(x["input_ids"].size(0) for x in batch)
    out = {
        "input_ids": torch.full((len(batch), max_len), pad_id,
                                dtype=torch.long),
        "attention_mask": torch.zeros(len(batch), max_len, dtype=torch.long),
        "labels": torch.stack([x["labels"] for x in batch], 0),
    }
    for i, x in enumerate(batch):
        n = x["input_ids"].size(0)
        out["input_ids"][i, :n] = x["input_ids"]
        out["attention_mask"][i, :n] = x["attention_mask"]
    return out


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def build_glue_loaders(task: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       batch_size: int,
                       train_subset: Optional[int] = None,
                       val_subset: Optional[int] = None,
                       seed: int = 42,
                       ) -> GlueLoaders:
    """Build train/val/test loaders for ``task``.

    The validation split of GLUE is halved deterministically into val and
    test. ``train_subset`` and ``val_subset`` cap the respective sizes for
    CPU smoke tests.
    """
    if task not in TASK_INFO:
        raise ValueError(f"Unknown GLUE task: {task}")
    info = TASK_INFO[task]
    key_a, key_b = info["keys"]

    ds = load_dataset("glue", task)
    val_name = VALIDATION_SPLIT_NAMES.get(task, "validation")

    def _tokenize(batch):
        if key_b is None:
            enc = tokenizer(batch[key_a], truncation=True,
                            max_length=max_length)
        else:
            enc = tokenizer(batch[key_a], batch[key_b], truncation=True,
                            max_length=max_length)
        return enc

    train_ds = ds["train"].map(_tokenize, batched=True)
    valtest_ds = ds[val_name].map(_tokenize, batched=True)

    # Deterministic halving of validation into val / test.
    valtest_ds = valtest_ds.shuffle(seed=seed)
    n_val = len(valtest_ds) // 2
    val_ds = valtest_ds.select(range(n_val))
    test_ds = valtest_ds.select(range(n_val, len(valtest_ds)))

    if train_subset is not None and train_subset < len(train_ds):
        train_ds = train_ds.shuffle(seed=seed).select(range(train_subset))
    if val_subset is not None and val_subset < len(val_ds):
        val_ds = val_ds.select(range(val_subset))
        test_ds = test_ds.select(range(min(val_subset, len(test_ds))))

    pad_id = tokenizer.pad_token_id

    def _make_loader(hf_ds, shuffle):
        return DataLoader(_TokenizedDataset(hf_ds),
                          batch_size=batch_size, shuffle=shuffle,
                          collate_fn=lambda b: _collate_pad(b, pad_id),
                          num_workers=0)

    return GlueLoaders(
        train=_make_loader(train_ds, shuffle=True),
        val=_make_loader(val_ds, shuffle=False),
        test=_make_loader(test_ds, shuffle=False),
        num_labels=info["num_labels"],
        task=task,
    )
