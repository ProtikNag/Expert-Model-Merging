"""Metrics computed on merged models and during the merge procedure itself.

We compute every quantity we might later want for a paper, so the analysis
stage never has to re-run the pipeline. Metrics fall into two groups:

1. **Task metrics** (accuracy, F1, Matthews correlation) -- per-task and
   aggregated across tasks.
2. **Parameter-space metrics** -- L2 distance from the merged model to the
   pretrained init, to the ensemble mean, and to each expert; cosine
   similarity; per-layer norms. These are useful for interpreting *where* a
   merging method places its solution.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Task metrics
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy."""
    return float((logits.argmax(-1) == labels).float().mean().item())


def matthews_corrcoef(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Matthews correlation coefficient for binary classification."""
    p = preds.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    tp = int(((p == 1) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def binary_f1(preds: torch.Tensor, labels: torch.Tensor,
              positive: int = 1) -> float:
    """Binary F1 with respect to ``positive``."""
    p = preds.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    tp = int(((p == positive) & (y == positive)).sum())
    fp = int(((p == positive) & (y != positive)).sum())
    fn = int(((p != positive) & (y == positive)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return float(2 * prec * rec / (prec + rec))


def task_metric(task: str,
                logits: torch.Tensor,
                labels: torch.Tensor) -> Dict[str, float]:
    """All relevant metrics for a GLUE task. The 'primary' key is the
    GLUE-standard metric used for per-task scoring; 'accuracy' is always
    reported as a second line."""
    preds = logits.argmax(-1)
    acc = accuracy(logits, labels)
    out = {"accuracy": acc}
    if task == "cola":
        out["matthews"] = matthews_corrcoef(preds, labels)
        out["primary"] = out["matthews"]
    elif task in {"mrpc", "qqp"}:
        out["f1"] = binary_f1(preds, labels)
        out["primary"] = (out["f1"] + acc) / 2.0  # GLUE-standard avg
    else:                                       # sst2, mnli, qnli, rte
        out["primary"] = acc
    return out


def aggregate_primary(per_task: Dict[str, Dict[str, float]]) -> float:
    """Average 'primary' score across tasks."""
    vals = [m["primary"] for m in per_task.values()]
    return float(sum(vals) / len(vals))


# ---------------------------------------------------------------------------
# Parameter-space metrics
# ---------------------------------------------------------------------------

def _flat(state: Dict[str, torch.Tensor],
          keys: Iterable[str]) -> torch.Tensor:
    return torch.cat([state[k].detach().reshape(-1) for k in keys])


def mergeable_keys(state: Dict[str, torch.Tensor]) -> List[str]:
    return [k for k, v in state.items() if v.dtype.is_floating_point]


def l2_distance(s1: Dict[str, torch.Tensor],
                s2: Dict[str, torch.Tensor]) -> float:
    keys = mergeable_keys(s1)
    v1, v2 = _flat(s1, keys), _flat(s2, keys)
    return float(torch.norm(v1 - v2).item())


def cosine_similarity(s1: Dict[str, torch.Tensor],
                      s2: Dict[str, torch.Tensor]) -> float:
    keys = mergeable_keys(s1)
    v1, v2 = _flat(s1, keys), _flat(s2, keys)
    denom = torch.norm(v1) * torch.norm(v2)
    if denom.item() == 0.0:
        return 0.0
    return float((v1 @ v2 / denom).item())


def param_space_summary(merged: Dict[str, torch.Tensor],
                        pretrained: Dict[str, torch.Tensor],
                        experts: List[Dict[str, torch.Tensor]],
                        ) -> Dict[str, float]:
    """Compute a fixed set of distances that characterizes *where* the merged
    model sits in parameter space relative to the landmarks."""
    keys = mergeable_keys(merged)
    ensemble_mean = {k: torch.stack([e[k] for e in experts], 0).mean(0)
                     for k in keys}
    out: Dict[str, float] = {
        "l2_to_pretrained": l2_distance(merged, pretrained),
        "l2_to_ensemble_mean": l2_distance(merged, ensemble_mean),
        "cos_to_pretrained": cosine_similarity(merged, pretrained),
        "cos_to_ensemble_mean": cosine_similarity(merged, ensemble_mean),
    }
    per_expert = [l2_distance(merged, e) for e in experts]
    out["l2_to_experts_mean"] = float(np.mean(per_expert))
    out["l2_to_experts_min"] = float(np.min(per_expert))
    out["l2_to_experts_max"] = float(np.max(per_expert))
    out["l2_to_experts_std"] = float(np.std(per_expert))
    return out


# ---------------------------------------------------------------------------
# Fisher / curvature statistics
# ---------------------------------------------------------------------------

def curvature_stats(curvatures: List[Dict[str, torch.Tensor]]
                    ) -> Dict[str, float]:
    """Summary stats of a list of curvature dicts: distribution over their
    values. Useful for diagnosing 'is the Fisher dominated by a few layers'
    and 'how does the task-vector proxy compare to true Fisher'."""
    flats = []
    for F in curvatures:
        flats.append(torch.cat([v.reshape(-1) for v in F.values()]))
    stacked = torch.stack(flats, dim=0)
    return {
        "global_mean": float(stacked.mean().item()),
        "global_median": float(stacked.median().item()),
        "global_max": float(stacked.max().item()),
        "global_min": float(stacked.min().item()),
        "global_std": float(stacked.std().item()),
        "per_expert_mean": [float(f.mean().item()) for f in flats],
        "per_expert_nnz_frac": [float((f > 1e-12).float().mean().item())
                                for f in flats],
    }
