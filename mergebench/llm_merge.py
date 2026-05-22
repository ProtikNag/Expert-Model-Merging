"""Parameter-space merging of large HuggingFace causal LMs.

Three merge rules, all elementwise and memory-bounded (one parameter key at
a time), so they scale to billion-parameter models:

- ``simple``      : ensemble mean of the experts.
- ``task_arith``  : w_pre + scale * sum_i (w_i - w_pre).
- ``whc_diag``    : the diagonal WHC closed form,

      w_M = (sum_i F_i * w_i + lam * w_bar) / (sum_i F_i + lam),

  with per-parameter curvature F_i. With ``curvature="taskvec"`` (default,
  dataless) F_i = (w_i - w_pre)^2; with ``curvature="fisher"`` the caller
  supplies a diagonal Fisher per expert. As lam -> inf the result tends to
  the simple mean; as lam -> 0 it tends to the curvature-weighted mean. This
  is the N-expert closed form (CRL Eq. 9) specialized to diagonal curvature
  with an ensemble-mean anchor.

Each routine streams over the base model's keys, merges float parameters
whose shapes agree across base and all experts, and copies the base tensor
for everything else (e.g. vocab-sized embeddings that differ across
finetunes, integer buffers). Merge arithmetic is done in float32 and the
result is cast back to the base tensor's dtype (typically bfloat16).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .io_utils import ShardedStateReader, save_merged

_EPS = 1e-12


def _shapes_agree(key: str,
                  base: ShardedStateReader,
                  experts: List[ShardedStateReader]) -> bool:
    """True if ``key`` exists in every expert with the base's shape."""
    base_shape = base.get(key).shape
    for e in experts:
        if not e.has(key) or e.get(key).shape != base_shape:
            return False
    return True


def _is_float(t: torch.Tensor) -> bool:
    return t.dtype.is_floating_point


def merge_checkpoints(method: str,
                      base_dir: str,
                      expert_dirs: List[str],
                      save_dir: str,
                      *,
                      scale: float = 0.4,
                      lam: float = 1e-4,
                      curvature: str = "taskvec",
                      fisher_dirs: Optional[List[str]] = None,
                      log_every: int = 50) -> Dict[str, float]:
    """Merge ``expert_dirs`` into a single HF checkpoint at ``save_dir``.

    Parameters
    ----------
    method:
        One of ``"simple"``, ``"task_arith"``, ``"whc_diag"``.
    base_dir:
        Local path to the shared pretrained base checkpoint.
    expert_dirs:
        Local paths to the fine-tuned expert checkpoints.
    save_dir:
        Output directory for the merged HF model.
    scale:
        Task-arithmetic scaling coefficient (``task_arith`` only).
    lam:
        Tikhonov / anchor coefficient (``whc_diag`` only).
    curvature:
        ``"taskvec"`` (dataless squared task vector) or ``"fisher"``
        (``whc_diag`` only).
    fisher_dirs:
        When ``curvature="fisher"``, local paths to per-expert diagonal
        Fisher checkpoints (same key layout as the models).
    log_every:
        Print progress every this many merged keys.

    Returns
    -------
    Dict[str, float]
        Summary counters (merged vs. copied keys).
    """
    base = ShardedStateReader(base_dir)
    experts = [ShardedStateReader(d) for d in expert_dirs]
    fishers = ([ShardedStateReader(d) for d in fisher_dirs]
               if (method == "whc_diag" and curvature == "fisher"
                   and fisher_dirs is not None) else None)
    n = len(experts)

    keys = base.keys()
    merged: Dict[str, torch.Tensor] = {}
    n_merged, n_copied = 0, 0

    for idx, key in enumerate(keys):
        base_t = base.get(key)
        # Non-float or shape-mismatched params: copy the base tensor as-is.
        if not _is_float(base_t) or not _shapes_agree(key, base, experts):
            merged[key] = base_t.clone()
            n_copied += 1
        else:
            out_dtype = base_t.dtype
            w_pre = base_t.float()
            w_experts = [e.get(key).float() for e in experts]

            if method == "simple":
                acc = torch.zeros_like(w_pre)
                for w in w_experts:
                    acc += w
                out = acc / n

            elif method == "task_arith":
                tv_sum = torch.zeros_like(w_pre)
                for w in w_experts:
                    tv_sum += (w - w_pre)
                out = w_pre + scale * tv_sum

            elif method == "whc_diag":
                if fishers is not None:
                    curv = [f.get(key).float() for f in fishers]
                else:
                    curv = [(w - w_pre) ** 2 for w in w_experts]
                w_bar = torch.zeros_like(w_pre)
                for w in w_experts:
                    w_bar += w
                w_bar /= n
                num = lam * w_bar
                den = torch.full_like(w_pre, lam)
                for f_i, w_i in zip(curv, w_experts):
                    num += f_i * w_i
                    den += f_i
                out = num / (den + _EPS)

            else:
                raise ValueError(f"Unknown merge method: {method}")

            merged[key] = out.to(out_dtype)
            n_merged += 1

        if (idx + 1) % log_every == 0 or (idx + 1) == len(keys):
            print(f"  [{method}] {idx + 1}/{len(keys)} keys "
                  f"(merged={n_merged}, copied={n_copied})", flush=True)

    save_merged(merged, save_dir, aux_src_dir=base_dir)
    print(f"  [{method}] saved merged checkpoint -> {save_dir}", flush=True)
    return {"n_merged": float(n_merged), "n_copied": float(n_copied),
            "n_experts": float(n)}
