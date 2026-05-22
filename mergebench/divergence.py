"""Tier 0 divergence diagnostic for the shared-basin assumption.

Every parameter-space merger (Simple, Task Arithmetic, Fisher, RegMean, WHC)
assumes the experts live near a shared basin around the pretrained init, so
that task vectors tau_i = w_i - w_pre are small and not mutually orthogonal.
This module measures that assumption directly from weights alone, with no
forward passes and no data:

  - ||tau_i||                : how far each expert drifted from init.
  - cos(tau_i, tau_j)        : pairwise task-vector alignment.
  - ||tau_i|| / ||w_pre||    : relative drift, comparable across models.

Low pairwise cosines plus large relative drift indicate the high-divergence
regime where all mergers degrade toward the same point; structured cosines
leave room for a curvature-aware method to win.

Tensors are read one key at a time so this scales to billion-parameter
models. Statistics accumulate in float32.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from .io_utils import ShardedStateReader


def compute_divergence(base_dir: str,
                       expert_dirs: List[str],
                       expert_names: List[str],
                       log_every: int = 50) -> Dict:
    """Compute task-vector norms and pairwise cosine similarities.

    Parameters
    ----------
    base_dir:
        Local path to the shared pretrained base checkpoint.
    expert_dirs:
        Local paths to the fine-tuned expert checkpoints, aligned with
        ``expert_names``.
    expert_names:
        Human-readable labels (e.g. domain names) for reporting.
    log_every:
        Print progress every this many parameter keys.

    Returns
    -------
    Dict
        Nested summary with per-expert norms, the pairwise cosine matrix,
        and relative-drift figures.
    """
    base = ShardedStateReader(base_dir)
    experts = [ShardedStateReader(d) for d in expert_dirs]
    n = len(experts)

    # Accumulators over the flattened, concatenated parameter space.
    sq_norm_tau = [0.0] * n            # sum of tau_i^2
    dot_tau = [[0.0] * n for _ in range(n)]   # sum of tau_i . tau_j
    sq_norm_pre = 0.0                  # sum of w_pre^2

    keys = [k for k in base.keys() if base.get(k).dtype.is_floating_point]
    for idx, key in enumerate(keys):
        w_pre = base.get(key).float()
        sq_norm_pre += float((w_pre * w_pre).sum())

        # Skip keys not present with matching shape in every expert.
        taus = []
        ok = True
        for e in experts:
            if not e.has(key) or e.get(key).shape != w_pre.shape:
                ok = False
                break
            taus.append((e.get(key).float() - w_pre).reshape(-1))
        if not ok:
            continue

        for i in range(n):
            sq_norm_tau[i] += float(taus[i].dot(taus[i]))
            for j in range(i, n):
                d = float(taus[i].dot(taus[j]))
                dot_tau[i][j] += d
                if i != j:
                    dot_tau[j][i] += d

        if (idx + 1) % log_every == 0 or (idx + 1) == len(keys):
            print(f"  [divergence] {idx + 1}/{len(keys)} keys", flush=True)

    norm_tau = [s ** 0.5 for s in sq_norm_tau]
    norm_pre = sq_norm_pre ** 0.5

    cosine = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            denom = norm_tau[i] * norm_tau[j]
            cosine[i][j] = (dot_tau[i][j] / denom) if denom > 0 else 0.0

    # Mean off-diagonal cosine is the single-number summary of alignment.
    off_diag = [cosine[i][j] for i in range(n) for j in range(n) if i != j]
    mean_off_diag = sum(off_diag) / len(off_diag) if off_diag else 0.0

    return {
        "expert_names": expert_names,
        "norm_pretrained": norm_pre,
        "norm_taskvec": {expert_names[i]: norm_tau[i] for i in range(n)},
        "relative_drift": {expert_names[i]: (norm_tau[i] / norm_pre
                                             if norm_pre > 0 else 0.0)
                           for i in range(n)},
        "cosine_matrix": cosine,
        "mean_offdiag_cosine": mean_off_diag,
    }


def format_report(div: Dict) -> str:
    """Render the divergence summary as a readable text block."""
    names = div["expert_names"]
    lines = ["", "=" * 60, "TIER 0 DIVERGENCE DIAGNOSTIC", "=" * 60]
    lines.append(f"||w_pretrained|| = {div['norm_pretrained']:.4f}")
    lines.append("")
    lines.append(f"{'expert':<16}{'||tau||':>12}{'rel.drift':>12}")
    for name in names:
        lines.append(f"{name:<16}{div['norm_taskvec'][name]:>12.4f}"
                     f"{div['relative_drift'][name]:>12.4f}")
    lines.append("")
    lines.append("pairwise cos(tau_i, tau_j):")
    header = " " * 16 + "".join(f"{nm[:8]:>10}" for nm in names)
    lines.append(header)
    for i, name in enumerate(names):
        row = "".join(f"{div['cosine_matrix'][i][j]:>10.3f}"
                      for j in range(len(names)))
        lines.append(f"{name:<16}{row}")
    lines.append("")
    lines.append(f"mean off-diagonal cosine = {div['mean_offdiag_cosine']:.4f}")
    lines.append("=" * 60)
    return "\n".join(lines)
