"""Adapter to run MergeBench's own baseline mergers on a chosen subset.

Rather than reimplement Task Arithmetic, TIES, DARE, Consensus, RegMean,
RegMean++, or Localize-and-Stitch, we drive MergeBench's validated classes
directly. Their ``Merger(base, ft_models, save_path)`` accepts local
checkpoint directories, so we hand it the base dir and the subset of expert
dirs we want to merge, then call ``.merge(**hp)``. This keeps WHC and every
baseline on identical inputs, save format, and downstream eval.

The MergeBench repo must be cloned locally; pass its path via ``mergebench_dir``
(its ``merging`` folder is added to ``sys.path`` so ``import merging_methods``
and ``import taskloader`` resolve). Data-using baselines (RegMean, RegMean++,
Fisher) additionally need MergeBench's data/eval environment and a GPU; the
dataless baselines need only the model weights.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

# domain -> the dataset name MergeBench's taskloader expects (data-using
# methods key their per-task data on these names).
DOMAIN_TO_DATASET = {
    "instruction": "Tulu3IF",
    "math": "DartMath",
    "coding": "MagiCoder",
    "safety": "WildguardMix",
    "multilingual": "Aya",
}

# Which baselines need MergeBench's data/taskloader pipeline (and a GPU).
DATA_USING = {"RegMean", "RegMeanPlusPlus", "Fisher"}


def _ensure_on_path(mergebench_dir: str) -> None:
    merging = str(Path(mergebench_dir) / "merging")
    if merging not in sys.path:
        sys.path.insert(0, merging)


def run_baseline(algo: str,
                 base_dir: str,
                 expert_dirs: List[str],
                 domains: List[str],
                 save_dir: str,
                 mergebench_dir: str,
                 hp: Dict) -> None:
    """Instantiate and run one MergeBench baseline merger.

    Parameters
    ----------
    algo:
        MergeBench class name (e.g. ``"TaskArithmetic"``, ``"TIES"``,
        ``"DARE"``, ``"Consensus"``, ``"LocalizeAndStitch"``, ``"RegMean"``,
        ``"RegMeanPlusPlus"``).
    base_dir, expert_dirs:
        Local checkpoint directories. ``expert_dirs`` is aligned with
        ``domains``.
    domains:
        Domain names for the experts (used to build ``task_names`` for
        data-using methods).
    save_dir:
        Output directory for the merged checkpoint.
    mergebench_dir:
        Path to a local clone of the MergeBench repo.
    hp:
        Method hyperparameters passed through to ``.merge(**hp)``. For
        data-using methods we inject ``task_names`` automatically.
    """
    _ensure_on_path(mergebench_dir)
    import merging_methods  # noqa: WPS433 (resolved via sys.path)

    merge_kwargs = dict(hp)
    if algo in DATA_USING:
        # MergeBench keys per-task data on dataset names, ordered to match the
        # expert list we pass in.
        merge_kwargs["task_names"] = "-".join(DOMAIN_TO_DATASET[d]
                                              for d in domains)

    merger_cls = getattr(merging_methods, algo)
    merger = merger_cls(base_dir, expert_dirs, save_dir)
    print(f"[baseline:{algo}] merging {domains} -> {save_dir} "
          f"kwargs={merge_kwargs}", flush=True)
    merger.merge(**merge_kwargs)
    print(f"[baseline:{algo}] done", flush=True)
