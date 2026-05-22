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


def _stub_trl_if_absent() -> None:
    """Register a placeholder ``trl`` module when the real one is missing.

    MergeBench's ``merging_methods/__init__.py`` eagerly imports every merger,
    including ``LocalizeAndStitch``, whose import chain ends at
    ``from trl import SFTConfig, SFTTrainer`` (via ``taskloader``). That makes
    even data-free baselines (TaskArithmetic, TIES, DARE, Consensus) fail to
    import when ``trl`` is not installed. ``trl`` cannot be installed alongside
    the pinned ``transformers`` here without a major-version upgrade, so for the
    data-free path we satisfy the symbol lookup with stubs that are never
    actually called. Data-using methods still require a real ``trl`` install.
    """
    import importlib.util

    if importlib.util.find_spec("trl") is not None:
        return  # real trl available; do not shadow it

    import types

    stub = types.ModuleType("trl")

    class _Unavailable:  # noqa: WPS431 (local placeholder)
        """Raises only if a data-using method actually instantiates it."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "trl is not installed; this MergeBench method needs the "
                "data/SFT pipeline. Install trl in a transformers>=4.40 env."
            )

    stub.SFTConfig = _Unavailable
    stub.SFTTrainer = _Unavailable
    sys.modules["trl"] = stub


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
    if algo not in DATA_USING:
        # Data-free baselines never touch trl; stub it so the package's eager
        # __init__ import does not drag in the SFT pipeline.
        _stub_trl_if_absent()
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
