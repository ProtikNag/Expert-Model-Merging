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


# Optional deps that MergeBench's eager ``merging_methods/__init__.py`` drags in
# (via LocalizeAndStitch -> localize_utils -> taskloader) but that data-free
# baselines never call at runtime. None can be installed alongside the pinned
# transformers 4.32.1 without a transformers-5 upgrade that breaks ml_env.
_DATA_PIPELINE_DEPS = ("trl", "accelerate")


def _stub_data_pipeline_deps() -> None:
    """Register placeholder modules for absent data/SFT-pipeline deps.

    MergeBench's ``merging_methods/__init__.py`` eagerly imports every merger,
    including ``LocalizeAndStitch``, whose chain pulls ``trl`` (SFTConfig,
    SFTTrainer) and ``accelerate`` (dispatch_model). That makes even data-free
    baselines (TaskArithmetic, TIES, DARE, Consensus) fail to import when those
    deps are missing. They cannot be installed alongside the pinned transformers
    here without a major-version upgrade, so for the data-free path we satisfy
    the symbol lookups with placeholders that are never actually invoked. Each
    stub module's ``__getattr__`` (PEP 562) returns a class usable as a base or
    callable; it raises only if a data-using method genuinely touches it. Data-
    using methods still require real installs in a transformers>=4.40 env.
    """
    import importlib.machinery
    import importlib.util
    import types

    for name in _DATA_PIPELINE_DEPS:
        if name in sys.modules or importlib.util.find_spec(name) is not None:
            continue  # real module available; do not shadow it

        stub = types.ModuleType(name)
        # A bare ModuleType has __spec__ = None. transformers 5's availability
        # check calls importlib.util.find_spec(name), which raises
        # "ValueError: <name>.__spec__ is None" when the module is present but
        # spec-less. Give it a real (loader-less) spec so find_spec returns it;
        # transformers then queries importlib.metadata.version(name), gets
        # PackageNotFoundError, and correctly treats the dep as unavailable.
        stub.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)

        def _make_placeholder(mod_name: str):
            class _Unavailable:  # noqa: WPS431 (local placeholder)
                """Raises only if a data-using method actually uses it."""

                def __init__(self, *args, **kwargs):
                    raise ModuleNotFoundError(
                        f"{mod_name} is not installed; this MergeBench method "
                        f"needs the data/SFT pipeline. Install it in a "
                        f"transformers>=4.40 env."
                    )

            return _Unavailable

        placeholder = _make_placeholder(name)

        def _stub_getattr(attr, _p=placeholder):
            # Dunder lookups (__file__, __path__, __spec__, ...) must NOT resolve
            # to the placeholder: import/inspect machinery probes them on every
            # module in sys.modules (e.g. torch's lazy custom_op registration
            # calls inspect.getmodule, which reads __file__ and calls
            # .endswith on it). Returning a class there breaks that walk. Raise
            # AttributeError so callers fall back to their defaults; resolve only
            # genuine symbol lookups (from X import SFTConfig) to the placeholder.
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            return _p

        stub.__getattr__ = _stub_getattr  # type: ignore[attr-defined]
        sys.modules[name] = stub


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
        # Data-free baselines never touch the SFT pipeline; stub its optional
        # deps so the package's eager __init__ import does not require them.
        _stub_data_pipeline_deps()
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
