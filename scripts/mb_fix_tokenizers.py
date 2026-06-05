"""Copy the base tokenizer into the MergeBench baseline merge dirs.

Our own merges (``task_arith``, ``whc_diag``) already copy the base tokenizer via
``save_merged``. MergeBench's baseline ``Merger.save`` instead re-serializes
``tokenizer.json`` with a newer ``tokenizers`` lib than the eval envs can parse
("data did not match any variant of untagged enum ModelWrapper"). The base
tokenizer is identical and loads cleanly, so we overwrite the baseline dirs'
tokenizer files with the base ones. lm-eval can override the tokenizer with a
flag, but bigcode and safety-eval (vLLM) load it from the model dir, so the fix
must live on disk.

Run once after merging, before the bigcode / safety evals::

    python scripts/mb_fix_tokenizers.py --config configs/mergebench_tier2.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mergebench.io_utils import copy_aux_files  # noqa: E402
from scripts.mb_download import local_dir_for  # noqa: E402
from src.utils import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    base_dir = local_dir_for(Path(cfg["download_dir"]), cfg["base_model"])
    merged_root = Path(cfg["paths"]["merged"]) / cfg["base_name"]

    # Only the MergeBench baseline dirs need the fix; our own dirs are clean.
    baseline_methods = list(cfg.get("baselines_dataless", {}).keys())
    print(f"[fix-tok] base={base_dir} -> baseline merge dirs under {merged_root}",
          flush=True)
    for name in baseline_methods:
        dst = merged_root / name
        if not dst.exists():
            print(f"  [skip] {name}: {dst} not found (merge not run?)", flush=True)
            continue
        copy_aux_files(base_dir, dst)
        print(f"  [ok] copied base tokenizer/config -> {dst}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
