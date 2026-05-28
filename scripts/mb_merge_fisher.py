"""Run the two Fisher-curvature merges for the data-tier comparison.

Requires per-expert diagonal Fisher computed first by
scripts/mb_fisher_estimate.py, laid out as
``<fisher_root>/<domain>/model.safetensors`` (one per merged domain). Produces:

  - fisher_merge     : plain Fisher-weighted average (Matena & Raffel 2022).
  - whc_diag_fisher  : our anchored whc_diag with TRUE Fisher curvature (the
                       ablation isolating the dataless task-vector proxy).

Both run through the same memory-bounded path as the other merges, so they are
directly comparable. CPU-only (no GPU needed for the merge itself).

Usage
-----
    python -u scripts/mb_merge_fisher.py --config configs/mergebench.yaml \
        --fisher-root mb_fisher/gemma-2-2b --domains math,coding
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mergebench.llm_merge import merge_checkpoints  # noqa: E402
from scripts.mb_download import expert_repo, local_dir_for  # noqa: E402
from src.utils import ensure_dir, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fisher-root", default="mb_fisher/gemma-2-2b")
    ap.add_argument("--domains", default=None,
                    help="Comma-separated; defaults to cfg tier_domains.")
    ap.add_argument("--lam", type=float, default=1e-4,
                    help="Anchor coefficient for whc_diag_fisher.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    download_dir = Path(cfg["download_dir"])
    domains = (args.domains.split(",") if args.domains else cfg["tier_domains"])
    base_dir = str(local_dir_for(download_dir, cfg["base_model"]))
    expert_dirs = [str(local_dir_for(download_dir,
                                     expert_repo(cfg["base_name"], d)))
                   for d in domains]
    fisher_dirs = [str(Path(args.fisher_root) / d) for d in domains]
    merged_root = ensure_dir(Path(cfg["paths"]["merged"]) / cfg["base_name"])

    for d in fisher_dirs:
        if not (Path(d) / "model.safetensors").exists():
            raise FileNotFoundError(
                f"missing Fisher at {d}/model.safetensors — run "
                f"scripts/mb_fisher_estimate.py for each domain first.")

    print(f"[fisher-merge] domains={domains} fisher_dirs={fisher_dirs}", flush=True)

    # 1) plain Fisher-weighted average (precursor baseline).
    merge_checkpoints(method="fisher_merge", base_dir=base_dir,
                      expert_dirs=expert_dirs,
                      save_dir=str(merged_root / "fisher_merge"),
                      fisher_dirs=fisher_dirs)

    # 2) our anchored merge with true Fisher (proxy-vs-Fisher ablation).
    merge_checkpoints(method="whc_diag", base_dir=base_dir,
                      expert_dirs=expert_dirs,
                      save_dir=str(merged_root / "whc_diag_fisher"),
                      lam=args.lam, curvature="fisher", fisher_dirs=fisher_dirs)

    print("[fisher-merge] done: fisher_merge + whc_diag_fisher", flush=True)


if __name__ == "__main__":
    main()
