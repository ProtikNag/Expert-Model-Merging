"""Tier 0: measure expert divergence on downloaded MergeBench checkpoints.

This is the cheapest go/no-go gate. It reads weights only (no GPU, no data),
computes task-vector norms and pairwise cosine similarities, prints a report,
and writes the full summary to JSON. Interpret as follows:

  - mean off-diagonal cosine near 0 and large relative drift  ->  high
    divergence; every merger (including WHC) will land near the same point.
  - structured (non-trivial) cosines  ->  room for a curvature-aware merge to
    win; proceed to Tier 1.

Example
-------
    python -u scripts/mb_tier0_divergence.py --config configs/mergebench.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mergebench.divergence import compute_divergence, format_report  # noqa: E402
from mergebench.io_utils import ShardedStateReader  # noqa: E402
from scripts.mb_download import expert_repo, local_dir_for  # noqa: E402
from src.logging_utils import RunLogger  # noqa: E402
from src.utils import ensure_dir, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--domains", default=None,
                    help="Comma-separated override of tier_domains.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    download_dir = Path(cfg["download_dir"])
    results_dir = ensure_dir(cfg["paths"]["results"])

    logger = RunLogger(experiment=f"mb_tier0_{cfg['base_name']}",
                       log_root=cfg["paths"]["logs"],
                       config_path=args.config)

    domains = (args.domains.split(",") if args.domains
               else cfg["tier_domains"])
    base_dir = str(local_dir_for(download_dir, cfg["base_model"]))
    expert_dirs = [str(local_dir_for(download_dir,
                                     expert_repo(cfg["base_name"], d)))
                   for d in domains]

    # Fail early with a clear message if a checkpoint is missing.
    for d in [base_dir] + expert_dirs:
        ShardedStateReader(d)   # raises FileNotFoundError if no safetensors

    print(f"[tier0] base={cfg['base_model']} domains={domains}", flush=True)
    div = compute_divergence(base_dir, expert_dirs, domains)
    print(format_report(div), flush=True)

    out_path = Path(results_dir) / f"divergence_{cfg['base_name']}.json"
    with open(out_path, "w") as f:
        json.dump(div, f, indent=2)
    print(f"\n[done] {out_path}", flush=True)
    logger.record(event="tier0_divergence", domains=domains,
                  mean_offdiag_cosine=div["mean_offdiag_cosine"],
                  norm_taskvec=div["norm_taskvec"],
                  relative_drift=div["relative_drift"])
    logger.close()


if __name__ == "__main__":
    main()
