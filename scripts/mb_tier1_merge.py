"""Tier 1: produce merged checkpoints on downloaded MergeBench experts.

Runs one or more merge methods (simple, task_arith, whc_diag) over the
selected domains and writes each merged model as a plain HF checkpoint under
``paths.merged/<base_name>/<method>/``. Those directories are consumed
directly by MergeBench's own ``scripts/evaluate.sh`` for the (subset) eval
that forms the Tier 1 go/no-go gate.

Examples
--------
    # all methods in the config
    python -u scripts/mb_tier1_merge.py --config configs/mergebench.yaml

    # a single method
    python -u scripts/mb_tier1_merge.py --config configs/mergebench.yaml \
        --only whc_diag
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mergebench.io_utils import ShardedStateReader  # noqa: E402
from mergebench.llm_merge import merge_checkpoints  # noqa: E402
from scripts.mb_download import expert_repo, local_dir_for  # noqa: E402
from src.logging_utils import RunLogger  # noqa: E402
from src.utils import ensure_dir, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--domains", default=None,
                    help="Comma-separated override of tier_domains.")
    ap.add_argument("--only", default=None,
                    help="Comma-separated subset of method names to run.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    download_dir = Path(cfg["download_dir"])
    results_dir = ensure_dir(cfg["paths"]["results"])
    merged_root = ensure_dir(Path(cfg["paths"]["merged"]) / cfg["base_name"])

    logger = RunLogger(experiment=f"mb_tier1_{cfg['base_name']}",
                       log_root=cfg["paths"]["logs"],
                       config_path=args.config)

    domains = (args.domains.split(",") if args.domains
               else cfg["tier_domains"])
    base_dir = str(local_dir_for(download_dir, cfg["base_model"]))
    expert_dirs = [str(local_dir_for(download_dir,
                                     expert_repo(cfg["base_name"], d)))
                   for d in domains]

    for d in [base_dir] + expert_dirs:
        ShardedStateReader(d)   # fail early on missing checkpoints

    only = set(args.only.split(",")) if args.only else None
    methods = cfg["methods"]
    print(f"[tier1] base={cfg['base_model']} domains={domains}", flush=True)

    summaries = {}
    for name, hp in methods.items():
        if only is not None and name not in only:
            continue
        save_dir = str(merged_root / name)
        print(f"\n[method={name}] hp={hp} -> {save_dir}", flush=True)
        t0 = time.time()
        summary = merge_checkpoints(
            method=name, base_dir=base_dir, expert_dirs=expert_dirs,
            save_dir=save_dir,
            scale=hp.get("scale", 0.4),
            lam=hp.get("lam", 1e-4),
            curvature=hp.get("curvature", "taskvec"),
        )
        summary["merge_time_s"] = time.time() - t0
        summary["save_dir"] = save_dir
        summaries[name] = summary
        logger.record(event="tier1_merge", method=name, domains=domains,
                      hp=hp, **summary)
        print(f"  [{name}] done in {summary['merge_time_s']:.1f}s", flush=True)

    out_path = Path(results_dir) / f"tier1_merges_{cfg['base_name']}.json"
    with open(out_path, "w") as f:
        json.dump({"domains": domains, "methods": summaries}, f, indent=2)
    print(f"\n[done] {out_path}", flush=True)
    print("\nNext: evaluate each merged dir with MergeBench's harness, e.g.\n"
          f"  bash scripts/evaluate.sh {merged_root}/whc_diag 0 "
          "results/mb_eval/whc_diag", flush=True)
    logger.close()


if __name__ == "__main__":
    main()
