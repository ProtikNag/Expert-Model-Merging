"""Tier 1: merge a subset of MergeBench experts with WHC and all baselines.

Produces one merged HF checkpoint per method under
``paths.merged/<base_name>/<method>/``, ready for MergeBench's eval. WHC and
our cross-check Task Arithmetic run via the tested standalone path
(mergebench/llm_merge.py); the baselines run via MergeBench's own classes
(mergebench/mb_baselines.py) so the comparison shares inputs and save format.

Tiers of methods (selectable with --tier):
  ours      : whc_diag (+ task_arith cross-check)            [weights only]
  dataless  : TaskArithmetic, TIES, DARE, Consensus, L&S     [weights only]
  data      : RegMean, RegMeanPlusPlus                       [needs data env + GPU]
  all       : ours + dataless (default; data added only if --tier data/all)

Examples
--------
    # fail-fast: our methods + dataless baselines on math+coding
    python -u scripts/mb_tier1_merge.py --config configs/mergebench.yaml

    # just WHC
    python -u scripts/mb_tier1_merge.py --config configs/mergebench.yaml --only whc_diag
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mergebench.io_utils import ShardedStateReader  # noqa: E402
from mergebench.llm_merge import merge_checkpoints  # noqa: E402
from mergebench.mb_baselines import run_baseline  # noqa: E402
from scripts.mb_download import expert_repo, local_dir_for  # noqa: E402
from src.logging_utils import RunLogger  # noqa: E402
from src.utils import ensure_dir, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--domains", default=None,
                    help="Comma-separated override of tier_domains.")
    ap.add_argument("--tier", default="all",
                    choices=["ours", "dataless", "data", "all"],
                    help="Which group(s) of methods to run.")
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

    domains = (args.domains.split(",") if args.domains else cfg["tier_domains"])
    base_dir = str(local_dir_for(download_dir, cfg["base_model"]))
    expert_dirs = [str(local_dir_for(download_dir,
                                     expert_repo(cfg["base_name"], d)))
                   for d in domains]
    for d in [base_dir] + expert_dirs:
        ShardedStateReader(d)   # fail early on missing checkpoints

    # Assemble the run plan: (method_name, kind, hp).
    plan = []
    if args.tier in ("ours", "all"):
        for name, hp in cfg.get("our_methods", {}).items():
            plan.append((name, "ours", hp))
    if args.tier in ("dataless", "all"):
        for name, hp in cfg.get("baselines_dataless", {}).items():
            plan.append((name, "baseline", hp))
    if args.tier == "data":
        for name, hp in cfg.get("baselines_data", {}).items():
            plan.append((name, "baseline", hp))

    only = set(args.only.split(",")) if args.only else None
    print(f"[tier1] base={cfg['base_model']} domains={domains} "
          f"tier={args.tier}", flush=True)

    summaries = {}
    for name, kind, hp in plan:
        if only is not None and name not in only:
            continue
        save_dir = str(merged_root / name)
        print(f"\n[method={name}] kind={kind} hp={hp} -> {save_dir}", flush=True)
        t0 = time.time()
        # Isolate each method: a failure (e.g. a baseline OOMing, or breaking
        # under a newer transformers) is recorded and skipped so the remaining
        # methods still produce their checkpoints. The summary keeps the error.
        try:
            if kind == "ours":
                summary = merge_checkpoints(
                    method=name, base_dir=base_dir, expert_dirs=expert_dirs,
                    save_dir=save_dir,
                    scale=hp.get("scale", 0.4),
                    lam=hp.get("lam", 1e-4),
                    curvature=hp.get("curvature", "taskvec"))
            else:
                run_baseline(algo=name, base_dir=base_dir,
                             expert_dirs=expert_dirs, domains=domains,
                             save_dir=save_dir,
                             mergebench_dir=cfg["mergebench_dir"], hp=hp)
                summary = {}
        except Exception as exc:  # noqa: BLE001 (want to continue the sweep)
            traceback.print_exc()
            summary = {"error": repr(exc)}
            print(f"  [{name}] FAILED: {exc}", flush=True)
        summary["merge_time_s"] = time.time() - t0
        summary["save_dir"] = save_dir
        summaries[name] = summary
        logger.record(event="tier1_merge", method=name, kind=kind,
                      domains=domains, hp=hp, **summary)
        status = "FAILED" if "error" in summary else "done"
        print(f"  [{name}] {status} in {summary['merge_time_s']:.1f}s",
              flush=True)

    out_path = Path(results_dir) / f"tier1_merges_{cfg['base_name']}.json"
    with open(out_path, "w") as f:
        json.dump({"domains": domains, "tier": args.tier,
                   "methods": summaries}, f, indent=2)
    print(f"\n[done] {out_path}", flush=True)
    logger.close()


if __name__ == "__main__":
    main()
