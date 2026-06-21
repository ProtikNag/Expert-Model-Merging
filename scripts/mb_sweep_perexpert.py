"""Per-EXPERT scaled task arithmetic sweep:  w = w_pre + sum_i s_i * tau_i.

The weight-space routing probes (per-parameter ownership, per-layer energy)
failed because the N=5 domains are entangled in parameter space. The separable
axis is the EXPERT. Diagnosis: instruction is diluted (wants high s), coding
over-extrapolates (wants moderate s); math/safety/multilingual sit at the
global-0.4 baseline. This driver sweeps a handful of per-expert coefficient
vectors in ONE pass over the six models.

Coefficients are given per domain via repeated --set DOMAIN=v1,v2,... ; the
sweep is the CARTESIAN product over the listed domains, with unlisted domains
held at --base. Expert order follows cfg['tier_domains'].

    python -u scripts/mb_sweep_perexpert.py --config configs/mergebench_tier2.yaml \
        --base 0.4 --set instruction=0.6,0.8,1.0 --set coding=0.3,0.4 \
        --manifest mb_merged/Llama-3.1-8B/perexpert_r0.txt
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mergebench.llm_merge import merge_task_arith_perexpert_multi  # noqa: E402
from scripts.mb_download import expert_repo, local_dir_for  # noqa: E402
from scripts.mb_sweep_pscale import _fmt  # noqa: E402
from src.utils import ensure_dir, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--domains", default=None)
    ap.add_argument("--base", default=0.4, type=float,
                    help="Coefficient for any domain not given via --set.")
    ap.add_argument("--set", action="append", default=[], dest="sets",
                    help="DOMAIN=v1,v2,... (repeatable); swept as a cartesian "
                         "product over the listed domains.")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    download_dir = Path(cfg["download_dir"])
    domains = (args.domains.split(",") if args.domains else cfg["tier_domains"])
    base_dir = str(local_dir_for(download_dir, cfg["base_model"]))
    expert_dirs = [str(local_dir_for(download_dir,
                                     expert_repo(cfg["base_name"], d)))
                   for d in domains]
    merged_root = ensure_dir(Path(cfg["paths"]["merged"]) / cfg["base_name"])

    swept = {}                       # domain -> [values]
    for s in args.sets:
        dom, vals = s.split("=")
        if dom not in domains:
            raise ValueError(f"--set domain {dom!r} not in {domains}")
        swept[dom] = [float(x) for x in vals.split(",") if x.strip()]
    swept_doms = list(swept.keys())
    print(f"[perexpert] order={domains} base={args.base} "
          f"swept={swept}", flush=True)

    variants = []
    combos = (list(itertools.product(*[swept[d] for d in swept_doms]))
              if swept_doms else [()])
    for combo in combos:
        coeffs = [args.base] * len(domains)
        parts = []
        for d, val in zip(swept_doms, combo):
            coeffs[domains.index(d)] = val
            parts.append(f"{d[:4]}{_fmt(val)}")
        tag = "ta_pe_" + "_".join(parts) if parts else f"ta_pe_all{_fmt(args.base)}"
        variants.append(dict(tag=tag, coeffs=coeffs))

    save_dirs = [str(merged_root / v["tag"]) for v in variants]
    manifest_path = Path(args.manifest) if args.manifest else (
        merged_root / "perexpert_manifest.txt")
    print(f"[perexpert] variants={len(variants)} (single-pass) -> {manifest_path}",
          flush=True)
    for v in variants:
        print(f"  - {v['tag']}: coeffs={v['coeffs']}", flush=True)

    t0 = time.time()
    merge_task_arith_perexpert_multi(base_dir=base_dir, expert_dirs=expert_dirs,
                                     variants=variants, save_dirs=save_dirs)
    print(f"\n[done] {len(variants)} variants in {time.time() - t0:.1f}s",
          flush=True)
    with open(manifest_path, "w") as f:
        f.write("\n".join(f"{v['tag']} {sd}"
                          for v, sd in zip(variants, save_dirs)) + "\n")
    print(f"[done] manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
