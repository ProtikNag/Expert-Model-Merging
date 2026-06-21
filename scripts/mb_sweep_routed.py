"""Sweep dominance-ROUTED per-parameter update scaling for HTCL (whc_diag).

Round 0/1 showed a hard coupling: instruction monotonically wants a high update
scale (instr 15->32 as alpha 1->3) while coding monotonically wants a low one
(mbpp+ 56->40). A single global -- or even per-parameter consensus -- scale
averages them out at the saturated ~51 gate. This driver breaks the coupling
dataless-ly: route the per-parameter cap by which expert OWNS the weight.

For each parameter, the dominant expert = argmax_i |w_i - w_pre|. Parameters
owned by a "low" expert (default: coding) are capped at ``alpha_low`` (protect
the coding capability that over-boosting destroys); all other parameters use the
consensus rescale capped at ``alpha_high`` (give instruction its large scale).
The merge is otherwise the curvature-weighted whc_diag closed form, dataless,
closed-form, one pass over the six models (``merge_whc_diag_pscale_multi``).

    python -u scripts/mb_sweep_routed.py --config configs/mergebench_tier2.yaml \
        --lam 1e-3 --highs 3,3.5,4,5 --lows 1 --low-domains coding \
        --manifest mb_merged/Llama-3.1-8B/routed_r0.txt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mergebench.llm_merge import merge_whc_diag_pscale_multi  # noqa: E402
from scripts.mb_download import expert_repo, local_dir_for  # noqa: E402
from scripts.mb_sweep_pscale import _fmt  # noqa: E402
from src.utils import ensure_dir, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--domains", default=None,
                    help="Comma-separated override of tier_domains (also fixes "
                         "the expert ordering used to resolve --low-domains).")
    ap.add_argument("--lam", default="1e-3", type=str)
    ap.add_argument("--highs", default="3,3.5,4,5",
                    help="Comma-separated alpha_high (cap on non-low params).")
    ap.add_argument("--lows", default="1",
                    help="Comma-separated alpha_low (cap on low-expert params).")
    ap.add_argument("--low-domains", default="coding",
                    help="Comma-separated domains whose owned params are capped "
                         "at alpha_low (protected from over-boost).")
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

    lam = float(args.lam)
    lam_tag = _fmt(lam)

    low_domains = [d.strip() for d in args.low_domains.split(",") if d.strip()]
    low_idx = [domains.index(d) for d in low_domains]      # KeyError-safe below
    print(f"[routed] domain order={domains}  low-domains={low_domains} "
          f"-> low_idx={low_idx}", flush=True)

    def _floats(s):
        return [float(x) for x in s.split(",") if x.strip()]

    lo_tag = "".join(d[0] for d in low_domains)            # 'coding' -> 'c'
    variants = []
    for h in _floats(args.highs):
        for lo in _floats(args.lows):
            variants.append(dict(
                tag=f"whc_route_l{lam_tag}_h{_fmt(h)}_l{_fmt(lo)}_{lo_tag}",
                pscale="consensus_routed", alpha_max=h, alpha_low=lo,
                low_idx=low_idx, beta=1.0))

    save_dirs = [str(merged_root / v["tag"]) for v in variants]
    manifest_path = Path(args.manifest) if args.manifest else (
        merged_root / "routed_manifest.txt")
    print(f"[routed] base={cfg['base_name']} lam={lam} variants={len(variants)} "
          f"(single-pass) -> {manifest_path}", flush=True)
    for v in variants:
        print(f"  - {v['tag']}: high={v['alpha_max']} low={v['alpha_low']} "
              f"low_idx={v['low_idx']}", flush=True)

    t0 = time.time()
    merge_whc_diag_pscale_multi(base_dir=base_dir, expert_dirs=expert_dirs,
                                variants=variants, save_dirs=save_dirs, lam=lam)
    print(f"\n[done] {len(variants)} variants in {time.time() - t0:.1f}s",
          flush=True)

    with open(manifest_path, "w") as f:
        f.write("\n".join(f"{v['tag']} {sd}"
                          for v, sd in zip(variants, save_dirs)) + "\n")
    print(f"[done] manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
