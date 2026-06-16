"""Sweep HTCL (whc_diag) per-parameter update scaling at N=5.

The global-alpha sweep (scripts/mb_sweep_whc.py) capped HTCL at the dataless tie
because one scalar cannot serve every domain (instruction wants a large scale,
coding a small one). This driver replaces the single global ``alpha`` with a
per-parameter scale derived from inter-expert agreement (see
``mergebench.llm_merge.merge_checkpoints`` ``pscale``):

  - consensus: a_p = clip(|agreeing-expert task-vector sum| / |update|, 1, amax)
    -> recovers task arithmetic's additive sum where experts agree, leaves
       single-expert params unscaled (no over-extrapolation), bounds conflicts.
  - coherence: a_p = 1 + (amax-1) * (|sum tau| / sum|tau|)^beta   (AAAI-plan form)

Each variant is saved as ``mb_merged/<base>/whc_<mode-tag>/`` and appended to a
manifest the existing sweep eval drivers (mb_eval_sweep_{lm,code}.sh) and ranker
(mb_sweep_table.py) read unchanged. CPU-only; run on BigMem via
scripts/mb_sweep_pscale.sh.

    # consensus alpha_max in {3,5,8}, coherence (amax,beta) in {(3,1),(5,1),(5,2)}
    python -u scripts/mb_sweep_pscale.py --config configs/mergebench_tier2.yaml \
        --lam 1e-3 --cons-amaxes 3,5,8 --coh-amaxes 5 --coh-betas 1,2
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
from src.utils import ensure_dir, load_config  # noqa: E402


def _fmt(x: float) -> str:
    """Compact, filesystem-safe float tag, e.g. 1e-3 -> '1e-3', 5.0 -> '5'."""
    if x == int(x):
        return str(int(x))
    return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--domains", default=None,
                    help="Comma-separated override of tier_domains.")
    ap.add_argument("--lam", default="1e-3", type=str,
                    help="Single anchor coefficient (the global-sweep best).")
    ap.add_argument("--cons-amaxes", default="3,5,8",
                    help="Comma-separated alpha_max for the consensus mode "
                         "(empty string disables consensus variants).")
    ap.add_argument("--coh-amaxes", default="5",
                    help="Comma-separated alpha_max for the coherence mode "
                         "(empty string disables coherence variants).")
    ap.add_argument("--coh-betas", default="1,2",
                    help="Comma-separated beta for the coherence mode.")
    ap.add_argument("--manifest", default=None,
                    help="Where to write the variant manifest "
                         "(default mb_merged/<base>/pscale_manifest.txt).")
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

    def _floats(s):
        return [float(x) for x in s.split(",") if x.strip()]

    # All variants share the whc_diag closed form and differ only in the
    # per-parameter scale, so merge them in ONE pass over the keys (reads the
    # six models once, not once-per-variant -- the dominant NFS I/O at 8B).
    variants = []
    for am in _floats(args.cons_amaxes):
        variants.append(dict(tag=f"whc_cons_l{lam_tag}_am{_fmt(am)}",
                             pscale="consensus", alpha_max=am, beta=1.0))
    for am in _floats(args.coh_amaxes):
        for b in _floats(args.coh_betas):
            variants.append(dict(tag=f"whc_coh_l{lam_tag}_am{_fmt(am)}_b{_fmt(b)}",
                                pscale="coherence", alpha_max=am, beta=b))

    save_dirs = [str(merged_root / v["tag"]) for v in variants]
    manifest_path = Path(args.manifest) if args.manifest else (
        merged_root / "pscale_manifest.txt")
    print(f"[pscale-sweep] base={cfg['base_name']} domains={domains} "
          f"lam={lam} variants={len(variants)} (single-pass) -> {manifest_path}",
          flush=True)
    for v in variants:
        print(f"  - {v['tag']}: {v['pscale']} alpha_max={v['alpha_max']} "
              f"beta={v['beta']}", flush=True)

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
