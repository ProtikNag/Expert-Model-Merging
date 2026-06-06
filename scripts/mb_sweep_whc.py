"""Sweep HTCL (whc_diag) hyperparameters to make it competitive at N=5.

Tier 2 showed the plain closed form (lam=1e-4, alpha=1) underperforming the
dataless baselines on Llama-3.1-8B: the curvature-weighted *mean* dilutes each
expert's update by ~1/N versus a *sum* of task vectors, and lam was never tuned
for this base/N. This driver re-merges whc_diag over a (lam, alpha) grid for the
dataless task-vector curvature and, if per-expert Fisher dirs are supplied, the
data Fisher curvature too. Each variant is saved as

    mb_merged/<base>/whc_<curv>_l<lam>_a<alpha>/

and every variant tag+dir is appended to a manifest the sweep eval drivers read
(one "tag dir" per line). CPU-only; run on BigMem (see mb_sweep_merge.sh).

    python -u scripts/mb_sweep_whc.py --config configs/mergebench_tier2.yaml \
        --lams 1e-3,1e-4,1e-5 --alphas 1,2,3,4
    # add data curvature too:
    python -u scripts/mb_sweep_whc.py --config configs/mergebench_tier2.yaml \
        --lams 1e-4,1e-5 --alphas 1,2,3 --fisher-root mb_fisher/Llama-3.1-8B
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mergebench.llm_merge import merge_checkpoints  # noqa: E402
from scripts.mb_download import expert_repo, local_dir_for  # noqa: E402
from src.utils import ensure_dir, load_config  # noqa: E402


def _fmt(x: float) -> str:
    """Compact, filesystem-safe float tag, e.g. 1e-4 -> '1e-04', 2.0 -> '2'."""
    if x == int(x):
        return str(int(x))
    return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--domains", default=None,
                    help="Comma-separated override of tier_domains.")
    ap.add_argument("--lams", default="1e-3,1e-4,1e-5",
                    help="Comma-separated anchor coefficients.")
    ap.add_argument("--alphas", default="1,2,3,4",
                    help="Comma-separated task-vector scales.")
    ap.add_argument("--fisher-root", default=None,
                    help="If set, also sweep curvature=fisher reading "
                         "<root>/<domain>/model.safetensors per expert.")
    ap.add_argument("--manifest", default=None,
                    help="Where to write the variant manifest "
                         "(default mb_merged/<base>/sweep_manifest.txt).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    download_dir = Path(cfg["download_dir"])
    domains = (args.domains.split(",") if args.domains else cfg["tier_domains"])
    base_dir = str(local_dir_for(download_dir, cfg["base_model"]))
    expert_dirs = [str(local_dir_for(download_dir,
                                     expert_repo(cfg["base_name"], d)))
                   for d in domains]
    merged_root = ensure_dir(Path(cfg["paths"]["merged"]) / cfg["base_name"])

    lams = [float(x) for x in args.lams.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    # (curvature_tag, curvature_arg, fisher_dirs)
    curvatures = [("tv", "taskvec", None)]
    if args.fisher_root:
        fisher_dirs = [str(Path(args.fisher_root) / d) for d in domains]
        for d in fisher_dirs:
            if not (Path(d) / "model.safetensors").exists():
                raise FileNotFoundError(
                    f"missing Fisher at {d}/model.safetensors — run "
                    f"scripts/mb_fisher_estimate.py per domain first.")
        curvatures.append(("fish", "fisher", fisher_dirs))

    manifest_path = Path(args.manifest) if args.manifest else (
        merged_root / "sweep_manifest.txt")
    plan = [(ct, ca, fd, lam, a)
            for (ct, ca, fd) in curvatures for lam in lams for a in alphas]
    print(f"[sweep] base={cfg['base_name']} domains={domains} "
          f"variants={len(plan)} -> {manifest_path}", flush=True)

    lines = []
    for ct, ca, fd, lam, a in plan:
        tag = f"whc_{ct}_l{_fmt(lam)}_a{_fmt(a)}"
        save_dir = merged_root / tag
        print(f"\n[variant {tag}] curvature={ca} lam={lam} alpha={a}", flush=True)
        t0 = time.time()
        merge_checkpoints(method="whc_diag", base_dir=base_dir,
                          expert_dirs=expert_dirs, save_dir=str(save_dir),
                          lam=lam, alpha=a, curvature=ca, fisher_dirs=fd)
        lines.append(f"{tag} {save_dir}")
        print(f"  [{tag}] done in {time.time() - t0:.1f}s", flush=True)

    with open(manifest_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[done] {len(lines)} variants; manifest -> {manifest_path}",
          flush=True)


if __name__ == "__main__":
    main()
