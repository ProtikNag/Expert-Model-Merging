"""Run the data-using whc_gram merge (HTCL-data) for the RegMean comparison.

Requires per-expert input Grams computed first by scripts/mb_gram_estimate.py,
laid out as ``<gram_root>/<domain>/model.safetensors`` (one per merged domain).
Optionally also reads per-expert diagonal Fishers (``<fisher_root>/<domain>/``)
for the gamma ridge.

Produces one merged checkpoint per (lam, gamma) requested, named
``whc_gram_l<lam>_g<gamma>`` under ``mb_merged/<base_name>/``, and appends each
``tag dir`` line to ``mb_merged/<base_name>/whc_gram_manifest.txt`` so the
existing sweep-eval drivers (``mb_eval_sweep_{lm,code}.sh``) score them with no
new eval code (pass ``MANIFEST=.../whc_gram_manifest.txt``). Runs through the
same memory-bounded streaming path as every other merge; CPU-only, but the
per-key ``[in, in]`` solve is heavy for ``down_proj`` (14336x14336) so give it a
high-RAM node.

This is the single-pass (K=0) merge. The iterative "catch-up" (K>=1) is a shell
loop around this script + mb_gram_estimate.py: re-estimate Grams of each domain
*on the round-(k-1) merged model*, then re-merge the original experts with the
refreshed Grams (see TIER2_RUNBOOK.md, data tier).

Usage
-----
    python -u scripts/mb_merge_whc_gram.py --config configs/mergebench_tier2.yaml \
        --gram-root mb_grams/Llama-3.1-8B --lams 0,1e-3,1e-2 --gammas 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mergebench.llm_merge import merge_checkpoints  # noqa: E402
from scripts.mb_download import expert_repo, local_dir_for  # noqa: E402
from src.utils import ensure_dir, load_config  # noqa: E402


def _fmt(x: float) -> str:
    """Compact tag for a hyperparameter value (1e-3 -> '1e-3', 0.0 -> '0')."""
    if x == 0:
        return "0"
    if x == int(x):
        return str(int(x))
    return f"{x:g}"


def _parse_floats(s: str) -> List[float]:
    return [float(v) for v in s.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gram-root", default="mb_grams/Llama-3.1-8B")
    ap.add_argument("--fisher-root", default=None,
                    help="Per-domain diagonal Fisher root, for gamma>0 only.")
    ap.add_argument("--domains", default=None,
                    help="Comma-separated; defaults to cfg tier_domains.")
    ap.add_argument("--lams", default="0,1e-3,1e-2",
                    help="Comma-separated ridge-toward-mean coefficients.")
    ap.add_argument("--gammas", default="0",
                    help="Comma-separated Fisher-ridge coefficients (needs "
                         "--fisher-root when nonzero).")
    ap.add_argument("--manifest", default=None,
                    help="Where to append the variant manifest "
                         "(default mb_merged/<base>/whc_gram_manifest.txt).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    download_dir = Path(cfg["download_dir"])
    domains = (args.domains.split(",") if args.domains else cfg["tier_domains"])
    base_dir = str(local_dir_for(download_dir, cfg["base_model"]))
    expert_dirs = [str(local_dir_for(download_dir,
                                     expert_repo(cfg["base_name"], d)))
                   for d in domains]
    gram_dirs = [str(Path(args.gram_root) / d) for d in domains]
    fisher_dirs = ([str(Path(args.fisher_root) / d) for d in domains]
                   if args.fisher_root else None)
    merged_root = ensure_dir(Path(cfg["paths"]["merged"]) / cfg["base_name"])

    for d in gram_dirs:
        if not (Path(d) / "model.safetensors").exists():
            raise FileNotFoundError(
                f"missing Gram at {d}/model.safetensors — run "
                f"scripts/mb_gram_estimate.py for each domain first.")

    lams = _parse_floats(args.lams)
    gammas = _parse_floats(args.gammas)
    if any(g != 0 for g in gammas) and fisher_dirs is None:
        raise ValueError("gamma>0 requires --fisher-root.")

    manifest_path = (Path(args.manifest) if args.manifest
                     else merged_root / "whc_gram_manifest.txt")
    print(f"[whc-gram] domains={domains} grams={gram_dirs} "
          f"lams={lams} gammas={gammas} -> manifest {manifest_path}", flush=True)

    lines: List[str] = []
    for lam in lams:
        for gamma in gammas:
            tag = f"whc_gram_l{_fmt(lam)}_g{_fmt(gamma)}"
            save_dir = merged_root / tag
            print(f"\n[variant {tag}] lam={lam} gamma={gamma}", flush=True)
            merge_checkpoints(
                method="whc_gram", base_dir=base_dir, expert_dirs=expert_dirs,
                save_dir=str(save_dir), lam=lam, gamma=gamma,
                grams_dirs=gram_dirs,
                fisher_dirs=(fisher_dirs if gamma != 0 else None))
            lines.append(f"{tag} {save_dir}")

    with open(manifest_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[whc-gram] done: {len(lines)} variants; manifest -> {manifest_path}",
          flush=True)


if __name__ == "__main__":
    main()
