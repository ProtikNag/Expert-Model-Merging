"""Download the MergeBench base model and selected domain experts locally.

Uses ``huggingface_hub.snapshot_download`` so the checkpoints land in a
predictable local directory the Tier 0 / Tier 1 scripts can read. Run this
once on the HPC node before merging.

Example
-------
    python scripts/mb_download.py --config configs/mergebench.yaml

Requires a HuggingFace login (``huggingface-cli login``) and acceptance of
the base model license (Gemma / Llama are gated).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huggingface_hub import snapshot_download  # noqa: E402

from src.utils import load_config  # noqa: E402


def expert_repo(base_name: str, domain: str) -> str:
    """MergeBench expert repo id, e.g. 'MergeBench/gemma-2-2b_math'."""
    return f"MergeBench/{base_name}_{domain}"


def local_dir_for(download_dir: Path, repo_id: str) -> Path:
    """Local directory for a repo (repo id with '/' replaced by '__')."""
    return download_dir / repo_id.replace("/", "__")


def fetch(repo_id: str, dest: Path) -> None:
    print(f"[download] {repo_id} -> {dest}", flush=True)
    snapshot_download(repo_id=repo_id, local_dir=str(dest),
                      ignore_patterns=["*.pth", "*.bin", "*.h5", "*.msgpack"])
    print(f"[done] {repo_id}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--domains", default=None,
                    help="Comma-separated override of tier_domains.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    download_dir = Path(cfg["download_dir"])
    download_dir.mkdir(parents=True, exist_ok=True)

    domains = (args.domains.split(",") if args.domains
               else cfg["tier_domains"])

    # Base model.
    base_dest = local_dir_for(download_dir, cfg["base_model"])
    fetch(cfg["base_model"], base_dest)

    # Domain experts.
    for domain in domains:
        repo = expert_repo(cfg["base_name"], domain)
        fetch(repo, local_dir_for(download_dir, repo))

    print("\n[all done] checkpoints under:", download_dir, flush=True)


if __name__ == "__main__":
    main()
