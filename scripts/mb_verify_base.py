"""Verify the ungated base mirror matches Meta's canonical Llama-3.1-8B init.

Tier 2 anchors every task vector at the pretrained base (``w_pre``). Because the
official ``meta-llama/Llama-3.1-8B`` repo was gated at run time, the merge used the
ungated ``NousResearch/Meta-Llama-3.1-8B`` re-upload. This script confirms the two
are bit-identical so the task-vector anchor is correct. Run it once official
access lands and the canonical repo is downloaded.

Compares every shared float tensor: exact equality first, then max abs diff for a
tolerant report. A clean run (0 mismatches, max diff 0) means the mirror was a
faithful stand-in and the Tier 2 merges need no redo.

    # after: python scripts/mb_download.py --config <cfg pointing base at meta-llama/...>
    python scripts/mb_verify_base.py \
        --mirror mb_ckpts/NousResearch__Meta-Llama-3.1-8B \
        --official mb_ckpts/meta-llama__Llama-3.1-8B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mergebench.io_utils import ShardedStateReader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", required=True,
                    help="Base dir actually used for merging (the mirror).")
    ap.add_argument("--official", required=True,
                    help="Canonical meta-llama base dir, once access is granted.")
    ap.add_argument("--rtol", type=float, default=0.0,
                    help="Report tensors whose max abs diff exceeds this.")
    args = ap.parse_args()

    a = ShardedStateReader(args.mirror)
    b = ShardedStateReader(args.official)
    keys_a, keys_b = set(a.keys()), set(b.keys())

    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    shared = sorted(keys_a & keys_b)
    if only_a:
        print(f"[warn] {len(only_a)} keys only in mirror, e.g. {only_a[:3]}")
    if only_b:
        print(f"[warn] {len(only_b)} keys only in official, e.g. {only_b[:3]}")

    n_exact, n_close, n_diff = 0, 0, 0
    worst_key, worst_diff = None, 0.0
    for i, key in enumerate(shared):
        ta, tb = a.get(key), b.get(key)
        if ta.shape != tb.shape:
            n_diff += 1
            print(f"  [SHAPE] {key}: {tuple(ta.shape)} vs {tuple(tb.shape)}")
            continue
        if torch.equal(ta, tb):
            n_exact += 1
        else:
            diff = (ta.float() - tb.float()).abs().max().item()
            if diff > worst_diff:
                worst_diff, worst_key = diff, key
            if diff <= args.rtol:
                n_close += 1
            else:
                n_diff += 1
        if (i + 1) % 50 == 0 or (i + 1) == len(shared):
            print(f"  {i + 1}/{len(shared)} checked "
                  f"(exact={n_exact}, close={n_close}, diff={n_diff})",
                  flush=True)

    print("\n[summary]")
    print(f"  shared tensors : {len(shared)}")
    print(f"  bit-identical  : {n_exact}")
    print(f"  within rtol    : {n_close}")
    print(f"  mismatched     : {n_diff}")
    print(f"  worst max-diff : {worst_diff:.3e} @ {worst_key}")
    if n_diff == 0 and not only_a and not only_b and worst_diff == 0.0:
        print("\n[OK] mirror is bit-identical to the official base. "
              "Tier 2 anchor is correct; no redo needed.")
    else:
        print("\n[CHECK] differences found. If non-trivial, re-merge from the "
              "official base and re-run the affected evals.")


if __name__ == "__main__":
    main()
