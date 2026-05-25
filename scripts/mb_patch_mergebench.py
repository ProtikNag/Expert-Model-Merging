"""Idempotent, hardware-specific patches to a local MergeBench clone.

MergeBench's mergers assume a beefy GPU node with flash-attn and enough RAM to
hold several billion-parameter task vectors at once. Our HPC node has neither
(128G RAM ceiling, gcc 4.8.5 so flash-attn cannot build). Two of its baselines
fail as written; both fixes below are numerically faithful (same merged result,
only the memory profile / attention kernel changes):

1. TIES ``topk_values_mask`` materializes ``M.abs()`` twice on the stacked
   ``[n_experts, num_params]`` matrix, OOMing above 120G. Rows are independent
   experts, so we take the per-row kth-value in a loop and keep only one row's
   ``.abs()`` resident at a time.
2. LocalizeAndStitch hardcodes ``attn_implementation="flash_attention_2"`` in
   two ``from_pretrained`` calls. We switch to ``"eager"`` (flash-attn is absent
   and unbuildable here; the dataless path runs no attention anyway).

Run after cloning MergeBench and before the merge stage. Safe to re-run: each
patch checks for its own marker and skips if already applied.

Usage
-----
    python -u scripts/mb_patch_mergebench.py --mergebench ./MergeBench
"""
from __future__ import annotations

import argparse
from pathlib import Path

# ── TIES: replace the dense two-copy trim with a per-row loop ────────────────
_TIES_ORIG = """\
    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask"""

_TIES_PATCHED = """\
    # [whc-patch] memory-bounded: kthvalue per row so we never hold two full
    # M.abs() copies of a [n_experts, num_params] matrix at once (OOMs above
    # 128G for billion-param models). Numerically identical to the original.
    mask = torch.zeros_like(M, dtype=torch.bool)
    for _row in range(n):
        _row_abs = M[_row].abs()
        _kth = _row_abs.kthvalue(k).values
        mask[_row] = _row_abs >= _kth
        del _row_abs
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask"""

_TIES_MARKER = "[whc-patch] memory-bounded"

# ── LocalizeAndStitch: flash_attention_2 -> eager ────────────────────────────
_FA2 = 'attn_implementation="flash_attention_2"'
_EAGER = 'attn_implementation="eager"'  # [whc-patch] flash-attn unavailable here


def _patch_file(path: Path, orig: str, patched: str, marker: str) -> str:
    """Replace ``orig`` with ``patched`` in ``path`` unless ``marker`` present."""
    text = path.read_text()
    if marker in text:
        return f"already patched: {path.name}"
    if orig not in text:
        return f"WARNING: target text not found in {path.name} (upstream changed?)"
    path.write_text(text.replace(orig, patched))
    return f"patched: {path.name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mergebench", default="./MergeBench",
                    help="Path to the local MergeBench clone.")
    args = ap.parse_args()

    mm = Path(args.mergebench) / "merging" / "merging_methods"
    ties = mm / "ties_merging_utils.py"
    localize = mm / "localize_utils.py"

    print(_patch_file(ties, _TIES_ORIG, _TIES_PATCHED, _TIES_MARKER), flush=True)

    # localize: replace every flash_attention_2 occurrence (two as of writing).
    loc_text = localize.read_text()
    if _FA2 not in loc_text:
        status = ("already patched: localize_utils.py"
                  if _EAGER in loc_text else
                  "WARNING: flash_attention_2 not found in localize_utils.py")
    else:
        n = loc_text.count(_FA2)
        localize.write_text(loc_text.replace(_FA2, _EAGER))
        status = f"patched: localize_utils.py ({n} attn_implementation site(s))"
    print(status, flush=True)


if __name__ == "__main__":
    main()
