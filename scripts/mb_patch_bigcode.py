"""Idempotent patch to force eager attention in the bigcode-eval harness.

gemma2's SDPA path hits "cutlassF: no kernel found to launch!" on this node's
GPU (same failure lmeval had until we passed attn_implementation=eager). The
bigcode harness exposes no CLI flag for it and calls
``AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)`` with a
fixed ``model_kwargs`` dict, so we inject ``attn_implementation="eager"`` into
that dict in ``main.py``. Eager is the reference attention implementation, so
this does not change results, only the kernel.

Usage
-----
    python -u scripts/mb_patch_bigcode.py --bigcode /work/pnag/bigcode-evaluation-harness
"""
from __future__ import annotations

import argparse
from pathlib import Path

_ORIG = '''    model_kwargs = {
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
        "token": args.use_auth_token,
    }'''

_PATCHED = '''    model_kwargs = {
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
        "token": args.use_auth_token,
        "attn_implementation": "eager",  # [whc-patch] gemma2 SDPA cutlassF fails here
    }'''

_MARKER = "[whc-patch] gemma2 SDPA"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bigcode", default="/work/pnag/bigcode-evaluation-harness",
                    help="Path to the local bigcode-evaluation-harness clone.")
    args = ap.parse_args()

    main_py = Path(args.bigcode) / "main.py"
    text = main_py.read_text()
    if _MARKER in text:
        print(f"already patched: {main_py}", flush=True)
        return
    if _ORIG not in text:
        print(f"WARNING: model_kwargs block not found in {main_py} verbatim "
              f"(upstream changed?). Inspect and patch by hand.", flush=True)
        return
    main_py.write_text(text.replace(_ORIG, _PATCHED))
    print(f"patched: {main_py} (attn_implementation=eager)", flush=True)


if __name__ == "__main__":
    main()
