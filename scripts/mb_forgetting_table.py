"""Forgetting table for the Tier-2 sweep: how far each merge falls below the
per-domain specialist, under the IDENTICAL eval protocol used for the merges.

Forgetting_d(method) = specialist_d - method_d, where the specialist on domain d
is the MergeBench expert fine-tuned for d, evaluated with the SAME harness as the
merges (base tokenizer, no chat template; gsm8k_cot 8-shot, ifeval 0-shot,
humanevalplus/mbppplus n_samples). Positive = the merge lost capability vs the
specialist; negative = the merge EXCEEDS the specialist on this protocol (common
on gsm8k_cot, where the chat-SFT math specialist underperforms the base
completion format that merging partly restores).

Specialist diagonal (each expert on its own benchmark):
  math   -> math_expert        / gsm8k_cot
  instr  -> instruction_expert / ifeval
  heval+ -> coding_expert      / humanevalplus
  mbpp+  -> coding_expert      / mbppplus

    python scripts/mb_forgetting_table.py --config configs/mergebench_tier2.yaml \
        --manifest mb_merged/Llama-3.1-8B/pscale_manifest.txt \
        --manifest mb_merged/Llama-3.1-8B/pscale_r1a.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.mb_sweep_table import gate_scores, BASELINE_TAGS  # noqa: E402
from src.utils import load_config  # noqa: E402

# Which expert is the specialist for each gate benchmark.
SPECIALIST = {
    "math": "math_expert",
    "instr": "instruction_expert",
    "heval": "coding_expert",
    "mbpp": "coding_expert",
}
COLS = ["math", "instr", "heval", "mbpp"]


def _read_manifest(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [ln.split()[0] for ln in path.read_text().splitlines() if ln.strip()]


def _pct(x: Optional[float]) -> str:
    return "   --" if x is None else f"{100.0 * x:5.1f}"


def _signed(x: Optional[float]) -> str:
    return "   --" if x is None else f"{100.0 * x:+5.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", action="append", default=[],
                    help="Sweep manifest(s); repeatable.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = cfg["base_name"]
    eval_root = ROOT / "results" / "mb_eval" / base

    # Specialist reference per benchmark (from the expert eval dirs).
    spec_cache: Dict[str, Dict[str, Optional[float]]] = {}
    spec: Dict[str, Optional[float]] = {}
    for col in COLS:
        ex = SPECIALIST[col]
        if ex not in spec_cache:
            spec_cache[ex] = gate_scores(eval_root / ex)
        spec[col] = spec_cache[ex][col]

    variant_tags: List[str] = []
    for m in args.manifest:
        variant_tags += _read_manifest(Path(m))

    rows = []
    for tag in variant_tags + BASELINE_TAGS:
        mdir = eval_root / tag
        if not mdir.exists():
            continue
        g = gate_scores(mdir)
        forget = {c: (None if (spec[c] is None or g[c] is None)
                      else spec[c] - g[c]) for c in COLS}
        fvals = [v for v in forget.values() if v is not None]
        mean_f = sum(fvals) / len(fvals) if fvals else None
        rows.append((tag, g, forget, mean_f, tag in variant_tags))

    # Sort by mean forgetting ascending (least forgetting / most gain first).
    rows.sort(key=lambda r: (r[3] is None, r[3] if r[3] is not None else 0))

    print(f"\n[forgetting] {base}  forgetting_d = specialist_d - merged_d  "
          f"(negative = merge BEATS the specialist on this protocol)\n")
    print("  specialist diagonal: "
          + "  ".join(f"{c}({SPECIALIST[c].replace('_expert','')})="
                      f"{_pct(spec[c])}" for c in COLS))
    print()
    hdr = f"  {'method':28s}"
    for c in COLS:
        hdr += f" {c:>6}"
    hdr += f" {'MEANF':>7}  kind"
    print(hdr)
    print("  " + "-" * 78)
    for tag, g, forget, mean_f, is_var in rows:
        line = f"  {tag:28s}"
        for c in COLS:
            line += f" {_signed(forget[c])}"
        mf = "    --" if mean_f is None else f"{100.0 * mean_f:+6.1f}"
        line += f" {mf}  {'sweep' if is_var else 'baseline'}"
        print(line)
    print("\n  (MEANF<0 => on average the merge exceeds the per-domain "
          "specialists under this completion-format protocol.)")


if __name__ == "__main__":
    main()
