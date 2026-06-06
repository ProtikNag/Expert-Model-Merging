"""Rank the HTCL sweep variants by their gate score (math + instr + coding).

Reads the sweep manifest and each variant's eval outputs, computes the mean of
gsm8k / ifeval / humanevalplus / mbppplus (the three fast domains), and prints a
ranked table. The best variant is the candidate to promote to a full four-domain
eval (multilingual on L40S) via the main Tier 2 drivers.

For reference it also prints the dataless-baseline gate scores from the main run
(Consensus/TIES/DARE/TaskArithmetic) if those results are present, so you can see
whether the tuned HTCL clears the baseline cluster.

    python scripts/mb_sweep_table.py --config configs/mergebench_tier2.yaml
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402

BASELINE_TAGS = ["Consensus", "TIES", "DARE", "TaskArithmetic",
                 "task_arith", "whc_diag"]


def _all_lm(model_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for p in sorted(glob.glob(str(model_dir / "**" / "results_*.json"),
                              recursive=True),
                    key=lambda q: Path(q).stat().st_mtime):
        try:
            out.update(json.load(open(p)).get("results", {}))
        except json.JSONDecodeError:
            continue
    return out


def _metric(block: dict, *names: str) -> Optional[float]:
    for n in names:
        if n in block:
            return float(block[n])
    return None


def gate_scores(model_dir: Path) -> Dict[str, Optional[float]]:
    res = _all_lm(model_dir)
    math = (_metric(res.get("gsm8k_cot", {}), "exact_match,flexible-extract",
                    "exact_match,strict-match", "exact_match"))
    instr = _metric(res.get("ifeval", {}), "prompt_level_strict_acc,none",
                    "prompt_level_strict_acc")
    heval = mbpp = None
    code_path = model_dir / "code_eval.json"
    if code_path.exists():
        data = json.load(open(code_path))
        for key, dst in (("humanevalplus", "heval"), ("mbppplus", "mbpp")):
            blk = data.get(key)
            if isinstance(blk, dict):
                v = blk.get("pass@1")
                if dst == "heval":
                    heval = v
                else:
                    mbpp = v
    return {"math": math, "instr": instr, "heval": heval, "mbpp": mbpp}


def _mean(*xs: Optional[float]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def _pct(x: Optional[float]) -> str:
    return "  --" if x is None else f"{100.0 * x:5.1f}"


def _read_manifest(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [ln.split()[0] for ln in path.read_text().splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = cfg["base_name"]
    eval_root = ROOT / "results" / "mb_eval" / base
    merged_root = ROOT / "mb_merged" / base
    manifest = Path(args.manifest) if args.manifest else (
        merged_root / "sweep_manifest.txt")

    variant_tags = _read_manifest(manifest)
    rows = []
    for tag in variant_tags + BASELINE_TAGS:
        mdir = eval_root / tag
        if not mdir.exists():
            continue
        g = gate_scores(mdir)
        gate = _mean(g["math"], g["instr"], g["heval"], g["mbpp"])
        rows.append((tag, g, gate, tag in variant_tags))

    rows.sort(key=lambda r: (r[2] is None, -(r[2] or 0)))

    print(f"\n[sweep-table] {base}  (gate = mean of math/instr/heval+/mbpp+, %)\n")
    print(f"  {'variant':28s} {'math':>5} {'instr':>5} {'heval':>5} "
          f"{'mbpp':>5} {'GATE':>6}  kind")
    print("  " + "-" * 70)
    for tag, g, gate, is_variant in rows:
        kind = "sweep" if is_variant else "baseline"
        gate_s = "   --" if gate is None else f"{100.0 * gate:5.1f}"
        print(f"  {tag:28s} {_pct(g['math'])} {_pct(g['instr'])} "
              f"{_pct(g['heval'])} {_pct(g['mbpp'])} {gate_s}  {kind}")

    sweep_rows = [r for r in rows if r[3] and r[2] is not None]
    base_rows = [r for r in rows if not r[3] and r[2] is not None]
    if sweep_rows:
        best = sweep_rows[0]
        print(f"\n  best HTCL variant: {best[0]}  gate={100 * best[2]:.1f}")
        if base_rows:
            top_base = max(base_rows, key=lambda r: r[2])
            verdict = ("CLEARS" if best[2] >= top_base[2] else "below")
            print(f"  top baseline:      {top_base[0]}  gate={100 * top_base[2]:.1f}"
                  f"   ->  tuned HTCL {verdict} the baseline cluster")


if __name__ == "__main__":
    main()
