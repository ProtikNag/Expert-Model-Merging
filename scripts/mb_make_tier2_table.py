"""Assemble the Tier 2 (Llama-3.1-8B, five domains) results table.

Reads the per-model eval outputs written by the three Tier 2 drivers under
``results/mb_eval/<base>/<tag>/`` and emits, per model:

  - instruction : ifeval ``prompt_level_strict_acc``                 (lm-eval)
  - math        : gsm8k_cot ``exact_match,flexible-extract``         (lm-eval)
  - multilingual: mean over m_mmlu/arc/hellaswag x {fr,es,de,ru};
                  acc for m_mmlu, acc_norm for arc/hellaswag (Table 3)   (lm-eval)
  - coding      : humanevalplus / mbppplus pass@1                    (bigcode)
  - safety      : mean of wildguardtest/harmbench/do_anything_now RTA
                  and xstest accuracy                              (safety-eval)

Writes ``results/mergebench/tier2_table_<base>.json`` and prints HTML <tr> rows
ready to paste into results/mergebench/TIER1_TABLE.md. Safety JSON schema varies
by safety-eval version, so that parser is defensive and reports what it found;
verify the safety column against the raw safety_eval.json before trusting it.

    python scripts/mb_make_tier2_table.py --config configs/mergebench_tier2.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402

# Display order: tag -> (category, pretty label).
ROW_ORDER = [
    ("base", "Reference", "Llama-3.1-8B (no merge)"),
    ("Consensus", "Dataless merge", "Consensus"),
    ("LocalizeAndStitch", "Dataless merge", "LocalizeAndStitch"),
    ("TIES", "Dataless merge", "TIES"),
    ("DARE", "Dataless merge", "DARE"),
    ("TaskArithmetic", "Dataless merge", "TaskArithmetic"),
    ("task_arith", "Dataless merge", "task_arith (cross-check)"),
    ("whc_diag", "Dataless merge", "HTCL"),
    ("ta_pe_inst0.8_codi0.4", "Dataless merge", "HTCL per-expert (champion)"),
    ("instruction_expert", "Specialist", "instruction expert"),
    ("math_expert", "Specialist", "math expert"),
    ("coding_expert", "Specialist", "coding expert"),
    ("safety_expert", "Specialist", "safety expert"),
    ("multilingual_expert", "Specialist", "multilingual expert"),
]

ML_ACC_NORM = {"arc", "hellaswag"}   # these report acc_norm; m_mmlu reports acc
ML_LANGS = ["fr", "es", "de", "ru"]
ML_STEMS = ["m_mmlu", "arc", "hellaswag"]
SAFETY_TASKS = ["wildguardtest", "harmbench", "do_anything_now", "xstest"]


def _newest_lm_results(model_dir: Path) -> Optional[dict]:
    """Newest lm-eval ``results_*.json`` anywhere under ``model_dir``."""
    cands = sorted(model_dir.rglob("results_*.json"),
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        return None
    with open(cands[-1]) as f:
        return json.load(f)


def _all_lm_results(model_dir: Path) -> Dict[str, dict]:
    """Merge the ``results`` blocks across every results_*.json (the three
    lm-eval groups write separate timestamped files into the same dir)."""
    out: Dict[str, dict] = {}
    for p in sorted(model_dir.rglob("results_*.json"),
                    key=lambda q: q.stat().st_mtime):
        try:
            with open(p) as f:
                res = json.load(f).get("results", {})
        except json.JSONDecodeError:
            continue
        out.update(res)   # newer files win on key collision
    return out


def _pct(x: Optional[float]) -> str:
    return "" if x is None else f"{100.0 * x:.1f}"


def _get_metric(block: dict, *names: str) -> Optional[float]:
    for n in names:
        if n in block:
            return float(block[n])
    return None


def parse_lm(model_dir: Path) -> Dict[str, Optional[float]]:
    res = _all_lm_results(model_dir)
    out: Dict[str, Optional[float]] = {"instruction": None, "math": None,
                                       "multilingual": None}
    if "gsm8k_cot" in res:
        out["math"] = _get_metric(res["gsm8k_cot"],
                                  "exact_match,flexible-extract",
                                  "exact_match,strict-match", "exact_match")
    if "ifeval" in res:
        out["instruction"] = _get_metric(res["ifeval"],
                                          "prompt_level_strict_acc,none",
                                          "prompt_level_strict_acc")
    ml_vals: List[float] = []
    for stem in ML_STEMS:
        for lang in ML_LANGS:
            task = f"{stem}_{lang}"
            if task not in res:
                continue
            if stem in ML_ACC_NORM:
                v = _get_metric(res[task], "acc_norm,none", "acc_norm",
                                "acc,none", "acc")
            else:
                v = _get_metric(res[task], "acc,none", "acc")
            if v is not None:
                ml_vals.append(v)
    if ml_vals:
        out["multilingual"] = sum(ml_vals) / len(ml_vals)
    return out


def parse_code(model_dir: Path) -> Dict[str, Optional[float]]:
    out = {"humanevalplus": None, "mbppplus": None}
    p = model_dir / "code_eval.json"
    if not p.exists():
        return out
    with open(p) as f:
        data = json.load(f)
    for task in ("humanevalplus", "mbppplus"):
        blk = data.get(task) or data.get(task.replace("plus", "+"))
        if isinstance(blk, dict):
            out[task] = blk.get("pass@1")
    return out


def parse_safety(model_dir: Path) -> Dict[str, Optional[float]]:
    """Best-effort safety parse. safety-eval report schemas differ across
    versions, so we walk the JSON and pull a plausible scalar per task. xstest
    is an accuracy; the other three are RTA (refuse-to-answer, higher = safer)."""
    out: Dict[str, Optional[float]] = {t: None for t in SAFETY_TASKS}
    p = model_dir / "safety_eval.json"
    if not p.exists():
        return out
    with open(p) as f:
        data = json.load(f)

    def _scalar(obj) -> Optional[float]:
        # Pull the first float-like value from a metric dict / number.
        if isinstance(obj, (int, float)):
            return float(obj)
        if isinstance(obj, dict):
            for key in ("RTA", "rta", "refusal", "accuracy", "acc",
                        "score", "overall"):
                if key in obj and isinstance(obj[key], (int, float)):
                    return float(obj[key])
            for v in obj.values():
                s = _scalar(v)
                if s is not None:
                    return s
        return None

    for task in SAFETY_TASKS:
        for key, val in data.items():
            if task in key.lower():
                out[task] = _scalar(val)
                break
    return out


def _safety_mean(s: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [v for v in s.values() if v is not None]
    return sum(vals) / len(vals) if vals else None


def _avg(*xs: Optional[float]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    base = cfg["base_name"]
    eval_root = ROOT / "results" / "mb_eval" / base
    out_rows: Dict[str, dict] = {}

    print(f"[tier2-table] reading {eval_root}\n", flush=True)
    for tag, category, label in ROW_ORDER:
        mdir = eval_root / tag
        if not mdir.exists():
            print(f"  [skip] {tag}: no results dir", flush=True)
            continue
        lm = parse_lm(mdir)
        code = parse_code(mdir)
        safety = parse_safety(mdir)
        coding_mean = _avg(code["humanevalplus"], code["mbppplus"])
        safety_mean = _safety_mean(safety)
        avg = _avg(lm["instruction"], lm["math"], coding_mean,
                   safety_mean, lm["multilingual"])
        out_rows[tag] = {
            "category": category, "label": label,
            "instruction": lm["instruction"], "math": lm["math"],
            "humanevalplus": code["humanevalplus"], "mbppplus": code["mbppplus"],
            "safety": safety_mean, "safety_breakdown": safety,
            "multilingual": lm["multilingual"], "avg": avg,
        }
        print(f"  {label:28s} instr={_pct(lm['instruction']):>5} "
              f"math={_pct(lm['math']):>5} heval+={_pct(code['humanevalplus']):>5} "
              f"mbpp+={_pct(code['mbppplus']):>5} safe={_pct(safety_mean):>5} "
              f"ml={_pct(lm['multilingual']):>5} avg={_pct(avg):>5}", flush=True)

    out_json = ROOT / "results" / "mergebench" / f"tier2_table_{base}.json"
    with open(out_json, "w") as f:
        json.dump(out_rows, f, indent=2)
    print(f"\n[json] {out_json}", flush=True)

    # HTML <tr> rows to paste into TIER1_TABLE.md's Tier-2 table.
    print("\n[html rows]\n")
    for tag, _cat, _lbl in ROW_ORDER:
        if tag not in out_rows:
            continue
        r = out_rows[tag]
        lbl = f"<b>{r['label']}</b>" if tag == "whc_diag" else r["label"]
        cells = [_pct(r["instruction"]), _pct(r["math"]),
                 _pct(r["humanevalplus"]), _pct(r["mbppplus"]),
                 _pct(r["safety"]), _pct(r["multilingual"]), _pct(r["avg"])]
        tds = "".join(f"<td>{c}</td>" for c in cells)
        print(f"<tr><td>{lbl}</td>{tds}</tr>")


if __name__ == "__main__":
    main()
