"""Run every merging method and evaluate on each rotation.

Reads expert checkpoints produced by ``scripts/train_experts.py``, runs the
merging methods (sweeping hyperparameters where applicable), and writes a
single ``results/pilot_results.json`` with all accuracies, runtimes, and
selected hyperparameters.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_loader  # noqa: E402
from src.merging.fisher_merge import fisher_merge  # noqa: E402
from src.merging.htcl import fisher_ridge, htcl_a, htcl_b  # noqa: E402
from src.merging.regmean import regmean_merge  # noqa: E402
from src.merging.simple import simple_average  # noqa: E402
from src.merging.task_arith import task_arithmetic  # noqa: E402
from src.merging.ties import ties_merging  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train import evaluate  # noqa: E402
from src.utils import ensure_dir, load_config, set_seed  # noqa: E402


def evaluate_on_all(model, rotations, cfg, device) -> Dict[int, float]:
    """Return per-rotation test accuracy for the given model state."""
    out: Dict[int, float] = {}
    for angle in rotations:
        loader = build_loader(
            cfg["dataset"]["root"], theta_deg=float(angle),
            train=False, batch_size=cfg["train"]["batch_size"],
            subset=cfg["train"]["test_subset"], seed=cfg["seed"],
        )
        out[int(angle)] = evaluate(model, loader, device)
    return out


def summarize(acc_by_angle: Dict[int, float]) -> Dict[str, float]:
    v = list(acc_by_angle.values())
    return {"avg": sum(v) / len(v),
            "min": min(v),
            "max": max(v),
            "std": float(torch.tensor(v).std(unbiased=False))}


def load_experts(ckpt_dir: Path, angles: List[int]):
    states, fishers, grams = [], [], []
    for a in angles:
        states.append(torch.load(ckpt_dir / f"expert_{a:03d}.pt",
                                 map_location="cpu", weights_only=True))
        fishers.append(torch.load(ckpt_dir / f"expert_{a:03d}_fisher.pt",
                                  map_location="cpu", weights_only=True))
        grams.append(torch.load(ckpt_dir / f"expert_{a:03d}_grams.pt",
                                map_location="cpu", weights_only=True))
    return states, fishers, grams


def run_method(method_name, build_fn, model, rotations, cfg, device):
    """Apply ``build_fn`` to get a merged state, time it, and evaluate."""
    t0 = time.time()
    merged_state = build_fn()
    dt = time.time() - t0
    model.load_state_dict(merged_state, strict=True)
    acc = evaluate_on_all(model, rotations, cfg, device)
    summ = summarize(acc)
    print(f"  {method_name:>28s}: avg={summ['avg']:.4f} "
          f"(min={summ['min']:.4f}, std={summ['std']:.4f}), t={dt:.2f}s")
    return {"per_rotation": acc, "summary": summ, "time_s": dt}


def sweep_best(method_name, build_fn_for_param, param_values,
               model, rotations, cfg, device):
    """Evaluate every hyperparameter value; return the one with best avg."""
    best = None
    curves = []
    for p in param_values:
        out = run_method(f"{method_name}(p={p})",
                         lambda p=p: build_fn_for_param(p),
                         model, rotations, cfg, device)
        out["param"] = p
        curves.append(out)
        if best is None or out["summary"]["avg"] > best["summary"]["avg"]:
            best = out
    return best, curves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cpu")
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    results_dir = ensure_dir(cfg["paths"]["results"])

    angles = list(cfg["dataset"]["rotations"])
    model = build_model(cfg["model"]).to(device)
    pretrained = torch.load(ckpt_dir / "pretrained.pt",
                            map_location="cpu", weights_only=True)
    states, fishers, grams_list = load_experts(ckpt_dir, angles)

    results: Dict[str, dict] = {}

    # ---- Sanity: pretrained (shared init) accuracy -------------------------
    print("[eval] shared initialization (pre-fine-tune) baseline...")
    model.load_state_dict(pretrained, strict=True)
    acc = evaluate_on_all(model, angles, cfg, device)
    results["pretrained"] = {"per_rotation": acc, "summary": summarize(acc),
                             "time_s": 0.0}

    # ---- Sanity: individual experts (diagonal = best achievable per rotation)
    print("[eval] individual experts (per-rotation topline)...")
    expert_diag_accs: Dict[int, float] = {}
    expert_matrix: Dict[int, Dict[int, float]] = {}
    for a, s in zip(angles, states):
        model.load_state_dict(s, strict=True)
        acc = evaluate_on_all(model, angles, cfg, device)
        expert_matrix[int(a)] = acc
        expert_diag_accs[int(a)] = acc[int(a)]
    results["experts"] = {"matrix": {str(k): v for k, v in expert_matrix.items()},
                          "diagonal_mean": sum(expert_diag_accs.values())
                                           / len(expert_diag_accs)}

    # ---- Simple averaging --------------------------------------------------
    print("[merge] simple averaging")
    results["simple"] = run_method(
        "simple", lambda: simple_average(states),
        model, angles, cfg, device)

    # ---- Task arithmetic (sweep scale) ------------------------------------
    print("[merge] task arithmetic (scale sweep)")
    best, sweep = sweep_best(
        "task_arith",
        lambda p: task_arithmetic(states, pretrained, scale=p),
        cfg["merging"]["task_arithmetic"]["scale_sweep"],
        model, angles, cfg, device)
    results["task_arithmetic"] = {"best": best, "sweep": sweep}

    # ---- TIES (sweep over k and scale) ------------------------------------
    print("[merge] TIES (k x scale grid)")
    best_ties, sweep_ties = None, []
    for k in cfg["merging"]["ties"]["k_sweep"]:
        for s_scale in cfg["merging"]["ties"]["scale_sweep"]:
            out = run_method(
                f"ties(k={k},s={s_scale})",
                lambda k=k, s_scale=s_scale: ties_merging(
                    states, pretrained, keep_frac=k, scale=s_scale),
                model, angles, cfg, device)
            out["param"] = {"keep_frac": k, "scale": s_scale}
            sweep_ties.append(out)
            if best_ties is None or out["summary"]["avg"] > best_ties["summary"]["avg"]:
                best_ties = out
    results["ties"] = {"best": best_ties, "sweep": sweep_ties}

    # ---- Fisher merging ---------------------------------------------------
    print("[merge] Fisher merging")
    results["fisher"] = run_method(
        "fisher", lambda: fisher_merge(states, fishers),
        model, angles, cfg, device)

    # ---- RegMean ----------------------------------------------------------
    print("[merge] RegMean")
    alpha = cfg["merging"]["regmean"]["alpha"]
    results["regmean"] = run_method(
        f"regmean(a={alpha})",
        lambda: regmean_merge(states, grams_list, alpha=alpha),
        model, angles, cfg, device)

    # ---- Fisher+epsI ablation (anchor vs. stabilization) ------------------
    print("[merge] Fisher+epsI ablation (origin-anchored)")
    best_fr, sweep_fr = sweep_best(
        "fisher_ridge",
        lambda p: fisher_ridge(states, fishers, eps=p),
        cfg["merging"]["htcl_a"]["lam_sweep"],
        model, angles, cfg, device)
    results["fisher_ridge"] = {"best": best_fr, "sweep": sweep_fr}

    # ---- HTCL-A (sweep lambda) --------------------------------------------
    print("[merge] HTCL-A (lambda sweep)")
    best_a, sweep_a = sweep_best(
        "htcl_a",
        lambda p: htcl_a(states, fishers, lam=p),
        cfg["merging"]["htcl_a"]["lam_sweep"],
        model, angles, cfg, device)
    results["htcl_a"] = {"best": best_a, "sweep": sweep_a}

    # ---- HTCL-B (hierarchical tree) ---------------------------------------
    print("[merge] HTCL-B (hierarchical tree)")
    hb = cfg["merging"]["htcl_b"]
    results["htcl_b"] = run_method(
        f"htcl_b(fanout={hb['fanout']},lam0={hb['lam0']},rho={hb['rho']})",
        lambda: htcl_b(states, fishers,
                       fanout=hb["fanout"], lam0=hb["lam0"], rho=hb["rho"]),
        model, angles, cfg, device)

    out_path = results_dir / "pilot_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] results -> {out_path}")


if __name__ == "__main__":
    main()
