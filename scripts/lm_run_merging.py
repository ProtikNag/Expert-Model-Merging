"""Run every merging method on the fine-tuned GLUE experts and write results.

Fair protocol:
- Every method sees the same expert backbones, heads, Fishers, Grams.
- Hyperparameter selection uses **val** primary metric, averaged across
  tasks.
- Final reported metric is computed on **test** with the val-winning hparam.
- Sweep budget is controlled by config; currently ~8 configs per method.

All per-expert, per-method, per-hparam scalars are streamed to the run's
metrics JSONL so the downstream analysis stage never has to re-evaluate.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.glue_data import TASK_INFO, build_glue_loaders, load_tokenizer  # noqa: E402
from src.lm_models import build_encoder_classifier  # noqa: E402
from src.lm_train import (backbone_gradient_on_task,  # noqa: E402
                          collect_backbone_linear_grams, evaluate)
from src.logging_utils import RunLogger  # noqa: E402
from src.merging.fisher_merge import fisher_merge  # noqa: E402
from src.merging.regmean import regmean_merge  # noqa: E402
from src.merging.regmean_plus import regmean_plusplus_merge_simple  # noqa: E402
from src.merging.simple import simple_average  # noqa: E402
from src.merging.task_arith import task_arithmetic  # noqa: E402
from src.merging.ties import ties_merging  # noqa: E402
from src.merging.whc import whc_tree  # noqa: E402
from src.metrics import (aggregate_primary, curvature_stats,  # noqa: E402
                         param_space_summary, task_metric)
from src.utils import ensure_dir, load_config, set_seed  # noqa: E402


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def load_all_experts(ckpt_dir: Path, tasks: List[str]):
    pretrained = torch.load(ckpt_dir / "pretrained_backbone.pt",
                            map_location="cpu", weights_only=True)
    backbones, heads, fishers, grams = [], [], [], []
    for t in tasks:
        tdir = ckpt_dir / t
        backbones.append(torch.load(tdir / "backbone.pt",
                                    map_location="cpu", weights_only=True))
        heads.append(torch.load(tdir / "head.pt",
                                map_location="cpu", weights_only=True))
        fishers.append(torch.load(tdir / "fisher.pt",
                                  map_location="cpu", weights_only=True))
        grams.append(torch.load(tdir / "grams.pt",
                                map_location="cpu", weights_only=True))
    return pretrained, backbones, heads, fishers, grams


def load_or_compute_gradients(ckpt_dir: Path,
                              tasks: List[str],
                              model,
                              loaders_per_task: Dict[str, any],
                              backbones: List[Dict[str, torch.Tensor]],
                              heads: List[Dict[str, torch.Tensor]],
                              device: torch.device,
                              n_samples: int,
                              ) -> List[Dict[str, torch.Tensor]]:
    """Per-task gradient at the expert weights, cached at
    ``<ckpt_dir>/<task>/gradient.pt``. Computed lazily if missing.
    """
    out: List[Dict[str, torch.Tensor]] = []
    for i, task in enumerate(tasks):
        path = ckpt_dir / task / "gradient.pt"
        if path.exists():
            out.append(torch.load(path, map_location="cpu",
                                  weights_only=True))
            continue
        print(f"  [grad] computing for task={task}...")
        model.load_backbone_state_dict(backbones[i], strict=True)
        out_dim = heads[i]["weight"].shape[0]
        model.head = torch.nn.Linear(model.head.in_features, out_dim).to(device)
        model.load_head_state_dict(heads[i], strict=True)
        model.to(device)
        g = backbone_gradient_on_task(
            model, loaders_per_task[task].train, device, n_samples=n_samples,
        )
        g_cpu = {k: v.detach().cpu() for k, v in g.items()}
        torch.save(g_cpu, path)
        out.append(g_cpu)
    return out


def make_recollect_grams_fn(model,
                            tasks: List[str],
                            loaders_per_task: Dict[str, any],
                            heads: List[Dict[str, torch.Tensor]],
                            device: torch.device,
                            n_samples: int):
    """Return a callable that, given a merged backbone state, re-collects
    per-task input Grams on that backbone. Used by whc_tree(K>0).
    """
    def fn(merged_state: Dict[str, torch.Tensor]
           ) -> List[Dict[str, torch.Tensor]]:
        model.load_backbone_state_dict(merged_state, strict=True)
        model.to(device)
        new_grams: List[Dict[str, torch.Tensor]] = []
        for i, task in enumerate(tasks):
            out_dim = heads[i]["weight"].shape[0]
            model.head = torch.nn.Linear(
                model.head.in_features, out_dim).to(device)
            model.load_head_state_dict(heads[i], strict=True)
            grams_dev = collect_backbone_linear_grams(
                model, loaders_per_task[task].train, device,
                n_samples=n_samples,
            )
            new_grams.append({k: v.detach().cpu()
                              for k, v in grams_dev.items()})
        return new_grams
    return fn


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_merged_on_all(model,
                            merged_backbone: Dict[str, torch.Tensor],
                            heads: List[Dict[str, torch.Tensor]],
                            loaders_per_task: Dict[str, any],
                            split: str,
                            tasks: List[str],
                            device: torch.device,
                            ) -> Dict[str, Dict[str, float]]:
    """Load ``merged_backbone`` into ``model``, then for each task swap the
    head and evaluate on the requested split.
    """
    model.load_backbone_state_dict(merged_backbone, strict=True)
    model.to(device)
    out: Dict[str, Dict[str, float]] = {}
    for task, head in zip(tasks, heads):
        model.head = torch.nn.Linear(
            model.head.in_features, head["weight"].shape[0]
        ).to(device)
        model.load_head_state_dict(head, strict=True)
        loader = getattr(loaders_per_task[task], split)
        logits, labels = evaluate(model, loader, device)
        out[task] = task_metric(task, logits, labels)
    return out


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def expand_grid(spec: Dict[str, List]) -> List[Dict]:
    """Cartesian product of named grids into a list of config dicts."""
    keys = list(spec.keys())
    vals = [spec[k] for k in keys]
    combos = list(itertools.product(*vals))
    return [dict(zip(keys, c)) for c in combos]


def build_method_registry(cfg: Dict,
                          backbones: List[Dict[str, torch.Tensor]],
                          pretrained: Dict[str, torch.Tensor],
                          fishers: List[Dict[str, torch.Tensor]],
                          grams: List[Dict[str, torch.Tensor]],
                          curvatures_taskvec: List[Dict[str, torch.Tensor]],
                          gradients: List[Dict[str, torch.Tensor]],
                          recollect_grams_fn: Callable,
                          ) -> Dict[str, Tuple[Callable, List[Dict]]]:
    """Return a dict mapping method_name -> (merge_fn, hparam_grid).

    ``merge_fn(hparams) -> merged_backbone_state_dict``.
    """
    m = cfg["merging"]
    reg: Dict[str, Tuple[Callable, List[Dict]]] = {}

    reg["simple"] = (
        lambda h: simple_average(backbones),
        [{}],
    )
    reg["task_arith"] = (
        lambda h: task_arithmetic(backbones, pretrained, scale=h["scale"]),
        expand_grid({"scale": m["task_arith"]["scale_grid"]}),
    )
    reg["ties"] = (
        lambda h: ties_merging(backbones, pretrained,
                               keep_frac=h["keep_frac"], scale=h["scale"]),
        expand_grid({"keep_frac": m["ties"]["keep_frac_grid"],
                     "scale": m["ties"]["scale_grid"]}),
    )
    reg["fisher_merge"] = (
        lambda h: fisher_merge(backbones, fishers),
        [{}],
    )
    reg["regmean"] = (
        lambda h: regmean_merge(backbones, grams, alpha=h["alpha"]),
        expand_grid({"alpha": m["regmean"]["alpha_grid"]}),
    )
    reg["regmean_plus"] = (
        lambda h: regmean_plusplus_merge_simple(backbones, grams,
                                                alpha=h["alpha"]),
        expand_grid({"alpha": m["regmean_plus"]["alpha_grid"]}),
    )
    # WHC tree, base: λ only.
    reg["whc_tree"] = (
        lambda h: whc_tree(backbones, grams, lam=h["lam"]),
        expand_grid({"lam": m["whc_tree"]["lam_grid"]}),
    )
    # WHC tree + Fisher-anchored ridge (γ): whc.pdf Eq. 19/20 ablation.
    reg["whc_tree_fisher"] = (
        lambda h: whc_tree(backbones, grams, lam=h["lam"],
                           fishers=fishers, gamma=h["gamma"]),
        expand_grid({"lam":   m["whc_tree_fisher"]["lam_grid"],
                     "gamma": m["whc_tree_fisher"]["gamma_grid"]}),
    )
    # WHC tree + gradient correction (β): Eq. (27) -β α_i g_i term.
    reg["whc_tree_grad"] = (
        lambda h: whc_tree(backbones, grams, lam=h["lam"],
                           gradients=gradients, beta=h["beta"]),
        expand_grid({"lam":  m["whc_tree_grad"]["lam_grid"],
                     "beta": m["whc_tree_grad"]["beta_grid"]}),
    )
    # WHC tree, full Eq. (29): λ + γ + β jointly.
    reg["whc_tree_full"] = (
        lambda h: whc_tree(backbones, grams, lam=h["lam"],
                           fishers=fishers, gamma=h["gamma"],
                           gradients=gradients, beta=h["beta"]),
        expand_grid({"lam":   m["whc_tree_full"]["lam_grid"],
                     "gamma": m["whc_tree_full"]["gamma_grid"],
                     "beta":  m["whc_tree_full"]["beta_grid"]}),
    )
    # WHC tree, iterative re-linearization: refresh Grams K times.
    reg["whc_tree_iter"] = (
        lambda h: whc_tree(backbones, grams, lam=h["lam"],
                           fishers=fishers, gamma=h["gamma"],
                           gradients=gradients, beta=h["beta"],
                           K=h["K"], recollect_grams_fn=recollect_grams_fn),
        expand_grid({"lam":   m["whc_tree_iter"]["lam_grid"],
                     "gamma": m["whc_tree_iter"]["gamma_grid"],
                     "beta":  m["whc_tree_iter"]["beta_grid"],
                     "K":     m["whc_tree_iter"]["K_grid"]}),
    )
    return reg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", default=None,
                    help="Comma-separated subset of method names to run.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    results_dir = ensure_dir(cfg["paths"]["results"])
    figures_dir = ensure_dir(cfg["paths"]["figures"])
    ensure_dir(Path(figures_dir) / "png")
    ensure_dir(Path(figures_dir) / "svg")

    logger = RunLogger(experiment=f"lm_run_merging_{cfg['experiment']}",
                       log_root=cfg["paths"]["logs"],
                       config_path=args.config)

    tasks = cfg["data"]["tasks"]
    tokenizer = load_tokenizer(cfg["model"]["name"])

    # Build per-task loaders once (reused across all methods).
    loaders_per_task = {}
    for task in tasks:
        loaders_per_task[task] = build_glue_loaders(
            task=task, tokenizer=tokenizer,
            max_length=cfg["data"]["max_length"],
            batch_size=cfg["data"]["batch_size"],
            train_subset=cfg["data"]["train_subset"],
            val_subset=cfg["data"]["val_subset"],
            seed=cfg["seed"],
        )

    # Load expert artifacts.
    pretrained, backbones, heads, fishers, grams = load_all_experts(
        ckpt_dir, tasks)

    # Curvature summary (audit).
    cv_task = [
        {k: (b[k] - pretrained[k]) ** 2
         for k in pretrained if pretrained[k].dtype.is_floating_point}
        for b in backbones
    ]
    logger.record(event="curvature_stats",
                  source="taskvec",
                  stats=curvature_stats(cv_task))
    logger.record(event="curvature_stats",
                  source="fisher",
                  stats=curvature_stats(fishers))

    # Build a fresh model (used just as a shell for loading states and evaluating).
    # We instantiate with a max-label head; it will be replaced per-task.
    max_labels = max(h["weight"].shape[0] for h in heads)
    model = build_encoder_classifier(cfg["model"]["name"], max_labels,
                                     head_dropout=cfg["model"]["head_dropout"])
    model.to(device)

    # Sanity: expert topline (load each task's own backbone+head, test set).
    print("\n[topline] per-task evaluation of each expert's own model")
    topline: Dict[str, Dict[str, float]] = {}
    for task, bb, hd in zip(tasks, backbones, heads):
        metrics = _evaluate_merged_on_all(
            model, bb, [hd], {task: loaders_per_task[task]}, "test",
            [task], device)
        topline[task] = metrics[task]
        print(f"  {task}: primary={metrics[task]['primary']:.4f}")
        logger.record(event="topline", task=task, **metrics[task])
    topline_avg = aggregate_primary(topline)

    # Shared init reference.
    print("[baseline] shared-init (no merging) primary scores")
    init_metrics = _evaluate_merged_on_all(
        model, pretrained, heads, loaders_per_task, "test", tasks, device)
    init_avg = aggregate_primary(init_metrics)
    print(f"  avg primary = {init_avg:.4f}")
    logger.record(event="shared_init", per_task=init_metrics, avg=init_avg)

    # Per-task gradients at expert weights, cached on disk (whc.pdf Eq. 27 -β g term).
    print("\n[grad] loading or computing per-task gradients at expert weights")
    n_grad_samples = cfg.get("fisher", {}).get("n_samples", 1024)
    gradients = load_or_compute_gradients(
        ckpt_dir, tasks, model, loaders_per_task, backbones, heads,
        device, n_samples=min(n_grad_samples, 1024),
    )

    # Recollect-Grams callback used by whc_tree_iter (whc.pdf §7.2 Fix 3).
    recollect_n = cfg.get("grams", {}).get("recollect_n_samples", 256)
    recollect_grams_fn = make_recollect_grams_fn(
        model, tasks, loaders_per_task, heads, device,
        n_samples=recollect_n,
    )

    # Build the method registry and run.
    registry = build_method_registry(cfg, backbones, pretrained, fishers,
                                     grams, cv_task, gradients,
                                     recollect_grams_fn)
    only = set(args.only.split(",")) if args.only else None

    results: Dict[str, Dict] = {
        "topline": {"per_task": topline, "avg": topline_avg},
        "shared_init": {"per_task": init_metrics, "avg": init_avg},
        "methods": {},
    }

    for name, (merge_fn, grid) in registry.items():
        if only is not None and name not in only:
            continue
        print(f"\n[method={name}] sweeping {len(grid)} hparam(s)...")
        sweep_records: List[Dict] = []
        best = None
        for hp in grid:
            t0 = time.time()
            merged = merge_fn(hp)
            merge_t = time.time() - t0
            val_metrics = _evaluate_merged_on_all(
                model, merged, heads, loaders_per_task, "val", tasks, device)
            val_avg = aggregate_primary(val_metrics)
            rec = {
                "hp": hp, "val_per_task": val_metrics, "val_avg": val_avg,
                "merge_time_s": merge_t,
            }
            sweep_records.append(rec)
            logger.record(event="sweep", method=name, **rec)
            print(f"  hp={hp}  val_avg={val_avg:.4f}  t={merge_t:.2f}s")
            if best is None or val_avg > best["val_avg"]:
                best = rec

        # Apply winning hparam -> test.
        merged = merge_fn(best["hp"])
        test_metrics = _evaluate_merged_on_all(
            model, merged, heads, loaders_per_task, "test", tasks, device)
        test_avg = aggregate_primary(test_metrics)
        param_stats = param_space_summary(merged, pretrained, backbones)

        results["methods"][name] = {
            "best_hp": best["hp"],
            "val_per_task": best["val_per_task"],
            "val_avg": best["val_avg"],
            "test_per_task": test_metrics,
            "test_avg": test_avg,
            "param_stats": param_stats,
            "sweep": sweep_records,
        }
        logger.record(event="method_done", method=name,
                      best_hp=best["hp"], val_avg=best["val_avg"],
                      test_avg=test_avg, param_stats=param_stats)
        print(f"  [best] hp={best['hp']}  val={best['val_avg']:.4f}"
              f"  test={test_avg:.4f}")

    out_path = Path(results_dir) / "merge_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[done] {out_path}")
    logger.close()


if __name__ == "__main__":
    main()
