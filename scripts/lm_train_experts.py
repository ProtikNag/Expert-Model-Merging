"""Fine-tune one LM expert per GLUE task and dump all merge-time artifacts.

Per task, we save under ``<ckpt_dir>/<task>/``:

- ``backbone.pt``  : backbone state dict (merge target).
- ``head.pt``      : task-specific head (never merged; reloaded at eval).
- ``fisher.pt``    : diagonal empirical Fisher over backbone params.
- ``grams.pt``     : per-Linear-layer Grams over backbone.
- ``train_hist.json`` : per-epoch loss/acc for audit.
- ``eval.json``    : expert's own val/test metrics for topline reference.

Also saves the shared pretrained backbone to ``<ckpt_dir>/pretrained_backbone.pt``.

Every scalar metric is streamed into the run's structured metrics JSONL so
downstream analysis needs no re-runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.glue_data import TASK_INFO, build_glue_loaders, load_tokenizer  # noqa: E402
from src.lm_models import build_encoder_classifier, pretrained_backbone_state_dict  # noqa: E402
from src.lm_train import (backbone_gradient_on_task,  # noqa: E402
                          collect_backbone_linear_grams,
                          diagonal_empirical_fisher, evaluate,
                          train_one_epoch)
from src.logging_utils import RunLogger  # noqa: E402
from src.metrics import task_metric  # noqa: E402
from src.utils import ensure_dir, load_config, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None,
                    help="Override compute device (default: auto).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = ensure_dir(cfg["paths"]["checkpoints"])
    logger = RunLogger(experiment=f"lm_train_experts_{cfg['experiment']}",
                       log_root=cfg["paths"]["logs"],
                       config_path=args.config)

    model_name = cfg["model"]["name"]
    print(f"[tokenizer] loading {model_name}")
    tokenizer = load_tokenizer(model_name)
    tokenizer.save_pretrained(ckpt_dir / "tokenizer")

    print(f"[pretrained] downloading backbone {model_name}")
    pretrained_bb = pretrained_backbone_state_dict(model_name)
    torch.save(pretrained_bb, ckpt_dir / "pretrained_backbone.pt")
    logger.record(event="pretrained_backbone_saved",
                  n_params=sum(v.numel() for v in pretrained_bb.values()))

    for task in cfg["data"]["tasks"]:
        info = TASK_INFO[task]
        task_ckpt = ensure_dir(ckpt_dir / task)
        print(f"\n[task={task}] num_labels={info['num_labels']}  "
              f"primary_metric={info['primary_metric']}")

        loaders = build_glue_loaders(
            task=task, tokenizer=tokenizer,
            max_length=cfg["data"]["max_length"],
            batch_size=cfg["data"]["batch_size"],
            train_subset=cfg["data"]["train_subset"],
            val_subset=cfg["data"]["val_subset"],
            seed=cfg["seed"],
        )
        print(f"  train={len(loaders.train.dataset)}  "
              f"val={len(loaders.val.dataset)}  "
              f"test={len(loaders.test.dataset)}")

        set_seed(cfg["seed"])   # reset for reproducibility per task
        model = build_encoder_classifier(model_name, loaders.num_labels,
                                         head_dropout=cfg["model"]["head_dropout"])
        model.to(device)
        optim = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])

        t0 = time.time()
        hist = {"train_loss": [], "train_acc": [], "val_primary": []}
        for ep in range(cfg["train"]["epochs"]):
            loss, acc = train_one_epoch(model, loaders.train, optim, device)
            v_logits, v_labels = evaluate(model, loaders.val, device)
            v_met = task_metric(task, v_logits, v_labels)
            hist["train_loss"].append(loss)
            hist["train_acc"].append(acc)
            hist["val_primary"].append(v_met["primary"])
            print(f"  epoch {ep}: train_loss={loss:.4f} train_acc={acc:.4f}"
                  f"  val_primary={v_met['primary']:.4f}")
            logger.record(event="epoch", task=task, epoch=ep,
                          train_loss=loss, train_acc=acc,
                          val_primary=v_met["primary"])
        dt_train = time.time() - t0

        # Own-model val and test evaluation (per-expert topline).
        v_logits, v_labels = evaluate(model, loaders.val, device)
        t_logits, t_labels = evaluate(model, loaders.test, device)
        v_met = task_metric(task, v_logits, v_labels)
        t_met = task_metric(task, t_logits, t_labels)
        print(f"  topline: val={v_met['primary']:.4f}  test={t_met['primary']:.4f}")

        # Diagonal Fisher over backbone on a small fisher-sized subset.
        print("  [fisher] computing diagonal empirical Fisher...")
        fisher = diagonal_empirical_fisher(
            model, loaders.train, device,
            n_samples=cfg["fisher"]["n_samples"],
        )

        # Grams for RegMean.
        print("  [grams] collecting Linear-layer Grams...")
        grams = collect_backbone_linear_grams(
            model, loaders.train, device,
            n_samples=cfg["grams"]["n_samples"],
        )

        # Gradient at fine-tuned optimum (sanity: should be near zero).
        print("  [grad-check] mean |g| at optimum (should be small)...")
        g_opt = backbone_gradient_on_task(
            model, loaders.train, device,
            n_samples=min(cfg["fisher"]["n_samples"], 256),
        )
        g_norm = float(torch.sqrt(
            sum((v ** 2).sum() for v in g_opt.values())).item())
        print(f"    ||g|| = {g_norm:.6f}")
        logger.record(event="grad_at_optimum", task=task, g_norm=g_norm)

        # Save everything.
        torch.save(model.backbone_state_dict(), task_ckpt / "backbone.pt")
        torch.save(model.head_state_dict(), task_ckpt / "head.pt")
        torch.save(fisher, task_ckpt / "fisher.pt")
        torch.save(grams, task_ckpt / "grams.pt")
        torch.save({k: v.detach().cpu() for k, v in g_opt.items()},
                   task_ckpt / "gradient.pt")
        with open(task_ckpt / "train_hist.json", "w") as f:
            json.dump({"history": hist, "t_train_s": dt_train,
                       "val": v_met, "test": t_met}, f, indent=2)
        logger.record(event="task_done", task=task,
                      val_primary=v_met["primary"],
                      test_primary=t_met["primary"],
                      train_time_s=dt_train)

        # Free memory before the next task.
        del model, optim, fisher, grams, g_opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[done] experts and artifacts -> {ckpt_dir}")
    logger.close()


if __name__ == "__main__":
    main()
