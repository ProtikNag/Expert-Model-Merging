"""Train a shared initialization and N RotatedMNIST experts on CPU.

Artifacts written to ``checkpoints/``:

- ``pretrained.pt``          : state dict of the shared init.
- ``expert_<angle>.pt``      : fine-tuned expert for each rotation angle.
- ``expert_<angle>_fisher.pt`` : diagonal Fisher for each expert.
- ``expert_<angle>_grams.pt`` : per-Linear-layer Gram matrices for RegMean.
- ``meta.pt``                : bookkeeping (angles, accuracies, seed, etc.).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_loader, build_pretrain_loader  # noqa: E402
from src.fisher import diagonal_empirical_fisher  # noqa: E402
from src.merging.regmean import collect_linear_grams  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train import evaluate, fine_tune, train_one_epoch  # noqa: E402
from src.utils import ensure_dir, load_config, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cpu")
    ckpt_dir = ensure_dir(cfg["paths"]["checkpoints"])

    # ---- 1. Shared init: train on un-rotated MNIST for a couple of epochs.
    print("[pretrain] training shared initialization on clean MNIST...")
    base_model = build_model(cfg["model"]).to(device)
    pre_loader = build_pretrain_loader(
        cfg["dataset"]["root"],
        batch_size=cfg["train"]["batch_size"],
        subset=cfg["train"]["train_subset"],
        seed=cfg["seed"],
    )
    optim = torch.optim.Adam(base_model.parameters(), lr=cfg["train"]["lr"])
    for ep in range(cfg["train"]["pretrain_epochs"]):
        t0 = time.time()
        loss, acc = train_one_epoch(base_model, pre_loader, optim, device)
        print(f"  pre epoch {ep}: loss={loss:.4f} acc={acc:.4f} "
              f"({time.time() - t0:.1f}s)")
    torch.save(base_model.state_dict(), ckpt_dir / "pretrained.pt")
    pretrained_state = copy.deepcopy(base_model.state_dict())

    # ---- 2. For each rotation angle, fine-tune an expert from the shared init.
    meta = {"angles": [], "expert_acc_own": {}, "expert_acc_matrix": {}}
    for angle in cfg["dataset"]["rotations"]:
        print(f"[expert {angle:>3d}] fine-tuning from shared init...")
        set_seed(cfg["seed"] + angle)   # different fine-tuning noise per expert
        model = build_model(cfg["model"]).to(device)
        model.load_state_dict(pretrained_state)
        loader = build_loader(
            cfg["dataset"]["root"], theta_deg=float(angle),
            train=True, batch_size=cfg["train"]["batch_size"],
            subset=cfg["train"]["train_subset"], seed=cfg["seed"] + angle,
        )
        fine_tune(model, loader,
                  epochs=cfg["train"]["expert_epochs"],
                  lr=cfg["train"]["lr"],
                  weight_decay=cfg["train"]["weight_decay"],
                  device=device)
        # Evaluate expert on its own rotation (sanity).
        test_loader = build_loader(
            cfg["dataset"]["root"], theta_deg=float(angle),
            train=False, batch_size=cfg["train"]["batch_size"],
            subset=cfg["train"]["test_subset"], seed=cfg["seed"],
        )
        acc_own = evaluate(model, test_loader, device)
        print(f"  expert {angle}: own-rotation test acc = {acc_own:.4f}")

        # Diagonal empirical Fisher on the expert's training data.
        fisher_loader = build_loader(
            cfg["dataset"]["root"], theta_deg=float(angle),
            train=True, batch_size=1,
            subset=cfg["fisher"]["n_samples"], seed=cfg["seed"] + angle,
        )
        fisher = diagonal_empirical_fisher(
            model, fisher_loader, device,
            n_samples=cfg["fisher"]["n_samples"],
        )

        # Linear-layer Grams for RegMean.
        gram_loader = build_loader(
            cfg["dataset"]["root"], theta_deg=float(angle),
            train=True, batch_size=cfg["train"]["batch_size"],
            subset=cfg["merging"]["regmean"]["n_activation_samples"],
            seed=cfg["seed"] + angle,
        )
        grams = collect_linear_grams(
            model, gram_loader, device,
            n_samples=cfg["merging"]["regmean"]["n_activation_samples"],
        )

        torch.save(model.state_dict(), ckpt_dir / f"expert_{angle:03d}.pt")
        torch.save(fisher, ckpt_dir / f"expert_{angle:03d}_fisher.pt")
        torch.save(grams, ckpt_dir / f"expert_{angle:03d}_grams.pt")
        meta["angles"].append(int(angle))
        meta["expert_acc_own"][str(angle)] = acc_own

    torch.save(meta, ckpt_dir / "meta.pt")
    # Human-readable copy.
    with open(ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] checkpoints written to {ckpt_dir}")


if __name__ == "__main__":
    main()
