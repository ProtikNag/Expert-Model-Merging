"""CPU smoke test for the MergeBench Tier 0 / Tier 1 tooling.

Builds tiny synthetic safetensors checkpoints (a few small tensors that mimic
a model's key layout), then exercises the divergence diagnostic and all merge
methods, checking the merged outputs against hand-computed references. No
network, no GPU, runs in seconds. Use this to confirm the logic is correct
before pulling on HPC.

    python scripts/mb_smoke_test.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safetensors.torch import save_file  # noqa: E402

from mergebench.divergence import compute_divergence  # noqa: E402
from mergebench.io_utils import ShardedStateReader  # noqa: E402
from mergebench.llm_merge import merge_checkpoints  # noqa: E402


def _write_ckpt(path: Path, tensors: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()},
              str(path / "model.safetensors"), metadata={"format": "pt"})
    # Minimal aux file so copy_aux_files has something to copy.
    (path / "config.json").write_text('{"model_type": "smoke"}')


def main() -> None:
    torch.manual_seed(0)
    tmp = Path(tempfile.mkdtemp(prefix="mb_smoke_"))
    print(f"[smoke] workdir {tmp}", flush=True)

    # Two float weights (one "linear", one "embed") + one int buffer.
    base = {
        "model.layer.weight": torch.zeros(4, 3, dtype=torch.bfloat16),
        "model.embed.weight": torch.zeros(5, 3, dtype=torch.bfloat16),
        "model.step": torch.tensor([0], dtype=torch.long),
    }
    e1 = {
        "model.layer.weight": torch.ones(4, 3, dtype=torch.bfloat16),
        "model.embed.weight": torch.full((5, 3), 2.0, dtype=torch.bfloat16),
        "model.step": torch.tensor([1], dtype=torch.long),
    }
    e2 = {
        "model.layer.weight": torch.full((4, 3), 3.0, dtype=torch.bfloat16),
        "model.embed.weight": torch.full((5, 3), -2.0, dtype=torch.bfloat16),
        "model.step": torch.tensor([2], dtype=torch.long),
    }
    _write_ckpt(tmp / "base", base)
    _write_ckpt(tmp / "e1", e1)
    _write_ckpt(tmp / "e2", e2)
    base_dir, expert_dirs = str(tmp / "base"), [str(tmp / "e1"), str(tmp / "e2")]

    failures = []

    def check(name: str, cond: bool) -> None:
        status = "ok" if cond else "FAIL"
        print(f"  [{status}] {name}", flush=True)
        if not cond:
            failures.append(name)

    # --- divergence ---------------------------------------------------------
    div = compute_divergence(base_dir, expert_dirs, ["e1", "e2"])
    # tau_e1 = ones (layer) + 2 (embed): ||tau||^2 = 4*3*1 + 5*3*4 = 72
    check("divergence ||tau_e1||", abs(div["norm_taskvec"]["e1"] - 72 ** 0.5) < 1e-3)
    # cos(tau_e1, tau_e2): dot = 12*(1*3) + 15*(2*-2) = 36 - 60 = -24
    # ||tau_e2||^2 = 12*9 + 15*4 = 168 ; cos = -24 / sqrt(72*168)
    expected_cos = -24.0 / (72 * 168) ** 0.5
    check("divergence cos(e1,e2)",
          abs(div["cosine_matrix"][0][1] - expected_cos) < 1e-3)

    # --- simple -------------------------------------------------------------
    merge_checkpoints("simple", base_dir, expert_dirs, str(tmp / "m_simple"))
    r = ShardedStateReader(str(tmp / "m_simple"))
    layer = r.get("model.layer.weight").float()
    check("simple layer == mean(1,3)=2", torch.allclose(layer, torch.full((4, 3), 2.0), atol=1e-2))
    check("simple keeps int buffer", r.get("model.step").item() == 0)

    # --- task_arith (scale=0.5) --------------------------------------------
    merge_checkpoints("task_arith", base_dir, expert_dirs, str(tmp / "m_ta"),
                      scale=0.5)
    r = ShardedStateReader(str(tmp / "m_ta"))
    layer = r.get("model.layer.weight").float()
    # w_pre=0 + 0.5*(1+3) = 2.0
    check("task_arith layer == 2.0", torch.allclose(layer, torch.full((4, 3), 2.0), atol=1e-2))

    # --- whc_diag (taskvec) -------------------------------------------------
    # layer: F1=(1)^2=1, F2=(3)^2=9 ; w_bar=2 ; lam small
    # num = 1*1 + 9*3 + lam*2 = 28 + 2*lam ; den = 1+9+lam = 10+lam
    # ~ 2.8 as lam->0
    merge_checkpoints("whc_diag", base_dir, expert_dirs, str(tmp / "m_whc"),
                      lam=1e-4, curvature="taskvec")
    r = ShardedStateReader(str(tmp / "m_whc"))
    layer = r.get("model.layer.weight").float()
    check("whc_diag layer ~ 2.8", torch.allclose(layer, torch.full((4, 3), 2.8), atol=2e-2))

    # whc_diag with huge lam -> simple mean (2.0)
    merge_checkpoints("whc_diag", base_dir, expert_dirs, str(tmp / "m_whc_big"),
                      lam=1e6, curvature="taskvec")
    r = ShardedStateReader(str(tmp / "m_whc_big"))
    layer = r.get("model.layer.weight").float()
    check("whc_diag(lam->inf) -> mean 2.0",
          torch.allclose(layer, torch.full((4, 3), 2.0), atol=1e-2))

    print("\n[smoke] " + ("ALL PASS" if not failures
                          else f"FAILURES: {failures}"), flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
