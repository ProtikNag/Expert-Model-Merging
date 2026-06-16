"""Correctness tests for per-parameter update-scaling in ``whc_diag``.

No model download: builds tiny fake checkpoints on disk and pins the limit
behaviours of the ``coherence`` and ``consensus`` per-parameter scale modes
against the plain global-alpha closed form. Run in the `merging` env (needs
torch + safetensors):

    python tests/test_pscale.py            # plain asserts, exits 0 on success
    pytest tests/test_pscale.py            # also works under pytest
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mergebench.io_utils import ShardedStateReader  # noqa: E402
from mergebench.llm_merge import merge_checkpoints   # noqa: E402

_TOL = 1e-5


def _write_ckpt(d: Path, state: dict) -> str:
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in state.items()},
              str(d / "model.safetensors"), metadata={"format": "pt"})
    return str(d)


def _merge(tmp: Path, experts, *, base=None, lam=1e-12, **kwargs):
    """experts: list of dict[key->tensor]. base defaults to zeros."""
    keys = list(experts[0].keys())
    if base is None:
        base = {k: torch.zeros_like(experts[0][k]) for k in keys}
    base_dir = _write_ckpt(tmp / "base", base)
    expert_dirs = []
    for i, e in enumerate(experts):
        expert_dirs.append(_write_ckpt(tmp / f"exp{i}", e))
    save_dir = str(tmp / "merged")
    merge_checkpoints(method="whc_diag", base_dir=base_dir,
                      expert_dirs=expert_dirs, save_dir=save_dir,
                      lam=lam, **kwargs)
    return ShardedStateReader(save_dir)


def test_global_default_is_plain_closed_form():
    """pscale='global', alpha=1 leaves the curvature-weighted mean untouched."""
    with tempfile.TemporaryDirectory() as td:
        e = [{"w": torch.tensor([1.0, 2.0])},
             {"w": torch.tensor([3.0, -1.0])}]
        out = _merge(Path(td), e).get("w").float()
        # taskvec curvature F_i = tau_i^2, base 0 -> w_M = sum F_i w_i / sum F_i.
        w = torch.stack([torch.tensor([1.0, 2.0]), torch.tensor([3.0, -1.0])])
        f = w ** 2
        ref = (f * w).sum(0) / f.sum(0)
        assert torch.allclose(out, ref, atol=_TOL), (out - ref)


def test_coherence_one_recovers_global_alpha():
    """When every expert agrees on every param (coherence_p == 1), the
    coherence mode multiplies the update by exactly alpha_max."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # identical-sign experts -> coherence = 1 everywhere
        e = [{"w": torch.tensor([1.0, 2.0])},
             {"w": torch.tensor([2.0, 5.0])}]
        base_out = _merge(tmp / "g", e).get("w").float()             # alpha=1
        coh_out = _merge(tmp / "c", e, pscale="coherence",
                         alpha_max=3.0, beta=1.0).get("w").float()
        assert torch.allclose(coh_out, 3.0 * base_out, atol=_TOL), \
            (coh_out, base_out)


def test_coherence_zero_leaves_update_unscaled():
    """A perfectly conflicted param (tau = +c and -c) has coherence 0, so the
    coherence mode leaves a_p = 1 (no boost) regardless of alpha_max."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        e = [{"w": torch.tensor([4.0])}, {"w": torch.tensor([-4.0])}]
        g1 = _merge(tmp / "g", e).get("w").float()                  # alpha=1
        c = _merge(tmp / "c", e, pscale="coherence",
                   alpha_max=9.0, beta=1.0).get("w").float()
        assert torch.allclose(c, g1, atol=_TOL), (c, g1)


def test_consensus_single_expert_param_stays_unscaled():
    """A param moved by only one expert must NOT be boosted (a_p ~= 1): the
    curvature-weighted mean already equals that expert's task vector, so scaling
    it up would over-extrapolate beyond the expert. This is the property pure
    coherence lacks and consensus provides."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # param 0: only expert 0 active; param 1: only expert 1 active
        e = [{"w": torch.tensor([5.0, 0.0])},
             {"w": torch.tensor([0.0, 7.0])}]
        g1 = _merge(tmp / "g", e).get("w").float()
        cons = _merge(tmp / "c", e, pscale="consensus",
                      alpha_max=5.0).get("w").float()
        assert torch.allclose(cons, g1, atol=1e-4), (cons, g1)


def test_consensus_agreeing_experts_recover_the_sum():
    """When N experts agree on a param, consensus rescales the curvature mean up
    to the additive task-vector SUM of the agreeing experts (capped at
    alpha_max). With two equal-magnitude agreeing experts the update doubles."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        e = [{"w": torch.tensor([3.0])}, {"w": torch.tensor([3.0])}]
        g1 = _merge(tmp / "g", e).get("w").float()      # = 3.0 (mean of equals)
        cons = _merge(tmp / "c", e, pscale="consensus",
                      alpha_max=5.0).get("w").float()
        # consensus sum = 6.0; u = 3.0 -> a_p = 2 -> out = 6.0
        assert torch.allclose(cons, torch.tensor([6.0]), atol=1e-4), cons
        assert torch.allclose(g1, torch.tensor([3.0]), atol=1e-4), g1


def test_consensus_respects_alpha_max_cap():
    """Five agreeing equal experts would want a_p = 5; alpha_max=3 caps it."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        e = [{"w": torch.tensor([2.0])} for _ in range(5)]
        cons = _merge(tmp / "c", e, pscale="consensus",
                      alpha_max=3.0).get("w").float()
        # u = 2.0 (mean), consensus sum = 10.0, ratio 5 capped at 3 -> 6.0
        assert torch.allclose(cons, torch.tensor([6.0]), atol=1e-4), cons


if __name__ == "__main__":
    test_global_default_is_plain_closed_form()
    test_coherence_one_recovers_global_alpha()
    test_coherence_zero_leaves_update_unscaled()
    test_consensus_single_expert_param_stays_unscaled()
    test_consensus_agreeing_experts_recover_the_sum()
    test_consensus_respects_alpha_max_cap()
    print("all pscale tests passed")
