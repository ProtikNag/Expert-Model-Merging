"""Correctness tests for the data-using ``whc_gram`` merge.

No model download: builds tiny fake checkpoints + Gram dirs on disk and checks
the closed form, the fallbacks, and the lam/gamma limits. Run in the `merging`
env (needs torch + safetensors):

    python tests/test_whc_gram.py          # plain asserts, exits 0 on success
    pytest tests/test_whc_gram.py          # also works under pytest
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

_TOL = 1e-4


def _write_ckpt(d: Path, state: dict) -> str:
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in state.items()},
              str(d / "model.safetensors"), metadata={"format": "pt"})
    return str(d)


def _ref_whc_gram(w_experts, g_list, lam, eps=1e-12,
                  f_in_avg=None, gamma=0.0):
    """Independent reference for one Linear weight."""
    in_dim = w_experts[0].shape[1]
    g_sum = sum(g_list)
    w_bar = sum(w_experts) / len(w_experts)
    rhs = sum(w @ g for w, g in zip(w_experts, g_list)) + lam * w_bar
    lhs = g_sum + (lam + eps) * torch.eye(in_dim)
    if gamma != 0.0 and f_in_avg is not None:
        lhs = lhs + gamma * torch.diag(f_in_avg)
    return torch.linalg.solve(lhs, rhs.t()).t()


def _spd(in_dim: int, seed: int) -> torch.Tensor:
    """A symmetric positive-definite [in, in] Gram."""
    g = torch.randn(in_dim, in_dim, generator=torch.Generator().manual_seed(seed))
    return g @ g.t() + in_dim * torch.eye(in_dim)


def _run(tmp: Path, lam: float, *, gamma: float = 0.0, with_fisher: bool = False):
    torch.manual_seed(0)
    out_dim, in_dim, n = 2, 3, 2
    base = {"lin.weight": torch.zeros(out_dim, in_dim),
            "norm.weight": torch.zeros(in_dim),           # 1D -> mean fallback
            "buf": torch.arange(4, dtype=torch.int64)}     # non-float -> copied
    w_experts, g_list, f_list = [], [], []
    expert_dirs, gram_dirs, fisher_dirs = [], [], []
    for i in range(n):
        w = torch.randn(out_dim, in_dim)
        nrm = torch.randn(in_dim)
        w_experts.append(w)
        _write_ckpt(tmp / f"exp{i}", {"lin.weight": w, "norm.weight": nrm,
                                      "buf": torch.arange(4, dtype=torch.int64)})
        expert_dirs.append(str(tmp / f"exp{i}"))
        g = _spd(in_dim, seed=10 + i)
        g_list.append(g)
        _write_ckpt(tmp / f"gram{i}", {"lin.weight": g})
        gram_dirs.append(str(tmp / f"gram{i}"))
        if with_fisher:
            f = torch.rand(out_dim, in_dim) + 0.1
            f_list.append(f)
            _write_ckpt(tmp / f"fish{i}", {"lin.weight": f})
            fisher_dirs.append(str(tmp / f"fish{i}"))

    base_dir = _write_ckpt(tmp / "base", base)
    save_dir = str(tmp / "merged")
    merge_checkpoints(method="whc_gram", base_dir=base_dir,
                      expert_dirs=expert_dirs, save_dir=save_dir,
                      lam=lam, gamma=gamma, grams_dirs=gram_dirs,
                      fisher_dirs=(fisher_dirs if with_fisher else None))
    out = ShardedStateReader(save_dir)
    f_in_avg = (sum(f.mean(dim=0) for f in f_list) / n) if with_fisher else None
    ref = _ref_whc_gram(w_experts, g_list, lam, f_in_avg=f_in_avg, gamma=gamma)
    return out, w_experts, ref


def test_closed_form_matches_reference():
    with tempfile.TemporaryDirectory() as td:
        out, w_experts, ref = _run(Path(td), lam=1e-2)
        got = out.get("lin.weight").float()
        assert torch.allclose(got, ref, atol=_TOL), (got - ref).abs().max()
        # non-Linear float param -> ensemble mean
        norm_mean = out.get("norm.weight").float()
        # buf is non-float -> copied verbatim from base
        assert out.get("buf").dtype == torch.int64


def test_gamma_fisher_ridge():
    with tempfile.TemporaryDirectory() as td:
        out, _, ref = _run(Path(td), lam=1e-3, gamma=0.5, with_fisher=True)
        got = out.get("lin.weight").float()
        assert torch.allclose(got, ref, atol=_TOL), (got - ref).abs().max()


def test_large_lam_tends_to_mean():
    with tempfile.TemporaryDirectory() as td:
        out, w_experts, _ = _run(Path(td), lam=1e8)
        got = out.get("lin.weight").float()
        mean = sum(w_experts) / len(w_experts)
        assert torch.allclose(got, mean, atol=1e-3), (got - mean).abs().max()


def test_identity_gram_lam0_is_mean():
    """G_i = I, lam=0 -> RegMean = (sum W_i)(N I)^-1 = mean(W)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        torch.manual_seed(1)
        out_dim, in_dim, n = 2, 3, 3
        w_experts, expert_dirs, gram_dirs = [], [], []
        for i in range(n):
            w = torch.randn(out_dim, in_dim)
            w_experts.append(w)
            _write_ckpt(tmp / f"exp{i}", {"lin.weight": w})
            expert_dirs.append(str(tmp / f"exp{i}"))
            _write_ckpt(tmp / f"gram{i}", {"lin.weight": torch.eye(in_dim)})
            gram_dirs.append(str(tmp / f"gram{i}"))
        base_dir = _write_ckpt(tmp / "base", {"lin.weight": torch.zeros(out_dim, in_dim)})
        save_dir = str(tmp / "merged")
        merge_checkpoints(method="whc_gram", base_dir=base_dir,
                          expert_dirs=expert_dirs, save_dir=save_dir,
                          lam=0.0, grams_dirs=gram_dirs)
        got = ShardedStateReader(save_dir).get("lin.weight").float()
        mean = sum(w_experts) / n
        assert torch.allclose(got, mean, atol=1e-4), (got - mean).abs().max()


def _ab_merge(tmp: Path, **kwargs):
    """Two experts, key 'a.weight' has a Gram, key 'b.weight' does not.
    Returns (merged_reader, [(a,b) per expert], base)."""
    torch.manual_seed(2)
    out_dim, in_dim, n = 2, 3, 2
    we, expert_dirs, gram_dirs = [], [], []
    for i in range(n):
        w = torch.randn(out_dim, in_dim)
        w2 = torch.randn(out_dim, in_dim)                  # second Linear, NO gram
        we.append((w, w2))
        _write_ckpt(tmp / f"exp{i}", {"a.weight": w, "b.weight": w2})
        expert_dirs.append(str(tmp / f"exp{i}"))
        _write_ckpt(tmp / f"gram{i}", {"a.weight": _spd(in_dim, 5 + i)})
        gram_dirs.append(str(tmp / f"gram{i}"))
    base = {"a.weight": torch.zeros(out_dim, in_dim),
            "b.weight": torch.zeros(out_dim, in_dim)}
    base_dir = _write_ckpt(tmp / "base", base)
    save_dir = str(tmp / "merged")
    merge_checkpoints(method="whc_gram", base_dir=base_dir,
                      expert_dirs=expert_dirs, save_dir=save_dir,
                      grams_dirs=gram_dirs, **kwargs)
    return ShardedStateReader(save_dir), we


def test_missing_gram_falls_back_to_mean():
    """A Linear key absent from the Gram dir is averaged, not solved."""
    with tempfile.TemporaryDirectory() as td:
        out, we = _ab_merge(Path(td), lam=1e-3)            # default fallback=mean
        got_b = out.get("b.weight").float()
        mean_b = sum(b for _, b in we) / len(we)
        assert torch.allclose(got_b, mean_b, atol=1e-5), (got_b - mean_b).abs().max()


def test_task_arith_fallback():
    """gram_fallback='task_arith' makes non-Gram keys w_pre + scale*sum(tv).
    Base is 0, so out_b = scale * sum_i b_i."""
    with tempfile.TemporaryDirectory() as td:
        out, we = _ab_merge(Path(td), lam=1e-3,
                            gram_fallback="task_arith", scale=0.4)
        got_b = out.get("b.weight").float()
        ref_b = 0.4 * sum(b for _, b in we)                # w_pre=0
        assert torch.allclose(got_b, ref_b, atol=1e-5), (got_b - ref_b).abs().max()


def test_alpha_scales_all_keys():
    """alpha rescales the deviation from base for BOTH the solved key and the
    fallback key: out = w_pre + alpha*(out_base - w_pre), w_pre=0 here."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        out1, _ = _ab_merge(tmp / "a1", lam=1e-2, alpha=1.0)
        out3, _ = _ab_merge(tmp / "a3", lam=1e-2, alpha=3.0)
        for k in ("a.weight", "b.weight"):
            base_dev = out1.get(k).float()                 # w_pre=0 -> dev == value
            scaled = out3.get(k).float()
            assert torch.allclose(scaled, 3.0 * base_dev, atol=1e-4), \
                (k, (scaled - 3.0 * base_dev).abs().max())


if __name__ == "__main__":
    test_closed_form_matches_reference()
    test_gamma_fisher_ridge()
    test_large_lam_tends_to_mean()
    test_identity_gram_lam0_is_mean()
    test_missing_gram_falls_back_to_mean()
    test_task_arith_fallback()
    test_alpha_scales_all_keys()
    print("all whc_gram tests passed")
