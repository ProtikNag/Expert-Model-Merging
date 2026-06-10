"""Parameter-space merging of large HuggingFace causal LMs.

Merge rules, all memory-bounded (one parameter key at a time), so they scale
to billion-parameter models:

- ``simple``      : ensemble mean of the experts.
- ``task_arith``  : w_pre + scale * sum_i (w_i - w_pre).
- ``whc_diag``    : the diagonal WHC closed form,

      w_M = (sum_i F_i * w_i + lam * w_bar) / (sum_i F_i + lam),

  with per-parameter curvature F_i. With ``curvature="taskvec"`` (default,
  dataless) F_i = (w_i - w_pre)^2; with ``curvature="fisher"`` the caller
  supplies a diagonal Fisher per expert. As lam -> inf the result tends to
  the simple mean; as lam -> 0 it tends to the curvature-weighted mean. This
  is the N-expert closed form (CRL Eq. 9) specialized to diagonal curvature
  with an ensemble-mean anchor.
- ``whc_gram``    : the *data-using* layer-wise WHC merge (the LLM port of the
  GLUE ``whc_tree`` winner). For each Linear weight ``W_i in R^{out x in}`` with
  input Gram ``G_i = (1/n) sum_t x_t x_t^T in R^{in x in}`` (collected on the
  expert's own domain data), the N-expert single-pass closed form is

      W_M = (sum_i W_i G_i + lam * W_bar)(sum_i G_i + lam I + gamma diag(F_in))^{-1},

  with ``W_bar`` the ensemble mean and ``F_in`` the input-dim projection of the
  diagonal Fisher (optional, gamma>0). As lam -> 0 this is exactly RegMean
  (Jin et al. 2023); the ``lam W_bar`` ridge pulls the solve toward the mean
  (lam -> inf gives the simple mean), and the gamma term injects curvature that
  pure-Gram RegMean discards. Non-Linear params and any key lacking a Gram fall
  back to the ensemble mean (RegMean convention). This is the head-to-head
  competitor to RegMean / RegMean++ / Fisher merging.

Each routine streams over the base model's keys, merges float parameters
whose shapes agree across base and all experts, and copies the base tensor
for everything else (e.g. vocab-sized embeddings that differ across
finetunes, integer buffers). Merge arithmetic is done in float32 and the
result is cast back to the base tensor's dtype (typically bfloat16).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .io_utils import ShardedStateReader, save_merged

_EPS = 1e-12


def _shapes_agree(key: str,
                  base: ShardedStateReader,
                  experts: List[ShardedStateReader]) -> bool:
    """True if ``key`` exists in every expert with the base's shape."""
    base_shape = base.get(key).shape
    for e in experts:
        if not e.has(key) or e.get(key).shape != base_shape:
            return False
    return True


def _is_float(t: torch.Tensor) -> bool:
    return t.dtype.is_floating_point


def _project_fisher_to_in_dim(fisher_t: torch.Tensor) -> torch.Tensor:
    """Project a diagonal Fisher for a Linear weight (shape ``[out, in]``) to
    the input dimension (shape ``[in]``) by averaging over the output dim.

    The layer-wise ridge in ``whc_gram`` lives in input-dim space (we solve an
    ``[in, in]`` system per layer), so the per-parameter Fisher diagonal is
    collapsed to a per-input-dimension prior strength that adds cleanly to the
    Tikhonov term. Mirrors ``src/merging/whc.py::_project_fisher_to_in_dim``.
    """
    return fisher_t.mean(dim=0)


def merge_checkpoints(method: str,
                      base_dir: str,
                      expert_dirs: List[str],
                      save_dir: str,
                      *,
                      scale: float = 0.4,
                      lam: float = 1e-4,
                      alpha: float = 1.0,
                      gamma: float = 0.0,
                      curvature: str = "taskvec",
                      fisher_dirs: Optional[List[str]] = None,
                      grams_dirs: Optional[List[str]] = None,
                      log_every: int = 50) -> Dict[str, float]:
    """Merge ``expert_dirs`` into a single HF checkpoint at ``save_dir``.

    Parameters
    ----------
    method:
        One of ``"simple"``, ``"task_arith"``, ``"whc_diag"``.
    base_dir:
        Local path to the shared pretrained base checkpoint.
    expert_dirs:
        Local paths to the fine-tuned expert checkpoints.
    save_dir:
        Output directory for the merged HF model.
    scale:
        Task-arithmetic scaling coefficient (``task_arith`` only).
    lam:
        Tikhonov / anchor coefficient (``whc_diag`` and ``whc_gram``). For
        ``whc_gram`` it is the weight on the ridge that pulls the layer-wise
        solve toward the ensemble mean (``lam=0`` recovers plain RegMean).
    gamma:
        Fisher-anchored ridge weight for ``whc_gram`` only. When ``gamma>0`` and
        ``fisher_dirs`` is supplied, adds ``gamma * diag(F_in)`` to the
        ``[in, in]`` ridge, injecting curvature that pure-Gram RegMean discards.
    alpha:
        Task-vector scale for ``whc_diag`` only. The closed form returns a
        curvature-weighted *mean* of the experts, which dilutes each expert's
        update by ~1/N relative to a *sum* of task vectors (task arithmetic).
        ``alpha`` rescales the net deviation from base, ``w_M = w_pre + alpha *
        (w_M^HTCL - w_pre)``, so ``alpha=1`` is the plain closed form and
        ``alpha>1`` compensates the averaging dilution (try ``alpha ~ N``). This
        is the analogue of task arithmetic's scaling coefficient.
    curvature:
        ``"taskvec"`` (dataless squared task vector) or ``"fisher"``
        (``whc_diag`` only).
    fisher_dirs:
        When ``curvature="fisher"`` (``whc_diag``/``fisher_merge``) or
        ``gamma>0`` (``whc_gram``), local paths to per-expert diagonal Fisher
        checkpoints (same key layout as the models).
    grams_dirs:
        When ``method="whc_gram"``, local paths to per-expert input-Gram
        checkpoints (``model.safetensors`` keyed by the Linear weight name,
        each value an ``[in, in]`` matrix). Produced by
        ``scripts/mb_gram_estimate.py``.
    log_every:
        Print progress every this many merged keys.

    Returns
    -------
    Dict[str, float]
        Summary counters (merged vs. copied keys).
    """
    base = ShardedStateReader(base_dir)
    experts = [ShardedStateReader(d) for d in expert_dirs]
    needs_fisher = (fisher_dirs is not None and
                    (method == "fisher_merge"
                     or (method == "whc_diag" and curvature == "fisher")
                     or (method == "whc_gram" and gamma != 0.0)))
    fishers = ([ShardedStateReader(d) for d in fisher_dirs]
               if needs_fisher else None)
    if method == "whc_gram":
        if grams_dirs is None:
            raise ValueError("whc_gram requires grams_dirs.")
        grams = [ShardedStateReader(d) for d in grams_dirs]
    else:
        grams = None
    n = len(experts)

    keys = base.keys()
    merged: Dict[str, torch.Tensor] = {}
    n_merged, n_copied = 0, 0

    for idx, key in enumerate(keys):
        base_t = base.get(key)
        # Non-float or shape-mismatched params: copy the base tensor as-is.
        if not _is_float(base_t) or not _shapes_agree(key, base, experts):
            merged[key] = base_t.clone()
            n_copied += 1
        else:
            out_dtype = base_t.dtype
            w_pre = base_t.float()
            w_experts = [e.get(key).float() for e in experts]

            if method == "simple":
                acc = torch.zeros_like(w_pre)
                for w in w_experts:
                    acc += w
                out = acc / n

            elif method == "task_arith":
                tv_sum = torch.zeros_like(w_pre)
                for w in w_experts:
                    tv_sum += (w - w_pre)
                out = w_pre + scale * tv_sum

            elif method == "whc_diag":
                if fishers is not None:
                    curv = [f.get(key).float() for f in fishers]
                else:
                    curv = [(w - w_pre) ** 2 for w in w_experts]
                w_bar = torch.zeros_like(w_pre)
                for w in w_experts:
                    w_bar += w
                w_bar /= n
                num = lam * w_bar
                den = torch.full_like(w_pre, lam)
                for f_i, w_i in zip(curv, w_experts):
                    num += f_i * w_i
                    den += f_i
                out = num / (den + _EPS)
                # Rescale the net update away from base. alpha=1 is the plain
                # closed form; alpha>1 undoes the ~1/N averaging dilution so the
                # merged model applies more of each expert's task vector.
                if alpha != 1.0:
                    out = w_pre + alpha * (out - w_pre)

            elif method == "fisher_merge":
                # Plain Fisher-weighted average (Matena & Raffel 2022), NO
                # anchor: w_M = sum_i F_i w_i / sum_i F_i. Where the Fisher is
                # ~0 for every expert (the divide-by-zero collapse they patch by
                # "defaulting to a target model"), fall back to the ensemble mean.
                if fishers is None:
                    raise ValueError("fisher_merge requires fisher_dirs.")
                curv = [f.get(key).float() for f in fishers]
                num = torch.zeros_like(w_pre)
                den = torch.zeros_like(w_pre)
                for f_i, w_i in zip(curv, w_experts):
                    num += f_i * w_i
                    den += f_i
                w_bar = torch.zeros_like(w_pre)
                for w in w_experts:
                    w_bar += w
                w_bar /= n
                out = torch.where(den > 1e-8, num / (den + _EPS), w_bar)

            elif method == "whc_gram":
                # Data-using layer-wise WHC (RegMean + ridge-toward-mean +
                # optional Fisher ridge). Only 2D Linear weights that every
                # expert has a Gram for go through the [in, in] solve; all other
                # float params (norms, biases, embeddings, lm_head, and any key
                # without a Gram) fall back to the ensemble mean.
                have_gram = (w_pre.dim() == 2
                             and all(g.has(key) for g in grams))
                if not have_gram:
                    acc = torch.zeros_like(w_pre)
                    for w in w_experts:
                        acc += w
                    out = acc / n
                else:
                    in_dim = w_pre.shape[1]
                    g_list = [g.get(key).float() for g in grams]
                    g_sum = torch.zeros((in_dim, in_dim), dtype=torch.float32)
                    rhs = torch.zeros_like(w_pre)
                    w_bar = torch.zeros_like(w_pre)
                    for w_i, g_i in zip(w_experts, g_list):
                        g_sum += g_i
                        rhs += w_i @ g_i          # [out,in] @ [in,in]
                        w_bar += w_i
                    w_bar /= n
                    rhs += lam * w_bar
                    eye = torch.eye(in_dim, dtype=torch.float32)
                    lhs = g_sum + (lam + _EPS) * eye
                    if gamma != 0.0 and fishers is not None:
                        f_in = torch.zeros(in_dim, dtype=torch.float32)
                        for f in fishers:
                            f_in += _project_fisher_to_in_dim(f.get(key).float())
                        f_in /= n
                        lhs += gamma * torch.diag(f_in)
                    # lhs is symmetric PSD; solve lhs X^T = rhs^T, transpose back.
                    out = torch.linalg.solve(lhs, rhs.t()).t()

            else:
                raise ValueError(f"Unknown merge method: {method}")

            merged[key] = out.to(out_dtype)
            n_merged += 1

        if (idx + 1) % log_every == 0 or (idx + 1) == len(keys):
            print(f"  [{method}] {idx + 1}/{len(keys)} keys "
                  f"(merged={n_merged}, copied={n_copied})", flush=True)

    save_merged(merged, save_dir, aux_src_dir=base_dir)
    print(f"  [{method}] saved merged checkpoint -> {save_dir}", flush=True)
    return {"n_merged": float(n_merged), "n_copied": float(n_copied),
            "n_experts": float(n)}
