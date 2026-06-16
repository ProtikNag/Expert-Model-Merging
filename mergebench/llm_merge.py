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


def _pscale_factor(pscale: str, *, u: torch.Tensor, tvs: List[torch.Tensor],
                   sum_tv: torch.Tensor, sum_abs: torch.Tensor,
                   alpha_max: float, beta: float) -> torch.Tensor:
    """Per-parameter update scale a_p for a ``whc_diag`` update ``u`` given the
    expert task vectors ``tvs`` (and their precomputed sum / abs-sum). See the
    ``pscale`` docstring in :func:`merge_checkpoints` for the definitions."""
    if pscale == "coherence":
        coh = sum_tv.abs() / (sum_abs + _EPS)
        return 1.0 + (alpha_max - 1.0) * coh.pow(beta)
    if pscale == "consensus":
        s = torch.sign(u)
        cons = torch.zeros_like(u)
        for tv in tvs:
            cons += torch.where(torch.sign(tv) == s, tv, torch.zeros_like(tv))
        return (cons.abs() / (u.abs() + _EPS)).clamp(min=1.0, max=alpha_max)
    raise ValueError(f"Unknown pscale mode: {pscale}")


def merge_whc_diag_pscale_multi(base_dir: str,
                                expert_dirs: List[str],
                                variants: List[dict],
                                save_dirs: List[str],
                                *,
                                lam: float = 1e-3,
                                log_every: int = 50) -> Dict[str, float]:
    """Single-pass ``whc_diag`` merge that emits MANY per-parameter-scale
    variants at once (dataless ``curvature="taskvec"`` only).

    All variants share the entire expensive computation -- reading the six
    models off disk and forming the curvature-weighted-mean update
    ``u = w_M^HTCL - w_pre`` -- and differ only in the per-parameter scale
    ``a_p`` applied to ``u``. Sweeping them with N separate
    :func:`merge_checkpoints` calls re-reads all experts N times (the NFS I/O
    that dominates wall time at 8B); this reads each key's tensors ONCE and
    writes all N variants, cutting merge I/O ~Nx. Numerically identical to
    calling :func:`merge_checkpoints` with ``method="whc_diag",
    curvature="taskvec", pscale=...`` per variant.

    Parameters
    ----------
    variants:
        List of dicts, each ``{"tag": str, "pscale": "consensus"|"coherence",
        "alpha_max": float, "beta": float}`` (``beta`` used by coherence only).
    save_dirs:
        Parallel list of output directories, one per variant.
    """
    if len(variants) != len(save_dirs):
        raise ValueError("variants and save_dirs must be the same length.")
    base = ShardedStateReader(base_dir)
    experts = [ShardedStateReader(d) for d in expert_dirs]
    n = len(experts)
    merged: List[Dict[str, torch.Tensor]] = [{} for _ in variants]

    keys = base.keys()
    n_merged, n_copied = 0, 0
    for idx, key in enumerate(keys):
        base_t = base.get(key)
        if not _is_float(base_t) or not _shapes_agree(key, base, experts):
            for m in merged:
                m[key] = base_t.clone()
            n_copied += 1
        else:
            out_dtype = base_t.dtype
            w_pre = base_t.float()
            w_experts = [e.get(key).float() for e in experts]
            # whc_diag closed form (taskvec curvature) + ensemble-mean anchor.
            w_bar = torch.zeros_like(w_pre)
            for w in w_experts:
                w_bar += w
            w_bar /= n
            num = lam * w_bar
            den = torch.full_like(w_pre, lam)
            for w_i in w_experts:
                f_i = (w_i - w_pre) ** 2
                num += f_i * w_i
                den += f_i
            out0 = num / (den + _EPS)             # alpha=1 closed form
            u = out0 - w_pre
            tvs = [w - w_pre for w in w_experts]
            sum_tv = torch.zeros_like(w_pre)
            sum_abs = torch.zeros_like(w_pre)
            for tv in tvs:
                sum_tv += tv
                sum_abs += tv.abs()
            for m, v in zip(merged, variants):
                a_p = _pscale_factor(v["pscale"], u=u, tvs=tvs, sum_tv=sum_tv,
                                     sum_abs=sum_abs,
                                     alpha_max=float(v["alpha_max"]),
                                     beta=float(v.get("beta", 1.0)))
                m[key] = (w_pre + a_p * u).to(out_dtype)
            n_merged += 1

        if (idx + 1) % log_every == 0 or (idx + 1) == len(keys):
            print(f"  [whc_pscale_multi] {idx + 1}/{len(keys)} keys "
                  f"(merged={n_merged}, copied={n_copied}, "
                  f"variants={len(variants)})", flush=True)

    for m, sd in zip(merged, save_dirs):
        save_merged(m, sd, aux_src_dir=base_dir)
        print(f"  [whc_pscale_multi] saved -> {sd}", flush=True)
    return {"n_merged": float(n_merged), "n_copied": float(n_copied),
            "n_experts": float(n), "n_variants": float(len(variants))}


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
                      pscale: str = "global",
                      alpha_max: float = 1.0,
                      beta: float = 1.0,
                      gamma: float = 0.0,
                      curvature: str = "taskvec",
                      gram_fallback: str = "mean",
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
    gram_fallback:
        For ``whc_gram`` only, how to merge the non-Gram keys (norms,
        embeddings, lm_head, and any Linear lacking a Gram such as the excluded
        ``down_proj``). ``"mean"`` (default) is the ensemble mean; ``"task_arith"``
        is the scaled task-vector sum ``w_pre + scale * sum_i (w_i - w_pre)``,
        which undoes the mean's ~1/N dilution on those keys.
    alpha:
        Task-vector scale for ``whc_diag`` and ``whc_gram``. The closed form
        returns a
        curvature-weighted *mean* of the experts, which dilutes each expert's
        update by ~1/N relative to a *sum* of task vectors (task arithmetic).
        ``alpha`` rescales the net deviation from base, ``w_M = w_pre + alpha *
        (w_M^HTCL - w_pre)``, so ``alpha=1`` is the plain closed form and
        ``alpha>1`` compensates the averaging dilution (try ``alpha ~ N``). This
        is the analogue of task arithmetic's scaling coefficient. Used only when
        ``pscale="global"`` (the default).
    pscale:
        Per-parameter update-scale mode for ``whc_diag`` only. A single global
        ``alpha`` cannot serve all domains at once (instruction wants a large
        scale, coding a small one), which caps the merge at the dataless tie.
        Instead of one scalar, scale each parameter's update ``u_p = w_M^HTCL_p
        - w_pre_p`` by a per-parameter factor ``a_p`` derived from the experts'
        task vectors ``tau_i = w_i - w_pre``:

        - ``"global"`` (default): ``a_p = alpha`` everywhere (backward compatible).
        - ``"coherence"``: ``a_p = 1 + (alpha_max - 1) * coherence_p ** beta``,
          where ``coherence_p = |sum_i tau_i,p| / (sum_i |tau_i,p| + eps)`` in
          [0, 1]. Params where the experts agree are boosted toward ``alpha_max``;
          conflicted params stay near 1. (The AAAI-plan mechanism.)
        - ``"consensus"``: ``a_p = clip(|c_p| / (|u_p| + eps), 1, alpha_max)``,
          where ``c_p = sum_{i: sign(tau_i,p)==sign(u_p)} tau_i,p`` is the summed
          task vector of the experts that agree with the merged update's
          direction. This rescales the curvature-weighted *mean* up to the
          additive *sum* of the agreeing experts, directly undoing the ~1/N
          dilution where it occurs while leaving single-expert params at a_p~=1
          (no over-extrapolation) and bounding conflicted params (tiny u_p, cap).
    alpha_max:
        Upper bound on the per-parameter scale when ``pscale != "global"``.
    beta:
        Coherence exponent when ``pscale="coherence"`` (sharpens/softens the
        boost as a function of inter-expert agreement).
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
                # Rescale the net update away from base. A single global alpha
                # (pscale="global") cannot serve all domains; the per-parameter
                # modes derive the scale from inter-expert agreement so coherent
                # updates recover the additive sum while single-expert and
                # conflicted params are left ~unscaled. See the docstring.
                if pscale == "global":
                    if alpha != 1.0:
                        out = w_pre + alpha * (out - w_pre)
                else:
                    u = out - w_pre
                    tvs = [w - w_pre for w in w_experts]
                    if pscale == "coherence":
                        sum_tv = torch.zeros_like(w_pre)
                        sum_abs = torch.zeros_like(w_pre)
                        for tv in tvs:
                            sum_tv += tv
                            sum_abs += tv.abs()
                        coh = sum_tv.abs() / (sum_abs + _EPS)
                        a_p = 1.0 + (alpha_max - 1.0) * coh.pow(beta)
                    elif pscale == "consensus":
                        s = torch.sign(u)
                        cons = torch.zeros_like(w_pre)
                        for tv in tvs:
                            cons += torch.where(torch.sign(tv) == s, tv,
                                                torch.zeros_like(tv))
                        ratio = cons.abs() / (u.abs() + _EPS)
                        a_p = ratio.clamp(min=1.0, max=alpha_max)
                    else:
                        raise ValueError(f"Unknown pscale mode: {pscale}")
                    out = w_pre + a_p * u

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
                # without a Gram) take the ``gram_fallback`` rule.
                #
                # Both the Gram solve AND the mean fallback are weighted AVERAGES
                # of the experts, so whc_gram inherits the same ~1/N update
                # dilution analysed for whc_diag (NOTES Sec 11). Two knobs undo
                # it: ``gram_fallback="task_arith"`` makes the non-Gram keys a
                # scaled task-vector SUM instead of a mean, and ``alpha`` rescales
                # the final deviation from base uniformly (alpha~N), exactly the
                # whc_diag fix lifted to the data tier.
                have_gram = (w_pre.dim() == 2
                             and all(g.has(key) for g in grams))
                if not have_gram:
                    if gram_fallback == "task_arith":
                        tv_sum = torch.zeros_like(w_pre)
                        for w in w_experts:
                            tv_sum += (w - w_pre)
                        out = w_pre + scale * tv_sum
                    else:  # "mean"
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
                # Global update-scale (applies to solved AND fallback keys).
                if alpha != 1.0:
                    out = w_pre + alpha * (out - w_pre)

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
