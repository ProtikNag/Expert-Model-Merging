"""Offline per-expert coefficient derivation from a cached derivative dump.

mb_fit_perexpert_surrogate.py --dump-derivatives writes, at the s0 linearisation
point, the per-example directional derivatives d_{n,i} = <grad l_n, tau_i>, the
domain index of each example, and the per-example loss. Given those, the entire
Gauss-Newton solve is closed-form numpy -- so we can explore the derivation rule
WITHOUT touching the GPU again (the expensive 8B forward/backward pass ran once).

The knob this exposes is the own-vs-cross interference weight mu in [0,1]:

    grad_i(mu) = mean_n [ d_{n,i} * (1 if domain(n)==i else mu) ]

  * mu = 0  -> own_domain: coordinate i only sees domain-i data. No interference
              brake, optimum at s_i ~ 1 (be your own expert) -> OVER-merges.
  * mu = 1  -> pooled: full interference. Optimum below 0.4 -> UNDER-merges.
  * 0<mu<1  -> the moderate valley. Cross-domain damage is counted, but down-
              weighted to reflect that argmax-thresholded benchmarks tolerate the
              distributional drift that a soft loss over-penalises.

The curvature metric is the pooled empirical-Fisher A = mean_n outer(d_n, d_n)
(full, off-diagonals = cross-talk geometry); the Newton step is

    delta = -(A + ridge I)^{-1} grad(mu),   s* = clip(s0 + delta, box).

Note the RELATIVE coefficients (which expert gets the most) come from the
curvature/derivatives; mu is a single scalar setting the overall aggressiveness.

    python scripts/mb_derive_s_from_dump.py --dump deriv.npz --mu 0.3
    python scripts/mb_derive_s_from_dump.py --dump deriv.npz --sweep 0,0.1,0.25,0.5,1.0
"""
from __future__ import annotations

import argparse

import numpy as np


def derive(d, dom, s0, mu, ridge, clip, s_min, s_max):
    n_ex, n = d.shape
    # own/cross-weighted gradient
    w = np.where(dom[:, None] == np.arange(n)[None, :], 1.0, mu)   # (n_ex, N)
    grad = (d * w).mean(axis=0)
    # pooled empirical-Fisher curvature (full, with off-diagonals)
    A = np.einsum("ni,nj->ij", d, d) / n_ex
    delta = -np.linalg.solve(A + ridge * np.eye(n), grad)
    delta = np.clip(delta, -clip, clip)
    return np.clip(s0 + delta, s_min, s_max), grad, A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--mu", type=float, default=0.3)
    ap.add_argument("--sweep", default=None,
                    help="comma list of mu values to print a table for")
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--clip", type=float, default=1.0,
                    help="max |delta_i| (single-shot solve, so allow a full step)")
    ap.add_argument("--s-min", type=float, default=0.0)
    ap.add_argument("--s-max", type=float, default=1.5)
    args = ap.parse_args()

    z = np.load(args.dump, allow_pickle=True)
    d, dom, s0 = z["d"], z["domain"], z["s0"]
    domains = [str(x) for x in z["domains"]]
    n = d.shape[1]
    print(f"[derive] {d.shape[0]} examples, N={n}, domains={domains}, s0={s0.tolist()}")
    print(f"[derive] grad sign check (mean d per coord, own only):")
    for i in range(n):
        own = d[dom == i, i]
        print(f"    {domains[i]:14} own-mean d={own.mean():+.4f}  "
              f"(neg => wants to grow)")

    mus = ([float(x) for x in args.sweep.split(",")] if args.sweep else [args.mu])
    print(f"\n{'mu':>6} | " + " ".join(f"{x[:5]:>6}" for x in domains) + " | total")
    print("-" * (10 + 7 * n + 8))
    for mu in mus:
        s, grad, A = derive(d, dom, s0, mu, args.ridge, args.clip,
                            args.s_min, args.s_max)
        print(f"{mu:>6.2f} | " + " ".join(f"{v:>6.3f}" for v in s) +
              f" | {s.sum():.2f}")


if __name__ == "__main__":
    main()
