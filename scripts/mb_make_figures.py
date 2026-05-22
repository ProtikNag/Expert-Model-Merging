"""Figures for the Tier 0 divergence diagnostic.

Reads ``results/mergebench/divergence_<base>.json`` and produces, in the
academic palette, both PNG (300 dpi) and SVG:

  - fig_divergence_heatmap : pairwise task-vector cosine matrix.
  - fig_divergence_components : per-component mean cosine and relative drift.

    python scripts/mb_make_figures.py --config configs/mergebench.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, load_config  # noqa: E402

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica", "Arial"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E9ECEF",
    "grid.linewidth": 0.6,
    "axes.edgecolor": "#495057",
    "axes.labelcolor": "#212529",
    "xtick.color": "#6C757D",
    "ytick.color": "#6C757D",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

AC_BLUE = "#2563EB"
AC_AMBER = "#D97706"
AC_TEXT = "#212529"
# Sequential white -> blue colormap (colorblind-safe, no rainbow).
AC_CMAP = LinearSegmentedColormap.from_list("ac_blue", ["#FFFFFF", AC_BLUE])


def _save(fig, fig_dir: Path, name: str) -> None:
    ensure_dir(fig_dir / "png")
    ensure_dir(fig_dir / "svg")
    fig.savefig(fig_dir / "png" / f"{name}.png")
    fig.savefig(fig_dir / "svg" / f"{name}.svg")
    plt.close(fig)
    print(f"  wrote {name}.png / .svg", flush=True)


def fig_heatmap(div: dict, fig_dir: Path) -> None:
    names = div["expert_names"]
    mat = np.array(div["cosine_matrix"])
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.1 * n + 1.5, 1.1 * n + 1))
    # Off-diagonal range is the interesting part; cap at a modest value so the
    # near-1.0 diagonal does not wash out the contrast.
    im = ax.imshow(mat, cmap=AC_CMAP, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="#FFFFFF" if val > 0.5 else AC_TEXT, fontsize=9)
    ax.set_title(r"Pairwise task-vector cosine  $\cos(\tau_i,\tau_j)$",
                 color=AC_TEXT, fontsize=13)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor("#DEE2E6")
    _save(fig, fig_dir, "fig_divergence_heatmap")


def fig_components(div: dict, fig_dir: Path) -> None:
    comps = list(div["per_component"].keys())
    names = div["expert_names"]
    mean_cos = [div["per_component"][c]["mean_offdiag_cosine"] for c in comps]
    # Mean relative drift across experts, per component.
    mean_drift = [float(np.mean([div["per_component"][c]["relative_drift"][nm]
                                 for nm in names])) for c in comps]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(comps))

    axes[0].bar(x, mean_cos, color=AC_BLUE, width=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comps)
    axes[0].set_ylabel("mean off-diag cosine")
    axes[0].set_title("Task-vector alignment by component", color=AC_TEXT)
    for xi, v in zip(x, mean_cos):
        axes[0].text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9,
                     color=AC_TEXT)
    axes[0].set_ylim(0, 1.08)

    axes[1].bar(x, mean_drift, color=AC_AMBER, width=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(comps)
    axes[1].set_ylabel(r"mean relative drift  $\|\tau\|/\|w_{pre}\|$")
    axes[1].set_title("Drift from init by component", color=AC_TEXT)
    for xi, v in zip(x, mean_drift):
        axes[1].text(xi, v + max(mean_drift) * 0.02, f"{v:.3f}",
                     ha="center", fontsize=9, color=AC_TEXT)

    fig.tight_layout()
    _save(fig, fig_dir, "fig_divergence_components")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    results_dir = Path(cfg["paths"]["results"])
    div_path = results_dir / f"divergence_{cfg['base_name']}.json"
    with open(div_path, "r") as f:
        div = json.load(f)

    fig_dir = ensure_dir(results_dir / "figures")
    print(f"[figures] from {div_path}", flush=True)
    fig_heatmap(div, fig_dir)
    if "per_component" in div:
        fig_components(div, fig_dir)
    print(f"[done] figures under {fig_dir}", flush=True)


if __name__ == "__main__":
    main()
