"""Academic-style figures for the pilot.

Generates, in the user's academic palette:

1. ``fig_methods_bar.{png,svg}``       - mean accuracy per method with error bars.
2. ``fig_per_rotation.{png,svg}``      - per-rotation accuracy, methods overlaid.
3. ``fig_htcl_lambda.{png,svg}``       - HTCL-A sensitivity to lambda.
4. ``fig_task_arith_scale.{png,svg}``  - Task Arithmetic scale sensitivity (ref).
5. ``fig_expert_matrix.{png,svg}``     - expert-vs-rotation accuracy heatmap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, load_config  # noqa: E402

AC_BLUE, AC_AMBER, AC_GREEN, AC_RED = "#2563EB", "#D97706", "#059669", "#DC2626"
AC_VIOLET, AC_TEAL, AC_ROSE, AC_SIENNA = "#7C3AED", "#0891B2", "#BE185D", "#92400E"
AC_SERIES = [AC_BLUE, AC_AMBER, AC_GREEN, AC_RED,
             AC_VIOLET, AC_TEAL, AC_ROSE, AC_SIENNA]

# ---- Matplotlib academic defaults ------------------------------------------
mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#E9ECEF",
    "grid.linewidth":    0.6,
    "axes.edgecolor":    "#495057",
    "axes.labelcolor":   "#212529",
    "xtick.color":       "#6C757D",
    "ytick.color":       "#6C757D",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.titleweight":  "semibold",
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "legend.frameon":    False,
})


METHOD_ORDER = [
    "simple", "task_arithmetic", "ties", "regmean", "fisher", "htcl_a", "htcl_b",
]
METHOD_LABELS = {
    "simple":          "Simple Avg.",
    "task_arithmetic": "Task Arith.",
    "ties":            "TIES",
    "regmean":         "RegMean",
    "fisher":          "Fisher",
    "htcl_a":          "HTCL-A (ours)",
    "htcl_b":          "HTCL-B (ours)",
}
METHOD_COLORS = {
    "simple":          "#ADB5BD",
    "task_arithmetic": AC_SIENNA,
    "ties":            AC_VIOLET,
    "regmean":         AC_AMBER,
    "fisher":          AC_GREEN,
    "htcl_a":          AC_BLUE,
    "htcl_b":          AC_RED,
}


def _best(results: Dict, key: str) -> Dict:
    """Return the `best` entry for swept methods, or the entry itself."""
    node = results[key]
    return node.get("best", node)


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.svg")
    plt.close(fig)


# ---- Figure 1: methods bar chart -------------------------------------------
def fig_methods_bar(results: Dict, out_dir: Path) -> None:
    names, avgs, stds, colors = [], [], [], []
    for m in METHOD_ORDER:
        e = _best(results, m)
        names.append(METHOD_LABELS[m])
        avgs.append(100 * e["summary"]["avg"])
        stds.append(100 * e["summary"]["std"])
        colors.append(METHOD_COLORS[m])

    exp_topline = 100 * results["experts"]["diagonal_mean"]
    pre_avg = 100 * results["pretrained"]["summary"]["avg"]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(names))
    bars = ax.bar(x, avgs, yerr=stds, capsize=3, color=colors,
                  edgecolor="#212529", linewidth=0.6, width=0.72,
                  error_kw={"ecolor": "#495057", "elinewidth": 0.8})
    for rect, v in zip(bars, avgs):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.9,
                f"{v:.1f}", ha="center", va="bottom",
                fontsize=9, color="#212529")
    # Topline and baseline annotations.
    ax.axhline(exp_topline, color="#495057", linestyle="--",
               linewidth=0.9, zorder=0)
    ax.text(-0.4, exp_topline + 0.8,
            f"expert own-rotation  {exp_topline:.1f}",
            ha="left", va="bottom", fontsize=9, color="#495057")
    ax.axhline(pre_avg, color=METHOD_COLORS["simple"], linestyle=":",
               linewidth=0.9, zorder=0)
    ax.text(-0.4, pre_avg + 0.8,
            f"shared init  {pre_avg:.1f}",
            ha="left", va="bottom", fontsize=9, color="#6C757D")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Avg. accuracy across 8 rotations (%)")
    ax.set_title("RotatedMNIST merging: avg accuracy (std across rotations as error bars)")
    ax.set_ylim(0, max(exp_topline + 8, max(avgs) + 12))
    _save(fig, out_dir, "fig_methods_bar")


# ---- Figure 2: per-rotation accuracy ---------------------------------------
def fig_per_rotation(results: Dict, out_dir: Path, rotations: List[int]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    angles = sorted(rotations)
    for m in METHOD_ORDER:
        e = _best(results, m)
        ys = [100 * e["per_rotation"][str(a)] for a in angles]
        ax.plot(angles, ys, marker="o", markersize=4,
                color=METHOD_COLORS[m], linewidth=1.8,
                label=METHOD_LABELS[m])
    # Expert topline (diagonal of matrix).
    mat = results["experts"]["matrix"]
    ys = [100 * mat[str(a)][str(a)] for a in angles]
    ax.plot(angles, ys, color="#495057", linestyle="--", linewidth=1.2,
            label="own-rotation expert")

    ax.set_xlabel("Rotation angle (deg)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Per-rotation accuracy of each merged model")
    ax.set_xticks(angles)
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.32))
    _save(fig, out_dir, "fig_per_rotation")


# ---- Figure 3: HTCL-A lambda sensitivity -----------------------------------
def fig_htcl_lambda(results: Dict, out_dir: Path) -> None:
    sweep = results["htcl_a"]["sweep"]
    lams = np.array([s["param"] for s in sweep])
    avgs = np.array([100 * s["summary"]["avg"] for s in sweep])
    stds = np.array([100 * s["summary"]["std"] for s in sweep])

    # Replace lam=0 for log axis with a sentinel on the left.
    nonzero = lams > 0
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.fill_between(lams[nonzero], avgs[nonzero] - stds[nonzero],
                    avgs[nonzero] + stds[nonzero],
                    color=AC_BLUE, alpha=0.18, linewidth=0)
    ax.plot(lams[nonzero], avgs[nonzero], "-o", color=AC_BLUE,
            markersize=5, linewidth=1.8, label="HTCL-A")

    # Reference lines: Fisher merging and simple averaging.
    fisher_avg = 100 * results["fisher"]["summary"]["avg"]
    simple_avg = 100 * results["simple"]["summary"]["avg"]
    ax.axhline(fisher_avg, color=AC_GREEN, linewidth=1.2, linestyle="--",
               label=f"Fisher ({fisher_avg:.1f})")
    ax.axhline(simple_avg, color="#ADB5BD", linewidth=1.2, linestyle=":",
               label=f"Simple Avg ({simple_avg:.1f})")

    ax.set_xscale("log")
    # Lam=0 collapses to ~11% (rank-deficient Fisher inverse); callout in box.
    if (lams == 0).any():
        lam0_avg = float(avgs[lams == 0][0])
        ax.text(0.98, 0.96,
                f"λ=0: unregularized Eq. 7\ncollapses to {lam0_avg:.1f}%",
                transform=ax.transAxes, fontsize=9, color=AC_RED,
                ha="right", va="top",
                bbox=dict(facecolor="#FEF3C7", edgecolor=AC_RED,
                          linewidth=0.6, boxstyle="round,pad=0.3"))
    ax.set_xlabel(r"Tikhonov coefficient  $\lambda$  (log scale)")
    ax.set_ylabel("Avg. accuracy (%)")
    ax.set_title(r"HTCL-A sensitivity to the ensemble-mean anchor strength $\lambda$")
    ax.legend(loc="lower left")
    _save(fig, out_dir, "fig_htcl_lambda")


# ---- Figure 4: task arithmetic scale sweep ---------------------------------
def fig_task_arith_scale(results: Dict, out_dir: Path) -> None:
    sweep = results["task_arithmetic"]["sweep"]
    scales = np.array([s["param"] for s in sweep])
    avgs = np.array([100 * s["summary"]["avg"] for s in sweep])
    stds = np.array([100 * s["summary"]["std"] for s in sweep])

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.fill_between(scales, avgs - stds, avgs + stds,
                    color=AC_SIENNA, alpha=0.18, linewidth=0)
    ax.plot(scales, avgs, "-o", color=AC_SIENNA,
            markersize=5, linewidth=1.8)
    ax.set_xlabel("Task-arithmetic scale")
    ax.set_ylabel("Avg. accuracy (%)")
    ax.set_title("Task arithmetic: coefficient sweep")
    _save(fig, out_dir, "fig_task_arith_scale")


# ---- Figure 5: expert x rotation matrix heatmap ----------------------------
def fig_expert_matrix(results: Dict, out_dir: Path,
                      rotations: List[int]) -> None:
    mat = results["experts"]["matrix"]
    angles = sorted(rotations)
    M = np.zeros((len(angles), len(angles)))
    for i, ai in enumerate(angles):
        for j, aj in enumerate(angles):
            M[i, j] = 100 * mat[str(ai)][str(aj)]

    # Sequential colormap from ac-surface to ac-blue.
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ac_blues",
                                             ["#F8F9FA", AC_BLUE])

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="equal")
    for i in range(len(angles)):
        for j in range(len(angles)):
            color = "#FFFFFF" if M[i, j] > 60 else "#212529"
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_xticks(range(len(angles)))
    ax.set_yticks(range(len(angles)))
    ax.set_xticklabels([f"{a}°" for a in angles])
    ax.set_yticklabels([f"{a}°" for a in angles])
    ax.set_xlabel("Evaluated rotation")
    ax.set_ylabel("Expert trained on rotation")
    ax.set_title("Individual experts: acc. (%) by train/eval rotation")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9, color="#495057")
    cbar.outline.set_edgecolor("#DEE2E6")
    _save(fig, out_dir, "fig_expert_matrix")


def main() -> None:
    cfg = load_config(ROOT / "configs" / "pilot.yaml")
    results_path = Path(cfg["paths"]["results"]) / "pilot_results.json"
    with open(results_path) as f:
        results = json.load(f)

    out_dir = ensure_dir(cfg["paths"]["figures"])
    rotations = list(cfg["dataset"]["rotations"])

    fig_methods_bar(results, out_dir)
    fig_per_rotation(results, out_dir, rotations)
    fig_htcl_lambda(results, out_dir)
    fig_task_arith_scale(results, out_dir)
    fig_expert_matrix(results, out_dir, rotations)

    print(f"[done] figures -> {out_dir}")


if __name__ == "__main__":
    main()
