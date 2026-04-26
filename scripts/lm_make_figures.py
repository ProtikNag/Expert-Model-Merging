"""Academic figures for the LM merging experiment.

Figures produced (each as PNG under ``figures/png/`` and SVG under
``figures/svg/``):

1. ``fig_methods_bar``              : avg test primary per method, grouped
   by tier (dataless vs. statistics-using).
2. ``fig_methods_per_task_heatmap`` : per-task primary for every method.
3. ``fig_whc_lambda_sensitivity``   : WHC (tree) sweep vs. RegMean, Fisher,
   Simple Avg, and TIES.
4. ``fig_param_space_positions``    : L2 distance to pretrained, ensemble
   mean, and experts-mean for each method.
5. ``fig_sweep_curves``             : one subplot per method showing its
   sweep curve.
"""
from __future__ import annotations

import argparse
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


TIER_OF = {
    "simple":         "dataless",
    "task_arith":     "dataless",
    "ties":           "dataless",
    "fisher_merge":   "statistics",
    "regmean":        "statistics",
    "regmean_plus":   "statistics",
    "whc_tree":       "statistics",
}
LABEL_OF = {
    "simple":         "Simple Avg.",
    "task_arith":     "Task Arith.",
    "ties":           "TIES",
    "fisher_merge":   "Fisher",
    "regmean":        "RegMean",
    "regmean_plus":   "RegMean++",
    "whc_tree":       "WHC (tree)",
}
COLOR_OF = {
    "simple":         "#ADB5BD",
    "task_arith":     AC_SIENNA,
    "ties":           AC_VIOLET,
    "fisher_merge":   AC_GREEN,
    "regmean":        AC_AMBER,
    "regmean_plus":   "#92400E",
    "whc_tree":       AC_BLUE,
}


def _save(fig: plt.Figure, png_dir: Path, svg_dir: Path, stem: str) -> None:
    fig.savefig(png_dir / f"{stem}.png")
    fig.savefig(svg_dir / f"{stem}.svg")
    plt.close(fig)


def fig_methods_bar(results: Dict, png: Path, svg: Path) -> None:
    methods = [m for m in LABEL_OF if m in results["methods"]]
    dataless = [m for m in methods if TIER_OF[m] == "dataless"]
    stats = [m for m in methods if TIER_OF[m] == "statistics"]
    ordered = dataless + stats
    vals = [100 * results["methods"][m]["test_avg"] for m in ordered]
    colors = [COLOR_OF[m] for m in ordered]
    labels = [LABEL_OF[m] for m in ordered]

    fig, ax = plt.subplots(figsize=(max(8, 0.95 * len(ordered)), 4.5))
    x = np.arange(len(ordered))
    bars = ax.bar(x, vals, color=colors, edgecolor="#212529",
                  linewidth=0.6, width=0.72)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.4,
                f"{v:.1f}", ha="center", va="bottom",
                fontsize=9, color="#212529")

    tl = 100 * results["topline"]["avg"]
    si = 100 * results["shared_init"]["avg"]
    ax.axhline(tl, color="#495057", linestyle="--", linewidth=0.9, zorder=0)
    ax.text(-0.4, tl + 0.6, f"per-task topline  {tl:.1f}",
            fontsize=9, color="#495057")
    ax.axhline(si, color="#ADB5BD", linestyle=":", linewidth=0.9, zorder=0)
    ax.text(-0.4, si + 0.6, f"shared init  {si:.1f}",
            fontsize=9, color="#6C757D")

    # Tier separator.
    if dataless and stats:
        sep = len(dataless) - 0.5
        ax.axvline(sep, color="#DEE2E6", linewidth=1.0, zorder=0)
        ax.text(sep - 0.1, ax.get_ylim()[1] * 0.98, "dataless",
                ha="right", va="top", fontsize=9, color="#495057")
        ax.text(sep + 0.1, ax.get_ylim()[1] * 0.98, "+ statistics",
                ha="left", va="top", fontsize=9, color="#495057")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Avg. test primary metric across tasks (%)")
    ax.set_title("GLUE merging: test-set primary metric by method")
    ax.set_ylim(0, max(tl + 6, max(vals) + 8))
    _save(fig, png, svg, "fig_methods_bar")


def fig_methods_per_task_heatmap(results: Dict, tasks: List[str],
                                 png: Path, svg: Path) -> None:
    methods = [m for m in LABEL_OF if m in results["methods"]]
    M = np.zeros((len(methods), len(tasks)))
    for i, m in enumerate(methods):
        per = results["methods"][m]["test_per_task"]
        for j, t in enumerate(tasks):
            M[i, j] = 100 * per[t]["primary"]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ac_blues", ["#F8F9FA", AC_BLUE])
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(tasks)),
                                    max(4, 0.4 * len(methods))))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(len(methods)):
        for j in range(len(tasks)):
            color = "#FFFFFF" if M[i, j] > 60 else "#212529"
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_xticks(range(len(tasks)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels([t.upper() for t in tasks])
    ax.set_yticklabels([LABEL_OF[m] for m in methods])
    ax.set_title("Per-task test primary metric (%) per method")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_edgecolor("#DEE2E6")
    _save(fig, png, svg, "fig_methods_per_task_heatmap")


def fig_whc_lambda_sensitivity(results: Dict, png: Path, svg: Path) -> None:
    if "whc_tree" not in results["methods"]:
        return
    sweep = results["methods"]["whc_tree"]["sweep"]
    lams = np.array([s["hp"]["lam"] for s in sweep])
    vals = np.array([100 * s["val_avg"] for s in sweep])

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    nz = lams > 0
    ax.plot(lams[nz], vals[nz], "-o", color=AC_BLUE, linewidth=1.8,
            markersize=5, label="WHC (tree)")
    for ref, col, name in [
        ("regmean",      AC_AMBER, "RegMean"),
        ("fisher_merge", AC_GREEN, "Fisher"),
        ("simple",       "#ADB5BD", "Simple Avg"),
        ("ties",         AC_VIOLET, "TIES"),
    ]:
        if ref in results["methods"]:
            v = 100 * results["methods"][ref]["val_avg"]
            ax.axhline(v, color=col, linestyle="--", linewidth=1.0,
                       label=f"{name} ({v:.1f})")
    ax.set_xscale("log")
    ax.set_xlabel(r"Tikhonov coefficient  $\lambda$  (log)")
    ax.set_ylabel("Avg. val primary (%)")
    ax.set_title(r"WHC (tree) sensitivity to $\lambda$")
    ax.legend(loc="best")
    _save(fig, png, svg, "fig_whc_lambda_sensitivity")


def fig_param_space_positions(results: Dict, png: Path, svg: Path) -> None:
    methods = [m for m in LABEL_OF if m in results["methods"]]
    d_pre = [results["methods"][m]["param_stats"]["l2_to_pretrained"]
             for m in methods]
    d_mean = [results["methods"][m]["param_stats"]["l2_to_ensemble_mean"]
              for m in methods]
    d_exp = [results["methods"][m]["param_stats"]["l2_to_experts_mean"]
             for m in methods]
    x = np.arange(len(methods))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 0.95 * len(methods)), 4.0))
    ax.bar(x - w, d_pre, width=w, color=AC_AMBER, label="to pretrained")
    ax.bar(x,     d_mean, width=w, color=AC_BLUE, label="to ensemble mean")
    ax.bar(x + w, d_exp,  width=w, color=AC_GREEN, label="to experts (mean)")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_OF[m] for m in methods], rotation=30,
                       ha="right")
    ax.set_ylabel(r"$\ell_2$ distance")
    ax.set_title("Where each merged model lives in parameter space")
    ax.legend(loc="upper right")
    _save(fig, png, svg, "fig_param_space_positions")


def fig_sweep_curves(results: Dict, png: Path, svg: Path) -> None:
    methods = [m for m in LABEL_OF
               if m in results["methods"]
               and len(results["methods"][m]["sweep"]) > 1]
    if not methods:
        return
    cols = 3
    rows = (len(methods) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.8 * rows),
                             squeeze=False)
    for ax, m in zip(axes.flat, methods):
        sweep = results["methods"][m]["sweep"]
        xs = list(range(len(sweep)))
        ys = [100 * s["val_avg"] for s in sweep]
        ax.plot(xs, ys, "-o", color=COLOR_OF[m], linewidth=1.6, markersize=4)
        best_idx = int(np.argmax(ys))
        ax.axvline(best_idx, color="#DEE2E6", linewidth=0.8, zorder=0)
        ax.set_title(LABEL_OF[m])
        ax.set_xlabel("config idx")
        ax.set_ylabel("val primary (%)")
    for ax in axes.flat[len(methods):]:
        ax.axis("off")
    fig.suptitle("Per-method hparam sweep (val primary)", y=1.02)
    fig.tight_layout()
    _save(fig, png, svg, "fig_sweep_curves")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(cfg["paths"]["results"])
    with open(results_dir / "merge_results.json") as f:
        results = json.load(f)

    png = ensure_dir(Path(cfg["paths"]["figures"]) / "png")
    svg = ensure_dir(Path(cfg["paths"]["figures"]) / "svg")
    tasks = cfg["data"]["tasks"]

    fig_methods_bar(results, png, svg)
    fig_methods_per_task_heatmap(results, tasks, png, svg)
    fig_whc_lambda_sensitivity(results, png, svg)
    fig_param_space_positions(results, png, svg)
    fig_sweep_curves(results, png, svg)
    print(f"[done] figures -> {png} and {svg}")


if __name__ == "__main__":
    main()
