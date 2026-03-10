"""
Visualization functions for Neura PLDI 2026 artifact evaluation.

Generates: speedup (fig13), IPC (fig14), PPA (fig15), energy (fig16),
           optimization (fig17), scalability (fig18).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional


# ── Shared grouped-bar chart (speedup / PPA / IPC) ──────────────────────

def _grouped_bar_chart(
    bar_data: dict[str, list],
    geomeans: dict[str, Optional[float]],
    benchmarks: list[str],
    archs: list[str],
    save_path: Path,
    *,
    ylabel: str,
    figsize: tuple = (13, 2.4),
    geomean_fmt: str = "{:.2f}x",
    annotation_offset: float = 0.05,
    y_grid_step: Optional[float] = 0.5,
    y_step_int: Optional[int] = None,
    x_label_offset: float = -0.2,
) -> None:
    """
    Common routine for speedup, PPA, and IPC grouped bar charts.

    y_grid_step : fixed grid step for speedup-style (0.5 spacing).
    y_step_int  : if set, use integer tick step and auto-ceil y-axis (PPA/IPC).
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not available; skipping figure generation.")
        return

    plt.rcParams.update({"font.size": 14})

    bench_labels = benchmarks + ["Geomean"]
    data: dict[str, list] = {}
    for arch in archs:
        vals = list(bar_data[arch])
        gm = geomeans[arch]
        vals.append(gm if gm is not None else 0.0)
        data[arch] = [v if v is not None else 0.0 for v in vals]

    avg_labels = {
        arch: geomean_fmt.format(geomeans[arch]) if geomeans[arch] else "N/A"
        for arch in archs
    }

    fig, ax = plt.subplots(figsize=figsize)
    num_archs = len(archs)
    bar_width = (1 - 0.3) / num_archs
    index = np.arange(len(bench_labels))
    colors = ["#A9A9A9", "#72bcd5", "#386795", "#fee6b4", "#e86252"]

    for i, arch in enumerate(archs):
        pos = index + (i - num_archs / 2 + 0.5) * bar_width
        ax.bar(pos, data[arch], bar_width,
               color=colors[i % len(colors)], label=arch, edgecolor="black")

    # Annotate geomean bars
    last_idx = len(bench_labels) - 1
    for i, arch in enumerate(archs):
        val = data[arch][last_idx]
        pos = last_idx + (i - num_archs / 2 + 0.5) * bar_width
        ax.text(pos, val + annotation_offset, avg_labels[arch],
                rotation=90, ha="center", va="bottom",
                fontsize=11 if y_step_int is None else 12, fontweight="bold")

    ax.set_ylabel(ylabel, fontsize=14)
    all_vals = [v for vs in data.values() for v in vs]

    if y_step_int is not None:
        # PPA / IPC style: integer step, auto-ceil
        y_max = max(all_vals) * 1.3 if any(v > 0 for v in all_vals) else 15
        y_ceil = int(y_max / y_step_int + 1) * y_step_int
        ax.set_ylim(0, y_ceil)
        y_ticks = list(range(0, y_ceil + 1, y_step_int))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(y) for y in y_ticks], fontsize=14)
        ax.tick_params(axis="y", direction="inout", length=6)
        grid_ys = y_ticks[1:]
    else:
        # Speedup style: fixed grid lines
        y_max = max(all_vals) * 1.3 if any(v > 0 for v in all_vals) else 4
        ax.set_ylim(0, y_max)
        grid_ys = [y_grid_step * i for i in range(1, int(y_max / y_grid_step) + 1)]

    if ylabel == "IPC":
        ax.yaxis.set_label_coords(-0.025, 0.5)

    ax.set_xlim(-0.5, len(bench_labels) - 0.5)
    dividers = [i - 0.5 for i in range(len(bench_labels) + 1)]
    ax.set_xticks(dividers)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", direction="inout", length=6)

    for i, b in enumerate(bench_labels):
        w = "bold" if b == "Geomean" else "normal"
        ax.text(i, x_label_offset, b, ha="center", va="top",
                fontsize=13, fontweight=w)

    for y in grid_ys:
        ax.axhline(y, color="lightgray", linestyle=(0, (5, 5)),
                   linewidth=1, alpha=0.7, zorder=0)
    for x in dividers:
        ax.axvline(x, color="lightgray", linewidth=0.8, alpha=0.8, zorder=0)

    ax.spines["top"].set_linestyle("--")

    legend_y = 1.5 if y_step_int is not None else 1.35
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, legend_y),
              ncol=num_archs, handleheight=0.7, handlelength=0.7,
              columnspacing=4, handletextpad=0.3, frameon=False)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    plt.close(fig)


# ── Public figure functions ──────────────────────────────────────────────

def generate_speedup_figure(speedup_data, geomeans, benchmarks, archs, save_path):
    _grouped_bar_chart(
        speedup_data, geomeans, benchmarks, archs, save_path,
        ylabel="Normalised\nSpeedup",
        y_grid_step=0.5, x_label_offset=-0.2,
    )


def generate_ppa_figure(ppa_data, geomeans, benchmarks, archs, save_path):
    _grouped_bar_chart(
        ppa_data, geomeans, benchmarks, archs, save_path,
        ylabel="Normalized\nPerf/Area",
        y_step_int=5, annotation_offset=0.18, x_label_offset=-0.7,
    )


def generate_ipc_figure(ipc_data, geomeans, benchmarks, archs, save_path):
    _grouped_bar_chart(
        ipc_data, geomeans, benchmarks, archs, save_path,
        ylabel="IPC",
        y_step_int=5, geomean_fmt="{:.2f}",
        annotation_offset=0.15, x_label_offset=-0.3,
    )


def generate_energy_figure(
    energy_data: dict[str, list],
    geomeans: dict[str, Optional[float]],
    benchmarks: list[str],
    archs: list[str],
    save_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not available; skipping energy figure.")
        return

    plt.rcParams.update({"font.size": 14})

    bench_labels = benchmarks + ["Geomean"]
    data: dict[str, list] = {}
    for arch in archs:
        vals = list(energy_data[arch])
        gm = geomeans[arch]
        vals.append(gm if gm is not None else 0.0)
        data[arch] = [v if v is not None else 0.0 for v in vals]

    avg_labels = {
        arch: f"{geomeans[arch]:.2f}x" if geomeans[arch] else "N/A"
        for arch in archs
    }

    fig, ax = plt.subplots(figsize=(8, 2.5))
    num_archs = len(archs)
    bar_width = (1 - 0.5) / num_archs
    index = np.arange(len(bench_labels))
    colors = ["#386795", "#fee6b4"]

    for i, arch in enumerate(archs):
        pos = index + (i - num_archs / 2 + 0.5) * bar_width
        ax.bar(pos, data[arch], bar_width,
               color=colors[i % len(colors)], label=arch, edgecolor="black")

    last_idx = len(bench_labels) - 1
    for i, arch in enumerate(archs):
        val = data[arch][last_idx]
        pos = last_idx + (i - num_archs / 2 + 0.5) * bar_width
        ax.text(pos, val + 0.03, avg_labels[arch],
                rotation=90, ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_ylabel("Normalized\nTotal Energy", fontsize=14)
    all_vals = [v for vs in data.values() for v in vs if v > 0]
    y_max = max(all_vals) * 1.45 if all_vals else 2.0
    ax.set_ylim(0, y_max)
    ax.set_xlim(-0.5, len(bench_labels) - 0.5)

    dividers = [i - 0.5 for i in range(len(bench_labels) + 1)]
    ax.set_xticks(dividers)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", direction="inout", length=6)

    for i, b in enumerate(bench_labels):
        w = "bold" if b == "Geomean" else "normal"
        ax.text(i + 0.15, -0.02, b, ha="right", va="top",
                fontsize=14, rotation=20, fontweight=w)

    y_ticks = [v * 0.5 for v in range(int(y_max / 0.5) + 2) if v * 0.5 <= y_max]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(
        [str(int(y)) if y == int(y) else "" for y in y_ticks], fontsize=14
    )
    ax.tick_params(axis="y", direction="inout", length=6)

    for y in y_ticks[1:]:
        ax.axhline(y, color="lightgray", linestyle=(0, (5, 5)),
                   linewidth=1, alpha=0.7, zorder=0)
    for x in dividers:
        ax.axvline(x, color="lightgray", linewidth=0.8, alpha=0.8, zorder=0)

    ax.spines["top"].set_linestyle("--")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.35),
              ncol=num_archs, handleheight=0.7, handlelength=0.7,
              columnspacing=4, handletextpad=0.3, frameon=False)

    plt.tight_layout(pad=0.8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    plt.close(fig)


_OPT_VARIANT_NAMES = [
    "w/o Optimization",
    "Computational Pattern Fusion",
    "Data Type Alignment",
    "Data Type Alignment + Constant Folding",
    "HW-Agnostic + Loop Streaming",
]
_OPT_COLORS = ["#A9A9A9", "#a9d18e", "#409cd8", "#fed06e", "#e86252"]


def generate_optimization_figure(
    speedup_data: Dict[str, List[Optional[float]]],
    benchmarks: List[str],
    save_path: Path,
) -> None:
    """Stacked bar chart of cumulative optimization speedup (marginal gains)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not available; skipping optimisation figure.")
        return

    plt.rcParams.update({"font.size": 14})

    bench_labels = list(benchmarks) + ["Geomean"]
    n = len(bench_labels)

    total: Dict[str, List[float]] = {
        v: [x if x is not None else 0.0 for x in speedup_data[v]]
        for v in _OPT_VARIANT_NAMES
    }

    # Compute marginal contributions
    marginal: List[List[float]] = []
    for idx, vname in enumerate(_OPT_VARIANT_NAMES):
        if idx == 0:
            marginal.append(list(total[vname]))
        else:
            prev = total[_OPT_VARIANT_NAMES[idx - 1]]
            cur  = total[vname]
            marginal.append([max(0.0, c - p) for c, p in zip(cur, prev)])

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_width = 0.5
    index = np.arange(n)
    bottoms = np.zeros(n)

    for layer_data, color, vname in zip(marginal, _OPT_COLORS, _OPT_VARIANT_NAMES):
        ax.bar(index, layer_data, bar_width, bottom=bottoms,
               color=color, label=vname, edgecolor="black", linewidth=0.5)
        bottoms += np.array(layer_data)

    # Annotate total speedup
    top_vals = total["HW-Agnostic + Loop Streaming"]
    for i, val in enumerate(top_vals):
        if val > 0:
            ax.text(i, val + 0.05, f"{val:.2f}\u00d7",
                    ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="black")

    ax.set_ylabel("Normalized Speedup", fontsize=14)
    all_tops = [v for v in top_vals if v > 0]
    y_max = max(max(all_tops) * 1.2, 5.0) if all_tops else 5.0
    y_top = int(y_max) + 1
    ax.set_ylim(0, y_top)
    ax.set_xlim(-0.5, n - 0.5)

    y_ticks = list(range(0, y_top + 1))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(y) for y in y_ticks], fontsize=14)
    ax.tick_params(axis="y", direction="inout", length=6)

    dividers = [i - 0.5 for i in range(n + 1)]
    ax.set_xticks(dividers)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", direction="inout", length=6, width=0)

    for i, b in enumerate(bench_labels):
        w = "bold" if b == "Geomean" else "normal"
        ax.text(i + 0.2, -0.1, b, ha="right", va="top",
                fontsize=14, rotation=22, fontweight=w)

    for y in y_ticks[1:]:
        ax.axhline(y, color="lightgray", linestyle=(0, (5, 5)),
                   linewidth=1, alpha=0.7, zorder=0)
    for x in dividers:
        ax.axvline(x, color="lightgray", linewidth=0.8, alpha=0.8, zorder=0)

    ax.spines["top"].set_linestyle("--")

    # Two-row legend
    from matplotlib.patches import Patch
    colors_map = dict(zip(_OPT_VARIANT_NAMES, _OPT_COLORS))
    row1_entries = [
        ("w/o Optimization",  "w/o Optimization"),
        ("Data Type Alignment", "Data Type Alignment"),
        ("Constant Folding",  "Data Type Alignment + Constant Folding"),
    ]
    row2_entries = [
        ("Computational Pattern Fusion", "Computational Pattern Fusion"),
        ("Loop Streaming",               "HW-Agnostic + Loop Streaming"),
    ]

    h1 = [Patch(facecolor=colors_map[k], edgecolor="black", linewidth=0.5) for _, k in row1_entries]
    h2 = [Patch(facecolor=colors_map[k], edgecolor="black", linewidth=0.5) for _, k in row2_entries]

    leg2 = ax.legend(h2, [lbl for lbl, _ in row2_entries], loc="upper center",
                     bbox_to_anchor=(0.5, 1.17), ncol=2, fontsize=12,
                     handleheight=0.7, handlelength=0.7,
                     columnspacing=2.5, handletextpad=0.3, frameon=False)
    ax.legend(h1, [lbl for lbl, _ in row1_entries], loc="upper center",
              bbox_to_anchor=(0.5, 1.28), ncol=3, fontsize=12,
              handleheight=0.7, handlelength=0.7,
              columnspacing=2.5, handletextpad=0.3, frameon=False)
    ax.add_artist(leg2)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    plt.close(fig)


def generate_scalability_figure(
    speedup_data: Dict[str, List[float]],
    improvement_rate: List[float],
    benchmarks: List[str],
    save_path: Path,
) -> None:
    """Grouped bars (4x4 vs 6x6) + improvement-rate line on twin Y-axis."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        print("[WARN] matplotlib not available; skipping scalability figure.")
        return

    plt.rcParams.update({"font.size": 14})

    bench_labels = list(benchmarks) + ["Geomean"]
    config_names = list(speedup_data.keys())

    data: Dict[str, List[float]] = {
        cfg: [v if v is not None else 0.0 for v in speedup_data[cfg]]
        for cfg in config_names
    }

    last_idx = len(bench_labels) - 1
    avg_speedups = {cfg: f"{data[cfg][last_idx]:.2f}x" for cfg in config_names}

    fig, ax1 = plt.subplots(figsize=(7, 3))
    line_color = "#c50313"

    num_configs = len(config_names)
    group_spacing = 0.5
    bar_width = (1 - group_spacing) / num_configs
    index = np.arange(len(bench_labels))

    colors = ["#72bcd5", "#ffd06e"]
    bars = []
    for i, cfg in enumerate(config_names):
        position = index + (i - num_configs / 2 + 0.5) * bar_width
        bar = ax1.bar(position, data[cfg], bar_width,
                      color=colors[i % len(colors)], label=cfg, edgecolor="black")
        bars.append(bar)

    ax1.set_ylabel("Normalized Speedup", fontsize=14)

    divider_positions_x = [i - 0.5 for i in range(len(bench_labels) + 1)]
    ax1.set_xticks(divider_positions_x)
    ax1.set_xticklabels([])
    ax1.tick_params(axis="x", which="major", direction="inout", length=6, width=1)

    for i, benchmark in enumerate(bench_labels):
        weight = "bold" if benchmark == "Geomean" else "normal"
        ax1.text(i + 0.15, -0.02, benchmark, ha="right", va="top",
                 fontsize=14, rotation=20, fontweight=weight)

    ax1.set_yticks([0, 0.5, 1, 1.5, 2])
    ax1.set_yticklabels(["0", "", "1", "", "2"], fontsize=14)
    ax1.tick_params(axis="y", which="major", direction="inout", length=6, width=1)
    ax1.set_ylim(0, 2)
    ax1.set_xlim(-0.5, len(bench_labels) - 0.5)

    ax1.spines["top"].set_visible(True)
    ax1.spines["top"].set_linestyle((0, (5, 5)))
    ax1.spines["top"].set_linewidth(1)

    for y in [0.5, 1, 1.5]:
        ax1.axhline(y, color="lightgray", linestyle=(0, (5, 5)),
                    linewidth=1, alpha=0.7, zorder=0)
    for x in divider_positions_x:
        ax1.axvline(x, color="lightgray", linewidth=0.8, alpha=0.8, zorder=0)

    # Twin right Y-axis: improvement rate
    ax2 = ax1.twinx()
    ax2.set_ylabel("Improvement Rate (%)", fontsize=14, color=line_color)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(line_color)

    line = ax2.plot(index + 0.05, improvement_rate, marker="v",
                    color=line_color, linewidth=2,
                    markeredgecolor="black", markeredgewidth=0.75, markersize=8,
                    label="Improvement Rate")

    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_yticklabels(["0", "25", "50", "75", "100"], fontsize=14)
    ax2.tick_params(axis="y", which="major", direction="inout", length=6, width=1,
                    right=True, left=False, color=line_color, labelcolor=line_color)

    def _pct_fmt(x, pos):
        return f"{int(x)}" if x == int(x) else f"{x:.1f}"
    ax2.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))

    # Annotate geomean bar tops
    for i, cfg in enumerate(config_names):
        avg_val = data[cfg][last_idx]
        pos = last_idx + (i - num_configs / 2 + 0.5) * bar_width
        ax1.text(pos, avg_val + 0.03, avg_speedups[cfg],
                 rotation=90, ha="center", va="bottom",
                 fontsize=12, fontweight="bold", color="black")

    # Combined legend
    all_handles = bars + line
    all_labels = [h.get_label() for h in bars] + [line[0].get_label()]
    legend = ax1.legend(all_handles, all_labels,
                        loc="upper center", bbox_to_anchor=(0.5, 1.3),
                        ncol=3, handleheight=0.7, handlelength=0.7,
                        columnspacing=1.0, handletextpad=0.3, frameon=False)
    legend.get_texts()[-1].set_color(line_color)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    plt.close(fig)
