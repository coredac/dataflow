"""
Figure generation utilities for Neura PLDI 2026 artifact evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


def generate_speedup_figure(
    speedup_data: dict[str, list],
    geomeans: dict[str, Optional[float]],
    benchmarks: list[str],
    archs: list[str],
    save_path: Path,
) -> None:
    """
    Plot a grouped bar chart of normalised speedup (relative to Marionette).

    speedup_data : dict mapping arch name → list of per-benchmark speedup values
                   (None for missing/failed entries).
    geomeans     : dict mapping arch name → geometric mean speedup (or None).
    benchmarks   : ordered list of benchmark names.
    archs        : ordered list of architecture names (controls bar order / colours).
    save_path    : destination PDF/PNG path; parent directory must exist.
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
        vals = list(speedup_data[arch])
        gm = geomeans[arch]
        vals.append(gm if gm is not None else 0.0)
        data[arch] = [v if v is not None else 0.0 for v in vals]

    avg_labels = {
        arch: f"{geomeans[arch]:.2f}x" if geomeans[arch] else "N/A"
        for arch in archs
    }

    fig, ax = plt.subplots(figsize=(13, 2.4))

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
        ax.text(pos, val + 0.05, avg_labels[arch],
                rotation=90, ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_ylabel("Normalised\nSpeedup", fontsize=14)
    ax.set_ylim(0, max(max(v) for v in data.values()) * 1.3 or 4)
    ax.set_xlim(-0.5, len(bench_labels) - 0.5)

    dividers = [i - 0.5 for i in range(len(bench_labels) + 1)]
    ax.set_xticks(dividers)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", direction="inout", length=6)

    for i, b in enumerate(bench_labels):
        w = "bold" if b == "Geomean" else "normal"
        ax.text(i, -0.2, b, ha="center", va="top", fontsize=13, fontweight=w)

    for y in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        ax.axhline(y, color="lightgray", linestyle=(0, (5, 5)),
                   linewidth=1, alpha=0.7, zorder=0)
    for x in dividers:
        ax.axvline(x, color="lightgray", linewidth=0.8, alpha=0.8, zorder=0)

    ax.spines["top"].set_linestyle("--")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.35),
              ncol=num_archs, handleheight=0.7, handlelength=0.7,
              columnspacing=4, handletextpad=0.3, frameon=False)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    plt.close(fig)


def generate_ppa_figure(
    ppa_data: dict[str, list],
    geomeans: dict[str, Optional[float]],
    benchmarks: list[str],
    archs: list[str],
    save_path: Path,
) -> None:
    """
    Plot a grouped bar chart of normalised Performance-per-Area (relative to
    Marionette).  Styling mirrors the speedup figure.

    ppa_data : dict mapping arch name → list of per-benchmark PPA values
               (None for missing/failed entries).
    geomeans : dict mapping arch name → geometric mean PPA (or None).
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not available; skipping PPA figure generation.")
        return

    plt.rcParams.update({"font.size": 14})

    bench_labels = benchmarks + ["Geomean"]
    data: dict[str, list] = {}
    for arch in archs:
        vals = list(ppa_data[arch])
        gm = geomeans[arch]
        vals.append(gm if gm is not None else 0.0)
        data[arch] = [v if v is not None else 0.0 for v in vals]

    avg_labels = {
        arch: f"{geomeans[arch]:.2f}x" if geomeans[arch] else "N/A"
        for arch in archs
    }

    fig, ax = plt.subplots(figsize=(13, 2.4))

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
        ax.text(pos, val + 0.18, avg_labels[arch],
                rotation=90, ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_ylabel("Normalized\nPerf/Area", fontsize=14)

    all_vals = [v for vs in data.values() for v in vs]
    y_max = max(all_vals) * 1.3 if any(v > 0 for v in all_vals) else 15
    ax.set_ylim(0, y_max)
    ax.set_xlim(-0.5, len(bench_labels) - 0.5)

    # Y-axis ticks: choose round steps that cover the range
    step = 5
    y_ticks = list(range(0, int(y_max) + step, step))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(y) for y in y_ticks], fontsize=14)
    ax.tick_params(axis="y", direction="inout", length=6)

    dividers = [i - 0.5 for i in range(len(bench_labels) + 1)]
    ax.set_xticks(dividers)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", direction="inout", length=6)

    for i, b in enumerate(bench_labels):
        w = "bold" if b == "Geomean" else "normal"
        ax.text(i, -0.7, b, ha="center", va="top", fontsize=13, fontweight=w)

    for y in y_ticks[1:]:  # horizontal grid (skip 0)
        ax.axhline(y, color="lightgray", linestyle=(0, (5, 5)),
                   linewidth=1, alpha=0.7, zorder=0)
    for x in dividers:
        ax.axvline(x, color="lightgray", linewidth=0.8, alpha=0.8, zorder=0)

    ax.spines["top"].set_linestyle("--")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.5),
              ncol=num_archs, handleheight=0.7, handlelength=0.7,
              columnspacing=4, handletextpad=0.3, frameon=False)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    plt.close(fig)


def generate_ipc_figure(
    ipc_data: dict[str, list],
    geomeans: dict[str, Optional[float]],
    benchmarks: list[str],
    archs: list[str],
    save_path: Path,
) -> None:
    """
    Plot a grouped bar chart of IPC (instructions per cycle) per architecture.

    ipc_data : dict mapping arch name → list of per-benchmark IPC values
               (None for entries where instruction count is not yet set).
    geomeans : dict mapping arch name → geometric mean IPC (or None).
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not available; skipping IPC figure generation.")
        return

    plt.rcParams.update({"font.size": 14})

    bench_labels = benchmarks + ["Geomean"]
    data: dict[str, list] = {}
    for arch in archs:
        vals = list(ipc_data[arch])
        gm = geomeans[arch]
        vals.append(gm if gm is not None else 0.0)
        data[arch] = [v if v is not None else 0.0 for v in vals]

    avg_labels = {
        arch: f"{geomeans[arch]:.2f}" if geomeans[arch] else "N/A"
        for arch in archs
    }

    fig, ax = plt.subplots(figsize=(13, 2.4))

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
        ax.text(pos, val + 0.15, avg_labels[arch],
                rotation=90, ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_ylabel("IPC", fontsize=14)
    ax.yaxis.set_label_coords(-0.025, 0.5)

    all_vals = [v for vs in data.values() for v in vs]
    y_max = max(all_vals) * 1.3 if any(v > 0 for v in all_vals) else 10
    # Round up to a clean multiple of 5
    step = 5
    y_ceil = int(y_max / step + 1) * step
    ax.set_ylim(0, y_ceil)
    ax.set_xlim(-0.5, len(bench_labels) - 0.5)

    y_ticks = list(range(0, y_ceil + 1, step))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(y) for y in y_ticks], fontsize=14)
    ax.tick_params(axis="y", direction="inout", length=6)

    dividers = [i - 0.5 for i in range(len(bench_labels) + 1)]
    ax.set_xticks(dividers)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", direction="inout", length=6)

    for i, b in enumerate(bench_labels):
        w = "bold" if b == "Geomean" else "normal"
        ax.text(i, -0.3, b, ha="center", va="top", fontsize=13, fontweight=w)

    for y in y_ticks[1:]:
        ax.axhline(y, color="lightgray", linestyle=(0, (5, 5)),
                   linewidth=1, alpha=0.7, zorder=0)
    for x in dividers:
        ax.axvline(x, color="lightgray", linewidth=0.8, alpha=0.8, zorder=0)

    ax.spines["top"].set_linestyle("--")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.4),
              ncol=num_archs, handleheight=0.7, handlelength=0.7,
              columnspacing=4, handletextpad=0.3, frameon=False)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    plt.close(fig)


def generate_energy_figure(
    energy_data: dict[str, list],
    geomeans: dict[str, Optional[float]],
    benchmarks: list[str],
    archs: list[str],
    save_path: Path,
) -> None:
    """
    Plot a grouped bar chart of normalised total energy (relative to RipTide).

    Energy per arch per bench = ARCH_POWER_MW[arch] * BENCH_ARCH_INSTRUCTIONS[(bench, arch)].
    All values are normalised so RipTide = 1.0.

    energy_data : dict arch → list of per-benchmark normalised energy (None if unavailable).
    geomeans    : dict arch → geometric mean normalised energy (or None).
    archs       : expected ["RipTide", "NEURA-SO"].
    save_path   : destination PDF/PNG path.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not available; skipping energy figure generation.")
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
    # Match energy.py reference colours: RipTide→#386795, NEURA-SO→#fee6b4
    colors = ["#386795", "#fee6b4"]

    for i, arch in enumerate(archs):
        pos = index + (i - num_archs / 2 + 0.5) * bar_width
        ax.bar(pos, data[arch], bar_width,
               color=colors[i % len(colors)], label=arch, edgecolor="black")

    # Annotate geomean bars
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

    # Y-axis ticks at multiples of 0.5
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


# ─────────────────────────────────────────────────────────────────────────────
# Ordered variant names (must match keys in speedup_data passed by main.py)
_OPT_VARIANT_NAMES = [
    "w/o Optimization",
    "Computational Pattern Fusion",
    "Data Type Alignment",
    "Data Type Alignment + Constant Folding",
    "HW-Agnostic + Loop Streaming",
]

# Stack colours, one per variant (same order as above)
_OPT_COLORS = ["#A9A9A9", "#a9d18e", "#409cd8", "#fed06e", "#e86252"]


def generate_optimization_figure(
    speedup_data: Dict[str, List[Optional[float]]],
    benchmarks: List[str],
    save_path: Path,
) -> None:
    """
    Stacked bar chart showing the cumulative speedup contribution of each
    optimisation pass level relative to 'w/o Optimization' (= 1.0 baseline).

    speedup_data : dict mapping each variant name in _OPT_VARIANT_NAMES →
                   list of per-benchmark speedup values (last entry = geomean).
                   None values are treated as 0.
    benchmarks   : ordered benchmark names (without "Geomean"; added internally).
    save_path    : destination PDF/PNG path.

    The stacked layers show MARGINAL gains, i.e. how much each additional
    optimisation contributed beyond the previous level.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not available; skipping optimisation figure.")
        return

    plt.rcParams.update({"font.size": 14})

    bench_labels = list(benchmarks) + ["Geomean"]
    n = len(bench_labels)

    # ── resolve None → 0 ─────────────────────────────────────────────────
    total: Dict[str, List[float]] = {
        v: [x if x is not None else 0.0 for x in speedup_data[v]]
        for v in _OPT_VARIANT_NAMES
    }

    # ── compute marginal contributions ───────────────────────────────────
    # marginal[0] = total["w/o Opt"]  (always 1.0)
    # marginal[k] = total[k] - total[k-1]
    marginal: List[List[float]] = []
    for idx, vname in enumerate(_OPT_VARIANT_NAMES):
        if idx == 0:
            marginal.append(list(total[vname]))
        else:
            prev = total[_OPT_VARIANT_NAMES[idx - 1]]
            cur  = total[vname]
            marginal.append([max(0.0, c - p) for c, p in zip(cur, prev)])

    # ── draw stacked bars ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    bar_width = 0.5
    index = np.arange(n)
    bottoms = np.zeros(n)

    for layer_data, color, vname in zip(marginal, _OPT_COLORS, _OPT_VARIANT_NAMES):
        ax.bar(index, layer_data, bar_width, bottom=bottoms,
               color=color, label=vname, edgecolor="black", linewidth=0.5)
        bottoms += np.array(layer_data)

    # ── annotate total speedup on each bar ───────────────────────────────
    top_vals = total["HW-Agnostic + Loop Streaming"]
    for i, val in enumerate(top_vals):
        if val > 0:
            ax.text(i, val + 0.05, f"{val:.2f}×",
                    ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="black")

    # ── axes ─────────────────────────────────────────────────────────────
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

    # ── two-row legend ───────────────────────────────────────────────────
    from matplotlib.patches import Patch
    row1_names = ["w/o Optimization", "Data Type Alignment", "Constant Folding"]
    row2_names = ["Computational Pattern Fusion", "Loop Streaming"]
    colors_map  = dict(zip(_OPT_VARIANT_NAMES, _OPT_COLORS))

    h1 = [Patch(facecolor=colors_map[n], edgecolor="black", linewidth=0.5) for n in row1_names]
    h2 = [Patch(facecolor=colors_map[n], edgecolor="black", linewidth=0.5) for n in row2_names]

    leg2 = ax.legend(h2, row2_names, loc="upper center",
                     bbox_to_anchor=(0.5, 1.17), ncol=2, fontsize=12,
                     handleheight=0.7, handlelength=0.7,
                     columnspacing=2.5, handletextpad=0.3, frameon=False)
    ax.legend(h1, row1_names, loc="upper center",
              bbox_to_anchor=(0.5, 1.28), ncol=3, fontsize=12,
              handleheight=0.7, handlelength=0.7,
              columnspacing=2.5, handletextpad=0.3, frameon=False)
    ax.add_artist(leg2)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    plt.close(fig)
