"""
Figure generation utilities for Neura PLDI 2026 artifact evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


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
