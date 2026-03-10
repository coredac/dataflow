#!/usr/bin/env python3
"""
Artifact Evaluation – Neura PLDI 2026
=====================================
End-to-end pipeline:
  C++ source -> compile (per-arch) -> mapped MLIR -> extract II+steps
             -> latency model -> normalised metrics -> figures

Figures produced:
  fig13  Speedup (normalised to Marionette)
  fig14  IPC (instructions per cycle)
  fig15  Performance per Area
  fig16  Total Energy (RipTide vs NEURA-SO)
  fig17  Optimisation pass comparison
  fig18  Scalability (4x4 vs 6x6 NEURA-ST)

Run:  python3 evaluation/main.py
"""

from __future__ import annotations
import csv
import functools
import sys
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from util.config import (
    BENCHMARKS_DIR, RESULTS_DIR, FIGS_DIR,
    NEURA_OPT, NEURA_OPT_4_4,
    ARCH_AREA_MM2, ARCH_POWER_MW, ENERGY_ARCHS,
    BENCH_ARCH_INSTRUCTIONS,
    ARCH_CONFIGS, BENCH_ARCH_SEGS, OPT_VARIANTS,
    BENCHMARKS, ARCHS, SCALABILITY_BENCHMARKS,
)
from util.compiler import compile_segment, compile_opt_segment, extract_ii_steps
from util.latency import bench_latency
from util.visualizer import (
    generate_speedup_figure, generate_ppa_figure, generate_ipc_figure,
    generate_energy_figure, generate_optimization_figure, generate_scalability_figure,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _geomean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return functools.reduce(lambda a, b: a * b, values, 1) ** (1 / len(values))


def _print_table(header: list[str], rows: list[list], col_width: int = 14):
    print("".join(f"{h:<{col_width}s}" for h in header))
    for row in rows:
        print("".join(f"{str(v):<{col_width}s}" for v in row))


def _write_csv(path: Path, header: list[str], rows: list[list]):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    print(f"Data written to {path}")


def _fmt(v, fmt=".3f"):
    return f"{v:{fmt}}" if v is not None else "N/A"


# ── Benchmark runners ────────────────────────────────────────────────────

def run_benchmark(bench, arch_name, work_root, verbose=False):
    """Compile + extract II+steps + compute latency for (bench, arch)."""
    key = (bench, arch_name)
    if key not in BENCH_ARCH_SEGS:
        if verbose:
            print(f"  [SKIP] no config for ({bench}, {arch_name})")
        return None

    arch_cfg = ARCH_CONFIGS[arch_name]
    arch_dir = BENCHMARKS_DIR / bench / arch_cfg.folder
    work_dir = work_root / bench / arch_name
    work_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seg in BENCH_ARCH_SEGS[key]:
        if verbose:
            print(f"  Compiling  {bench}/{arch_name}/{seg.cpp_file} ...")
        try:
            mapped = compile_segment(seg, arch_cfg, arch_dir, work_dir)
            ii, steps = extract_ii_steps(mapped)
            if verbose:
                print(f"    -> II={ii}  steps={steps}")
            results.append((seg, ii, steps))
        except Exception as e:
            print(f"  [ERROR] {bench}/{arch_name}/{seg.cpp_file}: {e}")
            return None

    return bench_latency(results)


def run_benchmark_scalability(bench, arch_cfg, neura_opt_binary, work_dir, verbose=False):
    """Compile NEURA-ST segments using given binary, return latency (with steps)."""
    key = (bench, "NEURA-ST")
    if key not in BENCH_ARCH_SEGS:
        return None

    arch_dir = BENCHMARKS_DIR / bench / arch_cfg.folder
    work_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seg in BENCH_ARCH_SEGS[key]:
        if verbose:
            print(f"  Compiling  {bench}/{seg.cpp_file} ...")
        try:
            mapped = compile_segment(seg, arch_cfg, arch_dir, work_dir,
                                     neura_opt_binary=neura_opt_binary)
            ii, steps = extract_ii_steps(mapped)
            if verbose:
                print(f"    -> II={ii}  steps={steps}")
            results.append((seg, ii, steps))
        except Exception as e:
            print(f"  [ERROR] {bench}/{seg.cpp_file}: {e}")
            return None

    return bench_latency(results, include_steps=True)


# ── Evaluation steps (called from main) ─────────────────────────────────

def evaluate_latencies(work_root):
    """Step 1: Compile all benchmarks x archs, return latencies dict."""
    latencies = {arch: {} for arch in ARCHS}
    for bench in BENCHMARKS:
        print(f"\n{'─'*60}\nBenchmark: {bench}")
        for arch in ARCHS:
            lat = run_benchmark(bench, arch, work_root, verbose=True)
            latencies[arch][bench] = lat
            status = f"{lat:.0f} cycles" if lat is not None else "FAILED"
            print(f"  {arch:<12s}  {status}")
    return latencies


def evaluate_speedup(latencies):
    """Step 2: Normalised speedup relative to Marionette -> CSV + figure."""
    speedup_data = {arch: [] for arch in ARCHS}
    for bench in BENCHMARKS:
        base = latencies["Marionette"].get(bench)
        for arch in ARCHS:
            lat = latencies[arch].get(bench)
            speedup_data[arch].append(base / lat if base and lat else None)

    geomeans = {a: _geomean([v for v in speedup_data[a] if v]) for a in ARCHS}

    # Print + CSV
    rows = []
    for i, bench in enumerate(BENCHMARKS):
        rows.append([bench] + [_fmt(speedup_data[a][i]) for a in ARCHS])
    rows.append(["Geomean"] + [_fmt(geomeans[a]) for a in ARCHS])
    print(f"\n{'═'*60}\nNormalised speedup (relative to Marionette)")
    _print_table(["Bench"] + ARCHS, rows)
    _write_csv(RESULTS_DIR / "speedup.csv", ["benchmark"] + ARCHS,
               [[r[0]] + [_fmt(speedup_data[a][i], ".4f") if i < len(BENCHMARKS)
                          else _fmt(geomeans[a], ".4f")
                          for a in ARCHS]
                for i, r in enumerate(rows)])

    # Also write raw latency CSV
    lat_rows = [[bench] + [_fmt(latencies[a].get(bench), ".0f") for a in ARCHS]
                for bench in BENCHMARKS]
    _write_csv(RESULTS_DIR / "latency_cycles.csv", ["benchmark"] + ARCHS, lat_rows)

    generate_speedup_figure(speedup_data, geomeans, BENCHMARKS, ARCHS,
                            save_path=FIGS_DIR / "fig13.pdf")
    return speedup_data, geomeans


def evaluate_ppa(latencies, speedup_data):
    """Step 3: Normalised Performance per Area -> CSV + figure."""
    area_mario = ARCH_AREA_MM2["Marionette"]
    ppa_data = {arch: [] for arch in ARCHS}

    for i, bench in enumerate(BENCHMARKS):
        for arch in ARCHS:
            sp = speedup_data[arch][i]
            if sp is not None and arch in ARCH_AREA_MM2:
                ppa_data[arch].append(sp * (area_mario / ARCH_AREA_MM2[arch]))
            else:
                ppa_data[arch].append(None)

    geomeans = {a: _geomean([v for v in ppa_data[a] if v]) for a in ARCHS}

    rows = []
    for i, bench in enumerate(BENCHMARKS):
        rows.append([bench] + [_fmt(ppa_data[a][i]) for a in ARCHS])
    rows.append(["Geomean"] + [_fmt(geomeans[a]) for a in ARCHS])
    print(f"\n{'═'*60}\nNormalised Perf/Area (relative to Marionette)")
    _print_table(["Bench"] + ARCHS, rows)
    _write_csv(RESULTS_DIR / "perf_per_area.csv", ["benchmark"] + ARCHS,
               [[r[0]] + [_fmt(ppa_data[a][i], ".4f") if i < len(BENCHMARKS)
                          else _fmt(geomeans[a], ".4f")
                          for a in ARCHS]
                for i, r in enumerate(rows)])

    generate_ppa_figure(ppa_data, geomeans, BENCHMARKS, ARCHS,
                        save_path=FIGS_DIR / "fig15.pdf")


def evaluate_ipc(latencies):
    """Step 4: Instructions per Cycle -> CSV + figure."""
    ipc_data = {arch: [] for arch in ARCHS}
    for bench in BENCHMARKS:
        for arch in ARCHS:
            n_instr = BENCH_ARCH_INSTRUCTIONS.get((bench, arch))
            cycles = latencies[arch].get(bench)
            ipc_data[arch].append(n_instr / cycles if n_instr and cycles else None)

    geomeans = {a: _geomean([v for v in ipc_data[a] if v]) for a in ARCHS}

    rows = []
    for i, bench in enumerate(BENCHMARKS):
        rows.append([bench] + [_fmt(ipc_data[a][i]) for a in ARCHS])
    rows.append(["Geomean"] + [_fmt(geomeans[a]) for a in ARCHS])
    print(f"\n{'═'*60}\nIPC (instructions per cycle)")
    _print_table(["Bench"] + ARCHS, rows)
    _write_csv(RESULTS_DIR / "ipc.csv", ["benchmark"] + ARCHS,
               [[r[0]] + [_fmt(ipc_data[a][i], ".4f") if i < len(BENCHMARKS)
                          else _fmt(geomeans[a], ".4f")
                          for a in ARCHS]
                for i, r in enumerate(rows)])

    generate_ipc_figure(ipc_data, geomeans, BENCHMARKS, ARCHS,
                        save_path=FIGS_DIR / "fig14.pdf")


def evaluate_energy():
    """Step 5: Normalised total energy (RipTide baseline) -> CSV + figure."""
    energy_data = {arch: [] for arch in ENERGY_ARCHS}
    for bench in BENCHMARKS:
        rip_instr = BENCH_ARCH_INSTRUCTIONS.get((bench, "RipTide"))
        rip_energy = ARCH_POWER_MW["RipTide"] * rip_instr if rip_instr else None
        for arch in ENERGY_ARCHS:
            n_instr = BENCH_ARCH_INSTRUCTIONS.get((bench, arch))
            if n_instr and rip_energy:
                energy_data[arch].append((ARCH_POWER_MW[arch] * n_instr) / rip_energy)
            else:
                energy_data[arch].append(None)

    geomeans = {a: _geomean([v for v in energy_data[a] if v]) for a in ENERGY_ARCHS}

    rows = []
    for i, bench in enumerate(BENCHMARKS):
        rows.append([bench] + [_fmt(energy_data[a][i]) for a in ENERGY_ARCHS])
    rows.append(["Geomean"] + [_fmt(geomeans[a]) for a in ENERGY_ARCHS])
    print(f"\n{'═'*60}\nNormalised total energy (relative to RipTide)")
    _print_table(["Bench"] + ENERGY_ARCHS, rows)
    _write_csv(RESULTS_DIR / "energy.csv", ["benchmark"] + ENERGY_ARCHS,
               [[r[0]] + [_fmt(energy_data[a][i], ".4f") if i < len(BENCHMARKS)
                          else _fmt(geomeans[a], ".4f")
                          for a in ENERGY_ARCHS]
                for i, r in enumerate(rows)])

    generate_energy_figure(energy_data, geomeans, BENCHMARKS, ENERGY_ARCHS,
                           save_path=FIGS_DIR / "fig16.pdf")


def evaluate_optimization(work_root):
    """Step 6: Optimisation pass comparison on NEURA-ST -> CSV + figure."""
    neura_st_cfg = ARCH_CONFIGS["NEURA-ST"]
    opt_latencies = {v.name: {} for v in OPT_VARIANTS}

    print(f"\n{'═'*60}\nOptimisation comparison (NEURA-ST)")
    for bench in BENCHMARKS:
        print(f"  Benchmark: {bench}")
        key = (bench, "NEURA-ST")
        if key not in BENCH_ARCH_SEGS:
            for v in OPT_VARIANTS:
                opt_latencies[v.name][bench] = None
            continue

        segs = BENCH_ARCH_SEGS[key]
        for variant in OPT_VARIANTS:
            variant_work = work_root / "opt" / bench / variant.name.replace(" ", "_")
            variant_work.mkdir(parents=True, exist_ok=True)
            try:
                seg_results = []
                for seg in segs:
                    mapped = compile_opt_segment(
                        seg, neura_st_cfg,
                        BENCHMARKS_DIR / bench / neura_st_cfg.folder,
                        variant_work, variant,
                    )
                    ii, steps = extract_ii_steps(mapped)
                    seg_results.append((seg, ii, steps))
                lat = bench_latency(seg_results)
                opt_latencies[variant.name][bench] = lat
                print(f"    {variant.name:<42s}  {lat:.0f} cycles")
            except Exception as e:
                print(f"    [ERROR] {variant.name}: {e}")
                opt_latencies[variant.name][bench] = None

    # Speedup relative to 'w/o Optimization'
    vnames = [v.name for v in OPT_VARIANTS]
    opt_speedup = {vn: [] for vn in vnames}
    for bench in BENCHMARKS:
        base_lat = opt_latencies["w/o Optimization"].get(bench)
        for vn in vnames:
            lat = opt_latencies[vn].get(bench)
            opt_speedup[vn].append(base_lat / lat if base_lat and lat else None)

    # Append geomean
    for vn in vnames:
        vals = [v for v in opt_speedup[vn] if v is not None]
        opt_speedup[vn].append(_geomean(vals) or 0.0)

    print(f"\nOptimisation geomeans:")
    for vn in vnames:
        print(f"  {vn:<42s}  {_fmt(opt_speedup[vn][-1])}")

    _write_csv(RESULTS_DIR / "optimization.csv", ["benchmark"] + vnames,
               [[BENCHMARKS[i]] + [_fmt(opt_speedup[vn][i], ".4f") for vn in vnames]
                for i in range(len(BENCHMARKS))]
               + [["Geomean"] + [_fmt(opt_speedup[vn][-1], ".4f") for vn in vnames]])

    generate_optimization_figure(opt_speedup, BENCHMARKS,
                                 save_path=FIGS_DIR / "fig17.pdf")


def evaluate_scalability(work_root):
    """Step 7: 4x4 vs 6x6 NEURA-ST scalability -> CSV + figure."""
    _4x4, _6x6 = r"4x4 NEURA-ST", r"6x6 NEURA-ST"
    configs = {_4x4: NEURA_OPT_4_4, _6x6: NEURA_OPT}
    neura_st_cfg = ARCH_CONFIGS["NEURA-ST"]

    scal_latencies = {name: {} for name in configs}

    print(f"\n{'═'*60}\nScalability comparison (4x4 vs 6x6 NEURA-ST)")
    for bench in SCALABILITY_BENCHMARKS:
        print(f"  Benchmark: {bench}")
        for cfg_name, binary in configs.items():
            tag = "4x4" if binary == NEURA_OPT_4_4 else "6x6"
            work_dir = work_root / "scalability" / bench / tag
            lat = run_benchmark_scalability(bench, neura_st_cfg, binary, work_dir, verbose=True)
            scal_latencies[cfg_name][bench] = lat
            status = f"{lat:.0f} cycles" if lat is not None else "FAILED"
            print(f"    {cfg_name:<22s}  {status}")

    # Normalise: 4x4 = 1.0
    scal_speedup = {name: [] for name in configs}
    for bench in SCALABILITY_BENCHMARKS:
        lat_4 = scal_latencies[_4x4].get(bench)
        lat_6 = scal_latencies[_6x6].get(bench)
        scal_speedup[_4x4].append(1.0 if lat_4 is not None else None)
        scal_speedup[_6x6].append(lat_4 / lat_6 if lat_4 and lat_6 else None)

    # Append geomeans
    for name in configs:
        vals = [v for v in scal_speedup[name] if v is not None]
        scal_speedup[name].append(_geomean(vals) or 0.0)

    # Improvement rate
    n = len(SCALABILITY_BENCHMARKS) + 1
    improvement_rate = [
        (scal_speedup[_6x6][i] - scal_speedup[_4x4][i]) * 100
        if scal_speedup[_6x6][i] is not None and scal_speedup[_4x4][i] is not None
        else 0.0
        for i in range(n)
    ]

    print(f"\nScalability geomeans:")
    for name in configs:
        print(f"  {name:<22s}  {scal_speedup[name][-1]:.3f}x")

    cfg_names = list(configs.keys())
    _write_csv(RESULTS_DIR / "scalability.csv",
               ["benchmark"] + cfg_names + ["improvement_rate_%"],
               [[SCALABILITY_BENCHMARKS[i]]
                + [_fmt(scal_speedup[c][i], ".4f") for c in cfg_names]
                + [f"{improvement_rate[i]:.2f}"]
                for i in range(len(SCALABILITY_BENCHMARKS))]
               + [["Geomean"]
                  + [_fmt(scal_speedup[c][-1], ".4f") for c in cfg_names]
                  + [f"{improvement_rate[-1]:.2f}"]])

    generate_scalability_figure(scal_speedup, improvement_rate,
                                SCALABILITY_BENCHMARKS,
                                save_path=FIGS_DIR / "fig18.pdf")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="neura_ae_") as tmp:
        work_root = Path(tmp)

        # Step 1: Compile all benchmarks, collect latencies
        latencies = evaluate_latencies(work_root)

        # Step 2: Speedup (fig13)
        speedup_data, _ = evaluate_speedup(latencies)

        # Step 3: Performance per Area (fig15)
        evaluate_ppa(latencies, speedup_data)

        # Step 4: IPC (fig14)
        evaluate_ipc(latencies)

        # Step 5: Energy (fig16)
        evaluate_energy()

        # Step 6: Optimisation comparison (fig17)
        evaluate_optimization(work_root)

        # Step 7: Scalability (fig18)
        evaluate_scalability(work_root)


if __name__ == "__main__":
    main()
