"""
Configuration data for Neura PLDI 2026 artifact evaluation.

All tool paths, architecture definitions, benchmark segment configs,
optimization variants, and physical constants live here.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Tool paths (adjust if locations differ) ──────────────────────────────

CGEIST    = "/home/lucas/Project/dataflow/thirdparty/polygeist/cgeist"
MLIR_OPT  = "mlir-opt"
NEURA_OPT = "/home/lucas/Project/dataflow/build/tools/mlir-neura-opt/mlir-neura-opt"
NEURA_OPT_4_4 = "/home/lucas/Project/dataflow/build/tools/mlir-neura-opt/mlir-neura-opt-4x4"

BENCHMARKS_DIR = Path("/home/lucas/Project/dataflow/evaluation/benchmarks")
RESULTS_DIR   = Path("/home/lucas/Project/dataflow/evaluation/results")
FIGS_DIR      = Path("/home/lucas/Project/dataflow/evaluation/figs")

# ── Physical constants ───────────────────────────────────────────────────

CPU_TRANSITION_CYCLES       = 20   # CPU <-> CGRA context switch
MARIONETTE_TRANSITION_CYCLES = 5   # Marionette controller-based switch

ARCH_POWER_MW: dict[str, float] = {
    "RipTide":  9.311,
    "NEURA-SO": 9.262,
}
ENERGY_ARCHS = ["RipTide", "NEURA-SO"]

ARCH_AREA_MM2: dict[str, float] = {
    "Marionette": 2.68353,
    "ICED":       0.882533225,
    "RipTide":    0.679615265,
    "NEURA-SO":   0.668016,
    "NEURA-ST":   0.922508964,
}

# ── Runtime instruction counts per (benchmark, arch) ────────────────────
# Used for IPC = instructions / cycles and energy = power * instructions.

BENCH_ARCH_INSTRUCTIONS: dict[tuple, Optional[int]] = {
    ("conv", "Marionette"): 114573312,  ("conv", "ICED"): 305528832,
    ("conv", "RipTide"): 196411392,     ("conv", "NEURA-SO"): 190955520,
    ("conv", "NEURA-ST"): 190955520,

    ("relu", "Marionette"): 10752,      ("relu", "ICED"): 11776,
    ("relu", "RipTide"): 9216,          ("relu", "NEURA-SO"): 15360,
    ("relu", "NEURA-ST"): 15360,

    ("spmv", "Marionette"): 79040,      ("spmv", "ICED"): 271700,
    ("spmv", "RipTide"): 167960,        ("spmv", "NEURA-SO"): 167960,
    ("spmv", "NEURA-ST"): 207480,

    ("gemm", "Marionette"): 4718592,    ("gemm", "ICED"): 13893632,
    ("gemm", "RipTide"): 8650752,       ("gemm", "NEURA-SO"): 9175040,
    ("gemm", "NEURA-ST"): 9699328,

    ("bicg", "Marionette"): 87790500,   ("bicg", "ICED"): 211470000,
    ("bicg", "RipTide"): 199500000,     ("bicg", "NEURA-SO"): 191520000,
    ("bicg", "NEURA-ST"): 159600000,

    ("mvt", "Marionette"): 80000000,    ("mvt", "ICED"): 208000000,
    ("mvt", "RipTide"): 136000000,      ("mvt", "NEURA-SO"): 140000000,
    ("mvt", "NEURA-ST"): 144000000,

    ("jacobi", "Marionette"): 28000000, ("jacobi", "ICED"): 86000000,
    ("jacobi", "RipTide"): 56000000,    ("jacobi", "NEURA-SO"): 48000000,
    ("jacobi", "NEURA-ST"): 48000000,

    ("fft", "Marionette"): 3456,        ("fft", "ICED"): 7296,
    ("fft", "RipTide"): 3584,           ("fft", "NEURA-SO"): 3456,
    ("fft", "NEURA-ST"): 5376,

    ("merge-sort", "Marionette"): 23552, ("merge-sort", "ICED"): 25600,
    ("merge-sort", "RipTide"): 13312,    ("merge-sort", "NEURA-SO"): 23552,
    ("merge-sort", "NEURA-ST"): 45056,

    ("bfs", "Marionette"): 6144,        ("bfs", "ICED"): 6656,
    ("bfs", "RipTide"): 5120,           ("bfs", "NEURA-SO"): 6144,
    ("bfs", "NEURA-ST"): 15360,

    ("floyd", "Marionette"): 12000000000, ("floyd", "ICED"): 12000000000,
    ("floyd", "RipTide"): 12000000000,    ("floyd", "NEURA-SO"): 12000000000,
    ("floyd", "NEURA-ST"): 39000000000,
}

# ── Architecture configurations ──────────────────────────────────────────

@dataclass
class ArchConfig:
    name: str
    folder: str              # subfolder inside PLDI-Test/<bench>/
    dataflow_passes: list
    mapping_mode: str        # "spatial-temporal" | "spatial-only"
    mapped_suffix: str
    is_body_only: bool = False


ARCH_CONFIGS: dict[str, ArchConfig] = {
    "NEURA-ST": ArchConfig(
        name="NEURA-ST", folder="NEURA-ST",
        dataflow_passes=["--canonicalize-cast", "--fold-constant",
                         "--canonicalize-live-in", "--leverage-predicated-value",
                         "--transform-ctrl-to-data-flow", "--fold-constant"],
        mapping_mode="spatial-temporal", mapped_suffix="_st_map.mlir",
    ),
    "NEURA-SO": ArchConfig(
        name="NEURA-SO", folder="NEURA-SO",
        dataflow_passes=["--canonicalize-cast", "--fold-constant",
                         "--canonicalize-live-in", "--leverage-predicated-value",
                         "--transform-ctrl-to-data-flow", "--fold-constant"],
        mapping_mode="spatial-only", mapped_suffix="_so_map.mlir",
    ),
    "ICED": ArchConfig(
        name="ICED", folder="ICED",
        dataflow_passes=["--canonicalize-cast",
                         "--canonicalize-live-in", "--leverage-predicated-value",
                         "--transform-ctrl-to-data-flow"],
        mapping_mode="spatial-temporal", mapped_suffix="_st_map.mlir",
    ),
    "Marionette": ArchConfig(
        name="Marionette", folder="Marionette",
        dataflow_passes=["--canonicalize-cast", "--fold-constant",
                         "--canonicalize-live-in", "--leverage-predicated-value",
                         "--transform-ctrl-to-data-flow", "--fold-constant"],
        mapping_mode="spatial-only", mapped_suffix="_so_map.mlir",
        is_body_only=True,
    ),
    "RipTide": ArchConfig(
        name="RipTide", folder="RipTide",
        dataflow_passes=["--canonicalize-cast",
                         "--canonicalize-live-in", "--leverage-predicated-value",
                         "--transform-ctrl-to-data-flow",
                         "--transform-to-steer-control", "--remove-predicated-type"],
        mapping_mode="spatial-only", mapped_suffix="_steer_map.mlir",
    ),
}

# ── Segment configuration ────────────────────────────────────────────────
#
# SegConfig describes one compiled segment (one .cpp -> one mapped MLIR).
#
# body_only=False: CGRA pipelines cgra_trips inner iterations; CPU drives outer.
# body_only=True:  CGRA executes single body; CPU drives ALL loops.
# group >= 0:      Segments share an outer CPU loop (costs combined, not summed
#                  independently).  See bench_latency() for details.

@dataclass
class SegConfig:
    cpp_file: str
    body_only: bool
    cpu_trips: list
    cgra_trips: int = 1
    group: int = -1
    group_outer_trips: list = field(default_factory=list)
    fast_switch: bool = True


def _SM(cpp, outer, inner):
    """Self-managed: CGRA pipelines inner loop, CPU drives outer."""
    return SegConfig(cpp, body_only=False, cpu_trips=outer, cgra_trips=inner)

def _CL(cpp, all_trips, fast_switch=True):
    """Controller-based: CPU drives all loops, CGRA maps body."""
    return SegConfig(cpp, body_only=True, cpu_trips=list(all_trips),
                     fast_switch=fast_switch)

def _MG(cpp, inner_trips, group_id, outer_trips):
    """Marionette grouped: shares outer loop with sibling segments."""
    return SegConfig(cpp, body_only=True, cpu_trips=list(inner_trips),
                     group=group_id, group_outer_trips=list(outer_trips))


BENCH_ARCH_SEGS: dict[tuple, list] = {
    # conv
    ("conv", "NEURA-ST"):   [_SM("conv.cpp",  [28416], 192)],
    ("conv", "NEURA-SO"):   [_SM("conv.cpp",  [28416], 192)],
    ("conv", "ICED"):       [_SM("conv.cpp",  [28416], 192)],
    ("conv", "RipTide"):    [_SM("conv.cpp",  [28416], 192)],
    ("conv", "Marionette"): [_CL("conv.cpp", [28416, 192])],
    # relu
    ("relu", "NEURA-ST"):   [_SM("relu.cpp", [], 512)],
    ("relu", "NEURA-SO"):   [_SM("relu.cpp", [], 512)],
    ("relu", "ICED"):       [_CL("relu.cpp", [512], fast_switch=False)],
    ("relu", "RipTide"):    [_CL("relu.cpp", [512], fast_switch=False)],
    ("relu", "Marionette"): [_CL("relu.cpp", [512])],
    # spmv
    ("spmv", "NEURA-ST"):   [_SM("spmv.cpp", [494], 10)],
    ("spmv", "NEURA-SO"):   [_SM("spmv.cpp", [494], 10)],
    ("spmv", "ICED"):       [_SM("spmv.cpp", [494], 10)],
    ("spmv", "RipTide"):    [_SM("spmv.cpp", [494], 10)],
    ("spmv", "Marionette"): [_CL("spmv.cpp", [494, 10])],
    # gemm
    ("gemm", "NEURA-ST"):   [_SM("gemm.cpp", [4096], 64)],
    ("gemm", "NEURA-SO"):   [_SM("gemm.cpp", [4096], 64)],
    ("gemm", "ICED"):       [_SM("gemm.cpp", [4096], 64)],
    ("gemm", "RipTide"):    [_SM("gemm.cpp", [4096], 64)],
    ("gemm", "Marionette"): [_CL("gemm.cpp", [4096, 64])],
    # bicg
    ("bicg", "NEURA-ST"):   [_SM("bicg.cpp", [2100], 1900)],
    ("bicg", "ICED"):       [_SM("bicg.cpp", [2100], 1900)],
    ("bicg", "NEURA-SO"):   [_SM("bicg_kernel1.cpp", [2100], 1900),
                              _SM("bicg_kernel2.cpp", [2100], 1900)],
    ("bicg", "RipTide"):    [_SM("bicg_kernel1.cpp", [2100], 1900),
                              _SM("bicg_kernel2.cpp", [2100], 1900)],
    ("bicg", "Marionette"): [_MG("bicg_outer.cpp", [1],    group_id=0, outer_trips=[2100]),
                              _MG("bicg.cpp",       [1900], group_id=0, outer_trips=[2100])],
    # mvt
    ("mvt", "NEURA-ST"):   [_SM("mvt.cpp", [2000], 2000)],
    ("mvt", "NEURA-SO"):   [_SM("mvt.cpp", [2000], 2000)],
    ("mvt", "ICED"):       [_SM("mvt.cpp", [2000], 2000)],
    ("mvt", "RipTide"):    [_SM("mvt.cpp", [2000], 2000)],
    ("mvt", "Marionette"): [_CL("mvt.cpp", [2000, 2000])],
    # jacobi
    ("jacobi", "NEURA-ST"):   [_SM("jacobi_kernel1.cpp", [500], 2000),
                                _SM("jacobi_kernel2.cpp", [500], 2000)],
    ("jacobi", "NEURA-SO"):   [_SM("jacobi_kernel1.cpp", [500], 2000),
                                _SM("jacobi_kernel2.cpp", [500], 2000)],
    ("jacobi", "ICED"):       [_SM("jacobi_kernel1.cpp", [500], 2000),
                                _SM("jacobi_kernel2.cpp", [500], 2000)],
    ("jacobi", "RipTide"):    [_SM("jacobi_kernel1.cpp", [500], 2000),
                                _SM("jacobi_kernel2.cpp", [500], 2000)],
    ("jacobi", "Marionette"): [_CL("jacobi_kernel1.cpp", [500, 2000]),
                                _CL("jacobi_kernel2.cpp", [500, 2000])],
    # fft
    ("fft", "NEURA-ST"):   [_SM("fft.cpp", [], 128)],
    ("fft", "NEURA-SO"):   [_CL("fft.cpp", [128], fast_switch=False)],
    ("fft", "ICED"):       [_SM("fft.cpp", [], 128)],
    ("fft", "RipTide"):    [_CL("fft.cpp", [128], fast_switch=False)],
    ("fft", "Marionette"): [_CL("fft.cpp", [128])],
    # merge-sort
    ("merge-sort", "NEURA-ST"):   [_SM("merge-sort.cpp", [], 1024)],
    ("merge-sort", "NEURA-SO"):   [_CL("merge-sort.cpp", [1024], fast_switch=False)],
    ("merge-sort", "ICED"):       [_CL("merge-sort.cpp", [1024], fast_switch=False)],
    ("merge-sort", "RipTide"):    [_CL("merge-sort.cpp", [1024], fast_switch=False)],
    ("merge-sort", "Marionette"): [_CL("merge-sort.cpp", [1024])],
    # bfs
    ("bfs", "NEURA-ST"):   [_SM("bfs.cpp", [], 256)],
    ("bfs", "NEURA-SO"):   [_CL("bfs.cpp", [256], fast_switch=False)],
    ("bfs", "ICED"):       [_CL("bfs.cpp", [256], fast_switch=False)],
    ("bfs", "RipTide"):    [_CL("bfs.cpp", [256], fast_switch=False)],
    ("bfs", "Marionette"): [_CL("bfs.cpp", [256])],
    # floyd
    ("floyd", "NEURA-ST"):   [_SM("floyd.cpp", [1000, 1000], 1000)],
    ("floyd", "NEURA-SO"):   [_CL("floyd.cpp", [1000, 1000, 1000], fast_switch=False)],
    ("floyd", "ICED"):       [_CL("floyd.cpp", [1000, 1000, 1000], fast_switch=False)],
    ("floyd", "RipTide"):    [_CL("floyd.cpp", [1000, 1000, 1000], fast_switch=False)],
    ("floyd", "Marionette"): [_CL("floyd.cpp", [1000, 1000, 1000])],
}

# ── Optimization variant configurations (for fig17) ─────────────────────

@dataclass
class OptVariant:
    name: str
    use_finalize_memref: bool   = False
    dataflow_passes: list       = field(default_factory=list)
    extra_pre_map_passes: list  = field(default_factory=list)


OPT_VARIANTS: list[OptVariant] = [
    OptVariant(
        name="w/o Optimization",
        use_finalize_memref=True,
        dataflow_passes=["--canonicalize-live-in", "--leverage-predicated-value",
                         "--transform-ctrl-to-data-flow"],
    ),
    OptVariant(
        name="Computational Pattern Fusion",
        dataflow_passes=["--canonicalize-live-in", "--leverage-predicated-value",
                         "--transform-ctrl-to-data-flow"],
    ),
    OptVariant(
        name="Data Type Alignment",
        dataflow_passes=["--canonicalize-cast", "--canonicalize-live-in",
                         "--leverage-predicated-value", "--transform-ctrl-to-data-flow"],
    ),
    OptVariant(
        name="Data Type Alignment + Constant Folding",
        dataflow_passes=["--canonicalize-cast", "--fold-constant",
                         "--canonicalize-live-in", "--leverage-predicated-value",
                         "--transform-ctrl-to-data-flow", "--fold-constant"],
    ),
    OptVariant(
        name="HW-Agnostic + Loop Streaming",
        dataflow_passes=["--canonicalize-cast", "--fold-constant",
                         "--canonicalize-live-in", "--leverage-predicated-value",
                         "--transform-ctrl-to-data-flow", "--fold-constant"],
        extra_pre_map_passes=["--fuse-loop-control"],
    ),
]

# ── Benchmark and architecture lists ─────────────────────────────────────

BENCHMARKS = [
    "conv", "relu", "spmv", "gemm", "bicg", "mvt",
    "jacobi", "fft", "merge-sort", "bfs", "floyd",
]
ARCHS = ["Marionette", "ICED", "RipTide", "NEURA-SO", "NEURA-ST"]
SCALABILITY_BENCHMARKS = ["bicg", "mvt", "jacobi", "fft", "merge-sort", "bfs", "floyd"]
