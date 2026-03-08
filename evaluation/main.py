#!/usr/bin/env python3
"""
Artifact Evaluation – Neura PLDI 2026
=====================================
End-to-end pipeline:
  C++ source  →  (per-arch compilation)  →  Mapped MLIR
                         ↓
               Extract compiled_II + steps
                         ↓
            Latency model per architecture
                         ↓
          Speedup normalised to Marionette → plot

Run:  python3 evaluation/main.py
"""

from __future__ import annotations
import csv
import functools
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Allow   from util.figures import ...   regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))
from util.figures import generate_speedup_figure, generate_ppa_figure, generate_ipc_figure

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – TOOL PATHS  (adjust if locations differ)
# ════════════════════════════════════════════════════════════════════════════

CGEIST    = "/home/lucas/Project/dataflow/thirdparty/polygeist/cgeist"
MLIR_OPT  = "mlir-opt"
NEURA_OPT = "/home/lucas/Project/dataflow/build/tools/mlir-neura-opt/mlir-neura-opt"

PLDI_TEST_DIR = Path("/home/lucas/Project/dataflow/PLDI-Test")
RESULTS_DIR   = Path("/home/lucas/Project/dataflow/evaluation/results")
FIGS_DIR      = Path("/home/lucas/Project/dataflow/evaluation/figs")

# Cycles lost on each CPU ↔ CGRA context switch
CPU_TRANSITION_CYCLES = 20
# Cycles lost on Marionette controller-based context switch
MARIONETTE_TRANSITION_CYCLES = 5

# ── Total instruction counts per (benchmark, architecture) ────────────────
# IPC = BENCH_ARCH_INSTRUCTIONS[(bench, arch)] / total_cycles
# These are manually collected from the mapped MLIR files, counting the number of instructions and computing total instructions during runtime.
BENCH_ARCH_INSTRUCTIONS: dict[tuple, Optional[int]] = {
    # ── conv ────────────────────────────────────────────────────────
    ("conv", "Marionette"): 114573312,
    ("conv", "ICED"):       305528832,
    ("conv", "RipTide"):    196411392,
    ("conv", "NEURA-SO"):   190955520,
    ("conv", "NEURA-ST"):   190955520,
    # ── relu ────────────────────────────────────────────────────────
    ("relu", "Marionette"): 10752,
    ("relu", "ICED"):       11776,
    ("relu", "RipTide"):    9216,
    ("relu", "NEURA-SO"):   15360,
    ("relu", "NEURA-ST"):   15360,
    # ── spmv ────────────────────────────────────────────────────────
    ("spmv", "Marionette"): 79040,
    ("spmv", "ICED"):       271700,
    ("spmv", "RipTide"):    167960,
    ("spmv", "NEURA-SO"):   167960,
    ("spmv", "NEURA-ST"):   207480,
    # ── gemm ────────────────────────────────────────────────────────
    ("gemm", "Marionette"): 4718592,
    ("gemm", "ICED"):       13893632,
    ("gemm", "RipTide"):    8650752,
    ("gemm", "NEURA-SO"):   9175040,
    ("gemm", "NEURA-ST"):   9699328,
    # ── bicg ────────────────────────────────────────────────────────
    ("bicg", "Marionette"): 87790500,
    ("bicg", "ICED"):       211470000,
    ("bicg", "RipTide"):    199500000,
    ("bicg", "NEURA-SO"):   191520000,
    ("bicg", "NEURA-ST"):   159600000,
    # ── mvt ─────────────────────────────────────────────────────────
    ("mvt", "Marionette"):  80000000,
    ("mvt", "ICED"):        208000000,
    ("mvt", "RipTide"):     136000000,
    ("mvt", "NEURA-SO"):    140000000,
    ("mvt", "NEURA-ST"):    144000000,
    # ── jacobi ──────────────────────────────────────────────────────
    ("jacobi", "Marionette"): 28000000,
    ("jacobi", "ICED"):       86000000,
    ("jacobi", "RipTide"):    56000000,
    ("jacobi", "NEURA-SO"):   48000000,
    ("jacobi", "NEURA-ST"):   48000000,
    # ── fft ─────────────────────────────────────────────────────────
    ("fft", "Marionette"):  3456,
    ("fft", "ICED"):        7296,
    ("fft", "RipTide"):     3584,
    ("fft", "NEURA-SO"):    3456,
    ("fft", "NEURA-ST"):    5376,
    # ── merge-sort ──────────────────────────────────────────────────
    ("merge-sort", "Marionette"): 23552,
    ("merge-sort", "ICED"):       25600,
    ("merge-sort", "RipTide"):    13312,
    ("merge-sort", "NEURA-SO"):   23552,
    ("merge-sort", "NEURA-ST"):   45056,
    # ── bfs ─────────────────────────────────────────────────────────
    ("bfs", "Marionette"):  6144,
    ("bfs", "ICED"):        6656,
    ("bfs", "RipTide"):     5120,
    ("bfs", "NEURA-SO"):    6144,
    ("bfs", "NEURA-ST"):    15360,
    # ── floyd ───────────────────────────────────────────────────────
    ("floyd", "Marionette"): 12000000000,
    ("floyd", "ICED"):       12000000000,
    ("floyd", "RipTide"):    12000000000,
    ("floyd", "NEURA-SO"):   12000000000,
    ("floyd", "NEURA-ST"):   39000000000
}

# CGRA area in mm²  (used for normalised Perf/Area computation)
ARCH_AREA_MM2: dict[str, float] = {
    "Marionette": 2.68353,
    "ICED":       0.882533225,
    "RipTide":    0.679615265,
    "NEURA-SO":   0.668016,
    "NEURA-ST":   0.922508964,
}

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – ARCHITECTURE CONFIGURATIONS
#
#  Each ArchConfig bundles together:
#   • the folder name inside PLDI-Test/<bench>/
#   • the neura-opt pass sequence that produces the dataflow IR
#   • the mapping mode and the expected suffix of the resulting mapped file
#   • is_body_only: Marionette maps the innermost body (both loops CPU-driven)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ArchConfig:
    name: str
    folder: str              # e.g. "NEURA-ST", "OpenCGRA", "Marionette"
    dataflow_passes: list    # passed directly to neura-opt after neura lowering
    mapping_mode: str        # "spatial-temporal" | "spatial-only"
    mapped_suffix: str       # suffix of the final mapped MLIR file
    is_body_only: bool = False


ARCH_CONFIGS: dict[str, ArchConfig] = {

    "NEURA-ST": ArchConfig(
        name="NEURA-ST", folder="NEURA-ST",
        dataflow_passes=[
            "--canonicalize-cast", "--fold-constant",
            "--canonicalize-live-in", "--leverage-predicated-value",
            "--transform-ctrl-to-data-flow", "--fold-constant",
        ],
        mapping_mode="spatial-temporal",
        mapped_suffix="_st_map.mlir",
    ),

    "NEURA-SO": ArchConfig(
        name="NEURA-SO", folder="NEURA-SO",
        dataflow_passes=[
            "--canonicalize-cast", "--fold-constant",
            "--canonicalize-live-in", "--leverage-predicated-value",
            "--transform-ctrl-to-data-flow", "--fold-constant",
        ],
        mapping_mode="spatial-only",
        mapped_suffix="_so_map.mlir",
    ),

    # ICED baseline
    "ICED": ArchConfig(
        name="ICED", folder="OpenCGRA",
        dataflow_passes=[
            "--canonicalize-cast",
            "--canonicalize-live-in", "--leverage-predicated-value",
            "--transform-ctrl-to-data-flow",
        ],
        mapping_mode="spatial-temporal",
        mapped_suffix="_st_map.mlir",
    ),

    # Marionette: spatial-only, maps the innermost BODY (CPU drives every loop)
    "Marionette": ArchConfig(
        name="Marionette", folder="Marionette",
        dataflow_passes=[
            "--canonicalize-cast","--fold-constant",
            "--canonicalize-live-in", "--leverage-predicated-value",
            "--transform-ctrl-to-data-flow", "--fold-constant",
        ],
        mapping_mode="spatial-only",
        mapped_suffix="_so_map.mlir",
        is_body_only=True,
    ),

    # RipTide: spatial-only, steering-based dataflow
    "RipTide": ArchConfig(
        name="RipTide", folder="RipTide",
        dataflow_passes=[
            "--canonicalize-cast",
            "--canonicalize-live-in", "--leverage-predicated-value",
            "--transform-ctrl-to-data-flow",
            "--transform-to-steer-control", "--remove-predicated-type",
        ],
        mapping_mode="spatial-only",
        mapped_suffix="_steer_map.mlir",
    ),
}

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – BENCHMARK EXECUTION CONFIGURATIONS
#
#  SegConfig describes ONE compiled segment (one .cpp → one mapped MLIR).
#
#  Two execution models
#  ─────────────────────────────────────────────────────────────────────────
#  body_only = False  (NEURA-ST / NEURA-SO / ICED / RipTide)
#    • CGRA pipelines `cgra_trips` inner iterations.
#    • CPU drives all enclosing loops  (trip counts in `cpu_trips`).
#    • Latency = prod(cpu_trips) × [(cgra_trips − 1) × II + steps
#                                    + CPU_TRANSITION_CYCLES]
#
#  body_only = True   (Marionette)
#    • CGRA executes a single body invocation (no loop inside the function).
#    • CPU drives ALL loops; `cpu_trips` lists every level top→bottom.
#    • Latency = prod(cpu_trips) × [steps + CPU_TRANSITION_CYCLES]
#
#  Multi-kernel benchmarks (jacobi, bicg-SO, bicg-RipTide) are stored as a
#  list of SegConfigs.  Latencies are combined in two ways:
#
#  • Independent segments (group=-1): summed independently.
#  • Grouped segments (group≥0): share the same outer CPU loop.  Their
#    per-outer-iteration costs are summed first, then multiplied by the
#    shared outer trip counts.  This avoids double-counting the outer t.
#    Example bicg Marionette, t=5:
#      cost = 2100 × [(steps_outer+t)·1 + (steps_inner+t)·1900]
#           = ((5+5)·1900 + 5 + 1) × 2100
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SegConfig:
    cpp_file: str        # relative to PLDI_TEST_DIR/<bench>/<arch_folder>/
    body_only: bool      # False → CGRA owns inner loop; True → Marionette body
    cpu_trips: list      # CPU-driven loop trip counts for THIS segment (outermost → innermost)
    cgra_trips: int = 1  # inner loop trip count handled by CGRA (ignored when body_only)
    # ── shared-outer-loop grouping (Marionette only) ──────────────────────
    # Segments with the same group id share an outer CPU loop.  Their costs
    # are combined as: prod(group_outer_trips) × Σ cost_per_outer_iter(seg).
    group: int = -1                                         # -1 = independent
    group_outer_trips: list = field(default_factory=list)  # shared outer trips
    fast_switch: bool = True  # Whether the arch supports fast CPU↔CGRA context switch.
                              # body_only=True:  True  → t=5  (hardware-managed switch, e.g. Marionette)
                              #                  False → t=20 (standard CPU context switch)
                              # body_only=False: True  → modulo-scheduled pipeline
                              #                           latency = prod(outer)×[(inner−1)×II+steps+t]
                              #                  False → sequential invocations
                              #                           latency = prod(outer)×inner×(steps+t)


# ── helper ────────────────────────────────────────────────────────────────
def _SM(cpp, outer, inner):
    """Shorthand for a self-managed control scheme"""
    return SegConfig(cpp, body_only=False, cpu_trips=outer, cgra_trips=inner)

def _CL(cpp, all_trips, fast_switch=True):
    """Shorthand for a controller-based scheme

    The CPU owns every loop level; the CGRA is invoked once per innermost
    body.  All loop levels (outermost → innermost) are listed in all_trips.

        fast_switch=True  (default, e.g. Marionette) → t = 5
        fast_switch=False (standard CPU switch)       → t = 20

    cost = _marionette_cost(all_trips, steps, t)
    """
    return SegConfig(cpp, body_only=True, cpu_trips=list(all_trips),
                     fast_switch=fast_switch)

def _MG(cpp, inner_trips, group_id, outer_trips):
    """Marionette segment whose outer loop is shared with sibling segments.

    inner_trips : trip counts owned by this segment alone.
                  Use [1] for a single-call body (e.g. an outer init).
    group_id    : integer key linking all co-grouped segments (scope = one
                  benchmark × arch entry).
    outer_trips : shared enclosing loop trip counts (same for all siblings).

    Combined cost = prod(outer_trips) × Σ _marionette_cost(inner_trips, steps, t)
    """
    return SegConfig(cpp, body_only=True, cpu_trips=list(inner_trips),
                     group=group_id, group_outer_trips=list(outer_trips))


# BENCH_ARCH_SEGS[(bench_name, arch_name)] → list[SegConfig]
BENCH_ARCH_SEGS: dict[tuple, list] = {

    # ── conv ─────────────────────────────────────────────────────────────
    ("conv", "NEURA-ST"):   [_SM("conv.cpp",  [28416], 192)],
    ("conv", "NEURA-SO"):   [_SM("conv.cpp",  [28416], 192)],
    ("conv", "ICED"):       [_SM("conv.cpp",  [28416], 192)],
    ("conv", "RipTide"):    [_SM("conv.cpp",  [28416], 192)],
    ("conv", "Marionette"): [_CL("conv.cpp", [28416, 192], fast_switch=True)],

    # ── relu ──────────────────────────────────────────────────────────────
    #  NEURA-ST/SO/RipTide pipeline the loop; ICED cannot → sequential model.
    ("relu", "NEURA-ST"):   [_SM("relu.cpp", [], 512)],
    ("relu", "NEURA-SO"):   [_SM("relu.cpp", [], 512)],
    ("relu", "ICED"):       [_CL("relu.cpp", [512], fast_switch=False)],
    ("relu", "RipTide"):    [_CL("relu.cpp", [512], fast_switch=False)],
    ("relu", "Marionette"): [_CL("relu.cpp", [512], fast_switch=True)],

    # ── spmv ──────────────────────────────────────────────────────────────
    ("spmv", "NEURA-ST"):   [_SM("spmv.cpp", [494], 10)],
    ("spmv", "NEURA-SO"):   [_SM("spmv.cpp", [494], 10)],
    ("spmv", "ICED"):       [_SM("spmv.cpp", [494], 10)],
    ("spmv", "RipTide"):    [_SM("spmv.cpp", [494], 10)],
    ("spmv", "Marionette"): [_CL("spmv.cpp", [494, 10], fast_switch=True)],

    # ── gemm ──────────────────────────────────────────────────────────────
    ("gemm", "NEURA-ST"):   [_SM("gemm.cpp", [4096], 64)],
    ("gemm", "NEURA-SO"):   [_SM("gemm.cpp", [4096], 64)],
    ("gemm", "ICED"):       [_SM("gemm.cpp", [4096], 64)],
    ("gemm", "RipTide"):    [_SM("gemm.cpp", [4096], 64)],
    ("gemm", "Marionette"): [_CL("gemm.cpp", [4096, 64], fast_switch=True)],

    # ── bicg ──────────────────────────────────────────────────────────────
    ("bicg", "NEURA-ST"):   [_SM("bicg.cpp", [2100], 1900)],
    ("bicg", "ICED"):       [_SM("bicg.cpp", [2100], 1900)],
    ("bicg", "NEURA-SO"):   [_SM("bicg_kernel1.cpp", [2100], 1900),
                              _SM("bicg_kernel2.cpp", [2100], 1900)],
    ("bicg", "RipTide"):    [_SM("bicg_kernel1.cpp", [2100], 1900),
                              _SM("bicg_kernel2.cpp", [2100], 1900)],
    ("bicg", "Marionette"): [
        _MG("bicg_outer.cpp", [1],    group_id=0, outer_trips=[2100]),
        _MG("bicg.cpp",       [1900], group_id=0, outer_trips=[2100]),
    ],

    # ── mvt ───────────────────────────────────────────────────────────────
    ("mvt", "NEURA-ST"):   [_SM("mvt.cpp", [2000], 2000)],
    ("mvt", "NEURA-SO"):   [_SM("mvt.cpp", [2000], 2000)],
    ("mvt", "ICED"):       [_SM("mvt.cpp", [2000], 2000)],
    ("mvt", "RipTide"):    [_SM("mvt.cpp", [2000], 2000)],
    ("mvt", "Marionette"): [_CL("mvt.cpp", [2000, 2000], fast_switch=True)],

    # ── jacobi ────────────────────────────────────────────────────────────
    ("jacobi", "NEURA-ST"):   [_SM("jacobi_kernel1.cpp", [500], 2000),
                                _SM("jacobi_kernel2.cpp", [500], 2000)],
    ("jacobi", "NEURA-SO"):   [_SM("jacobi_kernel1.cpp", [500], 2000),
                                _SM("jacobi_kernel2.cpp", [500], 2000)],
    ("jacobi", "ICED"):       [_SM("jacobi_kernel1.cpp", [500], 2000),
                                _SM("jacobi_kernel2.cpp", [500], 2000)],
    ("jacobi", "RipTide"):    [_SM("jacobi_kernel1.cpp", [500], 2000),
                                _SM("jacobi_kernel2.cpp", [500], 2000)],
    ("jacobi", "Marionette"): [
        _CL("jacobi_kernel1.cpp", [500, 2000], fast_switch=True),
        _CL("jacobi_kernel2.cpp", [500, 2000], fast_switch=True),
    ],

    # ── fft ───────────────────────────────────────────────────────────────
    ("fft", "NEURA-ST"):   [_SM("fft.cpp", [], 128)],
    ("fft", "NEURA-SO"):   [_CL("fft.cpp", [128], fast_switch=False)],
    ("fft", "ICED"):       [_SM("fft.cpp", [], 128)],
    ("fft", "RipTide"):    [_CL("fft.cpp", [128], fast_switch=False)],
    ("fft", "Marionette"): [_CL("fft.cpp", [128], fast_switch=True)],

    # ── merge-sort ────────────────────────────────────────────────────────
    ("merge-sort", "NEURA-ST"):   [_SM("merge-sort.cpp", [], 1024)],
    ("merge-sort", "NEURA-SO"):   [_CL("merge-sort.cpp", [1024], fast_switch=False)],
    ("merge-sort", "ICED"):       [_CL("merge-sort.cpp", [1024], fast_switch=False)],
    ("merge-sort", "RipTide"):    [_CL("merge-sort.cpp", [1024], fast_switch=False)],
    ("merge-sort", "Marionette"): [_CL("merge-sort.cpp", [1024], fast_switch=True)],

    # ── bfs ───────────────────────────────────────────────────────────────
    ("bfs", "NEURA-ST"):   [_SM("bfs.cpp", [], 256)],
    ("bfs", "NEURA-SO"):   [_CL("bfs.cpp", [256], fast_switch=False)],
    ("bfs", "ICED"):       [_CL("bfs.cpp", [256], fast_switch=False)],
    ("bfs", "RipTide"):    [_CL("bfs.cpp", [256], fast_switch=False)],
    ("bfs", "Marionette"): [_CL("bfs.cpp", [256], fast_switch=True)],

    # ── floyd ─────────────────────────────────────────────────────────────
    ("floyd", "NEURA-ST"):   [_SM("floyd.cpp", [1000, 1000], 1000)],
    ("floyd", "NEURA-SO"):   [_CL("floyd.cpp", [1000, 1000, 1000], fast_switch=False)],
    ("floyd", "ICED"):       [_CL("floyd.cpp", [1000, 1000, 1000], fast_switch=False)],
    ("floyd", "RipTide"):    [_CL("floyd.cpp", [1000, 1000, 1000], fast_switch=False)],
    ("floyd", "Marionette"): [_CL("floyd.cpp", [1000, 1000, 1000], fast_switch=True)],
}

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – COMPILATION PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def _run(cmd: list, label: str = "") -> str:
    """Run a shell command, raise on failure, return stdout."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tag = f"[{label}] " if label else ""
        raise RuntimeError(
            f"{tag}Command failed:\n  {' '.join(cmd)}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def strip_module_attributes(mlir_text: str) -> str:
    """
    cgeist emits:  module attributes { ... } {
    Subsequent passes require plain:  module {
    This strips all module-level attributes in one regex pass.
    Handles the typical single-line attribute block produced by cgeist.
    """
    return re.sub(
        r'(?m)^(\s*module)\s+attributes\s*\{[^\n]*\}\s*(\{)',
        r'\1 \2',
        mlir_text,
    )


def compile_segment(
    seg: SegConfig,
    arch_cfg: ArchConfig,
    arch_dir: Path,
    work_dir: Path,
) -> Path:
    """
    Run the full compilation pipeline for one SegConfig and return the path
    to the resulting mapped MLIR file.

    Pipeline
    ────────
      1. cgeist  cpp → scf MLIR
      2. (strip module attributes)
      3. mlir-opt  scf → llvm dialect
      4. neura-opt  llvm → neura dialect
      5. neura-opt  neura → dataflow IR   (arch-specific passes)
      6. neura-opt  dataflow → mapped IR  (insert-data-mov + map-to-accelerator)
    """
    cpp_path  = arch_dir / seg.cpp_file
    base_name = cpp_path.stem          # e.g. "gemm", "bicg_kernel1"

    scf_raw   = work_dir / f"{base_name}_scf_raw.mlir"
    scf_clean = work_dir / f"{base_name}_scf.mlir"
    llvm_mlir = work_dir / f"{base_name}_llvm.mlir"
    neura     = work_dir / f"{base_name}_neura.mlir"
    dataflow  = work_dir / f"{base_name}_dataflow.mlir"
    mapped    = work_dir / f"{base_name}{arch_cfg.mapped_suffix}"

    # Step 1 – C++ → scf MLIR
    _run([CGEIST, str(cpp_path), "-S", "-O3", "-o", str(scf_raw)],
         f"cgeist:{base_name}")

    # Step 2 – strip module attributes
    scf_clean.write_text(strip_module_attributes(scf_raw.read_text()))

    # Step 3 – scf MLIR → llvm dialect
    _run([MLIR_OPT,
          "--lower-affine", "--convert-scf-to-cf", "--convert-cf-to-llvm",
          "--convert-arith-to-llvm", "--convert-index-to-llvm",
          "--reconcile-unrealized-casts",
          str(scf_clean), "-o", str(llvm_mlir)],
         f"mlir-opt:{base_name}")

    # Step 4 – llvm dialect → neura dialect
    _run([NEURA_OPT,
          "--assign-accelerator",
          "--lower-arith-to-neura", "--lower-memref-to-neura",
          "--lower-builtin-to-neura", "--lower-llvm-to-neura",
          str(llvm_mlir), "-o", str(neura)],
         f"neura-lower:{base_name}")

    # Step 5 – neura → dataflow IR  (arch-specific pass sequence)
    _run([NEURA_OPT, *arch_cfg.dataflow_passes, str(neura), "-o", str(dataflow)],
         f"dataflow:{base_name}")

    # Step 6 – dataflow → mapped IR
    mapping_flag = (
        f"--map-to-accelerator="
        f"mapping-strategy=heuristic "
        f"mapping-mode={arch_cfg.mapping_mode} "
        f"backtrack-config=customized "
        f"sort-strategy=mixed"
    )
    _run([NEURA_OPT, "--insert-data-mov", mapping_flag,
          str(dataflow), "-o", str(mapped)],
         f"map:{base_name}")

    return mapped


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – II AND STEPS EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def extract_ii_steps(mapped_mlir: Path) -> tuple[int, int]:
    """
    Parse the mapped MLIR file and return (compiled_ii, steps).

    compiled_ii  – read from  mapping_info = {compiled_ii = N : i32, ...}
    steps        – max(time_step) across all mapping_locs entries.
                   Represents the pipeline depth / circuit latency in cycles.
    """
    text = mapped_mlir.read_text()

    # compiled_ii
    m = re.search(r'compiled_ii\s*=\s*(\d+)\s*:', text)
    if m is None:
        raise ValueError(f"compiled_ii not found in {mapped_mlir}")
    ii = int(m.group(1))

    # steps = max time_step across all mapping_locs entries
    time_steps = [int(v) for v in re.findall(r'time_step\s*=\s*(\d+)', text)]
    if not time_steps:
        raise ValueError(f"No time_step values found in {mapped_mlir}")
    steps = max(time_steps)

    return ii, steps


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 6 – LATENCY MODEL
# ════════════════════════════════════════════════════════════════════════════

def _marionette_cost(trips: list, steps: int, t: int) -> float:
    """
    Recursive latency for Marionette-style execution.

    Marionette maps only the innermost loop body onto the CGRA; the CPU
    drives every loop level.  Each CGRA invocation costs (steps + t).
    Each CPU loop level wraps the inner cost and adds one extra transition
    for the loop-control overhead itself:

        cost([N, ...rest]) = N * (cost([...rest]) + t)
        cost([])           = steps          ← single body invocation

    Example  steps=10, t=5, trips=[28416, 192]:
        cost([192])        = 192 * (10 + 5)       = 2880
        cost([28416, 192]) = 28416 * (2880 + 5)   = 81,993,600
        which equals ((10+5)*192 + 5) * 28416  ✓
    """
    if not trips:
        return steps
    return trips[0] * (_marionette_cost(trips[1:], steps, t) + t)


def segment_latency(
    seg: SegConfig,
    ii: int,
    steps: int,
    cpu_transition: int = CPU_TRANSITION_CYCLES,
) -> float:
    """
    Compute cycles for one SegConfig given its compiled II and steps.

    Non-Marionette (body_only=False):
        CGRA pipelines cgra_trips inner iterations.
        cycles = prod(cpu_trips) × [(cgra_trips − 1) × II + steps + cpu_transition]

    Marionette (body_only=True):
        CGRA maps innermost body; CPU drives all loops recursively.
        cycles = _marionette_cost(cpu_trips, steps, MARIONETTE_TRANSITION_CYCLES)
    """
    if seg.body_only:
        t = MARIONETTE_TRANSITION_CYCLES if seg.fast_switch else cpu_transition
        return _marionette_cost(seg.cpu_trips, steps, t)
    else:
        outer = functools.reduce(lambda a, b: a * b, seg.cpu_trips, 1) if seg.cpu_trips else 1
        if seg.fast_switch:
            # Modulo-scheduled pipeline: fill cost amortised over cgra_trips iterations
            if outer == 1:
                return outer * ((seg.cgra_trips - 1) * ii)
            else:
                return outer * ((seg.cgra_trips - 1) * ii + cpu_transition)
        else:
            # Sequential: every iteration runs to completion before the next
            return outer * seg.cgra_trips * (steps + cpu_transition)


def bench_latency(
    segs_with_results: list[tuple[SegConfig, int, int]],
    cpu_transition: int = CPU_TRANSITION_CYCLES,
) -> float:
    """
    Compute total latency for one benchmark.

    Independent segments (group=-1) are evaluated via segment_latency() and summed.

    Grouped segments (group≥0) share an outer CPU loop; their per-outer-iteration
    costs are summed first and then multiplied by the shared outer trip counts:

        total += prod(group_outer_trips) × Σ _marionette_cost(seg.cpu_trips, steps, t)

    This avoids double-counting the outer-loop transition t that would occur if
    each segment were evaluated independently with the full trip list.
    """
    t = MARIONETTE_TRANSITION_CYCLES
    total = 0.0
    groups: dict[int, list] = {}

    for item in segs_with_results:
        seg = item[0]
        if seg.group >= 0:
            groups.setdefault(seg.group, []).append(item)
        else:
            seg2, ii2, steps2 = item
            total += segment_latency(seg2, ii2, steps2, cpu_transition)

    for group_items in groups.values():
        outer_trips = group_items[0][0].group_outer_trips
        prod_outer = functools.reduce(lambda a, b: a * b, outer_trips, 1)
        per_outer = sum(
            _marionette_cost(seg.cpu_trips, steps, t)
            for seg, ii, steps in group_items
        )
        total += prod_outer * per_outer

    return total


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 7 – MAIN ORCHESTRATION
# ════════════════════════════════════════════════════════════════════════════

BENCHMARKS = [
    "conv", "relu", "spmv", "gemm", "bicg", "mvt",
    "jacobi", "fft", "merge-sort", "bfs", "floyd",
]
# BENCHMARKS = ["bfs", "bicg", "conv", "fft", "floyd", "gemm", "jacobi", "merge-sort", "relu", "spmv", "mvt"]
ARCHS = ["Marionette", "ICED", "RipTide", "NEURA-SO", "NEURA-ST"]
# ARCHS = ["NEURA-ST"]


def run_benchmark(
    bench: str,
    arch_name: str,
    work_root: Path,
    verbose: bool = False,
) -> Optional[float]:
    """
    Compile every segment for (bench, arch_name), extract II+steps,
    compute and return total latency in cycles.
    Returns None if the config is not defined or compilation fails.
    """
    key = (bench, arch_name)
    if key not in BENCH_ARCH_SEGS:
        print(f"  [SKIP] no config for ({bench}, {arch_name})")
        return None

    arch_cfg  = ARCH_CONFIGS[arch_name]
    arch_dir  = PLDI_TEST_DIR / bench / arch_cfg.folder
    work_dir  = work_root / bench / arch_name
    work_dir.mkdir(parents=True, exist_ok=True)

    segs = BENCH_ARCH_SEGS[key]
    results = []

    for seg in segs:
        base = Path(seg.cpp_file).stem
        if verbose:
            print(f"  Compiling  {bench}/{arch_name}/{seg.cpp_file} ...")
        try:
            mapped = compile_segment(seg, arch_cfg, arch_dir, work_dir)
            ii, steps = extract_ii_steps(mapped)
            if verbose:
                print(f"    → II={ii}  steps={steps}")
            results.append((seg, ii, steps))
        except Exception as e:
            print(f"  [ERROR] {bench}/{arch_name}/{seg.cpp_file}: {e}")
            return None

    return bench_latency(results)


def main():
    with tempfile.TemporaryDirectory(prefix="neura_ae_") as tmp:
        work_root = Path(tmp)

        # ── compile everything and collect latencies ──────────────────────
        # latencies[arch_name][bench] = cycles  (or None on failure)
        latencies: dict[str, dict[str, Optional[float]]] = {
            arch: {} for arch in ARCHS
        }

        for bench in BENCHMARKS:
            print(f"\n{'─'*60}")
            print(f"Benchmark: {bench}")
            for arch in ARCHS:
                lat = run_benchmark(bench, arch, work_root, verbose=True)
                latencies[arch][bench] = lat
                status = f"{lat:.0f} cycles" if lat is not None else "FAILED"
                print(f"  {arch:<12s}  {status}")

        # ── normalise to Marionette = 1.0 ────────────────────────────────
        print(f"\n{'═'*60}")
        print("Normalised speedup (relative to Marionette)")
        print(f"{'Bench':<14s}", end="")
        for arch in ARCHS:
            print(f"  {arch:<12s}", end="")
        print()

        speedup_data: dict[str, list[Optional[float]]] = {
            arch: [] for arch in ARCHS
        }

        for bench in BENCHMARKS:
            base = latencies["Marionette"].get(bench)
            print(f"{bench:<14s}", end="")
            for arch in ARCHS:
                lat = latencies[arch].get(bench)
                if base and lat:
                    sp = base / lat
                else:
                    sp = None
                speedup_data[arch].append(sp)
                tag = f"{sp:.3f}" if sp is not None else "N/A"
                print(f"  {tag:<12s}", end="")
            print()

        # ── geometric mean speedup ────────────────────────────────────────
        print(f"{'Geomean':<14s}", end="")
        geomeans: dict[str, Optional[float]] = {}
        for arch in ARCHS:
            vals = [v for v in speedup_data[arch] if v is not None]
            gmean = functools.reduce(lambda a, b: a * b, vals, 1) ** (1 / len(vals)) if vals else None
            geomeans[arch] = gmean
            tag = f"{gmean:.3f}" if gmean is not None else "N/A"
            print(f"  {tag:<12s}", end="")
        print()

        # ── write raw latency CSV ─────────────────────────────────────
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGS_DIR.mkdir(parents=True, exist_ok=True)
        latency_csv = RESULTS_DIR / "latency_cycles.csv"
        with latency_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["benchmark"] + ARCHS)
            for bench in BENCHMARKS:
                row = [bench] + [
                    f"{latencies[arch].get(bench):.0f}"
                    if latencies[arch].get(bench) is not None else "N/A"
                    for arch in ARCHS
                ]
                writer.writerow(row)
        print(f"\nLatency data written to {latency_csv}")

        # ── write speedup CSV ─────────────────────────────────────────────
        speedup_csv = RESULTS_DIR / "speedup.csv"
        with speedup_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["benchmark"] + ARCHS)
            for i, bench in enumerate(BENCHMARKS):
                row = [bench] + [
                    f"{speedup_data[arch][i]:.4f}"
                    if speedup_data[arch][i] is not None else "N/A"
                    for arch in ARCHS
                ]
                writer.writerow(row)
            gmean_row = ["Geomean"] + [
                f"{geomeans[arch]:.4f}" if geomeans[arch] is not None else "N/A"
                for arch in ARCHS
            ]
            writer.writerow(gmean_row)
        print(f"Speedup data written to {speedup_csv}")

        # ── generate speedup figure ───────────────────────────────────────
        generate_speedup_figure(
            speedup_data, geomeans, BENCHMARKS, ARCHS,
            save_path=FIGS_DIR / "fig13.pdf",
        )

        # ── compute normalised Perf/Area ──────────────────────────────────
        # PPA_normalised_arch = (lat_mario / area_mario) / (lat_arch / area_arch)
        #                     = speedup * (area_mario / area_arch)
        area_mario = ARCH_AREA_MM2["Marionette"]
        ppa_data: dict[str, list[Optional[float]]] = {arch: [] for arch in ARCHS}
        for i, bench in enumerate(BENCHMARKS):
            lat_mario = latencies["Marionette"].get(bench)
            for arch in ARCHS:
                lat = latencies[arch].get(bench)
                sp  = speedup_data[arch][i]
                if sp is not None and arch in ARCH_AREA_MM2:
                    ppa = sp * (area_mario / ARCH_AREA_MM2[arch])
                else:
                    ppa = None
                ppa_data[arch].append(ppa)

        geomeans_ppa: dict[str, Optional[float]] = {}
        print(f"\n{'═'*60}")
        print("Normalised Perf/Area (relative to Marionette)")
        print(f"{'Bench':<14s}", end="")
        for arch in ARCHS:
            print(f"  {arch:<12s}", end="")
        print()
        for i, bench in enumerate(BENCHMARKS):
            print(f"{bench:<14s}", end="")
            for arch in ARCHS:
                v = ppa_data[arch][i]
                print(f"  {f'{v:.3f}' if v is not None else 'N/A':<12s}", end="")
            print()
        print(f"{'Geomean':<14s}", end="")
        for arch in ARCHS:
            vals = [v for v in ppa_data[arch] if v is not None]
            gm = functools.reduce(lambda a, b: a * b, vals, 1) ** (1 / len(vals)) if vals else None
            geomeans_ppa[arch] = gm
            print(f"  {f'{gm:.3f}' if gm is not None else 'N/A':<12s}", end="")
        print()

        # ── write Perf/Area CSV ───────────────────────────────────────────
        ppa_csv = RESULTS_DIR / "perf_per_area.csv"
        with ppa_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["benchmark"] + ARCHS)
            for i, bench in enumerate(BENCHMARKS):
                row = [bench] + [
                    f"{ppa_data[arch][i]:.4f}" if ppa_data[arch][i] is not None else "N/A"
                    for arch in ARCHS
                ]
                writer.writerow(row)
            writer.writerow(["Geomean"] + [
                f"{geomeans_ppa[arch]:.4f}" if geomeans_ppa[arch] is not None else "N/A"
                for arch in ARCHS
            ])
        print(f"Perf/Area data written to {ppa_csv}")

        # ── generate Perf/Area figure ─────────────────────────────────────
        generate_ppa_figure(
            ppa_data, geomeans_ppa, BENCHMARKS, ARCHS,
            save_path=FIGS_DIR / "fig15.pdf",
        )

        # ── compute IPC ───────────────────────────────────────────────────
        # IPC = total_instructions / total_cycles
        ipc_data: dict[str, list[Optional[float]]] = {arch: [] for arch in ARCHS}
        for bench in BENCHMARKS:
            for arch in ARCHS:
                n_instr = BENCH_ARCH_INSTRUCTIONS.get((bench, arch))
                cycles  = latencies[arch].get(bench)
                if n_instr is not None and cycles:
                    ipc_data[arch].append(n_instr / cycles)
                else:
                    ipc_data[arch].append(None)

        geomeans_ipc: dict[str, Optional[float]] = {}
        print(f"\n{'═'*60}")
        print("IPC (instructions per cycle)")
        print(f"{'Bench':<14s}", end="")
        for arch in ARCHS:
            print(f"  {arch:<12s}", end="")
        print()
        for i, bench in enumerate(BENCHMARKS):
            print(f"{bench:<14s}", end="")
            for arch in ARCHS:
                v = ipc_data[arch][i]
                print(f"  {f'{v:.3f}' if v is not None else 'N/A':<12s}", end="")
            print()
        print(f"{'Geomean':<14s}", end="")
        for arch in ARCHS:
            vals = [v for v in ipc_data[arch] if v is not None]
            gm = functools.reduce(lambda a, b: a * b, vals, 1) ** (1 / len(vals)) if vals else None
            geomeans_ipc[arch] = gm
            print(f"  {f'{gm:.3f}' if gm is not None else 'N/A':<12s}", end="")
        print()

        # ── write IPC CSV ─────────────────────────────────────────────────
        ipc_csv = RESULTS_DIR / "ipc.csv"
        with ipc_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["benchmark"] + ARCHS)
            for i, bench in enumerate(BENCHMARKS):
                row = [bench] + [
                    f"{ipc_data[arch][i]:.4f}" if ipc_data[arch][i] is not None else "N/A"
                    for arch in ARCHS
                ]
                writer.writerow(row)
            writer.writerow(["Geomean"] + [
                f"{geomeans_ipc[arch]:.4f}" if geomeans_ipc[arch] is not None else "N/A"
                for arch in ARCHS
            ])
        print(f"IPC data written to {ipc_csv}")

        # ── generate IPC figure ───────────────────────────────────────────
        generate_ipc_figure(
            ipc_data, geomeans_ipc, BENCHMARKS, ARCHS,
            save_path=FIGS_DIR / "fig14.pdf",
        )


# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()