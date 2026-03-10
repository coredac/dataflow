"""
Compilation pipeline for Neura PLDI 2026 artifact evaluation.

Handles: C++ -> scf MLIR -> LLVM dialect -> Neura dialect -> dataflow -> mapped IR.
"""

from __future__ import annotations
import re
import subprocess
from pathlib import Path

from .config import (
    CGEIST, MLIR_OPT, NEURA_OPT,
    ArchConfig, SegConfig, OptVariant,
)


def _run(cmd: list, label: str = "") -> str:
    """Run a shell command, raise on failure, return stdout."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tag = f"[{label}] " if label else ""
        raise RuntimeError(
            f"{tag}Command failed:\n  {' '.join(cmd)}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def _strip_module_attributes(mlir_text: str) -> str:
    """Strip cgeist module-level attributes so downstream passes accept the IR."""
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
    neura_opt_binary: str = NEURA_OPT,
) -> Path:
    """
    Full compilation pipeline for one SegConfig.  Returns path to mapped MLIR.

    Steps: cgeist -> strip attrs -> mlir-opt (llvm) -> neura-opt (lower)
           -> neura-opt (dataflow) -> neura-opt (map)
    """
    cpp_path  = arch_dir / seg.cpp_file
    base_name = cpp_path.stem

    scf_raw   = work_dir / f"{base_name}_scf_raw.mlir"
    scf_clean = work_dir / f"{base_name}_scf.mlir"
    llvm_mlir = work_dir / f"{base_name}_llvm.mlir"
    neura     = work_dir / f"{base_name}_neura.mlir"
    dataflow  = work_dir / f"{base_name}_dataflow.mlir"
    mapped    = work_dir / f"{base_name}{arch_cfg.mapped_suffix}"

    # C++ -> scf MLIR
    _run([CGEIST, str(cpp_path), "-S", "-O3", "-o", str(scf_raw)],
         f"cgeist:{base_name}")

    # Strip module attributes
    scf_clean.write_text(_strip_module_attributes(scf_raw.read_text()))

    # scf -> LLVM dialect
    _run([MLIR_OPT,
          "--lower-affine", "--convert-scf-to-cf", "--convert-cf-to-llvm",
          "--convert-arith-to-llvm", "--convert-index-to-llvm",
          "--reconcile-unrealized-casts",
          str(scf_clean), "-o", str(llvm_mlir)],
         f"mlir-opt:{base_name}")

    # LLVM -> Neura dialect
    _run([neura_opt_binary,
          "--assign-accelerator",
          "--lower-arith-to-neura", "--lower-memref-to-neura",
          "--lower-builtin-to-neura", "--lower-llvm-to-neura",
          str(llvm_mlir), "-o", str(neura)],
         f"neura-lower:{base_name}")

    # Neura -> dataflow IR (arch-specific passes)
    _run([neura_opt_binary, *arch_cfg.dataflow_passes,
          str(neura), "-o", str(dataflow)],
         f"dataflow:{base_name}")

    # Dataflow -> mapped IR
    mapping_flag = (
        f"--map-to-accelerator="
        f"mapping-strategy=heuristic "
        f"mapping-mode={arch_cfg.mapping_mode} "
        f"backtrack-config=customized "
        f"sort-strategy=mixed"
    )
    _run([neura_opt_binary, "--insert-data-mov", mapping_flag,
          str(dataflow), "-o", str(mapped)],
         f"map:{base_name}")

    return mapped


def compile_opt_segment(
    seg: SegConfig,
    arch_cfg: ArchConfig,
    arch_dir: Path,
    work_dir: Path,
    variant: OptVariant,
) -> Path:
    """
    Compile one segment for the optimization-comparison experiment.
    Same as compile_segment but with variant-specific passes and optional
    --finalize-memref-to-llvm / --fuse-loop-control steps.
    """
    cpp_path  = arch_dir / seg.cpp_file
    base_name = cpp_path.stem

    scf_raw    = work_dir / f"{base_name}_scf_raw.mlir"
    scf_clean  = work_dir / f"{base_name}_scf.mlir"
    llvm_mlir  = work_dir / f"{base_name}_llvm.mlir"
    final_llvm = work_dir / f"{base_name}_final_llvm.mlir"
    neura      = work_dir / f"{base_name}_neura.mlir"
    dataflow   = work_dir / f"{base_name}_dataflow.mlir"
    stream     = work_dir / f"{base_name}_stream.mlir"
    mapped     = work_dir / f"{base_name}{arch_cfg.mapped_suffix}"

    label = f"opt/{variant.name}/{base_name}"

    # C++ -> scf MLIR
    _run([CGEIST, str(cpp_path), "-S", "-O3", "-o", str(scf_raw)],
         f"cgeist:{label}")
    scf_clean.write_text(_strip_module_attributes(scf_raw.read_text()))

    # scf -> LLVM dialect
    _run([MLIR_OPT,
          "--lower-affine", "--convert-scf-to-cf", "--convert-cf-to-llvm",
          "--convert-arith-to-llvm", "--convert-index-to-llvm",
          "--reconcile-unrealized-casts",
          str(scf_clean), "-o", str(llvm_mlir)],
         f"mlir-opt:{label}")

    # Optional: finalize memref (baseline variant only)
    if variant.use_finalize_memref:
        _run([MLIR_OPT, "--finalize-memref-to-llvm",
              str(llvm_mlir), "-o", str(final_llvm)],
             f"finalize-memref:{label}")
        neura_input = final_llvm
    else:
        neura_input = llvm_mlir

    # LLVM -> Neura dialect
    _run([NEURA_OPT,
          "--assign-accelerator",
          "--lower-arith-to-neura", "--lower-memref-to-neura",
          "--lower-builtin-to-neura", "--lower-llvm-to-neura",
          str(neura_input), "-o", str(neura)],
         f"neura-lower:{label}")

    # Neura -> dataflow IR (variant-specific passes)
    _run([NEURA_OPT, *variant.dataflow_passes,
          str(neura), "-o", str(dataflow)],
         f"dataflow:{label}")

    # Optional: loop streaming pass
    if variant.extra_pre_map_passes:
        _run([NEURA_OPT, *variant.extra_pre_map_passes,
              str(dataflow), "-o", str(stream)],
             f"stream:{label}")
        pre_map = stream
    else:
        pre_map = dataflow

    # Map to accelerator
    mapping_flag = (
        f"--map-to-accelerator="
        f"mapping-strategy=heuristic "
        f"mapping-mode={arch_cfg.mapping_mode} "
        f"backtrack-config=customized "
        f"sort-strategy=mixed"
    )
    _run([NEURA_OPT, "--insert-data-mov", mapping_flag,
          str(pre_map), "-o", str(mapped)],
         f"map:{label}")

    return mapped


def extract_ii_steps(mapped_mlir: Path) -> tuple[int, int]:
    """
    Parse mapped MLIR and return (compiled_ii, steps).
    steps = max(time_step) = pipeline depth in cycles.
    """
    text = mapped_mlir.read_text()

    m = re.search(r'compiled_ii\s*=\s*(\d+)\s*:', text)
    if m is None:
        raise ValueError(f"compiled_ii not found in {mapped_mlir}")
    ii = int(m.group(1))

    time_steps = [int(v) for v in re.findall(r'time_step\s*=\s*(\d+)', text)]
    if not time_steps:
        raise ValueError(f"No time_step values found in {mapped_mlir}")

    return ii, max(time_steps)
