# ─────────────────────────────────────────────────────────────────────────
# Dockerfile — Neura PLDI 2026 Artifact Evaluation
#
# Two modes:
#   1. Reproduce experiments:  python3 evaluation/main.py
#   2. Rebuild from source:    cd build && cmake .. <flags> && make -j$(nproc)
#
# Build:
#   docker build -t neura-pldi26:latest .
#
# Run (interactive):
#   docker run --rm -it -v $(pwd)/output:/workspace/dataflow/evaluation/output neura-pldi26:latest
# ─────────────────────────────────────────────────────────────────────────

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages (build tools + Python) ───────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        python3 \
        python3-pip \
        zlib1g-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir matplotlib numpy

# ── Directory layout ─────────────────────────────────────────────────────
#   /workspace/llvm-project/         Pre-built LLVM/MLIR (for rebuilding)
#   /workspace/dataflow/             Project source + pre-built binaries
RUN mkdir -p /workspace/llvm-project /workspace/dataflow

# ── LLVM/MLIR: pre-built artifacts needed for cmake find_package ─────────
#   lib/    → static libraries (.a) + cmake config files
#   include/→ generated headers (build-time)
#   bin/    → llvm-tblgen, mlir-tblgen, mlir-opt, FileCheck
#
#   mlir/include/ → MLIR source headers
#   llvm/include/ → LLVM source headers
COPY llvm-project/build/lib/         /workspace/llvm-project/build/lib/
COPY llvm-project/build/include/     /workspace/llvm-project/build/include/
# Only copy essential binaries (not the entire 3.8 GB bin/)
COPY llvm-project/build/bin/llvm-tblgen     /workspace/llvm-project/build/bin/llvm-tblgen
COPY llvm-project/build/bin/mlir-tblgen     /workspace/llvm-project/build/bin/mlir-tblgen
COPY llvm-project/build/bin/mlir-opt        /workspace/llvm-project/build/bin/mlir-opt
COPY llvm-project/build/bin/FileCheck       /workspace/llvm-project/build/bin/FileCheck
COPY llvm-project/build/bin/llvm-lit        /workspace/llvm-project/build/bin/llvm-lit
COPY llvm-project/build/bin/mlir-translate  /workspace/llvm-project/build/bin/mlir-translate
COPY llvm-project/build/bin/llc             /workspace/llvm-project/build/bin/llc
COPY llvm-project/build/bin/clang-20        /workspace/llvm-project/build/bin/clang-20
RUN ln -sf clang-20 /workspace/llvm-project/build/bin/clang && \
    ln -sf clang-20 /workspace/llvm-project/build/bin/clang++
# Generated MLIR headers (e.g. dialect .inc files from tablegen)
COPY llvm-project/build/tools/mlir/include/ /workspace/llvm-project/build/tools/mlir/include/
# Source headers (needed by cmake include_directories)
COPY llvm-project/mlir/include/      /workspace/llvm-project/mlir/include/
COPY llvm-project/llvm/include/      /workspace/llvm-project/llvm/include/
# LLVM cmake modules (TableGen.cmake, AddLLVM.cmake, etc.)
COPY llvm-project/llvm/cmake/        /workspace/llvm-project/llvm/cmake/
# lit Python library (used by llvm-lit)
COPY llvm-project/llvm/utils/        /workspace/llvm-project/llvm/utils/

# ── Fix hardcoded build-host paths in cmake config files ─────────────────
RUN find /workspace/llvm-project/build/lib/cmake -name '*.cmake' \
        -exec sed -i 's|/home/lucas/llvm-project|/workspace/llvm-project|g' {} +

# ── Dataflow project source ──────────────────────────────────────────────
COPY dataflow/CMakeLists.txt         /workspace/dataflow/CMakeLists.txt
COPY dataflow/include/               /workspace/dataflow/include/
COPY dataflow/lib/                   /workspace/dataflow/lib/
COPY dataflow/tools/                 /workspace/dataflow/tools/
COPY dataflow/test/                  /workspace/dataflow/test/

# ── Pre-built binaries (for direct experiment reproduction) ──────────────
COPY dataflow/build/tools/mlir-neura-opt/mlir-neura-opt       /workspace/dataflow/build/tools/mlir-neura-opt/mlir-neura-opt
COPY dataflow/build/tools/mlir-neura-opt/mlir-neura-opt-4x4   /workspace/dataflow/build/tools/mlir-neura-opt/mlir-neura-opt-4x4
COPY dataflow/build/tools/neura-interpreter/neura-interpreter  /workspace/dataflow/build/tools/neura-interpreter/neura-interpreter
COPY dataflow/build/tools/neura-compiler/neura-compiler        /workspace/dataflow/build/tools/neura-compiler/neura-compiler
COPY dataflow/thirdparty/polygeist/cgeist                     /workspace/dataflow/thirdparty/polygeist/cgeist
RUN chmod +x /workspace/dataflow/build/tools/mlir-neura-opt/mlir-neura-opt \
             /workspace/dataflow/build/tools/mlir-neura-opt/mlir-neura-opt-4x4 \
             /workspace/dataflow/build/tools/neura-interpreter/neura-interpreter \
             /workspace/dataflow/build/tools/neura-compiler/neura-compiler \
             /workspace/dataflow/thirdparty/polygeist/cgeist

# ── Evaluation pipeline + benchmarks ─────────────────────────────────────
COPY dataflow/evaluation/main.py     /workspace/dataflow/evaluation/main.py
COPY dataflow/evaluation/util/       /workspace/dataflow/evaluation/util/
COPY dataflow/evaluation/benchmarks/ /workspace/dataflow/evaluation/benchmarks/
COPY dataflow/evaluation/arch_reports/ /workspace/dataflow/evaluation/arch_reports/

# ── Put mlir-opt on PATH so "mlir-opt" works without full path ───────────
ENV PATH="/workspace/llvm-project/build/bin:${PATH}"

# ── Run cmake configure to generate lit.cfg with Docker-correct paths ────
RUN cd /workspace/dataflow && mkdir -p build && cd build && \
    cmake .. \
        -DMLIR_DIR=/workspace/llvm-project/build/lib/cmake/mlir \
        -DLLVM_DIR=/workspace/llvm-project/build/lib/cmake/llvm \
        -DLLVM_TOOLS_BINARY_DIR=/workspace/llvm-project/build/bin \
        -DMLIR_BINARY_DIR=/workspace/llvm-project/build

# ── Entrypoint ───────────────────────────────────────────────────────────
COPY dataflow/docker-entrypoint.sh   /workspace/dataflow/docker-entrypoint.sh
RUN chmod +x /workspace/dataflow/docker-entrypoint.sh

WORKDIR /workspace/dataflow
ENTRYPOINT ["/workspace/dataflow/docker-entrypoint.sh"]
CMD ["reproduce"]
