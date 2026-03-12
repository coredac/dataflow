#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# build_docker.sh — Stage files and build the Neura PLDI 2026 Docker image
#
# Usage:
#   ./build_docker.sh                           # defaults
#   LLVM_BUILD=/path/to/llvm-project ./build_docker.sh
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLVM_PROJECT="${LLVM_PROJECT:-/home/lucas/llvm-project}"
DATAFLOW_DIR="${DATAFLOW_DIR:-$SCRIPT_DIR}"
IMAGE_NAME="${IMAGE_NAME:-neura-pldi26:latest}"

STAGING_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGING_DIR"' EXIT

echo "══════════════════════════════════════════════════════════"
echo " Neura PLDI 2026 — Building Docker Image"
echo "══════════════════════════════════════════════════════════"
echo "  LLVM source:   $LLVM_PROJECT"
echo "  Dataflow:      $DATAFLOW_DIR"
echo "  Staging dir:   $STAGING_DIR"
echo "  Image name:    $IMAGE_NAME"
echo ""

# ── Stage LLVM artifacts ─────────────────────────────────────────────────
echo "[1/4] Staging LLVM/MLIR build artifacts..."

LLVM_STAGE="$STAGING_DIR/llvm-project"
mkdir -p "$LLVM_STAGE/build/bin"

# Static libraries + cmake configs (needed for rebuild)
cp -a "$LLVM_PROJECT/build/lib"     "$LLVM_STAGE/build/lib"
# Generated headers
cp -a "$LLVM_PROJECT/build/include" "$LLVM_STAGE/build/include"
# Essential binaries only (not the full 3.8 GB bin/)
for bin in llvm-tblgen mlir-tblgen mlir-opt FileCheck llvm-lit mlir-translate llc clang-20; do
    cp "$LLVM_PROJECT/build/bin/$bin" "$LLVM_STAGE/build/bin/$bin"
done
# clang is a symlink to clang-20
ln -sf clang-20 "$LLVM_STAGE/build/bin/clang"
ln -sf clang-20 "$LLVM_STAGE/build/bin/clang++"
# Source headers (needed for cmake include_directories)
mkdir -p "$LLVM_STAGE/mlir" "$LLVM_STAGE/llvm"
cp -a "$LLVM_PROJECT/mlir/include" "$LLVM_STAGE/mlir/include"
cp -a "$LLVM_PROJECT/llvm/include" "$LLVM_STAGE/llvm/include"
# LLVM cmake modules (TableGen.cmake, AddLLVM.cmake, etc.)
cp -a "$LLVM_PROJECT/llvm/cmake"   "$LLVM_STAGE/llvm/cmake"
# lit Python library (used by llvm-lit)
cp -a "$LLVM_PROJECT/llvm/utils"   "$LLVM_STAGE/llvm/utils"
# Generated MLIR headers (dialect .inc files from tablegen)
mkdir -p "$LLVM_STAGE/build/tools/mlir"
cp -a "$LLVM_PROJECT/build/tools/mlir/include" "$LLVM_STAGE/build/tools/mlir/include"

echo "   Done. $(du -sh "$LLVM_STAGE" | cut -f1) staged."

# ── Stage dataflow project ───────────────────────────────────────────────
echo "[2/4] Staging dataflow project..."

DF_STAGE="$STAGING_DIR/dataflow"
mkdir -p "$DF_STAGE"

# Source code for rebuild
cp    "$DATAFLOW_DIR/CMakeLists.txt"     "$DF_STAGE/"
cp -a "$DATAFLOW_DIR/include"            "$DF_STAGE/include"
cp -a "$DATAFLOW_DIR/lib"                "$DF_STAGE/lib"
cp -a "$DATAFLOW_DIR/tools"              "$DF_STAGE/tools"
cp -a "$DATAFLOW_DIR/test"               "$DF_STAGE/test"

# Pre-built binaries
mkdir -p "$DF_STAGE/build/tools/mlir-neura-opt"
cp "$DATAFLOW_DIR/build/tools/mlir-neura-opt/mlir-neura-opt"     "$DF_STAGE/build/tools/mlir-neura-opt/"
cp "$DATAFLOW_DIR/build/tools/mlir-neura-opt/mlir-neura-opt-4x4" "$DF_STAGE/build/tools/mlir-neura-opt/"
mkdir -p "$DF_STAGE/build/tools/neura-interpreter"
cp "$DATAFLOW_DIR/build/tools/neura-interpreter/neura-interpreter" "$DF_STAGE/build/tools/neura-interpreter/"
mkdir -p "$DF_STAGE/build/tools/neura-compiler"
cp "$DATAFLOW_DIR/build/tools/neura-compiler/neura-compiler"       "$DF_STAGE/build/tools/neura-compiler/"
mkdir -p "$DF_STAGE/thirdparty/polygeist"
cp "$DATAFLOW_DIR/thirdparty/polygeist/cgeist"                    "$DF_STAGE/thirdparty/polygeist/"

# Evaluation pipeline
mkdir -p "$DF_STAGE/evaluation"
cp    "$DATAFLOW_DIR/evaluation/main.py"       "$DF_STAGE/evaluation/"
cp -a "$DATAFLOW_DIR/evaluation/util"          "$DF_STAGE/evaluation/util"
cp -a "$DATAFLOW_DIR/evaluation/benchmarks"    "$DF_STAGE/evaluation/benchmarks"
cp -a "$DATAFLOW_DIR/evaluation/arch_reports"   "$DF_STAGE/evaluation/arch_reports"

# Entrypoint
cp "$DATAFLOW_DIR/docker-entrypoint.sh" "$DF_STAGE/"

echo "   Done. $(du -sh "$DF_STAGE" | cut -f1) staged."

# ── Copy Dockerfile into staging root ────────────────────────────────────
echo "[3/4] Preparing Docker build context..."
cp "$DATAFLOW_DIR/Dockerfile" "$STAGING_DIR/Dockerfile"

echo "   Total context: $(du -sh "$STAGING_DIR" | cut -f1)"

# ── Build Docker image ───────────────────────────────────────────────────
echo "[4/4] Building Docker image: $IMAGE_NAME ..."
echo ""
docker build -t "$IMAGE_NAME" "$STAGING_DIR"

echo ""
echo "══════════════════════════════════════════════════════════"
echo " Image built: $IMAGE_NAME"
echo ""
echo " Quick start:"
echo "   docker run --rm -it $IMAGE_NAME             # reproduce"
echo "   docker run --rm -it $IMAGE_NAME build        # rebuild from source"
echo "   docker run --rm -it $IMAGE_NAME bash          # interactive shell"
echo "══════════════════════════════════════════════════════════"
