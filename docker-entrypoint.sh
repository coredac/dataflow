#!/usr/bin/env bash
set -e

show_help() {
    echo ""
    echo "Neura PLDI 2026 — Artifact Evaluation Docker"
    echo "=============================================="
    echo ""
    echo "Usage:"
    echo "  docker run --rm -it -v \$(pwd)/output:/workspace/dataflow/evaluation/output <image>"
    echo "  docker run --rm -it -v \$(pwd)/output:/workspace/dataflow/evaluation/output <image> reproduce"
    echo "  docker run --rm -it -v \$(pwd)/output:/workspace/dataflow/evaluation/output <image> build"
    echo "  docker run --rm -it <image> bash"
    echo ""
    echo "Commands:"
    echo "  reproduce  Run the full evaluation pipeline (default, ~50 min)"
    echo "             Generates fig13-fig18 and CSV data."
    echo "  quick      Run lit tests to verify compiler tools (~30 sec)"
    echo "             For kick-the-tires evaluation."
    echo "  build      Rebuild the dataflow project from source against LLVM/MLIR."
    echo "  bash       Drop into an interactive shell."
    echo "  help       Show this help message."
    echo ""
}

case "${1:-reproduce}" in
    reproduce)
        echo "════════════════════════════════════════════════════════════"
        echo " Neura PLDI 2026 — Reproducing Experiments"
        echo "════════════════════════════════════════════════════════════"
        cd /workspace/dataflow
        python3 evaluation/main.py
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo " Done! Results:"
        echo "   CSV files:  evaluation/results/"
        echo "   Figures:    evaluation/figs/"
        echo ""
        echo " If you mounted -v \$(pwd)/output:/workspace/dataflow/evaluation/output"
        echo " the results are also copied to your host."
        echo "════════════════════════════════════════════════════════════"
        # Copy to output mount point if it exists
        if [ -d /workspace/dataflow/evaluation/output ]; then
            cp -r /workspace/dataflow/evaluation/results /workspace/dataflow/evaluation/output/ 2>/dev/null || true
            cp -r /workspace/dataflow/evaluation/figs    /workspace/dataflow/evaluation/output/ 2>/dev/null || true
        fi
        ;;
    quick)
        echo "════════════════════════════════════════════════════════════"
        echo " Neura PLDI 2026 — Kick-the-Tires (lit tests, ~30 sec)"
        echo "════════════════════════════════════════════════════════════"
        cd /workspace/dataflow
        echo "Running lit tests..."
        python3 /workspace/llvm-project/build/bin/llvm-lit test/ -v
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo " All tests passed! The compiler tools are working correctly."
        echo "════════════════════════════════════════════════════════════"
        ;;
    build)
        echo "════════════════════════════════════════════════════════════"
        echo " Neura PLDI 2026 — Rebuilding from Source"
        echo "════════════════════════════════════════════════════════════"
        LLVM_BUILD=/workspace/llvm-project/build
        cd /workspace/dataflow
        # Back up pre-built binaries so they survive cmake reconfiguration
        if [ -f build/tools/mlir-neura-opt/mlir-neura-opt ] && [ ! -f build/.prebuilt_backup ]; then
            cp build/tools/mlir-neura-opt/mlir-neura-opt     /tmp/mlir-neura-opt.bak
            cp build/tools/mlir-neura-opt/mlir-neura-opt-4x4 /tmp/mlir-neura-opt-4x4.bak
            touch build/.prebuilt_backup
        fi
        mkdir -p build && cd build
        cmake .. \
            -DMLIR_DIR="${LLVM_BUILD}/lib/cmake/mlir" \
            -DLLVM_DIR="${LLVM_BUILD}/lib/cmake/llvm" \
            -DLLVM_TOOLS_BINARY_DIR="${LLVM_BUILD}/bin" \
            -DMLIR_BINARY_DIR="${LLVM_BUILD}"
        make -j"$(nproc)"
        echo ""
        echo "Build complete. Binaries:"
        echo "  $(pwd)/tools/mlir-neura-opt/mlir-neura-opt"
        echo ""
        echo "You can now run:  python3 /workspace/dataflow/evaluation/main.py"
        echo "════════════════════════════════════════════════════════════"
        ;;
    bash|sh)
        echo "Dropping into interactive shell..."
        echo "  To reproduce:  python3 evaluation/main.py"
        echo "  To rebuild:    cd build && cmake .. -DMLIR_DIR=... && make -j\$(nproc)"
        exec /bin/bash
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        # Pass through any other command
        exec "$@"
        ;;
esac
