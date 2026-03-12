# Neura — PLDI 2026 Artifact

This is the artifact for *"NEURA: A Unified and Retargetable Compilation Framework for Coarse-Grained Reconfigurable Architectures"* (PLDI 2026).

The artifact is packaged as a self-contained Docker image (~4 GB). No
network access or external dependencies are required.

**Hardware used for paper results:** Intel Core i9-12900 (16-core, 24-thread),
32 GB RAM, Ubuntu 22.04. The artifact is CPU-only and runs on any x86-64
Linux/macOS/WSL2 machine.

**Third-party dependencies bundled inside the docker image:**
- [LLVM/MLIR](https://llvm.org/) v20.1.7 (pre-built, Apache-2.0)
- [Polygeist/cgeist](https://github.com/llvm/Polygeist) C/C++ → MLIR frontend (pre-built binary)

**System Requirements:**
- Docker 20.10+ (Linux, macOS, or Windows with WSL2)
- ~10 GB disk space for the image
- ~32 GB RAM
---

## Part A — Getting Started Guide

*Expected time: < 5 minutes.*

### A.1 Pull the docker image

```bash
docker pull shangkunli/neura-pldi26:pldi2026
```

### A.2 Kick-the-tires: run compiler test suite

```bash
docker run --rm shangkunli/neura-pldi26:pldi2026 quick
```

This runs 64 `llvm-lit` tests that exercise the core compiler passes
(dialect lowering, dataflow conversion, code generation, interpretation).
All 64 tests should pass in ~30 seconds. Sample output:

```
PASS: Neura Dialect Tests :: affine2neura/affine_to_neura.mlir (1 of 64)
...
PASS: Neura Dialect Tests :: controflow_fuse/perfect_nested/perfect_nested.mlir (64 of 64)

Testing Time: 22.59s
Total Discovered Tests: 64
  Passed: 64 (100.00%)
```

If all 64 tests pass, the compiler tools are working correctly and you can
proceed to Part B.

---

## Part B — Step-by-Step Instructions

### B.1 Reproduce all experiments (one command)

```bash
mkdir -p output
docker run --rm -v $(pwd)/output:/workspace/dataflow/evaluation/output \
    shangkunli/neura-pldi26:pldi2026 reproduce
```

This runs the full evaluation pipeline. **Expected runtime: ~50 minutes.**
Progress is printed to stdout as each benchmark × architecture is compiled.

Upon completion, figures and CSV data are written to `output/` on the host:

| Output file | Paper reference | Description |
|---|---|---|
| `figs/fig13.pdf` | Figure 13 | Speedup comparison across 5 architectures (11 regular + irregular benchmarks), normalised to Marionette |
| `figs/fig14.pdf` | Figure 14 | Instructions Per Cycle (IPC) comparison |
| `figs/fig15.pdf` | Figure 15 | Performance per Area (PPA) comparison |
| `figs/fig16.pdf` | Figure 16 | Total energy comparison (RipTide vs NEURA-SO) |
| `figs/fig17.pdf` | Figure 17 | Optimisation ablation study on NEURA-ST |
| `figs/fig18.pdf` | Figure 18 | Scalability: 4×4 vs 6×6 CGRA (NEURA-ST) |
| `results/latency_cycles.csv` | §8 | Raw latency in cycles per (benchmark, architecture) |
| `results/speedup.csv` | §8 | Normalised speedup data |
| `results/ipc.csv` | §8 | IPC data |
| `results/perf_per_area.csv` | §8 | PPA data |
| `results/energy.csv` | §8 | Energy data |
| `results/optimization.csv` | §8 | Optimisation pass comparison data |
| `results/scalability.csv` | §8 | Scalability data (4×4 vs 6×6) |

### B.2 Pipeline details

The pipeline (`evaluation/main.py`) performs seven steps:

1. **Compile all benchmarks** — For each of the 11 benchmarks × 5
   architectures, the pipeline: (a) translates C++ to MLIR via `cgeist`,
   (b) applies architecture-specific dataflow passes and mapping via
   `mlir-neura-opt`, (c) extracts the Initiation Interval (II) and
   configuration steps from the mapped MLIR.

2. **Compute latencies** — A cycle-accurate latency model combines II,
   pipeline steps, trip counts, and CPU–CGRA transition overhead to produce
   total execution cycles.

3. **Generate figures** — Latencies are normalised and plotted as the six
   figures in the paper.

### B.3 Expected output

Figures that are closely match the ones in the submitted paper.

---

## Claims Supported by the Artifact
1. **Area Overhead (Figure 12):** Due to the EDA license, we are not able to provide access to commercial EDA tools. As a solution, we provide the original power, area, and timing reports for the evaluation in `/workspace/dataflow/evaluation/arch_reports`. This figure is not generated automatically, you can just check the reports for area overhead.
1. **Performance (Figure 13):** The artifact compiles all 11 benchmarks
   across 5 architectures and reproduces Figure 13.

2. **IPC (Figure 14):** The
   artifact uses instruction counts and computed latencies to reproduce
   Figure 14.

3. **Performance per Area (Figure 15):** The artifact uses synthesis
   area data from `/workspace/dataflow/evaluation/arch_reports/` and computed latencies to
   reproduce Figure 15.

4. **Energy efficiency (Figure 16):** The artifact uses synthesis power in `/workspace/dataflow/evaluation/arch_reports` to
   reproduce Figure 16.

5. **Optimisation effectiveness (Figure 17):** Each optimisation pass
   (computational pattern fusion, data type alignment, constant folding,
   loop streaming) contributes incrementally to performance. The artifact
   compiles benchmarks with each pass combination disabled/enabled and
   reproduces Figure 17.

6. **Scalability (Figure 18):** The artifact compiles benchmarks with both array sizes (using `/workspace/dataflow/build/tools/mlir-neura-opt/mlir-neura-opt` and `/workspace/dataflow/build/tools/mlir-neura-opt/mlir-neura-opt-4x4` respectively) and
   reproduces Figure 18.

## Claims NOT Supported by the Artifact

1. **Hardware synthesis results (area, power, timing):** The area and power
   results in Fig.12 and Fig.15–16 originate from Synopsys Design
   Compiler synthesis reports (TSMC 22nm). These are provided as-is under
   `/workspace/dataflow/evaluation/arch_reports/` but **cannot be regenerated** without
   commercial EDA tools and a foundry PDK due to **license issues**. The artifact uses these
   pre-existing values directly. Furthermore, for the hardware metrics of Marionette, we estimated them by scaling the REVEL RTL synthesis results based on the data reported in the original Marionette paper.

2. **RTL designs:** The CGRA RTL designs are not
   included. These require the PyMTL3 framework and our CGRA RTL generator,
   which are part of a separate project.

---

## Rebuilding the Compiler from Source

Reviewers can recompile the Neura compiler from source inside Docker:

```bash
docker run --rm -it shangkunli/neura-pldi26:pldi2026 build
```

This runs `cmake` + `make` against the bundled LLVM/MLIR (v20.1.7) and
produces fresh `mlir-neura-opt`, `neura-interpreter`, and `neura-compiler`
binaries. After building, run the lit tests or full experiments with the
newly-built binaries.

---

## Interactive Exploration

```bash
docker run --rm -it shangkunli/neura-pldi26:pldi2026 bash
```

### Container layout

```
/workspace/
├── llvm-project/                           Pre-built LLVM/MLIR (v20.1.7)
│   ├── build/
│   │   ├── lib/                            Static libraries + cmake configs
│   │   ├── include/                        Generated LLVM headers
│   │   ├── tools/mlir/include/             Generated MLIR dialect headers
│   │   └── bin/                            llvm-tblgen, mlir-tblgen, mlir-opt,
│   │                                       FileCheck, llvm-lit, clang, llc, ...
│   ├── mlir/include/                       MLIR source headers
│   └── llvm/
│       ├── include/                        LLVM source headers
│       ├── cmake/                          CMake modules (TableGen, AddLLVM, ...)
│       └── utils/lit/                      Lit test framework (Python)
│
└── dataflow/                               Neura compiler project
    ├── CMakeLists.txt
    ├── include/, lib/                      Compiler source (dialect definitions,
    │                                       passes, conversions)
    ├── tools/
    │   ├── mlir-neura-opt/                 Main compiler driver source
    │   ├── neura-compiler/                 MLIR → LLVM IR code generator source
    │   └── neura-interpreter/              Dataflow graph interpreter source
    ├── test/                               Lit test suite (64 tests)
    ├── build/tools/
    │   ├── mlir-neura-opt/
    │   │   ├── mlir-neura-opt              Pre-built compiler (6×6 CGRA)
    │   │   └── mlir-neura-opt-4x4          Pre-built compiler (4×4 CGRA)
    │   ├── neura-interpreter/              Pre-built interpreter
    │   └── neura-compiler/                 Pre-built code generator
    ├── thirdparty/polygeist/cgeist         C/C++ → MLIR frontend
    └── evaluation/
        ├── main.py                         Experiment orchestration
        ├── util/                           Config, compiler, latency, visualizer
        ├── benchmarks/                     C++ benchmark sources (11 benchmarks)
        └── arch_reports/                   Synthesis reports per architecture
```

### Tool descriptions

| Binary | Purpose |
|---|---|
| `mlir-neura-opt` | Main compiler driver. Applies dataflow lowering, optimisation passes, and spatial/spatial-temporal mapping to MLIR programs. |
| `neura-compiler` | Used to run pass pipelines without specifying multiple passes. |
| `neura-interpreter` | Dataflow graph interpreter. Simulates execution of Neura dataflow IR for functional verification. |
| `cgeist` | Polygeist C/C++ frontend. Translates C/C++ source to MLIR (used in the evaluation pipeline). |

### Manual commands

```bash
# Run lit tests
cd /workspace/dataflow
python3 /workspace/llvm-project/build/bin/llvm-lit test/ -v

# Run full evaluation
python3 evaluation/main.py

# Rebuild from source in the docker
cd /workspace/dataflow/build
cmake .. \
    -DMLIR_DIR=/workspace/llvm-project/build/lib/cmake/mlir \
    -DLLVM_DIR=/workspace/llvm-project/build/lib/cmake/llvm \
    -DLLVM_TOOLS_BINARY_DIR=/workspace/llvm-project/build/bin \
    -DMLIR_BINARY_DIR=/workspace/llvm-project/build
make -j$(nproc)
```

---

## Building from Scratch (without Docker, ~2 h)

To build LLVM and Neura from source on a host machine:

1. Clone [llvm-project](https://github.com/llvm/llvm-project) and check out
   commit `6146a88`.

2. Build LLVM:
```bash
cd llvm-project && mkdir build && cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="Native" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="-std=c++17 -frtti" \
    -DLLVM_ENABLE_RTTI=ON
cmake --build .
```

3. Build Neura:
```bash
cd neura && mkdir build && cd build
cmake -G Ninja .. \
    -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
    -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
    -DMLIR_BINARY_DIR=/path/to/llvm-project/build \
    -DLLVM_TOOLS_BINARY_DIR=/path/to/llvm-project/build/bin
ninja
```

4. Run tests:
```bash
/path/to/llvm-project/build/bin/llvm-lit test/ -v
```

---

