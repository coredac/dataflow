// Compiles the original C kernel to mlir with vectorization enabled, then lowers it via Neura.
// RUN: clang++ -S -emit-llvm -O3 -fno-unroll-loops -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/fir/fir_int.cpp
// RUN: llvm-extract --rfunc=".*kernel.*" %t-kernel-full.ll -o %t-kernel-only.ll
// RUN: mlir-translate --import-llvm %t-kernel-only.ll -o %t-kernel.mlir

// RUN: mlir-neura-opt %t-kernel.mlir \
// RUN:   --assign-accelerator \
// RUN:   -o %t-1-assign-accelerator.mlir

// RUN: mlir-neura-opt %t-1-assign-accelerator.mlir \
// RUN:   --lower-llvm-to-neura \
// RUN:   -o %t-2-lower-llvm-to-neura.mlir

// RUN: mlir-neura-opt %t-2-lower-llvm-to-neura.mlir \
// RUN:   --promote-func-arg-to-const \
// RUN:   -o %t-3-promote-func-arg-to-const.mlir

// RUN: mlir-neura-opt %t-3-promote-func-arg-to-const.mlir \
// RUN:   --fold-constant \
// RUN:   -o %t-4-fold-constant.mlir

// RUN: mlir-neura-opt %t-4-fold-constant.mlir \
// RUN:   --canonicalize-live-in \
// RUN:   -o %t-5-canonicalize-live-in.mlir

// RUN: mlir-neura-opt %t-5-canonicalize-live-in.mlir \
// RUN:   --leverage-predicated-value \
// RUN:   -o %t-6-leverage-predicated-value.mlir

// RUN: mlir-neura-opt %t-6-leverage-predicated-value.mlir \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   -o %t-7-transform-ctrl-to-data-flow.mlir

// RUN: mlir-neura-opt %t-7-transform-ctrl-to-data-flow.mlir \
// RUN:   --fold-constant \
// RUN:   -o %t-8-fold-constant-2.mlir

// RUN: mlir-neura-opt %t-8-fold-constant-2.mlir \
// RUN:   --insert-data-mov \
// RUN:   -o %t-9-insert-data-mov.mlir

// RUN: mlir-neura-opt %t-9-insert-data-mov.mlir \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --architecture-spec=../../arch_spec/architecture.yaml \
// RUN:   --generate-code -o %t-mapping.mlir 

// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING

// MAPPING: module
// MAPPING:      func.func
// MAPPING-SAME:   mapping_mode = "spatial-temporal"
// MAPPING-SAME:   mapping_strategy = "heuristic"

