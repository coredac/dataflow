// Tests FIR kernel with removed tiles in architecture.
// This test verifies that tiles marked with existence:false are not used in mapping.

// RUN: clang++ -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -o %t-kernel-full.ll %S/../benchmark/CGRA-Bench/kernels/fir/fir_int.cpp
// RUN: llvm-extract --rfunc=".*kernel.*" %t-kernel-full.ll -o %t-kernel-only.ll
// RUN: mlir-translate --import-llvm %t-kernel-only.ll -o %t-kernel.mlir

// RUN: mlir-neura-opt %t-kernel.mlir \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-input-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --architecture-spec=%S/../arch_spec/custom_arch_with_removed_tiles.yaml \
// RUN:   -o %t-after-mapping.mlir

// RUN: FileCheck %s --input-file=%t-after-mapping.mlir

// Verifies that the function was successfully mapped.
// CHECK: func.func @_Z6kernelPiS_S_
// CHECK: mapping_info =

// Verifies that removed tiles are NOT used in mapping.
// These tiles are used in the default FIR mapping, so removing them forces alternative placement.
// CHECK-NOT: x = 1 : i32, y = 1 : i32
// CHECK-NOT: x = 0 : i32, y = 1 : i32

// Verifies that some operations were successfully mapped.
// CHECK: mapping_locs =
// CHECK: neura.load
// CHECK: neura.return
