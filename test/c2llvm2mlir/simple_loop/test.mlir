// Compiles the original kernel.
// RUN: clang++ kernel.cpp -o %t-kernel.out

// Compiles the original kernel to mlir, then lower back to llvm, eventually binary.
// RUN: clang++ -S -emit-llvm -o %t-kernel.ll kernel.cpp
// RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir
// RUN: mlir-opt %t-kernel.mlir | mlir-translate -mlir-to-llvmir -o %t-kernel_back.ll
// RUN: llc %t-kernel_back.ll -relocation-model=pic -filetype=obj -o %t-kernel_back.o
// RUN: clang++ %t-kernel_back.o -o %t-kernel_back.out

// RUN: %t-kernel.out > %t-dumped_output.txt
// RUN: %t-kernel_back.out >> %t-dumped_output.txt
// RUN: FileCheck %s < %t-dumped_output.txt

// Verifies the output values are the same for the original and re-compiled kernel.
// CHECK: output: [[OUTPUT:[0-9]+\.[0-9]+]]
// CHECK: output: [[OUTPUT]]

// Test LLVM to NEURA lowering
// RUN: clang++ -S -emit-llvm -O3 -fno-unroll-loops -fno-vectorize -ffp-contract=off kernel.cpp -o %t-kernel.ll
// RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir

// RUN: mlir-neura-opt --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --view-op-graph \
// RUN:   --architecture-spec=../../arch_spec/architecture.yaml \
// RUN:   --insert-data-mov %t-kernel.mlir -o %t-kernel-neura.mlir
// RUN: FileCheck %s --check-prefix=CHECK-LLVM2NEURA < %t-kernel-neura.mlir

// RUN: mlir-neura-opt --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --view-op-graph \
// RUN:   --architecture-spec=../../arch_spec/architecture.yaml \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic backtrack-config=customized=5,3" %t-kernel.mlir -o %t-kernel-mapped.mlir
// RUN: FileCheck %s --check-prefix=CHECK-LLVM2NEURA-MAP < %t-kernel-mapped.mlir

// CHECK-LLVM2NEURA: accelerator = "neura"
// CHECK-LLVM2NEURA: dataflow_mode = "predicate"
// CHECK-LLVM2NEURA: neura.phi
// CHECK-LLVM2NEURA: neura.gep
// CHECK-LLVM2NEURA: neura.load
// CHECK-LLVM2NEURA: neura.fmul
// CHECK-LLVM2NEURA: neura.fadd

// CHECK-LLVM2NEURA-MAP: func.func @_Z6kernelPfS_S_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 5 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 5 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
