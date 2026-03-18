// Compiles the original kernel.
// RUN: clang++ kernel.cpp -o %t-kernel.out

// Compiles the original kernel to mlir, then lowers back to llvm, eventually binary.
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

// Tests LLVM to NEURA lowering.
// RUN: clang++ -S -emit-llvm -O3 -fno-unroll-loops -fno-vectorize -ffp-contract=off kernel.cpp -o %t-kernel.ll
// RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir

// RUN: mlir-neura-opt --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-input-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --view-op-graph \
// RUN:   --architecture-spec=../../arch_spec/architecture.yaml \
// RUN:   --insert-data-mov %t-kernel.mlir -o %t-kernel-neura.mlir
// RUN: FileCheck %s --check-prefix=CHECK-LLVM2NEURA < %t-kernel-neura.mlir

// Test with mapping table dump enabled
// RUN: mlir-neura-opt --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-input-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --view-op-graph \
// RUN:   --architecture-spec=../../arch_spec/architecture.yaml \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic backtrack-config=customized=5,3 dump-mapping-table=true" %t-kernel.mlir -o %t-kernel-mapped.mlir 2>&1 | tee %t-kernel-mapping-output.txt
// RUN: FileCheck %s --check-prefix=CHECK-MAPPING-TABLE < %t-kernel-mapping-output.txt
// RUN: FileCheck %s --check-prefix=CHECK-LLVM2NEURA-MAP < %t-kernel-mapped.mlir

// Checks the resource allocation table output
// CHECK-MAPPING-TABLE: === MappingState: Resource Allocation Table ===
// CHECK-MAPPING-TABLE: II = 5
// CHECK-MAPPING-TABLE: Tile     | t%{{[0-9]+}}={{[0-9]+}}
// CHECK-MAPPING-TABLE: ---------+
// CHECK-MAPPING-TABLE: Tile#{{[0-9]+}} |

// CHECK-LLVM2NEURA: accelerator = "neura"
// CHECK-LLVM2NEURA: dataflow_mode = "predicate"
// CHECK-LLVM2NEURA: neura.phi_start
// CHECK-LLVM2NEURA: neura.gep
// CHECK-LLVM2NEURA-SAME: operandSegmentSizes = array<i32: 0, 1>
// CHECK-LLVM2NEURA-SAME: lhs_value
// CHECK-LLVM2NEURA: neura.load
// CHECK-LLVM2NEURA: neura.fmul
// CHECK-LLVM2NEURA: neura.fadd
// CHECK-LLVM2NEURA: neura.store
// CHECK-LLVM2NEURA-SAME: rhs_value

// CHECK-LLVM2NEURA-MAP:      func.func @
// CHECK-LLVM2NEURA-MAP-SAME:  accelerator = "neura"
// CHECK-LLVM2NEURA-MAP-SAME:  dataflow_mode = "predicate"
// CHECK-LLVM2NEURA-MAP-SAME:  mapping_info = {
// CHECK-LLVM2NEURA-MAP-SAME:   compiled_ii = 5 : i32, 
// CHECK-LLVM2NEURA-MAP-SAME:   mapping_mode = "spatial-temporal"
// CHECK-LLVM2NEURA-MAP-SAME:   mapping_strategy = "heuristic"
// CHECK-LLVM2NEURA-MAP-SAME:   rec_mii = 5 : i32
// CHECK-LLVM2NEURA-MAP-SAME:   res_mii = 2 : i32
// CHECK-LLVM2NEURA-MAP-SAME:   x_tiles = 4 : i32
// CHECK-LLVM2NEURA-MAP-SAME:   y_tiles = 4 : i32}