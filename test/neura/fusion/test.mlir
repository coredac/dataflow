// RUN: clang++ -S -emit-llvm -O3 -fno-unroll-loops -fno-vectorize -o %t-kernel.ll %S/kernel.cpp
// RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir
// RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --assign-accelerator \
// RUN:           --lower-llvm-to-neura \
// RUN:           --promote-func-arg-to-const \
// RUN:           --canonicalize-return \
// RUN:           --canonicalize-live-in \
// RUN:           --leverage-predicated-value \
// RUN:           --fold-constant \
// RUN:           --transform-ctrl-to-data-flow \
// RUN:           --fold-constant \
// RUN:           --fuse-pattern \
// RUN:           --insert-data-mov %t-kernel.mlir \
// RUN: | FileCheck %s --check-prefix=CHECK-FUSED

// RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --assign-accelerator \
// RUN:           --lower-llvm-to-neura \
// RUN:           --promote-func-arg-to-const \
// RUN:           --canonicalize-return \
// RUN:           --canonicalize-live-in \
// RUN:           --leverage-predicated-value \
// RUN:           --fold-constant \
// RUN:           --transform-ctrl-to-data-flow \
// RUN:           --fold-constant \
// RUN:           --fuse-pattern \
// RUN:           --insert-data-mov \
// RUN:           --map-to-accelerator="mapping-strategy=heuristic backtrack-config=simple" %t-kernel.mlir | FileCheck %s --check-prefix=CHECK-MAPPING

// CHECK-FUSED: func.func @_Z6kernel
// CHECK-FUSED-SAME: accelerator = "neura"
// CHECK-FUSED:      neura.load_indexed
// CHECK-FUSED:      "neura.mul_add"
// CHECK-FUSED:      "neura.mul_add"

// CHECK-MAPPING: mapping_info = {compiled_ii = {{1[0-5]}} : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = {{[0-9]+}} : i32, res_mii = {{[0-9]+}} : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}

// RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --verify-each=true \
// RUN:           --assign-accelerator \
// RUN:           --lower-llvm-to-neura \
// RUN:           --promote-func-arg-to-const \
// RUN:           --canonicalize-cast \
// RUN:           --canonicalize-return \
// RUN:           --canonicalize-live-in \
// RUN:           --leverage-predicated-value \
// RUN:           --fold-constant \
// RUN:           --transform-ctrl-to-data-flow \
// RUN:           --fold-constant \
// RUN:           --iter-merge-pattern="min-support=3 max-iter=4" %t-kernel.mlir \
// RUN: | FileCheck %s --check-prefix=CHECK-ITER-MERGE-PATTERN

// CHECK-ITER-MERGE-PATTERN:       "neura.fused_op"
// CHECK-ITER-MERGE-PATTERN-SAME:  pattern_name = "grant_once->phi_start"
// CHECK-ITER-MERGE-PATTERN:       "neura.grant_once"
// CHECK-ITER-MERGE-PATTERN:       neura.phi_start
// CHECK-ITER-MERGE-PATTERN:       neura.yield

// CHECK-ITER-MERGE-PATTERN:       "neura.fused_op"
// CHECK-ITER-MERGE-PATTERN-SAME:  pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"
// CHECK-ITER-MERGE-PATTERN:       neura.phi_start
// CHECK-ITER-MERGE-PATTERN:       neura.phi_start
// CHECK-ITER-MERGE-PATTERN:       "neura.gep"
// CHECK-ITER-MERGE-PATTERN:       "neura.load"
// CHECK-ITER-MERGE-PATTERN:       neura.yield

// CHECK-ITER-MERGE-PATTERN:       "neura.fused_op"
// CHECK-ITER-MERGE-PATTERN-SAME:  pattern_name = "phi_start->fused_op:gep->load"
// CHECK-ITER-MERGE-PATTERN:       neura.phi_start
// CHECK-ITER-MERGE-PATTERN:       "neura.gep"
// CHECK-ITER-MERGE-PATTERN:       "neura.load"
// CHECK-ITER-MERGE-PATTERN:       neura.yield

// RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --verify-each=true \
// RUN:           --assign-accelerator \
// RUN:           --lower-llvm-to-neura \
// RUN:           --promote-func-arg-to-const \
// RUN:           --canonicalize-cast \
// RUN:           --canonicalize-return \
// RUN:           --canonicalize-live-in \
// RUN:           --leverage-predicated-value \
// RUN:           --fold-constant \
// RUN:           --transform-ctrl-to-data-flow \
// RUN:           --fold-constant \
// RUN:           --init-pattern %t-kernel.mlir \
// RUN:           | FileCheck %s --check-prefix=CHECK-INIT-PATTERN

// CHECK-INIT-PATTERN:         "neura.fused_op"
// CHECK-INIT-PATTERN-SAME:    pattern_name = "gep->load"
// CHECK-INIT-PATTERN:         "neura.gep"
// CHECK-INIT-PATTERN:         "neura.load"
// CHECK-INIT-PATTERN:         neura.yield

// RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --verify-each=true \
// RUN:           --assign-accelerator \
// RUN:           --lower-llvm-to-neura \
// RUN:           --promote-func-arg-to-const \
// RUN:           --canonicalize-cast \
// RUN:           --canonicalize-return \
// RUN:           --canonicalize-live-in \
// RUN:           --leverage-predicated-value \
// RUN:           --fold-constant \
// RUN:           --transform-ctrl-to-data-flow \
// RUN:           --fold-constant \
// RUN:           --iter-merge-pattern="min-support=3 max-iter=4" \
// RUN:           --insert-data-mov \
// RUN:           --map-to-accelerator="mapping-strategy=heuristic backtrack-config=simple" %t-kernel.mlir | FileCheck %s --check-prefix=CHECK-ITER-MERGE-PATTERN-MAPPING

// CHECK-ITER-MERGE-PATTERN-MAPPING: mapping_info = {compiled_ii = {{1[0-5]}} : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = {{[0-9]+}} : i32, res_mii = {{[0-9]+}} : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}