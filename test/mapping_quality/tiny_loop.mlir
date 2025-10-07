// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --canonicalize-live-in \
// RUN: | FileCheck %s

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: --fuse-loop-control \
// RUN: --fold-constant \
// RUN: --insert-data-mov \
// RUN: --map-to-accelerator="mapping-strategy=heuristic mapping-mode=spatial-only backtrack-config=customized=4,3" \
// RUN: --architecture-spec=../test_architecture_spec/architecture.yaml \
// RUN: | FileCheck %s -check-prefix=SPATIAL

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: --fuse-loop-control \
// RUN: --fold-constant \
// RUN: --insert-data-mov \
// RUN: --map-to-accelerator="mapping-strategy=heuristic mapping-mode=spatial-temporal backtrack-config=customized=4,4" \
// RUN: --architecture-spec=../test_architecture_spec/architecture.yaml \
// RUN: | FileCheck %s -check-prefix=SPATIAL-TEMPORAL

module {
  func.func @simple_add_loop() -> i64 {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    %result = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %c10) -> (i64) {
      %sum = arith.addi %acc, %acc : i64
      scf.yield %sum : i64
    }
    return %result : i64
  }
}

// CHECK:     func.func @simple_add_loop() -> i64 attributes {accelerator = "neura"} {
// CHECK-NEXT: %0 = "neura.constant"() <{value = 16 : i64}> : () -> i64
// CHECK-NEXT: %1 = "neura.constant"() <{value = 1 : i64}> : () -> i64
// CHECK-NEXT: %2 = "neura.constant"() <{value = 1 : i64}> : () -> i64
// CHECK-NEXT: %3 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CHECK-NEXT: neura.br %3, %2, %0, %1 : i64, i64, i64, i64 to ^bb1
// CHECK-NEXT: ^bb1(%4: i64, %5: i64, %6: i64, %7: i64):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT: %8 = "neura.icmp"(%4, %6) <{cmpType = "slt"}> : (i64, i64) -> i1
// CHECK-NEXT: neura.cond_br %8 : i1 then %5, %4, %7, %6 : i64, i64, i64, i64 to ^bb2 else %5 : i64 to ^bb3
// CHECK-NEXT: ^bb2(%9: i64, %10: i64, %11: i64, %12: i64):  // pred: ^bb1
// CHECK-NEXT: %13 = "neura.add"(%9, %9) : (i64, i64) -> i64
// CHECK-NEXT: %14 = "neura.add"(%10, %11) : (i64, i64) -> i64
// CHECK-NEXT: neura.br %14, %13, %12, %11 : i64, i64, i64, i64 to ^bb1
// CHECK-NEXT: ^bb3(%15: i64):  // pred: ^bb1
// CHECK-NEXT: "neura.return"(%15) : (i64) -> ()
// CHECK-NEXT: }

// SPATIAL:          func.func @simple_add_loop() -> i64 attributes {accelerator = "neura", mapping_info = {compiled_ii = 7 : i32, mapping_mode = "spatial-only", mapping_strategy = "heuristic", rec_mii = 3 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {

// SPATIAL-TEMPORAL:        func.func @simple_add_loop() -> i64 attributes {accelerator = "neura", mapping_info = {compiled_ii = 3 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 3 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {