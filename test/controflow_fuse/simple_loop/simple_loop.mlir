// RUN: mlir-opt %s \
// RUN: --lower-affine \
// RUN: --convert-scf-to-cf \
// RUN: --convert-cf-to-llvm -o %t-llvm.mlir

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: | FileCheck %s

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --promote-func-arg-to-const \
// RUN: --canonicalize-live-in | FileCheck %s --check-prefix=CANONICALIZE

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --promote-func-arg-to-const \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: | FileCheck %s -check-prefix=CTRL2DATA

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --promote-func-arg-to-const \
// RUN: --fold-constant \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: --fold-constant \
// RUN: --fuse-loop-control \
// RUN: --fold-constant \
// RUN: | FileCheck %s --check-prefix=FUSE

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --promote-func-arg-to-const \
// RUN: --fold-constant \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: --fold-constant \
// RUN: --fuse-loop-control \
// RUN: --fold-constant \
// RUN: --insert-data-mov \
// RUN: --map-to-accelerator="mapping-strategy=heuristic backtrack-config=customized" \
// RUN: --architecture-spec=../../arch_spec/architecture.yaml \
// RUN: | FileCheck %s --check-prefix=FUSE-MAPPING

module attributes {} {
  func.func @_Z11simple_loopPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    affine.for %arg2 = 0 to 128 {
      %0 = affine.load %arg0[%arg2] : memref<?xi32>
      %1 = arith.muli %0, %c2_i32 : i32
      %2 = arith.addi %1, %c1_i32 : i32
      affine.store %2, %arg1[%arg2] : memref<?xi32>
    }
    return
  }
}


// CHECK:      func.func @_Z11simple_loopPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = "neura.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT:     %1 = "neura.constant"() <{value = 128 : index}> : () -> index
// CHECK-NEXT:     %2 = "neura.constant"() <{value = 1 : i32}> : () -> i32
// CHECK-NEXT:     %3 = "neura.constant"() <{value = 2 : i32}> : () -> i32
// CHECK-NEXT:     %4 = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:     %5 = "neura.cast"(%4) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %5 : i64 to ^bb1
// CHECK-NEXT:   ^bb1(%6: i64):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %7 = "neura.cast"(%6) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %8 = "neura.icmp"(%7, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %8 : i1 then to ^bb2 else to ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %9 = neura.load_indexed %arg0[%7 : index] memref<?xi32> : i32
// CHECK-NEXT:     %10 = "neura.mul"(%9, %3) : (i32, i32) -> i32
// CHECK-NEXT:     %11 = "neura.add"(%10, %2) : (i32, i32) -> i32
// CHECK-NEXT:     neura.store_indexed %11 to %arg1[%7 : index] memref<?xi32> : i32
// CHECK-NEXT:     %12 = "neura.add"(%7, %0) : (index, index) -> index
// CHECK-NEXT:     %13 = "neura.cast"(%12) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %13 : i64 to ^bb1
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     "neura.return"() : () -> ()
// CHECK-NEXT:   }

// CANONICALIZE:       func.func @_Z11simple_loopPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CANONICALIZE-NEXT:     %0 = "neura.constant"() <{value = "%arg0"}> : () -> memref<?xi32>
// CANONICALIZE-NEXT:     %1 = "neura.constant"() <{value = "%arg1"}> : () -> memref<?xi32>
// CANONICALIZE-NEXT:     %2 = "neura.constant"() <{value = 1 : i64}> : () -> i64
// CANONICALIZE-NEXT:     %3 = "neura.constant"() <{value = 128 : i64}> : () -> i64
// CANONICALIZE-NEXT:     %4 = "neura.constant"() <{value = 1 : i32}> : () -> i32
// CANONICALIZE-NEXT:     %5 = "neura.constant"() <{value = 2 : i32}> : () -> i32
// CANONICALIZE-NEXT:     %6 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CANONICALIZE-NEXT:     neura.br %6, %3, %0, %5, %4, %1, %2 : i64, i64, memref<?xi32>, i32, i32, memref<?xi32>, i64 to ^bb1
// CANONICALIZE-NEXT:   ^bb1(%7: i64, %8: i64, %9: memref<?xi32>, %10: i32, %11: i32, %12: memref<?xi32>, %13: i64):  // 2 preds: ^bb0, ^bb2
// CANONICALIZE-NEXT:     %14 = "neura.icmp"(%7, %8) <{cmpType = "slt"}> : (i64, i64) -> i1
// CANONICALIZE-NEXT:     neura.cond_br %14 : i1 then %9, %7, %10, %11, %12, %13, %8 : memref<?xi32>, i64, i32, i32, memref<?xi32>, i64, i64 to ^bb2 else to ^bb3
// CANONICALIZE-NEXT:   ^bb2(%15: memref<?xi32>, %16: i64, %17: i32, %18: i32, %19: memref<?xi32>, %20: i64, %21: i64):  // pred: ^bb1
// CANONICALIZE-NEXT:     %22 = neura.load_indexed %15[%16 : i64] memref<?xi32> : i32
// CANONICALIZE-NEXT:     %23 = "neura.mul"(%22, %17) : (i32, i32) -> i32
// CANONICALIZE-NEXT:     %24 = "neura.add"(%23, %18) : (i32, i32) -> i32
// CANONICALIZE-NEXT:     neura.store_indexed %24 to %19[%16 : i64] memref<?xi32> : i32
// CANONICALIZE-NEXT:     %25 = "neura.add"(%16, %20) : (i64, i64) -> i64
// CANONICALIZE-NEXT:     neura.br %25, %21, %15, %17, %18, %19, %20 : i64, i64, memref<?xi32>, i32, i32, memref<?xi32>, i64 to ^bb1
// CANONICALIZE-NEXT:   ^bb3:  // pred: ^bb1
// CANONICALIZE-NEXT:     "neura.return"() : () -> ()
// CANONICALIZE-NEXT:   }

// CTRL2DATA:        func.func @_Z11simple_loopPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {accelerator = "neura", dataflow_mode = "predicate", llvm.linkage = #llvm.linkage<external>} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = "%arg0"}> : () -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = "%arg1"}> : () -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{value = 128 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{value = 1 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %10 = "neura.constant"() <{value = 2 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %11 = "neura.grant_once"(%10) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %12 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %13 = "neura.grant_once"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %14 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %15 = neura.phi_start %5, %14 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %16 = neura.reserve : !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %17 = neura.phi_start %3, %16 : !neura.data<memref<?xi32>, i1>, !neura.data<memref<?xi32>, i1> -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %18 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %19 = neura.phi_start %9, %18 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %20 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %21 = neura.phi_start %11, %20 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %22 = neura.reserve : !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %23 = neura.phi_start %1, %22 : !neura.data<memref<?xi32>, i1>, !neura.data<memref<?xi32>, i1> -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %24 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %25 = neura.phi_start %7, %24 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %26 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %27 = neura.phi_start %13, %26 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %28 = "neura.icmp"(%27, %25) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %29 = neura.grant_predicate %23, %28 : !neura.data<memref<?xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %30 = neura.grant_predicate %27, %28 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %31 = neura.grant_predicate %21, %28 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %32 = neura.grant_predicate %19, %28 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %33 = neura.grant_predicate %17, %28 : !neura.data<memref<?xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %34 = neura.grant_predicate %15, %28 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %35 = neura.grant_predicate %25, %28 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %36 = neura.load_indexed %29[%30 : !neura.data<i64, i1>] !neura.data<memref<?xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %37 = "neura.mul"(%36, %31) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %38 = "neura.add"(%37, %32) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %38 to %33[%30 : !neura.data<i64, i1>] !neura.data<memref<?xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %39 = "neura.add"(%30, %34) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %39 -> %26 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %35 -> %24 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %29 -> %22 : !neura.data<memref<?xi32>, i1> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %31 -> %20 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %32 -> %18 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %33 -> %16 : !neura.data<memref<?xi32>, i1> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %34 -> %14 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     "neura.return"() : () -> ()
// CTRL2DATA-NEXT:   }


// FUSE:        func.func @_Z11simple_loopPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {accelerator = "neura", dataflow_mode = "predicate", llvm.linkage = #llvm.linkage<external>} {
// FUSE-NEXT:     %0 = "neura.grant_always"() <{constant_value = true}> : () -> !neura.data<i1, i1>
// FUSE-NEXT:     %nextindex, %valid = "neura.loop_control"(%0) <{end = 128 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (!neura.data<i1, i1>) -> (!neura.data<i64, i1>, !neura.data<i1, i1>)
// FUSE-NEXT:     %1 = neura.load_indexed [%nextindex : !neura.data<i64, i1>]  {lhs_value = "%arg0"} : !neura.data<i32, i1>
// FUSE-NEXT:     %2 = "neura.mul"(%1) {rhs_value = 2 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-NEXT:     %3 = "neura.add"(%2) {rhs_value = 1 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-NEXT:     neura.store_indexed %3 to [%nextindex : !neura.data<i64, i1>]  {rhs_value = "%arg1"} : !neura.data<i32, i1>
// FUSE-NEXT:     "neura.return"() : () -> ()
// FUSE-NEXT:   }

// FUSE-MAPPING:  func.func @_Z11simple_loopPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {accelerator = "neura", dataflow_mode = "predicate", llvm.linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 1 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 1 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {