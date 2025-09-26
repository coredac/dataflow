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
// RUN: --canonicalize-live-in | FileCheck %s --check-prefix=CANONICALIZE

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow | FileCheck %s -check-prefix=CTRL2DATA

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
// RUN: | FileCheck %s -check-prefix=FUSE

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
// RUN: --map-to-accelerator="mapping-strategy=heuristic backtrack-config=customized" | FileCheck %s -check-prefix=FUSE-MAPPING

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
// CHECK-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 1 : index}> : () -> index
// CHECK-NEXT:     %1 = "neura.constant"() <{predicate = true, value = 128 : index}> : () -> index
// CHECK-NEXT:     %2 = "neura.constant"() <{predicate = true, value = 1 : i32}> : () -> i32
// CHECK-NEXT:     %3 = "neura.constant"() <{predicate = true, value = 2 : i32}> : () -> i32
// CHECK-NEXT:     %4 = "neura.constant"() <{predicate = true, value = 0 : index}> : () -> index
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
// CANONICALIZE-NEXT:     %0 = "neura.constant"() <{predicate = true, value = "%arg0"}> : () -> memref<?xi32>
// CANONICALIZE-NEXT:     %1 = "neura.constant"() <{predicate = true, value = "%arg1"}> : () -> memref<?xi32>
// CANONICALIZE-NEXT:     %2 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> i64
// CANONICALIZE-NEXT:     %3 = "neura.constant"() <{predicate = true, value = 128 : i64}> : () -> i64
// CANONICALIZE-NEXT:     %4 = "neura.constant"() <{predicate = true, value = 1 : i32}> : () -> i32
// CANONICALIZE-NEXT:     %5 = "neura.constant"() <{predicate = true, value = 2 : i32}> : () -> i32
// CANONICALIZE-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> i64
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

// CTRL2DATA:        func.func @_Z11simple_loopPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{predicate = true, value = "%arg0"}> : () -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{predicate = true, value = "%arg1"}> : () -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 128 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{predicate = true, value = 1 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %10 = "neura.constant"() <{predicate = true, value = 2 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %11 = "neura.grant_once"(%10) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %12 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %13 = "neura.grant_once"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %14 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %15 = "neura.phi"(%14, %5) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %16 = neura.reserve : !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %17 = "neura.phi"(%16, %3) : (!neura.data<memref<?xi32>, i1>, !neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %18 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %19 = "neura.phi"(%18, %9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %20 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %21 = "neura.phi"(%20, %11) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %22 = neura.reserve : !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %23 = "neura.phi"(%22, %1) : (!neura.data<memref<?xi32>, i1>, !neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %24 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %25 = "neura.phi"(%24, %7) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %26 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %27 = "neura.phi"(%26, %13) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
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


// FUSE:        func.func @_Z11simple_loopPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// FUSE-NEXT:     %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<memref<?xi32>, i1>
// FUSE-NEXT:     %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<memref<?xi32>, i1>
// FUSE-NEXT:     %2 = "neura.grant_always"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
// FUSE-NEXT:     %3 = "neura.grant_always"() <{constant_value = 128 : i64}> : () -> !neura.data<i64, i1>
// FUSE-NEXT:     %4 = "neura.grant_once"() <{constant_value = 1 : i32}> : () -> !neura.data<i32, i1>
// FUSE-NEXT:     %5 = "neura.grant_once"() <{constant_value = 2 : i32}> : () -> !neura.data<i32, i1>
// FUSE-NEXT:     %6 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// FUSE-NEXT:     %7 = "neura.grant_always"() <{constant_value = true}> : () -> !neura.data<i1, i1>
// FUSE-NEXT:     %nextindex, %valid = neura.loop_control(parent_valid = %7, start = %6, end = %3, step = %2) {iterationType = "increment"} : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>, !neura.data<i1, i1>
// FUSE-NEXT:     %8 = neura.reserve : !neura.data<memref<?xi32>, i1>
// FUSE-NEXT:     %9 = "neura.phi"(%8, %1) : (!neura.data<memref<?xi32>, i1>, !neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// FUSE-NEXT:     %10 = neura.reserve : !neura.data<i32, i1>
// FUSE-NEXT:     %11 = "neura.phi"(%10, %4) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-NEXT:     %12 = neura.reserve : !neura.data<i32, i1>
// FUSE-NEXT:     %13 = "neura.phi"(%12, %5) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-NEXT:     %14 = neura.reserve : !neura.data<memref<?xi32>, i1>
// FUSE-NEXT:     %15 = "neura.phi"(%14, %0) : (!neura.data<memref<?xi32>, i1>, !neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// FUSE-NEXT:     %16 = neura.grant_predicate %15, %valid : !neura.data<memref<?xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?xi32>, i1>
// FUSE-NEXT:     %17 = neura.grant_predicate %13, %valid : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-NEXT:     %18 = neura.grant_predicate %11, %valid : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-NEXT:     %19 = neura.grant_predicate %9, %valid : !neura.data<memref<?xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?xi32>, i1>
// FUSE-NEXT:     %20 = neura.load_indexed %16[%nextindex : !neura.data<i64, i1>] !neura.data<memref<?xi32>, i1> : !neura.data<i32, i1>
// FUSE-NEXT:     %21 = "neura.mul"(%20, %17) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-NEXT:     %22 = "neura.add"(%21, %18) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-NEXT:     neura.store_indexed %22 to %19[%nextindex : !neura.data<i64, i1>] !neura.data<memref<?xi32>, i1> : !neura.data<i32, i1>
// FUSE-NEXT:     neura.ctrl_mov %16 -> %14 : !neura.data<memref<?xi32>, i1> !neura.data<memref<?xi32>, i1>
// FUSE-NEXT:     neura.ctrl_mov %17 -> %12 : !neura.data<i32, i1> !neura.data<i32, i1>
// FUSE-NEXT:     neura.ctrl_mov %18 -> %10 : !neura.data<i32, i1> !neura.data<i32, i1>
// FUSE-NEXT:     neura.ctrl_mov %19 -> %8 : !neura.data<memref<?xi32>, i1> !neura.data<memref<?xi32>, i1>
// FUSE-NEXT:     "neura.return"() : () -> ()
// FUSE-NEXT:   }

// FUSE-MAPPING:      func.func @_Z11simple_loopPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 4 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 2 : i32, res_mii = 1 : i32, x_tiles = 6 : i32, y_tiles = 6 : i32}} {
// FUSE-MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = "%arg0"}> {mapping_locs = [{id = 21 : i32, resource = "tile", time_step = 0 : i32, x = 3 : i32, y = 3 : i32}]} : () -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %1 = "neura.grant_once"() <{constant_value = "%arg1"}> {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 1 : i32}]} : () -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %2 = "neura.grant_always"() <{constant_value = 1 : i64}> {mapping_locs = [{id = 20 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 3 : i32}]} : () -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %3 = "neura.grant_always"() <{constant_value = 128 : i64}> {mapping_locs = [{id = 16 : i32, resource = "tile", time_step = 0 : i32, x = 4 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %4 = "neura.grant_once"() <{constant_value = 1 : i32}> {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 2 : i32, x = 4 : i32, y = 1 : i32}]} : () -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %5 = "neura.grant_once"() <{constant_value = 2 : i32}> {mapping_locs = [{id = 26 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 4 : i32}]} : () -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %6 = "neura.grant_once"() <{constant_value = 0 : i64}> {mapping_locs = [{id = 18 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 3 : i32}]} : () -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %7 = "neura.grant_always"() <{constant_value = true}> {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 0 : i32, x = 3 : i32, y = 1 : i32}]} : () -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %8 = "neura.data_mov"(%7) {mapping_locs = [{id = 27 : i32, resource = "link", time_step = 0 : i32}, {id = 26 : i32, resource = "link", time_step = 1 : i32}, {id = 112 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %9 = "neura.data_mov"(%6) {mapping_locs = [{id = 60 : i32, resource = "link", time_step = 0 : i32}, {id = 64 : i32, resource = "link", time_step = 1 : i32}, {id = 69 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %10 = "neura.data_mov"(%3) {mapping_locs = [{id = 53 : i32, resource = "link", time_step = 0 : i32}, {id = 49 : i32, resource = "link", time_step = 1 : i32}, {id = 113 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %11 = "neura.data_mov"(%2) {mapping_locs = [{id = 69 : i32, resource = "link", time_step = 0 : i32}, {id = 114 : i32, resource = "register", time_step = 1 : i32}, {id = 114 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %nextindex, %valid = neura.loop_control(parent_valid = %8, start = %9, end = %10, step = %11) {iterationType = "increment", mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>, !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %12 = neura.reserve : !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %13 = "neura.data_mov"(%1) {mapping_locs = [{id = 27 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %14 = "neura.phi"(%12, %13) {mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<memref<?xi32>, i1>, !neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %15 = neura.reserve : !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %16 = "neura.data_mov"(%4) {mapping_locs = [{id = 80 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %17 = "neura.phi"(%15, %16) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 3 : i32, x = 4 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %18 = neura.reserve : !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %19 = "neura.data_mov"(%5) {mapping_locs = [{id = 208 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %20 = "neura.phi"(%18, %19) {mapping_locs = [{id = 26 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 4 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %21 = neura.reserve : !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %22 = "neura.data_mov"(%0) {mapping_locs = [{id = 168 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %23 = "neura.phi"(%21, %22) {mapping_locs = [{id = 21 : i32, resource = "tile", time_step = 1 : i32, x = 3 : i32, y = 3 : i32}]} : (!neura.data<memref<?xi32>, i1>, !neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %24 = "neura.data_mov"(%23) {mapping_locs = [{id = 73 : i32, resource = "link", time_step = 1 : i32}, {id = 120 : i32, resource = "register", time_step = 2 : i32}, {id = 120 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %25 = "neura.data_mov"(%valid) {mapping_locs = [{id = 46 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %26 = neura.grant_predicate %24, %25 {mapping_locs = [{id = 15 : i32, resource = "tile", time_step = 4 : i32, x = 3 : i32, y = 2 : i32}]} : !neura.data<memref<?xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %27 = "neura.data_mov"(%20) {mapping_locs = [{id = 91 : i32, resource = "link", time_step = 2 : i32}, {id = 69 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %28 = "neura.data_mov"(%valid) {mapping_locs = [{id = 112 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %29 = neura.grant_predicate %27, %28 {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %30 = "neura.data_mov"(%17) {mapping_locs = [{id = 31 : i32, resource = "link", time_step = 3 : i32}, {id = 72 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %31 = "neura.data_mov"(%valid) {mapping_locs = [{id = 47 : i32, resource = "link", time_step = 3 : i32}, {id = 24 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %32 = neura.grant_predicate %30, %31 {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 5 : i32, x = 3 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %33 = "neura.data_mov"(%14) {mapping_locs = [{id = 64 : i32, resource = "register", time_step = 4 : i32}, {id = 64 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %34 = "neura.data_mov"(%valid) {mapping_locs = [{id = 114 : i32, resource = "register", time_step = 3 : i32}, {id = 47 : i32, resource = "link", time_step = 4 : i32}, {id = 65 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %35 = neura.grant_predicate %33, %34 {mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<memref<?xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %36 = "neura.data_mov"(%26) {mapping_locs = [{id = 49 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %37 = "neura.data_mov"(%nextindex) {mapping_locs = [{id = 113 : i32, resource = "register", time_step = 3 : i32}, {id = 113 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %38 = neura.load_indexed %36[%37 : !neura.data<i64, i1>] !neura.data<memref<?xi32>, i1> {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %39 = "neura.data_mov"(%38) {mapping_locs = [{id = 45 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %40 = "neura.data_mov"(%29) {mapping_locs = [{id = 45 : i32, resource = "link", time_step = 4 : i32}, {id = 104 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %41 = "neura.mul"(%39, %40) {mapping_locs = [{id = 13 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %42 = "neura.data_mov"(%41) {mapping_locs = [{id = 43 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %43 = "neura.data_mov"(%32) {mapping_locs = [{id = 27 : i32, resource = "link", time_step = 5 : i32}, {id = 23 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %44 = "neura.add"(%42, %43) {mapping_locs = [{id = 7 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %45 = "neura.data_mov"(%44) {mapping_locs = [{id = 21 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %46 = "neura.data_mov"(%35) {mapping_locs = [{id = 25 : i32, resource = "link", time_step = 6 : i32}, {id = 5 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %47 = "neura.data_mov"(%nextindex) {mapping_locs = [{id = 45 : i32, resource = "link", time_step = 3 : i32}, {id = 43 : i32, resource = "link", time_step = 4 : i32}, {id = 21 : i32, resource = "link", time_step = 5 : i32}, {id = 8 : i32, resource = "register", time_step = 6 : i32}, {id = 8 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     neura.store_indexed %45 to %46[%47 : !neura.data<i64, i1>] !neura.data<memref<?xi32>, i1> {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     neura.ctrl_mov %26 -> %21 {mapping_locs = [{id = 52 : i32, resource = "link", time_step = 4 : i32}]} : !neura.data<memref<?xi32>, i1> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     neura.ctrl_mov %29 -> %18 {mapping_locs = [{id = 48 : i32, resource = "link", time_step = 4 : i32}, {id = 70 : i32, resource = "link", time_step = 5 : i32}]} : !neura.data<i32, i1> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     neura.ctrl_mov %32 -> %15 {mapping_locs = [{id = 28 : i32, resource = "link", time_step = 5 : i32}, {id = 81 : i32, resource = "register", time_step = 6 : i32}]} : !neura.data<i32, i1> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     neura.ctrl_mov %35 -> %12 {mapping_locs = [{id = 64 : i32, resource = "register", time_step = 6 : i32}, {id = 64 : i32, resource = "register", time_step = 7 : i32}]} : !neura.data<memref<?xi32>, i1> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     "neura.return"() {mapping_locs = [{id = 22 : i32, resource = "tile", time_step = 6 : i32, x = 4 : i32, y = 3 : i32}]} : () -> ()
// FUSE-MAPPING-NEXT:   }