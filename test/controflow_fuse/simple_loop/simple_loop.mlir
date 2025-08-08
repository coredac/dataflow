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
// RUN: --canonicalize-live-in | FileCheck %s --check-prefix=CANO

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
// RUN: --fuse-control-flow \
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
// RUN: --fuse-control-flow \
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

// CANO:     func.func @_Z11simple_loopPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CANO-NEXT:     %0 = "neura.constant"() <{predicate = true, value = "%arg0"}> : () -> memref<?xi32>
// CANO-NEXT:     %1 = "neura.constant"() <{predicate = true, value = "%arg1"}> : () -> memref<?xi32>
// CANO-NEXT:     %2 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> i64
// CANO-NEXT:     %3 = "neura.constant"() <{predicate = true, value = 128 : i64}> : () -> i64
// CANO-NEXT:     %4 = "neura.constant"() <{predicate = true, value = 1 : i32}> : () -> i32
// CANO-NEXT:     %5 = "neura.constant"() <{predicate = true, value = 2 : i32}> : () -> i32
// CANO-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> i64
// CANO-NEXT:     neura.br %6, %3 : i64, i64 to ^bb1
// CANO-NEXT:   ^bb1(%7: i64, %8: i64):  // 2 preds: ^bb0, ^bb2
// CANO-NEXT:     %9 = "neura.icmp"(%7, %8) <{cmpType = "slt"}> : (i64, i64) -> i1
// CANO-NEXT:     neura.cond_br %9 : i1 then %0, %7, %5, %4, %1, %2, %3 : memref<?xi32>, i64, i32, i32, memref<?xi32>, i64, i64 to ^bb2 else to ^bb3
// CANO-NEXT:   ^bb2(%10: memref<?xi32>, %11: i64, %12: i32, %13: i32, %14: memref<?xi32>, %15: i64, %16: i64):  // pred: ^bb1
// CANO-NEXT:     %17 = neura.load_indexed %10[%11 : i64] memref<?xi32> : i32
// CANO-NEXT:     %18 = "neura.mul"(%17, %12) : (i32, i32) -> i32
// CANO-NEXT:     %19 = "neura.add"(%18, %13) : (i32, i32) -> i32
// CANO-NEXT:     neura.store_indexed %19 to %14[%11 : i64] memref<?xi32> : i32
// CANO-NEXT:     %20 = "neura.add"(%11, %15) : (i64, i64) -> i64
// CANO-NEXT:     neura.br %20, %16 : i64, i64 to ^bb1
// CANO-NEXT:   ^bb3:  // pred: ^bb1
// CANO-NEXT:     "neura.return"() : () -> ()
// CANO-NEXT:   }

// CTRL2DATA:      func.func @_Z11simple_loopPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
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
// CTRL2DATA-NEXT:     %15 = "neura.phi"(%14, %7) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %16 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %17 = "neura.phi"(%16, %13) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %18 = "neura.icmp"(%17, %15) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %19 = "neura.not"(%18) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %20 = neura.grant_predicate %1, %18 : !neura.data<memref<?xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %21 = neura.grant_predicate %17, %18 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %22 = neura.grant_predicate %11, %18 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %23 = neura.grant_predicate %9, %18 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %24 = neura.grant_predicate %3, %18 : !neura.data<memref<?xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %25 = neura.grant_predicate %5, %18 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %26 = neura.grant_predicate %7, %18 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %27 = neura.load_indexed %20[%21 : !neura.data<i64, i1>] !neura.data<memref<?xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %28 = "neura.mul"(%27, %22) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %29 = "neura.add"(%28, %23) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %29 to %24[%21 : !neura.data<i64, i1>] !neura.data<memref<?xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %30 = "neura.add"(%21, %25) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %30 -> %16 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %26 -> %14 : !neura.data<i64, i1> !neura.data<i64, i1>
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
// FUSE-NEXT:     %8 = "neura.not"(%valid) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-NEXT:     %9 = neura.grant_predicate %0, %valid : !neura.data<memref<?xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?xi32>, i1>
// FUSE-NEXT:     %10 = neura.grant_predicate %5, %valid : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-NEXT:     %11 = neura.grant_predicate %4, %valid : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-NEXT:     %12 = neura.grant_predicate %1, %valid : !neura.data<memref<?xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?xi32>, i1>
// FUSE-NEXT:     %13 = neura.load_indexed %9[%nextindex : !neura.data<i64, i1>] !neura.data<memref<?xi32>, i1> : !neura.data<i32, i1>
// FUSE-NEXT:     %14 = "neura.mul"(%13, %10) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-NEXT:     %15 = "neura.add"(%14, %11) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-NEXT:     neura.store_indexed %15 to %12[%nextindex : !neura.data<i64, i1>] !neura.data<memref<?xi32>, i1> : !neura.data<i32, i1>
// FUSE-NEXT:     "neura.return"() : () -> ()
// FUSE-NEXT:   }

// FUSE-MAPPING:        func.func @_Z11simple_loopPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {CompiledII = 5 : i32, RecMII = 1 : i32, ResMII = 2 : i32, accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// FUSE-MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = "%arg0"}> {mapping_locs = [{id = 11 : i32, resource = "tile", time_step = 1 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %1 = "neura.grant_once"() <{constant_value = "%arg1"}> {mapping_locs = [{id = 11 : i32, resource = "tile", time_step = 4 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %2 = "neura.grant_always"() <{constant_value = 1 : i64}> {mapping_locs = [{id = 11 : i32, resource = "tile", time_step = 0 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %3 = "neura.grant_always"() <{constant_value = 128 : i64}> {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %4 = "neura.grant_once"() <{constant_value = 1 : i32}> {mapping_locs = [{id = 11 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %5 = "neura.grant_once"() <{constant_value = 2 : i32}> {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %6 = "neura.grant_once"() <{constant_value = 0 : i64}> {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 1 : i32}]} : () -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %7 = "neura.grant_always"() <{constant_value = true}> {mapping_locs = [{id = 4 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 1 : i32}]} : () -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %8 = "neura.data_mov"(%7) {mapping_locs = [{id = 10 : i32, resource = "link", time_step = 0 : i32}, {id = 14 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %9 = "neura.data_mov"(%6) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %10 = "neura.data_mov"(%3) {mapping_locs = [{id = 28 : i32, resource = "link", time_step = 0 : i32}, {id = 33 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %11 = "neura.data_mov"(%2) {mapping_locs = [{id = 36 : i32, resource = "link", time_step = 0 : i32}, {id = 21 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %nextindex, %valid = neura.loop_control(parent_valid = %8, start = %9, end = %10, step = %11) {iterationType = "increment", mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>, !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %12 = "neura.data_mov"(%valid) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %13 = "neura.not"(%12) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %14 = "neura.data_mov"(%0) {mapping_locs = [{id = 36 : i32, resource = "link", time_step = 1 : i32}, {id = 28 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %15 = "neura.data_mov"(%valid) {mapping_locs = [{id = 18 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %16 = neura.grant_predicate %14, %15 {mapping_locs = [{id = 7 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 1 : i32}]} : !neura.data<memref<?xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %17 = "neura.data_mov"(%5) {mapping_locs = [{id = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 4 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %18 = "neura.data_mov"(%valid) {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 2 : i32}, {id = 15 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %19 = neura.grant_predicate %17, %18 {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %20 = "neura.data_mov"(%4) {mapping_locs = [{id = 35 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %21 = "neura.data_mov"(%valid) {mapping_locs = [{id = 20 : i32, resource = "link", time_step = 2 : i32}, {id = 40 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %22 = neura.grant_predicate %20, %21 {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %23 = "neura.data_mov"(%1) {mapping_locs = [{id = 36 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %24 = "neura.data_mov"(%valid) {mapping_locs = [{id = 19 : i32, resource = "link", time_step = 2 : i32}, {id = 6 : i32, resource = "link", time_step = 3 : i32}, {id = 9 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %25 = neura.grant_predicate %23, %24 {mapping_locs = [{id = 7 : i32, resource = "tile", time_step = 5 : i32, x = 3 : i32, y = 1 : i32}]} : !neura.data<memref<?xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %26 = "neura.data_mov"(%16) {mapping_locs = [{id = 21 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %27 = "neura.data_mov"(%nextindex) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %28 = neura.load_indexed %26[%27 : !neura.data<i64, i1>] !neura.data<memref<?xi32>, i1> {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %29 = "neura.data_mov"(%28) {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %30 = "neura.data_mov"(%19) {mapping_locs = [{id = 4 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %31 = "neura.mul"(%29, %30) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %32 = "neura.data_mov"(%31) {mapping_locs = [{id = 16 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %33 = "neura.data_mov"(%22) {mapping_locs = [{id = 31 : i32, resource = "link", time_step = 4 : i32}, {id = 36 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %34 = "neura.add"(%32, %33) {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %35 = "neura.data_mov"(%34) {mapping_locs = [{id = 28 : i32, resource = "link", time_step = 6 : i32}, {id = 33 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %36 = "neura.data_mov"(%25) {mapping_locs = [{id = 21 : i32, resource = "link", time_step = 5 : i32}, {id = 24 : i32, resource = "register", time_step = 6 : i32}, {id = 24 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %37 = "neura.data_mov"(%nextindex) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     neura.store_indexed %35 to %36[%37 : !neura.data<i64, i1>] !neura.data<memref<?xi32>, i1> {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 8 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     "neura.return"() {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 0 : i32}]} : () -> ()
// FUSE-MAPPING-NEXT:   }