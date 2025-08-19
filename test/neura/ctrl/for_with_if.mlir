// Source code:
// int for_with_if(int data[128]) {
//   int output = 0;
//   int threshold = 1000;
//   for (int i = 0; i < 128; ++i) {
//     if (output >= threshold) {
//       // Simulate backpressure by halting processing
//       output -= 5;
//     }
//     output += data[i] * 2 + 1;
//   }
//   return output;
// }

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
// RUN: --fold-constant \
// RUN: --insert-data-mov \
// RUN: --map-to-accelerator="mapping-strategy=heuristic backtrack-config=customized" | FileCheck %s -check-prefix=FUSE-MAPPING

module attributes {} {
  func.func @_Z11for_with_ifPi(%arg0: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c-5_i32 = arith.constant -5 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1000_i32 = arith.constant 1000 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %c0_i32) -> (i32) {
      %1 = arith.cmpi sge, %arg2, %c1000_i32 : i32
      %2 = scf.if %1 -> (i32) {
        %7 = arith.addi %arg2, %c-5_i32 : i32
        scf.yield %7 : i32
      } else {
        scf.yield %arg2 : i32
      }
      %3 = memref.load %arg0[%arg1] : memref<?xi32>
      %4 = arith.muli %3, %c2_i32 : i32
      %5 = arith.addi %4, %c1_i32 : i32
      %6 = arith.addi %2, %5 : i32
      scf.yield %6 : i32
    }
    return %0 : i32
  }
}

// CHECK:        func.func @_Z11for_with_ifPi(%arg0: memref<?xi32>) -> i32 attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 0 : i32}> : () -> i32
// CHECK-NEXT:     %1 = "neura.constant"() <{predicate = true, value = 1000 : i32}> : () -> i32
// CHECK-NEXT:     %2 = "neura.constant"() <{predicate = true, value = 2 : i32}> : () -> i32
// CHECK-NEXT:     %3 = "neura.constant"() <{predicate = true, value = 1 : i32}> : () -> i32
// CHECK-NEXT:     %4 = "neura.constant"() <{predicate = true, value = -5 : i32}> : () -> i32
// CHECK-NEXT:     %5 = "neura.constant"() <{predicate = true, value = 1 : index}> : () -> index
// CHECK-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 128 : index}> : () -> index
// CHECK-NEXT:     %7 = "neura.constant"() <{predicate = true, value = 0 : index}> : () -> index
// CHECK-NEXT:     %8 = "neura.cast"(%7) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %8, %0 : i64, i32 to ^bb1
// CHECK-NEXT:   ^bb1(%9: i64, %10: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %11 = "neura.cast"(%9) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %12 = "neura.icmp"(%11, %6) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %12 : i1 then to ^bb2 else to ^bb7
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %13 = "neura.icmp"(%10, %1) <{cmpType = "sge"}> : (i32, i32) -> i1
// CHECK-NEXT:     neura.cond_br %13 : i1 then to ^bb3 else to ^bb4
// CHECK-NEXT:   ^bb3:  // pred: ^bb2
// CHECK-NEXT:     %14 = "neura.add"(%10, %4) : (i32, i32) -> i32
// CHECK-NEXT:     neura.br %14 : i32 to ^bb5
// CHECK-NEXT:   ^bb4:  // pred: ^bb2
// CHECK-NEXT:     neura.br %10 : i32 to ^bb5
// CHECK-NEXT:   ^bb5(%15: i32):  // 2 preds: ^bb3, ^bb4
// CHECK-NEXT:     neura.br to ^bb6
// CHECK-NEXT:   ^bb6:  // pred: ^bb5
// CHECK-NEXT:     %16 = neura.load_indexed %arg0[%11 : index] memref<?xi32> : i32
// CHECK-NEXT:     %17 = "neura.mul"(%16, %2) : (i32, i32) -> i32
// CHECK-NEXT:     %18 = "neura.add"(%17, %3) : (i32, i32) -> i32
// CHECK-NEXT:     %19 = "neura.add"(%15, %18) : (i32, i32) -> i32
// CHECK-NEXT:     %20 = "neura.add"(%11, %5) : (index, index) -> index
// CHECK-NEXT:     %21 = "neura.cast"(%20) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %21, %19 : i64, i32 to ^bb1
// CHECK-NEXT:   ^bb7:  // pred: ^bb1
// CHECK-NEXT:     "neura.return"(%10) : (i32) -> ()
// CHECK-NEXT:   }

// CTRL2DATA:        func.func @_Z11for_with_ifPi(%arg0: memref<?xi32>) -> i32 attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{predicate = true, value = "%arg0"}> : () -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{predicate = true, value = 0 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{predicate = true, value = 1000 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 2 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{predicate = true, value = 1 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %10 = "neura.constant"() <{predicate = true, value = -5 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %11 = "neura.grant_once"(%10) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %12 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %13 = "neura.grant_once"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %14 = "neura.constant"() <{predicate = true, value = 128 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %15 = "neura.grant_once"(%14) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %16 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %17 = "neura.grant_once"(%16) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %18 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %19 = "neura.phi"(%18, %15) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %20 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %21 = "neura.phi"(%20, %3) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %22 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %23 = "neura.phi"(%22, %17) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %24 = "neura.icmp"(%23, %19) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %25 = "neura.not"(%24) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %26 = neura.grant_predicate %21, %24 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %27 = neura.grant_predicate %5, %24 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %28 = neura.grant_predicate %21, %25 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %29 = "neura.icmp"(%26, %27) <{cmpType = "sge"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %30 = "neura.not"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %31 = "neura.and"(%24, %30) : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %32 = "neura.and"(%24, %29) : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %33 = neura.grant_predicate %21, %29 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %34 = neura.grant_predicate %11, %29 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %35 = neura.grant_predicate %21, %30 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %36 = "neura.add"(%33, %34) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %37 = "neura.or"(%32, %31) : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %38 = "neura.phi"(%36, %35) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %39 = neura.grant_predicate %1, %37 : !neura.data<memref<?xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %40 = neura.grant_predicate %7, %37 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %41 = neura.grant_predicate %9, %37 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %42 = neura.grant_predicate %13, %37 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %43 = neura.grant_predicate %15, %37 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %44 = neura.load_indexed %39[%23 : !neura.data<i64, i1>] !neura.data<memref<?xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %45 = "neura.mul"(%44, %40) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %46 = "neura.add"(%45, %41) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %47 = "neura.add"(%38, %46) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %48 = "neura.add"(%23, %42) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %48 -> %22 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %47 -> %20 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %43 -> %18 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     "neura.return"(%28) : (!neura.data<i32, i1>) -> ()
// CTRL2DATA-NEXT:   }

// FUSE-MAPPING:        func.func @_Z11for_with_ifPi(%arg0: memref<?xi32>) -> i32 attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 13 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 11 : i32, res_mii = 3 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
// FUSE-MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = "%arg0"}> {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 1 : i32}]} : () -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %1 = "neura.grant_once"() <{constant_value = 0 : i32}> {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %2 = "neura.grant_once"() <{constant_value = 1000 : i32}> {mapping_locs = [{id = 4 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}]} : () -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %3 = "neura.grant_once"() <{constant_value = 2 : i32}> {mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 8 : i32, x = 0 : i32, y = 2 : i32}]} : () -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %4 = "neura.grant_once"() <{constant_value = 1 : i32}> {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 9 : i32, x = 2 : i32, y = 1 : i32}]} : () -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %5 = "neura.grant_once"() <{constant_value = -5 : i32}> {mapping_locs = [{id = 12 : i32, resource = "tile", time_step = 8 : i32, x = 0 : i32, y = 3 : i32}]} : () -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %6 = "neura.grant_once"() <{constant_value = 1 : i64}> {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 10 : i32, x = 1 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %7 = "neura.grant_once"() <{constant_value = 128 : i64}> {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %8 = "neura.grant_once"() <{constant_value = 0 : i64}> {mapping_locs = [{id = 4 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 1 : i32}]} : () -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %9 = neura.reserve : !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %10 = "neura.data_mov"(%7) {mapping_locs = [{id = 1 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %11 = "neura.phi"(%9, %10) {mapping_locs = [{id = 4 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %12 = neura.reserve : !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %13 = "neura.data_mov"(%1) {mapping_locs = []} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %14 = "neura.phi"(%12, %13) {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %15 = neura.reserve : !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %16 = "neura.data_mov"(%8) {mapping_locs = [{id = 10 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %17 = "neura.phi"(%15, %16) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %18 = "neura.data_mov"(%17) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %19 = "neura.data_mov"(%11) {mapping_locs = [{id = 10 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %20 = "neura.icmp"(%18, %19) <{cmpType = "slt"}> {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %21 = "neura.data_mov"(%20) {mapping_locs = [{id = 16 : i32, resource = "link", time_step = 2 : i32}, {id = 28 : i32, resource = "link", time_step = 3 : i32}, {id = 40 : i32, resource = "register", time_step = 4 : i32}, {id = 40 : i32, resource = "register", time_step = 5 : i32}, {id = 40 : i32, resource = "register", time_step = 6 : i32}, {id = 40 : i32, resource = "register", time_step = 7 : i32}, {id = 40 : i32, resource = "register", time_step = 8 : i32}, {id = 40 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %22 = "neura.not"(%21) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 10 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %23 = "neura.data_mov"(%14) {mapping_locs = [{id = 1 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %24 = "neura.data_mov"(%20) {mapping_locs = [{id = 13 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %25 = neura.grant_predicate %23, %24 {mapping_locs = [{id = 4 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %26 = "neura.data_mov"(%2) {mapping_locs = [{id = 10 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %27 = "neura.data_mov"(%20) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %28 = neura.grant_predicate %26, %27 {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %29 = "neura.data_mov"(%14) {mapping_locs = []} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %30 = "neura.data_mov"(%22) {mapping_locs = [{id = 31 : i32, resource = "link", time_step = 10 : i32}, {id = 27 : i32, resource = "link", time_step = 11 : i32}, {id = 25 : i32, resource = "link", time_step = 12 : i32}, {id = 11 : i32, resource = "link", time_step = 13 : i32}, {id = 0 : i32, resource = "register", time_step = 14 : i32}, {id = 0 : i32, resource = "register", time_step = 15 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %31 = neura.grant_predicate %29, %30 {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 16 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %32 = "neura.data_mov"(%25) {mapping_locs = [{id = 10 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %33 = "neura.data_mov"(%28) {mapping_locs = []} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %34 = "neura.icmp"(%32, %33) <{cmpType = "sge"}> {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %35 = "neura.data_mov"(%34) {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %36 = "neura.not"(%35) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %37 = "neura.data_mov"(%20) {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 2 : i32}, {id = 24 : i32, resource = "register", time_step = 3 : i32}, {id = 24 : i32, resource = "register", time_step = 4 : i32}, {id = 24 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %38 = "neura.data_mov"(%36) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %39 = "neura.and"(%37, %38) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %40 = "neura.data_mov"(%20) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %41 = "neura.data_mov"(%34) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %42 = "neura.and"(%40, %41) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %43 = "neura.data_mov"(%14) {mapping_locs = [{id = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 4 : i32, resource = "link", time_step = 3 : i32}, {id = 20 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %44 = "neura.data_mov"(%34) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %45 = neura.grant_predicate %43, %44 {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %46 = "neura.data_mov"(%5) {mapping_locs = [{id = 38 : i32, resource = "link", time_step = 8 : i32}, {id = 42 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %47 = "neura.data_mov"(%34) {mapping_locs = [{id = 16 : i32, resource = "link", time_step = 4 : i32}, {id = 36 : i32, resource = "register", time_step = 5 : i32}, {id = 36 : i32, resource = "register", time_step = 6 : i32}, {id = 36 : i32, resource = "register", time_step = 7 : i32}, {id = 36 : i32, resource = "register", time_step = 8 : i32}, {id = 36 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %48 = neura.grant_predicate %46, %47 {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 10 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %49 = "neura.data_mov"(%14) {mapping_locs = []} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %50 = "neura.data_mov"(%36) {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 5 : i32}, {id = 13 : i32, resource = "link", time_step = 6 : i32}, {id = 11 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %51 = neura.grant_predicate %49, %50 {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 8 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %52 = "neura.data_mov"(%45) {mapping_locs = []} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %53 = "neura.data_mov"(%48) {mapping_locs = [{id = 29 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %54 = "neura.add"(%52, %53) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 11 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %55 = "neura.data_mov"(%42) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %56 = "neura.data_mov"(%39) {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %57 = "neura.or"(%55, %56) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %58 = "neura.data_mov"(%54) {mapping_locs = []} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %59 = "neura.data_mov"(%51) {mapping_locs = [{id = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 4 : i32, resource = "link", time_step = 9 : i32}, {id = 20 : i32, resource = "register", time_step = 10 : i32}, {id = 20 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %60 = "neura.phi"(%58, %59) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 12 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %61 = "neura.data_mov"(%0) {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %62 = "neura.data_mov"(%57) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %63 = neura.grant_predicate %61, %62 {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<memref<?xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %64 = "neura.data_mov"(%3) {mapping_locs = [{id = 24 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %65 = "neura.data_mov"(%57) {mapping_locs = [{id = 16 : i32, resource = "link", time_step = 7 : i32}, {id = 37 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %66 = neura.grant_predicate %64, %65 {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %67 = "neura.data_mov"(%4) {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %68 = "neura.data_mov"(%57) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %69 = neura.grant_predicate %67, %68 {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 10 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %70 = "neura.data_mov"(%6) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %71 = "neura.data_mov"(%57) {mapping_locs = [{id = 15 : i32, resource = "link", time_step = 7 : i32}, {id = 4 : i32, resource = "register", time_step = 8 : i32}, {id = 4 : i32, resource = "register", time_step = 9 : i32}, {id = 4 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %72 = neura.grant_predicate %70, %71 {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 11 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %73 = "neura.data_mov"(%7) {mapping_locs = [{id = 0 : i32, resource = "link", time_step = 0 : i32}, {id = 5 : i32, resource = "register", time_step = 1 : i32}, {id = 5 : i32, resource = "register", time_step = 2 : i32}, {id = 5 : i32, resource = "register", time_step = 3 : i32}, {id = 5 : i32, resource = "register", time_step = 4 : i32}, {id = 5 : i32, resource = "register", time_step = 5 : i32}, {id = 5 : i32, resource = "register", time_step = 6 : i32}, {id = 5 : i32, resource = "register", time_step = 7 : i32}, {id = 5 : i32, resource = "register", time_step = 8 : i32}, {id = 5 : i32, resource = "register", time_step = 9 : i32}, {id = 5 : i32, resource = "register", time_step = 10 : i32}, {id = 5 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %74 = "neura.data_mov"(%57) {mapping_locs = [{id = 13 : i32, resource = "link", time_step = 7 : i32}, {id = 11 : i32, resource = "link", time_step = 8 : i32}, {id = 0 : i32, resource = "link", time_step = 9 : i32}, {id = 6 : i32, resource = "register", time_step = 10 : i32}, {id = 6 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-MAPPING-NEXT:     %75 = neura.grant_predicate %73, %74 {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 12 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %76 = "neura.data_mov"(%63) {mapping_locs = []} : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// FUSE-MAPPING-NEXT:     %77 = "neura.data_mov"(%17) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %78 = neura.load_indexed %76[%77 : !neura.data<i64, i1>] !neura.data<memref<?xi32>, i1> {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %79 = "neura.data_mov"(%78) {mapping_locs = [{id = 16 : i32, resource = "link", time_step = 9 : i32}, {id = 36 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %80 = "neura.data_mov"(%66) {mapping_locs = []} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %81 = "neura.mul"(%79, %80) {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 11 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %82 = "neura.data_mov"(%81) {mapping_locs = []} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %83 = "neura.data_mov"(%69) {mapping_locs = [{id = 16 : i32, resource = "link", time_step = 10 : i32}, {id = 36 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %84 = "neura.add"(%82, %83) {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 12 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %85 = "neura.data_mov"(%60) {mapping_locs = [{id = 15 : i32, resource = "link", time_step = 12 : i32}, {id = 4 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %86 = "neura.data_mov"(%84) {mapping_locs = [{id = 29 : i32, resource = "link", time_step = 12 : i32}, {id = 15 : i32, resource = "link", time_step = 13 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %87 = "neura.add"(%85, %86) {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 14 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     %88 = "neura.data_mov"(%17) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %89 = "neura.data_mov"(%72) {mapping_locs = [{id = 4 : i32, resource = "link", time_step = 11 : i32}, {id = 20 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %90 = "neura.add"(%88, %89) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 13 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     neura.ctrl_mov %90 -> %15 {mapping_locs = []} : !neura.data<i64, i1> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     neura.ctrl_mov %87 -> %12 {mapping_locs = [{id = 2 : i32, resource = "link", time_step = 14 : i32}]} : !neura.data<i32, i1> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     neura.ctrl_mov %75 -> %9 {mapping_locs = [{id = 4 : i32, resource = "link", time_step = 12 : i32}, {id = 13 : i32, resource = "link", time_step = 13 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// FUSE-MAPPING-NEXT:     %91 = "neura.data_mov"(%31) {mapping_locs = []} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-MAPPING-NEXT:     "neura.return"(%91) {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 17 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>) -> ()
// FUSE-MAPPING-NEXT:   }