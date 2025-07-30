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
// RUN: --canonicalize-cast | FileCheck %s --check-prefix=CAST

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
// RUN: --fuse-control-flow | FileCheck %s -check-prefix=CTRLFUSE

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
// RUN: --insert-data-mov \
// RUN: --map-to-accelerator="mapping-strategy=heuristic" | FileCheck %s -check-prefix=CTRLFUSE-MAPPING

module attributes {} {
  func.func @_Z10simpleloopv() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.for %arg0 = 0 to 128 iter_args(%arg1 = %c0_i32) -> (i32) {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = arith.addi %arg1, %1 : i32
      affine.yield %2 : i32
    }
    return %0 : i32
  }
}

// CHECK: func.func @_Z10simpleloopv() -> i32 attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 1 : index}> : () -> index
// CHECK-NEXT:     %1 = "neura.constant"() <{predicate = true, value = 128 : index}> : () -> index
// CHECK-NEXT:     %2 = "neura.constant"() <{predicate = true, value = 0 : i32}> : () -> i32
// CHECK-NEXT:     %3 = "neura.constant"() <{predicate = true, value = 0 : index}> : () -> index
// CHECK-NEXT:     %4 = "neura.cast"(%3) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %4, %2 : i64, i32 to ^bb1
// CHECK-NEXT:   ^bb1(%5: i64, %6: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %7 = "neura.cast"(%5) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %8 = "neura.icmp"(%7, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %8 : i1 then to ^bb2 else to ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %9 = "neura.cast"(%7) <{cast_type = "index_to_int"}> : (index) -> i32
// CHECK-NEXT:     %10 = "neura.add"(%6, %9) : (i32, i32) -> i32
// CHECK-NEXT:     %11 = "neura.add"(%7, %0) : (index, index) -> index
// CHECK-NEXT:     %12 = "neura.cast"(%11) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %12, %10 : i64, i32 to ^bb1
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     "neura.return"(%6) : (i32) -> ()
// CHECK-NEXT:   }

// CAST:     func.func @_Z10simpleloopv() -> i32 attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CAST-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> i64
// CAST-NEXT:     %1 = "neura.constant"() <{predicate = true, value = 128 : i64}> : () -> i64
// CAST-NEXT:     %2 = "neura.constant"() <{predicate = true, value = 0 : i32}> : () -> i32
// CAST-NEXT:     %3 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> i64
// CAST-NEXT:     neura.br %3, %2 : i64, i32 to ^bb1
// CAST-NEXT:   ^bb1(%4: i64, %5: i32):  // 2 preds: ^bb0, ^bb2
// CAST-NEXT:     %6 = "neura.icmp"(%4, %1) <{cmpType = "slt"}> : (i64, i64) -> i1
// CAST-NEXT:     neura.cond_br %6 : i1 then to ^bb2 else to ^bb3
// CAST-NEXT:   ^bb2:  // pred: ^bb1
// CAST-NEXT:     %7 = "neura.cast"(%4) <{cast_type = "i64_to_i32"}> : (i64) -> i32
// CAST-NEXT:     %8 = "neura.add"(%5, %7) : (i32, i32) -> i32
// CAST-NEXT:     %9 = "neura.add"(%4, %0) : (i64, i64) -> i64
// CAST-NEXT:     neura.br %9, %8 : i64, i32 to ^bb1
// CAST-NEXT:   ^bb3:  // pred: ^bb1
// CAST-NEXT:     "neura.return"(%5) : (i32) -> ()
// CAST-NEXT:   }

// CTRL2DATA: func.func @_Z10simpleloopv() -> i32 attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{predicate = true, value = 128 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{predicate = true, value = 0 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %8 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %9 = "neura.phi"(%8, %3) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %10 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %11 = "neura.phi"(%10, %5) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %12 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %13 = "neura.phi"(%12, %7) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %14 = "neura.icmp"(%13, %9) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %15 = neura.grant_predicate %13, %14 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %16 = neura.grant_predicate %11, %14 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %17 = neura.grant_predicate %1, %14 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %18 = neura.grant_predicate %3, %14 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %19 = "neura.not"(%14) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %20 = neura.grant_predicate %11, %19 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %21 = "neura.cast"(%15) <{cast_type = "i64_to_i32"}> : (!neura.data<i64, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %22 = "neura.add"(%16, %21) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %23 = "neura.add"(%15, %17) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %23 -> %12 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %22 -> %10 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %18 -> %8 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     "neura.return"(%20) : (!neura.data<i32, i1>) -> ()
// CTRL2DATA-NEXT:   }


// CTRLFUSE:     func.func @_Z10simpleloopv() -> i32 attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CTRLFUSE-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRLFUSE-NEXT:     %1 = "neura.constant"() <{predicate = true, value = 128 : i64}> : () -> !neura.data<i64, i1>
// CTRLFUSE-NEXT:     %2 = "neura.grant_always"(%1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRLFUSE-NEXT:     %3 = "neura.grant_always"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRLFUSE-NEXT:     %4 = "neura.constant"() <{predicate = true, value = 0 : i32}> : () -> !neura.data<i32, i1>
// CTRLFUSE-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRLFUSE-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRLFUSE-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRLFUSE-NEXT:     %8 = "neura.constant"() <{predicate = true, value = true}> : () -> !neura.data<i1, i1>
// CTRLFUSE-NEXT:     %nextindex, %valid = neura.loop_control(parent_valid = %8, start = %7, end = %2, step = %3) {iterationType = "increment"} : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>, !neura.data<i1, i1>
// CTRLFUSE-NEXT:     %9 = "neura.not"(%valid) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRLFUSE-NEXT:     %10 = neura.reserve : !neura.data<i32, i1>
// CTRLFUSE-NEXT:     %11 = "neura.phi"(%10, %5) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRLFUSE-NEXT:     %12 = neura.grant_predicate %11, %valid : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRLFUSE-NEXT:     %13 = neura.grant_predicate %11, %9 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRLFUSE-NEXT:     %14 = "neura.cast"(%nextindex) <{cast_type = "i64_to_i32"}> : (!neura.data<i64, i1>) -> !neura.data<i32, i1>
// CTRLFUSE-NEXT:     %15 = "neura.add"(%12, %14) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRLFUSE-NEXT:     neura.ctrl_mov %15 -> %10 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRLFUSE-NEXT:     "neura.return"(%13) : (!neura.data<i32, i1>) -> ()
// CTRLFUSE-NEXT:   }


// CTRLFUSE-MAPPING:   func.func @_Z10simpleloopv() -> i32 attributes {CompiledII = 3 : i32, RecMII = 3 : i32, ResMII = 2 : i32, accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CTRLFUSE-MAPPING-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 1 : i64}> {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 1 : i32}]} : () -> !neura.data<i64, i1>
// CTRLFUSE-MAPPING-NEXT:     %1 = "neura.constant"() <{predicate = true, value = 128 : i64}> {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 1 : i32}]} : () -> !neura.data<i64, i1>
// CTRLFUSE-MAPPING-NEXT:     %2 = "neura.data_mov"(%1) {mapping_locs = [{id = 16 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRLFUSE-MAPPING-NEXT:     %3 = "neura.grant_always"(%2) {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRLFUSE-MAPPING-NEXT:     %4 = "neura.data_mov"(%0) {mapping_locs = [{id = 20 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRLFUSE-MAPPING-NEXT:     %5 = "neura.grant_always"(%4) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRLFUSE-MAPPING-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 0 : i32}> {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 2 : i32}]} : () -> !neura.data<i32, i1>
// CTRLFUSE-MAPPING-NEXT:     %7 = "neura.data_mov"(%6) {mapping_locs = [{id = 34 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRLFUSE-MAPPING-NEXT:     %8 = "neura.grant_once"(%7) {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRLFUSE-MAPPING-NEXT:     %9 = "neura.constant"() <{predicate = true, value = 0 : i64}> {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// CTRLFUSE-MAPPING-NEXT:     %10 = "neura.data_mov"(%9) {mapping_locs = [{id = 29 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRLFUSE-MAPPING-NEXT:     %11 = "neura.grant_once"(%10) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRLFUSE-MAPPING-NEXT:     %12 = "neura.constant"() <{predicate = true, value = true}> {mapping_locs = [{id = 13 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 3 : i32}]} : () -> !neura.data<i1, i1>
// CTRLFUSE-MAPPING-NEXT:     %13 = "neura.data_mov"(%12) {mapping_locs = [{id = 42 : i32, resource = "link", time_step = 0 : i32}, {id = 36 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRLFUSE-MAPPING-NEXT:     %14 = "neura.data_mov"(%11) {mapping_locs = [{id = 16 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRLFUSE-MAPPING-NEXT:     %15 = "neura.data_mov"(%3) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRLFUSE-MAPPING-NEXT:     %16 = "neura.data_mov"(%5) {mapping_locs = [{id = 31 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRLFUSE-MAPPING-NEXT:     %nextindex, %valid = neura.loop_control(parent_valid = %13, start = %14, end = %15, step = %16) {iterationType = "increment", mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>, !neura.data<i1, i1>
// CTRLFUSE-MAPPING-NEXT:     %17 = "neura.data_mov"(%valid) {mapping_locs = [{id = 28 : i32, resource = "link", time_step = 2 : i32}, {id = 40 : i32, resource = "register", time_step = 3 : i32}, {id = 40 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRLFUSE-MAPPING-NEXT:     %18 = "neura.not"(%17) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRLFUSE-MAPPING-NEXT:     %19 = neura.reserve : !neura.data<i32, i1>
// CTRLFUSE-MAPPING-NEXT:     %20 = "neura.data_mov"(%8) {mapping_locs = []} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRLFUSE-MAPPING-NEXT:     %21 = "neura.phi"(%19, %20) {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRLFUSE-MAPPING-NEXT:     %22 = "neura.data_mov"(%21) {mapping_locs = [{id = 45 : i32, resource = "link", time_step = 2 : i32}, {id = 33 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRLFUSE-MAPPING-NEXT:     %23 = "neura.data_mov"(%valid) {mapping_locs = [{id = 29 : i32, resource = "link", time_step = 2 : i32}, {id = 14 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRLFUSE-MAPPING-NEXT:     %24 = neura.grant_predicate %22, %23 {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRLFUSE-MAPPING-NEXT:     %25 = "neura.data_mov"(%21) {mapping_locs = [{id = 44 : i32, resource = "link", time_step = 2 : i32}, {id = 47 : i32, resource = "link", time_step = 3 : i32}, {id = 44 : i32, resource = "register", time_step = 4 : i32}, {id = 44 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRLFUSE-MAPPING-NEXT:     %26 = "neura.data_mov"(%18) {mapping_locs = [{id = 32 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRLFUSE-MAPPING-NEXT:     %27 = neura.grant_predicate %25, %26 {mapping_locs = [{id = 11 : i32, resource = "tile", time_step = 6 : i32, x = 3 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRLFUSE-MAPPING-NEXT:     %28 = "neura.data_mov"(%nextindex) {mapping_locs = [{id = 27 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRLFUSE-MAPPING-NEXT:     %29 = "neura.cast"(%28) <{cast_type = "i64_to_i32"}> {mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i32, i1>
// CTRLFUSE-MAPPING-NEXT:     %30 = "neura.data_mov"(%24) {mapping_locs = [{id = 20 : i32, resource = "link", time_step = 4 : i32}, {id = 34 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRLFUSE-MAPPING-NEXT:     %31 = "neura.data_mov"(%29) {mapping_locs = [{id = 24 : i32, resource = "link", time_step = 3 : i32}, {id = 30 : i32, resource = "link", time_step = 4 : i32}, {id = 41 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRLFUSE-MAPPING-NEXT:     %32 = "neura.add"(%30, %31) {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRLFUSE-MAPPING-NEXT:     neura.ctrl_mov %32 -> %19 {mapping_locs = []} : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRLFUSE-MAPPING-NEXT:     %33 = "neura.data_mov"(%27) {mapping_locs = [{id = 35 : i32, resource = "link", time_step = 6 : i32}, {id = 33 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRLFUSE-MAPPING-NEXT:     "neura.return"(%33) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 8 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>) -> ()
// CTRLFUSE-MAPPING-NEXT:   }