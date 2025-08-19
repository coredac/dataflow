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
// RUN: --fuse-control-flow \
// RUN: --fold-constant \
// RUN: --insert-data-mov \
// RUN: --map-to-accelerator="mapping-strategy=heuristic mapping-mode=spatial-only backtrack-config=customized=4,3" \
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
// RUN: --fuse-control-flow \
// RUN: --fold-constant \
// RUN: --insert-data-mov \
// RUN: --map-to-accelerator="mapping-strategy=heuristic mapping-mode=spatial-temporal backtrack-config=customized=3,4" \
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

// CHECK:        func.func @simple_add_loop() -> i64 attributes {accelerator = "neura"} {
// CHECK-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 16 : i64}> : () -> i64
// CHECK-NEXT:     %1 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> i64
// CHECK-NEXT:     %2 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> i64
// CHECK-NEXT:     %3 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> i64
// CHECK-NEXT:     neura.br %3, %2, %0 : i64, i64, i64 to ^bb1
// CHECK-NEXT:   ^bb1(%4: i64, %5: i64, %6: i64):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %7 = "neura.icmp"(%4, %6) <{cmpType = "slt"}> : (i64, i64) -> i1
// CHECK-NEXT:     neura.cond_br %7 : i1 then %5, %4, %1, %0 : i64, i64, i64, i64 to ^bb2 else %5 : i64 to ^bb3
// CHECK-NEXT:   ^bb2(%8: i64, %9: i64, %10: i64, %11: i64):  // pred: ^bb1
// CHECK-NEXT:     %12 = "neura.add"(%8, %8) : (i64, i64) -> i64
// CHECK-NEXT:     %13 = "neura.add"(%9, %10) : (i64, i64) -> i64
// CHECK-NEXT:     neura.br %13, %12, %11 : i64, i64, i64 to ^bb1
// CHECK-NEXT:   ^bb3(%14: i64):  // pred: ^bb1
// CHECK-NEXT:     "neura.return"(%14) : (i64) -> ()
// CHECK-NEXT:   }

// SPATIAL:      func.func @simple_add_loop() -> i64 attributes {accelerator = "neura", mapping_info = {compiled_ii = 7 : i32, mapping_mode = "spatial-only", mapping_strategy = "heuristic", rec_mii = 3 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
// SPATIAL-NEXT:   %0 = "neura.grant_always"() <{constant_value = 16 : i64}> {mapping_locs = [{id = 2 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// SPATIAL-NEXT:   %1 = "neura.grant_always"() <{constant_value = 1 : i64}> {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// SPATIAL-NEXT:   %2 = "neura.grant_once"() <{constant_value = 1 : i64}> {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// SPATIAL-NEXT:   %3 = "neura.grant_once"() <{constant_value = 0 : i64}> {mapping_locs = [{id = 11 : i32, resource = "tile", time_step = 0 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// SPATIAL-NEXT:   %4 = "neura.grant_always"() <{constant_value = true}> {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 2 : i32}]} : () -> !neura.data<i1, i1>
// SPATIAL-NEXT:   %5 = "neura.data_mov"(%4) {mapping_locs = [{id = 33 : i32, resource = "link", time_step = 0 : i32}, {id = 24 : i32, resource = "register", time_step = 1 : i32}, {id = 24 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// SPATIAL-NEXT:   %6 = "neura.data_mov"(%3) {mapping_locs = [{id = 36 : i32, resource = "link", time_step = 0 : i32}, {id = 21 : i32, resource = "link", time_step = 1 : i32}, {id = 25 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-NEXT:   %7 = "neura.data_mov"(%0) {mapping_locs = [{id = 7 : i32, resource = "link", time_step = 0 : i32}, {id = 26 : i32, resource = "register", time_step = 1 : i32}, {id = 26 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-NEXT:   %8 = "neura.data_mov"(%1) {mapping_locs = [{id = 0 : i32, resource = "link", time_step = 0 : i32}, {id = 4 : i32, resource = "link", time_step = 1 : i32}, {id = 14 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-NEXT:   %nextindex, %valid = neura.loop_control(parent_valid = %5, start = %6, end = %7, step = %8) {iterationType = "increment", mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>, !neura.data<i1, i1>
// SPATIAL-NEXT:   %9 = "neura.data_mov"(%valid) {mapping_locs = [{id = 18 : i32, resource = "link", time_step = 3 : i32}, {id = 22 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// SPATIAL-NEXT:   %10 = "neura.not"(%9) {mapping_locs = [{id = 3 : i32, resource = "tile", time_step = 5 : i32, x = 3 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// SPATIAL-NEXT:   %11 = neura.reserve : !neura.data<i64, i1>
// SPATIAL-NEXT:   %12 = "neura.data_mov"(%2) {mapping_locs = [{id = 2 : i32, resource = "link", time_step = 1 : i32}, {id = 1 : i32, resource = "link", time_step = 2 : i32}, {id = 10 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-NEXT:   %13 = "neura.phi"(%11, %12) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-NEXT:   %14 = "neura.data_mov"(%13) {mapping_locs = [{id = 16 : i32, resource = "link", time_step = 4 : i32}, {id = 36 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-NEXT:   %15 = "neura.data_mov"(%valid) {mapping_locs = [{id = 20 : i32, resource = "link", time_step = 3 : i32}, {id = 31 : i32, resource = "link", time_step = 4 : i32}, {id = 37 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// SPATIAL-NEXT:   %16 = neura.grant_predicate %14, %15 {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// SPATIAL-NEXT:   %17 = "neura.data_mov"(%13) {mapping_locs = [{id = 13 : i32, resource = "link", time_step = 4 : i32}, {id = 12 : i32, resource = "link", time_step = 5 : i32}, {id = 24 : i32, resource = "link", time_step = 6 : i32}, {id = 28 : i32, resource = "link", time_step = 7 : i32}, {id = 34 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-NEXT:   %18 = "neura.data_mov"(%10) {mapping_locs = [{id = 9 : i32, resource = "link", time_step = 5 : i32}, {id = 23 : i32, resource = "link", time_step = 6 : i32}, {id = 37 : i32, resource = "link", time_step = 7 : i32}, {id = 46 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// SPATIAL-NEXT:   %19 = neura.grant_predicate %17, %18 {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 9 : i32, x = 2 : i32, y = 3 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// SPATIAL-NEXT:   %20 = "neura.data_mov"(%16) {mapping_locs = [{id = 30 : i32, resource = "link", time_step = 6 : i32}, {id = 52 : i32, resource = "register", time_step = 7 : i32}, {id = 52 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-NEXT:   %21 = "neura.data_mov"(%16) {mapping_locs = [{id = 27 : i32, resource = "link", time_step = 6 : i32}, {id = 26 : i32, resource = "link", time_step = 7 : i32}, {id = 38 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-NEXT:   %22 = "neura.add"(%20, %21) {mapping_locs = [{id = 13 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 3 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-NEXT:   neura.ctrl_mov %22 -> %11 {mapping_locs = [{id = 42 : i32, resource = "link", time_step = 9 : i32}, {id = 29 : i32, resource = "link", time_step = 10 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// SPATIAL-NEXT:   %23 = "neura.data_mov"(%19) {mapping_locs = [{id = 44 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-NEXT:   "neura.return"(%23) {mapping_locs = [{id = 15 : i32, resource = "tile", time_step = 10 : i32, x = 3 : i32, y = 3 : i32}]} : (!neura.data<i64, i1>) -> ()
// SPATIAL-NEXT: }

// SPATIAL-TEMPORAL:        func.func @simple_add_loop() -> i64 attributes {accelerator = "neura", mapping_info = {compiled_ii = 3 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 3 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
// SPATIAL-TEMPORAL-NEXT:     %0 = "neura.grant_always"() <{constant_value = 16 : i64}> {mapping_locs = [{id = 4 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 1 : i32}]} : () -> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     %1 = "neura.grant_always"() <{constant_value = 1 : i64}> {mapping_locs = [{id = 12 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 3 : i32}]} : () -> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     %2 = "neura.grant_once"() <{constant_value = 1 : i64}> {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     %3 = "neura.grant_once"() <{constant_value = 0 : i64}> {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     %4 = "neura.grant_always"() <{constant_value = true}> {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 2 : i32}]} : () -> !neura.data<i1, i1>
// SPATIAL-TEMPORAL-NEXT:     %5 = "neura.data_mov"(%4) {mapping_locs = [{id = 27 : i32, resource = "link", time_step = 0 : i32}, {id = 32 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// SPATIAL-TEMPORAL-NEXT:     %6 = "neura.data_mov"(%3) {mapping_locs = [{id = 1 : i32, resource = "link", time_step = 0 : i32}, {id = 12 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     %7 = "neura.data_mov"(%0) {mapping_locs = [{id = 12 : i32, resource = "link", time_step = 0 : i32}, {id = 33 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     %8 = "neura.data_mov"(%1) {mapping_locs = [{id = 39 : i32, resource = "link", time_step = 0 : i32}, {id = 34 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     %nextindex, %valid = neura.loop_control(parent_valid = %5, start = %6, end = %7, step = %8) {iterationType = "increment", mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>, !neura.data<i1, i1>
// SPATIAL-TEMPORAL-NEXT:     %9 = "neura.data_mov"(%valid) {mapping_locs = [{id = 25 : i32, resource = "link", time_step = 2 : i32}, {id = 16 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// SPATIAL-TEMPORAL-NEXT:     %10 = "neura.not"(%9) {mapping_locs = [{id = 4 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// SPATIAL-TEMPORAL-NEXT:     %11 = neura.reserve : !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     %12 = "neura.data_mov"(%2) {mapping_locs = [{id = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     %13 = "neura.phi"(%11, %12) {mapping_locs = [{id = 4 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     %14 = "neura.data_mov"(%13) {mapping_locs = [{id = 12 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     %15 = "neura.data_mov"(%valid) {mapping_locs = [{id = 32 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// SPATIAL-TEMPORAL-NEXT:     %16 = neura.grant_predicate %14, %15 {mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     %17 = "neura.data_mov"(%13) {mapping_locs = [{id = 11 : i32, resource = "link", time_step = 2 : i32}, {id = 0 : i32, resource = "register", time_step = 3 : i32}, {id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     %18 = "neura.data_mov"(%10) {mapping_locs = [{id = 11 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// SPATIAL-TEMPORAL-NEXT:     %19 = neura.grant_predicate %17, %18 {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 5 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     %20 = "neura.data_mov"(%16) {mapping_locs = [{id = 32 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     %21 = "neura.data_mov"(%16) {mapping_locs = [{id = 33 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     %22 = "neura.add"(%20, %21) {mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     neura.ctrl_mov %22 -> %11 {mapping_locs = [{id = 25 : i32, resource = "link", time_step = 4 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     %23 = "neura.data_mov"(%19) {mapping_locs = [{id = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// SPATIAL-TEMPORAL-NEXT:     "neura.return"(%23) {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>) -> ()
// SPATIAL-TEMPORAL-NEXT:   }