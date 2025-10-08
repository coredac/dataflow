// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
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
// RUN: --transform-to-steer-control \
// RUN: --remove-predicated-type \
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
// RUN: --transform-to-steer-control \
// RUN: --remove-predicated-type \
// RUN: --insert-data-mov \
// RUN: --map-to-accelerator="mapping-strategy=heuristic mapping-mode=spatial-only backtrack-config=customized sort-strategy=topological" \
// RUN: | FileCheck %s -check-prefix=MAPPING

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

// CHECK:      func.func @simple_add_loop() -> i64 attributes {accelerator = "neura", dataflow_mode = "steering"} {
// CHECK-NEXT:   %0 = neura.reserve : i64
// CHECK-NEXT:   %1 = neura.reserve : i64
// CHECK-NEXT:   %2 = neura.reserve : i1
// CHECK-NEXT:   %3 = "neura.constant"() <{value = 16 : i64}> : () -> i64
// CHECK-NEXT:   %4 = "neura.constant"() <{value = 1 : i64}> : () -> i64
// CHECK-NEXT:   %5 = "neura.constant"() <{value = 1 : i64}> : () -> i64
// CHECK-NEXT:   %6 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CHECK-NEXT:   %7 = neura.invariant %4, %2 : i64, i1 -> i64
// CHECK-NEXT:   %8 = neura.invariant %3, %2 : i64, i1 -> i64
// CHECK-NEXT:   %9 = neura.carry %5, %2, %0 : i64, i1, i64 -> i64
// CHECK-NEXT:   %10 = neura.carry %6, %2, %1 : i64, i1, i64 -> i64
// CHECK-NEXT:   %11 = "neura.icmp"(%10, %8) <{cmpType = "slt"}> : (i64, i64) -> i1
// CHECK-NEXT:   neura.ctrl_mov %11 -> %2 : i1 i1
// CHECK-NEXT:   %12 = neura.false_steer %9, %11 : i64, i1 -> i64
// CHECK-NEXT:   %13 = "neura.add"(%9, %9) : (i64, i64) -> i64
// CHECK-NEXT:   neura.ctrl_mov %13 -> %0 : i64 i64
// CHECK-NEXT:   %14 = "neura.add"(%10, %7) : (i64, i64) -> i64
// CHECK-NEXT:   neura.ctrl_mov %14 -> %1 : i64 i64
// CHECK-NEXT:   "neura.return"(%12) : (i64) -> ()
// CHECK-NEXT: }

// MAPPING:      func.func @simple_add_loop() -> i64 attributes {accelerator = "neura", dataflow_mode = "steering", mapping_info = {compiled_ii = 5 : i32, mapping_mode = "spatial-only", mapping_strategy = "heuristic", rec_mii = 2 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
// MAPPING-NEXT:   %0 = neura.reserve : i64
// MAPPING-NEXT:   %1 = neura.reserve : i64
// MAPPING-NEXT:   %2 = neura.reserve : i1
// MAPPING-NEXT:   %3 = "neura.constant"() <{value = 16 : i64}> {mapping_locs = [{id = 11 : i32, resource = "tile", time_step = 0 : i32, x = 3 : i32, y = 2 : i32}]} : () -> i64
// MAPPING-NEXT:   %4 = "neura.constant"() <{value = 1 : i64}> {mapping_locs = [{id = 12 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 3 : i32}]} : () -> i64
// MAPPING-NEXT:   %5 = "neura.constant"() <{value = 1 : i64}> {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> i64
// MAPPING-NEXT:   %6 = "neura.constant"() <{value = 0 : i64}> {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 3 : i32}]} : () -> i64
// MAPPING-NEXT:   %7 = "neura.data_mov"(%4) {mapping_locs = [{id = 38 : i32, resource = "link", time_step = 0 : i32}]} : (i64) -> i64
// MAPPING-NEXT:   %8 = neura.invariant %7, %2 {mapping_locs = [{id = 13 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 3 : i32}]} : i64, i1 -> i64
// MAPPING-NEXT:   %9 = "neura.data_mov"(%3) {mapping_locs = [{id = 35 : i32, resource = "link", time_step = 0 : i32}]} : (i64) -> i64
// MAPPING-NEXT:   %10 = neura.invariant %9, %2 {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 2 : i32}]} : i64, i1 -> i64
// MAPPING-NEXT:   %11 = "neura.data_mov"(%5) {mapping_locs = [{id = 0 : i32, resource = "link", time_step = 0 : i32}]} : (i64) -> i64
// MAPPING-NEXT:   %12 = neura.carry %11, %2, %0 {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 0 : i32}]} : i64, i1, i64 -> i64
// MAPPING-NEXT:   %13 = "neura.data_mov"(%6) {mapping_locs = [{id = 45 : i32, resource = "link", time_step = 0 : i32}, {id = 33 : i32, resource = "link", time_step = 1 : i32}]} : (i64) -> i64
// MAPPING-NEXT:   %14 = neura.carry %13, %2, %1 {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 1 : i32}]} : i64, i1, i64 -> i64
// MAPPING-NEXT:   %15 = "neura.data_mov"(%14) {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 2 : i32}, {id = 16 : i32, resource = "link", time_step = 3 : i32}]} : (i64) -> i64
// MAPPING-NEXT:   %16 = "neura.data_mov"(%10) {mapping_locs = [{id = 31 : i32, resource = "link", time_step = 1 : i32}, {id = 288 : i32, resource = "register", time_step = 2 : i32}, {id = 288 : i32, resource = "register", time_step = 3 : i32}]} : (i64) -> i64
// MAPPING-NEXT:   %17 = "neura.icmp"(%15, %16) <{cmpType = "slt"}> {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 2 : i32}]} : (i64, i64) -> i1
// MAPPING-NEXT:   neura.ctrl_mov %17 -> %2 {mapping_locs = [{id = 30 : i32, resource = "link", time_step = 4 : i32}, {id = 416 : i32, resource = "register", time_step = 5 : i32}]} : i1 i1
// MAPPING-NEXT:   %18 = "neura.data_mov"(%12) {mapping_locs = [{id = 3 : i32, resource = "link", time_step = 1 : i32}, {id = 64 : i32, resource = "register", time_step = 2 : i32}, {id = 64 : i32, resource = "register", time_step = 3 : i32}, {id = 64 : i32, resource = "register", time_step = 4 : i32}, {id = 64 : i32, resource = "register", time_step = 5 : i32}, {id = 64 : i32, resource = "register", time_step = 6 : i32}]} : (i64) -> i64
// MAPPING-NEXT:   %19 = "neura.data_mov"(%17) {mapping_locs = [{id = 29 : i32, resource = "link", time_step = 4 : i32}, {id = 14 : i32, resource = "link", time_step = 5 : i32}, {id = 19 : i32, resource = "link", time_step = 6 : i32}]} : (i1) -> i1
// MAPPING-NEXT:   %20 = neura.false_steer %18, %19 {mapping_locs = [{id = 2 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 0 : i32}]} : i64, i1 -> i64
// MAPPING-NEXT:   %21 = "neura.data_mov"(%12) {mapping_locs = [{id = 4 : i32, resource = "link", time_step = 1 : i32}, {id = 160 : i32, resource = "register", time_step = 2 : i32}, {id = 160 : i32, resource = "register", time_step = 3 : i32}]} : (i64) -> i64
// MAPPING-NEXT:   %22 = "neura.data_mov"(%12) {mapping_locs = [{id = 2 : i32, resource = "link", time_step = 1 : i32}, {id = 1 : i32, resource = "link", time_step = 2 : i32}, {id = 10 : i32, resource = "link", time_step = 3 : i32}]} : (i64) -> i64
// MAPPING-NEXT:   %23 = "neura.add"(%21, %22) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 1 : i32}]} : (i64, i64) -> i64
// MAPPING-NEXT:   neura.ctrl_mov %23 -> %0 {mapping_locs = [{id = 15 : i32, resource = "link", time_step = 4 : i32}, {id = 32 : i32, resource = "register", time_step = 5 : i32}]} : i64 i64
// MAPPING-NEXT:   %24 = "neura.data_mov"(%14) {mapping_locs = [{id = 18 : i32, resource = "link", time_step = 2 : i32}, {id = 224 : i32, resource = "register", time_step = 3 : i32}, {id = 224 : i32, resource = "register", time_step = 4 : i32}]} : (i64) -> i64
// MAPPING-NEXT:   %25 = "neura.data_mov"(%8) {mapping_locs = [{id = 41 : i32, resource = "link", time_step = 1 : i32}, {id = 44 : i32, resource = "link", time_step = 2 : i32}, {id = 47 : i32, resource = "link", time_step = 3 : i32}, {id = 36 : i32, resource = "link", time_step = 4 : i32}]} : (i64) -> i64
// MAPPING-NEXT:   %26 = "neura.add"(%24, %25) {mapping_locs = [{id = 7 : i32, resource = "tile", time_step = 5 : i32, x = 3 : i32, y = 1 : i32}]} : (i64, i64) -> i64
// MAPPING-NEXT:   neura.ctrl_mov %26 -> %1 {mapping_locs = [{id = 21 : i32, resource = "link", time_step = 5 : i32}, {id = 192 : i32, resource = "register", time_step = 6 : i32}]} : i64 i64
// MAPPING-NEXT:   %27 = "neura.data_mov"(%20) {mapping_locs = [{id = 6 : i32, resource = "link", time_step = 7 : i32}]} : (i64) -> i64
// MAPPING-NEXT:   "neura.return"(%27) {mapping_locs = [{id = 3 : i32, resource = "tile", time_step = 8 : i32, x = 3 : i32, y = 0 : i32}]} : (i64) -> ()
