// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --leverage-predicated-value \
// RUN:   | FileCheck %s

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --fold-constant \
// RUN:   --canonicalize-live-in \
// RUN:   | FileCheck %s -check-prefix=CANONICALIZE

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --fold-constant \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   | FileCheck %s -check-prefix=CTRL2DATA

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --fold-constant \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   | FileCheck %s -check-prefix=FUSE

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --fold-constant \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   | FileCheck %s -check-prefix=MOV

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --fold-constant \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic backtrack-config=simple" \
// RUN:   | FileCheck %s -check-prefix=MAPPING

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --fold-constant \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic backtrack-config=simple" \
// RUN:   --generate-code
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml -check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM

func.func @loop_test() -> f32 {
  %n = llvm.mlir.constant(10 : i64) : i64
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c1f = llvm.mlir.constant(3.0 : f32) : f32
  %acc_init = llvm.mlir.constant(0.0 : f32) : f32

  llvm.br ^bb1(%c0, %acc_init : i64, f32)

^bb1(%i: i64, %acc: f32):  // loop body + check + increment
  %next_acc = llvm.fadd %acc, %c1f : f32
  %i_next = llvm.add %i, %c1 : i64
  %cmp = llvm.icmp "slt" %i_next, %n : i64
  llvm.cond_br %cmp, ^bb1(%i_next, %next_acc : i64, f32), ^exit(%next_acc : f32)

^exit(%result: f32):
  return %result : f32
}

// CHECK:      func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// CHECK-NEXT:   %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:   %1 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:   %2 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:   %3 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CHECK-NEXT:   %4 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CHECK-NEXT:   neura.br %1, %4 : !neura.data<i64, i1>, !neura.data<f32, i1> to ^bb1
// CHECK-NEXT: ^bb1(%5: !neura.data<i64, i1>, %6: !neura.data<f32, i1>):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:   %7 = "neura.fadd"(%6, %3) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:   %8 = "neura.add"(%5, %2) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:   %9 = "neura.icmp"(%8, %0) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:   neura.cond_br %9 : !neura.data<i1, i1> then %8, %7 : !neura.data<i64, i1>, !neura.data<f32, i1> to ^bb1 else %7 : !neura.data<f32, i1> to ^bb2
// CHECK-NEXT: ^bb2(%10: !neura.data<f32, i1>):  // pred: ^bb1
// CHECK-NEXT:   "neura.return"(%10) : (!neura.data<f32, i1>) -> ()
// CHECK-NEXT: }

// CANONICALIZE:      func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// CANONICALIZE-NEXT:   %0 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> i64
// CANONICALIZE-NEXT:   %1 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> f32
// CANONICALIZE-NEXT:   %2 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> : () -> f32
// CANONICALIZE-NEXT:   neura.br %0, %2, %1 : i64, f32, f32 to ^bb1
// CANONICALIZE-NEXT: ^bb1(%3: i64, %4: f32, %5: f32):  // 2 preds: ^bb0, ^bb1
// CANONICALIZE-NEXT:   %6 = "neura.fadd"(%4, %5) : (f32, f32) -> f32
// CANONICALIZE-NEXT:   %7 = "neura.add"(%3) {rhs_const_value = 1 : i64} : (i64) -> i64
// CANONICALIZE-NEXT:   %8 = "neura.icmp"(%7) <{cmpType = "slt"}> {rhs_const_value = 10 : i64} : (i64) -> i1
// CANONICALIZE-NEXT:   neura.cond_br %8 : i1 then %7, %6, %5 : i64, f32, f32 to ^bb1 else %6 : f32 to ^bb2
// CANONICALIZE-NEXT: ^bb2(%9: f32):  // pred: ^bb1
// CANONICALIZE-NEXT:   "neura.return"(%9) : (f32) -> ()
// CANONICALIZE-NEXT: }

// CTRL2DATA:        func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %6 = neura.reserve : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %7 = "neura.phi"(%6, %3) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %8 = neura.reserve : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %9 = "neura.phi"(%8, %5) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %10 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %11 = "neura.phi"(%10, %1) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %12 = "neura.fadd"(%9, %7) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %13 = "neura.add"(%11) {rhs_const_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %14 = "neura.icmp"(%13) <{cmpType = "slt"}> {rhs_const_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %15 = neura.grant_predicate %13, %14 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %15 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %16 = neura.grant_predicate %12, %14 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %16 -> %8 : !neura.data<f32, i1> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %17 = neura.grant_predicate %7, %14 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %17 -> %6 : !neura.data<f32, i1> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %18 = "neura.not"(%14) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %19 = neura.grant_predicate %12, %18 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     "neura.return"(%19) : (!neura.data<f32, i1>) -> ()
// CTRL2DATA-NEXT:   }

// FUSE:      func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// FUSE-NEXT:   %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// FUSE-NEXT:   %1 = "neura.grant_once"() <{constant_value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// FUSE-NEXT:   %2 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// FUSE-NEXT:   %3 = neura.reserve : !neura.data<f32, i1>
// FUSE-NEXT:   %4 = "neura.phi"(%3, %1) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// FUSE-NEXT:   %5 = neura.reserve : !neura.data<f32, i1>
// FUSE-NEXT:   %6 = "neura.phi"(%5, %2) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// FUSE-NEXT:   %7 = neura.reserve : !neura.data<i64, i1>
// FUSE-NEXT:   %8 = "neura.phi"(%7, %0) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-NEXT:   %9 = "neura.fadd"(%6, %4) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// FUSE-NEXT:   %10 = "neura.add"(%8) {rhs_const_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-NEXT:   %11 = "neura.icmp"(%10) <{cmpType = "slt"}> {rhs_const_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// FUSE-NEXT:   %12 = neura.grant_predicate %10, %11 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// FUSE-NEXT:   neura.ctrl_mov %12 -> %7 : !neura.data<i64, i1> !neura.data<i64, i1>
// FUSE-NEXT:   %13 = neura.grant_predicate %9, %11 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// FUSE-NEXT:   neura.ctrl_mov %13 -> %5 : !neura.data<f32, i1> !neura.data<f32, i1>
// FUSE-NEXT:   %14 = neura.grant_predicate %4, %11 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// FUSE-NEXT:   neura.ctrl_mov %14 -> %3 : !neura.data<f32, i1> !neura.data<f32, i1>
// FUSE-NEXT:   %15 = "neura.not"(%11) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-NEXT:   %16 = neura.grant_predicate %9, %15 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// FUSE-NEXT:   "neura.return"(%16) : (!neura.data<f32, i1>) -> ()
// FUSE-NEXT: }

// MOV:        func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// MOV-NEXT:     %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:     %1 = "neura.grant_once"() <{constant_value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// MOV-NEXT:     %2 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// MOV-NEXT:     %3 = neura.reserve : !neura.data<f32, i1>
// MOV-NEXT:     %4 = "neura.data_mov"(%1) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %5 = "neura.phi"(%3, %4) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %6 = neura.reserve : !neura.data<f32, i1>
// MOV-NEXT:     %7 = "neura.data_mov"(%2) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %8 = "neura.phi"(%6, %7) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %9 = neura.reserve : !neura.data<i64, i1>
// MOV-NEXT:     %10 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %11 = "neura.phi"(%9, %10) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %12 = "neura.data_mov"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %13 = "neura.data_mov"(%5) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %14 = "neura.fadd"(%12, %13) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %15 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %16 = "neura.add"(%15) {rhs_const_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %17 = "neura.data_mov"(%16) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %18 = "neura.icmp"(%17) <{cmpType = "slt"}> {rhs_const_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %19 = "neura.data_mov"(%16) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %20 = "neura.data_mov"(%18) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %21 = neura.grant_predicate %19, %20 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MOV-NEXT:     neura.ctrl_mov %21 -> %9 : !neura.data<i64, i1> !neura.data<i64, i1>
// MOV-NEXT:     %22 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %23 = "neura.data_mov"(%18) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %24 = neura.grant_predicate %22, %23 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:     neura.ctrl_mov %24 -> %6 : !neura.data<f32, i1> !neura.data<f32, i1>
// MOV-NEXT:     %25 = "neura.data_mov"(%5) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %26 = "neura.data_mov"(%18) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %27 = neura.grant_predicate %25, %26 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:     neura.ctrl_mov %27 -> %3 : !neura.data<f32, i1> !neura.data<f32, i1>
// MOV-NEXT:     %28 = "neura.data_mov"(%18) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %29 = "neura.not"(%28) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %30 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %31 = "neura.data_mov"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %32 = neura.grant_predicate %30, %31 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:     %33 = "neura.data_mov"(%32) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     "neura.return"(%33) : (!neura.data<f32, i1>) -> ()
// MOV-NEXT:   }

// MAPPING:       func.func @loop_test() -> f32 attributes {accelerator = "neura", mapping_info = {compiled_ii = 5 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 4 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
// MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = 0 : i64}> {mapping_locs = [{id = 11 : i32, resource = "tile", time_step = 0 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %1 = "neura.grant_once"() <{constant_value = 3.000000e+00 : f32}> {mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 2 : i32}]} : () -> !neura.data<f32, i1>
// MAPPING-NEXT:     %2 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> {mapping_locs = [{id = 15 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 3 : i32}]} : () -> !neura.data<f32, i1>
// MAPPING-NEXT:     %3 = neura.reserve : !neura.data<f32, i1>
// MAPPING-NEXT:     %4 = "neura.data_mov"(%1) {mapping_locs = [{id = 24 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %5 = "neura.phi"(%3, %4) {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %6 = neura.reserve : !neura.data<f32, i1>
// MAPPING-NEXT:     %7 = "neura.data_mov"(%2) {mapping_locs = [{id = 60 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %8 = "neura.phi"(%6, %7) {mapping_locs = [{id = 15 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 3 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %9 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:     %10 = "neura.data_mov"(%0) {mapping_locs = [{id = 44 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %11 = "neura.phi"(%9, %10) {mapping_locs = [{id = 11 : i32, resource = "tile", time_step = 1 : i32, x = 3 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %12 = "neura.data_mov"(%8) {mapping_locs = [{id = 46 : i32, resource = "link", time_step = 3 : i32}, {id = 45 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %13 = "neura.data_mov"(%5) {mapping_locs = [{id = 28 : i32, resource = "link", time_step = 3 : i32}, {id = 40 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %14 = "neura.fadd"(%12, %13) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %15 = "neura.data_mov"(%11) {mapping_locs = [{id = 44 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %16 = "neura.add"(%15) {mapping_locs = [{id = 11 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 2 : i32}], rhs_const_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %17 = "neura.data_mov"(%16) {mapping_locs = [{id = 35 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %18 = "neura.icmp"(%17) <{cmpType = "slt"}> {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}], rhs_const_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %19 = "neura.data_mov"(%16) {mapping_locs = [{id = 44 : i32, resource = "register", time_step = 2 : i32}, {id = 44 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %20 = "neura.data_mov"(%18) {mapping_locs = [{id = 32 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %21 = neura.grant_predicate %19, %20 {mapping_locs = [{id = 11 : i32, resource = "tile", time_step = 4 : i32, x = 3 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %21 -> %9 {mapping_locs = [{id = 45 : i32, resource = "register", time_step = 4 : i32}, {id = 45 : i32, resource = "register", time_step = 5 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %22 = "neura.data_mov"(%14) {mapping_locs = [{id = 40 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %23 = "neura.data_mov"(%18) {mapping_locs = [{id = 41 : i32, resource = "register", time_step = 3 : i32}, {id = 41 : i32, resource = "register", time_step = 4 : i32}, {id = 41 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %24 = neura.grant_predicate %22, %23 {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %24 -> %6 {mapping_locs = [{id = 32 : i32, resource = "link", time_step = 6 : i32}, {id = 37 : i32, resource = "link", time_step = 7 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
// MAPPING-NEXT:     %25 = "neura.data_mov"(%5) {mapping_locs = [{id = 36 : i32, resource = "register", time_step = 3 : i32}, {id = 36 : i32, resource = "register", time_step = 4 : i32}, {id = 36 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %26 = "neura.data_mov"(%18) {mapping_locs = [{id = 31 : i32, resource = "link", time_step = 3 : i32}, {id = 37 : i32, resource = "register", time_step = 4 : i32}, {id = 37 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %27 = neura.grant_predicate %25, %26 {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %27 -> %3 {mapping_locs = [{id = 36 : i32, resource = "register", time_step = 6 : i32}, {id = 36 : i32, resource = "register", time_step = 7 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
// MAPPING-NEXT:     %28 = "neura.data_mov"(%18) {mapping_locs = [{id = 33 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %29 = "neura.not"(%28) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %30 = "neura.data_mov"(%14) {mapping_locs = [{id = 33 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %31 = "neura.data_mov"(%29) {mapping_locs = [{id = 24 : i32, resource = "register", time_step = 4 : i32}, {id = 24 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %32 = neura.grant_predicate %30, %31 {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     %33 = "neura.data_mov"(%32) {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     "neura.return"(%33) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<f32, i1>) -> ()
// MAPPING-NEXT:   }

// YAML:      array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   cores:
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "5"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - timestep: 7
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "RETURN"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"


// ASM:      PE(2,2):
// ASM-NEXT: {
// ASM-NEXT:   ICMP, [EAST, RED] -> [EAST, RED], [$41], [WEST, RED], [SOUTH, RED]
// ASM-NEXT: } (t=3)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$40]
// ASM-NEXT: } (t=4)
// ASM-NEXT: {
// ASM-NEXT:   FADD, [NORTH, RED], [$40] -> [$40], [SOUTH, RED]
// ASM-NEXT: } (t=5)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$40], [$41] -> [EAST, RED]
// ASM-NEXT: } (t=6)