// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --leverage-predicated-value \
// RUN:   | FileCheck %s

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   | FileCheck %s -check-prefix=CTRL2DATA

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   | FileCheck %s -check-prefix=MOV

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator \
// RUN:   | FileCheck %s -check-prefix=MAPPING

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator \
// RUN:   --generate-code

// RUN: FileCheck %s --input-file=generated-instructions.json -check-prefix=INST

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

// CTRL2DATA:      func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// CTRL2DATA-NEXT:   %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %1 = "neura.grant_always"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %2 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %[[INT1:.*]] = "neura.grant_once"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %4 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %5 = "neura.grant_always"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %6 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:   %7 = "neura.grant_always"(%6) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:   %8 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:   %[[FLOAT1:.*]] = "neura.grant_once"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-DAG:    %[[RESERVEINT:.*]] = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-DAG:    %[[PHIINT:.*]] = "neura.phi"(%[[RESERVEINT:.*]], %[[INT1:.*]]) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-DAG:    %[[RESERVEFLOAT:.*]] = neura.reserve : !neura.data<f32, i1>
// CTRL2DATA-DAG:   %[[PHIFLOAT:.*]] = "neura.phi"(%[[RESERVEFLOAT:.*]], %[[FLOAT1:.*]]) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:   %[[FLOAT2:.*]] = "neura.fadd"(%[[PHIFLOAT:.*]], %7) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:   %[[INT2:.*]] = "neura.add"(%[[PHIINT:.*]], %5) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %16 = "neura.icmp"(%15, %1) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-DAG:   %[[GRANTFLOAT:.*]] = neura.grant_predicate %[[FLOAT2:.*]], %16 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-DAG:   neura.ctrl_mov %[[GRANTFLOAT:.*]] -> %[[RESERVEFLOAT:.*]] : !neura.data<f32, i1> !neura.data<f32, i1>
// CTRL2DATA-DAG:   %[[GRANTINT:.*]] = neura.grant_predicate %[[INT2:.*]], %16 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-DAG:   neura.ctrl_mov %[[GRANTINT:.*]] -> %[[RESERVEINT:.*]] : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %19 = "neura.not"(%16) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %20 = neura.grant_predicate %14, %19 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:   "neura.return"(%20) : (!neura.data<f32, i1>) -> ()
// CTRL2DATA-NEXT: }

// MOV:      func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// MOV-NEXT:   %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:   %1 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %2 = "neura.grant_always"(%1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %3 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:   %4 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %5 = "neura.grant_once"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %6 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:   %7 = "neura.data_mov"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %8 = "neura.grant_always"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %9 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// MOV-NEXT:   %10 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:   %11 = "neura.grant_always"(%10) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:   %12 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// MOV-NEXT:   %13 = "neura.data_mov"(%12) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:   %14 = "neura.grant_once"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:   %15 = neura.reserve : !neura.data<i64, i1>
// MOV-NEXT:   %16 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %17 = "neura.phi"(%16, %15) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %18 = neura.reserve : !neura.data<f32, i1>
// MOV-NEXT:   %19 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:   %20 = "neura.phi"(%19, %18) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:   %21 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:   %22 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:   %23 = "neura.fadd"(%21, %22) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:   %24 = "neura.data_mov"(%17) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %25 = "neura.data_mov"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %26 = "neura.add"(%24, %25) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %27 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %28 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %29 = "neura.icmp"(%27, %28) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:   %30 = "neura.data_mov"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:   %31 = "neura.not"(%30) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:   %32 = "neura.data_mov"(%23) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:   %33 = "neura.data_mov"(%31) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:   %34 = neura.grant_predicate %32, %33 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:   %35 = "neura.data_mov"(%23) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:   %36 = "neura.data_mov"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:   %37 = neura.grant_predicate %35, %36 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:   neura.ctrl_mov %37 -> %18 : !neura.data<f32, i1> !neura.data<f32, i1>
// MOV-NEXT:   %38 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:   %39 = "neura.data_mov"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:   %40 = neura.grant_predicate %38, %39 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MOV-NEXT:   neura.ctrl_mov %40 -> %15 : !neura.data<i64, i1> !neura.data<i64, i1>
// MOV-NEXT:   %41 = "neura.data_mov"(%34) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:   "neura.return"(%41) : (!neura.data<f32, i1>) -> ()
// MOV-NEXT: }

// MAPPING:      func.func @loop_test() -> f32 attributes {CompiledII = 6 : i32, RecMII = 4 : i32, ResMII = 1 : i32, accelerator = "neura"} {
// MAPPING-NEXT:   %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 0 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:   %1 = "neura.data_mov"(%0) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %2 = "neura.grant_always"(%1) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %3 = "neura.constant"() <{predicate = true, value = 0 : i64}> {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 0 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:   %4 = "neura.data_mov"(%3) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %5 = "neura.grant_once"(%4) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %6 = "neura.constant"() <{predicate = true, value = 1 : i64}> {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 0 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:   %7 = "neura.data_mov"(%6) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %8 = "neura.grant_always"(%7) {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %9 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 0 : i32}]} : () -> !neura.data<f32, i1>
// MAPPING-NEXT:   %10 = "neura.data_mov"(%9) {mapping_locs = []} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:   %11 = "neura.grant_always"(%10) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 1 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:   %12 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 0 : i32}]} : () -> !neura.data<f32, i1>
// MAPPING-NEXT:   %13 = "neura.data_mov"(%12) {mapping_locs = []} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:   %14 = "neura.grant_once"(%13) {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 1 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:   %15 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:   %16 = "neura.data_mov"(%5) {mapping_locs = [{id = 19 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %17 = "neura.phi"(%16, %15) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %18 = neura.reserve : !neura.data<f32, i1>
// MAPPING-NEXT:   %19 = "neura.data_mov"(%14) {mapping_locs = [{id = 43 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:   %20 = "neura.phi"(%19, %18) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 2 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:   %21 = "neura.data_mov"(%20) {mapping_locs = []} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:   %22 = "neura.data_mov"(%11) {mapping_locs = []} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:   %23 = "neura.fadd"(%21, %22) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 3 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:   %24 = "neura.data_mov"(%17) {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %25 = "neura.data_mov"(%8) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %26 = "neura.add"(%24, %25) {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 3 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %27 = "neura.data_mov"(%26) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %28 = "neura.data_mov"(%2) {mapping_locs = [{id = 15 : i32, resource = "link", time_step = 1 : i32}, {id = 11 : i32, resource = "link", time_step = 2 : i32}, {id = 26 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %29 = "neura.icmp"(%27, %28) <{cmpType = "slt"}> {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 4 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %30 = "neura.data_mov"(%29) {mapping_locs = [{id = 27 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %31 = "neura.not"(%30) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %32 = "neura.data_mov"(%23) {mapping_locs = [{id = 31 : i32, resource = "link", time_step = 3 : i32}, {id = 17 : i32, resource = "link", time_step = 4 : i32}, {id = 6 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:   %33 = "neura.data_mov"(%31) {mapping_locs = [{id = 13 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %34 = neura.grant_predicate %32, %33 {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 6 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:   %35 = "neura.data_mov"(%23) {mapping_locs = []} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:   %36 = "neura.data_mov"(%29) {mapping_locs = [{id = 30 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %37 = neura.grant_predicate %35, %36 {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 5 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:   neura.ctrl_mov %37 -> %18 {mapping_locs = []} : !neura.data<f32, i1> !neura.data<f32, i1>
// MAPPING-NEXT:   %38 = "neura.data_mov"(%26) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %39 = "neura.data_mov"(%29) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %40 = neura.grant_predicate %38, %39 {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 5 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:   neura.ctrl_mov %40 -> %15 {mapping_locs = [{id = 27 : i32, resource = "link", time_step = 5 : i32}, {id = 27 : i32, resource = "link", time_step = 6 : i32}, {id = 27 : i32, resource = "link", time_step = 7 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:   %41 = "neura.data_mov"(%34) {mapping_locs = []} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:   "neura.return"(%41) {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 7 : i32}]} : (!neura.data<f32, i1>) -> ()
// MAPPING-NEXT: }

// INST:        "name": "neura.fadd",
// INST-NEXT:   "operands": [
// INST-NEXT:     "neura.data_mov",
// INST-NEXT:     "neura.data_mov"
// INST-NEXT:   ],
// INST-NEXT:   "result_types": [
// INST-NEXT:     "!neura.data<f32, i1>"
// INST-NEXT:   ]