// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --leverage-predicated-value \
// RUN:   | FileCheck %s

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --neura-canonicalize \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   | FileCheck %s -check-prefix=CTRL2DATA

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --neura-canonicalize \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   | FileCheck %s -check-prefix=MOV

// RU: mlir-neura-opt %s \
// RU:   --assign-accelerator \
// RU:   --lower-llvm-to-neura \
// RU:   --neura-canonicalize \
// RU:   --leverage-predicated-value \
// RU:   --transform-ctrl-to-data-flow \
// RU:   --insert-data-mov \
// RU:   --map-to-accelerator="mapping-strategy=heuristic" \
// RU:   | FileCheck %s -check-prefix=MAPPING

// RU: mlir-neura-opt %s \
// RU:   --assign-accelerator \
// RU:   --lower-llvm-to-neura \
// RU:   --neura-canonicalize \
// RU:   --leverage-predicated-value \
// RU:   --transform-ctrl-to-data-flow \
// RU:   --insert-data-mov \
// RU:   --map-to-accelerator="mapping-strategy=heuristic" \
// RU:   --generate-code
// RU: FileCheck %s --input-file=generated-instructions.json -check-prefix=INST

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

// CTRL2DATA:   func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_always"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %2 = "neura.grant_once"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %3 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %4 = "neura.grant_once"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %5 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %6 = "neura.grant_always"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_always"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %10 = "neura.grant_once"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %11 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %12 = "neura.grant_once"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %13 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %14 = "neura.phi"(%13, %2) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %15 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %16 = "neura.phi"(%15, %7) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %17 = neura.reserve : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %18 = "neura.phi"(%17, %10) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %19 = neura.reserve : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %20 = "neura.phi"(%19, %12) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %21 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %22 = "neura.phi"(%21, %4) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %23 = "neura.fadd"(%20, %18) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %24 = "neura.add"(%22, %16) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %25 = "neura.icmp"(%24, %14) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %26 = neura.grant_predicate %24, %25 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %26 -> %21 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %27 = neura.grant_predicate %23, %25 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %27 -> %19 : !neura.data<f32, i1> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %28 = neura.grant_predicate %10, %25 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %28 -> %17 : !neura.data<f32, i1> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %29 = neura.grant_predicate %7, %25 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %29 -> %15 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %30 = neura.grant_predicate %2, %25 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %30 -> %13 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %31 = "neura.not"(%25) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %32 = neura.grant_predicate %23, %31 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     "neura.return"(%32) : (!neura.data<f32, i1>) -> ()
// CTRL2DATA-NEXT:   }

// MOV:      func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// MOV-NEXT: %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:     %1 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %2 = "neura.grant_always"(%1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %3 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %4 = "neura.grant_once"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %5 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:     %6 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %8 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:     %9 = "neura.data_mov"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %10 = "neura.grant_always"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %11 = "neura.data_mov"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %12 = "neura.grant_once"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %13 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// MOV-NEXT:     %14 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %15 = "neura.grant_always"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %16 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %17 = "neura.grant_once"(%16) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %18 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// MOV-NEXT:     %19 = "neura.data_mov"(%18) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %20 = "neura.grant_once"(%19) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %21 = neura.reserve : !neura.data<i64, i1>
// MOV-NEXT:     %22 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %23 = "neura.phi"(%21, %22) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %24 = neura.reserve : !neura.data<i64, i1>
// MOV-NEXT:     %25 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %26 = "neura.phi"(%24, %25) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %27 = neura.reserve : !neura.data<f32, i1>
// MOV-NEXT:     %28 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %29 = "neura.phi"(%27, %28) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %30 = neura.reserve : !neura.data<f32, i1>
// MOV-NEXT:     %31 = "neura.data_mov"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %32 = "neura.phi"(%30, %31) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %33 = neura.reserve : !neura.data<i64, i1>
// MOV-NEXT:     %34 = "neura.data_mov"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %35 = "neura.phi"(%33, %34) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %36 = "neura.data_mov"(%32) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %37 = "neura.data_mov"(%29) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %38 = "neura.fadd"(%36, %37) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %39 = "neura.data_mov"(%35) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %40 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %41 = "neura.add"(%39, %40) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %42 = "neura.data_mov"(%41) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %43 = "neura.data_mov"(%23) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %44 = "neura.icmp"(%42, %43) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %45 = "neura.data_mov"(%41) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %46 = "neura.data_mov"(%44) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %47 = neura.grant_predicate %45, %46 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MOV-NEXT:     neura.ctrl_mov %47 -> %33 : !neura.data<i64, i1> !neura.data<i64, i1>
// MOV-NEXT:     %48 = "neura.data_mov"(%38) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %49 = "neura.data_mov"(%44) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %50 = neura.grant_predicate %48, %49 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:     neura.ctrl_mov %50 -> %30 : !neura.data<f32, i1> !neura.data<f32, i1>
// MOV-NEXT:     %51 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %52 = "neura.data_mov"(%44) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %53 = neura.grant_predicate %51, %52 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:     neura.ctrl_mov %53 -> %27 : !neura.data<f32, i1> !neura.data<f32, i1>
// MOV-NEXT:     %54 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %55 = "neura.data_mov"(%44) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %56 = neura.grant_predicate %54, %55 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MOV-NEXT:     neura.ctrl_mov %56 -> %24 : !neura.data<i64, i1> !neura.data<i64, i1>
// MOV-NEXT:     %57 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %58 = "neura.data_mov"(%44) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %59 = neura.grant_predicate %57, %58 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MOV-NEXT:     neura.ctrl_mov %59 -> %21 : !neura.data<i64, i1> !neura.data<i64, i1>
// MOV-NEXT:     %60 = "neura.data_mov"(%44) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %61 = "neura.not"(%60) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %62 = "neura.data_mov"(%38) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %63 = "neura.data_mov"(%61) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %64 = neura.grant_predicate %62, %63 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:     %65 = "neura.data_mov"(%64) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     "neura.return"(%65) : (!neura.data<f32, i1>) -> ()
// MOV-NEXT:   }

// MAPPING: func.func @loop_test() -> f32 attributes {CompiledII = 6 : i32, RecMII = 4 : i32, ResMII = 2 : i32, accelerator = "neura"} {
// MAPPING-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 1 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %1 = "neura.data_mov"(%0) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %2 = "neura.grant_always"(%1) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %3 = "neura.data_mov"(%0) {mapping_locs = [{id = 16 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %4 = "neura.grant_once"(%3) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %5 = "neura.constant"() <{predicate = true, value = 0 : i64}> {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %6 = "neura.data_mov"(%5) {mapping_locs = [{id = 20 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %7 = "neura.grant_once"(%6) {mapping_locs = [{id = 7 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %8 = "neura.constant"() <{predicate = true, value = 1 : i64}> {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %9 = "neura.data_mov"(%8) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %10 = "neura.grant_always"(%9) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %11 = "neura.data_mov"(%8) {mapping_locs = [{id = 33 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %12 = "neura.grant_once"(%11) {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %13 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 1 : i32}]} : () -> !neura.data<f32, i1>
// MAPPING-NEXT:     %14 = "neura.data_mov"(%13) {mapping_locs = [{id = 28 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %15 = "neura.grant_always"(%14) {mapping_locs = [{id = 13 : i32, resource = "tile", time_step = 1 : i32, x = 3 : i32, y = 1 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %16 = "neura.data_mov"(%13) {mapping_locs = [{id = 29 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %17 = "neura.grant_once"(%16) {mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %18 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 0 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<f32, i1>
// MAPPING-NEXT:     %19 = "neura.data_mov"(%18) {mapping_locs = []} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %20 = "neura.grant_once"(%19) {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 1 : i32, x = 3 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %21 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:     %22 = "neura.data_mov"(%4) {mapping_locs = [{id = 19 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %23 = "neura.phi"(%21, %22) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %24 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:     %25 = "neura.data_mov"(%12) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %26 = "neura.phi"(%24, %25) {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %27 = neura.reserve : !neura.data<f32, i1>
// MAPPING-NEXT:     %28 = "neura.data_mov"(%17) {mapping_locs = []} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %29 = "neura.phi"(%27, %28) {mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %30 = neura.reserve : !neura.data<f32, i1>
// MAPPING-NEXT:     %31 = "neura.data_mov"(%20) {mapping_locs = [{id = 43 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %32 = "neura.phi"(%30, %31) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %33 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:     %34 = "neura.data_mov"(%7) {mapping_locs = [{id = 23 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %35 = "neura.phi"(%33, %34) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %36 = "neura.data_mov"(%32) {mapping_locs = [{id = 31 : i32, resource = "link", time_step = 2 : i32}, {id = 19 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %37 = "neura.data_mov"(%29) {mapping_locs = [{id = 24 : i32, resource = "link", time_step = 2 : i32}, {id = 12 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %38 = "neura.fadd"(%36, %37) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %39 = "neura.data_mov"(%35) {mapping_locs = [{id = 18 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %40 = "neura.data_mov"(%26) {mapping_locs = [{id = 30 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %41 = "neura.add"(%39, %40) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %42 = "neura.data_mov"(%41) {mapping_locs = [{id = 33 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %43 = "neura.data_mov"(%23) {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 2 : i32}, {id = 14 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %44 = "neura.icmp"(%42, %43) <{cmpType = "slt"}> {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %45 = "neura.data_mov"(%41) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %46 = "neura.data_mov"(%44) {mapping_locs = [{id = 30 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %47 = neura.grant_predicate %45, %46 {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %47 -> %33 {mapping_locs = [{id = 31 : i32, resource = "link", time_step = 5 : i32}, {id = 31 : i32, resource = "link", time_step = 6 : i32}, {id = 31 : i32, resource = "link", time_step = 7 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %48 = "neura.data_mov"(%38) {mapping_locs = []} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %49 = "neura.data_mov"(%44) {mapping_locs = [{id = 27 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %50 = neura.grant_predicate %48, %49 {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %50 -> %30 {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 5 : i32}, {id = 30 : i32, resource = "link", time_step = 6 : i32}, {id = 30 : i32, resource = "link", time_step = 7 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
// MAPPING-NEXT:     %51 = "neura.data_mov"(%17) {mapping_locs = [{id = 25 : i32, resource = "link", time_step = 1 : i32}, {id = 39 : i32, resource = "link", time_step = 2 : i32}, {id = 39 : i32, resource = "link", time_step = 3 : i32}, {id = 39 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %52 = "neura.data_mov"(%44) {mapping_locs = [{id = 28 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %53 = neura.grant_predicate %51, %52 {mapping_locs = [{id = 13 : i32, resource = "tile", time_step = 5 : i32, x = 3 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %53 -> %27 {mapping_locs = [{id = 41 : i32, resource = "link", time_step = 5 : i32}, {id = 38 : i32, resource = "link", time_step = 6 : i32}, {id = 38 : i32, resource = "link", time_step = 7 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
// MAPPING-NEXT:     %54 = "neura.data_mov"(%12) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %55 = "neura.data_mov"(%44) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %56 = neura.grant_predicate %54, %55 {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 9 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %56 -> %24 {mapping_locs = []} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %57 = "neura.data_mov"(%4) {mapping_locs = [{id = 18 : i32, resource = "link", time_step = 1 : i32}, {id = 33 : i32, resource = "link", time_step = 2 : i32}, {id = 27 : i32, resource = "link", time_step = 3 : i32}, {id = 15 : i32, resource = "link", time_step = 4 : i32}, {id = 15 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %58 = "neura.data_mov"(%44) {mapping_locs = [{id = 29 : i32, resource = "link", time_step = 4 : i32}, {id = 24 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %59 = neura.grant_predicate %57, %58 {mapping_locs = [{id = 4 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %59 -> %21 {mapping_locs = [{id = 12 : i32, resource = "link", time_step = 6 : i32}, {id = 12 : i32, resource = "link", time_step = 7 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %60 = "neura.data_mov"(%44) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %61 = "neura.not"(%60) {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %62 = "neura.data_mov"(%38) {mapping_locs = [{id = 13 : i32, resource = "link", time_step = 4 : i32}, {id = 3 : i32, resource = "link", time_step = 5 : i32}, {id = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %63 = "neura.data_mov"(%61) {mapping_locs = [{id = 27 : i32, resource = "link", time_step = 5 : i32}, {id = 15 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %64 = neura.grant_predicate %62, %63 {mapping_locs = [{id = 4 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     %65 = "neura.data_mov"(%64) {mapping_locs = []} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     "neura.return"(%65) {mapping_locs = [{id = 4 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<f32, i1>) -> ()
// MAPPING-NEXT:   }

// INST:        "name": "neura.fadd",
// INST-NEXT:   "operands": [
// INST-NEXT:     "neura.data_mov",
// INST-NEXT:     "neura.data_mov"
// INST-NEXT:   ],
// INST-NEXT:   "result_types": [
// INST-NEXT:     "!neura.data<f32, i1>"
// INST-NEXT:   ]