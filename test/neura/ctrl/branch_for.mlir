// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --leverage-predicated-value \
// RUN:   | FileCheck %s

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   | FileCheck %s -check-prefix=CANONICALIZE

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   | FileCheck %s -check-prefix=CTRL2DATA

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   | FileCheck %s -check-prefix=MOV

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=simple" \
// RUN:   | FileCheck %s -check-prefix=MAPPING

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=simple" \
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

// CANONICALIZE:        func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// CANONICALIZE-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> : () -> !neura.data<i64, i1>
// CANONICALIZE-NEXT:     %1 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CANONICALIZE-NEXT:     %2 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// CANONICALIZE-NEXT:     %3 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CANONICALIZE-NEXT:     %4 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CANONICALIZE-NEXT:     neura.br %1, %4, %3, %2, %0 : !neura.data<i64, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<i64, i1>, !neura.data<i64, i1> to ^bb1
// CANONICALIZE-NEXT:   ^bb1(%5: !neura.data<i64, i1>, %6: !neura.data<f32, i1>, %7: !neura.data<f32, i1>, %8: !neura.data<i64, i1>, %9: !neura.data<i64, i1>):  // 2 preds: ^bb0, ^bb1
// CANONICALIZE-NEXT:     %10 = "neura.fadd"(%6, %7) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CANONICALIZE-NEXT:     %11 = "neura.add"(%5, %8) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CANONICALIZE-NEXT:     %12 = "neura.icmp"(%11, %9) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CANONICALIZE-NEXT:     neura.cond_br %12 : !neura.data<i1, i1> then %11, %10, %3, %2, %0 : !neura.data<i64, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<i64, i1>, !neura.data<i64, i1> to ^bb1 else %10 : !neura.data<f32, i1> to ^bb2
// CANONICALIZE-NEXT:   ^bb2(%13: !neura.data<f32, i1>):  // pred: ^bb1
// CANONICALIZE-NEXT:     "neura.return"(%13) : (!neura.data<f32, i1>) -> ()
// CANONICALIZE-NEXT:   }

// CTRL2DATA:        func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %10 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %11 = "neura.phi"(%10, %1) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %12 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %13 = "neura.phi"(%12, %5) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %14 = neura.reserve : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %15 = "neura.phi"(%14, %7) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %16 = neura.reserve : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %17 = "neura.phi"(%16, %9) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %18 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %19 = "neura.phi"(%18, %3) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %20 = "neura.fadd"(%17, %15) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %21 = "neura.add"(%19, %13) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %22 = "neura.icmp"(%21, %11) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %23 = neura.grant_predicate %21, %22 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %23 -> %18 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %24 = neura.grant_predicate %20, %22 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %24 -> %16 : !neura.data<f32, i1> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %25 = neura.grant_predicate %7, %22 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %25 -> %14 : !neura.data<f32, i1> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %26 = neura.grant_predicate %5, %22 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %26 -> %12 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %27 = neura.grant_predicate %1, %22 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %27 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %28 = "neura.not"(%22) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %29 = neura.grant_predicate %20, %28 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     "neura.return"(%29) : (!neura.data<f32, i1>) -> ()
// CTRL2DATA-NEXT:   }

// MOV:        func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// MOV-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:     %1 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %2 = "neura.grant_once"(%1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %3 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:     %4 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:     %7 = "neura.data_mov"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %8 = "neura.grant_once"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %9 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// MOV-NEXT:     %10 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %11 = "neura.grant_once"(%10) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %12 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// MOV-NEXT:     %13 = "neura.data_mov"(%12) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %14 = "neura.grant_once"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %15 = neura.reserve : !neura.data<i64, i1>
// MOV-NEXT:     %16 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %17 = "neura.phi"(%15, %16) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %18 = neura.reserve : !neura.data<i64, i1>
// MOV-NEXT:     %19 = "neura.data_mov"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %20 = "neura.phi"(%18, %19) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %21 = neura.reserve : !neura.data<f32, i1>
// MOV-NEXT:     %22 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %23 = "neura.phi"(%21, %22) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %24 = neura.reserve : !neura.data<f32, i1>
// MOV-NEXT:     %25 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %26 = "neura.phi"(%24, %25) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %27 = neura.reserve : !neura.data<i64, i1>
// MOV-NEXT:     %28 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %29 = "neura.phi"(%27, %28) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %30 = "neura.data_mov"(%26) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %31 = "neura.data_mov"(%23) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %32 = "neura.fadd"(%30, %31) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %33 = "neura.data_mov"(%29) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %34 = "neura.data_mov"(%20) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %35 = "neura.add"(%33, %34) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %36 = "neura.data_mov"(%35) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %37 = "neura.data_mov"(%17) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %38 = "neura.icmp"(%36, %37) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %39 = "neura.data_mov"(%35) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %40 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %41 = neura.grant_predicate %39, %40 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MOV-NEXT:     neura.ctrl_mov %41 -> %27 : !neura.data<i64, i1> !neura.data<i64, i1>
// MOV-NEXT:     %42 = "neura.data_mov"(%32) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %43 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %44 = neura.grant_predicate %42, %43 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:     neura.ctrl_mov %44 -> %24 : !neura.data<f32, i1> !neura.data<f32, i1>
// MOV-NEXT:     %45 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %46 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %47 = neura.grant_predicate %45, %46 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:     neura.ctrl_mov %47 -> %21 : !neura.data<f32, i1> !neura.data<f32, i1>
// MOV-NEXT:     %48 = "neura.data_mov"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %49 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %50 = neura.grant_predicate %48, %49 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MOV-NEXT:     neura.ctrl_mov %50 -> %18 : !neura.data<i64, i1> !neura.data<i64, i1>
// MOV-NEXT:     %51 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %52 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %53 = neura.grant_predicate %51, %52 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MOV-NEXT:     neura.ctrl_mov %53 -> %15 : !neura.data<i64, i1> !neura.data<i64, i1>
// MOV-NEXT:     %54 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %55 = "neura.not"(%54) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %56 = "neura.data_mov"(%32) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %57 = "neura.data_mov"(%55) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %58 = neura.grant_predicate %56, %57 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:     %59 = "neura.data_mov"(%58) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     "neura.return"(%59) : (!neura.data<f32, i1>) -> ()
// MOV-NEXT:   }

// MAPPING:        func.func @loop_test() -> f32 attributes {CompiledII = 6 : i32, RecMII = 4 : i32, ResMII = 2 : i32, accelerator = "neura"} {
// MAPPING-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %1 = "neura.data_mov"(%0) {mapping_locs = [{id = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %2 = "neura.grant_once"(%1) {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %3 = "neura.constant"() <{predicate = true, value = 0 : i64}> {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %4 = "neura.data_mov"(%3) {mapping_locs = [{id = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %5 = "neura.grant_once"(%4) {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 1 : i64}> {mapping_locs = [{id = 2 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %7 = "neura.data_mov"(%6) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %8 = "neura.grant_once"(%7) {mapping_locs = [{id = 2 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %9 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 3 : i32}]} : () -> !neura.data<f32, i1>
// MAPPING-NEXT:     %10 = "neura.data_mov"(%9) {mapping_locs = [{id = 45 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %11 = "neura.grant_once"(%10) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %12 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> {mapping_locs = [{id = 11 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<f32, i1>
// MAPPING-NEXT:     %13 = "neura.data_mov"(%12) {mapping_locs = [{id = 36 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %14 = "neura.grant_once"(%13) {mapping_locs = [{id = 7 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 1 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %15 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:     %16 = "neura.data_mov"(%2) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %17 = "neura.phi"(%15, %16) {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %18 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:     %19 = "neura.data_mov"(%8) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %20 = "neura.phi"(%18, %19) {mapping_locs = [{id = 2 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %21 = neura.reserve : !neura.data<f32, i1>
// MAPPING-NEXT:     %22 = "neura.data_mov"(%11) {mapping_locs = [{id = 34 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %23 = "neura.phi"(%21, %22) {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %24 = neura.reserve : !neura.data<f32, i1>
// MAPPING-NEXT:     %25 = "neura.data_mov"(%14) {mapping_locs = [{id = 21 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %26 = "neura.phi"(%24, %25) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %27 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:     %28 = "neura.data_mov"(%5) {mapping_locs = [{id = 4 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %29 = "neura.phi"(%27, %28) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %30 = "neura.data_mov"(%26) {mapping_locs = [{id = 20 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %31 = "neura.data_mov"(%23) {mapping_locs = [{id = 45 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %32 = "neura.fadd"(%30, %31) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %33 = "neura.data_mov"(%29) {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %34 = "neura.data_mov"(%20) {mapping_locs = [{id = 7 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %35 = "neura.add"(%33, %34) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %36 = "neura.data_mov"(%35) {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %37 = "neura.data_mov"(%17) {mapping_locs = [{id = 4 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %38 = "neura.icmp"(%36, %37) <{cmpType = "slt"}> {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %39 = "neura.data_mov"(%35) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %40 = "neura.data_mov"(%38) {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %41 = neura.grant_predicate %39, %40 {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %41 -> %27 {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 5 : i32}, {id = 20 : i32, resource = "register", time_step = 6 : i32}, {id = 20 : i32, resource = "register", time_step = 7 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %42 = "neura.data_mov"(%32) {mapping_locs = [{id = 33 : i32, resource = "link", time_step = 5 : i32}, {id = 17 : i32, resource = "link", time_step = 6 : i32}, {id = 21 : i32, resource = "register", time_step = 7 : i32}, {id = 21 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %43 = "neura.data_mov"(%38) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %44 = neura.grant_predicate %42, %43 {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %44 -> %24 {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 9 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
// MAPPING-NEXT:     %45 = "neura.data_mov"(%11) {mapping_locs = [{id = 31 : i32, resource = "link", time_step = 3 : i32}, {id = 27 : i32, resource = "link", time_step = 4 : i32}, {id = 32 : i32, resource = "register", time_step = 5 : i32}, {id = 32 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %46 = "neura.data_mov"(%38) {mapping_locs = [{id = 13 : i32, resource = "link", time_step = 4 : i32}, {id = 12 : i32, resource = "link", time_step = 5 : i32}, {id = 33 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %47 = neura.grant_predicate %45, %46 {mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 7 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %47 -> %21 {mapping_locs = [{id = 24 : i32, resource = "link", time_step = 7 : i32}, {id = 30 : i32, resource = "link", time_step = 8 : i32}, {id = 41 : i32, resource = "link", time_step = 9 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
// MAPPING-NEXT:     %48 = "neura.data_mov"(%8) {mapping_locs = [{id = 7 : i32, resource = "link", time_step = 1 : i32}, {id = 24 : i32, resource = "register", time_step = 2 : i32}, {id = 24 : i32, resource = "register", time_step = 3 : i32}, {id = 24 : i32, resource = "register", time_step = 4 : i32}, {id = 24 : i32, resource = "register", time_step = 5 : i32}, {id = 24 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %49 = "neura.data_mov"(%38) {mapping_locs = [{id = 16 : i32, resource = "link", time_step = 4 : i32}, {id = 28 : i32, resource = "link", time_step = 5 : i32}, {id = 33 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %50 = neura.grant_predicate %48, %49 {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %50 -> %18 {mapping_locs = [{id = 19 : i32, resource = "link", time_step = 7 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %51 = "neura.data_mov"(%2) {mapping_locs = [{id = 4 : i32, resource = "link", time_step = 2 : i32}, {id = 21 : i32, resource = "register", time_step = 3 : i32}, {id = 21 : i32, resource = "register", time_step = 4 : i32}, {id = 21 : i32, resource = "register", time_step = 5 : i32}, {id = 21 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %52 = "neura.data_mov"(%38) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %53 = neura.grant_predicate %51, %52 {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %53 -> %15 {mapping_locs = [{id = 15 : i32, resource = "link", time_step = 7 : i32}, {id = 4 : i32, resource = "register", time_step = 8 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %54 = "neura.data_mov"(%38) {mapping_locs = [{id = 15 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %55 = "neura.not"(%54) {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %56 = "neura.data_mov"(%32) {mapping_locs = [{id = 31 : i32, resource = "link", time_step = 5 : i32}, {id = 36 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %57 = "neura.data_mov"(%55) {mapping_locs = [{id = 4 : i32, resource = "link", time_step = 5 : i32}, {id = 16 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %58 = neura.grant_predicate %56, %57 {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     %59 = "neura.data_mov"(%58) {mapping_locs = [{id = 30 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     "neura.return"(%59) {mapping_locs = [{id = 13 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 3 : i32}]} : (!neura.data<f32, i1>) -> ()
// MAPPING-NEXT:   }

// INST:        "name": "neura.fadd",
// INST-NEXT:   "operands": [
// INST-NEXT:     "neura.data_mov",
// INST-NEXT:     "neura.data_mov"
// INST-NEXT:   ],
// INST-NEXT:   "result_types": [
// INST-NEXT:     "!neura.data<f32, i1>"
// INST-NEXT:   ]