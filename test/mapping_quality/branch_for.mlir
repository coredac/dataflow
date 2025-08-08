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
// RUN:   --fold-constant \
// RUN:   | FileCheck %s -check-prefix=FUSE

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   | FileCheck %s -check-prefix=MOV

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=backtrack backtrack-config=heuristic" \
// RUN:   | FileCheck %s -check-prefix=MAPPING

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
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

// FUSE:        func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// FUSE-NEXT:     %0 = "neura.grant_once"() <{constant_value = 10 : i64}> : () -> !neura.data<i64, i1>
// FUSE-NEXT:     %1 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// FUSE-NEXT:     %2 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
// FUSE-NEXT:     %3 = "neura.grant_once"() <{constant_value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// FUSE-NEXT:     %4 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// FUSE-NEXT:     %5 = neura.reserve : !neura.data<i64, i1>
// FUSE-NEXT:     %6 = "neura.phi"(%5, %0) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-NEXT:     %7 = neura.reserve : !neura.data<i64, i1>
// FUSE-NEXT:     %8 = "neura.phi"(%7, %2) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-NEXT:     %9 = neura.reserve : !neura.data<f32, i1>
// FUSE-NEXT:     %10 = "neura.phi"(%9, %3) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// FUSE-NEXT:     %11 = neura.reserve : !neura.data<f32, i1>
// FUSE-NEXT:     %12 = "neura.phi"(%11, %4) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// FUSE-NEXT:     %13 = neura.reserve : !neura.data<i64, i1>
// FUSE-NEXT:     %14 = "neura.phi"(%13, %1) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-NEXT:     %15 = "neura.fadd"(%12, %10) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// FUSE-NEXT:     %16 = "neura.add"(%14, %8) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-NEXT:     %17 = "neura.icmp"(%16, %6) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// FUSE-NEXT:     %18 = neura.grant_predicate %16, %17 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// FUSE-NEXT:     neura.ctrl_mov %18 -> %13 : !neura.data<i64, i1> !neura.data<i64, i1>
// FUSE-NEXT:     %19 = neura.grant_predicate %15, %17 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// FUSE-NEXT:     neura.ctrl_mov %19 -> %11 : !neura.data<f32, i1> !neura.data<f32, i1>
// FUSE-NEXT:     %20 = neura.grant_predicate %3, %17 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// FUSE-NEXT:     neura.ctrl_mov %20 -> %9 : !neura.data<f32, i1> !neura.data<f32, i1>
// FUSE-NEXT:     %21 = neura.grant_predicate %2, %17 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// FUSE-NEXT:     neura.ctrl_mov %21 -> %7 : !neura.data<i64, i1> !neura.data<i64, i1>
// FUSE-NEXT:     %22 = neura.grant_predicate %0, %17 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// FUSE-NEXT:     neura.ctrl_mov %22 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
// FUSE-NEXT:     %23 = "neura.not"(%17) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-NEXT:     %24 = neura.grant_predicate %15, %23 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// FUSE-NEXT:     "neura.return"(%24) : (!neura.data<f32, i1>) -> ()
// FUSE-NEXT:   }

// MOV:        func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// MOV-NEXT:     %0 = "neura.grant_once"() <{constant_value = 10 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:     %1 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:     %2 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:     %3 = "neura.grant_once"() <{constant_value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// MOV-NEXT:     %4 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// MOV-NEXT:     %5 = neura.reserve : !neura.data<i64, i1>
// MOV-NEXT:     %6 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %7 = "neura.phi"(%5, %6) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %8 = neura.reserve : !neura.data<i64, i1>
// MOV-NEXT:     %9 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %10 = "neura.phi"(%8, %9) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %11 = neura.reserve : !neura.data<f32, i1>
// MOV-NEXT:     %12 = "neura.data_mov"(%3) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %13 = "neura.phi"(%11, %12) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %14 = neura.reserve : !neura.data<f32, i1>
// MOV-NEXT:     %15 = "neura.data_mov"(%4) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %16 = "neura.phi"(%14, %15) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %17 = neura.reserve : !neura.data<i64, i1>
// MOV-NEXT:     %18 = "neura.data_mov"(%1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %19 = "neura.phi"(%17, %18) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %20 = "neura.data_mov"(%16) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %21 = "neura.data_mov"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %22 = "neura.fadd"(%20, %21) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %23 = "neura.data_mov"(%19) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %24 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %25 = "neura.add"(%23, %24) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %26 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %27 = "neura.data_mov"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %28 = "neura.icmp"(%26, %27) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %29 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %30 = "neura.data_mov"(%28) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %31 = neura.grant_predicate %29, %30 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MOV-NEXT:     neura.ctrl_mov %31 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1>
// MOV-NEXT:     %32 = "neura.data_mov"(%22) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %33 = "neura.data_mov"(%28) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %34 = neura.grant_predicate %32, %33 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:     neura.ctrl_mov %34 -> %14 : !neura.data<f32, i1> !neura.data<f32, i1>
// MOV-NEXT:     %35 = "neura.data_mov"(%3) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %36 = "neura.data_mov"(%28) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %37 = neura.grant_predicate %35, %36 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:     neura.ctrl_mov %37 -> %11 : !neura.data<f32, i1> !neura.data<f32, i1>
// MOV-NEXT:     %38 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %39 = "neura.data_mov"(%28) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %40 = neura.grant_predicate %38, %39 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MOV-NEXT:     neura.ctrl_mov %40 -> %8 : !neura.data<i64, i1> !neura.data<i64, i1>
// MOV-NEXT:     %41 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %42 = "neura.data_mov"(%28) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %43 = neura.grant_predicate %41, %42 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MOV-NEXT:     neura.ctrl_mov %43 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
// MOV-NEXT:     %44 = "neura.data_mov"(%28) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %45 = "neura.not"(%44) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %46 = "neura.data_mov"(%22) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %47 = "neura.data_mov"(%45) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %48 = neura.grant_predicate %46, %47 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:     %49 = "neura.data_mov"(%48) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     "neura.return"(%49) : (!neura.data<f32, i1>) -> ()
// MOV-NEXT:   }

// MAPPING:        func.func @loop_test() -> f32 attributes {CompiledII = 6 : i32, RecMII = 4 : i32, ResMII = 2 : i32, accelerator = "neura"} {
// MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = 10 : i64}> {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %1 = "neura.grant_once"() <{constant_value = 0 : i64}> {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %2 = "neura.grant_once"() <{constant_value = 1 : i64}> {mapping_locs = [{id = 2 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %3 = "neura.grant_once"() <{constant_value = 3.000000e+00 : f32}> {mapping_locs = [{id = 4 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}]} : () -> !neura.data<f32, i1>
// MAPPING-NEXT:     %4 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> {mapping_locs = [{id = 15 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 3 : i32}]} : () -> !neura.data<f32, i1>
// MAPPING-NEXT:     %5 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:     %6 = "neura.data_mov"(%0) {mapping_locs = [{id = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %7 = "neura.phi"(%5, %6) {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %8 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:     %9 = "neura.data_mov"(%2) {mapping_locs = [{id = 7 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %10 = "neura.phi"(%8, %9) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %11 = neura.reserve : !neura.data<f32, i1>
// MAPPING-NEXT:     %12 = "neura.data_mov"(%3) {mapping_locs = [{id = 11 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %13 = "neura.phi"(%11, %12) {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %14 = neura.reserve : !neura.data<f32, i1>
// MAPPING-NEXT:     %15 = "neura.data_mov"(%4) {mapping_locs = [{id = 47 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %16 = "neura.phi"(%14, %15) {mapping_locs = [{id = 11 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %17 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:     %18 = "neura.data_mov"(%1) {mapping_locs = [{id = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %19 = "neura.phi"(%17, %18) {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %20 = "neura.data_mov"(%16) {mapping_locs = [{id = 35 : i32, resource = "link", time_step = 3 : i32}, {id = 31 : i32, resource = "link", time_step = 4 : i32}, {id = 36 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %21 = "neura.data_mov"(%13) {mapping_locs = [{id = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 4 : i32, resource = "link", time_step = 4 : i32}, {id = 16 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %22 = "neura.fadd"(%20, %21) {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %23 = "neura.data_mov"(%19) {mapping_locs = [{id = 4 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %24 = "neura.data_mov"(%10) {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %25 = "neura.add"(%23, %24) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %26 = "neura.data_mov"(%25) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %27 = "neura.data_mov"(%7) {mapping_locs = [{id = 4 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %28 = "neura.icmp"(%26, %27) <{cmpType = "slt"}> {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %29 = "neura.data_mov"(%25) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %30 = "neura.data_mov"(%28) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %31 = neura.grant_predicate %29, %30 {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %31 -> %17 {mapping_locs = [{id = 15 : i32, resource = "link", time_step = 4 : i32}, {id = 4 : i32, resource = "register", time_step = 5 : i32}, {id = 4 : i32, resource = "register", time_step = 6 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %32 = "neura.data_mov"(%22) {mapping_locs = []} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %33 = "neura.data_mov"(%28) {mapping_locs = [{id = 16 : i32, resource = "link", time_step = 3 : i32}, {id = 37 : i32, resource = "register", time_step = 4 : i32}, {id = 37 : i32, resource = "register", time_step = 5 : i32}, {id = 37 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %34 = neura.grant_predicate %32, %33 {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %34 -> %14 {mapping_locs = [{id = 28 : i32, resource = "link", time_step = 7 : i32}, {id = 32 : i32, resource = "link", time_step = 8 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
// MAPPING-NEXT:     %35 = "neura.data_mov"(%3) {mapping_locs = [{id = 12 : i32, resource = "link", time_step = 2 : i32}, {id = 24 : i32, resource = "link", time_step = 3 : i32}, {id = 29 : i32, resource = "link", time_step = 4 : i32}, {id = 15 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %36 = "neura.data_mov"(%28) {mapping_locs = [{id = 13 : i32, resource = "link", time_step = 3 : i32}, {id = 11 : i32, resource = "link", time_step = 4 : i32}, {id = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %37 = neura.grant_predicate %35, %36 {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %37 -> %11 {mapping_locs = [{id = 2 : i32, resource = "link", time_step = 6 : i32}, {id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 0 : i32, resource = "register", time_step = 8 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
// MAPPING-NEXT:     %38 = "neura.data_mov"(%2) {mapping_locs = [{id = 6 : i32, resource = "link", time_step = 0 : i32}, {id = 9 : i32, resource = "link", time_step = 1 : i32}, {id = 21 : i32, resource = "link", time_step = 2 : i32}, {id = 24 : i32, resource = "register", time_step = 3 : i32}, {id = 24 : i32, resource = "register", time_step = 4 : i32}, {id = 24 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %39 = "neura.data_mov"(%28) {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 3 : i32}, {id = 25 : i32, resource = "register", time_step = 4 : i32}, {id = 25 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %40 = neura.grant_predicate %38, %39 {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %40 -> %8 {mapping_locs = []} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %41 = "neura.data_mov"(%0) {mapping_locs = [{id = 1 : i32, resource = "link", time_step = 1 : i32}, {id = 10 : i32, resource = "link", time_step = 2 : i32}, {id = 20 : i32, resource = "register", time_step = 3 : i32}, {id = 20 : i32, resource = "register", time_step = 4 : i32}, {id = 20 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %42 = "neura.data_mov"(%28) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %43 = neura.grant_predicate %41, %42 {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %43 -> %5 {mapping_locs = [{id = 15 : i32, resource = "link", time_step = 6 : i32}, {id = 4 : i32, resource = "register", time_step = 7 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %44 = "neura.data_mov"(%28) {mapping_locs = [{id = 15 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %45 = "neura.not"(%44) {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %46 = "neura.data_mov"(%22) {mapping_locs = [{id = 27 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %47 = "neura.data_mov"(%45) {mapping_locs = [{id = 2 : i32, resource = "link", time_step = 4 : i32}, {id = 1 : i32, resource = "link", time_step = 5 : i32}, {id = 12 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %48 = neura.grant_predicate %46, %47 {mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 7 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     %49 = "neura.data_mov"(%48) {mapping_locs = []} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     "neura.return"(%49) {mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 8 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>) -> ()
// MAPPING-NEXT:   }

// INST:        "name": "neura.fadd",
// INST-NEXT:   "operands": [
// INST-NEXT:     "neura.data_mov",
// INST-NEXT:     "neura.data_mov"
// INST-NEXT:   ],
// INST-NEXT:   "result_types": [
// INST-NEXT:     "!neura.data<f32, i1>"
// INST-NEXT:   ]