// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --promote-func-arg-to-const \
// RUN: --canonicalize-return \
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
// RUN: --promote-func-arg-to-const \
// RUN: --canonicalize-return \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: --transform-to-steer-control \
// RUN: --remove-predicated-type \
// RUN: --insert-data-mov 
// RU: --map-to-accelerator="mapping-strategy=heuristic mapping-mode=spatial-only backtrack-config=customized" 
// RU: | FileCheck %s -check-prefix=MAPPING

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
// CHECK-NEXT:   neura.return_value %12 : i64
// CHECK-NEXT:   %13 = "neura.add"(%9, %9) : (i64, i64) -> i64
// CHECK-NEXT:   neura.ctrl_mov %13 -> %0 : i64 i64
// CHECK-NEXT:   %14 = "neura.add"(%10, %7) : (i64, i64) -> i64
// CHECK-NEXT:   neura.ctrl_mov %14 -> %1 : i64 i64
// CHECK-NEXT:   neura.yield
// CHECK-NEXT: }

// MAPPING:        func.func @simple_add_loop() -> i64 attributes {accelerator = "neura", dataflow_mode = "steering"} {
// MAPPING-NEXT: %0 = neura.reserve : i64
// MAPPING-NEXT: %1 = neura.reserve : i64
// MAPPING-NEXT: %2 = neura.reserve : i1
// MAPPING-NEXT: %3 = "neura.constant"() <{value = 16 : i64}> : () -> i64
// MAPPING-NEXT: %4 = "neura.constant"() <{value = 1 : i64}> : () -> i64
// MAPPING-NEXT: %5 = "neura.constant"() <{value = 1 : i64}> : () -> i64
// MAPPING-NEXT: %6 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// MAPPING-NEXT: %7 = "neura.data_mov"(%4) : (i64) -> i64
// MAPPING-NEXT: %8 = neura.invariant %7, %2 : i64, i1 -> i64
// MAPPING-NEXT: %9 = "neura.data_mov"(%3) : (i64) -> i64
// MAPPING-NEXT: %10 = neura.invariant %9, %2 : i64, i1 -> i64
// MAPPING-NEXT: %11 = "neura.data_mov"(%5) : (i64) -> i64
// MAPPING-NEXT: %12 = neura.carry %11, %2, %0 : i64, i1, i64 -> i64
// MAPPING-NEXT: %13 = "neura.data_mov"(%6) : (i64) -> i64
// MAPPING-NEXT: %14 = neura.carry %13, %2, %1 : i64, i1, i64 -> i64
// MAPPING-NEXT: %15 = "neura.data_mov"(%14) : (i64) -> i64
// MAPPING-NEXT: %16 = "neura.data_mov"(%10) : (i64) -> i64
// MAPPING-NEXT: %17 = "neura.icmp"(%15, %16) <{cmpType = "slt"}> : (i64, i64) -> i1
// MAPPING-NEXT: neura.ctrl_mov %17 -> %2 : i1 i1
// MAPPING-NEXT: %18 = "neura.data_mov"(%12) : (i64) -> i64
// MAPPING-NEXT: %19 = "neura.data_mov"(%17) : (i1) -> i1
// MAPPING-NEXT: %20 = neura.false_steer %18, %19 : i64, i1 -> i64
// MAPPING-NEXT: %21 = "neura.data_mov"(%12) : (i64) -> i64
// MAPPING-NEXT: %22 = "neura.data_mov"(%12) : (i64) -> i64
// MAPPING-NEXT: %23 = "neura.add"(%21, %22) : (i64, i64) -> i64
// MAPPING-NEXT: neura.ctrl_mov %23 -> %0 : i64 i64
// MAPPING-NEXT: %24 = "neura.data_mov"(%14) : (i64) -> i64
// MAPPING-NEXT: %25 = "neura.data_mov"(%8) : (i64) -> i64
// MAPPING-NEXT: %26 = "neura.add"(%24, %25) : (i64, i64) -> i64
// MAPPING-NEXT: neura.ctrl_mov %26 -> %1 : i64 i64
// MAPPING-NEXT: %27 = "neura.data_mov"(%20) : (i64) -> i64
// MAPPING-NEXT: "neura.return"(%27) : (i64) -> ()
// MAPPING-NEXT: }
// MAPPING-NEXT: }
