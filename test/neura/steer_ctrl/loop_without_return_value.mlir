// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --promote-input-arg-to-const \
// RUN: --canonicalize-return \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: --transform-to-steer-control \
// RUN: --remove-predicated-type \
// RUN: | FileCheck %s

module attributes {} {
  func.func @_Z11simple_loopPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    affine.for %arg2 = 0 to 128 {
      %0 = affine.load %arg0[%arg2] : memref<?xi32>
      %1 = arith.muli %0, %c2_i32 : i32
      %2 = arith.addi %1, %c1_i32 : i32
      affine.store %2, %arg1[%arg2] : memref<?xi32>
    }
    return
  }
}

// CHECK:      func.func @_Z11simple_loopPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {accelerator = "neura", dataflow_mode = "steering", llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:   %0 = neura.reserve : i64
// CHECK-NEXT:   %1 = neura.reserve : i1
// CHECK-NEXT:   %2 = "neura.constant"() <{value = "%arg0"}> : () -> memref<?xi32>
// CHECK-NEXT:   %3 = "neura.constant"() <{value = "%arg1"}> : () -> memref<?xi32>
// CHECK-NEXT:   %4 = "neura.constant"() <{value = 1 : i64}> : () -> i64
// CHECK-NEXT:   %5 = "neura.constant"() <{value = 128 : i64}> : () -> i64
// CHECK-NEXT:   %6 = "neura.constant"() <{value = 1 : i32}> : () -> i32
// CHECK-NEXT:   %7 = "neura.constant"() <{value = 2 : i32}> : () -> i32
// CHECK-NEXT:   %8 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CHECK-NEXT:   %9 = neura.invariant %4, %1 : i64, i1 -> i64
// CHECK-NEXT:   %10 = neura.invariant %3, %1 : memref<?xi32>, i1 -> memref<?xi32>
// CHECK-NEXT:   %11 = neura.invariant %6, %1 : i32, i1 -> i32
// CHECK-NEXT:   %12 = neura.invariant %7, %1 : i32, i1 -> i32
// CHECK-NEXT:   %13 = neura.invariant %2, %1 : memref<?xi32>, i1 -> memref<?xi32>
// CHECK-NEXT:   %14 = neura.invariant %5, %1 : i64, i1 -> i64
// CHECK-NEXT:   %15 = neura.carry %8, %1, %0 : i64, i1, i64 -> i64
// CHECK-NEXT:   %16 = "neura.icmp"(%15, %14) <{cmpType = "slt"}> : (i64, i64) -> i1
// CHECK-NEXT:   neura.ctrl_mov %16 -> %1 : i1 i1
// CHECK-NEXT:   %17 = "neura.not"(%16) : (i1) -> i1
// CHECK-NEXT:   %18 = neura.false_steer %17, %16 : i1, i1 -> i1
// CHECK-NEXT:   neura.return_void %18 : i1
// CHECK-NEXT:   %19 = neura.load_indexed %13[%15 : i64] memref<?xi32> : i32
// CHECK-NEXT:   %20 = "neura.mul"(%19, %12) : (i32, i32) -> i32
// CHECK-NEXT:   %21 = "neura.add"(%20, %11) : (i32, i32) -> i32
// CHECK-NEXT:   neura.store_indexed %21 to %10[%15 : i64] memref<?xi32> : i32
// CHECK-NEXT:   %22 = "neura.add"(%15, %9) : (i64, i64) -> i64
// CHECK-NEXT:   neura.ctrl_mov %22 -> %0 : i64 i64
// CHECK-NEXT:   neura.yield