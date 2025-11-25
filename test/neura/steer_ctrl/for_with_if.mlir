// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --promote-func-arg-to-const \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: --transform-to-steer-control \
// RUN: --remove-predicated-type \
// RUN: | FileCheck %s

module attributes {} {
  func.func @_Z11for_with_ifPi(%arg0: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c-5_i32 = arith.constant -5 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1000_i32 = arith.constant 1000 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %c0_i32) -> (i32) {
      %1 = arith.cmpi sge, %arg2, %c1000_i32 : i32
      %2 = scf.if %1 -> (i32) {
        %7 = arith.addi %arg2, %c-5_i32 : i32
        scf.yield %7 : i32
      } else {
        scf.yield %arg2 : i32
      }
      %3 = memref.load %arg0[%arg1] : memref<?xi32>
      %4 = arith.muli %3, %c2_i32 : i32
      %5 = arith.addi %4, %c1_i32 : i32
      %6 = arith.addi %2, %5 : i32
      scf.yield %6 : i32
    }
    return %0 : i32
  }
}

// CHECK:      func.func @_Z11for_with_ifPi(%arg0: memref<?xi32>) -> i32 attributes {accelerator = "neura", dataflow_mode = "steering", llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:   %0 = neura.reserve : i64
// CHECK-NEXT:   %1 = neura.reserve : i32
// CHECK-NEXT:   %2 = neura.reserve : i32
// CHECK-NEXT:   %3 = neura.reserve : memref<?xi32>
// CHECK-NEXT:   %4 = neura.reserve : i32
// CHECK-NEXT:   %5 = neura.reserve : i32
// CHECK-NEXT:   %6 = neura.reserve : i64
// CHECK-NEXT:   %7 = neura.reserve : i32
// CHECK-NEXT:   %8 = neura.reserve : i64
// CHECK-NEXT:   %9 = neura.reserve : i1
// CHECK-NEXT:   %10 = "neura.constant"() <{value = "%arg0"}> : () -> memref<?xi32>
// CHECK-NEXT:   %11 = "neura.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-NEXT:   %12 = "neura.constant"() <{value = 1000 : i32}> : () -> i32
// CHECK-NEXT:   %13 = "neura.constant"() <{value = 2 : i32}> : () -> i32
// CHECK-NEXT:   %14 = "neura.constant"() <{value = 1 : i32}> : () -> i32
// CHECK-NEXT:   %15 = "neura.constant"() <{value = -5 : i32}> : () -> i32
// CHECK-NEXT:   %16 = "neura.constant"() <{value = 1 : i64}> : () -> i64
// CHECK-NEXT:   %17 = "neura.constant"() <{value = 128 : i64}> : () -> i64
// CHECK-NEXT:   %18 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CHECK-NEXT:   %19 = neura.carry %16, %9, %0 : i64, i1, i64 -> i64
// CHECK-NEXT:   %20 = neura.carry %14, %9, %1 : i32, i1, i32 -> i32
// CHECK-NEXT:   %21 = neura.carry %13, %9, %2 : i32, i1, i32 -> i32
// CHECK-NEXT:   %22 = neura.carry %10, %9, %3 : memref<?xi32>, i1, memref<?xi32> -> memref<?xi32>
// CHECK-NEXT:   %23 = neura.carry %15, %9, %4 : i32, i1, i32 -> i32
// CHECK-NEXT:   %24 = neura.carry %12, %9, %5 : i32, i1, i32 -> i32
// CHECK-NEXT:   %25 = neura.carry %17, %9, %6 : i64, i1, i64 -> i64
// CHECK-NEXT:   %26 = neura.carry %11, %9, %7 : i32, i1, i32 -> i32
// CHECK-NEXT:   %27 = neura.carry %18, %9, %8 : i64, i1, i64 -> i64
// CHECK-NEXT:   %28 = "neura.icmp"(%27, %25) <{cmpType = "slt"}> : (i64, i64) -> i1
// CHECK-NEXT:   neura.ctrl_mov %28 -> %9 : i1 i1
// CHECK-NEXT:   %29 = neura.false_steer %26, %28 : i32, i1 -> i32
// CHECK-NEXT:   %30 = "neura.icmp"(%26, %24) <{cmpType = "sge"}> : (i32, i32) -> i1
// CHECK-NEXT:   %31 = neura.true_steer %26, %30 : i32, i1 -> i32
// CHECK-NEXT:   %32 = neura.true_steer %23, %30 : i32, i1 -> i32
// CHECK-NEXT:   %33 = neura.false_steer %26, %30 : i32, i1 -> i32
// CHECK-NEXT:   %34 = "neura.add"(%31, %32) : (i32, i32) -> i32
// CHECK-NEXT:   %35 = neura.merge %30, %23, %23 : i1, i32, i32 -> i32
// CHECK-NEXT:   neura.ctrl_mov %35 -> %4 : i32 i32
// CHECK-NEXT:   %36 = neura.merge %30, %24, %24 : i1, i32, i32 -> i32
// CHECK-NEXT:   neura.ctrl_mov %36 -> %5 : i32 i32
// CHECK-NEXT:   %37 = neura.merge %30, %25, %25 : i1, i64, i64 -> i64
// CHECK-NEXT:   neura.ctrl_mov %37 -> %6 : i64 i64
// CHECK-NEXT:   %38 = neura.merge %30, %19, %19 : i1, i64, i64 -> i64
// CHECK-NEXT:   neura.ctrl_mov %38 -> %0 : i64 i64
// CHECK-NEXT:   %39 = neura.merge %30, %20, %20 : i1, i32, i32 -> i32
// CHECK-NEXT:   neura.ctrl_mov %39 -> %1 : i32 i32
// CHECK-NEXT:   %40 = neura.merge %30, %21, %21 : i1, i32, i32 -> i32
// CHECK-NEXT:   neura.ctrl_mov %40 -> %2 : i32 i32
// CHECK-NEXT:   %41 = neura.merge %30, %27, %27 : i1, i64, i64 -> i64
// CHECK-NEXT:   %42 = neura.merge %30, %22, %22 : i1, memref<?xi32>, memref<?xi32> -> memref<?xi32>
// CHECK-NEXT:   neura.ctrl_mov %42 -> %3 : memref<?xi32> memref<?xi32>
// CHECK-NEXT:   %43 = neura.merge %30, %34, %33 : i1, i32, i32 -> i32
// CHECK-NEXT:   %44 = neura.load_indexed %42[%41 : i64] memref<?xi32> : i32
// CHECK-NEXT:   %45 = "neura.mul"(%44, %40) : (i32, i32) -> i32
// CHECK-NEXT:   %46 = "neura.add"(%45, %39) : (i32, i32) -> i32
// CHECK-NEXT:   %47 = "neura.add"(%43, %46) : (i32, i32) -> i32
// CHECK-NEXT:   neura.ctrl_mov %47 -> %7 : i32 i32
// CHECK-NEXT:   %48 = "neura.add"(%41, %38) : (i64, i64) -> i64
// CHECK-NEXT:   neura.ctrl_mov %48 -> %8 : i64 i64
// CHECK-NEXT:   "neura.return"(%29) : (i32) -> ()
