// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura | FileCheck %s
module attributes {} {
  func.func @_Z10bert_node8PA128_A1_KfPA128_A1_f(%arg0: memref<?x128x1xf32>, %arg1: memref<?x128x1xf32>) attributes {} {
    %cst = arith.constant 7.680000e+02 : f32
    affine.for %arg2 = 0 to 128 {
      %0 = affine.load %arg0[0, %arg2, 0] : memref<?x128x1xf32>
      %1 = arith.divf %0, %cst : f32
      affine.store %1, %arg1[0, %arg2, 0] : memref<?x128x1xf32>
    }
    return
  }
}

// CHECK:  func.func @_Z10bert_node8PA128_A1_KfPA128_A1_f(%arg0: memref<?x128x1xf32>, %arg1: memref<?x128x1xf32>) attributes {accelerator = "neura"} {
// CHECK-NEXT: %0 = "neura.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT: %1 = "neura.constant"() <{value = 128 : index}> : () -> index
// CHECK-NEXT: %2 = "neura.constant"() <{value = 7.680000e+02 : f32}> : () -> f32
// CHECK-NEXT: %3 = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT: %4 = builtin.unrealized_conversion_cast %3 : index to i64
// CHECK-NEXT: llvm.br ^bb1(%4 : i64)
// CHECK-NEXT: ^bb1(%5: i64):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT: %6 = builtin.unrealized_conversion_cast %5 : i64 to index
// CHECK-NEXT: %7 = "neura.icmp"(%6, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT: llvm.cond_br %7, ^bb2, ^bb3
// CHECK-NEXT: ^bb2:  // pred: ^bb1
// CHECK-NEXT: %8 = memref.load %arg0[%3, %6, %3] : memref<?x128x1xf32>
// CHECK-NEXT: %9 = "neura.fdiv"(%8, %2) : (f32, f32) -> f32
// CHECK-NEXT: memref.store %9, %arg1[%3, %6, %3] : memref<?x128x1xf32>
// CHECK-NEXT: %10 = "neura.add"(%6, %0) : (index, index) -> index
// CHECK-NEXT: %11 = builtin.unrealized_conversion_cast %10 : index to i64
// CHECK-NEXT: llvm.br ^bb1(%11 : i64)
// CHECK-NEXT: ^bb3:  // pred: ^bb1
// CHECK-NEXT: return
