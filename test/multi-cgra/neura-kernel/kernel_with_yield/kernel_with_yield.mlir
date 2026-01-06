// Wraps the innermost loop within neura.kernel operation.
// RUN: mlir-neura-opt %s \
// RUN:  --wrap-loop-in-kernel \
// RUN:  | FileCheck %s

module attributes {} {
  func.func @_Z27perfect_nested_reduction_2dPA128_i(%arg0: memref<?x128xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.for %arg1 = 0 to 128 iter_args(%arg2 = %c0_i32) -> (i32) {
      %1 = affine.for %arg3 = 0 to 128 iter_args(%arg4 = %arg2) -> (i32) {
        %2 = affine.load %arg0[%arg1, %arg3] : memref<?x128xi32>
        %3 = arith.addi %arg4, %2 : i32
        affine.yield %3 : i32
      }
      affine.yield %1 : i32
    }
    return %0 : i32
  }
}

 // CHECK:      module {
 // CHECK-NEXT:   func.func @_Z27perfect_nested_reduction_2dPA128_i(%arg0: memref<?x128xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
 // CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
 // CHECK-NEXT:     %0 = affine.for %arg1 = 0 to 128 iter_args(%arg2 = %c0_i32) -> (i32) {
 // CHECK-NEXT:       %1 = neura.kernel ins(%arg0, %arg1 : memref<?x128xi32>, index) attributes {kernel_name = "kernel_0"} {
 // CHECK-NEXT:         %2 = affine.for %arg3 = 0 to 128 iter_args(%arg4 = %arg2) -> (i32) {
 // CHECK-NEXT:           %3 = affine.load %arg0[%arg1, %arg3] : memref<?x128xi32>
 // CHECK-NEXT:           %4 = arith.addi %arg4, %3 : i32
 // CHECK-NEXT:           affine.yield %4 : i32
 // CHECK-NEXT:         }
 // CHECK-NEXT:         neura.yield %2 : i32
 // CHECK-NEXT:       } : i32
 // CHECK-NEXT:       affine.yield %1 : i32
 // CHECK-NEXT:     }
 // CHECK-NEXT:     return %0 : i32
 // CHECK-NEXT:   }
 // CHECK-NEXT: }
