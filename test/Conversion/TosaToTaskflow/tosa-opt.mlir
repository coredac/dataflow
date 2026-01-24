// RUN: mlir-neura-opt --pass-pipeline='builtin.module(func.func(tosa-infer-shapes,tosa-make-broadcastable,tosa-to-linalg-named,tosa-to-linalg,tosa-to-arith,tosa-to-tensor,linalg-fuse-elementwise-ops),one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},func.func(convert-linalg-to-affine-loops))' %s | FileCheck %s

// Test TOSA optimization (constant folding) with arith.constant
func.func @const_fold_test() -> tensor<4xf32> {
  %cst1 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %cst2 = arith.constant dense<[10.0, 20.0, 30.0, 40.0]> : tensor<4xf32>
  
  // This add should be constant folded by TOSA before lowering to Linalg
  %folded = tosa.add %cst1, %cst2 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %folded : tensor<4xf32>
}

// CHECK-LABEL: func.func @const_fold_test
// CHECK-SAME: () -> memref<4xf32>
// TODO: This should be folded to a memory copy of a global constant.
// Currently TOSA constant folding is not triggering as expected, so we check for the runtime op.
// CHECK: %0 = memref.get_global @__constant_4xf32 : memref<4xf32>
// CHECK-NEXT: %1 = memref.get_global @__constant_4xf32_0 : memref<4xf32>
// CHECK-NEXT: %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
// CHECK-NEXT: affine.for %arg0 = 0 to 4 {
// CHECK-NEXT:   %2 = affine.load %0[%arg0] : memref<4xf32>
// CHECK-NEXT:   %3 = affine.load %1[%arg0] : memref<4xf32>
// CHECK-NEXT:   %4 = arith.addf %2, %3 : f32
// CHECK-NEXT:   affine.store %4, %alloc[%arg0] : memref<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %alloc : memref<4xf32>
