// RUN: mlir-neura-opt --pass-pipeline='builtin.module(func.func(tosa-infer-shapes,tosa-make-broadcastable,tosa-to-linalg-named,tosa-to-linalg,tosa-to-arith,tosa-to-tensor,linalg-fuse-elementwise-ops),one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},func.func(convert-linalg-to-affine-loops))' %s | FileCheck %s

// Test Linalg fusion capability
// We chain multiple elementwise ops. If fusion works, we should see ONE loop nest.
func.func @fusion_test(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tosa.add %arg0, %arg0 : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  %1 = tosa.mul %0, %0 : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  
  // A simple relu-like operation: max(0, x)
  %zeros = "tosa.const"() {value = dense<0.0> : tensor<16xf32>} : () -> tensor<16xf32>
  %2 = tosa.maximum %1, %zeros : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  
  return %2 : tensor<16xf32>
}

// CHECK-LABEL: func.func @fusion_test
// CHECK-SAME: (%arg0: memref<16xf32>) -> memref<16xf32>
// CHECK: %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %alloc = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
// CHECK-NEXT: affine.for %arg1 = 0 to 16 {
// CHECK-NEXT:   %0 = affine.load %arg0[%arg1] : memref<16xf32>
// CHECK-NEXT:   %1 = arith.addf %0, %0 : f32
// CHECK-NEXT:   %2 = arith.mulf %1, %1 : f32
// CHECK-NEXT:   %3 = arith.maximumf %2, %cst : f32
// CHECK-NEXT:   affine.store %3, %alloc[%arg1] : memref<16xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %alloc : memref<16xf32>
