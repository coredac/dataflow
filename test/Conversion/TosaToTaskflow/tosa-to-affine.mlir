// RUN: mlir-neura-opt --pass-pipeline='builtin.module(func.func(tosa-infer-shapes,tosa-make-broadcastable,tosa-to-linalg-named,tosa-to-linalg,tosa-to-arith,tosa-to-tensor,linalg-fuse-elementwise-ops),one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},func.func(convert-linalg-to-affine-loops))' %s | FileCheck %s

// Test TOSA to Affine lowering
func.func @simple_add(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func.func @simple_add
// CHECK-SAME: (%arg0: memref<16xf32>, %arg1: memref<16xf32>) -> memref<16xf32>
// CHECK: %alloc = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
// CHECK-NEXT: affine.for %arg2 = 0 to 16 {
// CHECK-NEXT:   %0 = affine.load %arg0[%arg2] : memref<16xf32>
// CHECK-NEXT:   %1 = affine.load %arg1[%arg2] : memref<16xf32>
// CHECK-NEXT:   %2 = arith.addf %0, %1 : f32
// CHECK-NEXT:   affine.store %2, %alloc[%arg2] : memref<16xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %alloc : memref<16xf32>
