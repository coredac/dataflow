// RUN: mlir-neura-opt --tosa-to-affine-pipeline %s | FileCheck %s

// Test TOSA to Affine lowering
func.func @simple_add(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func.func @simple_add
// CHECK-SAME: (%arg0: memref<16xf32>, %arg1: memref<16xf32>, %arg2: memref<16xf32>)
// CHECK: %alloc = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
// CHECK-NEXT: affine.for %arg3 = 0 to 16 {
// CHECK-NEXT:   %0 = affine.load %arg0[%arg3] : memref<16xf32>
// CHECK-NEXT:   %1 = affine.load %arg1[%arg3] : memref<16xf32>
// CHECK-NEXT:   %2 = arith.addf %0, %1 : f32
// CHECK-NEXT:   affine.store %2, %alloc[%arg3] : memref<16xf32>
// CHECK-NEXT: }
// CHECK-NEXT: memref.copy %alloc, %arg2
// CHECK-NEXT: return
