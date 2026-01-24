// RUN: mlir-neura-opt --pass-pipeline='builtin.module(func.func(tosa-infer-shapes,tosa-make-broadcastable,tosa-to-linalg-named,tosa-to-linalg,tosa-to-arith,tosa-to-tensor,linalg-fuse-elementwise-ops),one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},func.func(convert-linalg-to-affine-loops),convert-affine-to-taskflow)' %s 2>&1 | FileCheck %s
// Simple TOSA add lowering test

func.func @simple_add(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func.func @simple_add
// CHECK: %alloc = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
// CHECK: %[[RES:.*]] = "taskflow.task"(%arg0, %arg1, %alloc)
// CHECK-SAME: task_name = "Task_0"
// CHECK-NEXT: ^bb0(%[[BA1:.*]]: memref<16xf32>, %[[BA2:.*]]: memref<16xf32>, %[[BA3:.*]]: memref<16xf32>):
// CHECK-NEXT:   affine.for %[[IV:.*]] = 0 to 16 {
// CHECK-NEXT:     %0 = affine.load %[[BA1]][%[[IV]]] : memref<16xf32>
// CHECK-NEXT:     %1 = affine.load %[[BA2]][%[[IV]]] : memref<16xf32>
// CHECK-NEXT:     %2 = arith.addf %0, %1 : f32
// CHECK-NEXT:     affine.store %2, %[[BA3]][%[[IV]]] : memref<16xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   "taskflow.yield"(%[[BA3]])
// CHECK: return %[[RES]] : memref<16xf32>
