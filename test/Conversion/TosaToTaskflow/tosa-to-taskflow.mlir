// RUN: mlir-neura-opt --tosa-to-taskflow-pipeline %s 2>&1 | FileCheck %s
// Simple TOSA add lowering test

func.func @simple_add(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func.func @simple_add
// CHECK: %alloc = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
// CHECK-NEXT: %[[RES:.*]] = "taskflow.task"(%arg0, %arg1, %alloc)
// CHECK-SAME: task_name = "Task_0"
// CHECK-NEXT: ^bb0(%arg3: memref<16xf32>, %arg4: memref<16xf32>, %arg5: memref<16xf32>):
// CHECK-NEXT:   affine.for %arg6 = 0 to 16 {
// CHECK-NEXT:     %0 = affine.load %arg3[%arg6] : memref<16xf32>
// CHECK-NEXT:     %1 = affine.load %arg4[%arg6] : memref<16xf32>
// CHECK-NEXT:     %2 = arith.addf %0, %1 : f32
// CHECK-NEXT:     affine.store %2, %arg5[%arg6] : memref<16xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   "taskflow.yield"(%arg5)
// CHECK: memref.copy %[[RES]], %arg2
// CHECK-NEXT: return
