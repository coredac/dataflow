// RUN: mlir-neura-opt --convert-affine-to-taskflow %s 2>/dev/null | FileCheck %s

// Test Affine to Taskflow conversion
module {
  func.func @simple_add(%arg0: memref<16xf32>, %arg1: memref<16xf32>, %arg2: memref<16xf32>) {
    affine.for %i = 0 to 16 {
      %0 = affine.load %arg0[%i] : memref<16xf32>
      %1 = affine.load %arg1[%i] : memref<16xf32>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %arg2[%i] : memref<16xf32>
    }
    return
  }
}

// CHECK-LABEL: func.func @simple_add
// CHECK-NEXT: %memory_outputs = "taskflow.task"(%arg0, %arg1, %arg2)
// CHECK-SAME: task_name = "Task_0"
// CHECK-NEXT: ^bb0(%arg3: memref<16xf32>, %arg4: memref<16xf32>, %arg5: memref<16xf32>):
// CHECK-NEXT:   affine.for %arg6 = 0 to 16 {
// CHECK-NEXT:     %0 = affine.load %arg3[%arg6] : memref<16xf32>
// CHECK-NEXT:     %1 = affine.load %arg4[%arg6] : memref<16xf32>
// CHECK-NEXT:     %2 = arith.addf %0, %1 : f32
// CHECK-NEXT:     affine.store %2, %arg5[%arg6] : memref<16xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   "taskflow.yield"(%arg5)
// CHECK: return
