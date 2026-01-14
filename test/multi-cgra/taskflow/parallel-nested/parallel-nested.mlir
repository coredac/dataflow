// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: | FileCheck %s --check-prefixes=TASKFLOW

// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: | FileCheck %s --check-prefixes=HYPERBLOCK

module {
  // Example: Parallel nested loops scenario
  // Task 0: Single-level loop (vector scaling)
  // Task 1: Two-level nested loop (matrix multiplication)
  func.func @parallel_nested_example(%A: memref<16xf32>, 
                                      %B: memref<8x8xf32>, 
                                      %C: memref<8x8xf32>,
                                      %D: memref<8x8xf32>,
                                      %scalar: f32) {
    // Task 0: Single-level loop - Vector scaling
    // Computes: A[i] = A[i] * scalar
    affine.for %i = 0 to 16 {
      %v = affine.load %A[%i] : memref<16xf32>
      %scaled = arith.mulf %v, %scalar : f32
      affine.store %scaled, %A[%i] : memref<16xf32>
    }
    
    // Task 1: Two-level nested loop - Matrix multiplication
    // Computes: D[i][j] = B[i][j] * C[i][j] (element-wise)
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        %b_val = affine.load %B[%i, %j] : memref<8x8xf32>
        %c_val = affine.load %C[%i, %j] : memref<8x8xf32>
        %product = arith.mulf %b_val, %c_val : f32
        affine.store %product, %D[%i, %j] : memref<8x8xf32>
      }
    }
    return
  }
}

// TASKFLOW:      module {
// TASKFLOW-NEXT:   func.func @parallel_nested_example(%arg0: memref<16xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<8x8xf32>, %arg4: f32) {
// TASKFLOW-NEXT:     %memory_outputs = "taskflow.task"(%arg0, %arg4) <{operandSegmentSizes = array<i32: 1, 1>, resultSegmentSizes = array<i32: 1, 0>, task_name = "Task_0"}> ({
// TASKFLOW-NEXT:     ^bb0(%arg5: memref<16xf32>, %arg6: f32):
// TASKFLOW-NEXT:       affine.for %arg7 = 0 to 16 {
// TASKFLOW-NEXT:         %0 = affine.load %arg5[%arg7] : memref<16xf32>
// TASKFLOW-NEXT:         %1 = arith.mulf %0, %arg6 : f32
// TASKFLOW-NEXT:         affine.store %1, %arg5[%arg7] : memref<16xf32>
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       "taskflow.yield"(%arg5) <{operandSegmentSizes = array<i32: 1, 0>}> : (memref<16xf32>) -> ()
// TASKFLOW-NEXT:     }) : (memref<16xf32>, f32) -> memref<16xf32>
// TASKFLOW-NEXT:     %memory_outputs_0 = "taskflow.task"(%arg1, %arg2, %arg3) <{operandSegmentSizes = array<i32: 3, 0>, resultSegmentSizes = array<i32: 1, 0>, task_name = "Task_1"}> ({
// TASKFLOW-NEXT:     ^bb0(%arg5: memref<8x8xf32>, %arg6: memref<8x8xf32>, %arg7: memref<8x8xf32>):
// TASKFLOW-NEXT:       affine.for %arg8 = 0 to 8 {
// TASKFLOW-NEXT:         affine.for %arg9 = 0 to 8 {
// TASKFLOW-NEXT:           %0 = affine.load %arg5[%arg8, %arg9] : memref<8x8xf32>
// TASKFLOW-NEXT:           %1 = affine.load %arg6[%arg8, %arg9] : memref<8x8xf32>
// TASKFLOW-NEXT:           %2 = arith.mulf %0, %1 : f32
// TASKFLOW-NEXT:           affine.store %2, %arg7[%arg8, %arg9] : memref<8x8xf32>
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       "taskflow.yield"(%arg7) <{operandSegmentSizes = array<i32: 1, 0>}> : (memref<8x8xf32>) -> ()
// TASKFLOW-NEXT:     }) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> memref<8x8xf32>
// TASKFLOW-NEXT:     return
// TASKFLOW-NEXT:   }
// TASKFLOW-NEXT: }

// HYPERBLOCK:      module {
// HYPERBLOCK-NEXT:   func.func @parallel_nested_example(%arg0: memref<16xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<8x8xf32>, %arg4: f32) {
// HYPERBLOCK-NEXT:     %memory_outputs = "taskflow.task"(%arg0, %arg4) <{operandSegmentSizes = array<i32: 1, 1>, resultSegmentSizes = array<i32: 1, 0>, task_name = "Task_0"}> ({
// HYPERBLOCK-NEXT:     ^bb0(%arg5: memref<16xf32>, %arg6: f32):
// HYPERBLOCK-NEXT:       %0 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 16 : index} : index
// HYPERBLOCK-NEXT:       taskflow.hyperblock indices(%0 : index) {
// HYPERBLOCK-NEXT:       ^bb0(%arg7: index):
// HYPERBLOCK-NEXT:         %1 = memref.load %arg5[%arg7] : memref<16xf32>
// HYPERBLOCK-NEXT:         %2 = arith.mulf %1, %arg6 : f32
// HYPERBLOCK-NEXT:         memref.store %2, %arg5[%arg7] : memref<16xf32>
// HYPERBLOCK-NEXT:       } -> ()
// HYPERBLOCK-NEXT:       "taskflow.yield"(%arg5) <{operandSegmentSizes = array<i32: 1, 0>}> : (memref<16xf32>) -> ()
// HYPERBLOCK-NEXT:     }) : (memref<16xf32>, f32) -> memref<16xf32>
// HYPERBLOCK-NEXT:     %memory_outputs_0 = "taskflow.task"(%arg1, %arg2, %arg3) <{operandSegmentSizes = array<i32: 3, 0>, resultSegmentSizes = array<i32: 1, 0>, task_name = "Task_1"}> ({
// HYPERBLOCK-NEXT:     ^bb0(%arg5: memref<8x8xf32>, %arg6: memref<8x8xf32>, %arg7: memref<8x8xf32>):
// HYPERBLOCK-NEXT:       %0 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// HYPERBLOCK-NEXT:       %1 = taskflow.counter parent(%0 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// HYPERBLOCK-NEXT:       taskflow.hyperblock indices(%0, %1 : index, index) {
// HYPERBLOCK-NEXT:       ^bb0(%arg8: index, %arg9: index):
// HYPERBLOCK-NEXT:         %2 = memref.load %arg5[%arg8, %arg9] : memref<8x8xf32>
// HYPERBLOCK-NEXT:         %3 = memref.load %arg6[%arg8, %arg9] : memref<8x8xf32>
// HYPERBLOCK-NEXT:         %4 = arith.mulf %2, %3 : f32
// HYPERBLOCK-NEXT:         memref.store %4, %arg7[%arg8, %arg9] : memref<8x8xf32>
// HYPERBLOCK-NEXT:       } -> ()
// HYPERBLOCK-NEXT:       "taskflow.yield"(%arg7) <{operandSegmentSizes = array<i32: 1, 0>}> : (memref<8x8xf32>) -> ()
// HYPERBLOCK-NEXT:     }) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> memref<8x8xf32>
// HYPERBLOCK-NEXT:     return
// HYPERBLOCK-NEXT:   }
// HYPERBLOCK-NEXT: }