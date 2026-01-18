// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task --optimize-task-graph \
// RUN: | FileCheck %s

// Tests hyperblock fusion for nested loops with identical counter structures.
// Two independent nested loops in the same task should be fused.

module {
  func.func @test_nested_fusion(%A: memref<8x8xf32>, %B: memref<8x8xf32>,
                                 %C: memref<8x8xf32>, %D: memref<8x8xf32>) {
    // Outer loop creates a single task with two inner loops.
    affine.for %i = 0 to 8 {
      // First inner loop: copies A to C.
      affine.for %j = 0 to 8 {
        %v = affine.load %A[%i, %j] : memref<8x8xf32>
        affine.store %v, %C[%i, %j] : memref<8x8xf32>
      }
      // Second inner loop: copies B to D - independent from first.
      // Should be fused with first loop since same counter structure.
      affine.for %j = 0 to 8 {
        %v = affine.load %B[%i, %j] : memref<8x8xf32>
        affine.store %v, %D[%i, %j] : memref<8x8xf32>
      }
    }
    return
  }
}

// After optimization, both inner loops should be fused into ONE hyperblock.

// CHECK:      module {
// CHECK-NEXT:   func.func @test_nested_fusion(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>, %arg3: memref<8x8xf32>) {
// CHECK-NEXT:     %memory_outputs:2 = "taskflow.task"(%arg0, %arg1, %arg2, %arg3) <{operandSegmentSizes = array<i32: 4, 0>, resultSegmentSizes = array<i32: 2, 0>, task_name = "Task_0"}> ({
// CHECK-NEXT:     ^bb0(%arg4: memref<8x8xf32>, %arg5: memref<8x8xf32>, %arg6: memref<8x8xf32>, %arg7: memref<8x8xf32>):
// CHECK-NEXT:       %0 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// CHECK-NEXT:       %1 = taskflow.counter parent(%0 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// CHECK-NEXT:       %2 = taskflow.counter parent(%0 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// CHECK-NEXT:       taskflow.hyperblock indices(%0, %1 : index, index) {
// CHECK-NEXT:       ^bb0(%arg8: index, %arg9: index):
// CHECK-NEXT:         %3 = memref.load %arg4[%arg8, %arg9] : memref<8x8xf32>
// CHECK-NEXT:         memref.store %3, %arg6[%arg8, %arg9] : memref<8x8xf32>
// CHECK-NEXT:         %4 = memref.load %arg5[%arg8, %arg9] : memref<8x8xf32>
// CHECK-NEXT:         memref.store %4, %arg7[%arg8, %arg9] : memref<8x8xf32>
// CHECK-NEXT:       } -> ()
// CHECK-NEXT:       "taskflow.yield"(%arg6, %arg7) <{operandSegmentSizes = array<i32: 2, 0>}> : (memref<8x8xf32>, memref<8x8xf32>) -> ()
// CHECK-NEXT:     }) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> (memref<8x8xf32>, memref<8x8xf32>)
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }
