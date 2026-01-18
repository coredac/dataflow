// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task --optimize-task-graph \
// RUN: | FileCheck %s

// Tests hyperblock fusion behavior for adjacent hyperblocks with identical counter structures.
// Two independent top-level loops with the same bounds become separate tasks, each with its own hyperblock (no cross-task fusion).

module {
  func.func @test_hyperblock_fusion(%A: memref<16xf32>, %B: memref<16xf32>, %scale: f32) {
    // First loop: reads A, writes A.
    affine.for %i = 0 to 16 {
      %v = affine.load %A[%i] : memref<16xf32>
      %scaled = arith.mulf %v, %scale : f32
      affine.store %scaled, %A[%i] : memref<16xf32>
    }
    
    // Second loop: reads B, writes B - independent from first loop.
    affine.for %i = 0 to 16 {
      %v = affine.load %B[%i] : memref<16xf32>
      %scaled = arith.mulf %v, %scale : f32
      affine.store %scaled, %B[%i] : memref<16xf32>
    }
    
    return
  }
}

// After conversion and optimization, both top-level loops become separate tasks.
// Cross-task fusion is not performed; each task has one hyperblock.

// CHECK:      module {
// CHECK-NEXT:   func.func @test_hyperblock_fusion(%arg0: memref<16xf32>, %arg1: memref<16xf32>, %arg2: f32) {
// CHECK-NEXT:     %memory_outputs = "taskflow.task"(%arg0, %arg2) <{operandSegmentSizes = array<i32: 1, 1>, resultSegmentSizes = array<i32: 1, 0>, task_name = "Task_0"}> ({
// CHECK-NEXT:     ^bb0(%arg3: memref<16xf32>, %arg4: f32):
// CHECK-NEXT:       %0 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 16 : index} : index
// CHECK-NEXT:       taskflow.hyperblock indices(%0 : index) {
// CHECK-NEXT:       ^bb0(%arg5: index):
// CHECK-NEXT:         %1 = memref.load %arg3[%arg5] : memref<16xf32>
// CHECK-NEXT:         %2 = arith.mulf %1, %arg4 : f32
// CHECK-NEXT:         memref.store %2, %arg3[%arg5] : memref<16xf32>
// CHECK-NEXT:       } -> ()
// CHECK-NEXT:       "taskflow.yield"(%arg3) <{operandSegmentSizes = array<i32: 1, 0>}> : (memref<16xf32>) -> ()
// CHECK-NEXT:     }) : (memref<16xf32>, f32) -> memref<16xf32>
// CHECK-NEXT:     %memory_outputs_0 = "taskflow.task"(%arg1, %arg2) <{operandSegmentSizes = array<i32: 1, 1>, resultSegmentSizes = array<i32: 1, 0>, task_name = "Task_1"}> ({
// CHECK-NEXT:     ^bb0(%arg3: memref<16xf32>, %arg4: f32):
// CHECK-NEXT:       %0 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 16 : index} : index
// CHECK-NEXT:       taskflow.hyperblock indices(%0 : index) {
// CHECK-NEXT:       ^bb0(%arg5: index):
// CHECK-NEXT:         %1 = memref.load %arg3[%arg5] : memref<16xf32>
// CHECK-NEXT:         %2 = arith.mulf %1, %arg4 : f32
// CHECK-NEXT:         memref.store %2, %arg3[%arg5] : memref<16xf32>
// CHECK-NEXT:       } -> ()
// CHECK-NEXT:       "taskflow.yield"(%arg3) <{operandSegmentSizes = array<i32: 1, 0>}> : (memref<16xf32>) -> ()
// CHECK-NEXT:     }) : (memref<16xf32>, f32) -> memref<16xf32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }
