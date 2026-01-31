// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: -o %t.taskflow.mlir
// RUN: FileCheck %s --input-file=%t.taskflow.mlir --check-prefixes=TASKFLOW

// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: -o %t.hyperblock.mlir
// RUN: FileCheck %s --input-file=%t.hyperblock.mlir --check-prefixes=HYPERBLOCK

// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --canonicalize-task \
// RUN: -o %t.canonicalized.mlir
// RUN: FileCheck %s --input-file=%t.canonicalized.mlir --check-prefixes=CANONICALIZE

// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --canonicalize-task \
// RUN: --map-ct-on-cgra-array \
// RUN: -o %t.placement.mlir
// RUN: FileCheck %s --input-file=%t.placement.mlir --check-prefixes=PLACEMENT

#set = affine_set<(d0, d1) : (d0 - 3 == 0, d1 - 7 == 0)>
module attributes {} {
  func.func @_Z21irregularLoopExample1v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    %alloca_0 = memref.alloca() : memref<4x8xi32>
    %0 = affine.for %arg0 = 0 to 5 iter_args(%arg1 = %c0_i32) -> (i32) {
      %2 = arith.index_cast %arg0 : index to i32
      %3 = arith.addi %arg1, %2 : i32
      affine.yield %3 : i32
    }
    affine.for %arg0 = 0 to 4 {
      %2 = arith.index_cast %arg0 : index to i32
      %3 = arith.muli %2, %c8_i32 : i32
      affine.for %arg1 = 0 to 8 {
        %4 = arith.index_cast %arg1 : index to i32
        %5 = arith.addi %3, %4 : i32
        affine.store %5, %alloca_0[%arg0, %arg1] : memref<4x8xi32>
      }
      affine.for %arg1 = 0 to 8 {
        %4 = affine.load %alloca_0[%arg0, %arg1] : memref<4x8xi32>
        %5 = arith.addi %4, %0 : i32
        affine.if #set(%arg0, %arg1) {
          affine.store %5, %alloca[] : memref<i32>
          %6 = arith.muli %5, %c2_i32 : i32
          affine.store %6, %alloca[] : memref<i32>
        }
      }
    }
    %1 = affine.load %alloca[] : memref<i32>
    return %1 : i32
  }
}

// TASKFLOW:      #set = affine_set<(d0, d1) : (d0 - 3 == 0, d1 - 7 == 0)>
// TASKFLOW-NEXT: module {
// TASKFLOW-NEXT:   func.func @_Z21irregularLoopExample1v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// TASKFLOW-NEXT:     %c2_i32 = arith.constant 2 : i32
// TASKFLOW-NEXT:     %c8_i32 = arith.constant 8 : i32
// TASKFLOW-NEXT:     %c0_i32 = arith.constant 0 : i32
// TASKFLOW-NEXT:     %alloca = memref.alloca() : memref<i32>
// TASKFLOW-NEXT:     %alloca_0 = memref.alloca() : memref<4x8xi32>
// TASKFLOW-NEXT:     %value_outputs = "taskflow.task"(%c0_i32) <{operandSegmentSizes = array<i32: 0, 1>, resultSegmentSizes = array<i32: 0, 1>, task_name = "Task_0"}> ({
// TASKFLOW-NEXT:     ^bb0(%arg0: i32):
// TASKFLOW-NEXT:       %1 = affine.for %arg1 = 0 to 5 iter_args(%arg2 = %arg0) -> (i32) {
// TASKFLOW-NEXT:         %2 = arith.index_cast %arg1 : index to i32
// TASKFLOW-NEXT:         %3 = arith.addi %arg2, %2 : i32
// TASKFLOW-NEXT:         affine.yield %3 : i32
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       "taskflow.yield"(%1) <{operandSegmentSizes = array<i32: 0, 1>}> : (i32) -> ()
// TASKFLOW-NEXT:     }) : (i32) -> i32
// TASKFLOW-NEXT:     %memory_outputs:2 = "taskflow.task"(%alloca_0, %alloca, %c8_i32, %value_outputs, %c2_i32) <{operandSegmentSizes = array<i32: 2, 3>, resultSegmentSizes = array<i32: 2, 0>, task_name = "Task_1"}> ({
// TASKFLOW-NEXT:     ^bb0(%arg0: memref<4x8xi32>, %arg1: memref<i32>, %arg2: i32, %arg3: i32, %arg4: i32):
// TASKFLOW-NEXT:       affine.for %arg5 = 0 to 4 {
// TASKFLOW-NEXT:         %1 = arith.index_cast %arg5 : index to i32
// TASKFLOW-NEXT:         %2 = arith.muli %1, %arg2 : i32
// TASKFLOW-NEXT:         affine.for %arg6 = 0 to 8 {
// TASKFLOW-NEXT:           %3 = arith.index_cast %arg6 : index to i32
// TASKFLOW-NEXT:           %4 = arith.addi %2, %3 : i32
// TASKFLOW-NEXT:           affine.store %4, %arg0[%arg5, %arg6] : memref<4x8xi32>
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:         affine.for %arg6 = 0 to 8 {
// TASKFLOW-NEXT:           %3 = affine.load %arg0[%arg5, %arg6] : memref<4x8xi32>
// TASKFLOW-NEXT:           %4 = arith.addi %3, %arg3 : i32
// TASKFLOW-NEXT:           affine.if #set(%arg5, %arg6) {
// TASKFLOW-NEXT:             affine.store %4, %arg1[] : memref<i32>
// TASKFLOW-NEXT:             %5 = arith.muli %4, %arg4 : i32
// TASKFLOW-NEXT:             affine.store %5, %arg1[] : memref<i32>
// TASKFLOW-NEXT:           }
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       "taskflow.yield"(%arg0, %arg1) <{operandSegmentSizes = array<i32: 2, 0>}> : (memref<4x8xi32>, memref<i32>) -> ()
// TASKFLOW-NEXT:     }) : (memref<4x8xi32>, memref<i32>, i32, i32, i32) -> (memref<4x8xi32>, memref<i32>)
// TASKFLOW-NEXT:     %0 = affine.load %memory_outputs#1[] : memref<i32>
// TASKFLOW-NEXT:     return %0 : i32
// TASKFLOW-NEXT:   }
// TASKFLOW-NEXT: }

// HYPERBLOCK:      module {
// HYPERBLOCK-NEXT:   func.func @_Z21irregularLoopExample1v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// HYPERBLOCK-NEXT:     %c2_i32 = arith.constant 2 : i32
// HYPERBLOCK-NEXT:     %c8_i32 = arith.constant 8 : i32
// HYPERBLOCK-NEXT:     %c0_i32 = arith.constant 0 : i32
// HYPERBLOCK-NEXT:     %alloca = memref.alloca() : memref<i32>
// HYPERBLOCK-NEXT:     %alloca_0 = memref.alloca() : memref<4x8xi32>
// HYPERBLOCK-NEXT:     %value_outputs = "taskflow.task"(%c0_i32) <{operandSegmentSizes = array<i32: 0, 1>, resultSegmentSizes = array<i32: 0, 1>, task_name = "Task_0"}> ({
// HYPERBLOCK-NEXT:     ^bb0(%arg0: i32):
// HYPERBLOCK-NEXT:       %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 5 : index} : index
// HYPERBLOCK-NEXT:       %2 = "taskflow.hyperblock"(%1, %arg0) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg1: index, %arg2: i32):
// HYPERBLOCK-NEXT:         %3 = arith.index_cast %arg1 : index to i32
// HYPERBLOCK-NEXT:         %4 = arith.addi %arg2, %3 : i32
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield iter_args_next(%4 : i32) results(%4 : i32)
// HYPERBLOCK-NEXT:       }) : (index, i32) -> i32
// HYPERBLOCK-NEXT:       "taskflow.yield"(%2) <{operandSegmentSizes = array<i32: 0, 1>}> : (i32) -> ()
// HYPERBLOCK-NEXT:     }) : (i32) -> i32
// HYPERBLOCK-NEXT:     %memory_outputs:2 = "taskflow.task"(%alloca_0, %alloca, %c8_i32, %value_outputs, %c2_i32) <{operandSegmentSizes = array<i32: 2, 3>, resultSegmentSizes = array<i32: 2, 0>, task_name = "Task_1"}> ({
// HYPERBLOCK-NEXT:     ^bb0(%arg0: memref<4x8xi32>, %arg1: memref<i32>, %arg2: i32, %arg3: i32, %arg4: i32):
// HYPERBLOCK-NEXT:       %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 4 : index} : index
// HYPERBLOCK-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// HYPERBLOCK-NEXT:       %3 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// HYPERBLOCK-NEXT:       "taskflow.hyperblock"(%1, %2) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg5: index, %arg6: index):
// HYPERBLOCK-NEXT:         %4 = arith.index_cast %arg5 : index to i32
// HYPERBLOCK-NEXT:         %5 = arith.muli %4, %arg2 : i32
// HYPERBLOCK-NEXT:         %6 = arith.index_cast %arg6 : index to i32
// HYPERBLOCK-NEXT:         %7 = arith.addi %5, %6 : i32
// HYPERBLOCK-NEXT:         memref.store %7, %arg0[%arg5, %arg6] : memref<4x8xi32>
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:       }) : (index, index) -> ()
// HYPERBLOCK-NEXT:       "taskflow.hyperblock"(%1, %3) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg5: index, %arg6: index):
// HYPERBLOCK-NEXT:         %4 = memref.load %arg0[%arg5, %arg6] : memref<4x8xi32>
// HYPERBLOCK-NEXT:         %5 = arith.addi %4, %arg3 : i32
// HYPERBLOCK-NEXT:         %c0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:         %c-3 = arith.constant -3 : index
// HYPERBLOCK-NEXT:         %6 = arith.addi %arg5, %c-3 : index
// HYPERBLOCK-NEXT:         %7 = arith.cmpi eq, %6, %c0 : index
// HYPERBLOCK-NEXT:         %c-7 = arith.constant -7 : index
// HYPERBLOCK-NEXT:         %8 = arith.addi %arg6, %c-7 : index
// HYPERBLOCK-NEXT:         %9 = arith.cmpi eq, %8, %c0 : index
// HYPERBLOCK-NEXT:         %10 = arith.andi %7, %9 : i1
// HYPERBLOCK-NEXT:         scf.if %10 {
// HYPERBLOCK-NEXT:           memref.store %5, %arg1[] : memref<i32>
// HYPERBLOCK-NEXT:           %11 = arith.muli %5, %arg4 : i32
// HYPERBLOCK-NEXT:           memref.store %11, %arg1[] : memref<i32>
// HYPERBLOCK-NEXT:         }
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:       }) : (index, index) -> ()
// HYPERBLOCK-NEXT:       "taskflow.yield"(%arg0, %arg1) <{operandSegmentSizes = array<i32: 2, 0>}> : (memref<4x8xi32>, memref<i32>) -> ()
// HYPERBLOCK-NEXT:     }) : (memref<4x8xi32>, memref<i32>, i32, i32, i32) -> (memref<4x8xi32>, memref<i32>)
// HYPERBLOCK-NEXT:     %0 = affine.load %memory_outputs#1[] : memref<i32>
// HYPERBLOCK-NEXT:     return %0 : i32
// HYPERBLOCK-NEXT:   }
// HYPERBLOCK-NEXT: }



// CANONICALIZE:      module {
// CANONICALIZE-NEXT:   func.func @_Z21irregularLoopExample1v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CANONICALIZE-NEXT:     %c2_i32 = arith.constant 2 : i32
// CANONICALIZE-NEXT:     %c8_i32 = arith.constant 8 : i32
// CANONICALIZE-NEXT:     %c0_i32 = arith.constant 0 : i32
// CANONICALIZE-NEXT:     %alloca = memref.alloca() : memref<i32>
// CANONICALIZE-NEXT:     %alloca_0 = memref.alloca() : memref<4x8xi32>
// CANONICALIZE-NEXT:     %value_outputs = "taskflow.task"(%c0_i32) <{operandSegmentSizes = array<i32: 0, 1>, resultSegmentSizes = array<i32: 0, 1>, task_name = "Task_0"}> ({
// CANONICALIZE-NEXT:     ^bb0(%arg0: i32):
// CANONICALIZE-NEXT:       %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 5 : index} : index
// CANONICALIZE-NEXT:       %2 = "taskflow.hyperblock"(%1, %arg0) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CANONICALIZE-NEXT:       ^bb0(%arg1: index, %arg2: i32):
// CANONICALIZE-NEXT:         %3 = arith.index_cast %arg1 : index to i32
// CANONICALIZE-NEXT:         %4 = arith.addi %arg2, %3 : i32
// CANONICALIZE-NEXT:         taskflow.hyperblock.yield iter_args_next(%4 : i32) results(%4 : i32)
// CANONICALIZE-NEXT:       }) : (index, i32) -> i32
// CANONICALIZE-NEXT:       "taskflow.yield"(%2) <{operandSegmentSizes = array<i32: 0, 1>}> : (i32) -> ()
// CANONICALIZE-NEXT:     }) : (i32) -> i32
// CANONICALIZE-NEXT:     %memory_outputs = "taskflow.task"(%alloca_0, %c8_i32, %alloca_0) <{operandSegmentSizes = array<i32: 1, 2>, resultSegmentSizes = array<i32: 1, 0>, task_name = "Task_1"}> ({
// CANONICALIZE-NEXT:     ^bb0(%arg0: memref<4x8xi32>, %arg1: i32, %arg2: memref<4x8xi32>):
// CANONICALIZE-NEXT:       %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 4 : index} : index
// CANONICALIZE-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// CANONICALIZE-NEXT:       "taskflow.hyperblock"(%1, %2) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// CANONICALIZE-NEXT:       ^bb0(%arg3: index, %arg4: index):
// CANONICALIZE-NEXT:         %3 = arith.index_cast %arg3 : index to i32
// CANONICALIZE-NEXT:         %4 = arith.muli %3, %arg1 : i32
// CANONICALIZE-NEXT:         %5 = arith.index_cast %arg4 : index to i32
// CANONICALIZE-NEXT:         %6 = arith.addi %4, %5 : i32
// CANONICALIZE-NEXT:         memref.store %6, %arg2[%arg3, %arg4] : memref<4x8xi32>
// CANONICALIZE-NEXT:         taskflow.hyperblock.yield
// CANONICALIZE-NEXT:       }) : (index, index) -> ()
// CANONICALIZE-NEXT:       "taskflow.yield"(%arg2) <{operandSegmentSizes = array<i32: 1, 0>}> : (memref<4x8xi32>) -> ()
// CANONICALIZE-NEXT:     }) : (memref<4x8xi32>, i32, memref<4x8xi32>) -> memref<4x8xi32>
// CANONICALIZE-NEXT:     %memory_outputs_1 = "taskflow.task"(%memory_outputs, %alloca, %alloca_0, %value_outputs, %alloca, %c2_i32) <{operandSegmentSizes = array<i32: 2, 4>, resultSegmentSizes = array<i32: 1, 0>, task_name = "Task_2"}> ({
// CANONICALIZE-NEXT:     ^bb0(%arg0: memref<4x8xi32>, %arg1: memref<i32>, %arg2: memref<4x8xi32>, %arg3: i32, %arg4: memref<i32>, %arg5: i32):
// CANONICALIZE-NEXT:       %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 4 : index} : index
// CANONICALIZE-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// CANONICALIZE-NEXT:       "taskflow.hyperblock"(%1, %2) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// CANONICALIZE-NEXT:       ^bb0(%arg6: index, %arg7: index):
// CANONICALIZE-NEXT:         %3 = memref.load %arg2[%arg6, %arg7] : memref<4x8xi32>
// CANONICALIZE-NEXT:         %4 = arith.addi %3, %arg3 : i32
// CANONICALIZE-NEXT:         %c0 = arith.constant 0 : index
// CANONICALIZE-NEXT:         %c-3 = arith.constant -3 : index
// CANONICALIZE-NEXT:         %5 = arith.addi %arg6, %c-3 : index
// CANONICALIZE-NEXT:         %6 = arith.cmpi eq, %5, %c0 : index
// CANONICALIZE-NEXT:         %c-7 = arith.constant -7 : index
// CANONICALIZE-NEXT:         %7 = arith.addi %arg7, %c-7 : index
// CANONICALIZE-NEXT:         %8 = arith.cmpi eq, %7, %c0 : index
// CANONICALIZE-NEXT:         %9 = arith.andi %6, %8 : i1
// CANONICALIZE-NEXT:         scf.if %9 {
// CANONICALIZE-NEXT:           memref.store %4, %arg4[] : memref<i32>
// CANONICALIZE-NEXT:           %10 = arith.muli %4, %arg5 : i32
// CANONICALIZE-NEXT:           memref.store %10, %arg4[] : memref<i32>
// CANONICALIZE-NEXT:         }
// CANONICALIZE-NEXT:         taskflow.hyperblock.yield
// CANONICALIZE-NEXT:       }) : (index, index) -> ()
// CANONICALIZE-NEXT:       "taskflow.yield"(%arg4) <{operandSegmentSizes = array<i32: 1, 0>}> : (memref<i32>) -> ()
// CANONICALIZE-NEXT:     }) : (memref<4x8xi32>, memref<i32>, memref<4x8xi32>, i32, memref<i32>, i32) -> memref<i32>
// CANONICALIZE-NEXT:     %0 = affine.load %memory_outputs_1[] : memref<i32>
// CANONICALIZE-NEXT:     return %0 : i32
// CANONICALIZE-NEXT:   }
// CANONICALIZE-NEXT: }

// PLACEMENT: task_name = "Task_0"
// PLACEMENT: cgra_col = 0 : i32, cgra_count = 1 : i32, cgra_row = 0 : i32
// PLACEMENT: task_name = "Task_1"
// PLACEMENT: cgra_col = 1 : i32, cgra_count = 1 : i32, cgra_row = 1 : i32
// PLACEMENT: task_name = "Task_2"
// PLACEMENT: cgra_col = 1 : i32, cgra_count = 1 : i32, cgra_row = 0 : i32
