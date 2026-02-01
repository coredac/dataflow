// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: -o %t.serialized.mlir
// RUN: FileCheck %s --input-file=%t.serialized.mlir --check-prefixes=SERIALIZED

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: -o %t.taskflow.mlir
// RUN: FileCheck %s --input-file=%t.taskflow.mlir --check-prefixes=TASKFLOW

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: -o %t.hyperblock.mlir
// RUN: FileCheck %s --input-file=%t.hyperblock.mlir --check-prefixes=HYPERBLOCK

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

// SERIALIZED:      module {
// SERIALIZED-NEXT:   func.func @_Z21irregularLoopExample1v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// SERIALIZED-NEXT:     %c2_i32 = arith.constant 2 : i32
// SERIALIZED-NEXT:     %c8_i32 = arith.constant 8 : i32
// SERIALIZED-NEXT:     %c0_i32 = arith.constant 0 : i32
// SERIALIZED-NEXT:     %alloca = memref.alloca() : memref<i32>
// SERIALIZED-NEXT:     %alloca_0 = memref.alloca() : memref<4x8xi32>
// SERIALIZED-NEXT:     %0 = affine.for %arg0 = 0 to 5 iter_args(%arg1 = %c0_i32) -> (i32) {
// SERIALIZED-NEXT:       %2 = arith.index_cast %arg0 : index to i32
// SERIALIZED-NEXT:       %3 = arith.addi %arg1, %2 : i32
// SERIALIZED-NEXT:       affine.yield %3 : i32
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     affine.for %arg0 = 0 to 4 {
// SERIALIZED-NEXT:       %2 = arith.index_cast %arg0 : index to i32
// SERIALIZED-NEXT:       %3 = arith.muli %2, %c8_i32 : i32
// SERIALIZED-NEXT:       affine.for %arg1 = 0 to 8 {
// SERIALIZED-NEXT:         %4 = arith.index_cast %arg1 : index to i32
// SERIALIZED-NEXT:         %5 = arith.addi %3, %4 : i32
// SERIALIZED-NEXT:         affine.store %5, %alloca_0[%arg0, %arg1] : memref<4x8xi32>
// SERIALIZED-NEXT:       }
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     affine.for %arg0 = 0 to 4 {
// SERIALIZED-NEXT:       %2 = arith.index_cast %arg0 : index to i32
// SERIALIZED-NEXT:       %3 = arith.muli %2, %c8_i32 : i32
// SERIALIZED-NEXT:       affine.for %arg1 = 0 to 8 {
// SERIALIZED-NEXT:         %4 = affine.load %alloca_0[%arg0, %arg1] : memref<4x8xi32>
// SERIALIZED-NEXT:         %5 = arith.addi %4, %0 : i32
// SERIALIZED-NEXT:         affine.if #set(%arg0, %arg1) {
// SERIALIZED-NEXT:           affine.store %5, %alloca[] : memref<i32>
// SERIALIZED-NEXT:           %6 = arith.muli %5, %c2_i32 : i32
// SERIALIZED-NEXT:           affine.store %6, %alloca[] : memref<i32>
// SERIALIZED-NEXT:         }
// SERIALIZED-NEXT:       }
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     %1 = affine.load %alloca[] : memref<i32>
// SERIALIZED-NEXT:     return %1 : i32
// SERIALIZED-NEXT:   }
// SERIALIZED-NEXT: }

// TASKFLOW:      #set = affine_set<(d0, d1) : (d0 - 3 == 0, d1 - 7 == 0)>
// TASKFLOW-NEXT: module {
// TASKFLOW-NEXT:   func.func @_Z21irregularLoopExample1v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// TASKFLOW-NEXT:     %c2_i32 = arith.constant 2 : i32
// TASKFLOW-NEXT:     %c8_i32 = arith.constant 8 : i32
// TASKFLOW-NEXT:     %c0_i32 = arith.constant 0 : i32
// TASKFLOW-NEXT:     %alloca = memref.alloca() : memref<i32>
// TASKFLOW-NEXT:     %alloca_0 = memref.alloca() : memref<4x8xi32>
// TASKFLOW-NEXT:     %value_outputs = taskflow.task @Task_0 value_inputs(%c0_i32 : i32) : (i32) -> (i32) {
// TASKFLOW-NEXT:     ^bb0(%arg0: i32):
// TASKFLOW-NEXT:       %1 = affine.for %arg1 = 0 to 5 iter_args(%arg2 = %arg0) -> (i32) {
// TASKFLOW-NEXT:         %2 = arith.index_cast %arg1 : index to i32
// TASKFLOW-NEXT:         %3 = arith.addi %arg2, %2 : i32
// TASKFLOW-NEXT:         affine.yield %3 : i32
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield values(%1 : i32)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     %write_outputs = taskflow.task @Task_1 write_memrefs(%alloca_0 : memref<4x8xi32>) value_inputs(%c8_i32 : i32) [original_write_memrefs(%alloca_0)] : (memref<4x8xi32>, i32) -> (memref<4x8xi32>) {
// TASKFLOW-NEXT:     ^bb0(%arg0: memref<4x8xi32>, %arg1: i32):
// TASKFLOW-NEXT:       affine.for %arg2 = 0 to 4 {
// TASKFLOW-NEXT:         %1 = arith.index_cast %arg2 : index to i32
// TASKFLOW-NEXT:         %2 = arith.muli %1, %arg1 : i32
// TASKFLOW-NEXT:         affine.for %arg3 = 0 to 8 {
// TASKFLOW-NEXT:           %3 = arith.index_cast %arg3 : index to i32
// TASKFLOW-NEXT:           %4 = arith.addi %2, %3 : i32
// TASKFLOW-NEXT:           affine.store %4, %arg0[%arg2, %arg3] : memref<4x8xi32>
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield writes(%arg0 : memref<4x8xi32>)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     %write_outputs_1 = taskflow.task @Task_2 read_memrefs(%write_outputs : memref<4x8xi32>) write_memrefs(%alloca : memref<i32>) value_inputs(%c8_i32, %value_outputs, %c2_i32 : i32, i32, i32) [original_read_memrefs(%alloca_0), original_write_memrefs(%alloca)] : (memref<4x8xi32>, memref<i32>, i32, i32, i32) -> (memref<i32>) {
// TASKFLOW-NEXT:     ^bb0(%arg0: memref<4x8xi32>, %arg1: memref<i32>, %arg2: i32, %arg3: i32, %arg4: i32):
// TASKFLOW-NEXT:       affine.for %arg5 = 0 to 4 {
// TASKFLOW-NEXT:         %1 = arith.index_cast %arg5 : index to i32
// TASKFLOW-NEXT:         %2 = arith.muli %1, %arg2 : i32
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
// TASKFLOW-NEXT:       taskflow.yield writes(%arg1 : memref<i32>)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     %0 = affine.load %write_outputs_1[] : memref<i32>
// TASKFLOW-NEXT:     return %0 : i32
// TASKFLOW-NEXT:   }
// TASKFLOW-NEXT: }

// HYPERBLOCK:     module {
// HYPERBLOCK-NEXT:  func.func @_Z21irregularLoopExample1v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// HYPERBLOCK-NEXT:    %c2_i32 = arith.constant 2 : i32
// HYPERBLOCK-NEXT:    %c8_i32 = arith.constant 8 : i32
// HYPERBLOCK-NEXT:    %c0_i32 = arith.constant 0 : i32
// HYPERBLOCK-NEXT:    %alloca = memref.alloca() : memref<i32>
// HYPERBLOCK-NEXT:    %alloca_0 = memref.alloca() : memref<4x8xi32>
// HYPERBLOCK-NEXT:    %value_outputs = taskflow.task @Task_0 value_inputs(%c0_i32 : i32) : (i32) -> (i32) {
// HYPERBLOCK-NEXT:    ^bb0(%arg0: i32):
// HYPERBLOCK-NEXT:      %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 5 : index} : index
// HYPERBLOCK-NEXT:      %2 = "taskflow.hyperblock"(%1, %arg0) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// HYPERBLOCK-NEXT:      ^bb0(%arg1: index, %arg2: i32):
// HYPERBLOCK-NEXT:        %3 = arith.index_cast %arg1 : index to i32
// HYPERBLOCK-NEXT:        %4 = arith.addi %arg2, %3 : i32
// HYPERBLOCK-NEXT:        taskflow.hyperblock.yield iter_args_next(%4 : i32) results(%4 : i32)
// HYPERBLOCK-NEXT:      }) : (index, i32) -> i32
// HYPERBLOCK-NEXT:      taskflow.yield values(%2 : i32)
// HYPERBLOCK-NEXT:    }
// HYPERBLOCK-NEXT:    %write_outputs = taskflow.task @Task_1 write_memrefs(%alloca_0 : memref<4x8xi32>) value_inputs(%c8_i32 : i32) [original_write_memrefs(%alloca_0)] : (memref<4x8xi32>, i32) -> (memref<4x8xi32>) {
// HYPERBLOCK-NEXT:    ^bb0(%arg0: memref<4x8xi32>, %arg1: i32):
// HYPERBLOCK-NEXT:      %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 4 : index} : index
// HYPERBLOCK-NEXT:      %2 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// HYPERBLOCK-NEXT:      "taskflow.hyperblock"(%1) <{operandSegmentSizes = array<i32: 1, 0>}> ({
// HYPERBLOCK-NEXT:      ^bb0(%arg2: index):
// HYPERBLOCK-NEXT:        %3 = arith.index_cast %arg2 : index to i32
// HYPERBLOCK-NEXT:        %4 = arith.muli %3, %arg1 : i32
// HYPERBLOCK-NEXT:        %c0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:        %c8 = arith.constant 8 : index
// HYPERBLOCK-NEXT:        %c1 = arith.constant 1 : index
// HYPERBLOCK-NEXT:        scf.for %arg3 = %c0 to %c8 step %c1 {
// HYPERBLOCK-NEXT:          %5 = arith.index_cast %arg3 : index to i32
// HYPERBLOCK-NEXT:          %6 = arith.addi %4, %5 : i32
// HYPERBLOCK-NEXT:          memref.store %6, %arg0[%arg2, %arg3] : memref<4x8xi32>
// HYPERBLOCK-NEXT:        }
// HYPERBLOCK-NEXT:        taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:      }) : (index) -> ()
// HYPERBLOCK-NEXT:      taskflow.yield writes(%arg0 : memref<4x8xi32>)
// HYPERBLOCK-NEXT:    }
// HYPERBLOCK-NEXT:    %write_outputs_1 = taskflow.task @Task_2 read_memrefs(%write_outputs : memref<4x8xi32>) write_memrefs(%alloca : memref<i32>) value_inputs(%c8_i32, %value_outputs, %c2_i32 : i32, i32, i32) [original_read_memrefs(%alloca_0), original_write_memrefs(%alloca)] : (memref<4x8xi32>, memref<i32>, i32, i32, i32) -> (memref<i32>) {
// HYPERBLOCK-NEXT:    ^bb0(%arg0: memref<4x8xi32>, %arg1: memref<i32>, %arg2: i32, %arg3: i32, %arg4: i32):
// HYPERBLOCK-NEXT:      %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 4 : index} : index
// HYPERBLOCK-NEXT:      %2 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// HYPERBLOCK-NEXT:      "taskflow.hyperblock"(%1) <{operandSegmentSizes = array<i32: 1, 0>}> ({
// HYPERBLOCK-NEXT:      ^bb0(%arg5: index):
// HYPERBLOCK-NEXT:        %3 = arith.index_cast %arg5 : index to i32
// HYPERBLOCK-NEXT:        %4 = arith.muli %3, %arg2 : i32
// HYPERBLOCK-NEXT:        %c0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:        %c8 = arith.constant 8 : index
// HYPERBLOCK-NEXT:        %c1 = arith.constant 1 : index
// HYPERBLOCK-NEXT:        scf.for %arg6 = %c0 to %c8 step %c1 {
// HYPERBLOCK-NEXT:          %5 = memref.load %arg0[%arg5, %arg6] : memref<4x8xi32>
// HYPERBLOCK-NEXT:          %6 = arith.addi %5, %arg3 : i32
// HYPERBLOCK-NEXT:          %c0_2 = arith.constant 0 : index
// HYPERBLOCK-NEXT:          %c-3 = arith.constant -3 : index
// HYPERBLOCK-NEXT:          %7 = arith.addi %arg5, %c-3 : index
// HYPERBLOCK-NEXT:          %8 = arith.cmpi eq, %7, %c0_2 : index
// HYPERBLOCK-NEXT:          %c-7 = arith.constant -7 : index
// HYPERBLOCK-NEXT:          %9 = arith.addi %arg6, %c-7 : index
// HYPERBLOCK-NEXT:          %10 = arith.cmpi eq, %9, %c0_2 : index
// HYPERBLOCK-NEXT:          %11 = arith.andi %8, %10 : i1
// HYPERBLOCK-NEXT:          scf.if %11 {
// HYPERBLOCK-NEXT:            memref.store %6, %arg1[] : memref<i32>
// HYPERBLOCK-NEXT:            %12 = arith.muli %6, %arg4 : i32
// HYPERBLOCK-NEXT:            memref.store %12, %arg1[] : memref<i32>
// HYPERBLOCK-NEXT:          }
// HYPERBLOCK-NEXT:        }
// HYPERBLOCK-NEXT:        taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:      }) : (index) -> ()
// HYPERBLOCK-NEXT:      taskflow.yield writes(%arg1 : memref<i32>)
// HYPERBLOCK-NEXT:    }
// HYPERBLOCK-NEXT:    %0 = affine.load %write_outputs_1[] : memref<i32>
// HYPERBLOCK-NEXT:    return %0 : i32
// HYPERBLOCK-NEXT:  }
// HYPERBLOCK-NEXT:}