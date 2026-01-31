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
// RUN: --place-act-on-cgra \
// RUN: -o %t.placement.mlir
// RUN: FileCheck %s --input-file=%t.placement.mlir --check-prefixes=PLACEMENT

module attributes {} {
  func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg10 = 0 to 4 {
      affine.for %arg11 = 0 to 8 {
        affine.for %arg12 = 0 to 6 {
          %1 = affine.load %arg0[%arg10, %arg11, %arg12] : memref<?x8x6xi32>
          affine.store %1, %arg5[%arg12] : memref<?xi32>
        }
        affine.for %arg12 = 0 to 5 {
          %1 = affine.load %arg1[%arg10, %arg11, %arg12] : memref<?x8x5xi32>
          %2 = affine.load %arg2[%arg10, %arg11, %arg12] : memref<?x8x5xi32>
          %3 = arith.addi %1, %2 : i32
          affine.store %3, %arg6[%arg12] : memref<?xi32>
        }
        affine.for %arg12 = 0 to 6 {
          %1 = affine.load %arg5[%arg12] : memref<?xi32>
          %2 = affine.load %arg6[%arg12] : memref<?xi32>
          %3 = arith.addi %1, %2 : i32
          %4 = affine.load %arg9[0] : memref<?xi32>
          %5 = arith.addi %4, %3 : i32
          affine.store %5, %arg9[0] : memref<?xi32>
        }
      }
      affine.for %arg11 = 0 to 7 {
        %1 = affine.load %arg3[%arg10, %arg11] : memref<?x7xi32>
        affine.store %1, %arg7[%arg11] : memref<?xi32>
      }
      affine.for %arg11 = 0 to 9 {
        %1 = affine.load %arg4[%arg10, %arg11] : memref<?x9xi32>
        %2 = affine.load %arg7[%arg11] : memref<?xi32>
        %3 = arith.addi %1, %2 : i32
        affine.store %3, %arg8[%arg11] : memref<?xi32>
      }
    }
    %0 = affine.load %arg9[0] : memref<?xi32>
    return %0 : i32
  }
}

// TASKFLOW:      module {
// TASKFLOW-NEXT:   func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// TASKFLOW-NEXT:     %memory_outputs:5 = "taskflow.task"(%arg0, %arg1, %arg2, %arg5, %arg6, %arg9, %arg3, %arg4, %arg7, %arg8) <{operandSegmentSizes = array<i32: 10, 0>, resultSegmentSizes = array<i32: 5, 0>, task_name = "Task_0"}> ({
// TASKFLOW-NEXT:     ^bb0(%arg10: memref<?x8x6xi32>, %arg11: memref<?x8x5xi32>, %arg12: memref<?x8x5xi32>, %arg13: memref<?xi32>, %arg14: memref<?xi32>, %arg15: memref<?xi32>, %arg16: memref<?x7xi32>, %arg17: memref<?x9xi32>, %arg18: memref<?xi32>, %arg19: memref<?xi32>):
// TASKFLOW-NEXT:       affine.for %arg20 = 0 to 4 {
// TASKFLOW-NEXT:         affine.for %arg21 = 0 to 8 {
// TASKFLOW-NEXT:           affine.for %arg22 = 0 to 6 {
// TASKFLOW-NEXT:             %1 = affine.load %arg10[%arg20, %arg21, %arg22] : memref<?x8x6xi32>
// TASKFLOW-NEXT:             affine.store %1, %arg13[%arg22] : memref<?xi32>
// TASKFLOW-NEXT:           }
// TASKFLOW-NEXT:           affine.for %arg22 = 0 to 5 {
// TASKFLOW-NEXT:             %1 = affine.load %arg11[%arg20, %arg21, %arg22] : memref<?x8x5xi32>
// TASKFLOW-NEXT:             %2 = affine.load %arg12[%arg20, %arg21, %arg22] : memref<?x8x5xi32>
// TASKFLOW-NEXT:             %3 = arith.addi %1, %2 : i32
// TASKFLOW-NEXT:             affine.store %3, %arg14[%arg22] : memref<?xi32>
// TASKFLOW-NEXT:           }
// TASKFLOW-NEXT:           affine.for %arg22 = 0 to 6 {
// TASKFLOW-NEXT:             %1 = affine.load %arg13[%arg22] : memref<?xi32>
// TASKFLOW-NEXT:             %2 = affine.load %arg14[%arg22] : memref<?xi32>
// TASKFLOW-NEXT:             %3 = arith.addi %1, %2 : i32
// TASKFLOW-NEXT:             %4 = affine.load %arg15[0] : memref<?xi32>
// TASKFLOW-NEXT:             %5 = arith.addi %4, %3 : i32
// TASKFLOW-NEXT:             affine.store %5, %arg15[0] : memref<?xi32>
// TASKFLOW-NEXT:           }
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:         affine.for %arg21 = 0 to 7 {
// TASKFLOW-NEXT:           %1 = affine.load %arg16[%arg20, %arg21] : memref<?x7xi32>
// TASKFLOW-NEXT:           affine.store %1, %arg18[%arg21] : memref<?xi32>
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:         affine.for %arg21 = 0 to 9 {
// TASKFLOW-NEXT:           %1 = affine.load %arg17[%arg20, %arg21] : memref<?x9xi32>
// TASKFLOW-NEXT:           %2 = affine.load %arg18[%arg21] : memref<?xi32>
// TASKFLOW-NEXT:           %3 = arith.addi %1, %2 : i32
// TASKFLOW-NEXT:           affine.store %3, %arg19[%arg21] : memref<?xi32>
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       "taskflow.yield"(%arg13, %arg14, %arg15, %arg18, %arg19) <{operandSegmentSizes = array<i32: 5, 0>}> : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> ()
// TASKFLOW-NEXT:     }) : (memref<?x8x6xi32>, memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?x7xi32>, memref<?x9xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>)
// TASKFLOW-NEXT:     %0 = affine.load %memory_outputs#2[0] : memref<?xi32>
// TASKFLOW-NEXT:     return %0 : i32
// TASKFLOW-NEXT:   }
// TASKFLOW-NEXT: }

// HYPERBLOCK:      module {
// HYPERBLOCK-NEXT:   func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// HYPERBLOCK-NEXT:     %memory_outputs:5 = "taskflow.task"(%arg0, %arg1, %arg2, %arg5, %arg6, %arg9, %arg3, %arg4, %arg7, %arg8) <{operandSegmentSizes = array<i32: 10, 0>, resultSegmentSizes = array<i32: 5, 0>, task_name = "Task_0"}> ({
// HYPERBLOCK-NEXT:     ^bb0(%arg10: memref<?x8x6xi32>, %arg11: memref<?x8x5xi32>, %arg12: memref<?x8x5xi32>, %arg13: memref<?xi32>, %arg14: memref<?xi32>, %arg15: memref<?xi32>, %arg16: memref<?x7xi32>, %arg17: memref<?x9xi32>, %arg18: memref<?xi32>, %arg19: memref<?xi32>):
// HYPERBLOCK-NEXT:       %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 4 : index} : index
// HYPERBLOCK-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// HYPERBLOCK-NEXT:       %3 = taskflow.counter parent(%2 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 6 : index} : index
// HYPERBLOCK-NEXT:       %4 = taskflow.counter parent(%2 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 5 : index} : index
// HYPERBLOCK-NEXT:       %5 = taskflow.counter parent(%2 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 6 : index} : index
// HYPERBLOCK-NEXT:       %6 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 7 : index} : index
// HYPERBLOCK-NEXT:       %7 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 9 : index} : index
// HYPERBLOCK-NEXT:       "taskflow.hyperblock"(%1, %2, %3) <{operandSegmentSizes = array<i32: 3, 0>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg20: index, %arg21: index, %arg22: index):
// HYPERBLOCK-NEXT:         %8 = memref.load %arg10[%arg20, %arg21, %arg22] : memref<?x8x6xi32>
// HYPERBLOCK-NEXT:         memref.store %8, %arg13[%arg22] : memref<?xi32>
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:       }) : (index, index, index) -> ()
// HYPERBLOCK-NEXT:       "taskflow.hyperblock"(%1, %2, %4) <{operandSegmentSizes = array<i32: 3, 0>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg20: index, %arg21: index, %arg22: index):
// HYPERBLOCK-NEXT:         %8 = memref.load %arg11[%arg20, %arg21, %arg22] : memref<?x8x5xi32>
// HYPERBLOCK-NEXT:         %9 = memref.load %arg12[%arg20, %arg21, %arg22] : memref<?x8x5xi32>
// HYPERBLOCK-NEXT:         %10 = arith.addi %8, %9 : i32
// HYPERBLOCK-NEXT:         memref.store %10, %arg14[%arg22] : memref<?xi32>
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:       }) : (index, index, index) -> ()
// HYPERBLOCK-NEXT:       "taskflow.hyperblock"(%5) <{operandSegmentSizes = array<i32: 1, 0>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg20: index):
// HYPERBLOCK-NEXT:         %8 = memref.load %arg13[%arg20] : memref<?xi32>
// HYPERBLOCK-NEXT:         %9 = memref.load %arg14[%arg20] : memref<?xi32>
// HYPERBLOCK-NEXT:         %10 = arith.addi %8, %9 : i32
// HYPERBLOCK-NEXT:         %c0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:         %11 = memref.load %arg15[%c0] : memref<?xi32>
// HYPERBLOCK-NEXT:         %12 = arith.addi %11, %10 : i32
// HYPERBLOCK-NEXT:         %c0_0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:         memref.store %12, %arg15[%c0_0] : memref<?xi32>
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:       }) : (index) -> ()
// HYPERBLOCK-NEXT:       "taskflow.hyperblock"(%1, %6) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg20: index, %arg21: index):
// HYPERBLOCK-NEXT:         %8 = memref.load %arg16[%arg20, %arg21] : memref<?x7xi32>
// HYPERBLOCK-NEXT:         memref.store %8, %arg18[%arg21] : memref<?xi32>
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:       }) : (index, index) -> ()
// HYPERBLOCK-NEXT:       "taskflow.hyperblock"(%1, %7) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// HYPERBLOCK-NEXT:       ^bb0(%arg20: index, %arg21: index):
// HYPERBLOCK-NEXT:         %8 = memref.load %arg17[%arg20, %arg21] : memref<?x9xi32>
// HYPERBLOCK-NEXT:         %9 = memref.load %arg18[%arg21] : memref<?xi32>
// HYPERBLOCK-NEXT:         %10 = arith.addi %8, %9 : i32
// HYPERBLOCK-NEXT:         memref.store %10, %arg19[%arg21] : memref<?xi32>
// HYPERBLOCK-NEXT:         taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:       }) : (index, index) -> ()
// HYPERBLOCK-NEXT:       "taskflow.yield"(%arg13, %arg14, %arg15, %arg18, %arg19) <{operandSegmentSizes = array<i32: 5, 0>}> : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> ()
// HYPERBLOCK-NEXT:     }) : (memref<?x8x6xi32>, memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?x7xi32>, memref<?x9xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>)
// HYPERBLOCK-NEXT:     %0 = affine.load %memory_outputs#2[0] : memref<?xi32>
// HYPERBLOCK-NEXT:     return %0 : i32
// HYPERBLOCK-NEXT:   }
// HYPERBLOCK-NEXT: }

// CANONICALIZE:      module {
// CANONICALIZE-NEXT:   func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CANONICALIZE-NEXT:     %memory_outputs = "taskflow.task"(%arg0, %arg5, %arg0, %arg5) <{operandSegmentSizes = array<i32: 2, 2>, resultSegmentSizes = array<i32: 1, 0>, task_name = "Task_0"}> ({
// CANONICALIZE-NEXT:     ^bb0(%arg10: memref<?x8x6xi32>, %arg11: memref<?xi32>, %arg12: memref<?x8x6xi32>, %arg13: memref<?xi32>):
// CANONICALIZE-NEXT:       %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 4 : index} : index
// CANONICALIZE-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// CANONICALIZE-NEXT:       %3 = taskflow.counter parent(%2 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 6 : index} : index
// CANONICALIZE-NEXT:       "taskflow.hyperblock"(%1, %2, %3) <{operandSegmentSizes = array<i32: 3, 0>}> ({
// CANONICALIZE-NEXT:       ^bb0(%arg14: index, %arg15: index, %arg16: index):
// CANONICALIZE-NEXT:         %4 = memref.load %arg12[%arg14, %arg15, %arg16] : memref<?x8x6xi32>
// CANONICALIZE-NEXT:         memref.store %4, %arg13[%arg16] : memref<?xi32>
// CANONICALIZE-NEXT:         taskflow.hyperblock.yield
// CANONICALIZE-NEXT:       }) : (index, index, index) -> ()
// CANONICALIZE-NEXT:       "taskflow.yield"(%arg13) <{operandSegmentSizes = array<i32: 1, 0>}> : (memref<?xi32>) -> ()
// CANONICALIZE-NEXT:     }) : (memref<?x8x6xi32>, memref<?xi32>, memref<?x8x6xi32>, memref<?xi32>) -> memref<?xi32>
// CANONICALIZE-NEXT:     %memory_outputs_0 = "taskflow.task"(%arg1, %arg2, %arg6, %arg1, %arg2, %arg6) <{operandSegmentSizes = array<i32: 3, 3>, resultSegmentSizes = array<i32: 1, 0>, task_name = "Task_1"}> ({
// CANONICALIZE-NEXT:     ^bb0(%arg10: memref<?x8x5xi32>, %arg11: memref<?x8x5xi32>, %arg12: memref<?xi32>, %arg13: memref<?x8x5xi32>, %arg14: memref<?x8x5xi32>, %arg15: memref<?xi32>):
// CANONICALIZE-NEXT:       %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 4 : index} : index
// CANONICALIZE-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// CANONICALIZE-NEXT:       %3 = taskflow.counter parent(%2 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 5 : index} : index
// CANONICALIZE-NEXT:       "taskflow.hyperblock"(%1, %2, %3) <{operandSegmentSizes = array<i32: 3, 0>}> ({
// CANONICALIZE-NEXT:       ^bb0(%arg16: index, %arg17: index, %arg18: index):
// CANONICALIZE-NEXT:         %4 = memref.load %arg13[%arg16, %arg17, %arg18] : memref<?x8x5xi32>
// CANONICALIZE-NEXT:         %5 = memref.load %arg14[%arg16, %arg17, %arg18] : memref<?x8x5xi32>
// CANONICALIZE-NEXT:         %6 = arith.addi %4, %5 : i32
// CANONICALIZE-NEXT:         memref.store %6, %arg15[%arg18] : memref<?xi32>
// CANONICALIZE-NEXT:         taskflow.hyperblock.yield
// CANONICALIZE-NEXT:       }) : (index, index, index) -> ()
// CANONICALIZE-NEXT:       "taskflow.yield"(%arg15) <{operandSegmentSizes = array<i32: 1, 0>}> : (memref<?xi32>) -> ()
// CANONICALIZE-NEXT:     }) : (memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>, memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>) -> memref<?xi32>
// CANONICALIZE-NEXT:     %memory_outputs_1 = "taskflow.task"(%memory_outputs, %memory_outputs_0, %arg9, %arg5, %arg6, %arg9) <{operandSegmentSizes = array<i32: 3, 3>, resultSegmentSizes = array<i32: 1, 0>, task_name = "Task_2"}> ({
// CANONICALIZE-NEXT:     ^bb0(%arg10: memref<?xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>, %arg13: memref<?xi32>, %arg14: memref<?xi32>, %arg15: memref<?xi32>):
// CANONICALIZE-NEXT:       %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 4 : index} : index
// CANONICALIZE-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// CANONICALIZE-NEXT:       %3 = taskflow.counter parent(%2 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 6 : index} : index
// CANONICALIZE-NEXT:       "taskflow.hyperblock"(%3) <{operandSegmentSizes = array<i32: 1, 0>}> ({
// CANONICALIZE-NEXT:       ^bb0(%arg16: index):
// CANONICALIZE-NEXT:         %4 = memref.load %arg13[%arg16] : memref<?xi32>
// CANONICALIZE-NEXT:         %5 = memref.load %arg14[%arg16] : memref<?xi32>
// CANONICALIZE-NEXT:         %6 = arith.addi %4, %5 : i32
// CANONICALIZE-NEXT:         %c0 = arith.constant 0 : index
// CANONICALIZE-NEXT:         %7 = memref.load %arg15[%c0] : memref<?xi32>
// CANONICALIZE-NEXT:         %8 = arith.addi %7, %6 : i32
// CANONICALIZE-NEXT:         %c0_4 = arith.constant 0 : index
// CANONICALIZE-NEXT:         memref.store %8, %arg15[%c0_4] : memref<?xi32>
// CANONICALIZE-NEXT:         taskflow.hyperblock.yield
// CANONICALIZE-NEXT:       }) : (index) -> ()
// CANONICALIZE-NEXT:       "taskflow.yield"(%arg15) <{operandSegmentSizes = array<i32: 1, 0>}> : (memref<?xi32>) -> ()
// CANONICALIZE-NEXT:     }) : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> memref<?xi32>
// CANONICALIZE-NEXT:     %memory_outputs_2 = "taskflow.task"(%arg3, %arg7, %arg3, %arg7) <{operandSegmentSizes = array<i32: 2, 2>, resultSegmentSizes = array<i32: 1, 0>, task_name = "Task_3"}> ({
// CANONICALIZE-NEXT:     ^bb0(%arg10: memref<?x7xi32>, %arg11: memref<?xi32>, %arg12: memref<?x7xi32>, %arg13: memref<?xi32>):
// CANONICALIZE-NEXT:       %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 4 : index} : index
// CANONICALIZE-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 7 : index} : index
// CANONICALIZE-NEXT:       "taskflow.hyperblock"(%1, %2) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// CANONICALIZE-NEXT:       ^bb0(%arg14: index, %arg15: index):
// CANONICALIZE-NEXT:         %3 = memref.load %arg12[%arg14, %arg15] : memref<?x7xi32>
// CANONICALIZE-NEXT:         memref.store %3, %arg13[%arg15] : memref<?xi32>
// CANONICALIZE-NEXT:         taskflow.hyperblock.yield
// CANONICALIZE-NEXT:       }) : (index, index) -> ()
// CANONICALIZE-NEXT:       "taskflow.yield"(%arg13) <{operandSegmentSizes = array<i32: 1, 0>}> : (memref<?xi32>) -> ()
// CANONICALIZE-NEXT:     }) : (memref<?x7xi32>, memref<?xi32>, memref<?x7xi32>, memref<?xi32>) -> memref<?xi32>
// CANONICALIZE-NEXT:     %memory_outputs_3 = "taskflow.task"(%arg4, %memory_outputs_2, %arg8, %arg4, %arg7, %arg8) <{operandSegmentSizes = array<i32: 3, 3>, resultSegmentSizes = array<i32: 1, 0>, task_name = "Task_4"}> ({
// CANONICALIZE-NEXT:     ^bb0(%arg10: memref<?x9xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>, %arg13: memref<?x9xi32>, %arg14: memref<?xi32>, %arg15: memref<?xi32>):
// CANONICALIZE-NEXT:       %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 4 : index} : index
// CANONICALIZE-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 9 : index} : index
// CANONICALIZE-NEXT:       "taskflow.hyperblock"(%1, %2) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// CANONICALIZE-NEXT:       ^bb0(%arg16: index, %arg17: index):
// CANONICALIZE-NEXT:         %3 = memref.load %arg13[%arg16, %arg17] : memref<?x9xi32>
// CANONICALIZE-NEXT:         %4 = memref.load %arg14[%arg17] : memref<?xi32>
// CANONICALIZE-NEXT:         %5 = arith.addi %3, %4 : i32
// CANONICALIZE-NEXT:         memref.store %5, %arg15[%arg17] : memref<?xi32>
// CANONICALIZE-NEXT:         taskflow.hyperblock.yield
// CANONICALIZE-NEXT:       }) : (index, index) -> ()
// CANONICALIZE-NEXT:       "taskflow.yield"(%arg15) <{operandSegmentSizes = array<i32: 1, 0>}> : (memref<?xi32>) -> ()
// CANONICALIZE-NEXT:     }) : (memref<?x9xi32>, memref<?xi32>, memref<?xi32>, memref<?x9xi32>, memref<?xi32>, memref<?xi32>) -> memref<?xi32>
// CANONICALIZE-NEXT:     %0 = affine.load %memory_outputs_1[0] : memref<?xi32>
// CANONICALIZE-NEXT:     return %0 : i32
// CANONICALIZE-NEXT:   }
// CANONICALIZE-NEXT: }

// PLACEMENT: task_name = "Task_0"
// PLACEMENT: cgra_col = 0 : i32, cgra_count = 1 : i32, cgra_row = 0 : i32
// PLACEMENT: task_name = "Task_1"
// PLACEMENT: cgra_col = 1 : i32, cgra_count = 1 : i32, cgra_row = 1 : i32
// PLACEMENT: task_name = "Task_2"
// PLACEMENT: cgra_col = 1 : i32, cgra_count = 1 : i32, cgra_row = 0 : i32
// PLACEMENT: task_name = "Task_3"
// PLACEMENT: cgra_col = 2 : i32, cgra_count = 1 : i32, cgra_row = 2 : i32
// PLACEMENT: task_name = "Task_4"
// PLACEMENT: cgra_col = 3 : i32, cgra_count = 1 : i32, cgra_row = 2 : i32