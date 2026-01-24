// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: -o %t.taskflow.mlir
// RUN: FileCheck %s --input-file=%t.taskflow.mlir --check-prefixes=TASKFLOW

// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --canonicalize-task \
// RUN: -o %t.canonicalized.mlir
// RUN: FileCheck %s --input-file=%t.canonicalized.mlir --check-prefixes=CANONICALIZE

// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --canonicalize-task \
// RUN: --classify-counters \
// RUN: --convert-taskflow-to-neura \
// RUN: -o %t.kernel.mlir
// RUN: FileCheck %s --input-file=%t.kernel.mlir --check-prefixes=KERNEL

// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --canonicalize-task \
// RUN: --classify-counters \
// RUN: --convert-taskflow-to-neura \
// RUN: --lower-affine \
// RUN: --convert-scf-to-cf \
// RUN: --convert-cf-to-llvm \
// RUN: --assign-accelerator \
// RUN: --lower-memref-to-neura \
// RUN: --lower-arith-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: -o %t.neura.mlir
// RUN: FileCheck %s --input-file=%t.neura.mlir --check-prefixes=NEURA

// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --canonicalize-task \
// RUN: --classify-counters \
// RUN: --convert-taskflow-to-neura \
// RUN: --lower-affine \
// RUN: --convert-scf-to-cf \
// RUN: --convert-cf-to-llvm \
// RUN: --assign-accelerator \
// RUN: --lower-memref-to-neura \
// RUN: --lower-arith-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --promote-input-arg-to-const \
// RUN: --fold-constant \
// RUN: --canonicalize-return \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: --fold-constant \
// RUN: -o %t.dataflow.mlir
// RUN: FileCheck %s --input-file=%t.dataflow.mlir --check-prefixes=DATAFLOW

// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --canonicalize-task \
// RUN: --classify-counters \
// RUN: --convert-taskflow-to-neura \
// RUN: --lower-affine \
// RUN: --convert-scf-to-cf \
// RUN: --convert-cf-to-llvm \
// RUN: --assign-accelerator \
// RUN: --lower-memref-to-neura \
// RUN: --lower-arith-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --promote-input-arg-to-const \
// RUN: --fold-constant \
// RUN: --canonicalize-return \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: --fold-constant \
// RUN: --map-to-accelerator="mapping-strategy=heuristic" \
// RUN: --architecture-spec=%S/../../../arch_spec/architecture.yaml \
// RUN: -o %t.mapped.mlir
// RUN: FileCheck %s --input-file=%t.mapped.mlir --check-prefixes=MAPPED



module attributes {} {
  func.func @_Z6kernelPiS_S_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.for %arg3 = 0 to 32 iter_args(%arg4 = %c0_i32) -> (i32) {
      %1 = affine.load %arg0[%arg3] : memref<?xi32>
      %2 = affine.load %arg2[%arg3] : memref<?xi32>
      %3 = arith.muli %1, %2 : i32
      %4 = arith.addi %arg4, %3 : i32
      affine.yield %4 : i32
    }
    return %0 : i32
  }
}

// TASKFLOW:      module {
// TASKFLOW-NEXT:   func.func @_Z6kernelPiS_S_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// TASKFLOW-NEXT:     %c0_i32 = arith.constant 0 : i32
// TASKFLOW-NEXT:     %value_outputs = "taskflow.task"(%arg0, %arg2, %c0_i32) <{operandSegmentSizes = array<i32: 2, 1>, resultSegmentSizes = array<i32: 0, 1>, task_name = "Task_0"}> ({
// TASKFLOW-NEXT:     ^bb0(%arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// TASKFLOW-NEXT:       %0 = affine.for %arg6 = 0 to 32 iter_args(%arg7 = %arg5) -> (i32) {
// TASKFLOW-NEXT:         %1 = affine.load %arg3[%arg6] : memref<?xi32>
// TASKFLOW-NEXT:         %2 = affine.load %arg4[%arg6] : memref<?xi32>
// TASKFLOW-NEXT:         %3 = arith.muli %1, %2 : i32
// TASKFLOW-NEXT:         %4 = arith.addi %arg7, %3 : i32
// TASKFLOW-NEXT:         affine.yield %4 : i32
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       "taskflow.yield"(%0) <{operandSegmentSizes = array<i32: 0, 1>}> : (i32) -> ()
// TASKFLOW-NEXT:     }) : (memref<?xi32>, memref<?xi32>, i32) -> i32
// TASKFLOW-NEXT:     return %value_outputs : i32
// TASKFLOW-NEXT:   }
// TASKFLOW-NEXT: }

// CANONICALIZE:      module {
// CANONICALIZE-NEXT:   func.func @_Z6kernelPiS_S_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CANONICALIZE-NEXT:     %c0_i32 = arith.constant 0 : i32
// CANONICALIZE-NEXT:     %value_outputs = "taskflow.task"(%arg0, %arg2, %c0_i32) <{operandSegmentSizes = array<i32: 2, 1>, resultSegmentSizes = array<i32: 0, 1>, task_name = "Task_0"}> ({
// CANONICALIZE-NEXT:     ^bb0(%arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// CANONICALIZE-NEXT:       %0 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 32 : index} : index
// CANONICALIZE-NEXT:       %1 = "taskflow.hyperblock"(%0, %arg5) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CANONICALIZE-NEXT:       ^bb0(%arg6: index, %arg7: i32):
// CANONICALIZE-NEXT:         %2 = memref.load %arg3[%arg6] : memref<?xi32>
// CANONICALIZE-NEXT:         %3 = memref.load %arg4[%arg6] : memref<?xi32>
// CANONICALIZE-NEXT:         %4 = arith.muli %2, %3 : i32
// CANONICALIZE-NEXT:         %5 = arith.addi %arg7, %4 : i32
// CANONICALIZE-NEXT:         taskflow.hyperblock.yield iter_args_next(%5 : i32) results(%5 : i32)
// CANONICALIZE-NEXT:       }) : (index, i32) -> i32
// CANONICALIZE-NEXT:       "taskflow.yield"(%1) <{operandSegmentSizes = array<i32: 0, 1>}> : (i32) -> ()
// CANONICALIZE-NEXT:     }) : (memref<?xi32>, memref<?xi32>, i32) -> i32
// CANONICALIZE-NEXT:     return %value_outputs : i32
// CANONICALIZE-NEXT:   }
// CANONICALIZE-NEXT: }

// KERNEL:      module {
// KERNEL-NEXT:   func.func @_Z6kernelPiS_S_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// KERNEL-NEXT:     %c0_i32 = arith.constant 0 : i32
// KERNEL-NEXT:     %value_outputs = "taskflow.task"(%arg0, %arg2, %c0_i32) <{operandSegmentSizes = array<i32: 2, 1>, resultSegmentSizes = array<i32: 0, 1>, task_name = "Task_0"}> ({
// KERNEL-NEXT:     ^bb0(%arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// KERNEL-NEXT:       %0 = taskflow.counter attributes {counter_id = 0 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 32 : index} : index
// KERNEL-NEXT:       %1 = neura.kernel inputs(%arg3, %arg4 : memref<?xi32>, memref<?xi32>) iter_args_init(%arg5 : i32) {
// KERNEL-NEXT:       ^bb0(%arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: i32):
// KERNEL-NEXT:         %2 = neura.counter {counter_id = 0 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 32 : index} : index
// KERNEL-NEXT:         %3 = memref.load %arg6[%2] : memref<?xi32>
// KERNEL-NEXT:         %4 = memref.load %arg7[%2] : memref<?xi32>
// KERNEL-NEXT:         %5 = arith.muli %3, %4 : i32
// KERNEL-NEXT:         %6 = arith.addi %arg8, %5 : i32
// KERNEL-NEXT:         neura.yield iter_args_next(%6 : i32) results(%6 : i32)
// KERNEL-NEXT:       } : i32
// KERNEL-NEXT:       "taskflow.yield"(%1) <{operandSegmentSizes = array<i32: 0, 1>}> : (i32) -> ()
// KERNEL-NEXT:     }) : (memref<?xi32>, memref<?xi32>, i32) -> i32
// KERNEL-NEXT:     return %value_outputs : i32
// KERNEL-NEXT:   }
// KERNEL-NEXT: }

// NEURA:      module {
// NEURA-NEXT:   func.func @_Z6kernelPiS_S_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// NEURA-NEXT:     %c0_i32 = arith.constant 0 : i32
// NEURA-NEXT:     %value_outputs = "taskflow.task"(%arg0, %arg2, %c0_i32) <{operandSegmentSizes = array<i32: 2, 1>, resultSegmentSizes = array<i32: 0, 1>, task_name = "Task_0"}> ({
// NEURA-NEXT:     ^bb0(%arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// NEURA-NEXT:       %0 = taskflow.counter attributes {counter_id = 0 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 32 : index} : index
// NEURA-NEXT:       %1 = neura.kernel inputs(%arg3, %arg4 : memref<?xi32>, memref<?xi32>) iter_args_init(%arg5 : i32) attributes {accelerator = "neura"} {
// NEURA-NEXT:       ^bb0(%arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: i32):
// NEURA-NEXT:         %2 = neura.counter {counter_id = 0 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 32 : index} : index
// NEURA-NEXT:         %3 = neura.load_indexed %arg6[%2 : index] memref<?xi32> : i32
// NEURA-NEXT:         %4 = neura.load_indexed %arg7[%2 : index] memref<?xi32> : i32
// NEURA-NEXT:         %5 = "neura.mul"(%3, %4) : (i32, i32) -> i32
// NEURA-NEXT:         %6 = "neura.add"(%arg8, %5) : (i32, i32) -> i32
// NEURA-NEXT:         neura.yield iter_args_next(%6 : i32) results(%6 : i32)
// NEURA-NEXT:       } : i32
// NEURA-NEXT:       "taskflow.yield"(%1) <{operandSegmentSizes = array<i32: 0, 1>}> : (i32) -> ()
// NEURA-NEXT:     }) : (memref<?xi32>, memref<?xi32>, i32) -> i32
// NEURA-NEXT:     return %value_outputs : i32
// NEURA-NEXT:   }
// NEURA-NEXT: }

// DATAFLOW:      module {
// DATAFLOW-NEXT:   func.func @_Z6kernelPiS_S_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// DATAFLOW-NEXT:     %c0_i32 = arith.constant 0 : i32
// DATAFLOW-NEXT:     %value_outputs = "taskflow.task"(%arg0, %arg2, %c0_i32) <{operandSegmentSizes = array<i32: 2, 1>, resultSegmentSizes = array<i32: 0, 1>, task_name = "Task_0"}> ({
// DATAFLOW-NEXT:     ^bb0(%arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: i32):
// DATAFLOW-NEXT:       %0 = taskflow.counter attributes {counter_id = 0 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 32 : index} : index
// DATAFLOW-NEXT:       %1 = neura.kernel inputs(%arg3, %arg4 : memref<?xi32>, memref<?xi32>) iter_args_init(%arg5 : i32) attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW-NEXT:       ^bb0(%arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: i32):
// DATAFLOW-NEXT:         %2 = neura.counter {counter_id = 0 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 32 : index} : !neura.data<index, i1>
// DATAFLOW-NEXT:         %3 = neura.load_indexed [%2 : !neura.data<index, i1>]  {lhs_value = "%input0"} : !neura.data<i32, i1>
// DATAFLOW-NEXT:         %4 = neura.load_indexed [%2 : !neura.data<index, i1>]  {lhs_value = "%input1"} : !neura.data<i32, i1>
// DATAFLOW-NEXT:         %5 = "neura.mul"(%3, %4) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// DATAFLOW-NEXT:         %6 = "neura.add"(%5) {lhs_value = "%iter_arg_init0"} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// DATAFLOW-NEXT:         %7 = neura.extract_predicate %2 : !neura.data<index, i1> -> !neura.data<i1, i1>
// DATAFLOW-NEXT:         %8 = "neura.not"(%7) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// DATAFLOW-NEXT:         %9 = neura.grant_predicate %6, %8 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// DATAFLOW-NEXT:         neura.return_value %9 : !neura.data<i32, i1>
// DATAFLOW-NEXT:         neura.yield
// DATAFLOW-NEXT:       } : i32
// DATAFLOW-NEXT:       "taskflow.yield"(%1) <{operandSegmentSizes = array<i32: 0, 1>}> : (i32) -> ()
// DATAFLOW-NEXT:     }) : (memref<?xi32>, memref<?xi32>, i32) -> i32
// DATAFLOW-NEXT:     return %value_outputs : i32
// DATAFLOW-NEXT:   }
// DATAFLOW-NEXT: }