// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: -o %t.serialized.mlir
// RUN: FileCheck %s --input-file=%t.serialized.mlir --check-prefixes=SERIALIZED

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: -o %t.taskflow.mlir
// RUN: FileCheck %s --input-file=%t.taskflow.mlir --check-prefixes=TASKFLOW

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: --convert-taskflow-to-neura="mode=innermost" \
// RUN: -o %t.kernel.mlir
// RUN: FileCheck %s --input-file=%t.kernel.mlir --check-prefixes=KERNEL

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: -o %t.hyperblock.mlir
// RUN: FileCheck %s --input-file=%t.hyperblock.mlir --check-prefixes=HYPERBLOCK

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --map-ct-on-cgra-array \
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

// SERIALIZED:      module {
// SERIALIZED-NEXT:   func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// SERIALIZED-NEXT:     affine.for %arg10 = 0 to 4 {
// SERIALIZED-NEXT:       affine.for %arg11 = 0 to 8 {
// SERIALIZED-NEXT:         affine.for %arg12 = 0 to 6 {
// SERIALIZED-NEXT:           %1 = affine.load %arg0[%arg10, %arg11, %arg12] : memref<?x8x6xi32>
// SERIALIZED-NEXT:           affine.store %1, %arg5[%arg12] : memref<?xi32>
// SERIALIZED-NEXT:         }
// SERIALIZED-NEXT:       }
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     affine.for %arg10 = 0 to 4 {
// SERIALIZED-NEXT:       affine.for %arg11 = 0 to 8 {
// SERIALIZED-NEXT:         affine.for %arg12 = 0 to 5 {
// SERIALIZED-NEXT:           %1 = affine.load %arg1[%arg10, %arg11, %arg12] : memref<?x8x5xi32>
// SERIALIZED-NEXT:           %2 = affine.load %arg2[%arg10, %arg11, %arg12] : memref<?x8x5xi32>
// SERIALIZED-NEXT:           %3 = arith.addi %1, %2 : i32
// SERIALIZED-NEXT:           affine.store %3, %arg6[%arg12] : memref<?xi32>
// SERIALIZED-NEXT:         }
// SERIALIZED-NEXT:       }
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     affine.for %arg10 = 0 to 4 {
// SERIALIZED-NEXT:       affine.for %arg11 = 0 to 8 {
// SERIALIZED-NEXT:         affine.for %arg12 = 0 to 6 {
// SERIALIZED-NEXT:           %1 = affine.load %arg5[%arg12] : memref<?xi32>
// SERIALIZED-NEXT:           %2 = affine.load %arg6[%arg12] : memref<?xi32>
// SERIALIZED-NEXT:           %3 = arith.addi %1, %2 : i32
// SERIALIZED-NEXT:           %4 = affine.load %arg9[0] : memref<?xi32>
// SERIALIZED-NEXT:           %5 = arith.addi %4, %3 : i32
// SERIALIZED-NEXT:           affine.store %5, %arg9[0] : memref<?xi32>
// SERIALIZED-NEXT:         }
// SERIALIZED-NEXT:       }
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     affine.for %arg10 = 0 to 4 {
// SERIALIZED-NEXT:       affine.for %arg11 = 0 to 7 {
// SERIALIZED-NEXT:         %1 = affine.load %arg3[%arg10, %arg11] : memref<?x7xi32>
// SERIALIZED-NEXT:         affine.store %1, %arg7[%arg11] : memref<?xi32>
// SERIALIZED-NEXT:       }
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     affine.for %arg10 = 0 to 4 {
// SERIALIZED-NEXT:       affine.for %arg11 = 0 to 9 {
// SERIALIZED-NEXT:         %1 = affine.load %arg4[%arg10, %arg11] : memref<?x9xi32>
// SERIALIZED-NEXT:         %2 = affine.load %arg7[%arg11] : memref<?xi32>
// SERIALIZED-NEXT:         %3 = arith.addi %1, %2 : i32
// SERIALIZED-NEXT:         affine.store %3, %arg8[%arg11] : memref<?xi32>
// SERIALIZED-NEXT:       }
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     %0 = affine.load %arg9[0] : memref<?xi32>
// SERIALIZED-NEXT:     return %0 : i32
// SERIALIZED-NEXT:   }
// SERIALIZED-NEXT: }

// TASKFLOW:      module {
// TASKFLOW-NEXT:   func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// TASKFLOW-NEXT:     %write_outputs = taskflow.task @Task_0 read_memrefs(%arg0 : memref<?x8x6xi32>) write_memrefs(%arg5 : memref<?xi32>) [original_read_memrefs(%arg0 : memref<?x8x6xi32>), original_write_memrefs(%arg5 : memref<?xi32>)] : (memref<?x8x6xi32>, memref<?xi32>) -> (memref<?xi32>) {
// TASKFLOW-NEXT:     ^bb0(%arg10: memref<?x8x6xi32>, %arg11: memref<?xi32>):
// TASKFLOW-NEXT:       affine.for %arg12 = 0 to 4 {
// TASKFLOW-NEXT:         affine.for %arg13 = 0 to 8 {
// TASKFLOW-NEXT:           affine.for %arg14 = 0 to 6 {
// TASKFLOW-NEXT:             %1 = affine.load %arg10[%arg12, %arg13, %arg14] : memref<?x8x6xi32>
// TASKFLOW-NEXT:             affine.store %1, %arg11[%arg14] : memref<?xi32>
// TASKFLOW-NEXT:           }
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield writes(%arg11 : memref<?xi32>)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     %write_outputs_0 = taskflow.task @Task_1 read_memrefs(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>) write_memrefs(%arg6 : memref<?xi32>) [original_read_memrefs(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>), original_write_memrefs(%arg6 : memref<?xi32>)] : (memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>) -> (memref<?xi32>) {
// TASKFLOW-NEXT:     ^bb0(%arg10: memref<?x8x5xi32>, %arg11: memref<?x8x5xi32>, %arg12: memref<?xi32>):
// TASKFLOW-NEXT:       affine.for %arg13 = 0 to 4 {
// TASKFLOW-NEXT:         affine.for %arg14 = 0 to 8 {
// TASKFLOW-NEXT:           affine.for %arg15 = 0 to 5 {
// TASKFLOW-NEXT:             %1 = affine.load %arg10[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// TASKFLOW-NEXT:             %2 = affine.load %arg11[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// TASKFLOW-NEXT:             %3 = arith.addi %1, %2 : i32
// TASKFLOW-NEXT:             affine.store %3, %arg12[%arg15] : memref<?xi32>
// TASKFLOW-NEXT:           }
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield writes(%arg12 : memref<?xi32>)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     %write_outputs_1 = taskflow.task @Task_2 read_memrefs(%write_outputs, %write_outputs_0, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>) write_memrefs(%arg9 : memref<?xi32>) [original_read_memrefs(%arg5, %arg6, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>), original_write_memrefs(%arg9 : memref<?xi32>)] : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// TASKFLOW-NEXT:     ^bb0(%arg10: memref<?xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>, %arg13: memref<?xi32>):
// TASKFLOW-NEXT:       affine.for %arg14 = 0 to 4 {
// TASKFLOW-NEXT:         affine.for %arg15 = 0 to 8 {
// TASKFLOW-NEXT:           affine.for %arg16 = 0 to 6 {
// TASKFLOW-NEXT:             %1 = affine.load %arg10[%arg16] : memref<?xi32>
// TASKFLOW-NEXT:             %2 = affine.load %arg11[%arg16] : memref<?xi32>
// TASKFLOW-NEXT:             %3 = arith.addi %1, %2 : i32
// TASKFLOW-NEXT:             %4 = affine.load %arg13[0] : memref<?xi32>
// TASKFLOW-NEXT:             %5 = arith.addi %4, %3 : i32
// TASKFLOW-NEXT:             affine.store %5, %arg13[0] : memref<?xi32>
// TASKFLOW-NEXT:           }
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield writes(%arg13 : memref<?xi32>)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     %write_outputs_2 = taskflow.task @Task_3 read_memrefs(%arg3 : memref<?x7xi32>) write_memrefs(%arg7 : memref<?xi32>) [original_read_memrefs(%arg3 : memref<?x7xi32>), original_write_memrefs(%arg7 : memref<?xi32>)] : (memref<?x7xi32>, memref<?xi32>) -> (memref<?xi32>) {
// TASKFLOW-NEXT:     ^bb0(%arg10: memref<?x7xi32>, %arg11: memref<?xi32>):
// TASKFLOW-NEXT:       affine.for %arg12 = 0 to 4 {
// TASKFLOW-NEXT:         affine.for %arg13 = 0 to 7 {
// TASKFLOW-NEXT:           %1 = affine.load %arg10[%arg12, %arg13] : memref<?x7xi32>
// TASKFLOW-NEXT:           affine.store %1, %arg11[%arg13] : memref<?xi32>
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield writes(%arg11 : memref<?xi32>)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     %write_outputs_3 = taskflow.task @Task_4 read_memrefs(%arg4, %write_outputs_2 : memref<?x9xi32>, memref<?xi32>) write_memrefs(%arg8 : memref<?xi32>) [original_read_memrefs(%arg4, %arg7 : memref<?x9xi32>, memref<?xi32>), original_write_memrefs(%arg8 : memref<?xi32>)] : (memref<?x9xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// TASKFLOW-NEXT:     ^bb0(%arg10: memref<?x9xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>):
// TASKFLOW-NEXT:       affine.for %arg13 = 0 to 4 {
// TASKFLOW-NEXT:         affine.for %arg14 = 0 to 9 {
// TASKFLOW-NEXT:           %1 = affine.load %arg10[%arg13, %arg14] : memref<?x9xi32>
// TASKFLOW-NEXT:           %2 = affine.load %arg11[%arg14] : memref<?xi32>
// TASKFLOW-NEXT:           %3 = arith.addi %1, %2 : i32
// TASKFLOW-NEXT:           affine.store %3, %arg12[%arg14] : memref<?xi32>
// TASKFLOW-NEXT:         }
// TASKFLOW-NEXT:       }
// TASKFLOW-NEXT:       taskflow.yield writes(%arg12 : memref<?xi32>)
// TASKFLOW-NEXT:     }
// TASKFLOW-NEXT:     %0 = affine.load %write_outputs_1[0] : memref<?xi32>
// TASKFLOW-NEXT:     return %0 : i32
// TASKFLOW-NEXT:   }
// TASKFLOW-NEXT: }

// KERNEL:      module {
// KERNEL-NEXT:   func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// KERNEL-NEXT:     %write_outputs = taskflow.task @Task_0 read_memrefs(%arg0 : memref<?x8x6xi32>) write_memrefs(%arg5 : memref<?xi32>) [original_read_memrefs(%arg0 : memref<?x8x6xi32>), original_write_memrefs(%arg5 : memref<?xi32>)] : (memref<?x8x6xi32>, memref<?xi32>) -> (memref<?xi32>) {
// KERNEL-NEXT:     ^bb0(%arg10: memref<?x8x6xi32>, %arg11: memref<?xi32>):
// KERNEL-NEXT:       affine.for %arg12 = 0 to 4 {
// KERNEL-NEXT:         affine.for %arg13 = 0 to 8 {
// KERNEL-NEXT:           neura.kernel inputs(%arg10, %arg12, %arg13, %arg11 : memref<?x8x6xi32>, index, index, memref<?xi32>) attributes {kernel_name = "kernel_0"} {
// KERNEL-NEXT:           ^bb0(%arg14: memref<?x8x6xi32>, %arg15: index, %arg16: index, %arg17: memref<?xi32>):
// KERNEL-NEXT:             %c0 = arith.constant 0 : index
// KERNEL-NEXT:             %c6 = arith.constant 6 : index
// KERNEL-NEXT:             %c1 = arith.constant 1 : index
// KERNEL-NEXT:             scf.for %arg18 = %c0 to %c6 step %c1 {
// KERNEL-NEXT:               %1 = memref.load %arg14[%arg15, %arg16, %arg18] : memref<?x8x6xi32>
// KERNEL-NEXT:               memref.store %1, %arg17[%arg18] : memref<?xi32>
// KERNEL-NEXT:             }
// KERNEL-NEXT:             neura.yield
// KERNEL-NEXT:           }
// KERNEL-NEXT:         }
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg11 : memref<?xi32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %write_outputs_0 = taskflow.task @Task_1 read_memrefs(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>) write_memrefs(%arg6 : memref<?xi32>) [original_read_memrefs(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>), original_write_memrefs(%arg6 : memref<?xi32>)] : (memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>) -> (memref<?xi32>) {
// KERNEL-NEXT:     ^bb0(%arg10: memref<?x8x5xi32>, %arg11: memref<?x8x5xi32>, %arg12: memref<?xi32>):
// KERNEL-NEXT:       affine.for %arg13 = 0 to 4 {
// KERNEL-NEXT:         affine.for %arg14 = 0 to 8 {
// KERNEL-NEXT:           neura.kernel inputs(%arg10, %arg13, %arg14, %arg11, %arg12 : memref<?x8x5xi32>, index, index, memref<?x8x5xi32>, memref<?xi32>) attributes {kernel_name = "kernel_1"} {
// KERNEL-NEXT:           ^bb0(%arg15: memref<?x8x5xi32>, %arg16: index, %arg17: index, %arg18: memref<?x8x5xi32>, %arg19: memref<?xi32>):
// KERNEL-NEXT:             %c0 = arith.constant 0 : index
// KERNEL-NEXT:             %c5 = arith.constant 5 : index
// KERNEL-NEXT:             %c1 = arith.constant 1 : index
// KERNEL-NEXT:             scf.for %arg20 = %c0 to %c5 step %c1 {
// KERNEL-NEXT:               %1 = memref.load %arg15[%arg16, %arg17, %arg20] : memref<?x8x5xi32>
// KERNEL-NEXT:               %2 = memref.load %arg18[%arg16, %arg17, %arg20] : memref<?x8x5xi32>
// KERNEL-NEXT:               %3 = arith.addi %1, %2 : i32
// KERNEL-NEXT:               memref.store %3, %arg19[%arg20] : memref<?xi32>
// KERNEL-NEXT:             }
// KERNEL-NEXT:             neura.yield
// KERNEL-NEXT:           }
// KERNEL-NEXT:         }
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg12 : memref<?xi32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %write_outputs_1 = taskflow.task @Task_2 read_memrefs(%write_outputs, %write_outputs_0, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>) write_memrefs(%arg9 : memref<?xi32>) [original_read_memrefs(%arg5, %arg6, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>), original_write_memrefs(%arg9 : memref<?xi32>)] : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// KERNEL-NEXT:     ^bb0(%arg10: memref<?xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>, %arg13: memref<?xi32>):
// KERNEL-NEXT:       affine.for %arg14 = 0 to 4 {
// KERNEL-NEXT:         affine.for %arg15 = 0 to 8 {
// KERNEL-NEXT:           neura.kernel inputs(%arg10, %arg11, %arg13 : memref<?xi32>, memref<?xi32>, memref<?xi32>) attributes {kernel_name = "kernel_2"} {
// KERNEL-NEXT:           ^bb0(%arg16: memref<?xi32>, %arg17: memref<?xi32>, %arg18: memref<?xi32>):
// KERNEL-NEXT:             %c0 = arith.constant 0 : index
// KERNEL-NEXT:             %c6 = arith.constant 6 : index
// KERNEL-NEXT:             %c1 = arith.constant 1 : index
// KERNEL-NEXT:             scf.for %arg19 = %c0 to %c6 step %c1 {
// KERNEL-NEXT:               %1 = memref.load %arg16[%arg19] : memref<?xi32>
// KERNEL-NEXT:               %2 = memref.load %arg17[%arg19] : memref<?xi32>
// KERNEL-NEXT:               %3 = arith.addi %1, %2 : i32
// KERNEL-NEXT:               %c0_4 = arith.constant 0 : index
// KERNEL-NEXT:               %4 = memref.load %arg18[%c0_4] : memref<?xi32>
// KERNEL-NEXT:               %5 = arith.addi %4, %3 : i32
// KERNEL-NEXT:               %c0_5 = arith.constant 0 : index
// KERNEL-NEXT:               memref.store %5, %arg18[%c0_5] : memref<?xi32>
// KERNEL-NEXT:             }
// KERNEL-NEXT:             neura.yield
// KERNEL-NEXT:           }
// KERNEL-NEXT:         }
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg13 : memref<?xi32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %write_outputs_2 = taskflow.task @Task_3 read_memrefs(%arg3 : memref<?x7xi32>) write_memrefs(%arg7 : memref<?xi32>) [original_read_memrefs(%arg3 : memref<?x7xi32>), original_write_memrefs(%arg7 : memref<?xi32>)] : (memref<?x7xi32>, memref<?xi32>) -> (memref<?xi32>) {
// KERNEL-NEXT:     ^bb0(%arg10: memref<?x7xi32>, %arg11: memref<?xi32>):
// KERNEL-NEXT:       affine.for %arg12 = 0 to 4 {
// KERNEL-NEXT:         neura.kernel inputs(%arg10, %arg12, %arg11 : memref<?x7xi32>, index, memref<?xi32>) attributes {kernel_name = "kernel_3"} {
// KERNEL-NEXT:         ^bb0(%arg13: memref<?x7xi32>, %arg14: index, %arg15: memref<?xi32>):
// KERNEL-NEXT:           %c0 = arith.constant 0 : index
// KERNEL-NEXT:           %c7 = arith.constant 7 : index
// KERNEL-NEXT:           %c1 = arith.constant 1 : index
// KERNEL-NEXT:           scf.for %arg16 = %c0 to %c7 step %c1 {
// KERNEL-NEXT:             %1 = memref.load %arg13[%arg14, %arg16] : memref<?x7xi32>
// KERNEL-NEXT:             memref.store %1, %arg15[%arg16] : memref<?xi32>
// KERNEL-NEXT:           }
// KERNEL-NEXT:           neura.yield
// KERNEL-NEXT:         }
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg11 : memref<?xi32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %write_outputs_3 = taskflow.task @Task_4 read_memrefs(%arg4, %write_outputs_2 : memref<?x9xi32>, memref<?xi32>) write_memrefs(%arg8 : memref<?xi32>) [original_read_memrefs(%arg4, %arg7 : memref<?x9xi32>, memref<?xi32>), original_write_memrefs(%arg8 : memref<?xi32>)] : (memref<?x9xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// KERNEL-NEXT:     ^bb0(%arg10: memref<?x9xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>):
// KERNEL-NEXT:       affine.for %arg13 = 0 to 4 {
// KERNEL-NEXT:         neura.kernel inputs(%arg10, %arg13, %arg11, %arg12 : memref<?x9xi32>, index, memref<?xi32>, memref<?xi32>) attributes {kernel_name = "kernel_4"} {
// KERNEL-NEXT:         ^bb0(%arg14: memref<?x9xi32>, %arg15: index, %arg16: memref<?xi32>, %arg17: memref<?xi32>):
// KERNEL-NEXT:           %c0 = arith.constant 0 : index
// KERNEL-NEXT:           %c9 = arith.constant 9 : index
// KERNEL-NEXT:           %c1 = arith.constant 1 : index
// KERNEL-NEXT:           scf.for %arg18 = %c0 to %c9 step %c1 {
// KERNEL-NEXT:             %1 = memref.load %arg14[%arg15, %arg18] : memref<?x9xi32>
// KERNEL-NEXT:             %2 = memref.load %arg16[%arg18] : memref<?xi32>
// KERNEL-NEXT:             %3 = arith.addi %1, %2 : i32
// KERNEL-NEXT:             memref.store %3, %arg17[%arg18] : memref<?xi32>
// KERNEL-NEXT:           }
// KERNEL-NEXT:           neura.yield
// KERNEL-NEXT:         }
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg12 : memref<?xi32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %0 = affine.load %write_outputs_1[0] : memref<?xi32>
// KERNEL-NEXT:     return %0 : i32
// KERNEL-NEXT:   }
// KERNEL-NEXT: }

// HYPERBLOCK:     module {
// HYPERBLOCK-NEXT:  func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// HYPERBLOCK-NEXT:    %write_outputs = taskflow.task @Task_0 read_memrefs(%arg0 : memref<?x8x6xi32>) write_memrefs(%arg5 : memref<?xi32>) [original_read_memrefs(%arg0 : memref<?x8x6xi32>), original_write_memrefs(%arg5 : memref<?xi32>)] : (memref<?x8x6xi32>, memref<?xi32>) -> (memref<?xi32>) {
// HYPERBLOCK-NEXT:    ^bb0(%arg10: memref<?x8x6xi32>, %arg11: memref<?xi32>):
// HYPERBLOCK-NEXT:      %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 4 : index} : index
// HYPERBLOCK-NEXT:      %2 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// HYPERBLOCK-NEXT:      %3 = taskflow.counter parent(%2 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 6 : index} : index
// HYPERBLOCK-NEXT:      "taskflow.hyperblock"(%1, %2, %3) <{operandSegmentSizes = array<i32: 3, 0>}> ({
// HYPERBLOCK-NEXT:      ^bb0(%arg12: index, %arg13: index, %arg14: index):
// HYPERBLOCK-NEXT:        %4 = memref.load %arg10[%arg12, %arg13, %arg14] : memref<?x8x6xi32>
// HYPERBLOCK-NEXT:        memref.store %4, %arg11[%arg14] : memref<?xi32>
// HYPERBLOCK-NEXT:        taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:      }) : (index, index, index) -> ()
// HYPERBLOCK-NEXT:      taskflow.yield writes(%arg11 : memref<?xi32>)
// HYPERBLOCK-NEXT:    }
// HYPERBLOCK-NEXT:    %write_outputs_0 = taskflow.task @Task_1 read_memrefs(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>) write_memrefs(%arg6 : memref<?xi32>) [original_read_memrefs(%arg1, %arg2 : memref<?x8x5xi32>, memref<?x8x5xi32>), original_write_memrefs(%arg6 : memref<?xi32>)] : (memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>) -> (memref<?xi32>) {
// HYPERBLOCK-NEXT:    ^bb0(%arg10: memref<?x8x5xi32>, %arg11: memref<?x8x5xi32>, %arg12: memref<?xi32>):
// HYPERBLOCK-NEXT:      %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 4 : index} : index
// HYPERBLOCK-NEXT:      %2 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// HYPERBLOCK-NEXT:      %3 = taskflow.counter parent(%2 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 5 : index} : index
// HYPERBLOCK-NEXT:      "taskflow.hyperblock"(%1, %2, %3) <{operandSegmentSizes = array<i32: 3, 0>}> ({
// HYPERBLOCK-NEXT:      ^bb0(%arg13: index, %arg14: index, %arg15: index):
// HYPERBLOCK-NEXT:        %4 = memref.load %arg10[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// HYPERBLOCK-NEXT:        %5 = memref.load %arg11[%arg13, %arg14, %arg15] : memref<?x8x5xi32>
// HYPERBLOCK-NEXT:        %6 = arith.addi %4, %5 : i32
// HYPERBLOCK-NEXT:        memref.store %6, %arg12[%arg15] : memref<?xi32>
// HYPERBLOCK-NEXT:        taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:      }) : (index, index, index) -> ()
// HYPERBLOCK-NEXT:      taskflow.yield writes(%arg12 : memref<?xi32>)
// HYPERBLOCK-NEXT:    }
// HYPERBLOCK-NEXT:    %write_outputs_1 = taskflow.task @Task_2 read_memrefs(%write_outputs, %write_outputs_0, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>) write_memrefs(%arg9 : memref<?xi32>) [original_read_memrefs(%arg5, %arg6, %arg9 : memref<?xi32>, memref<?xi32>, memref<?xi32>), original_write_memrefs(%arg9 : memref<?xi32>)] : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// HYPERBLOCK-NEXT:    ^bb0(%arg10: memref<?xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>, %arg13: memref<?xi32>):
// HYPERBLOCK-NEXT:      %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 4 : index} : index
// HYPERBLOCK-NEXT:      %2 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// HYPERBLOCK-NEXT:      %3 = taskflow.counter parent(%2 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 6 : index} : index
// HYPERBLOCK-NEXT:      "taskflow.hyperblock"(%3) <{operandSegmentSizes = array<i32: 1, 0>}> ({
// HYPERBLOCK-NEXT:      ^bb0(%arg14: index):
// HYPERBLOCK-NEXT:        %4 = memref.load %arg10[%arg14] : memref<?xi32>
// HYPERBLOCK-NEXT:        %5 = memref.load %arg11[%arg14] : memref<?xi32>
// HYPERBLOCK-NEXT:        %6 = arith.addi %4, %5 : i32
// HYPERBLOCK-NEXT:        %c0 = arith.constant 0 : index
// HYPERBLOCK-NEXT:        %7 = memref.load %arg13[%c0] : memref<?xi32>
// HYPERBLOCK-NEXT:        %8 = arith.addi %7, %6 : i32
// HYPERBLOCK-NEXT:        %c0_4 = arith.constant 0 : index
// HYPERBLOCK-NEXT:        memref.store %8, %arg13[%c0_4] : memref<?xi32>
// HYPERBLOCK-NEXT:        taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:      }) : (index) -> ()
// HYPERBLOCK-NEXT:      taskflow.yield writes(%arg13 : memref<?xi32>)
// HYPERBLOCK-NEXT:    }
// HYPERBLOCK-NEXT:    %write_outputs_2 = taskflow.task @Task_3 read_memrefs(%arg3 : memref<?x7xi32>) write_memrefs(%arg7 : memref<?xi32>) [original_read_memrefs(%arg3 : memref<?x7xi32>), original_write_memrefs(%arg7 : memref<?xi32>)] : (memref<?x7xi32>, memref<?xi32>) -> (memref<?xi32>) {
// HYPERBLOCK-NEXT:    ^bb0(%arg10: memref<?x7xi32>, %arg11: memref<?xi32>):
// HYPERBLOCK-NEXT:      %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 4 : index} : index
// HYPERBLOCK-NEXT:      %2 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 7 : index} : index
// HYPERBLOCK-NEXT:      "taskflow.hyperblock"(%1, %2) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// HYPERBLOCK-NEXT:      ^bb0(%arg12: index, %arg13: index):
// HYPERBLOCK-NEXT:        %3 = memref.load %arg10[%arg12, %arg13] : memref<?x7xi32>
// HYPERBLOCK-NEXT:        memref.store %3, %arg11[%arg13] : memref<?xi32>
// HYPERBLOCK-NEXT:        taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:      }) : (index, index) -> ()
// HYPERBLOCK-NEXT:      taskflow.yield writes(%arg11 : memref<?xi32>)
// HYPERBLOCK-NEXT:    }
// HYPERBLOCK-NEXT:    %write_outputs_3 = taskflow.task @Task_4 read_memrefs(%arg4, %write_outputs_2 : memref<?x9xi32>, memref<?xi32>) write_memrefs(%arg8 : memref<?xi32>) [original_read_memrefs(%arg4, %arg7 : memref<?x9xi32>, memref<?xi32>), original_write_memrefs(%arg8 : memref<?xi32>)] : (memref<?x9xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>) {
// HYPERBLOCK-NEXT:    ^bb0(%arg10: memref<?x9xi32>, %arg11: memref<?xi32>, %arg12: memref<?xi32>):
// HYPERBLOCK-NEXT:      %1 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 4 : index} : index
// HYPERBLOCK-NEXT:      %2 = taskflow.counter parent(%1 : index) attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 9 : index} : index
// HYPERBLOCK-NEXT:      "taskflow.hyperblock"(%1, %2) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// HYPERBLOCK-NEXT:      ^bb0(%arg13: index, %arg14: index):
// HYPERBLOCK-NEXT:        %3 = memref.load %arg10[%arg13, %arg14] : memref<?x9xi32>
// HYPERBLOCK-NEXT:        %4 = memref.load %arg11[%arg14] : memref<?xi32>
// HYPERBLOCK-NEXT:        %5 = arith.addi %3, %4 : i32
// HYPERBLOCK-NEXT:        memref.store %5, %arg12[%arg14] : memref<?xi32>
// HYPERBLOCK-NEXT:        taskflow.hyperblock.yield
// HYPERBLOCK-NEXT:      }) : (index, index) -> ()
// HYPERBLOCK-NEXT:      taskflow.yield writes(%arg12 : memref<?xi32>)
// HYPERBLOCK-NEXT:    }
// HYPERBLOCK-NEXT:    %0 = affine.load %write_outputs_1[0] : memref<?xi32>
// HYPERBLOCK-NEXT:    return %0 : i32
// HYPERBLOCK-NEXT:  }

// HYPERBLOCK-NEXT:}

// PLACEMENT:      taskflow.task @Task_0
// PLACEMENT-SAME: mapping_info = {cgra_positions = [{col = 0 : i32, row = 0 : i32}], read_sram_ids = [0 : i32], write_sram_ids = [65536 : i32]}
// PLACEMENT:      taskflow.task @Task_1
// PLACEMENT-SAME: mapping_info = {cgra_positions = [{col = 1 : i32, row = 0 : i32}], read_sram_ids = [1 : i32, 1 : i32], write_sram_ids = [65537 : i32]}
// PLACEMENT:      taskflow.task @Task_2
// PLACEMENT-SAME: mapping_info = {cgra_positions = [{col = 0 : i32, row = 1 : i32}], read_sram_ids = [65536 : i32, 65537 : i32, 65536 : i32], write_sram_ids = [65536 : i32]}
// PLACEMENT:      taskflow.task @Task_3
// PLACEMENT-SAME: mapping_info = {cgra_positions = [{col = 2 : i32, row = 0 : i32}], read_sram_ids = [2 : i32], write_sram_ids = [65538 : i32]}
// PLACEMENT:      taskflow.task @Task_4
// PLACEMENT-SAME: mapping_info = {cgra_positions = [{col = 1 : i32, row = 1 : i32}], read_sram_ids = [65537 : i32, 65538 : i32], write_sram_ids = [65537 : i32]}