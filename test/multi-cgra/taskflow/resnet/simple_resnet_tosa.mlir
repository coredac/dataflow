// RUN: neura-compiler %s --tosa-to-affine-conversion \
// RUN: -o %t.affine.mlir
// RUN: FileCheck %s --input-file=%t.affine.mlir --check-prefixes=AFFINE

// RUN: neura-compiler %s --tosa-to-affine-conversion \
// RUN: --taskflow-conversion \
// RUN: -o %t.kernel.mlir
// RUN: FileCheck %s --input-file=%t.kernel.mlir --check-prefixes=KERNEL

// RUN: mlir-neura-opt %t.affine.mlir \
// RUN: --affine-loop-tree-serialization \
// RUN: --affine-loop-perfection \
// RUN: --convert-affine-to-taskflow \
// RUN: --memory-access-streaming-fusion \
// RUN: -o %t.stream.mlir
// RUN: FileCheck %s --input-file=%t.stream.mlir --check-prefixes=STREAM

// RUN: mlir-neura-opt %t.stream.mlir \
// RUN: --resource-aware-task-optimization \
// RUN: -o %t.resopt.mlir
// RUN: FileCheck %s --input-file=%t.resopt.mlir --check-prefixes=RESOPT


module attributes {torch.debug_module_name = "SimpleResNetBlock"} {
  func.func @forward(%arg0: tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32> {
    %0 = "tosa.const"() <{value = dense<"0x7BEEA13C"> : tensor<64x64x3x3xf32>}> : () -> tensor<64x64x3x3xf32>
    %1 = "tosa.const"() <{value = dense<"0x8B9878BC"> : tensor<64x64x3x3xf32>}> : () -> tensor<64x64x3x3xf32>
    %2 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %3 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5 = tosa.transpose %arg0, %3 : (tensor<1x64x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x64xf32>
    %6 = tosa.transpose %1, %3 : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %7 = tosa.conv2d %5, %6, %2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x8x8x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x8x8x64xf32>
    %8 = tosa.transpose %7, %4 : (tensor<1x8x8x64xf32>, tensor<4xi32>) -> tensor<1x64x8x8xf32>
    %9 = tosa.clamp %8 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
    %10 = tosa.transpose %9, %3 : (tensor<1x64x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x64xf32>
    %11 = tosa.transpose %0, %3 : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %12 = tosa.conv2d %10, %11, %2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x8x8x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x8x8x64xf32>
    %13 = tosa.transpose %12, %4 : (tensor<1x8x8x64xf32>, tensor<4xi32>) -> tensor<1x64x8x8xf32>
    %14 = tosa.add %13, %arg0 : (tensor<1x64x8x8xf32>, tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
    %15 = tosa.clamp %14 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
    return %15 : tensor<1x64x8x8xf32>
  }
}

// AFFINE:      module attributes {torch.debug_module_name = "SimpleResNetBlock"} {
// AFFINE-NEXT:   memref.global "private" constant @__constant_64xf32 : memref<64xf32> = dense<0.000000e+00> {alignment = 64 : i64}
// AFFINE-NEXT:   memref.global "private" constant @__constant_64x3x3x64xf32_0 : memref<64x3x3x64xf32> = dense<-0.0151730878> {alignment = 64 : i64}
// AFFINE-NEXT:   memref.global "private" constant @__constant_64x3x3x64xf32 : memref<64x3x3x64xf32> = dense<0.0197670367> {alignment = 64 : i64}
// AFFINE-NEXT:   func.func @forward(%arg0: memref<1x64x8x8xf32>) -> memref<1x64x8x8xf32> {
// AFFINE-NEXT:     %cst = arith.constant 0.0197670367 : f32
// AFFINE-NEXT:     %cst_0 = arith.constant -0.0151730878 : f32
// AFFINE-NEXT:     %cst_1 = arith.constant 3.40282347E+38 : f32
// AFFINE-NEXT:     %cst_2 = arith.constant 0.000000e+00 : f32
// AFFINE-NEXT:     %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 8 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 64 {
// AFFINE-NEXT:             %0 = affine.load %arg0[%arg1, %arg4, %arg2, %arg3] : memref<1x64x8x8xf32>
// AFFINE-NEXT:             affine.store %0, %alloc[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x64xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x10x10x64xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 10 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 10 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 64 {
// AFFINE-NEXT:             affine.store %cst_2, %alloc_3[%arg1, %arg2, %arg3, %arg4] : memref<1x10x10x64xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 8 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 64 {
// AFFINE-NEXT:             affine.store %cst_2, %alloc_4[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x64xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 8 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 64 {
// AFFINE-NEXT:             affine.for %arg5 = 0 to 3 {
// AFFINE-NEXT:               affine.for %arg6 = 0 to 3 {
// AFFINE-NEXT:                 affine.for %arg7 = 0 to 64 {
// AFFINE-NEXT:                   %0 = affine.load %alloc_3[%arg1, %arg2 + %arg5, %arg3 + %arg6, %arg7] : memref<1x10x10x64xf32>
// AFFINE-NEXT:                   %1 = affine.load %alloc_4[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x64xf32>
// AFFINE-NEXT:                   %2 = arith.mulf %0, %cst_0 : f32
// AFFINE-NEXT:                   %3 = arith.addf %1, %2 : f32
// AFFINE-NEXT:                   affine.store %3, %alloc_4[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x64xf32>
// AFFINE-NEXT:                 }
// AFFINE-NEXT:               }
// AFFINE-NEXT:             }
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 64 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 8 {
// AFFINE-NEXT:             %0 = affine.load %alloc_4[%arg1, %arg3, %arg4, %arg2] : memref<1x8x8x64xf32>
// AFFINE-NEXT:             affine.store %0, %alloc_5[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 64 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 8 {
// AFFINE-NEXT:             %0 = affine.load %alloc_5[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:             %1 = arith.minimumf %0, %cst_1 : f32
// AFFINE-NEXT:             %2 = arith.maximumf %1, %cst_2 : f32
// AFFINE-NEXT:             affine.store %2, %alloc_6[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 8 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 64 {
// AFFINE-NEXT:             %0 = affine.load %alloc_6[%arg1, %arg4, %arg2, %arg3] : memref<1x64x8x8xf32>
// AFFINE-NEXT:             affine.store %0, %alloc_7[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x64xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x10x10x64xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 10 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 10 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 64 {
// AFFINE-NEXT:             affine.store %cst_2, %alloc_8[%arg1, %arg2, %arg3, %arg4] : memref<1x10x10x64xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 8 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 64 {
// AFFINE-NEXT:             affine.store %cst_2, %alloc_9[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x64xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 8 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 64 {
// AFFINE-NEXT:             affine.for %arg5 = 0 to 3 {
// AFFINE-NEXT:               affine.for %arg6 = 0 to 3 {
// AFFINE-NEXT:                 affine.for %arg7 = 0 to 64 {
// AFFINE-NEXT:                   %0 = affine.load %alloc_8[%arg1, %arg2 + %arg5, %arg3 + %arg6, %arg7] : memref<1x10x10x64xf32>
// AFFINE-NEXT:                   %1 = affine.load %alloc_9[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x64xf32>
// AFFINE-NEXT:                   %2 = arith.mulf %0, %cst : f32
// AFFINE-NEXT:                   %3 = arith.addf %1, %2 : f32
// AFFINE-NEXT:                   affine.store %3, %alloc_9[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x64xf32>
// AFFINE-NEXT:                 }
// AFFINE-NEXT:               }
// AFFINE-NEXT:             }
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 64 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 8 {
// AFFINE-NEXT:             %0 = affine.load %alloc_9[%arg1, %arg3, %arg4, %arg2] : memref<1x8x8x64xf32>
// AFFINE-NEXT:             affine.store %0, %alloc_10[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 64 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 8 {
// AFFINE-NEXT:             %0 = affine.load %alloc_10[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:             %1 = affine.load %arg0[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:             %2 = arith.addf %0, %1 : f32
// AFFINE-NEXT:             affine.store %2, %alloc_11[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to 1 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 64 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 8 {
// AFFINE-NEXT:           affine.for %arg4 = 0 to 8 {
// AFFINE-NEXT:             %0 = affine.load %alloc_11[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:             %1 = arith.minimumf %0, %cst_1 : f32
// AFFINE-NEXT:             %2 = arith.maximumf %1, %cst_2 : f32
// AFFINE-NEXT:             affine.store %2, %alloc_12[%arg1, %arg2, %arg3, %arg4] : memref<1x64x8x8xf32>
// AFFINE-NEXT:           }
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }
// AFFINE-NEXT:     return %alloc_12 : memref<1x64x8x8xf32>
// AFFINE-NEXT:   }
// AFFINE-NEXT: }


// KERNEL:      module attributes {torch.debug_module_name = "SimpleResNetBlock"} {
// KERNEL-NEXT:   memref.global "private" constant @__constant_64xf32 : memref<64xf32> = dense<0.000000e+00> {alignment = 64 : i64}
// KERNEL-NEXT:   memref.global "private" constant @__constant_64x3x3x64xf32_0 : memref<64x3x3x64xf32> = dense<-0.0151730878> {alignment = 64 : i64}
// KERNEL-NEXT:   memref.global "private" constant @__constant_64x3x3x64xf32 : memref<64x3x3x64xf32> = dense<0.0197670367> {alignment = 64 : i64}
// KERNEL-NEXT:   func.func @forward(%arg0: memref<1x64x8x8xf32>) -> memref<1x64x8x8xf32> {
// KERNEL-NEXT:     %cst = arith.constant 0.0197670367 : f32
// KERNEL-NEXT:     %cst_0 = arith.constant -0.0151730878 : f32
// KERNEL-NEXT:     %cst_1 = arith.constant 3.40282347E+38 : f32
// KERNEL-NEXT:     %cst_2 = arith.constant 0.000000e+00 : f32
// KERNEL-NEXT:     %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// KERNEL-NEXT:     %write_outputs = taskflow.task @Task_0 read_memrefs(%arg0 : memref<1x64x8x8xf32>) write_memrefs(%alloc : memref<1x8x8x64xf32>) [original_read_memrefs(%arg0 : memref<1x64x8x8xf32>), original_write_memrefs(%alloc : memref<1x8x8x64xf32>)] : (memref<1x64x8x8xf32>, memref<1x8x8x64xf32>) -> (memref<1x8x8x64xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x64x8x8xf32>, %arg2: memref<1x8x8x64xf32>):
// KERNEL-NEXT:       %0 = taskflow.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg2 : memref<1x64x8x8xf32>, memref<1x8x8x64xf32>) {
// KERNEL-NEXT:       ^bb0(%arg3: memref<1x64x8x8xf32>, %arg4: memref<1x8x8x64xf32>):
// KERNEL-NEXT:         %4 = neura.counter {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:         %5 = neura.counter {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %6 = neura.counter {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %7 = neura.counter {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:         %8 = memref.load %arg3[%4, %7, %5, %6] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         memref.store %8, %arg4[%4, %5, %6, %7] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg2 : memref<1x8x8x64xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x10x10x64xf32>
// KERNEL-NEXT:     %write_outputs_4 = taskflow.task @Task_1 write_memrefs(%alloc_3 : memref<1x10x10x64xf32>) value_inputs(%cst_2 : f32) [original_write_memrefs(%alloc_3 : memref<1x10x10x64xf32>)] : (memref<1x10x10x64xf32>, f32) -> (memref<1x10x10x64xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: f32):
// KERNEL-NEXT:       %0 = taskflow.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 10 : index} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 10 : index} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg2, %arg1 : f32, memref<1x10x10x64xf32>) {
// KERNEL-NEXT:       ^bb0(%arg3: f32, %arg4: memref<1x10x10x64xf32>):
// KERNEL-NEXT:         %4 = neura.counter {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:         %5 = neura.counter {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 10 : index} : index
// KERNEL-NEXT:         %6 = neura.counter {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 10 : index} : index
// KERNEL-NEXT:         %7 = neura.counter {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:         memref.store %arg3, %arg4[%4, %5, %6, %7] : memref<1x10x10x64xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg1 : memref<1x10x10x64xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// KERNEL-NEXT:     %write_outputs_6 = taskflow.task @Task_2 write_memrefs(%alloc_5 : memref<1x8x8x64xf32>) value_inputs(%cst_2 : f32) [original_write_memrefs(%alloc_5 : memref<1x8x8x64xf32>)] : (memref<1x8x8x64xf32>, f32) -> (memref<1x8x8x64xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: f32):
// KERNEL-NEXT:       %0 = taskflow.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg2, %arg1 : f32, memref<1x8x8x64xf32>) {
// KERNEL-NEXT:       ^bb0(%arg3: f32, %arg4: memref<1x8x8x64xf32>):
// KERNEL-NEXT:         %4 = neura.counter {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:         %5 = neura.counter {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %6 = neura.counter {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %7 = neura.counter {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:         memref.store %arg3, %arg4[%4, %5, %6, %7] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg1 : memref<1x8x8x64xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %write_outputs_7 = taskflow.task @Task_3 read_memrefs(%write_outputs_4, %write_outputs_6 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>) write_memrefs(%write_outputs_6 : memref<1x8x8x64xf32>) value_inputs(%cst_0 : f32) [original_read_memrefs(%alloc_3, %alloc_5 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>), original_write_memrefs(%alloc_5 : memref<1x8x8x64xf32>)] : (memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>, f32) -> (memref<1x8x8x64xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: memref<1x8x8x64xf32>, %arg3: memref<1x8x8x64xf32>, %arg4: f32):
// KERNEL-NEXT:       %0 = taskflow.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) attributes {counter_id = 3 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:       %4 = taskflow.counter parent(%3 : index) attributes {counter_id = 4 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 3 : index} : index
// KERNEL-NEXT:       %5 = taskflow.counter parent(%4 : index) attributes {counter_id = 5 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 3 : index} : index
// KERNEL-NEXT:       %6 = taskflow.counter parent(%5 : index) attributes {counter_id = 6 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg3, %arg4 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, f32) {
// KERNEL-NEXT:       ^bb0(%arg5: memref<1x10x10x64xf32>, %arg6: memref<1x8x8x64xf32>, %arg7: f32):
// KERNEL-NEXT:         %7 = neura.counter {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:         %8 = neura.counter {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %9 = neura.counter {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %10 = neura.counter {counter_id = 3 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:         %11 = neura.counter {counter_id = 4 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 3 : index} : index
// KERNEL-NEXT:         %12 = neura.counter {counter_id = 5 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 3 : index} : index
// KERNEL-NEXT:         %13 = neura.counter {counter_id = 6 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:         %14 = arith.addi %8, %11 : index
// KERNEL-NEXT:         %15 = arith.addi %9, %12 : index
// KERNEL-NEXT:         %16 = memref.load %arg5[%7, %14, %15, %13] : memref<1x10x10x64xf32>
// KERNEL-NEXT:         %17 = memref.load %arg6[%7, %8, %9, %10] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         %18 = arith.mulf %16, %arg7 : f32
// KERNEL-NEXT:         %19 = arith.addf %17, %18 : f32
// KERNEL-NEXT:         memref.store %19, %arg6[%7, %8, %9, %10] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg3 : memref<1x8x8x64xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// KERNEL-NEXT:     %write_outputs_9 = taskflow.task @Task_4 read_memrefs(%write_outputs_7 : memref<1x8x8x64xf32>) write_memrefs(%alloc_8 : memref<1x64x8x8xf32>) [original_read_memrefs(%alloc_5 : memref<1x8x8x64xf32>), original_write_memrefs(%alloc_8 : memref<1x64x8x8xf32>)] : (memref<1x8x8x64xf32>, memref<1x64x8x8xf32>) -> (memref<1x64x8x8xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: memref<1x64x8x8xf32>):
// KERNEL-NEXT:       %0 = taskflow.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg2 : memref<1x8x8x64xf32>, memref<1x64x8x8xf32>) {
// KERNEL-NEXT:       ^bb0(%arg3: memref<1x8x8x64xf32>, %arg4: memref<1x64x8x8xf32>):
// KERNEL-NEXT:         %4 = neura.counter {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:         %5 = neura.counter {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:         %6 = neura.counter {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %7 = neura.counter {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %8 = memref.load %arg3[%4, %6, %7, %5] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         memref.store %8, %arg4[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg2 : memref<1x64x8x8xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// KERNEL-NEXT:     %write_outputs_11 = taskflow.task @Task_5 read_memrefs(%write_outputs_9 : memref<1x64x8x8xf32>) write_memrefs(%alloc_10 : memref<1x64x8x8xf32>) value_inputs(%cst_1, %cst_2 : f32, f32) [original_read_memrefs(%alloc_8 : memref<1x64x8x8xf32>), original_write_memrefs(%alloc_10 : memref<1x64x8x8xf32>)] : (memref<1x64x8x8xf32>, memref<1x64x8x8xf32>, f32, f32) -> (memref<1x64x8x8xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x64x8x8xf32>, %arg2: memref<1x64x8x8xf32>, %arg3: f32, %arg4: f32):
// KERNEL-NEXT:       %0 = taskflow.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg3, %arg4, %arg2 : memref<1x64x8x8xf32>, f32, f32, memref<1x64x8x8xf32>) {
// KERNEL-NEXT:       ^bb0(%arg5: memref<1x64x8x8xf32>, %arg6: f32, %arg7: f32, %arg8: memref<1x64x8x8xf32>):
// KERNEL-NEXT:         %4 = neura.counter {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:         %5 = neura.counter {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:         %6 = neura.counter {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %7 = neura.counter {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %8 = memref.load %arg5[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         %9 = arith.minimumf %8, %arg6 : f32
// KERNEL-NEXT:         %10 = arith.maximumf %9, %arg7 : f32
// KERNEL-NEXT:         memref.store %10, %arg8[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg2 : memref<1x64x8x8xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// KERNEL-NEXT:     %write_outputs_13 = taskflow.task @Task_6 read_memrefs(%write_outputs_11 : memref<1x64x8x8xf32>) write_memrefs(%alloc_12 : memref<1x8x8x64xf32>) [original_read_memrefs(%alloc_10 : memref<1x64x8x8xf32>), original_write_memrefs(%alloc_12 : memref<1x8x8x64xf32>)] : (memref<1x64x8x8xf32>, memref<1x8x8x64xf32>) -> (memref<1x8x8x64xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x64x8x8xf32>, %arg2: memref<1x8x8x64xf32>):
// KERNEL-NEXT:       %0 = taskflow.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg2 : memref<1x64x8x8xf32>, memref<1x8x8x64xf32>) {
// KERNEL-NEXT:       ^bb0(%arg3: memref<1x64x8x8xf32>, %arg4: memref<1x8x8x64xf32>):
// KERNEL-NEXT:         %4 = neura.counter {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:         %5 = neura.counter {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %6 = neura.counter {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %7 = neura.counter {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:         %8 = memref.load %arg3[%4, %7, %5, %6] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         memref.store %8, %arg4[%4, %5, %6, %7] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg2 : memref<1x8x8x64xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x10x10x64xf32>
// KERNEL-NEXT:     %write_outputs_15 = taskflow.task @Task_7 write_memrefs(%alloc_14 : memref<1x10x10x64xf32>) value_inputs(%cst_2 : f32) [original_write_memrefs(%alloc_14 : memref<1x10x10x64xf32>)] : (memref<1x10x10x64xf32>, f32) -> (memref<1x10x10x64xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: f32):
// KERNEL-NEXT:       %0 = taskflow.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 10 : index} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 10 : index} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg2, %arg1 : f32, memref<1x10x10x64xf32>) {
// KERNEL-NEXT:       ^bb0(%arg3: f32, %arg4: memref<1x10x10x64xf32>):
// KERNEL-NEXT:         %4 = neura.counter {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:         %5 = neura.counter {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 10 : index} : index
// KERNEL-NEXT:         %6 = neura.counter {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 10 : index} : index
// KERNEL-NEXT:         %7 = neura.counter {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:         memref.store %arg3, %arg4[%4, %5, %6, %7] : memref<1x10x10x64xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg1 : memref<1x10x10x64xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// KERNEL-NEXT:     %write_outputs_17 = taskflow.task @Task_8 write_memrefs(%alloc_16 : memref<1x8x8x64xf32>) value_inputs(%cst_2 : f32) [original_write_memrefs(%alloc_16 : memref<1x8x8x64xf32>)] : (memref<1x8x8x64xf32>, f32) -> (memref<1x8x8x64xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: f32):
// KERNEL-NEXT:       %0 = taskflow.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg2, %arg1 : f32, memref<1x8x8x64xf32>) {
// KERNEL-NEXT:       ^bb0(%arg3: f32, %arg4: memref<1x8x8x64xf32>):
// KERNEL-NEXT:         %4 = neura.counter {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:         %5 = neura.counter {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %6 = neura.counter {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %7 = neura.counter {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:         memref.store %arg3, %arg4[%4, %5, %6, %7] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg1 : memref<1x8x8x64xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %write_outputs_18 = taskflow.task @Task_9 read_memrefs(%write_outputs_15, %write_outputs_17 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>) write_memrefs(%write_outputs_17 : memref<1x8x8x64xf32>) value_inputs(%cst : f32) [original_read_memrefs(%alloc_14, %alloc_16 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>), original_write_memrefs(%alloc_16 : memref<1x8x8x64xf32>)] : (memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>, f32) -> (memref<1x8x8x64xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: memref<1x8x8x64xf32>, %arg3: memref<1x8x8x64xf32>, %arg4: f32):
// KERNEL-NEXT:       %0 = taskflow.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) attributes {counter_id = 3 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:       %4 = taskflow.counter parent(%3 : index) attributes {counter_id = 4 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 3 : index} : index
// KERNEL-NEXT:       %5 = taskflow.counter parent(%4 : index) attributes {counter_id = 5 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 3 : index} : index
// KERNEL-NEXT:       %6 = taskflow.counter parent(%5 : index) attributes {counter_id = 6 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg3, %arg4 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, f32) {
// KERNEL-NEXT:       ^bb0(%arg5: memref<1x10x10x64xf32>, %arg6: memref<1x8x8x64xf32>, %arg7: f32):
// KERNEL-NEXT:         %7 = neura.counter {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:         %8 = neura.counter {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %9 = neura.counter {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %10 = neura.counter {counter_id = 3 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:         %11 = neura.counter {counter_id = 4 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 3 : index} : index
// KERNEL-NEXT:         %12 = neura.counter {counter_id = 5 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 3 : index} : index
// KERNEL-NEXT:         %13 = neura.counter {counter_id = 6 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:         %14 = arith.addi %8, %11 : index
// KERNEL-NEXT:         %15 = arith.addi %9, %12 : index
// KERNEL-NEXT:         %16 = memref.load %arg5[%7, %14, %15, %13] : memref<1x10x10x64xf32>
// KERNEL-NEXT:         %17 = memref.load %arg6[%7, %8, %9, %10] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         %18 = arith.mulf %16, %arg7 : f32
// KERNEL-NEXT:         %19 = arith.addf %17, %18 : f32
// KERNEL-NEXT:         memref.store %19, %arg6[%7, %8, %9, %10] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg3 : memref<1x8x8x64xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// KERNEL-NEXT:     %write_outputs_20 = taskflow.task @Task_10 read_memrefs(%write_outputs_18 : memref<1x8x8x64xf32>) write_memrefs(%alloc_19 : memref<1x64x8x8xf32>) [original_read_memrefs(%alloc_16 : memref<1x8x8x64xf32>), original_write_memrefs(%alloc_19 : memref<1x64x8x8xf32>)] : (memref<1x8x8x64xf32>, memref<1x64x8x8xf32>) -> (memref<1x64x8x8xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: memref<1x64x8x8xf32>):
// KERNEL-NEXT:       %0 = taskflow.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg2 : memref<1x8x8x64xf32>, memref<1x64x8x8xf32>) {
// KERNEL-NEXT:       ^bb0(%arg3: memref<1x8x8x64xf32>, %arg4: memref<1x64x8x8xf32>):
// KERNEL-NEXT:         %4 = neura.counter {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:         %5 = neura.counter {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:         %6 = neura.counter {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %7 = neura.counter {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %8 = memref.load %arg3[%4, %6, %7, %5] : memref<1x8x8x64xf32>
// KERNEL-NEXT:         memref.store %8, %arg4[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg2 : memref<1x64x8x8xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// KERNEL-NEXT:     %write_outputs_22 = taskflow.task @Task_11 read_memrefs(%write_outputs_20, %arg0 : memref<1x64x8x8xf32>, memref<1x64x8x8xf32>) write_memrefs(%alloc_21 : memref<1x64x8x8xf32>) [original_read_memrefs(%alloc_19, %arg0 : memref<1x64x8x8xf32>, memref<1x64x8x8xf32>), original_write_memrefs(%alloc_21 : memref<1x64x8x8xf32>)] : (memref<1x64x8x8xf32>, memref<1x64x8x8xf32>, memref<1x64x8x8xf32>) -> (memref<1x64x8x8xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x64x8x8xf32>, %arg2: memref<1x64x8x8xf32>, %arg3: memref<1x64x8x8xf32>):
// KERNEL-NEXT:       %0 = taskflow.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg2, %arg3 : memref<1x64x8x8xf32>, memref<1x64x8x8xf32>, memref<1x64x8x8xf32>) {
// KERNEL-NEXT:       ^bb0(%arg4: memref<1x64x8x8xf32>, %arg5: memref<1x64x8x8xf32>, %arg6: memref<1x64x8x8xf32>):
// KERNEL-NEXT:         %4 = neura.counter {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:         %5 = neura.counter {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:         %6 = neura.counter {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %7 = neura.counter {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %8 = memref.load %arg4[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         %9 = memref.load %arg5[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         %10 = arith.addf %8, %9 : f32
// KERNEL-NEXT:         memref.store %10, %arg6[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg3 : memref<1x64x8x8xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// KERNEL-NEXT:     %write_outputs_24 = taskflow.task @Task_12 read_memrefs(%write_outputs_22 : memref<1x64x8x8xf32>) write_memrefs(%alloc_23 : memref<1x64x8x8xf32>) value_inputs(%cst_1, %cst_2 : f32, f32) [original_read_memrefs(%alloc_21 : memref<1x64x8x8xf32>), original_write_memrefs(%alloc_23 : memref<1x64x8x8xf32>)] : (memref<1x64x8x8xf32>, memref<1x64x8x8xf32>, f32, f32) -> (memref<1x64x8x8xf32>) {
// KERNEL-NEXT:     ^bb0(%arg1: memref<1x64x8x8xf32>, %arg2: memref<1x64x8x8xf32>, %arg3: f32, %arg4: f32):
// KERNEL-NEXT:       %0 = taskflow.counter attributes {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:       %1 = taskflow.counter parent(%0 : index) attributes {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:       %2 = taskflow.counter parent(%1 : index) attributes {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       %3 = taskflow.counter parent(%2 : index) attributes {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:       neura.kernel inputs(%arg1, %arg3, %arg4, %arg2 : memref<1x64x8x8xf32>, f32, f32, memref<1x64x8x8xf32>) {
// KERNEL-NEXT:       ^bb0(%arg5: memref<1x64x8x8xf32>, %arg6: f32, %arg7: f32, %arg8: memref<1x64x8x8xf32>):
// KERNEL-NEXT:         %4 = neura.counter {counter_id = 0 : i32, counter_type = "root", lower_bound = 0 : index, step = 1 : index, upper_bound = 1 : index} : index
// KERNEL-NEXT:         %5 = neura.counter {counter_id = 1 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 64 : index} : index
// KERNEL-NEXT:         %6 = neura.counter {counter_id = 2 : i32, counter_type = "relay", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %7 = neura.counter {counter_id = 3 : i32, counter_type = "leaf", lower_bound = 0 : index, step = 1 : index, upper_bound = 8 : index} : index
// KERNEL-NEXT:         %8 = memref.load %arg5[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         %9 = arith.minimumf %8, %arg6 : f32
// KERNEL-NEXT:         %10 = arith.maximumf %9, %arg7 : f32
// KERNEL-NEXT:         memref.store %10, %arg8[%4, %5, %6, %7] : memref<1x64x8x8xf32>
// KERNEL-NEXT:         neura.yield
// KERNEL-NEXT:       }
// KERNEL-NEXT:       taskflow.yield writes(%arg2 : memref<1x64x8x8xf32>)
// KERNEL-NEXT:     }
// KERNEL-NEXT:     return %write_outputs_24 : memref<1x64x8x8xf32>
// KERNEL-NEXT:   }
// KERNEL-NEXT: }

// STREAM:      module attributes {torch.debug_module_name = "SimpleResNetBlock"} {
// STREAM-NEXT:   memref.global "private" constant @__constant_64xf32 : memref<64xf32> = dense<0.000000e+00> {alignment = 64 : i64}
// STREAM-NEXT:   memref.global "private" constant @__constant_64x3x3x64xf32_0 : memref<64x3x3x64xf32> = dense<-0.0151730878> {alignment = 64 : i64}
// STREAM-NEXT:   memref.global "private" constant @__constant_64x3x3x64xf32 : memref<64x3x3x64xf32> = dense<0.0197670367> {alignment = 64 : i64}
// STREAM-NEXT:   func.func @forward(%arg0: memref<1x64x8x8xf32>) -> memref<1x64x8x8xf32> {
// STREAM-NEXT:     %cst = arith.constant 0.0197670367 : f32
// STREAM-NEXT:     %cst_0 = arith.constant -0.0151730878 : f32
// STREAM-NEXT:     %cst_1 = arith.constant 3.40282347E+38 : f32
// STREAM-NEXT:     %cst_2 = arith.constant 0.000000e+00 : f32
// STREAM-NEXT:     %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// STREAM-NEXT:     %write_outputs = taskflow.task @Task_0 read_memrefs(%arg0 : memref<1x64x8x8xf32>) write_memrefs(%alloc : memref<1x8x8x64xf32>) [original_read_memrefs(%arg0 : memref<1x64x8x8xf32>), original_write_memrefs(%alloc : memref<1x8x8x64xf32>)] : (memref<1x64x8x8xf32>, memref<1x8x8x64xf32>) -> (memref<1x8x8x64xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x64x8x8xf32>, %arg2: memref<1x8x8x64xf32>):
// STREAM-NEXT:       affine.for %arg3 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg4 = 0 to 8 {
// STREAM-NEXT:           affine.for %arg5 = 0 to 8 {
// STREAM-NEXT:             affine.for %arg6 = 0 to 64 {
// STREAM-NEXT:               %0 = affine.load %arg1[%arg3, %arg6, %arg4, %arg5] : memref<1x64x8x8xf32>
// STREAM-NEXT:               affine.store %0, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<1x8x8x64xf32>
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield writes(%arg2 : memref<1x8x8x64xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x10x10x64xf32>
// STREAM-NEXT:     %write_outputs_4 = taskflow.task @Task_1 write_memrefs(%alloc_3 : memref<1x10x10x64xf32>) value_inputs(%cst_2 : f32) [original_write_memrefs(%alloc_3 : memref<1x10x10x64xf32>)] : (memref<1x10x10x64xf32>, f32) -> (memref<1x10x10x64xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: f32):
// STREAM-NEXT:       affine.for %arg3 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg4 = 0 to 10 {
// STREAM-NEXT:           affine.for %arg5 = 0 to 10 {
// STREAM-NEXT:             affine.for %arg6 = 0 to 64 {
// STREAM-NEXT:               affine.store %arg2, %arg1[%arg3, %arg4, %arg5, %arg6] : memref<1x10x10x64xf32>
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield writes(%arg1 : memref<1x10x10x64xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// STREAM-NEXT:     %write_outputs_6 = taskflow.task @Task_2 write_memrefs(%alloc_5 : memref<1x8x8x64xf32>) value_inputs(%cst_2 : f32) [original_write_memrefs(%alloc_5 : memref<1x8x8x64xf32>)] : (memref<1x8x8x64xf32>, f32) -> (memref<1x8x8x64xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: f32):
// STREAM-NEXT:       affine.for %arg3 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg4 = 0 to 8 {
// STREAM-NEXT:           affine.for %arg5 = 0 to 8 {
// STREAM-NEXT:             affine.for %arg6 = 0 to 64 {
// STREAM-NEXT:               affine.store %arg2, %arg1[%arg3, %arg4, %arg5, %arg6] : memref<1x8x8x64xf32>
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield writes(%arg1 : memref<1x8x8x64xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %write_outputs_7 = taskflow.task @Task_3 read_memrefs(%write_outputs_4, %write_outputs_6 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>) write_memrefs(%write_outputs_6 : memref<1x8x8x64xf32>) value_inputs(%cst_0 : f32) [original_read_memrefs(%alloc_3, %alloc_5 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>), original_write_memrefs(%alloc_5 : memref<1x8x8x64xf32>)] : (memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>, f32) -> (memref<1x8x8x64xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: memref<1x8x8x64xf32>, %arg3: memref<1x8x8x64xf32>, %arg4: f32):
// STREAM-NEXT:       affine.for %arg5 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg6 = 0 to 8 {
// STREAM-NEXT:           affine.for %arg7 = 0 to 8 {
// STREAM-NEXT:             affine.for %arg8 = 0 to 64 {
// STREAM-NEXT:               affine.for %arg9 = 0 to 3 {
// STREAM-NEXT:                 affine.for %arg10 = 0 to 3 {
// STREAM-NEXT:                   affine.for %arg11 = 0 to 64 {
// STREAM-NEXT:                     %0 = affine.load %arg1[%arg5, %arg6 + %arg9, %arg7 + %arg10, %arg11] : memref<1x10x10x64xf32>
// STREAM-NEXT:                     %1 = affine.load %arg3[%arg5, %arg6, %arg7, %arg8] : memref<1x8x8x64xf32>
// STREAM-NEXT:                     %2 = arith.mulf %0, %arg4 : f32
// STREAM-NEXT:                     %3 = arith.addf %1, %2 : f32
// STREAM-NEXT:                     affine.store %3, %arg3[%arg5, %arg6, %arg7, %arg8] : memref<1x8x8x64xf32>
// STREAM-NEXT:                   }
// STREAM-NEXT:                 }
// STREAM-NEXT:               }
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield writes(%arg3 : memref<1x8x8x64xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// STREAM-NEXT:     %write_outputs_9 = taskflow.task @Task_4_Task_5_fused read_memrefs(%write_outputs_7 : memref<1x8x8x64xf32>) write_memrefs(%alloc_8 : memref<1x64x8x8xf32>) value_inputs(%cst_1, %cst_2 : f32, f32) [original_read_memrefs(%alloc_5 : memref<1x8x8x64xf32>), original_write_memrefs(%alloc_8 : memref<1x64x8x8xf32>)] : (memref<1x8x8x64xf32>, memref<1x64x8x8xf32>, f32, f32) -> (memref<1x64x8x8xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: memref<1x64x8x8xf32>, %arg3: f32, %arg4: f32):
// STREAM-NEXT:       affine.for %arg5 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg6 = 0 to 64 {
// STREAM-NEXT:           affine.for %arg7 = 0 to 8 {
// STREAM-NEXT:             affine.for %arg8 = 0 to 8 {
// STREAM-NEXT:               %0 = affine.load %arg1[%arg5, %arg7, %arg8, %arg6] : memref<1x8x8x64xf32>
// STREAM-NEXT:               %1 = arith.minimumf %0, %arg3 : f32
// STREAM-NEXT:               %2 = arith.maximumf %1, %arg4 : f32
// STREAM-NEXT:               affine.store %2, %arg2[%arg5, %arg6, %arg7, %arg8] : memref<1x64x8x8xf32>
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield writes(%arg2 : memref<1x64x8x8xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// STREAM-NEXT:     %write_outputs_11 = taskflow.task @Task_6 read_memrefs(%write_outputs_9 : memref<1x64x8x8xf32>) write_memrefs(%alloc_10 : memref<1x8x8x64xf32>) [original_read_memrefs(%alloc_8 : memref<1x64x8x8xf32>), original_write_memrefs(%alloc_10 : memref<1x8x8x64xf32>)] : (memref<1x64x8x8xf32>, memref<1x8x8x64xf32>) -> (memref<1x8x8x64xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x64x8x8xf32>, %arg2: memref<1x8x8x64xf32>):
// STREAM-NEXT:       affine.for %arg3 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg4 = 0 to 8 {
// STREAM-NEXT:           affine.for %arg5 = 0 to 8 {
// STREAM-NEXT:             affine.for %arg6 = 0 to 64 {
// STREAM-NEXT:               %0 = affine.load %arg1[%arg3, %arg6, %arg4, %arg5] : memref<1x64x8x8xf32>
// STREAM-NEXT:               affine.store %0, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<1x8x8x64xf32>
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield writes(%arg2 : memref<1x8x8x64xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x10x10x64xf32>
// STREAM-NEXT:     %write_outputs_13 = taskflow.task @Task_7 write_memrefs(%alloc_12 : memref<1x10x10x64xf32>) value_inputs(%cst_2 : f32) [original_write_memrefs(%alloc_12 : memref<1x10x10x64xf32>)] : (memref<1x10x10x64xf32>, f32) -> (memref<1x10x10x64xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: f32):
// STREAM-NEXT:       affine.for %arg3 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg4 = 0 to 10 {
// STREAM-NEXT:           affine.for %arg5 = 0 to 10 {
// STREAM-NEXT:             affine.for %arg6 = 0 to 64 {
// STREAM-NEXT:               affine.store %arg2, %arg1[%arg3, %arg4, %arg5, %arg6] : memref<1x10x10x64xf32>
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield writes(%arg1 : memref<1x10x10x64xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x64xf32>
// STREAM-NEXT:     %write_outputs_15 = taskflow.task @Task_8 write_memrefs(%alloc_14 : memref<1x8x8x64xf32>) value_inputs(%cst_2 : f32) [original_write_memrefs(%alloc_14 : memref<1x8x8x64xf32>)] : (memref<1x8x8x64xf32>, f32) -> (memref<1x8x8x64xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: f32):
// STREAM-NEXT:       affine.for %arg3 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg4 = 0 to 8 {
// STREAM-NEXT:           affine.for %arg5 = 0 to 8 {
// STREAM-NEXT:             affine.for %arg6 = 0 to 64 {
// STREAM-NEXT:               affine.store %arg2, %arg1[%arg3, %arg4, %arg5, %arg6] : memref<1x8x8x64xf32>
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield writes(%arg1 : memref<1x8x8x64xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %write_outputs_16 = taskflow.task @Task_9 read_memrefs(%write_outputs_13, %write_outputs_15 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>) write_memrefs(%write_outputs_15 : memref<1x8x8x64xf32>) value_inputs(%cst : f32) [original_read_memrefs(%alloc_12, %alloc_14 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>), original_write_memrefs(%alloc_14 : memref<1x8x8x64xf32>)] : (memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>, f32) -> (memref<1x8x8x64xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x10x10x64xf32>, %arg2: memref<1x8x8x64xf32>, %arg3: memref<1x8x8x64xf32>, %arg4: f32):
// STREAM-NEXT:       affine.for %arg5 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg6 = 0 to 8 {
// STREAM-NEXT:           affine.for %arg7 = 0 to 8 {
// STREAM-NEXT:             affine.for %arg8 = 0 to 64 {
// STREAM-NEXT:               affine.for %arg9 = 0 to 3 {
// STREAM-NEXT:                 affine.for %arg10 = 0 to 3 {
// STREAM-NEXT:                   affine.for %arg11 = 0 to 64 {
// STREAM-NEXT:                     %0 = affine.load %arg1[%arg5, %arg6 + %arg9, %arg7 + %arg10, %arg11] : memref<1x10x10x64xf32>
// STREAM-NEXT:                     %1 = affine.load %arg3[%arg5, %arg6, %arg7, %arg8] : memref<1x8x8x64xf32>
// STREAM-NEXT:                     %2 = arith.mulf %0, %arg4 : f32
// STREAM-NEXT:                     %3 = arith.addf %1, %2 : f32
// STREAM-NEXT:                     affine.store %3, %arg3[%arg5, %arg6, %arg7, %arg8] : memref<1x8x8x64xf32>
// STREAM-NEXT:                   }
// STREAM-NEXT:                 }
// STREAM-NEXT:               }
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield writes(%arg3 : memref<1x8x8x64xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<1x64x8x8xf32>
// STREAM-NEXT:     %write_outputs_18 = taskflow.task @Task_10_Task_11_Task_12_fused_fused read_memrefs(%write_outputs_16, %arg0 : memref<1x8x8x64xf32>, memref<1x64x8x8xf32>) write_memrefs(%alloc_17 : memref<1x64x8x8xf32>) value_inputs(%cst_1, %cst_2 : f32, f32) [original_read_memrefs(%alloc_14, %arg0 : memref<1x8x8x64xf32>, memref<1x64x8x8xf32>), original_write_memrefs(%alloc_17 : memref<1x64x8x8xf32>)] : (memref<1x8x8x64xf32>, memref<1x64x8x8xf32>, memref<1x64x8x8xf32>, f32, f32) -> (memref<1x64x8x8xf32>) {
// STREAM-NEXT:     ^bb0(%arg1: memref<1x8x8x64xf32>, %arg2: memref<1x64x8x8xf32>, %arg3: memref<1x64x8x8xf32>, %arg4: f32, %arg5: f32):
// STREAM-NEXT:       affine.for %arg6 = 0 to 1 {
// STREAM-NEXT:         affine.for %arg7 = 0 to 64 {
// STREAM-NEXT:           affine.for %arg8 = 0 to 8 {
// STREAM-NEXT:             affine.for %arg9 = 0 to 8 {
// STREAM-NEXT:               %0 = affine.load %arg1[%arg6, %arg8, %arg9, %arg7] : memref<1x8x8x64xf32>
// STREAM-NEXT:               %1 = affine.load %arg2[%arg6, %arg7, %arg8, %arg9] : memref<1x64x8x8xf32>
// STREAM-NEXT:               %2 = arith.addf %0, %1 : f32
// STREAM-NEXT:               %3 = arith.minimumf %2, %arg4 : f32
// STREAM-NEXT:               %4 = arith.maximumf %3, %arg5 : f32
// STREAM-NEXT:               affine.store %4, %arg3[%arg6, %arg7, %arg8, %arg9] : memref<1x64x8x8xf32>
// STREAM-NEXT:             }
// STREAM-NEXT:           }
// STREAM-NEXT:         }
// STREAM-NEXT:       }
// STREAM-NEXT:       taskflow.yield writes(%arg3 : memref<1x64x8x8xf32>)
// STREAM-NEXT:     }
// STREAM-NEXT:     return %write_outputs_18 : memref<1x64x8x8xf32>
// STREAM-NEXT:   }
// STREAM-NEXT: }


// RESOPT:      %write_outputs:3 = taskflow.task @Task_1_Task_0_Task_2_utilfused_utilfused
// RESOPT-SAME: {trip_count = 14592 : i64}
// RESOPT:      taskflow.yield writes(%arg2, %arg3, %arg4 : memref<1x10x10x64xf32>, memref<1x8x8x64xf32>, memref<1x8x8x64xf32>)
// RESOPT:      %write_outputs_5 = taskflow.task @Task_3
// RESOPT:      taskflow.yield writes(%arg3 : memref<1x8x8x64xf32>)
// RESOPT:      %write_outputs_9:2 = taskflow.task @Task_4_Task_5_fused_Task_7_utilfused
// RESOPT-SAME: {trip_count = 10496 : i64}
// RESOPT:      taskflow.yield writes(%arg2, %arg3 : memref<1x64x8x8xf32>, memref<1x10x10x64xf32>)
// RESOPT:      %write_outputs_11:2 = taskflow.task @Task_6_Task_8_utilfused
// RESOPT-SAME: {trip_count = 8192 : i64}
// RESOPT:      taskflow.yield writes(%arg2, %arg3 : memref<1x8x8x64xf32>, memref<1x8x8x64xf32>)
// RESOPT:      %write_outputs_12 = taskflow.task @Task_9
// RESOPT:      taskflow.yield writes(%arg3 : memref<1x8x8x64xf32>)
// RESOPT:      %write_outputs_14 = taskflow.task @Task_10_Task_11_Task_12_fused_fused
// RESOPT:      taskflow.yield writes(%arg3 : memref<1x64x8x8xf32>)
// RESOPT:      return %write_outputs_14 : memref<1x64x8x8xf32>
