// Wraps the innermost loop within neura.kernel operation.
// RUN: mlir-neura-opt %s \
// RUN:  --wrap-loop-in-kernel \
// RUN:  | FileCheck %s

module attributes {} {
  func.func @_Z17conv3x3_then_reluPA32_A32_KfPA3_A3_A3_S_PS_PA30_A30_fSA_(%arg0: memref<?x32x32xf32>, %arg1: memref<?x3x3x3xf32>, %arg2: memref<?xf32>, %arg3: memref<?x30x30xf32>, %arg4: memref<?x30x30xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 30 {
        affine.for %arg7 = 0 to 30 {
          %0 = affine.load %arg2[%arg5] : memref<?xf32>
          %1 = affine.for %arg8 = 0 to 3 iter_args(%arg9 = %0) -> (f32) {
            %2 = affine.for %arg10 = 0 to 3 iter_args(%arg11 = %arg9) -> (f32) {
              %3 = affine.for %arg12 = 0 to 3 iter_args(%arg13 = %arg11) -> (f32) {
                %4 = affine.load %arg0[%arg8, %arg6 + %arg10, %arg7 + %arg12] : memref<?x32x32xf32>
                %5 = affine.load %arg1[%arg5, %arg8, %arg10, %arg12] : memref<?x3x3x3xf32>
                %6 = arith.mulf %4, %5 : f32
                %7 = arith.addf %arg13, %6 : f32
                affine.yield %7 : f32
              }
              affine.yield %3 : f32
            }
            affine.yield %2 : f32
          }
          affine.store %1, %arg3[%arg5, %arg6, %arg7] : memref<?x30x30xf32>
        }
      }
    }
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 30 {
        affine.for %arg7 = 0 to 30 {
          %0 = affine.load %arg3[%arg5, %arg6, %arg7] : memref<?x30x30xf32>
          %1 = arith.cmpf ogt, %0, %cst : f32
          %2 = arith.select %1, %0, %cst : f32
          affine.store %2, %arg4[%arg5, %arg6, %arg7] : memref<?x30x30xf32>
        }
      }
    }
    return
  }
}


 // CHECK:      module {
 // CHECK-NEXT:   func.func @_Z17conv3x3_then_reluPA32_A32_KfPA3_A3_A3_S_PS_PA30_A30_fSA_(%arg0: memref<?x32x32xf32>, %arg1: memref<?x3x3x3xf32>, %arg2: memref<?xf32>, %arg3: memref<?x30x30xf32>, %arg4: memref<?x30x30xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
 // CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
 // CHECK-NEXT:     affine.for %arg5 = 0 to 64 {
 // CHECK-NEXT:       affine.for %arg6 = 0 to 30 {
 // CHECK-NEXT:         affine.for %arg7 = 0 to 30 {
 // CHECK-NEXT:           %0 = affine.load %arg2[%arg5] : memref<?xf32>
 // CHECK-NEXT:           %1 = affine.for %arg8 = 0 to 3 iter_args(%arg9 = %0) -> (f32) {
 // CHECK-NEXT:             %2 = affine.for %arg10 = 0 to 3 iter_args(%arg11 = %arg9) -> (f32) {
 // CHECK-NEXT:               %3 = neura.kernel ins(%arg0, %arg8, %arg6, %arg10, %arg7, %arg1, %arg5 : memref<?x32x32xf32>, index, index, index, index, memref<?x3x3x3xf32>, index) attributes {kernel_name = "kernel_0"} {
 // CHECK-NEXT:                 %4 = affine.for %arg12 = 0 to 3 iter_args(%arg13 = %arg11) -> (f32) {
 // CHECK-NEXT:                   %5 = affine.load %arg0[%arg8, %arg6 + %arg10, %arg7 + %arg12] : memref<?x32x32xf32>
 // CHECK-NEXT:                   %6 = affine.load %arg1[%arg5, %arg8, %arg10, %arg12] : memref<?x3x3x3xf32>
 // CHECK-NEXT:                   %7 = arith.mulf %5, %6 : f32
 // CHECK-NEXT:                   %8 = arith.addf %arg13, %7 : f32
 // CHECK-NEXT:                   affine.yield %8 : f32
 // CHECK-NEXT:                 }
 // CHECK-NEXT:                 neura.yield %4 : f32
 // CHECK-NEXT:               } : f32
 // CHECK-NEXT:               affine.yield %3 : f32
 // CHECK-NEXT:             }
 // CHECK-NEXT:             affine.yield %2 : f32
 // CHECK-NEXT:           }
 // CHECK-NEXT:           affine.store %1, %arg3[%arg5, %arg6, %arg7] : memref<?x30x30xf32>
 // CHECK-NEXT:         }
 // CHECK-NEXT:       }
 // CHECK-NEXT:     }
 // CHECK-NEXT:     affine.for %arg5 = 0 to 64 {
 // CHECK-NEXT:       affine.for %arg6 = 0 to 30 {
 // CHECK-NEXT:         neura.kernel ins(%arg3, %arg5, %arg6, %cst, %arg4 : memref<?x30x30xf32>, index, index, f32, memref<?x30x30xf32>) attributes {kernel_name = "kernel_1"} {
 // CHECK-NEXT:           affine.for %arg7 = 0 to 30 {
 // CHECK-NEXT:             %0 = affine.load %arg3[%arg5, %arg6, %arg7] : memref<?x30x30xf32>
 // CHECK-NEXT:             %1 = arith.cmpf ogt, %0, %cst : f32
 // CHECK-NEXT:             %2 = arith.select %1, %0, %cst : f32
 // CHECK-NEXT:             affine.store %2, %arg4[%arg5, %arg6, %arg7] : memref<?x30x30xf32>
 // CHECK-NEXT:           }
 // CHECK-NEXT:         }
 // CHECK-NEXT:       }
 // CHECK-NEXT:     }
 // CHECK-NEXT:     return
 // CHECK-NEXT:   }
 // CHECK-NEXT: }

