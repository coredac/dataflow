// Check that the affine loop nest is correctly transformed to neura.loop_control
// RUN: mlir-neura-opt %s --assign-accelerator --lower-affine-to-neura | FileCheck %s
module attributes {} {
  memref.global @input_data : memref<3x3x3xi32> = uninitialized
  memref.global @output_data : memref<3x3x3xi32> = uninitialized
  func.func @_Z11deep_nestedv() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @output_data : memref<3x3x3xi32>
    %1 = memref.get_global @input_data : memref<3x3x3xi32>
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 3 {
        affine.for %arg2 = 0 to 3 {
          affine.for %arg3 = 0 to 3 {
            affine.for %arg4 = 0 to 3 {
              affine.for %arg5 = 0 to 3 {
                affine.for %arg6 = 0 to 3 {
                  affine.for %arg7 = 0 to 3 {
                    %2 = affine.load %1[%arg0, %arg1, %arg2] : memref<3x3x3xi32>
                    affine.for %arg8 = 0 to 3 {
                      affine.for %arg9 = 0 to 3 {
                        %3 = affine.load %0[%arg0, %arg1, %arg2] : memref<3x3x3xi32>
                        %4 = arith.addi %3, %2 : i32
                        affine.store %4, %0[%arg0, %arg1, %arg2] : memref<3x3x3xi32>
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return %c0_i32 : i32
  }
}

// Verify function signature is preserved
// CHECK-LABEL: func.func @_Z11deep_nestedv() -> i32

// Verify all affine operations are eliminated
// CHECK-NOT: affine.for
// CHECK-NOT: affine.load
// CHECK-NOT: affine.store
// CHECK-NOT: affine.apply

// CHECK-COUNT-10: neura.loop_control
