// Wraps the innermost loop within neura.kernel operation.
// RUN: mlir-neura-opt %s \
// RUN:  --wrap-loop-in-kernel \
// RUN:  | FileCheck %s

module attributes {} {
  func.func @_Z10bert_node1PA1_A1_A1_A1_A128_bPA1_A128_S1_(%arg0: memref<?x1x1x1x1x128xi8>, %arg1: memref<?x1x128x1x1x128xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 128 {
        %0 = affine.load %arg0[0, 0, 0, 0, 0, %arg3] : memref<?x1x1x1x1x128xi8>
        affine.store %0, %arg1[0, 0, %arg2, 0, 0, %arg3] : memref<?x1x128x1x1x128xi8>
      }
    }
    return
  }
}

 // CHECK:      module {
 // CHECK-NEXT:   func.func @_Z10bert_node1PA1_A1_A1_A1_A128_bPA1_A128_S1_(%arg0: memref<?x1x1x1x1x128xi8>, %arg1: memref<?x1x128x1x1x128xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
 // CHECK-NEXT:     affine.for %arg2 = 0 to 128 {
 // CHECK-NEXT:       neura.kernel ins(%arg0, %arg1, %arg2 : memref<?x1x1x1x1x128xi8>, memref<?x1x128x1x1x128xi8>, index) attributes {kernel_name = "kernel_0"} {
 // CHECK-NEXT:         affine.for %arg3 = 0 to 128 {
 // CHECK-NEXT:           %0 = affine.load %arg0[0, 0, 0, 0, 0, %arg3] : memref<?x1x1x1x1x128xi8>
 // CHECK-NEXT:           affine.store %0, %arg1[0, 0, %arg2, 0, 0, %arg3] : memref<?x1x128x1x1x128xi8>
 // CHECK-NEXT:         }
 // CHECK-NEXT:       }
 // CHECK-NEXT:     }
 // CHECK-NEXT:     return
 // CHECK-NEXT:   }
 // CHECK-NEXT: }