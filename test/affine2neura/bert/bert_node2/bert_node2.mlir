// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-llvm-to-neura | FileCheck %s
module attributes {} {
  func.func @_Z10bert_node2PA128_KiPA768_KfPA128_A768_f(%arg0: memref<?x128xi32>, %arg1: memref<?x768xf32>, %arg2: memref<?x128x768xf32>) attributes {} {
    %false = arith.constant false
    %c30521_i32 = arith.constant 30521 : i32
    %c0_i32 = arith.constant 0 : i32
    %c30522_i32 = arith.constant 30522 : i32
    affine.for %arg3 = 0 to 128 {
      affine.for %arg4 = 0 to 768 {
        %0 = affine.load %arg0[0, %arg3] : memref<?x128xi32>
        %1 = arith.cmpi sge, %0, %c30522_i32 : i32
        %2 = arith.select %1, %c30521_i32, %0 : i32
        %3 = scf.if %1 -> (i1) {
          scf.yield %false : i1
        } else {
          %7 = arith.cmpi slt, %0, %c0_i32 : i32
          scf.yield %7 : i1
        }
        %4 = arith.select %3, %c0_i32, %2 : i32
        %5 = arith.index_cast %4 : i32 to index
        %6 = memref.load %arg1[%5, %arg4] : memref<?x768xf32>
        affine.store %6, %arg2[0, %arg3, %arg4] : memref<?x128x768xf32>
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @_Z10bert_node2PA128_KiPA768_KfPA128_A768_f
// CHECK-NOT: arith.
// CHECK-NOT: affine.
// CHECK-NOT: llvm.