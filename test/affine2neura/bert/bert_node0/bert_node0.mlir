// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura | FileCheck %s

module attributes {} {
  func.func @_Z10bert_node0PA128_KiPA128_b(%arg0: memref<?x128xi32>, %arg1: memref<?x128xi8>) attributes {} {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg2 = 0 to 128 {
      %0 = affine.load %arg0[0, %arg2] : memref<?x128xi32>
      %1 = arith.cmpi sgt, %0, %c0_i32 : i32
      %2 = arith.extui %1 : i1 to i8
      affine.store %2, %arg1[0, %arg2] : memref<?x128xi8>
    }
    return
  }
}

// CHECK: func.func @_Z10bert_node0PA128_KiPA128_b(%arg0: memref<?x128xi32>, %arg1: memref<?x128xi8>) attributes {accelerator = "neura"} {
// CHECK-NEXT:  %0 = "neura.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT:  %1 = "neura.constant"() <{value = 128 : index}> : () -> index
// CHECK-NEXT:  %2 = "neura.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-NEXT:  %3 = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %3 : index to i64
// CHECK-NEXT:  llvm.br ^bb1(%4 : i64)
// CHECK-NEXT: ^bb1(%5: i64):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %5 : i64 to index
// CHECK-NEXT:  %7 = "neura.icmp"(%6, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:  llvm.cond_br %7, ^bb2, ^bb3
// CHECK-NEXT: ^bb2:  // pred: ^bb1
// CHECK-NEXT:  %8 = memref.load %arg0[%3, %6] : memref<?x128xi32>
// CHECK-NEXT:  %9 = "neura.icmp"(%8, %2) <{cmpType = "sgt"}> : (i32, i32) -> i1
// CHECK-NEXT:  %10 = "neura.cast"(%9) <{cast_type = "extui"}> : (i1) -> i8
// CHECK-NEXT:  memref.store %10, %arg1[%3, %6] : memref<?x128xi8>
// CHECK-NEXT:  %11 = "neura.add"(%6, %0) : (index, index) -> index
// CHECK-NEXT:  %12 = builtin.unrealized_conversion_cast %11 : index to i64
// CHECK-NEXT:  llvm.br ^bb1(%12 : i64)
// CHECK-NEXT: ^bb3:  // pred: ^bb1
// CHECK-NEXT:  return
// CHECK-NEXT:  }
// CHECK-NEXT:  }
