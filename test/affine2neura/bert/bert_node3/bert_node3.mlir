// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura | FileCheck %s
module attributes {} {
  func.func @_Z10bert_node3PA128_A768_KfS2_PA128_A768_f(%arg0: memref<?x128x768xf32>, %arg1: memref<?x128x768xf32>, %arg2: memref<?x128x768xf32>) attributes {} {
    affine.for %arg3 = 0 to 128 {
      affine.for %arg4 = 0 to 768 {
        %0 = affine.load %arg0[0, %arg3, %arg4] : memref<?x128x768xf32>
        %1 = affine.load %arg1[0, %arg3, %arg4] : memref<?x128x768xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %arg2[0, %arg3, %arg4] : memref<?x128x768xf32>
      }
    }
    return
  }
}

// CHECK: func.func @_Z10bert_node3PA128_A768_KfS2_PA128_A768_f(%arg0: memref<?x128x768xf32>, %arg1: memref<?x128x768xf32>, %arg2: memref<?x128x768xf32>) attributes {accelerator = "neura"} {
// CHECK-NEXT: %0 = "neura.constant"() <{value = 768 : index}> : () -> index
// CHECK-NEXT: %1 = "neura.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT: %2 = "neura.constant"() <{value = 128 : index}> : () -> index
// CHECK-NEXT: %3 = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT: %4 = "neura.cast"(%3) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT: neura.br %4 : i64 to ^bb1
// CHECK-NEXT: ^bb1(%5: i64):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT: %6 = "neura.cast"(%5) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT: %7 = "neura.icmp"(%6, %2) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT: neura.cond_br %7 : i1 then :  to ^bb2 else :  to ^bb6
// CHECK-NEXT: ^bb2:  // pred: ^bb1
// CHECK-NEXT: %8 = "neura.cast"(%3) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT: neura.br %8 : i64 to ^bb3
// CHECK-NEXT: ^bb3(%9: i64):  // 2 preds: ^bb2, ^bb4
// CHECK-NEXT: %10 = "neura.cast"(%9) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT: %11 = "neura.icmp"(%10, %0) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT: neura.cond_br %11 : i1 then :  to ^bb4 else :  to ^bb5
// CHECK-NEXT: ^bb4:  // pred: ^bb3
// CHECK-NEXT: %12 = neura.load_indexed %arg0[%3, %6, %10] memref<?x128x768xf32> : f32
// CHECK-NEXT: %13 = neura.load_indexed %arg1[%3, %6, %10] memref<?x128x768xf32> : f32
// CHECK-NEXT: %14 = "neura.fadd"(%12, %13) : (f32, f32) -> f32
// CHECK-NEXT: neura.store_indexed %14 to %arg2[%3, %6, %10] memref<?x128x768xf32> : f32
// CHECK-NEXT: %15 = "neura.add"(%10, %1) : (index, index) -> index
// CHECK-NEXT: %16 = "neura.cast"(%15) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT: neura.br %16 : i64 to ^bb3
// CHECK-NEXT: ^bb5:  // pred: ^bb3
// CHECK-NEXT: %17 = "neura.add"(%6, %1) : (index, index) -> index
// CHECK-NEXT: %18 = "neura.cast"(%17) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT: neura.br %18 : i64 to ^bb1
// CHECK-NEXT: ^bb6:  // pred: ^bb1
// CHECK-NEXT: "neura.return"() : () -> ()
// CHECK-NEXT: }
