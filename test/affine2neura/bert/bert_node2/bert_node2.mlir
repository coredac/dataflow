// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura | FileCheck %s
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

// CHECK: func.func @_Z10bert_node2PA128_KiPA768_KfPA128_A768_f(%arg0: memref<?x128xi32>, %arg1: memref<?x768xf32>, %arg2: memref<?x128x768xf32>) attributes {accelerator = "neura"} {
// CHECK-NEXT:  %0 = "neura.constant"() <{value = 768 : index}> : () -> index
// CHECK-NEXT:  %1 = "neura.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT:  %2 = "neura.constant"() <{value = 128 : index}> : () -> index
// CHECK-NEXT:  %3 = "neura.constant"() <{value = false}> : () -> i1
// CHECK-NEXT:  %4 = "neura.constant"() <{value = 30521 : i32}> : () -> i32
// CHECK-NEXT:  %5 = "neura.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-NEXT:  %6 = "neura.constant"() <{value = 30522 : i32}> : () -> i32
// CHECK-NEXT:  %7 = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:  %8 = "neura.cast"(%7) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:  neura.br %8 : i64 to ^bb1
// CHECK-NEXT: ^bb1(%9: i64):  // 2 preds: ^bb0, ^bb9
// CHECK-NEXT:  %10 = "neura.cast"(%9) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:  %11 = "neura.icmp"(%10, %2) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:  neura.cond_br %11 : i1 then to ^bb2 else to ^bb10
// CHECK-NEXT: ^bb2:  // pred: ^bb1
// CHECK-NEXT:  %12 = "neura.cast"(%7) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:  neura.br %12 : i64 to ^bb3
// CHECK-NEXT: ^bb3(%13: i64):  // 2 preds: ^bb2, ^bb8
// CHECK-NEXT:  %14 = "neura.cast"(%13) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:  %15 = "neura.icmp"(%14, %0) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:  neura.cond_br %15 : i1 then to ^bb4 else to ^bb9
// CHECK-NEXT: ^bb4:  // pred: ^bb3
// CHECK-NEXT:  %16 = neura.load_indexed %arg0[%7, %10 : index, index] memref<?x128xi32> : i32
// CHECK-NEXT:  %17 = "neura.icmp"(%16, %6) <{cmpType = "sge"}> : (i32, i32) -> i1
// CHECK-NEXT:  %18 = "neura.sel"(%4, %16, %17) : (i32, i32, i1) -> i32
// CHECK-NEXT:  neura.cond_br %17 : i1 then to ^bb5 else to ^bb6
// CHECK-NEXT: ^bb5:  // pred: ^bb4
// CHECK-NEXT:  neura.br %3 : i1 to ^bb7
// CHECK-NEXT: ^bb6:  // pred: ^bb4
// CHECK-NEXT:  %19 = "neura.icmp"(%16, %5) <{cmpType = "slt"}> : (i32, i32) -> i1
// CHECK-NEXT:  neura.br %19 : i1 to ^bb7
// CHECK-NEXT: ^bb7(%20: i1):  // 2 preds: ^bb5, ^bb6
// CHECK-NEXT:  neura.br to ^bb8
// CHECK-NEXT: ^bb8:  // pred: ^bb7
// CHECK-NEXT:  %21 = "neura.sel"(%5, %18, %20) : (i32, i32, i1) -> i32
// CHECK-NEXT:  %22 = "neura.cast"(%21) <{cast_type = "int_to_index"}> : (i32) -> index
// CHECK-NEXT:  %23 = neura.load_indexed %arg1[%22, %14 : index, index] memref<?x768xf32> : f32
// CHECK-NEXT:  neura.store_indexed %23 to %arg2[%7, %10, %14 : index, index, index] memref<?x128x768xf32> : f32
// CHECK-NEXT:  %24 = "neura.add"(%14, %1) : (index, index) -> index
// CHECK-NEXT:  %25 = "neura.cast"(%24) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:  neura.br %25 : i64 to ^bb3
// CHECK-NEXT: ^bb9:  // pred: ^bb3
// CHECK-NEXT:  %26 = "neura.add"(%10, %1) : (index, index) -> index
// CHECK-NEXT:  %27 = "neura.cast"(%26) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:  neura.br %27 : i64 to ^bb1
// CHECK-NEXT: ^bb10:  // pred: ^bb1
// CHECK-NEXT:  "neura.return"() : () -> ()
// CHECK-NEXT: }
