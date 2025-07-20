// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura | FileCheck %s
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --neura-canonicalize --leverage-predicated-value --transform-ctrl-to-data-flow | FileCheck %s -check-prefix=CTRL2DATA

module attributes {} {
  func.func @_Z27perfect_nested_reduction_2dPA128_i(%arg0: memref<?x128xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.for %arg1 = 0 to 128 iter_args(%arg2 = %c0_i32) -> (i32) {
      %1 = affine.for %arg3 = 0 to 128 iter_args(%arg4 = %arg2) -> (i32) {
        %2 = affine.load %arg0[%arg1, %arg3] : memref<?x128xi32>
        %3 = arith.addi %arg4, %2 : i32
        affine.yield %3 : i32
      }
      affine.yield %1 : i32
    }
    return %0 : i32
  }
}


// CHECK: func.func @_Z27perfect_nested_reduction_2dPA128_i(%arg0: memref<?x128xi32>) -> i32 attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:   %0 = "neura.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT:   %1 = "neura.constant"() <{value = 128 : index}> : () -> index
// CHECK-NEXT:   %2 = "neura.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-NEXT:   %3 = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:   %4 = "neura.cast"(%3) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:   neura.br %4, %2 : i64, i32 to ^bb1
// CHECK-NEXT: ^bb1(%5: i64, %6: i32):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:   %7 = "neura.cast"(%5) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:   %8 = "neura.icmp"(%7, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:   neura.cond_br %8 : i1 then to ^bb2 else to ^bb6
// CHECK-NEXT: ^bb2:  // pred: ^bb1
// CHECK-NEXT:   %9 = "neura.cast"(%3) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:   neura.br %9, %6 : i64, i32 to ^bb3
// CHECK-NEXT: ^bb3(%10: i64, %11: i32):  // 2 preds: ^bb2, ^bb4
// CHECK-NEXT:   %12 = "neura.cast"(%10) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:   %13 = "neura.icmp"(%12, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:   neura.cond_br %13 : i1 then to ^bb4 else to ^bb5
// CHECK-NEXT: ^bb4:  // pred: ^bb3
// CHECK-NEXT:   %14 = neura.load_indexed %arg0[%7, %12 : index, index] memref<?x128xi32> : i32
// CHECK-NEXT:   %15 = "neura.add"(%11, %14) : (i32, i32) -> i32
// CHECK-NEXT:   %16 = "neura.add"(%12, %0) : (index, index) -> index
// CHECK-NEXT:   %17 = "neura.cast"(%16) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:   neura.br %17, %15 : i64, i32 to ^bb3
// CHECK-NEXT: ^bb5:  // pred: ^bb3
// CHECK-NEXT:   %18 = "neura.add"(%7, %0) : (index, index) -> index
// CHECK-NEXT:   %19 = "neura.cast"(%18) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:   neura.br %19, %11 : i64, i32 to ^bb1
// CHECK-NEXT: ^bb6:  // pred: ^bb1
// CHECK-NEXT:   "neura.return"(%6) : (i32) -> ()
// CHECK-NEXT: }

// CTRL2DATA: func.func @_Z27perfect_nested_reduction_2dPA128_i(%arg0: memref<?x128xi32>) -> i32 attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = 1 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_always"(%0) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = 128 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_always"(%2) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{value = 0 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{value = 0 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_always"(%6) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %8 = "neura.cast"(%6) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %10 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %11 = "neura.phi"(%10, %5) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %12 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %13 = "neura.phi"(%12, %9) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %14 = "neura.cast"(%13) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %15 = "neura.icmp"(%14, %3) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %16 = "neura.not"(%15) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %17 = neura.grant_predicate %7, %15 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %18 = "neura.cast"(%17) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %19 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %20 = "neura.phi"(%19, %11) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %21 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %22 = "neura.phi"(%21, %18) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %23 = "neura.cast"(%22) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %24 = "neura.icmp"(%23, %3) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %25 = "neura.not"(%24) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %26 = neura.grant_predicate %14, %24 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %27 = neura.grant_predicate %23, %24 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %28 = neura.load_indexed %arg0[%26, %27 : !neura.data<index, i1>, !neura.data<index, i1>] memref<?x128xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %29 = "neura.add"(%20, %28) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %30 = neura.grant_predicate %1, %24 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %31 = "neura.add"(%27, %30) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %32 = "neura.cast"(%31) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %32 -> %21 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %29 -> %19 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %33 = neura.grant_predicate %14, %25 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %34 = neura.grant_predicate %1, %25 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %35 = "neura.add"(%33, %34) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %36 = "neura.cast"(%35) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %36 -> %12 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %20 -> %10 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     "neura.return"(%11) : (!neura.data<i32, i1>) -> ()
// CTRL2DATA-NEXT:   }