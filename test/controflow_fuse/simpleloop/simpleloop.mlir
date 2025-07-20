// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura | FileCheck %s
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --neura-canonicalize --leverage-predicated-value --transform-ctrl-to-data-flow | FileCheck %s -check-prefix=CTRL2DATA

module attributes {} {
  func.func @_Z10simpleloopv() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.for %arg0 = 0 to 128 iter_args(%arg1 = %c0_i32) -> (i32) {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = arith.addi %arg1, %1 : i32
      affine.yield %2 : i32
    }
    return %0 : i32
  }
}

// CHECK: func.func @_Z10simpleloopv() -> i32 attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = "neura.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT:     %1 = "neura.constant"() <{value = 128 : index}> : () -> index
// CHECK-NEXT:     %2 = "neura.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-NEXT:     %3 = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:     %4 = "neura.cast"(%3) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %4, %2 : i64, i32 to ^bb1
// CHECK-NEXT:   ^bb1(%5: i64, %6: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     %7 = "neura.cast"(%5) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %8 = "neura.icmp"(%7, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %8 : i1 then to ^bb2 else to ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %9 = "neura.cast"(%7) <{cast_type = "index_to_int"}> : (index) -> i32
// CHECK-NEXT:     %10 = "neura.add"(%6, %9) : (i32, i32) -> i32
// CHECK-NEXT:     %11 = "neura.add"(%7, %0) : (index, index) -> index
// CHECK-NEXT:     %12 = "neura.cast"(%11) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %12, %10 : i64, i32 to ^bb1
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     "neura.return"(%6) : (i32) -> ()
// CHECK-NEXT:   }


// CTRL2DATA: func.func @_Z10simpleloopv() -> i32 attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = 1 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_always"(%0) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = 128 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_always"(%2) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{value = 0 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{value = 0 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %7 = "neura.cast"(%6) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %8 = "neura.grant_once"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %9 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %10 = "neura.phi"(%9, %5) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %11 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %12 = "neura.phi"(%11, %8) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %13 = "neura.cast"(%12) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %14 = "neura.icmp"(%13, %3) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %15 = "neura.not"(%14) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %16 = neura.grant_predicate %13, %14 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %17 = "neura.cast"(%16) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %18 = "neura.add"(%10, %17) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %19 = neura.grant_predicate %1, %14 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %20 = "neura.add"(%16, %19) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %21 = "neura.cast"(%20) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %21 -> %11 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %18 -> %9 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     "neura.return"(%10) : (!neura.data<i32, i1>) -> ()
// CTRL2DATA-NEXT:   }
