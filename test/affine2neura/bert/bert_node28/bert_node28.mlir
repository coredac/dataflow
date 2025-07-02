// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura | FileCheck %s
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --leverage-predicated-value --transform-ctrl-to-data-flow | FileCheck %s -check-prefix=CTRL2DATA
module attributes {} {
  func.func @_Z11bert_node28PA128_A768_KfPA768_S0_PA128_A768_f(%arg0: memref<?x128x768xf32>, %arg1: memref<?x768x768xf32>, %arg2: memref<?x128x768xf32>) attributes {} {
    affine.for %arg3 = 0 to 128 {
      affine.for %arg4 = 0 to 768 {
        affine.for %arg5 = 0 to 768 {
          %0 = affine.load %arg0[0, %arg3, %arg5] : memref<?x128x768xf32>
          %1 = affine.load %arg1[0, %arg5, %arg4] : memref<?x768x768xf32>
          %2 = affine.load %arg2[0, %arg3, %arg4] : memref<?x128x768xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          affine.store %4, %arg2[0, %arg3, %arg4] : memref<?x128x768xf32>
        }
      }
    }
    return
  }
}
// CHECK: func.func @_Z11bert_node28PA128_A768_KfPA768_S0_PA128_A768_f(%arg0: memref<?x128x768xf32>, %arg1: memref<?x768x768xf32>, %arg2: memref<?x128x768xf32>) attributes {accelerator = "neura"} {
// CHECK-NEXT: %0 = "neura.constant"() <{value = 768 : index}> : () -> index
// CHECK-NEXT: %1 = "neura.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT: %2 = "neura.constant"() <{value = 128 : index}> : () -> index
// CHECK-NEXT: %3 = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT: %4 = "neura.cast"(%3) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT: neura.br %4 : i64 to ^bb1
// CHECK-NEXT: ^bb1(%5: i64):  // 2 preds: ^bb0, ^bb8
// CHECK-NEXT: %6 = "neura.cast"(%5) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT: %7 = "neura.icmp"(%6, %2) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT: neura.cond_br %7 : i1 then to ^bb2 else to ^bb9
// CHECK-NEXT: ^bb2:  // pred: ^bb1
// CHECK-NEXT: %8 = "neura.cast"(%3) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT: neura.br %8 : i64 to ^bb3
// CHECK-NEXT: ^bb3(%9: i64):  // 2 preds: ^bb2, ^bb7
// CHECK-NEXT: %10 = "neura.cast"(%9) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT: %11 = "neura.icmp"(%10, %0) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT: neura.cond_br %11 : i1 then to ^bb4 else to ^bb8
// CHECK-NEXT: ^bb4:  // pred: ^bb3
// CHECK-NEXT: %12 = "neura.cast"(%3) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT: neura.br %12 : i64 to ^bb5
// CHECK-NEXT: ^bb5(%13: i64):  // 2 preds: ^bb4, ^bb6
// CHECK-NEXT: %14 = "neura.cast"(%13) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT: %15 = "neura.icmp"(%14, %0) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT: neura.cond_br %15 : i1 then to ^bb6 else to ^bb7
// CHECK-NEXT: ^bb6:  // pred: ^bb5
// CHECK-NEXT: %16 = neura.load_indexed %arg0[%3, %6, %14 : index, index, index] memref<?x128x768xf32> : f32
// CHECK-NEXT: %17 = neura.load_indexed %arg1[%3, %14, %10 : index, index, index] memref<?x768x768xf32> : f32
// CHECK-NEXT: %18 = neura.load_indexed %arg2[%3, %6, %10 : index, index, index] memref<?x128x768xf32> : f32
// CHECK-NEXT: %19 = "neura.fmul"(%16, %17) : (f32, f32) -> f32
// CHECK-NEXT: %20 = "neura.fadd"(%18, %19) : (f32, f32) -> f32
// CHECK-NEXT: neura.store_indexed %20 to %arg2[%3, %6, %10 : index, index, index] memref<?x128x768xf32> : f32
// CHECK-NEXT: %21 = "neura.add"(%14, %1) : (index, index) -> index
// CHECK-NEXT: %22 = "neura.cast"(%21) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT: neura.br %22 : i64 to ^bb5
// CHECK-NEXT: ^bb7:  // pred: ^bb5
// CHECK-NEXT: %23 = "neura.add"(%10, %1) : (index, index) -> index
// CHECK-NEXT: %24 = "neura.cast"(%23) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT: neura.br %24 : i64 to ^bb3
// CHECK-NEXT: ^bb8:  // pred: ^bb3
// CHECK-NEXT: %25 = "neura.add"(%6, %1) : (index, index) -> index
// CHECK-NEXT: %26 = "neura.cast"(%25) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT: neura.br %26 : i64 to ^bb1
// CHECK-NEXT: ^bb9:  // pred: ^bb1
// CHECK-NEXT: "neura.return"() : () -> ()


// CTRL2DATA: func.func @_Z11bert_node28PA128_A768_KfPA768_S0_PA128_A768_f(%arg0: memref<?x128x768xf32>, %arg1: memref<?x768x768xf32>, %arg2: memref<?x128x768xf32>) attributes {accelerator = "neura"} {
// CTRL2DATA-NEXT: %0 = "neura.constant"() <{value = 768 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %1 = "neura.grant_always"(%0) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %2 = "neura.constant"() <{value = 1 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %3 = "neura.grant_always"(%2) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %4 = "neura.constant"() <{value = 128 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %5 = "neura.grant_always"(%4) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %6 = "neura.constant"() <{value = 0 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %7 = "neura.grant_always"(%6) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %8 = "neura.cast"(%6) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT: %9 = "neura.grant_once"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT: %10 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT: %11 = "neura.phi"(%10, %9) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT: %12 = "neura.cast"(%11) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %13 = "neura.icmp"(%12, %5) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT: %14 = "neura.not"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT: %15 = neura.grant_predicate %7, %13 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %16 = "neura.cast"(%15) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT: %17 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT: %18 = "neura.phi"(%17, %16) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT: %19 = "neura.cast"(%18) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %20 = "neura.icmp"(%19, %1) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT: %21 = "neura.not"(%20) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT: %22 = neura.grant_predicate %7, %20 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %23 = "neura.cast"(%22) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT: %24 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT: %25 = "neura.phi"(%24, %23) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT: %26 = "neura.cast"(%25) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %27 = "neura.icmp"(%26, %1) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT: %28 = "neura.not"(%27) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT: %29 = neura.grant_predicate %7, %27 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %30 = neura.grant_predicate %12, %27 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %31 = neura.grant_predicate %26, %27 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %32 = neura.load_indexed %arg0[%29, %30, %31 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] memref<?x128x768xf32> : !neura.data<f32, i1>
// CTRL2DATA-NEXT: %33 = neura.grant_predicate %19, %27 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %34 = neura.load_indexed %arg1[%29, %31, %33 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] memref<?x768x768xf32> : !neura.data<f32, i1>
// CTRL2DATA-NEXT: %35 = neura.load_indexed %arg2[%29, %30, %33 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] memref<?x128x768xf32> : !neura.data<f32, i1>
// CTRL2DATA-NEXT: %36 = "neura.fmul"(%32, %34) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT: %37 = "neura.fadd"(%35, %36) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT: neura.store_indexed %37 to %arg2[%29, %30, %33 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] memref<?x128x768xf32> : !neura.data<f32, i1>
// CTRL2DATA-NEXT: %38 = neura.grant_predicate %3, %27 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %39 = "neura.add"(%31, %38) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %40 = "neura.cast"(%39) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT: neura.ctrl_mov %40 -> %24 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT: %41 = neura.grant_predicate %19, %28 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %42 = neura.grant_predicate %3, %28 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %43 = "neura.add"(%41, %42) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %44 = "neura.cast"(%43) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT: neura.ctrl_mov %44 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT: %45 = neura.grant_predicate %12, %21 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %46 = neura.grant_predicate %3, %21 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %47 = "neura.add"(%45, %46) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT: %48 = "neura.cast"(%47) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT: neura.ctrl_mov %48 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT: "neura.return"() : () -> ()
// CTRL2DATA-NEXT: }
