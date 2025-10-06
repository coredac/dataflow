// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: | FileCheck %s

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: | FileCheck %s -check-prefix=CTRL2DATA
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

// CTRL2DATA:        func.func @_Z11bert_node28PA128_A768_KfPA768_S0_PA128_A768_f(%arg0: memref<?x128x768xf32>, %arg1: memref<?x768x768xf32>, %arg2: memref<?x128x768xf32>) attributes {accelerator = "neura"} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = "%arg0"}> : () -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<memref<?x128x768xf32>, i1>) -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = "%arg1"}> : () -> !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<memref<?x768x768xf32>, i1>) -> !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{value = "%arg2"}> : () -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<memref<?x128x768xf32>, i1>) -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{value = 768 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %10 = "neura.constant"() <{value = 128 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %11 = "neura.grant_once"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %12 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %13 = "neura.grant_once"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %14 = neura.reserve : !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %15 = "neura.phi"(%14, %5) : (!neura.data<memref<?x128x768xf32>, i1>, !neura.data<memref<?x128x768xf32>, i1>) -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %16 = neura.reserve : !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     %17 = "neura.phi"(%16, %3) : (!neura.data<memref<?x768x768xf32>, i1>, !neura.data<memref<?x768x768xf32>, i1>) -> !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     %18 = neura.reserve : !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %19 = "neura.phi"(%18, %1) : (!neura.data<memref<?x128x768xf32>, i1>, !neura.data<memref<?x128x768xf32>, i1>) -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %20 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %21 = "neura.phi"(%20, %9) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %22 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %23 = "neura.phi"(%22, %7) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %24 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %25 = "neura.phi"(%24, %13) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %26 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %27 = "neura.phi"(%26, %11) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %28 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %29 = "neura.phi"(%28, %13) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %30 = "neura.icmp"(%29, %27) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %31 = neura.grant_predicate %25, %30 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %32 = neura.grant_predicate %23, %30 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %33 = neura.grant_predicate %29, %30 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %34 = neura.grant_predicate %21, %30 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %35 = neura.grant_predicate %27, %30 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %36 = neura.grant_predicate %19, %30 : !neura.data<memref<?x128x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %37 = neura.grant_predicate %17, %30 : !neura.data<memref<?x768x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     %38 = neura.grant_predicate %15, %30 : !neura.data<memref<?x128x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %39 = neura.reserve : !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %40 = "neura.phi"(%39, %38) : (!neura.data<memref<?x128x768xf32>, i1>, !neura.data<memref<?x128x768xf32>, i1>) -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %41 = neura.reserve : !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     %42 = "neura.phi"(%41, %37) : (!neura.data<memref<?x768x768xf32>, i1>, !neura.data<memref<?x768x768xf32>, i1>) -> !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     %43 = neura.reserve : !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %44 = "neura.phi"(%43, %36) : (!neura.data<memref<?x128x768xf32>, i1>, !neura.data<memref<?x128x768xf32>, i1>) -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %45 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %46 = "neura.phi"(%45, %35) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %47 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %48 = "neura.phi"(%47, %34) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %49 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %50 = "neura.phi"(%49, %33) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %51 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %52 = "neura.phi"(%51, %31) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %53 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %54 = "neura.phi"(%53, %32) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %55 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %56 = "neura.phi"(%55, %31) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %57 = "neura.icmp"(%56, %54) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %58 = neura.grant_predicate %52, %57 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %59 = neura.grant_predicate %54, %57 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %60 = neura.grant_predicate %44, %57 : !neura.data<memref<?x128x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %61 = neura.grant_predicate %50, %57 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %62 = neura.grant_predicate %42, %57 : !neura.data<memref<?x768x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     %63 = neura.grant_predicate %56, %57 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %64 = neura.grant_predicate %40, %57 : !neura.data<memref<?x128x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %65 = neura.grant_predicate %48, %57 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %66 = neura.grant_predicate %46, %57 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %67 = "neura.not"(%57) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %68 = neura.grant_predicate %50, %67 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %69 = neura.grant_predicate %48, %67 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %70 = neura.grant_predicate %46, %67 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %71 = neura.grant_predicate %52, %67 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %72 = neura.grant_predicate %54, %67 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %73 = neura.grant_predicate %44, %67 : !neura.data<memref<?x128x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %74 = neura.grant_predicate %42, %67 : !neura.data<memref<?x768x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     %75 = neura.grant_predicate %40, %67 : !neura.data<memref<?x128x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %76 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %77 = "neura.phi"(%76, %66) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %78 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %79 = "neura.phi"(%78, %65) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %80 = neura.reserve : !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %81 = "neura.phi"(%80, %64) : (!neura.data<memref<?x128x768xf32>, i1>, !neura.data<memref<?x128x768xf32>, i1>) -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %82 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %83 = "neura.phi"(%82, %63) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %84 = neura.reserve : !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     %85 = "neura.phi"(%84, %62) : (!neura.data<memref<?x768x768xf32>, i1>, !neura.data<memref<?x768x768xf32>, i1>) -> !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     %86 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %87 = "neura.phi"(%86, %61) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %88 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %89 = "neura.phi"(%88, %58) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %90 = neura.reserve : !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %91 = "neura.phi"(%90, %60) : (!neura.data<memref<?x128x768xf32>, i1>, !neura.data<memref<?x128x768xf32>, i1>) -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %92 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %93 = "neura.phi"(%92, %59) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %94 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %95 = "neura.phi"(%94, %58) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %96 = "neura.icmp"(%95, %93) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %97 = neura.grant_predicate %91, %96 : !neura.data<memref<?x128x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %98 = neura.grant_predicate %89, %96 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %99 = neura.grant_predicate %87, %96 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %100 = neura.grant_predicate %95, %96 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %101 = neura.grant_predicate %85, %96 : !neura.data<memref<?x768x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     %102 = neura.grant_predicate %83, %96 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %103 = neura.grant_predicate %81, %96 : !neura.data<memref<?x128x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %104 = neura.grant_predicate %79, %96 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %105 = neura.grant_predicate %93, %96 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %106 = neura.grant_predicate %77, %96 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %107 = "neura.not"(%96) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %108 = neura.grant_predicate %83, %107 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %109 = neura.grant_predicate %79, %107 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %110 = neura.grant_predicate %93, %107 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %111 = neura.grant_predicate %89, %107 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %112 = neura.grant_predicate %87, %107 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %113 = neura.grant_predicate %77, %107 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %114 = neura.grant_predicate %91, %107 : !neura.data<memref<?x128x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %115 = neura.grant_predicate %85, %107 : !neura.data<memref<?x768x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     %116 = neura.grant_predicate %81, %107 : !neura.data<memref<?x128x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %117 = neura.load_indexed %97[%98, %99, %100 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x128x768xf32>, i1> : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %118 = neura.load_indexed %101[%98, %100, %102 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x768x768xf32>, i1> : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %119 = neura.load_indexed %103[%98, %99, %102 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x128x768xf32>, i1> : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %120 = "neura.fmul"(%117, %118) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %121 = "neura.fadd"(%119, %120) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %121 to %103[%98, %99, %102 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x128x768xf32>, i1> : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %122 = "neura.add"(%100, %104) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %122 -> %94 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %105 -> %92 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %97 -> %90 : !neura.data<memref<?x128x768xf32>, i1> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %98 -> %88 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %99 -> %86 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %101 -> %84 : !neura.data<memref<?x768x768xf32>, i1> !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %102 -> %82 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %103 -> %80 : !neura.data<memref<?x128x768xf32>, i1> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %104 -> %78 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %106 -> %76 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %123 = "neura.add"(%108, %109) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %123 -> %55 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %110 -> %53 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %111 -> %51 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %112 -> %49 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %109 -> %47 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %113 -> %45 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %114 -> %43 : !neura.data<memref<?x128x768xf32>, i1> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %115 -> %41 : !neura.data<memref<?x768x768xf32>, i1> !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %116 -> %39 : !neura.data<memref<?x128x768xf32>, i1> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %124 = "neura.add"(%68, %69) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %124 -> %28 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %70 -> %26 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %71 -> %24 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %72 -> %22 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %69 -> %20 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %73 -> %18 : !neura.data<memref<?x128x768xf32>, i1> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %74 -> %16 : !neura.data<memref<?x768x768xf32>, i1> !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %75 -> %14 : !neura.data<memref<?x128x768xf32>, i1> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     "neura.return"() : () -> ()
// CTRL2DATA-NEXT:   }
