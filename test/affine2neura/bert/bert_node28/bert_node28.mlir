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
// CHECK-NEXT: %0 = "neura.constant"() <{predicate = true, value = 768 : index}> : () -> index
// CHECK-NEXT: %1 = "neura.constant"() <{predicate = true, value = 1 : index}> : () -> index
// CHECK-NEXT: %2 = "neura.constant"() <{predicate = true, value = 128 : index}> : () -> index
// CHECK-NEXT: %3 = "neura.constant"() <{predicate = true, value = 0 : index}> : () -> index
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


// CTRL2DATA:     func.func @_Z11bert_node28PA128_A768_KfPA768_S0_PA128_A768_f(%arg0: memref<?x128x768xf32>, %arg1: memref<?x768x768xf32>, %arg2: memref<?x128x768xf32>) attributes {accelerator = "neura"} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{predicate = true, value = "%arg0"}> : () -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<memref<?x128x768xf32>, i1>) -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{predicate = true, value = "%arg1"}> : () -> !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<memref<?x768x768xf32>, i1>) -> !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{predicate = true, value = "%arg2"}> : () -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<memref<?x128x768xf32>, i1>) -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 768 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %10 = "neura.constant"() <{predicate = true, value = 128 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %11 = "neura.grant_once"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %12 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %13 = "neura.grant_once"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %14 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %15 = "neura.phi"(%14, %11) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %16 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %17 = "neura.phi"(%16, %13) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %18 = "neura.icmp"(%17, %15) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %19 = "neura.not"(%18) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %20 = neura.grant_predicate %13, %18 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %21 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %22 = "neura.phi"(%21, %7) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %23 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %24 = "neura.phi"(%23, %20) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %25 = "neura.icmp"(%24, %22) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %26 = neura.grant_predicate %13, %25 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %27 = "neura.not"(%25) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %28 = neura.grant_predicate %17, %27 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %29 = neura.grant_predicate %9, %27 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %30 = neura.grant_predicate %11, %27 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %31 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %32 = "neura.phi"(%31, %7) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %33 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %34 = "neura.phi"(%33, %26) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %35 = "neura.icmp"(%34, %32) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %36 = neura.grant_predicate %1, %35 : !neura.data<memref<?x128x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %37 = neura.grant_predicate %13, %35 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %38 = neura.grant_predicate %17, %35 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %39 = neura.grant_predicate %34, %35 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %40 = neura.grant_predicate %3, %35 : !neura.data<memref<?x768x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x768x768xf32>, i1>
// CTRL2DATA-NEXT:     %41 = neura.grant_predicate %24, %35 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %42 = neura.grant_predicate %5, %35 : !neura.data<memref<?x128x768xf32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128x768xf32>, i1>
// CTRL2DATA-NEXT:     %43 = neura.grant_predicate %9, %35 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %44 = neura.grant_predicate %7, %35 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %45 = "neura.not"(%35) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %46 = neura.grant_predicate %24, %45 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %47 = neura.grant_predicate %9, %45 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %48 = neura.grant_predicate %7, %45 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %49 = neura.load_indexed %36[%37, %38, %39 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x128x768xf32>, i1> : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %50 = neura.load_indexed %40[%37, %39, %41 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x768x768xf32>, i1> : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %51 = neura.load_indexed %42[%37, %38, %41 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x128x768xf32>, i1> : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %52 = "neura.fmul"(%49, %50) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %53 = "neura.fadd"(%51, %52) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %53 to %42[%37, %38, %41 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x128x768xf32>, i1> : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %54 = "neura.add"(%39, %43) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %54 -> %33 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %44 -> %31 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %55 = "neura.add"(%46, %47) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %55 -> %23 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %48 -> %21 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %56 = "neura.add"(%28, %29) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %56 -> %16 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %30 -> %14 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     "neura.return"() : () -> ()
// CTRL2DATA-NEXT:   }
