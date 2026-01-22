// RUN: mlir-opt %s \
// RUN: --lower-affine \
// RUN: --convert-scf-to-cf \
// RUN: --convert-cf-to-llvm -o %t-llvm.mlir

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
// RUN: --promote-func-arg-to-const \
// RUN: --canonicalize-return \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: | FileCheck %s -check-prefix=CTRL2DATA

module attributes {} {
  func.func @_Z11for_with_ifPi(%arg0: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c-5_i32 = arith.constant -5 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1000_i32 = arith.constant 1000 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %c0_i32) -> (i32) {
      %1 = arith.cmpi sge, %arg2, %c1000_i32 : i32
      %2 = scf.if %1 -> (i32) {
        %7 = arith.addi %arg2, %c-5_i32 : i32
        scf.yield %7 : i32
      } else {
        scf.yield %arg2 : i32
      }
      %3 = memref.load %arg0[%arg1] : memref<?xi32>
      %4 = arith.muli %3, %c2_i32 : i32
      %5 = arith.addi %4, %c1_i32 : i32
      %6 = arith.addi %2, %5 : i32
      scf.yield %6 : i32
    }
    return %0 : i32
  }
}

// CHECK:        func.func @_Z11for_with_ifPi(%arg0: memref<?xi32>) -> i32 attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = "neura.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-NEXT:     %1 = "neura.constant"() <{value = 1000 : i32}> : () -> i32
// CHECK-NEXT:     %2 = "neura.constant"() <{value = 2 : i32}> : () -> i32
// CHECK-NEXT:     %3 = "neura.constant"() <{value = 1 : i32}> : () -> i32
// CHECK-NEXT:     %4 = "neura.constant"() <{value = -5 : i32}> : () -> i32
// CHECK-NEXT:     %5 = "neura.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT:     %6 = "neura.constant"() <{value = 128 : index}> : () -> index
// CHECK-NEXT:     %7 = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:     %8 = "neura.cast"(%7) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %8, %0 : i64, i32 to ^bb1
// CHECK-NEXT:   ^bb1(%9: i64, %10: i32):  // 2 preds: ^bb0, ^bb6
// CHECK-NEXT:     %11 = "neura.cast"(%9) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %12 = "neura.icmp"(%11, %6) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %12 : i1 then to ^bb2 else to ^bb7
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %13 = "neura.icmp"(%10, %1) <{cmpType = "sge"}> : (i32, i32) -> i1
// CHECK-NEXT:     neura.cond_br %13 : i1 then to ^bb3 else to ^bb4
// CHECK-NEXT:   ^bb3:  // pred: ^bb2
// CHECK-NEXT:     %14 = "neura.add"(%10, %4) : (i32, i32) -> i32
// CHECK-NEXT:     neura.br %14 : i32 to ^bb5
// CHECK-NEXT:   ^bb4:  // pred: ^bb2
// CHECK-NEXT:     neura.br %10 : i32 to ^bb5
// CHECK-NEXT:   ^bb5(%15: i32):  // 2 preds: ^bb3, ^bb4
// CHECK-NEXT:     neura.br to ^bb6
// CHECK-NEXT:   ^bb6:  // pred: ^bb5
// CHECK-NEXT:     %16 = neura.load_indexed %arg0[%11 : index] memref<?xi32> : i32
// CHECK-NEXT:     %17 = "neura.mul"(%16, %2) : (i32, i32) -> i32
// CHECK-NEXT:     %18 = "neura.add"(%17, %3) : (i32, i32) -> i32
// CHECK-NEXT:     %19 = "neura.add"(%15, %18) : (i32, i32) -> i32
// CHECK-NEXT:     %20 = "neura.add"(%11, %5) : (index, index) -> index
// CHECK-NEXT:     %21 = "neura.cast"(%20) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %21, %19 : i64, i32 to ^bb1
// CHECK-NEXT:   ^bb7:  // pred: ^bb1
// CHECK-NEXT:     "neura.return"(%10) : (i32) -> ()
// CHECK-NEXT:   }

// CTRL2DATA:      func.func @_Z11for_with_ifPi(%arg0: memref<?xi32>) -> i32 attributes {accelerator = "neura", dataflow_mode = "predicate", llvm.linkage = #llvm.linkage<external>} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = "%arg0"}> : () -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = 0 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{value = 1000 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{value = 2 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{value = 1 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %10 = "neura.constant"() <{value = -5 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %11 = "neura.grant_once"(%10) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %12 = "neura.constant"() <{value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %13 = "neura.grant_once"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %14 = "neura.constant"() <{value = 128 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %15 = "neura.grant_once"(%14) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %16 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %17 = "neura.grant_once"(%16) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %18 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %19 = neura.phi_start %13, %18 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %20 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %21 = neura.phi_start %9, %20 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %22 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %23 = neura.phi_start %7, %22 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %24 = neura.reserve : !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %25 = neura.phi_start %1, %24 : !neura.data<memref<?xi32>, i1>, !neura.data<memref<?xi32>, i1> -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %26 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %27 = neura.phi_start %11, %26 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %28 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %29 = neura.phi_start %5, %28 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %30 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %31 = neura.phi_start %15, %30 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %32 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %33 = neura.phi_start %3, %32 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %34 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %35 = neura.phi_start %17, %34 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %36 = "neura.icmp"(%35, %31) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %37 = neura.grant_predicate %33, %36 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %38 = neura.grant_predicate %29, %36 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %39 = neura.grant_predicate %27, %36 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %40 = neura.grant_predicate %25, %36 : !neura.data<memref<?xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %41 = neura.grant_predicate %35, %36 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %42 = neura.grant_predicate %23, %36 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %43 = neura.grant_predicate %21, %36 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %44 = neura.grant_predicate %19, %36 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %45 = neura.grant_predicate %31, %36 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %46 = "neura.not"(%36) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %47 = neura.grant_predicate %33, %46 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.return_value %47 : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %48 = "neura.icmp"(%37, %38) <{cmpType = "sge"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %49 = neura.grant_predicate %37, %48 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %50 = neura.grant_predicate %39, %48 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %51 = neura.grant_predicate %40, %48 : !neura.data<memref<?xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %52 = neura.grant_predicate %41, %48 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %53 = neura.grant_predicate %42, %48 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %54 = neura.grant_predicate %43, %48 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %55 = neura.grant_predicate %44, %48 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %56 = neura.grant_predicate %45, %48 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %57 = neura.grant_predicate %38, %48 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %58 = "neura.not"(%48) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %59 = neura.grant_predicate %37, %58 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %60 = neura.grant_predicate %40, %58 : !neura.data<memref<?xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %61 = neura.grant_predicate %41, %58 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %62 = neura.grant_predicate %42, %58 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %63 = neura.grant_predicate %43, %58 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %64 = neura.grant_predicate %44, %58 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %65 = neura.grant_predicate %45, %58 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %66 = neura.grant_predicate %38, %58 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %67 = neura.grant_predicate %39, %58 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %68 = "neura.add"(%49, %50) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %69 = "neura.phi"(%50, %67) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %70 = "neura.phi"(%57, %66) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %71 = "neura.phi"(%56, %65) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %72 = "neura.phi"(%55, %64) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %73 = "neura.phi"(%54, %63) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %74 = "neura.phi"(%53, %62) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %75 = "neura.phi"(%52, %61) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %76 = "neura.phi"(%51, %60) : (!neura.data<memref<?xi32>, i1>, !neura.data<memref<?xi32>, i1>) -> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     %77 = "neura.phi"(%68, %59) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %78 = neura.load_indexed %76[%75 : !neura.data<i64, i1>] !neura.data<memref<?xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %79 = "neura.mul"(%78, %74) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %80 = "neura.add"(%79, %73) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %81 = "neura.add"(%77, %80) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %82 = "neura.add"(%75, %72) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %82 -> %34 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %81 -> %32 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %71 -> %30 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %70 -> %28 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %69 -> %26 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %76 -> %24 : !neura.data<memref<?xi32>, i1> !neura.data<memref<?xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %74 -> %22 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %73 -> %20 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %72 -> %18 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.yield
// CTRL2DATA-NEXT:   }