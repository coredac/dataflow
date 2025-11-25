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
// RUN: | FileCheck %s --check-prefix=CAST

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --promote-func-arg-to-const \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: | FileCheck %s -check-prefix=CTRL2DATA

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --promote-func-arg-to-const \
// RUN: --fold-constant \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: --fold-constant \
// RUN: --insert-data-mov \
// RUN: --map-to-accelerator="mapping-strategy=heuristic backtrack-config=customized" \
// RUN: --architecture-spec=../../arch_spec/architecture.yaml \
// RUN: | FileCheck %s -check-prefix=MAPPING

module attributes {} {
  func.func @_Z10bert_node1PA1_A1_A1_A1_A128_bPA1_A128_S1_(%arg0: memref<?x1x1x1x1x128xi8>, %arg1: memref<?x1x128x1x1x128xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 128 {
        %0 = affine.load %arg0[0, 0, 0, 0, 0, %arg3] : memref<?x1x1x1x1x128xi8>
        affine.store %0, %arg1[0, 0, %arg2, 0, 0, %arg3] : memref<?x1x128x1x1x128xi8>
      }
    }
    return
  }
}


// CHECK:   func.func @_Z10bert_node1PA1_A1_A1_A1_A128_bPA1_A128_S1_(%arg0: memref<?x1x1x1x1x128xi8>, %arg1: memref<?x1x128x1x1x128xi8>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = "neura.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT:     %1 = "neura.constant"() <{value = 128 : index}> : () -> index
// CHECK-NEXT:     %2 = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:     %3 = "neura.cast"(%2) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %3 : i64 to ^bb1
// CHECK-NEXT:   ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:     %5 = "neura.cast"(%4) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %6 = "neura.icmp"(%5, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %6 : i1 then to ^bb2 else to ^bb6
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %7 = "neura.cast"(%2) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %7 : i64 to ^bb3
// CHECK-NEXT:   ^bb3(%8: i64):  // 2 preds: ^bb2, ^bb4
// CHECK-NEXT:     %9 = "neura.cast"(%8) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %10 = "neura.icmp"(%9, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %10 : i1 then to ^bb4 else to ^bb5
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %11 = neura.load_indexed %arg0[%2, %2, %2, %2, %2, %9 : index, index, index, index, index, index] memref<?x1x1x1x1x128xi8> : i8
// CHECK-NEXT:     neura.store_indexed %11 to %arg1[%2, %2, %5, %2, %2, %9 : index, index, index, index, index, index] memref<?x1x128x1x1x128xi8> : i8
// CHECK-NEXT:     %12 = "neura.add"(%9, %0) : (index, index) -> index
// CHECK-NEXT:     %13 = "neura.cast"(%12) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %13 : i64 to ^bb3
// CHECK-NEXT:   ^bb5:  // pred: ^bb3
// CHECK-NEXT:     %14 = "neura.add"(%5, %0) : (index, index) -> index
// CHECK-NEXT:     %15 = "neura.cast"(%14) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %15 : i64 to ^bb1
// CHECK-NEXT:   ^bb6:  // pred: ^bb1
// CHECK-NEXT:     "neura.return"() : () -> ()
// CHECK-NEXT:   }
// CAST:     func.func @_Z10bert_node1PA1_A1_A1_A1_A128_bPA1_A128_S1_(%arg0: memref<?x1x1x1x1x128xi8>, %arg1: memref<?x1x128x1x1x128xi8>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CAST-NEXT:     %0 = "neura.constant"() <{value = 1 : i64}> : () -> i64
// CAST-NEXT:     %1 = "neura.constant"() <{value = 128 : i64}> : () -> i64
// CAST-NEXT:     %2 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CAST-NEXT:     neura.br %2 : i64 to ^bb1
// CAST-NEXT:   ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb5
// CAST-NEXT:     %4 = "neura.icmp"(%3, %1) <{cmpType = "slt"}> : (i64, i64) -> i1
// CAST-NEXT:     neura.cond_br %4 : i1 then to ^bb2 else to ^bb6
// CAST-NEXT:   ^bb2:  // pred: ^bb1
// CAST-NEXT:     neura.br %2 : i64 to ^bb3
// CAST-NEXT:   ^bb3(%5: i64):  // 2 preds: ^bb2, ^bb4
// CAST-NEXT:     %6 = "neura.icmp"(%5, %1) <{cmpType = "slt"}> : (i64, i64) -> i1
// CAST-NEXT:     neura.cond_br %6 : i1 then to ^bb4 else to ^bb5
// CAST-NEXT:   ^bb4:  // pred: ^bb3
// CAST-NEXT:     %7 = neura.load_indexed %arg0[%2, %2, %2, %2, %2, %5 : i64, i64, i64, i64, i64, i64] memref<?x1x1x1x1x128xi8> : i8
// CAST-NEXT:     neura.store_indexed %7 to %arg1[%2, %2, %3, %2, %2, %5 : i64, i64, i64, i64, i64, i64] memref<?x1x128x1x1x128xi8> : i8
// CAST-NEXT:     %8 = "neura.add"(%5, %0) : (i64, i64) -> i64
// CAST-NEXT:     neura.br %8 : i64 to ^bb3
// CAST-NEXT:   ^bb5:  // pred: ^bb3
// CAST-NEXT:     %9 = "neura.add"(%3, %0) : (i64, i64) -> i64
// CAST-NEXT:     neura.br %9 : i64 to ^bb1
// CAST-NEXT:   ^bb6:  // pred: ^bb1
// CAST-NEXT:     "neura.return"() : () -> ()
// CAST-NEXT:   }
// CTRL2DATA:        func.func @_Z10bert_node1PA1_A1_A1_A1_A128_bPA1_A128_S1_(%arg0: memref<?x1x1x1x1x128xi8>, %arg1: memref<?x1x128x1x1x128xi8>) attributes {accelerator = "neura", dataflow_mode = "predicate", llvm.linkage = #llvm.linkage<external>} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = "%arg0"}> : () -> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<memref<?x1x1x1x1x128xi8>, i1>) -> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = "%arg1"}> : () -> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<memref<?x1x128x1x1x128xi8>, i1>) -> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{value = 128 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %10 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %11 = "neura.phi"(%10, %5) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %12 = neura.reserve : !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %13 = "neura.phi"(%12, %3) : (!neura.data<memref<?x1x128x1x1x128xi8>, i1>, !neura.data<memref<?x1x128x1x1x128xi8>, i1>) -> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %14 = neura.reserve : !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %15 = "neura.phi"(%14, %1) : (!neura.data<memref<?x1x1x1x1x128xi8>, i1>, !neura.data<memref<?x1x1x1x1x128xi8>, i1>) -> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %16 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %17 = "neura.phi"(%16, %9) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %18 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %19 = "neura.phi"(%18, %7) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %20 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %21 = "neura.phi"(%20, %9) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %22 = "neura.icmp"(%21, %19) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %23 = neura.grant_predicate %17, %22 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %24 = neura.grant_predicate %19, %22 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %25 = neura.grant_predicate %15, %22 : !neura.data<memref<?x1x1x1x1x128xi8>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %26 = neura.grant_predicate %13, %22 : !neura.data<memref<?x1x128x1x1x128xi8>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %27 = neura.grant_predicate %21, %22 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %28 = neura.grant_predicate %11, %22 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %29 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %30 = "neura.phi"(%29, %28) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %31 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %32 = "neura.phi"(%31, %27) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %33 = neura.reserve : !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %34 = "neura.phi"(%33, %26) : (!neura.data<memref<?x1x128x1x1x128xi8>, i1>, !neura.data<memref<?x1x128x1x1x128xi8>, i1>) -> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %35 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %36 = "neura.phi"(%35, %23) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %37 = neura.reserve : !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %38 = "neura.phi"(%37, %25) : (!neura.data<memref<?x1x1x1x1x128xi8>, i1>, !neura.data<memref<?x1x1x1x1x128xi8>, i1>) -> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %39 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %40 = "neura.phi"(%39, %24) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %41 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %42 = "neura.phi"(%41, %23) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %43 = "neura.icmp"(%42, %40) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %44 = neura.grant_predicate %38, %43 : !neura.data<memref<?x1x1x1x1x128xi8>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %45 = neura.grant_predicate %36, %43 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %46 = neura.grant_predicate %42, %43 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %47 = neura.grant_predicate %34, %43 : !neura.data<memref<?x1x128x1x1x128xi8>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %48 = neura.grant_predicate %32, %43 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %49 = neura.grant_predicate %30, %43 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %50 = neura.grant_predicate %40, %43 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %51 = "neura.not"(%43) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %52 = neura.grant_predicate %32, %51 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %53 = neura.grant_predicate %30, %51 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %54 = neura.grant_predicate %40, %51 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %55 = neura.grant_predicate %36, %51 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %56 = neura.grant_predicate %38, %51 : !neura.data<memref<?x1x1x1x1x128xi8>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %57 = neura.grant_predicate %34, %51 : !neura.data<memref<?x1x128x1x1x128xi8>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %58 = "neura.add"(%52, %53) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %58 -> %20 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %54 -> %18 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %55 -> %16 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %56 -> %14 : !neura.data<memref<?x1x1x1x1x128xi8>, i1> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %57 -> %12 : !neura.data<memref<?x1x128x1x1x128xi8>, i1> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %53 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %59 = neura.load_indexed %44[%45, %45, %45, %45, %45, %46 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x1x1x1x1x128xi8>, i1> : !neura.data<i8, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %59 to %47[%45, %45, %48, %45, %45, %46 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x1x128x1x1x128xi8>, i1> : !neura.data<i8, i1>
// CTRL2DATA-NEXT:     %60 = "neura.add"(%46, %49) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %60 -> %41 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %50 -> %39 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %44 -> %37 : !neura.data<memref<?x1x1x1x1x128xi8>, i1> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %45 -> %35 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %47 -> %33 : !neura.data<memref<?x1x128x1x1x128xi8>, i1> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %48 -> %31 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %49 -> %29 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     "neura.return"() : () -> ()
// CTRL2DATA-NEXT:   }

// MAPPING:      func.func @_Z10bert_node1PA1_A1_A1_A1_A128_bPA1_A128_S1_(%arg0: memref<?x1x1x1x1x128xi8>, %arg1: memref<?x1x128x1x1x128xi8>) attributes {accelerator = "neura", dataflow_mode = "predicate", llvm.linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 10 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 8 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {