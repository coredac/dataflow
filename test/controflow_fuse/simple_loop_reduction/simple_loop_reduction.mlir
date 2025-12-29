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
// RUN: --canonicalize-live-in \
// RUN: | FileCheck %s --check-prefix=CANONICALIZE

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

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --promote-func-arg-to-const \
// RUN: --fold-constant \
// RUN: --canonicalize-return \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: --fold-constant \
// RUN: --fuse-loop-control \
// RUN: --fold-constant \
// RUN: | FileCheck %s -check-prefix=FUSE

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --promote-func-arg-to-const \
// RUN: --fold-constant \
// RUN: --canonicalize-return \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: --fold-constant \
// RUN: --fuse-loop-control \
// RUN: --fold-constant \
// RUN: --insert-data-mov \
// RUN: --map-to-accelerator="mapping-strategy=heuristic backtrack-config=customized" \
// RUN: --architecture-spec=../../arch_spec/architecture.yaml | FileCheck %s -check-prefix=FUSE-MAPPING

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

// CANONICALIZE:        func.func @_Z10simpleloopv() -> i32 attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CANONICALIZE-NEXT:     %0 = "neura.constant"() <{value = 1 : i64}> : () -> i64
// CANONICALIZE-NEXT:     %1 = "neura.constant"() <{value = 128 : i64}> : () -> i64
// CANONICALIZE-NEXT:     %2 = "neura.constant"() <{value = 0 : i32}> : () -> i32
// CANONICALIZE-NEXT:     %3 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CANONICALIZE-NEXT:     neura.br %3, %2, %1, %0 : i64, i32, i64, i64 to ^bb1
// CANONICALIZE-NEXT:   ^bb1(%4: i64, %5: i32, %6: i64, %7: i64):  // 2 preds: ^bb0, ^bb2
// CANONICALIZE-NEXT:     %8 = "neura.icmp"(%4, %6) <{cmpType = "slt"}> : (i64, i64) -> i1
// CANONICALIZE-NEXT:     neura.cond_br %8 : i1 then %4, %5, %7, %6 : i64, i32, i64, i64 to ^bb2 else %5 : i32 to ^bb3
// CANONICALIZE-NEXT:   ^bb2(%9: i64, %10: i32, %11: i64, %12: i64):  // pred: ^bb1
// CANONICALIZE-NEXT:     %13 = "neura.cast"(%9) <{cast_type = "i64_to_i32"}> : (i64) -> i32
// CANONICALIZE-NEXT:     %14 = "neura.add"(%10, %13) : (i32, i32) -> i32
// CANONICALIZE-NEXT:     %15 = "neura.add"(%9, %11) : (i64, i64) -> i64
// CANONICALIZE-NEXT:     neura.br %15, %14, %12, %11 : i64, i32, i64, i64 to ^bb1
// CANONICALIZE-NEXT:   ^bb3(%16: i32):  // pred: ^bb1
// CANONICALIZE-NEXT:     "neura.return"(%16) : (i32) -> ()
// CANONICALIZE-NEXT:   }

// CTRL2DATA:        func.func @_Z10simpleloopv() -> i32 attributes {accelerator = "neura", dataflow_mode = "predicate", llvm.linkage = #llvm.linkage<external>} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = 128 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{value = 0 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %8 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %9 = neura.phi_start %1, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %10 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %11 = neura.phi_start %3, %10 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %12 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %13 = neura.phi_start %5, %12 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %14 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %15 = neura.phi_start %7, %14 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %16 = "neura.icmp"(%15, %11) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %17 = neura.grant_predicate %15, %16 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %18 = neura.grant_predicate %13, %16 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %19 = neura.grant_predicate %9, %16 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %20 = neura.grant_predicate %11, %16 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %21 = "neura.not"(%16) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %22 = neura.grant_predicate %13, %21 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.return_value %22 : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %23 = "neura.cast"(%17) <{cast_type = "i64_to_i32"}> : (!neura.data<i64, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %24 = "neura.add"(%18, %23) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %25 = "neura.add"(%17, %19) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %25 -> %14 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %24 -> %12 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %20 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %19 -> %8 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.yield
// CTRL2DATA-NEXT:   }


// FUSE:        func.func @_Z10simpleloopv() -> i32 attributes {accelerator = "neura", dataflow_mode = "predicate", llvm.linkage = #llvm.linkage<external>} {
// FUSE-NEXT:     %0 = "neura.grant_once"() <{constant_value = 0 : i32}> : () -> !neura.data<i32, i1>
// FUSE-NEXT:     %1 = neura.reserve : !neura.data<i32, i1>
// FUSE-NEXT:     %2 = neura.phi_start %0, %1 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// FUSE-NEXT:     %3 = "neura.grant_always"() <{constant_value = true}> : () -> !neura.data<i1, i1>
// FUSE-NEXT:     %nextindex, %valid = "neura.loop_control"(%3) <{end = 128 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (!neura.data<i1, i1>) -> (!neura.data<i64, i1>, !neura.data<i1, i1>)
// FUSE-NEXT:     %4 = "neura.not"(%valid) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-NEXT:     %5 = neura.grant_predicate %2, %valid : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-NEXT:     %6 = neura.grant_predicate %2, %4 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// FUSE-NEXT:     neura.return_value %6 : !neura.data<i32, i1>
// FUSE-NEXT:     %7 = "neura.cast"(%nextindex) <{cast_type = "i64_to_i32"}> : (!neura.data<i64, i1>) -> !neura.data<i32, i1>
// FUSE-NEXT:     %8 = "neura.add"(%5, %7) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// FUSE-NEXT:     neura.ctrl_mov %8 -> %1 : !neura.data<i32, i1> !neura.data<i32, i1>
// FUSE-NEXT:     neura.yield
// FUSE-NEXT:   }


// FUSE-MAPPING:      func.func @_Z10simpleloopv() -> i32 attributes {accelerator = "neura", dataflow_mode = "predicate", llvm.linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 3 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 3 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}
