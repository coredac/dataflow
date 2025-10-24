// Compiles the original C kernel to mlir, then lowers it via Neura.
// TODO: Got error when using -O3 -fno-vectorize -fno-slp-vectorize -mllvm -force-vector-width=1 
// Issue: https://github.com/coredac/dataflow/issues/164
// RUN: clang++ -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/histogram/histogram.cpp
// RUN: llvm-extract --rfunc=".*kernel.*" %t-kernel-full.ll -o %t-kernel-only.ll
// RUN: mlir-translate --import-llvm %t-kernel-only.ll -o %t-kernel.mlir

// RUN: mlir-neura-opt %t-kernel.mlir \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic backtrack-config=customized" \
// RUN:   --architecture-spec=%S/../../arch_spec/architecture.yaml \
// RUN:   --generate-code -o %t-mapping.mlir 
// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM


// MAPPING: module
// MAPPING:      func.func
// MAPPING-SAME:   compiled_ii = 8
// MAPPING-SAME:   mapping_mode = "spatial-temporal"
// MAPPING-SAME:   mapping_strategy = "heuristic"
// MAPPING-SAME:   rec_mii = 5
// MAPPING-SAME:   res_mii = 2
// MAPPING-SAME:   x_tiles = 4
// MAPPING-SAME:   y_tiles = 4
//
// MAPPING-NEXT:  %0 = "neura.grant_once"() <{constant_value = 0 : i64}> {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:  %1 = "neura.grant_once"() <{constant_value = 1.800000e+01 : f32}> {mapping_locs = [{id = 11 : i32, resource = "tile", time_step = 4 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<f32, i1>
// MAPPING-NEXT:  %2 = neura.reserve : !neura.data<f32, i1>
// MAPPING-NEXT:  %3 = "neura.data_mov"(%1) {mapping_locs = [{id = 35 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:  %4 = "neura.phi"(%2, %3) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:  %5 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:  %6 = "neura.data_mov"(%0) {mapping_locs = [{id = 320 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:  %7 = "neura.phi"(%5, %6) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:  %8 = "neura.data_mov"(%7) {mapping_locs = [{id = 34 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:  %9 = "neura.gep"(%8) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg0", mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:  %10 = "neura.data_mov"(%9) {mapping_locs = [{id = 448 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:  %11 = "neura.load"(%10) {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:  %12 = "neura.data_mov"(%11) {mapping_locs = [{id = 448 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:  %13 = "neura.fadd"(%12) {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 3 : i32}], rhs_value = -1.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:  %14 = "neura.data_mov"(%13) {mapping_locs = [{id = 448 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:  %15 = "neura.fmul"(%14) {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 3 : i32}], rhs_value = 5.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:  %16 = "neura.data_mov"(%15) {mapping_locs = [{id = 448 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:  %17 = "neura.data_mov"(%4) {mapping_locs = [{id = 34 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:  %18 = "neura.fdiv"(%16, %17) {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:  %19 = "neura.data_mov"(%18) {mapping_locs = [{id = 448 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:  %20 = "neura.cast"(%19) <{cast_type = "fptosi"}> {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:  %21 = "neura.data_mov"(%20) {mapping_locs = [{id = 43 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:  %22 = neura.sext %21 {mapping_locs = [{id = 13 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 3 : i32}]} : !neura.data<i32, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:  %23 = "neura.data_mov"(%22) {mapping_locs = [{id = 40 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:  %24 = "neura.gep"(%23) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg1", mapping_locs = [{id = 12 : i32, resource = "tile", time_step = 9 : i32, x = 0 : i32, y = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:  %25 = "neura.data_mov"(%24) {mapping_locs = [{id = 39 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:  %26 = "neura.load"(%25) {mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 10 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:  %27 = "neura.data_mov"(%26) {mapping_locs = [{id = 24 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:  %28 = "neura.add"(%27) {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 11 : i32, x = 1 : i32, y = 2 : i32}], rhs_value = 1 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:  %29 = "neura.data_mov"(%28) {mapping_locs = [{id = 288 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:  %30 = "neura.data_mov"(%24) {mapping_locs = [{id = 38 : i32, resource = "link", time_step = 9 : i32}, {id = 42 : i32, resource = "link", time_step = 10 : i32}, {id = 289 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:  "neura.store"(%29, %30) {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 12 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT:  %31 = "neura.data_mov"(%7) {mapping_locs = [{id = 320 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:  %32 = "neura.add"(%31) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 2 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:  %33 = "neura.data_mov"(%32) {mapping_locs = [{id = 320 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:  %34 = "neura.icmp"(%33) <{cmpType = "eq"}> {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}], rhs_value = 20 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:  %35 = "neura.data_mov"(%34) {mapping_locs = [{id = 320 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:  %36 = "neura.not"(%35) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:  %37 = "neura.data_mov"(%32) {mapping_locs = [{id = 321 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 2 : i32}, {id = 321 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 3 : i32}, {id = 321 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}, {id = 321 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:  %38 = "neura.data_mov"(%36) {mapping_locs = [{id = 320 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 320 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:  %39 = neura.grant_predicate %37, %38 {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:  neura.ctrl_mov %39 -> %5 {mapping_locs = [{id = 321 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}, {id = 321 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}, {id = 321 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:  %40 = "neura.data_mov"(%4) {mapping_locs = [{id = 33 : i32, resource = "link", time_step = 5 : i32}, {id = 192 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 192 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 192 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 192 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}, {id = 192 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}, {id = 192 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:  %41 = "neura.data_mov"(%36) {mapping_locs = [{id = 33 : i32, resource = "link", time_step = 4 : i32}, {id = 193 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}, {id = 193 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}, {id = 193 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}, {id = 193 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}, {id = 193 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 193 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}, {id = 193 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:  %42 = neura.grant_predicate %40, %41 {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 12 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:  neura.ctrl_mov %42 -> %2 {mapping_locs = [{id = 20 : i32, resource = "link", time_step = 12 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
// MAPPING-NEXT:  "neura.return"() {mapping_locs = [{id = 15 : i32, resource = "tile", time_step = 12 : i32, x = 3 : i32, y = 3 : i32}]} : () -> ()

// YAML: instructions:
// YAML: - opcode: "GRANT_ONCE"
// YAML: - opcode: "FDIV"
// YAML: - opcode: "CAST"

// ASM:      PE(2,2):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#0] -> [$0]
// ASM-NEXT: } (t=0)
