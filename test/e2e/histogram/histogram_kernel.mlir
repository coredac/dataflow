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
// MAPPING-SAME:   compiled_ii = 5
// MAPPING-SAME:   mapping_mode = "spatial-temporal"
// MAPPING-SAME:   mapping_strategy = "heuristic"
// MAPPING-SAME:   rec_mii = 5
// MAPPING-SAME:   res_mii = 2
// MAPPING-SAME:   x_tiles = 4
// MAPPING-SAME:   y_tiles = 4
//
// MAPPING-NEXT: %0 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 0 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT: %1 = neura.reserve {dfg_id = 1 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT: %2 = "neura.data_mov"(%0) {dfg_id = 3 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %3 = "neura.phi"(%1, %2) {dfg_id = 4 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 3 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %4 = "neura.data_mov"(%3) {dfg_id = 6 : i32, mapping_locs = [{id = 36 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %5 = "neura.gep"(%4) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 8 : i32, lhs_value = "%arg0", mapping_locs = [{id = 7 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %6 = "neura.data_mov"(%5) {dfg_id = 11 : i32, mapping_locs = [{id = 23 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %7 = "neura.load"(%6) {dfg_id = 13 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT: %8 = "neura.data_mov"(%7) {dfg_id = 15 : i32, mapping_locs = [{id = 37 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT: %9 = "neura.fadd"(%8) {dfg_id = 17 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 3 : i32, y = 3 : i32}], rhs_value = -1.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT: %10 = "neura.data_mov"(%9) {dfg_id = 19 : i32, mapping_locs = [{id = 46 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT: %11 = "neura.fmul"(%10) {dfg_id = 21 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 3 : i32}], rhs_value = 5.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT: %12 = "neura.data_mov"(%11) {dfg_id = 23 : i32, mapping_locs = [{id = 45 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT: %13 = "neura.fdiv"(%12) {dfg_id = 24 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 2 : i32}], rhs_value = 1.800000e+01 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT: %14 = "neura.data_mov"(%13) {dfg_id = 25 : i32, mapping_locs = [{id = 31 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT: %15 = "neura.cast"(%14) <{cast_type = "fptosi"}> {dfg_id = 26 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %16 = "neura.data_mov"(%15) {dfg_id = 27 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %17 = neura.sext %16 {dfg_id = 28 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i32, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: %18 = "neura.data_mov"(%17) {dfg_id = 29 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %19 = "neura.gep"(%18) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 30 : i32, lhs_value = "%arg1", mapping_locs = [{id = 5 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %20 = "neura.data_mov"(%19) {dfg_id = 32 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %21 = "neura.load"(%20) {dfg_id = 33 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 10 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %22 = "neura.data_mov"(%21) {dfg_id = 34 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %23 = "neura.add"(%22) {dfg_id = 35 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 1 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 11 : i32, x = 1 : i32, y = 2 : i32}], rhs_value = 1 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %24 = "neura.data_mov"(%23) {dfg_id = 36 : i32, mapping_locs = [{id = 30 : i32, index_per_ii = 1 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 11 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %25 = "neura.data_mov"(%19) {dfg_id = 31 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 9 : i32}, {id = 30 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 10 : i32}, {id = 416 : i32, index_per_ii = 1 : i32, invalid_iterations = 2 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: "neura.store"(%24, %25) {dfg_id = 37 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 2 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 12 : i32, x = 1 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT: %26 = "neura.data_mov"(%3) {dfg_id = 5 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %27 = "neura.add"(%26) {dfg_id = 7 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 2 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %28 = "neura.data_mov"(%27) {dfg_id = 10 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %29 = "neura.icmp"(%28) <{cmpType = "eq"}> {dfg_id = 12 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}], rhs_value = 20 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %30 = "neura.data_mov"(%29) {dfg_id = 14 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %31 = "neura.not"(%30) {dfg_id = 16 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %32 = "neura.data_mov"(%27) {dfg_id = 9 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 35 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 320 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %33 = "neura.data_mov"(%31) {dfg_id = 18 : i32, mapping_locs = [{id = 321 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %34 = neura.grant_predicate %32, %33 {dfg_id = 20 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: neura.ctrl_mov %34 -> %1 {dfg_id = 22 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT: "neura.return"() {dfg_id = 2 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 2 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 12 : i32, x = 0 : i32, y = 3 : i32}]} : () -> ()

// YAML:      compiled_ii: 5
// YAML:      cores:
// YAML:        - column: 1
// YAML:          row: 1
// YAML:          entries:
// YAML:            - entry_id: "entry0"
// YAML:              instructions:
// YAML:                - index_per_ii: 0
// YAML:                  operations:
// YAML:                    - opcode: "LOAD"

// ASM:      PE(3,2):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#0] -> [$0] (t=0, inv_iter=0)
// ASM-NEXT: } (idx_per_ii=0)

// RUN: mlir-neura-opt %t-kernel.mlir \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --view-op-graph 2>&1 | sed -n '/^digraph G {/,/^}$/p' > histogram_kernel.dot
// RUN: dot -Tpng histogram_kernel.dot -o histogram_kernel.png
// RUN: FileCheck %s --input-file=histogram_kernel.dot -check-prefix=DOT

// DOT: digraph G {