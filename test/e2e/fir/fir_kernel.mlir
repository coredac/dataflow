// Compiles the original C kernel to mlir, then lowers it via Neura.
// RUN: clang++ -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/fir/fir_int.cpp
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
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --architecture-spec=../../arch_spec/architecture.yaml \
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
// MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 0 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %1 = "neura.grant_once"() <{constant_value = 0 : i32}> {dfg_id = 1 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 1 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %2 = neura.reserve {dfg_id = 2 : i32} : !neura.data<i32, i1>
// MAPPING-NEXT:     %3 = "neura.data_mov"(%1) {dfg_id = 5 : i32, mapping_locs = [{id = 128 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %4 = "neura.phi"(%2, %3) {dfg_id = 7 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %5 = neura.reserve {dfg_id = 3 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %6 = "neura.data_mov"(%0) {dfg_id = 4 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %7 = "neura.phi"(%5, %6) {dfg_id = 6 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 3 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %8 = "neura.data_mov"(%7) {dfg_id = 10 : i32, mapping_locs = [{id = 37 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %9 = "neura.gep"(%8) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 14 : i32, lhs_value = "%arg0", mapping_locs = [{id = 15 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %10 = "neura.data_mov"(%9) {dfg_id = 18 : i32, mapping_locs = [{id = 480 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %11 = "neura.load"(%10) {dfg_id = 21 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 3 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %12 = "neura.data_mov"(%7) {dfg_id = 9 : i32, mapping_locs = [{id = 36 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %13 = "neura.gep"(%12) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 13 : i32, lhs_value = "%arg2", mapping_locs = [{id = 7 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %14 = "neura.data_mov"(%13) {dfg_id = 17 : i32, mapping_locs = [{id = 21 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %15 = "neura.load"(%14) {dfg_id = 20 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %16 = "neura.data_mov"(%15) {dfg_id = 24 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 34 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %17 = "neura.data_mov"(%11) {dfg_id = 25 : i32, mapping_locs = [{id = 46 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 448 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %18 = "neura.mul"(%16, %17) {dfg_id = 27 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %19 = "neura.data_mov"(%18) {dfg_id = 30 : i32, mapping_locs = [{id = 43 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}, {id = 42 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %20 = "neura.data_mov"(%4) {dfg_id = 11 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 16 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}, {id = 288 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %21 = "neura.add"(%19, %20) {dfg_id = 32 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %22 = "neura.data_mov"(%7) {dfg_id = 8 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %23 = "neura.add"(%22) {dfg_id = 12 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 2 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %24 = "neura.data_mov"(%23) {dfg_id = 16 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %25 = "neura.icmp"(%24) <{cmpType = "eq"}> {dfg_id = 19 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}], rhs_value = 32 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %26 = "neura.data_mov"(%25) {dfg_id = 23 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %27 = "neura.not"(%26) {dfg_id = 26 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %28 = "neura.data_mov"(%23) {dfg_id = 15 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 35 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 320 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %29 = "neura.data_mov"(%27) {dfg_id = 29 : i32, mapping_locs = [{id = 321 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %30 = neura.grant_predicate %28, %29 {dfg_id = 31 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %30 -> %5 {dfg_id = 33 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %31 = "neura.data_mov"(%21) {dfg_id = 35 : i32, mapping_locs = [{id = 27 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %32 = "neura.data_mov"(%27) {dfg_id = 28 : i32, mapping_locs = [{id = 31 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 27 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}, {id = 256 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 256 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %33 = neura.grant_predicate %31, %32 {dfg_id = 37 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 8 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %33 -> %2 {dfg_id = 39 : i32, mapping_locs = [{id = 25 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 8 : i32}]} : !neura.data<i32, i1> !neura.data<i32, i1>
// MAPPING-NEXT:     %34 = "neura.data_mov"(%21) {dfg_id = 34 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %35 = "neura.data_mov"(%25) {dfg_id = 22 : i32, mapping_locs = [{id = 31 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 29 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 160 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}, {id = 160 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 160 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %36 = neura.grant_predicate %34, %35 {dfg_id = 36 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %37 = "neura.data_mov"(%36) {dfg_id = 38 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     "neura.return"(%37) {dfg_id = 40 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 9 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>) -> ()

// YAML: instructions:
// YAML: - opcode: "GRANT_ONCE"
// YAML: - opcode: "RETURN"
// YAML: - opcode: "ICMP_EQ"

// ASM:      PE(0,1):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#0] -> [$0]
// ASM-NEXT: } (t=3)
// ASM:      PE(2,2):
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$0]
// ASM-NEXT: } (t=2)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_EQ, [EAST, RED], [#32] -> [$0], [WEST, RED]
// ASM-NEXT: } (t=3)

// RUN: mlir-neura-opt %t-kernel.mlir \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --view-op-graph 2>&1 | sed -n '/^digraph G {/,/^}$/p' > fir_kernel.dot
// RUN: dot -Tpng fir_kernel.dot -o fir_kernel.png
// RUN: dot -Tjson fir_kernel.dot -o fir_kernel.json
// RUN: FileCheck %s --input-file=fir_kernel.dot -check-prefix=DOT

// DOT: digraph G {
