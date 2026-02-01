// Compile the int ReLU C++ kernel to LLVM IR.
// RUN: clang++ -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -std=c++17 \
// RUN:   -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/relu/relu_int.cpp
//
// Extract only the kernel function(s) from relu_int.cpp. The regex keeps this
// robust across possible kernel name variants (e.g., names containing "kernel").
// RUN: llvm-extract --rfunc=".*kernel.*" %t-kernel-full.ll -o %t-kernel-only.ll
//
// Import the LLVM IR into MLIR (LLVM dialect).
// RUN: mlir-translate --import-llvm %t-kernel-only.ll -o %t-kernel.mlir
//
// RUN: mlir-neura-opt %t-kernel.mlir --view-op-graph 2>&1 | sed -n '/^digraph G {/,/^}$/p' > relu_kernel_original.dot
// RUN: dot -Tpng relu_kernel_original.dot -o relu_kernel_original.png
// RUN: dot -Tjson relu_kernel_original.dot -o relu_kernel_original.json
// RUN: mlir-neura-opt %t-kernel.mlir \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-input-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --architecture-spec=%S/../../arch_spec/architecture.yaml \
// RUN:   --generate-code -o %t-mapping.mlir
// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM
//
// Check the mapped MLIR contains key operations with full statements.
// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
// MAPPING:          %0 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 0 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING:          %1 = "neura.grant_once"() <{constant_value = 0 : i32}> {dfg_id = 1 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 3 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING:          %2 = neura.reserve {dfg_id = 2 : i32} : !neura.data<i32, i1>
// MAPPING:          %3 = "neura.data_mov"(%1) {dfg_id = 6 : i32, mapping_locs = [{id = 39 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING:          %4 = neura.phi_start %3, %2 {dfg_id = 8 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// MAPPING:          %5 = neura.reserve {dfg_id = 3 : i32} : !neura.data<i64, i1>
// MAPPING:          %6 = "neura.data_mov"(%0) {dfg_id = 5 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING:          %7 = neura.phi_start %6, %5 {dfg_id = 7 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING:          %8 = "neura.data_mov"(%7) {dfg_id = 11 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING:          %9 = "neura.gep"(%8) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 16 : i32, lhs_value = "%arg0", mapping_locs = [{id = 11 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING:          %10 = "neura.data_mov"(%9) {dfg_id = 20 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING:          %11 = "neura.load"(%10) {dfg_id = 22 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING:          %12 = "neura.data_mov"(%11) {dfg_id = 27 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING:          %13 = "neura.icmp"(%12) <{cmpType = "sge"}> {dfg_id = 30 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 3 : i32, y = 2 : i32}], rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING:          %14 = "neura.data_mov"(%13) {dfg_id = 34 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 31 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING:          %15 = "neura.data_mov"(%11) {dfg_id = 26 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 31 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 288 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING:          %16 = "neura.data_mov"(%4) {dfg_id = 13 : i32, mapping_locs = [{id = 24 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 289 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING:          %17 = "neura.sel"(%14, %15, %16) {dfg_id = 38 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING:          %18 = "neura.data_mov"(%7) {dfg_id = 10 : i32, mapping_locs = [{id = 34 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}, {id = 448 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 448 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}, {id = 448 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING:          %19 = "neura.gep"(%18) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 15 : i32, lhs_value = "%arg1", mapping_locs = [{id = 14 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING:          %20 = "neura.data_mov"(%17) {dfg_id = 41 : i32, mapping_locs = [{id = 30 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING:          %21 = "neura.data_mov"(%19) {dfg_id = 19 : i32, mapping_locs = [{id = 448 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}, {id = 43 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING:          "neura.store"(%20, %21) {dfg_id = 42 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING:          %22 = "neura.data_mov"(%7) {dfg_id = 9 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING:          %23 = "neura.add"(%22) {dfg_id = 14 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 2 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING:          %24 = "neura.data_mov"(%23) {dfg_id = 18 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING:          %25 = "neura.icmp"(%24) <{cmpType = "eq"}> {dfg_id = 21 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}], rhs_value = 32 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING:          %26 = "neura.data_mov"(%25) {dfg_id = 25 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING:          %27 = "neura.not"(%26) {dfg_id = 29 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING:          %28 = "neura.data_mov"(%23) {dfg_id = 17 : i32, mapping_locs = [{id = 321 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 2 : i32}, {id = 321 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 3 : i32}, {id = 321 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING:          %29 = "neura.data_mov"(%27) {dfg_id = 33 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING:          %30 = neura.grant_predicate %28, %29 {dfg_id = 37 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING:          neura.ctrl_mov %30 -> %5 {dfg_id = 40 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING:          %31 = "neura.data_mov"(%4) {dfg_id = 12 : i32, mapping_locs = [{id = 256 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 24 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}, {id = 288 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING:          %32 = "neura.data_mov"(%27) {dfg_id = 32 : i32, mapping_locs = [{id = 34 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 43 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}, {id = 42 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING:          %33 = neura.grant_predicate %31, %32 {dfg_id = 36 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING:          neura.ctrl_mov %33 -> %2 {dfg_id = 39 : i32, mapping_locs = [{id = 27 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 7 : i32}, {id = 256 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : !neura.data<i32, i1> !neura.data<i32, i1>
// MAPPING:          %34 = "neura.data_mov"(%25) {dfg_id = 23 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 192 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING:          %35 = "neura.data_mov"(%25) {dfg_id = 24 : i32, mapping_locs = [{id = 322 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 3 : i32}, {id = 33 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING:          %36 = neura.grant_predicate %34, %35 {dfg_id = 28 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
// MAPPING:          %37 = "neura.data_mov"(%36) {dfg_id = 31 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING:          neura.return_void %37 : !neura.data<i1, i1> {dfg_id = 35 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 1 : i32}]}
// MAPPING:          neura.yield {dfg_id = 4 : i32}
// MAPPING:        }
// MAPPING:      }

// YAML: array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   compiled_ii: 5
// YAML-NEXT:   cores:
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "6"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 28
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "RETURN_VOID"
// YAML-NEXT:                   id: 35
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 23
// YAML-NEXT:                   time_step: 4
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"

// ASM: # Compiled II: 5
// ASM: PE(2,1):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [NORTH, RED] -> [$0] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   RETURN_VOID, [$0] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$0] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
