// Compiles the original C kernel to mlir, then lowers it via Neura.
// RUN: clang++ -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/fir/fir_int.cpp
// RUN: llvm-extract --rfunc=".*kernel.*" %t-kernel-full.ll -o %t-kernel-only.ll
// RUN: mlir-translate --import-llvm %t-kernel-only.ll -o %t-kernel.mlir

// RUN: neura-compiler %t-kernel.mlir --neura-conversion --architecture-spec=%S/../../arch_spec/architecture.yaml -o %t-mapping.mlir
// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM

// MAPPING:      func.func
// MAPPING-SAME:   compiled_ii = 5 : i32
// MAPPING-SAME:   mapping_mode = "spatial-temporal"
// MAPPING-SAME:   mapping_strategy = "heuristic"
// MAPPING-SAME:   rec_mii = 5 : i32
// MAPPING-SAME:   res_mii = 1 : i32
// MAPPING-SAME:   x_tiles = 4 : i32
// MAPPING-SAME:   y_tiles = 4 : i32
//
// MAPPING-NEXT:   %0 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 0 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:   %1 = "neura.grant_once"() <{constant_value = 0 : i32}> {dfg_id = 1 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 1 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:   %2 = neura.reserve {dfg_id = 2 : i32} : !neura.data<i32, i1>
// MAPPING-NEXT:   %3 = "neura.data_mov"(%1) {dfg_id = 6 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:   %4 = neura.phi_start %3, %2 {dfg_id = 8 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:   %5 = neura.reserve {dfg_id = 3 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:   %6 = "neura.data_mov"(%0) {dfg_id = 5 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %7 = neura.phi_start %6, %5 {dfg_id = 7 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:   %8 = "neura.data_mov"(%7) {dfg_id = 11 : i32, mapping_locs = [{id = 34 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %9 = neura.load_indexed [%8 : !neura.data<i64, i1>]  {dfg_id = 15 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 3 : i32}]} : !neura.data<i32, i1>
// MAPPING-NEXT:   %10 = "neura.data_mov"(%7) {dfg_id = 10 : i32, mapping_locs = [{id = 31 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %11 = neura.load_indexed [%10 : !neura.data<i64, i1>]  {dfg_id = 14 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i32, i1>
// MAPPING-NEXT:   %12 = "neura.data_mov"(%11) {dfg_id = 18 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:   %13 = "neura.data_mov"(%9) {dfg_id = 19 : i32, mapping_locs = [{id = 45 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:   %14 = "neura.data_mov"(%4) {dfg_id = 12 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:   %15 = "neura.mul_add"(%12, %13, %14) {dfg_id = 21 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:   %16 = "neura.data_mov"(%7) {dfg_id = 9 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %17 = "neura.add"(%16) {dfg_id = 13 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 2 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %18 = "neura.data_mov"(%17) {dfg_id = 17 : i32, mapping_locs = [{id = 31 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %19 = "neura.icmp"(%18) <{cmpType = "eq"}> {dfg_id = 20 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 2 : i32}], rhs_value = 32 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %20 = "neura.data_mov"(%19) {dfg_id = 23 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %21 = "neura.not"(%20) {dfg_id = 26 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %22 = "neura.data_mov"(%17) {dfg_id = 16 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 31 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 288 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %23 = "neura.data_mov"(%21) {dfg_id = 29 : i32, mapping_locs = [{id = 289 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %24 = neura.grant_predicate %22, %23 {dfg_id = 32 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:   neura.ctrl_mov %24 -> %5 {dfg_id = 35 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:   %25 = "neura.data_mov"(%15) {dfg_id = 25 : i32, mapping_locs = [{id = 321 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 3 : i32}, {id = 321 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:   %26 = "neura.data_mov"(%21) {dfg_id = 28 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %27 = neura.grant_predicate %25, %26 {dfg_id = 31 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:   neura.ctrl_mov %27 -> %2 {dfg_id = 34 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}, {id = 193 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}]} : !neura.data<i32, i1> !neura.data<i32, i1>
// MAPPING-NEXT:   %28 = "neura.data_mov"(%15) {dfg_id = 24 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:   %29 = "neura.data_mov"(%19) {dfg_id = 22 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %30 = neura.grant_predicate %28, %29 {dfg_id = 27 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:   %31 = "neura.data_mov"(%30) {dfg_id = 30 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:   neura.return_value %31 : !neura.data<i32, i1> {dfg_id = 33 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 1 : i32}]}
// MAPPING-NEXT:   neura.yield {dfg_id = 4 : i32}


// ASM: # Compiled II: 5

// ASM: PE(2,1):
// ASM-NEXT: {
// ASM-NEXT:   RETURN_VALUE, [NORTH, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#0] -> [$0] (t=1, inv_iters=0)
// ASM-NEXT:   CTRL_MOV, [NORTH, RED] -> [$1] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$1] -> [NORTH, RED] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)

// ASM: PE(1,2):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$1] -> [EAST, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   LOAD_INDEXED, [EAST, RED] -> [EAST, RED] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_EQ, [EAST, RED], [#32] -> [$0], [EAST, RED] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   NOT, [$0] -> [$1], [EAST, RED] (t=4, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$0] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)

// ASM: PE(2,2):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$1], [WEST, RED] -> [SOUTH, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [EAST, RED], [WEST, RED] -> [NORTH, RED], [WEST, RED], [$0] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   ADD, [$0], [#1] -> [WEST, RED], [$0] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   MUL_ADD, [WEST, RED], [NORTH, RED], [SOUTH, RED] -> [$1], [$0] (t=3, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [$0] -> [WEST, RED] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [WEST, RED] -> [SOUTH, RED] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)

// ASM: PE(3,2):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#0] -> [WEST, RED] (t=0, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)

// ASM: PE(2,3):
// ASM-NEXT: {
// ASM-NEXT:   LOAD_INDEXED, [SOUTH, RED] -> [SOUTH, RED] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)



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
// YAML-NEXT:                 - opcode: "RETURN_VALUE"
// YAML-NEXT:                   id: 33
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 1
// YAML-NEXT:                   time_step: 1
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 34
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 8
// YAML-NEXT:                   time_step: 2
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "9"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 32
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "LOAD_INDEXED"
// YAML-NEXT:                   id: 14
// YAML-NEXT:                   time_step: 2
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ICMP_EQ"
// YAML-NEXT:                   id: 20
// YAML-NEXT:                   time_step: 3
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "#32"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "NOT"
// YAML-NEXT:                   id: 26
// YAML-NEXT:                   time_step: 4
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 16
// YAML-NEXT:                   time_step: 4
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "10"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 31
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 7
// YAML-NEXT:                   time_step: 1
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ADD"
// YAML-NEXT:                   id: 13
// YAML-NEXT:                   time_step: 2
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "#1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "MUL_ADD"
// YAML-NEXT:                   id: 21
// YAML-NEXT:                   time_step: 3
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 160000
// YAML-NEXT:                   time_step: 3
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 27
// YAML-NEXT:                   time_step: 4
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 3
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "11"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 0
// YAML-NEXT:                   time_step: 0
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 3
// YAML-NEXT:       core_id: "14"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "LOAD_INDEXED"
// YAML-NEXT:                   id: 15
// YAML-NEXT:                   time_step: 2
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                      color: "RED"
