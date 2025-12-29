// Compiles the original C kernel to mlir, then lowers it via Neura.
// TODO: Got error when using -O3 -fno-vectorize -fno-slp-vectorize -mllvm -force-vector-width=1 
// Issue: https://github.com/coredac/dataflow/issues/164
// RUN: clang++ -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/histogram/histogram_int.cpp
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


// MAPPING:      func.func @_Z6kernelPiS_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}
// MAPPING-SAME: compiled_ii = 5
// MAPPING-SAME: mapping_mode = "spatial-temporal"
// MAPPING-SAME: mapping_strategy = "heuristic"
// MAPPING-SAME: rec_mii = 5
// MAPPING-SAME: res_mii = 2
// MAPPING-SAME: x_tiles = 4
// MAPPING-SAME: y_tiles = 4
// MAPPING-NEXT:    %0 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 0 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:    %1 = neura.reserve {dfg_id = 1 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:    %2 = "neura.data_mov"(%0) {dfg_id = 3 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>

// YAML:      array_config:
// YAML-NEXT:     columns: 4
// YAML-NEXT:     rows: 4
// YAML-NEXT:     compiled_ii: 5
// YAML-NEXT:     cores:
// YAML-NEXT:     - column: 2
// YAML-NEXT:     row: 0
// YAML-NEXT:     core_id: "2"
// YAML-NEXT:     entries:
// YAML-NEXT:     - entry_id: "entry0"
// YAML-NEXT:     instructions:
// YAML-NEXT:     - index_per_ii: 1
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "RETURN"
// YAML-NEXT:     id: 2
// YAML-NEXT:     time_step: 11
// YAML-NEXT:     invalid_iterations: 2
// YAML-NEXT:     - column: 2
// YAML-NEXT:     row: 1
// YAML-NEXT:     core_id: "6"
// YAML-NEXT:     entries:
// YAML-NEXT:     - entry_id: "entry0"
// YAML-NEXT:     instructions:
// YAML-NEXT:     - index_per_ii: 0
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "ADD"
// YAML-NEXT:     id: 33
// YAML-NEXT:     time_step: 10
// YAML-NEXT:     invalid_iterations: 2
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "#1"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 1
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "STORE"
// YAML-NEXT:     id: 35
// YAML-NEXT:     time_step: 11
// YAML-NEXT:     invalid_iterations: 2
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "$1"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 3
// YAML-NEXT:     operations:

// ASM:      # Compiled II: 5
// ASM:     PE(2,0):
// ASM-NEXT:     {
// ASM-NEXT:     RETURN (t=11, inv_iters=2)
// ASM-NEXT:     } (idx_per_ii=1)
// ASM:     PE(2,1):
// ASM-NEXT:     {
// ASM-NEXT:     ADD, [$0], [#1] -> [$0] (t=10, inv_iters=2)
// ASM-NEXT:     } (idx_per_ii=0)
// ASM-NEXT:     {
// ASM-NEXT:     STORE, [$0], [$1] (t=11, inv_iters=2)
// ASM-NEXT:     } (idx_per_ii=1)
// ASM-NEXT:     {
// ASM-NEXT:     DATA_MOV, [EAST, RED] -> [$1] (t=8, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=3)
// ASM-NEXT:     {
// ASM-NEXT:     LOAD, [EAST, RED] -> [$0] (t=9, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=4)
// ASM:     PE(3,1):
// ASM-NEXT:     {
// ASM-NEXT:     ADD, [NORTH, RED], [#-5] -> [$0] (t=5, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=0)
// ASM-NEXT:     {
// ASM-NEXT:     DIV, [$0], [#18] -> [$0] (t=6, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=1)
// ASM-NEXT:     {
// ASM-NEXT:     SEXT, [$0] -> [$0] (t=7, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=2)
// ASM-NEXT:     {
// ASM-NEXT:     GEP, [$0] -> [WEST, RED] (t=8, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=3)
// ASM:     PE(2,2):
// ASM-NEXT:     {
// ASM-NEXT:     GRANT_PREDICATE, [$1], [$0] -> [$0] (t=5, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=0)
// ASM-NEXT:     {
// ASM-NEXT:     PHI_START, [EAST, RED], [$0] -> [EAST, RED], [$0] (t=1, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=1)
// ASM-NEXT:     {
// ASM-NEXT:     ADD, [$0], [#1] -> [$0], [$1] (t=2, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=2)
// ASM-NEXT:     {
// ASM-NEXT:     ICMP_EQ, [$0], [#20] -> [$0] (t=3, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=3)
// ASM-NEXT:     {
// ASM-NEXT:     NOT, [$0] -> [$0] (t=4, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=4)
// ASM:     PE(3,2):
// ASM-NEXT:     {
// ASM-NEXT:     GRANT_ONCE, [#0] -> [WEST, RED] (t=0, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=0)
// ASM-NEXT:     {
// ASM-NEXT:     GEP, [WEST, RED] -> [$0] (t=2, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=2)
// ASM-NEXT:     {
// ASM-NEXT:     LOAD, [$0] -> [$0] (t=3, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=3)
// ASM-NEXT:     {
// ASM-NEXT:     MUL, [$0], [#5] -> [SOUTH, RED] (t=4, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=4)

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
// RUN: dot -Tjson histogram_kernel.dot -o histogram_kernel.json
// RUN: FileCheck %s --input-file=histogram_kernel.dot -check-prefix=DOT

// DOT: digraph G {
