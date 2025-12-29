// Compile the C kernel to LLVM IR (let clang handle headers and macros).
// Use -I %S so local headers (relu.h, polybench.h) are visible.
// RUN: clang -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -std=c11 \
// RUN:   -I %S/../../benchmark/CGRA-Bench/kernels/relu -DSMALL_DATASET \
// RUN:   -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/relu/relu.c
//
// Extract only the kernel function(s). PolyBench typically uses kernel_relu,
// so a regex keeps this robust across name variants.
// RUN: llvm-extract --rfunc=".*kernel.*" %t-kernel-full.ll -o %t-kernel-only.ll
//
// Import the LLVM IR into MLIR (LLVM dialect).
// RUN: mlir-translate --import-llvm %t-kernel-only.ll -o %t-kernel.mlir
//
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
// RUN:   --architecture-spec=%S/../../arch_spec/architecture.yaml \
// RUN:   --generate-code -o %t-mapping.mlir
// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM
//
// Check the mapped MLIR contains proper structure and neura operations.
// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
// MAPPING:      func.func @kernel(%arg0: i32 {llvm.noundef}

// YAML:      array_config:
// YAML-NEXT:     columns: 4
// YAML-NEXT:     rows: 4
// YAML-NEXT:     compiled_ii: 5
// YAML-NEXT:     cores:
// YAML-NEXT:     - column: 3
// YAML-NEXT:     row: 0
// YAML-NEXT:     core_id: "3"
// YAML-NEXT:     entries:
// YAML-NEXT:     - entry_id: "entry0"
// YAML-NEXT:     instructions:
// YAML-NEXT:     - index_per_ii: 4
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "RETURN"
// YAML-NEXT:     id: 3
// YAML-NEXT:     time_step: 9
// YAML-NEXT:     invalid_iterations: 1
// YAML-NEXT:     - column: 2
// YAML-NEXT:     row: 1
// YAML-NEXT:     core_id: "6"
// YAML-NEXT:     entries:
// YAML-NEXT:     - entry_id: "entry0"
// YAML-NEXT:     instructions:
// YAML-NEXT:     - index_per_ii: 0
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "DATA_MOV"
// YAML-NEXT:     id: 31
// YAML-NEXT:     time_step: 5
// YAML-NEXT:     invalid_iterations: 1
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "EAST"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$1"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - opcode: "STORE"
// YAML-NEXT:     id: 48
// YAML-NEXT:     time_step: 10
// YAML-NEXT:     invalid_iterations: 2
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "$2"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 1
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "DATA_MOV"
// YAML-NEXT:     id: 42
// YAML-NEXT:     time_step: 6
// YAML-NEXT:     invalid_iterations: 1
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "EAST"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$1"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - opcode: "DATA_MOV"
// YAML-NEXT:     id: 29
// YAML-NEXT:     time_step: 6
// YAML-NEXT:     invalid_iterations: 1
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "NORTH"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$2"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 2
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "ICMP_SGE"
// YAML-NEXT:     id: 44
// YAML-NEXT:     time_step: 7
// YAML-NEXT:     invalid_iterations: 1
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "EAST"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "#0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 3
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "GEP"
// YAML-NEXT:     id: 35
// YAML-NEXT:     time_step: 8
// YAML-NEXT:     invalid_iterations: 1
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "$1"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "$2"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$2"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 4
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "SEL"
// YAML-NEXT:     id: 46
// YAML-NEXT:     time_step: 9
// YAML-NEXT:     invalid_iterations: 1
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "$1"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "NORTH"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - column: 3
// YAML-NEXT:     row: 1
// YAML-NEXT:     core_id: "7"
// YAML-NEXT:     entries:
// YAML-NEXT:     - entry_id: "entry0"
// YAML-NEXT:     instructions:
// YAML-NEXT:     - index_per_ii: 0
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "GEP"
// YAML-NEXT:     id: 36
// YAML-NEXT:     time_step: 5
// YAML-NEXT:     invalid_iterations: 1
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "NORTH"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 1
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "LOAD"
// YAML-NEXT:     id: 41
// YAML-NEXT:     time_step: 6
// YAML-NEXT:     invalid_iterations: 1
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "WEST"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 3
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "DIV"
// YAML-NEXT:     id: 20
// YAML-NEXT:     time_step: 3
// YAML-NEXT:     invalid_iterations: 0
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "NORTH"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "#70"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 4
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "ZEXT"
// YAML-NEXT:     id: 26
// YAML-NEXT:     time_step: 4
// YAML-NEXT:     invalid_iterations: 0
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "WEST"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:     row: 2
// YAML-NEXT:     core_id: "9"
// YAML-NEXT:     entries:
// YAML-NEXT:     - entry_id: "entry0"
// YAML-NEXT:     instructions:
// YAML-NEXT:     - index_per_ii: 0
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "DATA_MOV"
// YAML-NEXT:     id: 4
// YAML-NEXT:     time_step: 0
// YAML-NEXT:     invalid_iterations: 0
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "EAST"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - opcode: "DATA_MOV"
// YAML-NEXT:     id: 27
// YAML-NEXT:     time_step: 5
// YAML-NEXT:     invalid_iterations: 1
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "EAST"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$2"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 2
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "PHI_START"
// YAML-NEXT:     id: 6
// YAML-NEXT:     time_step: 7
// YAML-NEXT:     invalid_iterations: 1
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "$1"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "EAST"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "$1"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 4
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "GRANT_PREDICATE"
// YAML-NEXT:     id: 33
// YAML-NEXT:     time_step: 9
// YAML-NEXT:     invalid_iterations: 1
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "$1"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "$2"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$1"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:     row: 2
// YAML-NEXT:     core_id: "10"
// YAML-NEXT:     entries:
// YAML-NEXT:     - entry_id: "entry0"
// YAML-NEXT:     instructions:
// YAML-NEXT:     - index_per_ii: 0
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "GRANT_PREDICATE"
// YAML-NEXT:     id: 34
// YAML-NEXT:     time_step: 5
// YAML-NEXT:     invalid_iterations: 1
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "$1"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - opcode: "DATA_MOV"
// YAML-NEXT:     id: 290001
// YAML-NEXT:     time_step: 5
// YAML-NEXT:     invalid_iterations: 1
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "EAST"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "SOUTH"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 1
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "PHI_START"
// YAML-NEXT:     id: 7
// YAML-NEXT:     time_step: 1
// YAML-NEXT:     invalid_iterations: 0
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "EAST"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "EAST"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 2
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "ADD"
// YAML-NEXT:     id: 12
// YAML-NEXT:     time_step: 2
// YAML-NEXT:     invalid_iterations: 0
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "#1"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "$1"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - opcode: "DATA_MOV"
// YAML-NEXT:     id: 40001
// YAML-NEXT:     time_step: 2
// YAML-NEXT:     invalid_iterations: 0
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "EAST"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "WEST"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 3
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "ICMP_EQ"
// YAML-NEXT:     id: 18
// YAML-NEXT:     time_step: 3
// YAML-NEXT:     invalid_iterations: 0
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "#4200"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - opcode: "DATA_MOV"
// YAML-NEXT:     id: 90001
// YAML-NEXT:     time_step: 8
// YAML-NEXT:     invalid_iterations: 1
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "WEST"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "SOUTH"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 4
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "NOT"
// YAML-NEXT:     id: 24
// YAML-NEXT:     time_step: 4
// YAML-NEXT:     invalid_iterations: 0
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "WEST"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - column: 3
// YAML-NEXT:     row: 2
// YAML-NEXT:     core_id: "11"
// YAML-NEXT:     entries:
// YAML-NEXT:     - entry_id: "entry0"
// YAML-NEXT:     instructions:
// YAML-NEXT:     - index_per_ii: 0
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "GRANT_ONCE"
// YAML-NEXT:     id: 0
// YAML-NEXT:     time_step: 0
// YAML-NEXT:     invalid_iterations: 0
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "#0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "WEST"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 2
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "CAST_TRUNC"
// YAML-NEXT:     id: 13
// YAML-NEXT:     time_step: 2
// YAML-NEXT:     invalid_iterations: 0
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "WEST"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "SOUTH"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 3
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "REM"
// YAML-NEXT:     id: 19
// YAML-NEXT:     time_step: 3
// YAML-NEXT:     invalid_iterations: 0
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "#70"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - index_per_ii: 4
// YAML-NEXT:     operations:
// YAML-NEXT:     - opcode: "ZEXT"
// YAML-NEXT:     id: 25
// YAML-NEXT:     time_step: 4
// YAML-NEXT:     invalid_iterations: 0
// YAML-NEXT:     src_operands:
// YAML-NEXT:     - operand: "$0"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     dst_operands:
// YAML-NEXT:     - operand: "SOUTH"
// YAML-NEXT:     color: "RED"
// YAML-NEXT:     - operand: "WEST"
// YAML-NEXT:     color: "RED"


// ASM:      # Compiled II: 5
// ASM:     PE(3,0):
// ASM-NEXT:     {
// ASM-NEXT:     RETURN (t=9, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=4)
// ASM:     PE(2,1):
// ASM-NEXT:     {
// ASM-NEXT:     DATA_MOV, [EAST, RED] -> [$1] (t=5, inv_iters=1)
// ASM-NEXT:     STORE, [$0], [$2] (t=10, inv_iters=2)
// ASM-NEXT:     } (idx_per_ii=0)
// ASM-NEXT:     {
// ASM-NEXT:     DATA_MOV, [EAST, RED] -> [$1] (t=6, inv_iters=1)
// ASM-NEXT:     DATA_MOV, [NORTH, RED] -> [$2] (t=6, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=1)
// ASM-NEXT:     {
// ASM-NEXT:     ICMP_SGE, [EAST, RED], [#0] -> [$0] (t=7, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=2)
// ASM-NEXT:     {
// ASM-NEXT:     GEP, [$1], [$2] -> [$2] (t=8, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=3)
// ASM-NEXT:     {
// ASM-NEXT:     SEL, [$0], [$1], [NORTH, RED] -> [$0] (t=9, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=4)
// ASM:     PE(3,1):
// ASM-NEXT:     {
// ASM-NEXT:     GEP, [$0], [NORTH, RED] -> [$0] (t=5, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=0)
// ASM-NEXT:     {
// ASM-NEXT:     LOAD, [$0] -> [WEST, RED] (t=6, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=1)
// ASM-NEXT:     {
// ASM-NEXT:     DIV, [NORTH, RED], [#70] -> [$0] (t=3, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=3)
// ASM-NEXT:     {
// ASM-NEXT:     ZEXT, [$0] -> [$0], [WEST, RED] (t=4, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=4)
// ASM:     PE(1,2):
// ASM-NEXT:     {
// ASM-NEXT:     DATA_MOV, [EAST, RED] -> [$0] (t=0, inv_iters=0)
// ASM-NEXT:     DATA_MOV, [EAST, RED] -> [$2] (t=5, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=0)
// ASM-NEXT:     {
// ASM-NEXT:     PHI_START, [$0], [$1] -> [EAST, RED], [$1] (t=7, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=2)
// ASM-NEXT:     {
// ASM-NEXT:     GRANT_PREDICATE, [$1], [$2] -> [$1] (t=9, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=4)
// ASM:     PE(2,2):
// ASM-NEXT:     {
// ASM-NEXT:     GRANT_PREDICATE, [$1], [$0] -> [$0] (t=5, inv_iters=1)
// ASM-NEXT:     DATA_MOV, [EAST, RED] -> [SOUTH, RED] (t=5, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=0)
// ASM-NEXT:     {
// ASM-NEXT:     PHI_START, [EAST, RED], [$0] -> [EAST, RED], [$0] (t=1, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=1)
// ASM-NEXT:     {
// ASM-NEXT:     ADD, [$0], [#1] -> [$0], [$1] (t=2, inv_iters=0)
// ASM-NEXT:     DATA_MOV, [EAST, RED] -> [WEST, RED] (t=2, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=2)
// ASM-NEXT:     {
// ASM-NEXT:     ICMP_EQ, [$0], [#4200] -> [$0] (t=3, inv_iters=0)
// ASM-NEXT:     DATA_MOV, [WEST, RED] -> [SOUTH, RED] (t=8, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=3)
// ASM-NEXT:     {
// ASM-NEXT:     NOT, [$0] -> [$0], [WEST, RED] (t=4, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=4)
// ASM:     PE(3,2):
// ASM-NEXT:     {
// ASM-NEXT:     GRANT_ONCE, [#0] -> [WEST, RED] (t=0, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=0)
// ASM-NEXT:     {
// ASM-NEXT:     CAST_TRUNC, [WEST, RED] -> [SOUTH, RED], [$0] (t=2, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=2)
// ASM-NEXT:     {
// ASM-NEXT:     REM, [$0], [#70] -> [$0] (t=3, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=3)
// ASM-NEXT:     {
// ASM-NEXT:     ZEXT, [$0] -> [SOUTH, RED], [WEST, RED] (t=4, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=4)
