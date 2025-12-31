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
// RUN: mlir-neura-opt %t-kernel.mlir --view-op-graph 2>&1 | sed -n '/^digraph G {/,/^}$/p' > relu_kernel_original.dot
// RUN: dot -Tpng relu_kernel_original.dot -o relu_kernel_original.png
// RUN: dot -Tjson relu_kernel_original.dot -o relu_kernel_original.json
// RUN: mlir-neura-opt %t-kernel.mlir \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
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


// MAPPING: func.func @kernel
// MAPPING-SAME: accelerator = "neura"
// MAPPING-SAME: dataflow_mode = "predicate"
// MAPPING-SAME: mapping_info = {compiled_ii = 5 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 5 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}

// YAML:      array_config:
// YAML-NEXT:     columns: 4
// YAML-NEXT:     rows: 4
// YAML-NEXT:     compiled_ii: 5
// YAML-NEXT:     cores:
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "2"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 440001
// YAML-NEXT:                   time_step: 9
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"

// ASM:      # Compiled II: 5
// ASM:     PE(2,0):
// ASM-NEXT:     {
// ASM-NEXT:     DATA_MOV, [EAST, RED] -> [NORTH, RED] (t=9, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=4)
// ASM:     PE(3,0):
// ASM-NEXT:     {
// ASM-NEXT:     DATA_MOV, [NORTH, RED] -> [$0] (t=5, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=0)
// ASM-NEXT:     {
// ASM-NEXT:     GEP, [$0], [$1] -> [WEST, RED] (t=8, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=3)
// ASM-NEXT:     {
// ASM-NEXT:     DATA_MOV, [NORTH, RED] -> [$1] (t=4, inv_iters=0)
// ASM-NEXT:     } (idx_per_ii=4)
