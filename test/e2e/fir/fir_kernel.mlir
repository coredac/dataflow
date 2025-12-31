// Compiles the original C kernel to mlir, then lowers it via Neura.
// RUN: clang++ -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/fir/fir_int.cpp
// RUN: llvm-extract --rfunc=".*kernel.*" %t-kernel-full.ll -o %t-kernel-only.ll
// RUN: mlir-translate --import-llvm %t-kernel-only.ll -o %t-kernel.mlir

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


// MAPPING: func.func @_Z6kernelPiS_S_
// MAPPING-SAME: accelerator = "neura"
// MAPPING-SAME: dataflow_mode = "predicate"
// MAPPING-SAME: mapping_info = {compiled_ii = 5 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 5 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}

// YAML:      array_config:
// YAML-NEXT:     columns: 4
// YAML-NEXT:     rows: 4
// YAML-NEXT:     compiled_ii: 5
// YAML-NEXT:     cores:
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "4"
// YAML-NEXT:       entries:
// YAML-NEXT:       - entry_id: "entry0"
// YAML-NEXT:         instructions:
// YAML-NEXT:         - index_per_ii: 3
// YAML-NEXT:           operations:
// YAML-NEXT:           - opcode: "CTRL_MOV"
// YAML-NEXT:             id: 400001
// YAML-NEXT:             time_step: 8
// YAML-NEXT:             invalid_iterations: 1
// YAML-NEXT:             src_operands:
// YAML-NEXT:             - operand: "EAST"
// YAML-NEXT:               color: "RED"
// YAML-NEXT:             dst_operands:
// YAML-NEXT:             - operand: "NORTH"
// YAML-NEXT:               color: "RED"

// ASM:      # Compiled II: 5
// ASM:     PE(0,1):
// ASM-NEXT:     {
// ASM-NEXT:     CTRL_MOV, [EAST, RED] -> [NORTH, RED] (t=8, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=3)
// ASM:     PE(1,1):
// ASM-NEXT:     {
// ASM-NEXT:     DATA_MOV, [NORTH, RED] -> [$1] (t=5, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=0)
// ASM-NEXT:     {
// ASM-NEXT:     ADD, [EAST, RED], [NORTH, RED] -> [$0], [NORTH, RED] (t=6, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=1)


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
// RUN:   --view-op-graph 2>&1 | sed -n '/^digraph G {/,/^}$/p' > fir_kernel.dot
// RUN: dot -Tpng fir_kernel.dot -o fir_kernel.png
// RUN: dot -Tjson fir_kernel.dot -o fir_kernel.json
// RUN: FileCheck %s --input-file=fir_kernel.dot -check-prefix=DOT

// DOT: digraph G {
