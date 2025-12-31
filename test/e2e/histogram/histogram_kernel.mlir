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
// RUN:   --canonicalize-return \
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


// MAPPING: func.func @_Z6kernelPiS_
// MAPPING-SAME: accelerator = "neura"

// YAML:      array_config:
// YAML-NEXT:     columns: 4
// YAML-NEXT:     rows: 4
// YAML-NEXT:     compiled_ii: 5
// YAML-NEXT:     cores:
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "6"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ADD"
// YAML-NEXT:                   id: 38
// YAML-NEXT:                   time_step: 10
// YAML-NEXT:                   invalid_iterations: 2
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "#1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"

// ASM:      # Compiled II: 5
// ASM:     PE(2,1):
// ASM-NEXT:     {
// ASM-NEXT:     ADD, [$0], [#1] -> [$0] (t=10, inv_iters=2)
// ASM-NEXT:     } (idx_per_ii=0)
// ASM-NEXT:     {
// ASM-NEXT:     STORE, [$0], [$1] (t=11, inv_iters=2)
// ASM-NEXT:     } (idx_per_ii=1)
// ASM:     PE(3,1):
// ASM-NEXT:     {
// ASM-NEXT:     ADD, [NORTH, RED], [#-5] -> [$0] (t=5, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=0)
