// Compiles the original C kernel to mlir with vectorization enabled, then lowers it via Neura.
// RUN: clang++ -S -emit-llvm -O3 -fno-unroll-loops -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/fir/fir_int.cpp
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

// MAPPING:      func.func @_Z6kernelPiS_S_(
// MAPPING-SAME: accelerator = "neura"
// MAPPING-SAME: mapping_info = {compiled_ii = 5 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 5 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}
// MAPPING:      neura.phi_start
// MAPPING:      neura.gep
// MAPPING:      neura.load
// MAPPING:      "neura.vmul"
// MAPPING:      "neura.vadd"
// MAPPING:      neura.return_value

// YAML:      array_config:
// YAML:          columns: 4
// YAML:          rows: 4
// YAML:          compiled_ii: 5

// ASM:      # Compiled II: 5
// ASM:      PE(0,1):
// ASM:      GRANT_PREDICATE
// ASM:      PE(1,1):
// ASM:      VADD
// ASM:      RETURN_VALUE
