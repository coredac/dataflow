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
//
// Check the mapped MLIR contains key operations with full statements.
// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
// MAPPING:          %0 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 0 : i32, mapping_locs = {{.*}}} : () -> !neura.data<i64, i1>
// MAPPING:          %1 = neura.reserve {dfg_id = 1 : i32} : !neura.data<i64, i1>
// MAPPING:          %2 = "neura.data_mov"(%0) {dfg_id = 3 : i32, mapping_locs = {{.*}}} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING:          %3 = neura.phi_start %2, %1 {dfg_id = 4 : i32, mapping_locs = {{.*}}} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING:          %4 = "neura.data_mov"(%3) {dfg_id = 7 : i32, mapping_locs = {{.*}}} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING:          %5 = "neura.gep"(%4) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 9 : i32, lhs_value = "%arg0", mapping_locs = {{.*}}} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING:          %6 = "neura.data_mov"(%5) {dfg_id = 12 : i32, mapping_locs = {{.*}}} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING:          %7 = "neura.load"(%6) {dfg_id = 14 : i32, mapping_locs = {{.*}}} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING:          %8 = "neura.data_mov"(%7) {dfg_id = 19 : i32, mapping_locs = {{.*}}} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING:          %9 = "neura.icmp"(%8) <{cmpType = "sgt"}> {dfg_id = 22 : i32, mapping_locs = {{.*}}, rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING:          %10 = "neura.data_mov"(%3) {dfg_id = 6 : i32, mapping_locs = {{.*}}} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING:          %11 = "neura.data_mov"(%9) {dfg_id = 26 : i32, mapping_locs = {{.*}}} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING:          %12 = neura.grant_predicate %10, %11 {dfg_id = 30 : i32, mapping_locs = {{.*}}} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING:          %13 = "neura.data_mov"(%7) {dfg_id = 18 : i32, mapping_locs = {{.*}}} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING:          %14 = "neura.data_mov"(%9) {dfg_id = 25 : i32, mapping_locs = {{.*}}} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING:          %15 = neura.grant_predicate %13, %14 {dfg_id = 29 : i32, mapping_locs = {{.*}}} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING:          %16 = "neura.data_mov"(%12) {dfg_id = 33 : i32, mapping_locs = {{.*}}} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING:          %17 = "neura.gep"(%16) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 34 : i32, lhs_value = "%arg1", mapping_locs = {{.*}}} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING:          %18 = "neura.data_mov"(%17) {dfg_id = 36 : i32, mapping_locs = {{.*}}} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING:          %19 = "neura.load"(%18) {dfg_id = 37 : i32, mapping_locs = {{.*}}} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING:          %20 = "neura.data_mov"(%19) {dfg_id = 38 : i32, mapping_locs = {{.*}}} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING:          %21 = "neura.data_mov"(%15) {dfg_id = 32 : i32, mapping_locs = {{.*}}} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING:          %22 = "neura.add"(%20, %21) {dfg_id = 39 : i32, mapping_locs = {{.*}}} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING:          %23 = "neura.data_mov"(%22) {dfg_id = 40 : i32, mapping_locs = {{.*}}} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING:          %24 = "neura.data_mov"(%17) {dfg_id = 35 : i32, mapping_locs = {{.*}}} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING:          "neura.store"(%23, %24) {dfg_id = 41 : i32, mapping_locs = {{.*}}} : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING:          %25 = "neura.data_mov"(%3) {dfg_id = 5 : i32, mapping_locs = {{.*}}} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING:          %26 = "neura.add"(%25) {dfg_id = 8 : i32, mapping_locs = {{.*}}, rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING:          %27 = "neura.data_mov"(%26) {dfg_id = 11 : i32, mapping_locs = {{.*}}} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING:          %28 = "neura.icmp"(%27) <{cmpType = "eq"}> {dfg_id = 13 : i32, mapping_locs = {{.*}}, rhs_value = 32 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING:          %29 = "neura.data_mov"(%28) {dfg_id = 17 : i32, mapping_locs = {{.*}}} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING:          %30 = "neura.not"(%29) {dfg_id = 21 : i32, mapping_locs = {{.*}}} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING:          %31 = "neura.data_mov"(%26) {dfg_id = 10 : i32, mapping_locs = {{.*}}} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING:          %32 = "neura.data_mov"(%30) {dfg_id = 24 : i32, mapping_locs = {{.*}}} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING:          %33 = neura.grant_predicate %31, %32 {dfg_id = 28 : i32, mapping_locs = {{.*}}} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING:          neura.ctrl_mov %33 -> %1 {dfg_id = 31 : i32, mapping_locs = {{.*}}} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING:          %34 = "neura.data_mov"(%28) {dfg_id = 15 : i32, mapping_locs = {{.*}}} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING:          %35 = "neura.data_mov"(%28) {dfg_id = 16 : i32, mapping_locs = {{.*}}} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING:          %36 = neura.grant_predicate %34, %35 {dfg_id = 20 : i32, mapping_locs = {{.*}}} : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
// MAPPING:          %37 = "neura.data_mov"(%36) {dfg_id = 23 : i32, mapping_locs = {{.*}}} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING:          neura.return_void %37 : !neura.data<i1, i1> {dfg_id = 27 : i32, mapping_locs = {{.*}}}
// MAPPING:          neura.yield {dfg_id = 2 : i32}
// MAPPING:        }
// MAPPING:      }

// YAML:      array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   compiled_ii: 5
// YAML-NEXT:   cores:
// YAML-NEXT:        - column: 2
// YAML-NEXT:      row: 1
// YAML-NEXT:      core_id: "6"
// YAML-NEXT:     entries:
// YAML-NEXT:       - entry_id: "entry0"
// YAML-NEXT:         instructions:
// YAML-DAG:           - index_per_ii: 3
// YAML-DAG:             operations:
// YAML-DAG:               - opcode: "DATA_MOV"
// YAML-DAG:                 id: 380001
// YAML-DAG:                 time_step: 8
// YAML-DAG:                 invalid_iterations: 1
// YAML-DAG:                 src_operands:
// YAML-DAG:                   - operand: "EAST"
// YAML-DAG:                     color: "RED"
// YAML-DAG:                 dst_operands:
// YAML-DAG:                   - operand: "NORTH"
// YAML-DAG:                     color: "RED"

// ASM: # Compiled II: 5
// ASM: PE(3,2):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#0] -> [$0] (t=0, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [SOUTH, RED] -> [NORTH, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [WEST, RED] -> [WEST, RED], [SOUTH, RED], [$0] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   ADD, [$0], [#1] -> [$0], [WEST, RED] (t=2, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [SOUTH, RED] (t=12, inv_iters=2)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_EQ, [$0], [#32] -> [WEST, RED], [NORTH, RED], [SOUTH, RED] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_SGT, [WEST, RED], [#0] -> [SOUTH, RED], [NORTH, RED] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
