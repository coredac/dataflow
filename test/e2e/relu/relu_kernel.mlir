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
//
// Check the mapped MLIR contains proper structure and neura operations.
// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
// MAPPING:      func.func @kernel(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg4: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 5 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 5 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// MAPPING:          %0 = "neura.grant_once"() <{constant_value = 0 : i32}> {dfg_id = 0 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %1 = neura.reserve {dfg_id = 1 : i32} : !neura.data<i32, i1>
// YAML:      array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   compiled_ii: 5
// YAML-NEXT:   cores:
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
// YAML-NEXT:     - column: 3
// YAML-NEXT:       row: 0

// ASM:      # Compiled II: 5
// ASM:      PE(2,0):
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [NORTH, RED] (t=9, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=4)
// ASM:      PE(3,0):
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$0] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$1] (t=7, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   GEP, [$0], [$1] -> [WEST, RED] (t=8, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM:      PE(2,1):
// ASM-NEXT: {
// ASM-NEXT:   STORE, [$0], [SOUTH, RED] (t=10, inv_iters=2)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_SGE, [EAST, RED], [#0] -> [$0] (t=7, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$1] (t=8, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   SEL, [$0], [$1], [NORTH, RED] -> [$0] (t=9, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=4)
// ASM:      PE(3,1):
// ASM-NEXT: {
// ASM-NEXT:   GEP, [$0], [NORTH, RED] -> [$0] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [$0] -> [WEST, RED], [$0] (t=6, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [SOUTH, RED] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [$0] -> [WEST, RED] (t=7, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   DIV, [NORTH, RED], [#70] -> [$0] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   ZEXT, [$0] -> [$0], [SOUTH, RED] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM:      PE(1,2):
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$2] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$4] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$3] -> [EAST, RED], [$3] (t=7, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$0] (t=3, inv_iters=0)
// ASM-NEXT:   GRANT_PREDICATE, [$1], [$2] -> [NORTH, RED] (t=8, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$1] (t=4, inv_iters=0)
// ASM-NEXT:   GRANT_PREDICATE, [$3], [$4] -> [$3] (t=9, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=4)
// ASM:      PE(2,2):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$1], [$0] -> [$0] (t=5, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [$2] -> [WEST, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [EAST, RED], [$0] -> [EAST, RED], [$0] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   ADD, [$0], [#1] -> [$0], [$1] (t=2, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [WEST, RED] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_EQ, [$0], [#4200] -> [$0], [WEST, RED], [$2] (t=3, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [SOUTH, RED] (t=8, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
