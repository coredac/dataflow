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


// MAPPING:      module attributes {{.*}}
// MAPPING-NEXT:   func.func @_Z6kernelPiS_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 5 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 5 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 0 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %1 = neura.reserve {dfg_id = 1 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %2 = "neura.data_mov"(%0) {dfg_id = 3 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %3 = neura.phi_start %2, %1 {dfg_id = 5 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 3 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %4 = "neura.data_mov"(%3) {dfg_id = 8 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %5 = "neura.gep"(%4) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 10 : i32, lhs_value = "%arg0", mapping_locs = [{id = 10 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %6 = "neura.data_mov"(%5) {dfg_id = 13 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %7 = "neura.load"(%6) {dfg_id = 15 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %8 = "neura.data_mov"(%7) {dfg_id = 17 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %9 = "neura.mul"(%8) {dfg_id = 19 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 3 : i32, y = 2 : i32}], rhs_value = 5 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %10 = "neura.data_mov"(%9) {dfg_id = 21 : i32, mapping_locs = [{id = 37 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %11 = "neura.add"(%10) {dfg_id = 23 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 3 : i32, y = 3 : i32}], rhs_value = -5 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %12 = "neura.data_mov"(%11) {dfg_id = 25 : i32, mapping_locs = [{id = 480 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %13 = "neura.div"(%12) {dfg_id = 26 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 3 : i32, y = 3 : i32}], rhs_value = 18 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %14 = "neura.data_mov"(%13) {dfg_id = 27 : i32, mapping_locs = [{id = 480 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %15 = neura.sext %14 {dfg_id = 28 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 3 : i32, y = 3 : i32}]} : !neura.data<i32, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %16 = "neura.data_mov"(%15) {dfg_id = 29 : i32, mapping_locs = [{id = 480 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %17 = "neura.gep"(%16) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 30 : i32, lhs_value = "%arg1", mapping_locs = [{id = 15 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 8 : i32, x = 3 : i32, y = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %18 = "neura.data_mov"(%17) {dfg_id = 32 : i32, mapping_locs = [{id = 480 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %19 = "neura.load"(%18) {dfg_id = 33 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 9 : i32, x = 3 : i32, y = 3 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %20 = "neura.data_mov"(%19) {dfg_id = 34 : i32, mapping_locs = [{id = 46 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %21 = "neura.add"(%20) {dfg_id = 35 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 10 : i32, x = 2 : i32, y = 3 : i32}], rhs_value = 1 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %22 = "neura.data_mov"(%21) {dfg_id = 36 : i32, mapping_locs = [{id = 448 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %23 = "neura.data_mov"(%17) {dfg_id = 31 : i32, mapping_locs = [{id = 46 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 8 : i32}, {id = 449 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 449 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     "neura.store"(%22, %23) {dfg_id = 37 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 1 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 11 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT:     %24 = "neura.data_mov"(%3) {dfg_id = 7 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %25 = "neura.add"(%24) {dfg_id = 9 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 2 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %26 = "neura.data_mov"(%25) {dfg_id = 12 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %27 = "neura.icmp"(%26) <{cmpType = "eq"}> {dfg_id = 14 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 2 : i32}], rhs_value = 20 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %28 = "neura.data_mov"(%27) {dfg_id = 16 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %29 = "neura.not"(%28) {dfg_id = 18 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %30 = "neura.data_mov"(%25) {dfg_id = 11 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 320 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}, {id = 320 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %31 = "neura.data_mov"(%29) {dfg_id = 20 : i32, mapping_locs = [{id = 321 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %32 = neura.grant_predicate %30, %31 {dfg_id = 22 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %32 -> %1 {dfg_id = 24 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %33 = "neura.grant_once"() <{constant_value = true}> {dfg_id = 2 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 10 : i32, x = 2 : i32, y = 0 : i32}]} : () -> !neura.data<i1, i1>
// MAPPING-NEXT:     %34 = "neura.data_mov"(%33) {dfg_id = 4 : i32, mapping_locs = [{id = 64 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     "neura.return"(%34) {dfg_id = 6 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 1 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 11 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> ()
// MAPPING-NEXT:   }
// MAPPING-NEXT: }

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
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 2
// YAML-NEXT:                   time_step: 10
// YAML-NEXT:                   invalid_iterations: 2
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#-1"
// YAML-NEXT:                       color: "RED"

// ASM:      # Compiled II: 5
// ASM:      PE(2,0):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#-1] -> [$0] (t=10, inv_iters=2)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   RETURN, [$0] (t=11, inv_iters=2)
// ASM-NEXT: } (idx_per_ii=1)
// ASM:      PE(2,2):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$1] -> [EAST, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GEP, [EAST, RED] -> [$0] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [$0] -> [EAST, RED] (t=3, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$0] (t=3, inv_iters=0)

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

// DOT:      digraph G {
// DOT-NEXT:   compound = true;
// DOT-NEXT:   subgraph cluster_1 {
// DOT-NEXT:     v2 [label = " ", shape = plain];
// DOT-NEXT: label = "builtin.module {{.*}}";
// DOT-NEXT:     subgraph cluster_3 {
// DOT-NEXT:       v4 [label = " ", shape = plain];
// DOT-NEXT:       label = "";
// DOT-NEXT:       subgraph cluster_5 {
// DOT-NEXT:         v6 [label = " ", shape = plain];
// DOT-NEXT:         label = "func.func : ()\n\nCConv: #llvm.cconv<ccc>\naccelerator: \"neura\"\narg_attrs: [{llvm.nocapture, ll...\ndataflow_mode: \"predicate\"\nfunction_type: (!llvm.ptr, !llvm.pt...\nlinkage: #llvm.linkage<extern...\nmemory_effects: #llvm.memory_effects...\nno_unwind: unit\npassthrough: [\"mustprogress\", \"no...\nsym_name: \"_Z6kernelPiS_\"\ntarget_cpu: \"x86-64\"\ntarget_features: #llvm.target_feature...\ntune_cpu: \"generic\"\nunnamed_addr: 1 : i64\nvisibility_: 0 : i64";
// DOT-NEXT:         subgraph cluster_7 {
// DOT-NEXT:           v8 [label = " ", shape = plain];
// DOT-NEXT:           label = "";
// DOT-NEXT:           v9 [label = "arg0", shape = ellipse];
// DOT-NEXT:           v10 [label = "arg1", shape = ellipse];
// DOT-NEXT:           v11 [fillcolor = "0.000000 1.0 1.0", label = "neura.grant_once : (!neura.data<i64, i1>)\n\nconstant_value: 0 : i64", shape = ellipse, style = filled];
// DOT-NEXT:           v12 [fillcolor = "0.058824 1.0 1.0", label = "neura.reserve : (!neura.data<i64, i1>)\n", shape = ellipse, style = filled];
// DOT-NEXT:           v13 [fillcolor = "0.117647 1.0 1.0", label = "neura.phi_start : (!neura.data<i64, i1>)\n", shape = ellipse, style = filled];
// DOT-NEXT:           v14 [fillcolor = "0.176471 1.0 1.0", label = "neura.gep : (!neura.data<!llvm.pt...)\n\nlhs_value: \"%arg0\"\noperandSegmentSizes: array<i32: 0, 1>", shape = ellipse, style = filled];
