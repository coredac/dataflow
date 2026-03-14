// Compiles a GEMM kernel to LLVM IR, imports to MLIR, then lowers via Neura.
// RUN: clang -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -std=c11 \
// RUN:   -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/gemm/gemm.c
// RUN: llvm-extract --rfunc=".*kernel.*" %t-kernel-full.ll -o %t-kernel-only.ll
// RUN: mlir-translate --import-llvm %t-kernel-only.ll -o %t-kernel.mlir
//
// RUN: mlir-neura-opt %t-kernel.mlir \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-input-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-operation-on-tile="mapping-strategy=heuristic" \
// RUN:   --architecture-spec=%S/../../arch_spec/architecture.yaml \
// RUN:   --generate-code -o %t-mapping.mlir
// RUN: FileCheck %s --input-file=%t-mapping.mlir --check-prefix=MAPPING
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM
//
// MAPPING:   func.func @kernel(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg4: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 17 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 13 : i32, res_mii = 6 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = "%arg0"}> {dfg_id = 0 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %1 = "neura.constant"() <{value = "%arg0"}> {dfg_id = 1 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 2 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %2 = "neura.grant_once"() <{constant_value = "%arg1"}> {dfg_id = 2 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 1 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %3 = "neura.constant"() <{value = "%arg1"}> {dfg_id = 3 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 1 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %4 = "neura.grant_once"() <{constant_value = "%arg2"}> {dfg_id = 4 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 2 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %5 = "neura.constant"() <{value = "%arg2"}> {dfg_id = 5 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 2 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %6 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 6 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %7 = "neura.data_mov"(%1) {dfg_id = 28 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %8 = "neura.icmp"(%7) <{cmpType = "sgt"}> {dfg_id = 34 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 2 : i32}], rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %9 = "neura.data_mov"(%5) {dfg_id = 32 : i32, mapping_locs = [{id = 8000 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %10 = "neura.icmp"(%9) <{cmpType = "sgt"}> {dfg_id = 36 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 2 : i32}], rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %11 = "neura.data_mov"(%8) {dfg_id = 37 : i32, mapping_locs = [{id = 27 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %12 = "neura.data_mov"(%10) {dfg_id = 39 : i32, mapping_locs = [{id = 8000 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %13 = "neura.and"(%11, %12) {dfg_id = 40 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %14 = "neura.data_mov"(%3) {dfg_id = 30 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %15 = "neura.icmp"(%14) <{cmpType = "sgt"}> {dfg_id = 35 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}], rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %16 = "neura.data_mov"(%13) {dfg_id = 41 : i32, mapping_locs = [{id = 25 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %17 = "neura.data_mov"(%15) {dfg_id = 38 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %18 = "neura.and"(%16, %17) {dfg_id = 42 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %19 = "neura.data_mov"(%18) {dfg_id = 44 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %20 = "neura.grant_once"(%19) {dfg_id = 46 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %21 = "neura.data_mov"(%18) {dfg_id = 43 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 2 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 4 : i32}, {id = 2 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 5 : i32}, {id = 2 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 6 : i32}, {id = 2 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 7 : i32}, {id = 2 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 8 : i32}, {id = 2 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}, {id = 2 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 10 : i32}, {id = 2 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 11 : i32}, {id = 2 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 12 : i32}, {id = 2 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 13 : i32}, {id = 2 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 14 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %22 = "neura.not"(%21) {dfg_id = 45 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 15 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %23 = "neura.data_mov"(%22) {dfg_id = 47 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 15 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %24 = "neura.grant_once"(%23) {dfg_id = 53 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 16 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %25 = "neura.data_mov"(%0) {dfg_id = 27 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %26 = "neura.data_mov"(%20) {dfg_id = 52 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 1 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}, {id = 1 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}, {id = 1 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}, {id = 1 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %27 = neura.grant_predicate %25, %26 {dfg_id = 58 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %28 = "neura.data_mov"(%4) {dfg_id = 31 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %29 = "neura.data_mov"(%20) {dfg_id = 51 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 24 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 28 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %30 = neura.grant_predicate %28, %29 {dfg_id = 57 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %31 = "neura.data_mov"(%2) {dfg_id = 29 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %32 = "neura.data_mov"(%20) {dfg_id = 50 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %33 = neura.grant_predicate %31, %32 {dfg_id = 56 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %34 = "neura.data_mov"(%6) {dfg_id = 33 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %35 = "neura.data_mov"(%20) {dfg_id = 49 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %36 = neura.grant_predicate %34, %35 {dfg_id = 55 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %37 = "neura.data_mov"(%20) {dfg_id = 48 : i32, mapping_locs = [{id = 4002 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 4 : i32}, {id = 12 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 8000 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 8000 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 8000 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 8000 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}, {id = 8000 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}, {id = 8000 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}, {id = 8000 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}, {id = 8000 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}, {id = 8000 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 14 : i32}, {id = 8000 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 15 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %38 = "neura.not"(%37) {dfg_id = 54 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 16 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %39 = "neura.data_mov"(%24) {dfg_id = 59 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 16 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %40 = "neura.data_mov"(%38) {dfg_id = 60 : i32, mapping_locs = [{id = 25 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 16 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %41 = neura.grant_predicate %39, %40 {dfg_id = 66 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 17 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
// MAPPING-NEXT:     %42 = "neura.data_mov"(%27) {dfg_id = 65 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %43 = neura.zext %42 {dfg_id = 71 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i32, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %44 = "neura.data_mov"(%30) {dfg_id = 64 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %45 = neura.zext %44 {dfg_id = 70 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i32, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %46 = "neura.data_mov"(%33) {dfg_id = 63 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %47 = neura.zext %46 {dfg_id = 69 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i32, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %48 = neura.reserve {dfg_id = 7 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %49 = "neura.data_mov"(%43) {dfg_id = 78 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %50 = neura.phi_start %49, %48 {dfg_id = 84 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %51 = neura.reserve {dfg_id = 8 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %52 = "neura.data_mov"(%45) {dfg_id = 77 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %53 = neura.phi_start %52, %51 {dfg_id = 83 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %54 = neura.reserve {dfg_id = 9 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %55 = "neura.data_mov"(%47) {dfg_id = 76 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %56 = neura.phi_start %55, %54 {dfg_id = 82 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %57 = neura.reserve {dfg_id = 10 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %58 = "neura.data_mov"(%36) {dfg_id = 62 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %59 = neura.phi_start %58, %57 {dfg_id = 68 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %60 = neura.reserve {dfg_id = 11 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %61 = "neura.data_mov"(%36) {dfg_id = 61 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 14 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 192 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 192 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 192 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %62 = neura.phi_start %61, %60 {dfg_id = 67 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %63 = neura.reserve {dfg_id = 12 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %64 = "neura.data_mov"(%50) {dfg_id = 93 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %65 = neura.phi_start %64, %63 {dfg_id = 101 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 12 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %66 = neura.reserve {dfg_id = 13 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %67 = "neura.data_mov"(%53) {dfg_id = 92 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %68 = neura.phi_start %67, %66 {dfg_id = 100 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %69 = neura.reserve {dfg_id = 14 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %70 = "neura.data_mov"(%56) {dfg_id = 91 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %71 = neura.phi_start %70, %69 {dfg_id = 99 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %72 = neura.reserve {dfg_id = 15 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %73 = "neura.data_mov"(%59) {dfg_id = 75 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %74 = neura.phi_start %73, %72 {dfg_id = 81 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %75 = neura.reserve {dfg_id = 16 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %76 = "neura.data_mov"(%62) {dfg_id = 73 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %77 = neura.phi_start %76, %75 {dfg_id = 79 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %78 = neura.reserve {dfg_id = 17 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %79 = "neura.data_mov"(%59) {dfg_id = 74 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 4000 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 4000 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %80 = neura.phi_start %79, %78 {dfg_id = 80 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %81 = "neura.data_mov"(%77) {dfg_id = 86 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}, {id = 13 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 12 : i32}, {id = 4002 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 13 : i32}, {id = 4002 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 14 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %82 = "neura.data_mov"(%80) {dfg_id = 88 : i32, mapping_locs = [{id = 4003 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 9 : i32}, {id = 4003 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 10 : i32}, {id = 4003 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 11 : i32}, {id = 4003 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 12 : i32}, {id = 4003 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 13 : i32}, {id = 4003 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 14 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %83 = "neura.gep"(%81, %82) <{operandSegmentSizes = array<i32: 0, 2>}> {dfg_id = 96 : i32, lhs_value = "%arg4", mapping_locs = [{id = 4 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 15 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %84 = neura.reserve {dfg_id = 18 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %85 = "neura.data_mov"(%74) {dfg_id = 90 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 194 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 8 : i32}, {id = 194 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}, {id = 194 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 10 : i32}, {id = 194 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 11 : i32}, {id = 194 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 12 : i32}, {id = 194 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 13 : i32}, {id = 194 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 14 : i32}, {id = 194 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 15 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %86 = neura.phi_start %85, %84 {dfg_id = 98 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 16 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %87 = neura.reserve {dfg_id = 19 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %88 = "neura.data_mov"(%65) {dfg_id = 116 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %89 = neura.phi_start %88, %87 {dfg_id = 123 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 13 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %90 = neura.reserve {dfg_id = 20 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %91 = "neura.data_mov"(%68) {dfg_id = 115 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %92 = neura.phi_start %91, %90 {dfg_id = 122 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %93 = neura.reserve {dfg_id = 21 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %94 = "neura.data_mov"(%71) {dfg_id = 114 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %95 = neura.phi_start %94, %93 {dfg_id = 121 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %96 = neura.reserve {dfg_id = 22 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %97 = "neura.data_mov"(%77) {dfg_id = 85 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %98 = neura.phi_start %97, %96 {dfg_id = 94 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 12 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %99 = neura.reserve {dfg_id = 23 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %100 = "neura.data_mov"(%80) {dfg_id = 87 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %101 = neura.phi_start %100, %99 {dfg_id = 95 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %102 = neura.reserve {dfg_id = 24 : i32} : !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %103 = "neura.data_mov"(%83) {dfg_id = 108 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 15 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %104 = neura.phi_start %103, %102 {dfg_id = 117 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 16 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %105 = neura.reserve {dfg_id = 25 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %106 = "neura.data_mov"(%74) {dfg_id = 89 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %107 = neura.phi_start %106, %105 {dfg_id = 97 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %108 = "neura.data_mov"(%104) {dfg_id = 125 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 16 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %109 = "neura.load"(%108) {dfg_id = 138 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 17 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f64, i1>
// MAPPING-NEXT:     %110 = "neura.data_mov"(%101) {dfg_id = 107 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}, {id = 0 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}, {id = 1000 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}, {id = 1000 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}, {id = 1000 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 14 : i32}, {id = 1000 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 15 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %111 = "neura.data_mov"(%107) {dfg_id = 111 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 1001 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 1001 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}, {id = 1001 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}, {id = 1001 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 1001 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}, {id = 1001 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 1001 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %112 = "neura.gep"(%110, %111) <{operandSegmentSizes = array<i32: 0, 2>}> {dfg_id = 120 : i32, lhs_value = "%arg5", mapping_locs = [{id = 1 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 16 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %113 = "neura.data_mov"(%112) {dfg_id = 130 : i32, mapping_locs = [{id = 1000 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 16 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %114 = "neura.load"(%113) {dfg_id = 140 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 17 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f64, i1>
// MAPPING-NEXT:     %115 = "neura.data_mov"(%98) {dfg_id = 104 : i32, mapping_locs = [{id = 195 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 12 : i32}, {id = 195 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 13 : i32}, {id = 195 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 14 : i32}, {id = 195 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 15 : i32}, {id = 195 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 16 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %116 = "neura.data_mov"(%107) {dfg_id = 110 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 196 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 9 : i32}, {id = 196 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 10 : i32}, {id = 196 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 11 : i32}, {id = 196 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 12 : i32}, {id = 196 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 13 : i32}, {id = 196 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 14 : i32}, {id = 196 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 15 : i32}, {id = 196 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 16 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %117 = "neura.gep"(%115, %116) <{operandSegmentSizes = array<i32: 0, 2>}> {dfg_id = 119 : i32, lhs_value = "%arg3", mapping_locs = [{id = 6 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 17 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %118 = "neura.data_mov"(%117) {dfg_id = 129 : i32, mapping_locs = [{id = 19 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 17 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %119 = "neura.load"(%118) {dfg_id = 139 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 18 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f64, i1>
// MAPPING-NEXT:     %120 = "neura.data_mov"(%109) {dfg_id = 142 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 17 : i32}, {id = 1000 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 18 : i32}]} : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// MAPPING-NEXT:     %121 = "neura.data_mov"(%114) {dfg_id = 144 : i32, mapping_locs = [{id = 1001 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 17 : i32}, {id = 1001 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 18 : i32}]} : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// MAPPING-NEXT:     %122 = "neura.data_mov"(%119) {dfg_id = 143 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 18 : i32}]} : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// MAPPING-NEXT:     %123 = "neura.fmul_fadd"(%120, %121, %122) {dfg_id = 152 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 19 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<f64, i1>, !neura.data<f64, i1>, !neura.data<f64, i1>) -> !neura.data<f64, i1>
// MAPPING-NEXT:     %124 = "neura.data_mov"(%123) {dfg_id = 160 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 19 : i32}]} : (!neura.data<f64, i1>) -> !neura.data<f64, i1>
// MAPPING-NEXT:     %125 = "neura.data_mov"(%117) {dfg_id = 128 : i32, mapping_locs = [{id = 196 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 17 : i32}, {id = 19 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 18 : i32}, {id = 2000 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 19 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     "neura.store"(%124, %125) {dfg_id = 181 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 20 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<f64, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT:     %126 = "neura.data_mov"(%107) {dfg_id = 109 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %127 = "neura.add"(%126) {dfg_id = 118 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 1 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %128 = "neura.data_mov"(%127) {dfg_id = 127 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %129 = "neura.data_mov"(%95) {dfg_id = 133 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %130 = "neura.icmp"(%128, %129) <{cmpType = "eq"}> {dfg_id = 141 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %131 = "neura.data_mov"(%130) {dfg_id = 151 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %132 = "neura.not"(%131) {dfg_id = 159 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %133 = "neura.data_mov"(%127) {dfg_id = 126 : i32, mapping_locs = [{id = 162 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}, {id = 162 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 10 : i32}, {id = 162 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 11 : i32}, {id = 162 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 12 : i32}, {id = 162 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %134 = "neura.data_mov"(%132) {dfg_id = 180 : i32, mapping_locs = [{id = 163 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 11 : i32}, {id = 163 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 12 : i32}, {id = 163 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %135 = neura.grant_predicate %133, %134 {dfg_id = 190 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 14 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %135 -> %105 {dfg_id = 200 : i32, mapping_locs = [{id = 161 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 161 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}, {id = 161 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 16 : i32}, {id = 161 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 17 : i32}, {id = 161 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 18 : i32}, {id = 161 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 19 : i32}, {id = 161 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 20 : i32}, {id = 161 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 21 : i32}, {id = 161 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 22 : i32}, {id = 161 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 23 : i32}, {id = 161 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 24 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %136 = "neura.data_mov"(%104) {dfg_id = 124 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 16 : i32}, {id = 165 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 17 : i32}, {id = 165 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 18 : i32}, {id = 165 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 19 : i32}, {id = 165 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 20 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %137 = "neura.data_mov"(%132) {dfg_id = 179 : i32, mapping_locs = [{id = 167 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 11 : i32}, {id = 167 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 12 : i32}, {id = 167 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 13 : i32}, {id = 167 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 14 : i32}, {id = 167 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 15 : i32}, {id = 167 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 16 : i32}, {id = 167 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 17 : i32}, {id = 167 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 18 : i32}, {id = 167 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 19 : i32}, {id = 167 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 20 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %138 = neura.grant_predicate %136, %137 {dfg_id = 189 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 21 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     neura.ctrl_mov %138 -> %102 {dfg_id = 199 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 21 : i32}, {id = 4006 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 22 : i32}, {id = 4006 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 23 : i32}, {id = 4006 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 24 : i32}, {id = 4006 : i32, index_per_ii = 8 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 25 : i32}, {id = 4006 : i32, index_per_ii = 9 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 26 : i32}, {id = 4006 : i32, index_per_ii = 10 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 27 : i32}, {id = 4006 : i32, index_per_ii = 11 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 28 : i32}, {id = 4006 : i32, index_per_ii = 12 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 29 : i32}, {id = 4006 : i32, index_per_ii = 13 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 30 : i32}, {id = 4006 : i32, index_per_ii = 14 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 31 : i32}, {id = 4006 : i32, index_per_ii = 15 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 32 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %139 = "neura.data_mov"(%101) {dfg_id = 106 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}, {id = 8002 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 11 : i32}, {id = 8002 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 12 : i32}, {id = 8002 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 13 : i32}, {id = 8002 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 14 : i32}, {id = 8002 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 15 : i32}, {id = 8002 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 16 : i32}, {id = 8002 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 17 : i32}, {id = 8002 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 18 : i32}, {id = 8002 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 19 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %140 = "neura.data_mov"(%132) {dfg_id = 178 : i32, mapping_locs = [{id = 166 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 11 : i32}, {id = 15 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 12 : i32}, {id = 2 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 13 : i32}, {id = 1 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 14 : i32}, {id = 12 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 15 : i32}, {id = 8001 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 16 : i32}, {id = 8001 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 17 : i32}, {id = 8001 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 18 : i32}, {id = 8001 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 19 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %141 = neura.grant_predicate %139, %140 {dfg_id = 188 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 20 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %141 -> %99 {dfg_id = 198 : i32, mapping_locs = [{id = 25 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 20 : i32}, {id = 4005 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 21 : i32}, {id = 4005 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 22 : i32}, {id = 4005 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 23 : i32}, {id = 4005 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 24 : i32}, {id = 4005 : i32, index_per_ii = 8 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 25 : i32}, {id = 4005 : i32, index_per_ii = 9 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 26 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %142 = "neura.data_mov"(%98) {dfg_id = 103 : i32, mapping_locs = [{id = 19 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 12 : i32}, {id = 2000 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}, {id = 2000 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 14 : i32}, {id = 2000 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 15 : i32}, {id = 2000 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 16 : i32}, {id = 2000 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 17 : i32}, {id = 2000 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 18 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %143 = "neura.data_mov"(%132) {dfg_id = 177 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}, {id = 3 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 12 : i32}, {id = 2001 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}, {id = 2001 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 2001 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}, {id = 2001 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 16 : i32}, {id = 2001 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 17 : i32}, {id = 2001 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 18 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %144 = neura.grant_predicate %142, %143 {dfg_id = 187 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 19 : i32, x = 2 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %144 -> %96 {dfg_id = 197 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 19 : i32}, {id = 198 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 20 : i32}, {id = 198 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 21 : i32}, {id = 198 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 22 : i32}, {id = 198 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 23 : i32}, {id = 198 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 24 : i32}, {id = 198 : i32, index_per_ii = 8 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 25 : i32}, {id = 198 : i32, index_per_ii = 9 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 26 : i32}, {id = 198 : i32, index_per_ii = 10 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 27 : i32}, {id = 198 : i32, index_per_ii = 11 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 28 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %145 = "neura.data_mov"(%95) {dfg_id = 132 : i32, mapping_locs = [{id = 289 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 289 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}, {id = 289 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %146 = "neura.data_mov"(%132) {dfg_id = 176 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %147 = neura.grant_predicate %145, %146 {dfg_id = 186 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 12 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %147 -> %93 {dfg_id = 196 : i32, mapping_locs = [{id = 289 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 289 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}, {id = 289 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 289 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}, {id = 289 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 16 : i32}, {id = 289 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 17 : i32}, {id = 289 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 18 : i32}, {id = 289 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 19 : i32}, {id = 289 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 20 : i32}, {id = 289 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 21 : i32}, {id = 289 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 22 : i32}, {id = 289 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 23 : i32}, {id = 289 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 24 : i32}, {id = 289 : i32, index_per_ii = 8 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 25 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %148 = "neura.data_mov"(%92) {dfg_id = 135 : i32, mapping_locs = [{id = 322 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 11 : i32}, {id = 322 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 12 : i32}, {id = 322 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 13 : i32}, {id = 322 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 14 : i32}, {id = 322 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 15 : i32}, {id = 322 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 16 : i32}, {id = 322 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 17 : i32}, {id = 322 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 18 : i32}, {id = 322 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 19 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %149 = "neura.data_mov"(%132) {dfg_id = 175 : i32, mapping_locs = [{id = 165 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 11 : i32}, {id = 16 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 12 : i32}, {id = 28 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 13 : i32}, {id = 323 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 14 : i32}, {id = 323 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 15 : i32}, {id = 323 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 16 : i32}, {id = 323 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 17 : i32}, {id = 323 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 18 : i32}, {id = 323 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 19 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %150 = neura.grant_predicate %148, %149 {dfg_id = 185 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 20 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %150 -> %90 {dfg_id = 195 : i32, mapping_locs = [{id = 323 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 20 : i32}, {id = 323 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 21 : i32}, {id = 323 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 22 : i32}, {id = 323 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 23 : i32}, {id = 323 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 24 : i32}, {id = 323 : i32, index_per_ii = 8 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 25 : i32}, {id = 323 : i32, index_per_ii = 9 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 26 : i32}, {id = 323 : i32, index_per_ii = 10 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 27 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %151 = "neura.data_mov"(%89) {dfg_id = 137 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 13 : i32}, {id = 1 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 1 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}, {id = 1 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 16 : i32}, {id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 17 : i32}, {id = 1 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 18 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %152 = "neura.data_mov"(%132) {dfg_id = 174 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}, {id = 11 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 12 : i32}, {id = 3 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 13 : i32}, {id = 3 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 14 : i32}, {id = 3 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 15 : i32}, {id = 3 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 16 : i32}, {id = 3 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 17 : i32}, {id = 3 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 18 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %153 = neura.grant_predicate %151, %152 {dfg_id = 184 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 19 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %153 -> %87 {dfg_id = 194 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 19 : i32}, {id = 4004 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 20 : i32}, {id = 4004 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 21 : i32}, {id = 4004 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 22 : i32}, {id = 4004 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 23 : i32}, {id = 4004 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 24 : i32}, {id = 4004 : i32, index_per_ii = 8 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 25 : i32}, {id = 4004 : i32, index_per_ii = 9 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 26 : i32}, {id = 4004 : i32, index_per_ii = 10 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 27 : i32}, {id = 4004 : i32, index_per_ii = 11 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 28 : i32}, {id = 4004 : i32, index_per_ii = 12 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 29 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %154 = "neura.data_mov"(%86) {dfg_id = 113 : i32, mapping_locs = [{id = 18 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 16 : i32}, {id = 224 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 17 : i32}, {id = 224 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 18 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %155 = "neura.data_mov"(%132) {dfg_id = 173 : i32, mapping_locs = [{id = 164 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 11 : i32}, {id = 14 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 12 : i32}, {id = 18 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 13 : i32}, {id = 225 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 225 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}, {id = 225 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 16 : i32}, {id = 225 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 17 : i32}, {id = 225 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 18 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %156 = neura.grant_predicate %154, %155 {dfg_id = 183 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 19 : i32, x = 3 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %156 -> %84 {dfg_id = 193 : i32, mapping_locs = [{id = 21 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 19 : i32}, {id = 197 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 20 : i32}, {id = 197 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 21 : i32}, {id = 197 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 22 : i32}, {id = 197 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 23 : i32}, {id = 197 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 24 : i32}, {id = 197 : i32, index_per_ii = 8 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 25 : i32}, {id = 197 : i32, index_per_ii = 9 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 26 : i32}, {id = 197 : i32, index_per_ii = 10 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 27 : i32}, {id = 197 : i32, index_per_ii = 11 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 28 : i32}, {id = 197 : i32, index_per_ii = 12 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 29 : i32}, {id = 197 : i32, index_per_ii = 13 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 30 : i32}, {id = 197 : i32, index_per_ii = 14 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 31 : i32}, {id = 197 : i32, index_per_ii = 15 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 32 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %157 = "neura.data_mov"(%101) {dfg_id = 105 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}, {id = 160 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %158 = "neura.data_mov"(%130) {dfg_id = 150 : i32, mapping_locs = [{id = 161 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}, {id = 161 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %159 = neura.grant_predicate %157, %158 {dfg_id = 158 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 12 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %160 = "neura.data_mov"(%92) {dfg_id = 134 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %161 = "neura.data_mov"(%130) {dfg_id = 149 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}, {id = 20 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %162 = neura.grant_predicate %160, %161 {dfg_id = 157 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 12 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %163 = "neura.data_mov"(%98) {dfg_id = 102 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %164 = "neura.data_mov"(%130) {dfg_id = 148 : i32, mapping_locs = [{id = 163 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 10 : i32}, {id = 14 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}, {id = 193 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %165 = neura.grant_predicate %163, %164 {dfg_id = 156 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 13 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %166 = "neura.data_mov"(%89) {dfg_id = 136 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %167 = "neura.data_mov"(%130) {dfg_id = 147 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}, {id = 4001 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}, {id = 4001 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 4001 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %168 = neura.grant_predicate %166, %167 {dfg_id = 155 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 14 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %169 = "neura.data_mov"(%86) {dfg_id = 112 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 16 : i32}, {id = 192 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 17 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %170 = "neura.data_mov"(%130) {dfg_id = 146 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}, {id = 3 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}, {id = 7 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 12 : i32}, {id = 193 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}, {id = 193 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 193 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}, {id = 193 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 16 : i32}, {id = 193 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 17 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %171 = neura.grant_predicate %169, %170 {dfg_id = 154 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 18 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %172 = "neura.data_mov"(%95) {dfg_id = 131 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}, {id = 288 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %173 = "neura.data_mov"(%130) {dfg_id = 145 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %174 = neura.grant_predicate %172, %173 {dfg_id = 153 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %175 = "neura.data_mov"(%159) {dfg_id = 172 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %176 = "neura.add"(%175) {dfg_id = 182 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 13 : i32, x = 1 : i32, y = 1 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %177 = "neura.data_mov"(%176) {dfg_id = 192 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 13 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %178 = "neura.data_mov"(%162) {dfg_id = 171 : i32, mapping_locs = [{id = 31 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 12 : i32}, {id = 288 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %179 = "neura.icmp"(%177, %178) <{cmpType = "eq"}> {dfg_id = 201 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 14 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %180 = "neura.data_mov"(%179) {dfg_id = 207 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 14 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %181 = "neura.not"(%180) {dfg_id = 213 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 15 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %182 = "neura.data_mov"(%176) {dfg_id = 191 : i32, mapping_locs = [{id = 164 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 13 : i32}, {id = 164 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 14 : i32}, {id = 164 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 15 : i32}, {id = 164 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 16 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %183 = "neura.data_mov"(%181) {dfg_id = 225 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 15 : i32}, {id = 160 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 16 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %184 = neura.grant_predicate %182, %183 {dfg_id = 232 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 17 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %184 -> %78 {dfg_id = 240 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 17 : i32}, {id = 4001 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 18 : i32}, {id = 4001 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 19 : i32}, {id = 4001 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 20 : i32}, {id = 4001 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 21 : i32}, {id = 4001 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 22 : i32}, {id = 4001 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 23 : i32}, {id = 4001 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 24 : i32}, {id = 4001 : i32, index_per_ii = 8 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 25 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %185 = "neura.data_mov"(%165) {dfg_id = 168 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 13 : i32}, {id = 324 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 14 : i32}, {id = 324 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 15 : i32}, {id = 324 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 16 : i32}, {id = 324 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 17 : i32}, {id = 324 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 18 : i32}, {id = 324 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 19 : i32}, {id = 324 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 20 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %186 = "neura.data_mov"(%181) {dfg_id = 224 : i32, mapping_locs = [{id = 30 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 15 : i32}, {id = 41 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 16 : i32}, {id = 45 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 17 : i32}, {id = 325 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 18 : i32}, {id = 325 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 19 : i32}, {id = 325 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 20 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %187 = neura.grant_predicate %185, %186 {dfg_id = 231 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 21 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %187 -> %75 {dfg_id = 239 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 21 : i32}, {id = 195 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 22 : i32}, {id = 195 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 23 : i32}, {id = 195 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 24 : i32}, {id = 195 : i32, index_per_ii = 8 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 25 : i32}, {id = 195 : i32, index_per_ii = 9 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 26 : i32}, {id = 195 : i32, index_per_ii = 10 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 27 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %188 = "neura.data_mov"(%171) {dfg_id = 164 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 18 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %189 = "neura.data_mov"(%181) {dfg_id = 223 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 15 : i32}, {id = 28 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 16 : i32}, {id = 320 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 17 : i32}, {id = 320 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 18 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %190 = neura.grant_predicate %188, %189 {dfg_id = 230 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 19 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %190 -> %72 {dfg_id = 238 : i32, mapping_locs = [{id = 31 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 19 : i32}, {id = 29 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 20 : i32}, {id = 162 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 21 : i32}, {id = 162 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 22 : i32}, {id = 162 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 23 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %191 = "neura.data_mov"(%174) {dfg_id = 162 : i32, mapping_locs = [{id = 292 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 11 : i32}, {id = 292 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 12 : i32}, {id = 292 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 13 : i32}, {id = 292 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 14 : i32}, {id = 292 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 15 : i32}, {id = 292 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 16 : i32}, {id = 292 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 17 : i32}, {id = 292 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 18 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %192 = "neura.data_mov"(%181) {dfg_id = 222 : i32, mapping_locs = [{id = 293 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 15 : i32}, {id = 293 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 16 : i32}, {id = 293 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 17 : i32}, {id = 293 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 18 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %193 = neura.grant_predicate %191, %192 {dfg_id = 229 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 19 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %193 -> %69 {dfg_id = 237 : i32, mapping_locs = [{id = 290 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 19 : i32}, {id = 290 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 20 : i32}, {id = 290 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 21 : i32}, {id = 290 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 22 : i32}, {id = 290 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 23 : i32}, {id = 290 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 24 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %194 = "neura.data_mov"(%162) {dfg_id = 170 : i32, mapping_locs = [{id = 321 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 321 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}, {id = 321 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 321 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %195 = "neura.data_mov"(%181) {dfg_id = 221 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 15 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %196 = neura.grant_predicate %194, %195 {dfg_id = 228 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 16 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %196 -> %66 {dfg_id = 236 : i32, mapping_locs = [{id = 321 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 16 : i32}, {id = 321 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 17 : i32}, {id = 321 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 18 : i32}, {id = 321 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 19 : i32}, {id = 321 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 20 : i32}, {id = 321 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 21 : i32}, {id = 321 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 22 : i32}, {id = 321 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 23 : i32}, {id = 321 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 24 : i32}, {id = 321 : i32, index_per_ii = 8 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 25 : i32}, {id = 321 : i32, index_per_ii = 9 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 26 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %197 = "neura.data_mov"(%168) {dfg_id = 166 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 14 : i32}, {id = 16 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 15 : i32}, {id = 295 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 16 : i32}, {id = 295 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 17 : i32}, {id = 295 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 18 : i32}, {id = 295 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 19 : i32}, {id = 295 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 20 : i32}, {id = 295 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 21 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %198 = "neura.data_mov"(%181) {dfg_id = 220 : i32, mapping_locs = [{id = 296 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 15 : i32}, {id = 296 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 16 : i32}, {id = 296 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 17 : i32}, {id = 296 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 18 : i32}, {id = 296 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 19 : i32}, {id = 296 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 20 : i32}, {id = 296 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 21 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %199 = neura.grant_predicate %197, %198 {dfg_id = 227 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 22 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %199 -> %63 {dfg_id = 235 : i32, mapping_locs = [{id = 27 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 22 : i32}, {id = 25 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 23 : i32}, {id = 4007 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 24 : i32}, {id = 4007 : i32, index_per_ii = 8 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 25 : i32}, {id = 4007 : i32, index_per_ii = 9 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 26 : i32}, {id = 4007 : i32, index_per_ii = 10 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 27 : i32}, {id = 4007 : i32, index_per_ii = 11 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 28 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %200 = "neura.data_mov"(%165) {dfg_id = 167 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 13 : i32}, {id = 160 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 14 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %201 = "neura.data_mov"(%179) {dfg_id = 206 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 14 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %202 = neura.grant_predicate %200, %201 {dfg_id = 212 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 15 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %203 = "neura.data_mov"(%168) {dfg_id = 165 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 14 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %204 = "neura.data_mov"(%179) {dfg_id = 205 : i32, mapping_locs = [{id = 27 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 14 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %205 = neura.grant_predicate %203, %204 {dfg_id = 211 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 15 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %206 = "neura.data_mov"(%171) {dfg_id = 163 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 18 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %207 = "neura.data_mov"(%179) {dfg_id = 204 : i32, mapping_locs = [{id = 30 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 14 : i32}, {id = 41 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 15 : i32}, {id = 45 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 16 : i32}, {id = 33 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 17 : i32}, {id = 193 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 18 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %208 = neura.grant_predicate %206, %207 {dfg_id = 210 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 19 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %209 = "neura.data_mov"(%174) {dfg_id = 161 : i32, mapping_locs = [{id = 290 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 11 : i32}, {id = 290 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 12 : i32}, {id = 290 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 13 : i32}, {id = 290 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 14 : i32}, {id = 290 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 15 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %210 = "neura.data_mov"(%179) {dfg_id = 203 : i32, mapping_locs = [{id = 291 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 14 : i32}, {id = 291 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 15 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %211 = neura.grant_predicate %209, %210 {dfg_id = 209 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 16 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %212 = "neura.data_mov"(%162) {dfg_id = 169 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}, {id = 320 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}, {id = 320 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 14 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %213 = "neura.data_mov"(%179) {dfg_id = 202 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 14 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 14 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %214 = neura.grant_predicate %212, %213 {dfg_id = 208 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 15 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %215 = "neura.data_mov"(%202) {dfg_id = 219 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 15 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %216 = "neura.add"(%215) {dfg_id = 226 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 16 : i32, x = 1 : i32, y = 1 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %217 = "neura.data_mov"(%216) {dfg_id = 234 : i32, mapping_locs = [{id = 162 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 16 : i32}, {id = 162 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 17 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %218 = "neura.data_mov"(%205) {dfg_id = 218 : i32, mapping_locs = [{id = 24 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 15 : i32}, {id = 29 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 16 : i32}, {id = 160 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 17 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %219 = "neura.icmp"(%217, %218) <{cmpType = "eq"}> {dfg_id = 241 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 18 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %220 = "neura.data_mov"(%219) {dfg_id = 244 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 18 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %221 = "neura.not"(%220) {dfg_id = 246 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 19 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %222 = "neura.data_mov"(%216) {dfg_id = 233 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 16 : i32}, {id = 195 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 17 : i32}, {id = 195 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 18 : i32}, {id = 195 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 19 : i32}, {id = 195 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 20 : i32}, {id = 195 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 21 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %223 = "neura.data_mov"(%221) {dfg_id = 252 : i32, mapping_locs = [{id = 164 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 19 : i32}, {id = 14 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 20 : i32}, {id = 192 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 21 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %224 = neura.grant_predicate %222, %223 {dfg_id = 258 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 22 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %224 -> %60 {dfg_id = 264 : i32, mapping_locs = [{id = 193 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 22 : i32}, {id = 193 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 23 : i32}, {id = 193 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 24 : i32}, {id = 193 : i32, index_per_ii = 8 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 25 : i32}, {id = 193 : i32, index_per_ii = 9 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 26 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %225 = "neura.data_mov"(%208) {dfg_id = 216 : i32, mapping_locs = [{id = 19 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 19 : i32}, {id = 5 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 20 : i32}, {id = 2 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 21 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %226 = "neura.data_mov"(%221) {dfg_id = 251 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 19 : i32}, {id = 2 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 20 : i32}, {id = 0 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 21 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %227 = neura.grant_predicate %225, %226 {dfg_id = 257 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 22 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %227 -> %57 {dfg_id = 263 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 22 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %228 = "neura.data_mov"(%211) {dfg_id = 215 : i32, mapping_locs = [{id = 294 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 16 : i32}, {id = 294 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 17 : i32}, {id = 294 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 18 : i32}, {id = 294 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 19 : i32}, {id = 294 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 20 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %229 = "neura.data_mov"(%221) {dfg_id = 250 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 19 : i32}, {id = 16 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 20 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %230 = neura.grant_predicate %228, %229 {dfg_id = 256 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 21 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %230 -> %54 {dfg_id = 262 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 21 : i32}, {id = 288 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 22 : i32}, {id = 288 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 23 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %231 = "neura.data_mov"(%214) {dfg_id = 214 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 15 : i32}, {id = 194 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 16 : i32}, {id = 194 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 17 : i32}, {id = 194 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 18 : i32}, {id = 194 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 19 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %232 = "neura.data_mov"(%221) {dfg_id = 249 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 19 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %233 = neura.grant_predicate %231, %232 {dfg_id = 255 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 20 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %233 -> %51 {dfg_id = 261 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 20 : i32}, {id = 322 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 21 : i32}, {id = 322 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 22 : i32}, {id = 322 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 23 : i32}, {id = 322 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 24 : i32}, {id = 322 : i32, index_per_ii = 8 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 25 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %234 = "neura.data_mov"(%205) {dfg_id = 217 : i32, mapping_locs = [{id = 8001 : i32, index_per_ii = 15 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}, {id = 24 : i32, index_per_ii = 16 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 16 : i32}, {id = 291 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 17 : i32}, {id = 291 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 18 : i32}, {id = 291 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 19 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %235 = "neura.data_mov"(%221) {dfg_id = 248 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 19 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %236 = neura.grant_predicate %234, %235 {dfg_id = 254 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 20 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %236 -> %48 {dfg_id = 260 : i32, mapping_locs = [{id = 27 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 20 : i32}, {id = 25 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 21 : i32}, {id = 4002 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 22 : i32}, {id = 4002 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 23 : i32}, {id = 4002 : i32, index_per_ii = 7 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 24 : i32}, {id = 4002 : i32, index_per_ii = 8 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 25 : i32}, {id = 4002 : i32, index_per_ii = 9 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 26 : i32}, {id = 4002 : i32, index_per_ii = 10 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 27 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %237 = "neura.data_mov"(%219) {dfg_id = 242 : i32, mapping_locs = [{id = 162 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 18 : i32}, {id = 162 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 19 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %238 = "neura.data_mov"(%219) {dfg_id = 243 : i32, mapping_locs = [{id = 163 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 18 : i32}, {id = 163 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 19 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %239 = neura.grant_predicate %237, %238 {dfg_id = 245 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 20 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
// MAPPING-NEXT:     %240 = "neura.data_mov"(%41) {dfg_id = 72 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 17 : i32}, {id = 15 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 18 : i32}, {id = 1000 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 19 : i32}, {id = 1000 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 20 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %241 = "neura.data_mov"(%239) {dfg_id = 247 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 20 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %242 = "neura.phi"(%240, %241) {dfg_id = 253 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 21 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %243 = "neura.data_mov"(%242) {dfg_id = 259 : i32, mapping_locs = [{id = 1000 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 21 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     neura.return_void %243 : !neura.data<i1, i1> {dfg_id = 265 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 22 : i32, x = 1 : i32, y = 0 : i32}]}
// MAPPING-NEXT:     neura.yield {dfg_id = 26 : i32}
// MAPPING-NEXT:   }


// YAML: array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   compiled_ii: 17
// YAML-NEXT:   cores:
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "0"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "LOAD"
// YAML-NEXT:                   id: 138
// YAML-NEXT:                   time_step: 17
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 184
// YAML-NEXT:                   time_step: 19
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 6
// YAML-NEXT:                   time_step: 4
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 43
// YAML-NEXT:                   time_step: 4
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 251
// YAML-NEXT:                   time_step: 21
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 52
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 257
// YAML-NEXT:                   time_step: 22
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 8
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 0
// YAML-NEXT:                   time_step: 8
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "arg0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 9
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 58
// YAML-NEXT:                   time_step: 9
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 10
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ZEXT"
// YAML-NEXT:                   id: 71
// YAML-NEXT:                   time_step: 10
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 11
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 1070001
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 13
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 174
// YAML-NEXT:                   time_step: 13
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 14
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 1780002
// YAML-NEXT:                   time_step: 14
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 137
// YAML-NEXT:                   time_step: 14
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 15
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "NOT"
// YAML-NEXT:                   id: 45
// YAML-NEXT:                   time_step: 15
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 16
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 53
// YAML-NEXT:                   time_step: 16
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "1"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "LOAD"
// YAML-NEXT:                   id: 140
// YAML-NEXT:                   time_step: 17
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 142
// YAML-NEXT:                   time_step: 18
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "FMUL_FADD"
// YAML-NEXT:                   id: 152
// YAML-NEXT:                   time_step: 19
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 72
// YAML-NEXT:                   time_step: 19
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 2510001
// YAML-NEXT:                   time_step: 20
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI"
// YAML-NEXT:                   id: 253
// YAML-NEXT:                   time_step: 21
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 2160002
// YAML-NEXT:                   time_step: 21
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "RETURN_VOID"
// YAML-NEXT:                   id: 265
// YAML-NEXT:                   time_step: 22
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 9
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 111
// YAML-NEXT:                   time_step: 9
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 11
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 1460001
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 12
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 107
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 1770001
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 13
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 1780001
// YAML-NEXT:                   time_step: 13
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 16
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GEP"
// YAML-NEXT:                   id: 120
// YAML-NEXT:                   time_step: 16
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "2"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "LOAD"
// YAML-NEXT:                   id: 139
// YAML-NEXT:                   time_step: 18
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 187
// YAML-NEXT:                   time_step: 19
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 128
// YAML-NEXT:                   time_step: 19
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "STORE"
// YAML-NEXT:                   id: 181
// YAML-NEXT:                   time_step: 20
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 2160001
// YAML-NEXT:                   time_step: 20
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 12
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 1460002
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 13
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 103
// YAML-NEXT:                   time_step: 13
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 177
// YAML-NEXT:                   time_step: 13
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "4"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 66
// YAML-NEXT:                   time_step: 17
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CONSTANT"
// YAML-NEXT:                   id: 3
// YAML-NEXT:                   time_step: 1
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "arg1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 240
// YAML-NEXT:                   time_step: 18
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ICMP_SGT"
// YAML-NEXT:                   id: 35
// YAML-NEXT:                   time_step: 2
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "#0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "AND"
// YAML-NEXT:                   id: 42
// YAML-NEXT:                   time_step: 3
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 194
// YAML-NEXT:                   time_step: 20
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 46
// YAML-NEXT:                   time_step: 4
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 198
// YAML-NEXT:                   time_step: 21
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$5"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 55
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 480000
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 199
// YAML-NEXT:                   time_step: 22
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$6"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 260
// YAML-NEXT:                   time_step: 22
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 6
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 68
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 7
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 235
// YAML-NEXT:                   time_step: 24
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$7"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 9
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 80
// YAML-NEXT:                   time_step: 9
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 10
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 95
// YAML-NEXT:                   time_step: 10
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$5"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 11
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 84
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 147
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 12
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 101
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$7"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 1740001
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 13
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 123
// YAML-NEXT:                   time_step: 13
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 86
// YAML-NEXT:                   time_step: 13
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 14
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 155
// YAML-NEXT:                   time_step: 14
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 15
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GEP"
// YAML-NEXT:                   id: 96
// YAML-NEXT:                   time_step: 15
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 1780003
// YAML-NEXT:                   time_step: 15
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 16
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 117
// YAML-NEXT:                   time_step: 16
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$6"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "5"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 232
// YAML-NEXT:                   time_step: 17
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 124
// YAML-NEXT:                   time_step: 17
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$5"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 218
// YAML-NEXT:                   time_step: 17
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ICMP_EQ"
// YAML-NEXT:                   id: 241
// YAML-NEXT:                   time_step: 18
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 720001
// YAML-NEXT:                   time_step: 18
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "NOT"
// YAML-NEXT:                   id: 246
// YAML-NEXT:                   time_step: 19
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 245
// YAML-NEXT:                   time_step: 20
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 2520000
// YAML-NEXT:                   time_step: 20
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 2500000
// YAML-NEXT:                   time_step: 20
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 189
// YAML-NEXT:                   time_step: 21
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$5"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$7"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 238
// YAML-NEXT:                   time_step: 21
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 56
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 6
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ZEXT"
// YAML-NEXT:                   id: 69
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 610001
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 7
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 81
// YAML-NEXT:                   time_step: 7
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 8
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 97
// YAML-NEXT:                   time_step: 8
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 9
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ADD"
// YAML-NEXT:                   id: 118
// YAML-NEXT:                   time_step: 9
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "#1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 10
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ICMP_EQ"
// YAML-NEXT:                   id: 141
// YAML-NEXT:                   time_step: 10
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 11
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "NOT"
// YAML-NEXT:                   id: 159
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$7"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$6"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$5"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 105
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 1480000
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 12
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 158
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 860001
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 1780000
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$6"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 1750000
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$5"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 1730000
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 13
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ADD"
// YAML-NEXT:                   id: 182
// YAML-NEXT:                   time_step: 13
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "#1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 14
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 190
// YAML-NEXT:                   time_step: 14
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 167
// YAML-NEXT:                   time_step: 14
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 15
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 212
// YAML-NEXT:                   time_step: 15
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 1660001
// YAML-NEXT:                   time_step: 15
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 16
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ADD"
// YAML-NEXT:                   id: 226
// YAML-NEXT:                   time_step: 16
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "#1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 225
// YAML-NEXT:                   time_step: 16
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "6"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GEP"
// YAML-NEXT:                   id: 119
// YAML-NEXT:                   time_step: 17
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 233
// YAML-NEXT:                   time_step: 17
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 154
// YAML-NEXT:                   time_step: 18
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 1280000
// YAML-NEXT:                   time_step: 18
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 204
// YAML-NEXT:                   time_step: 18
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 210
// YAML-NEXT:                   time_step: 19
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 255
// YAML-NEXT:                   time_step: 20
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 197
// YAML-NEXT:                   time_step: 20
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$6"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 193
// YAML-NEXT:                   time_step: 20
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$5"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 2
// YAML-NEXT:                   time_step: 4
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "arg1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 252
// YAML-NEXT:                   time_step: 21
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 258
// YAML-NEXT:                   time_step: 22
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 239
// YAML-NEXT:                   time_step: 22
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 7
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 61
// YAML-NEXT:                   time_step: 7
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 8
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 90
// YAML-NEXT:                   time_step: 8
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 9
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 110
// YAML-NEXT:                   time_step: 9
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 10
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 67
// YAML-NEXT:                   time_step: 10
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 11
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 79
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 1490001
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 12
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 94
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$6"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 148
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 13
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 156
// YAML-NEXT:                   time_step: 13
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 1730001
// YAML-NEXT:                   time_step: 13
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 146
// YAML-NEXT:                   time_step: 13
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 16
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 98
// YAML-NEXT:                   time_step: 16
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$5"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 214
// YAML-NEXT:                   time_step: 16
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 3
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "7"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 113
// YAML-NEXT:                   time_step: 17
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 183
// YAML-NEXT:                   time_step: 19
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 14
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 173
// YAML-NEXT:                   time_step: 14
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "8"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CONSTANT"
// YAML-NEXT:                   id: 5
// YAML-NEXT:                   time_step: 0
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "arg2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ICMP_SGT"
// YAML-NEXT:                   id: 36
// YAML-NEXT:                   time_step: 1
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "#0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "AND"
// YAML-NEXT:                   id: 40
// YAML-NEXT:                   time_step: 2
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 188
// YAML-NEXT:                   time_step: 20
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 2600001
// YAML-NEXT:                   time_step: 21
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 510001
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 6
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 48
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 2350001
// YAML-NEXT:                   time_step: 23
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 11
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 106
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 15
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 211
// YAML-NEXT:                   time_step: 15
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 16
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "NOT"
// YAML-NEXT:                   id: 54
// YAML-NEXT:                   time_step: 16
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 178
// YAML-NEXT:                   time_step: 16
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 2170000
// YAML-NEXT:                   time_step: 16
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "9"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CONSTANT"
// YAML-NEXT:                   id: 1
// YAML-NEXT:                   time_step: 0
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "arg0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 217
// YAML-NEXT:                   time_step: 17
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ICMP_SGT"
// YAML-NEXT:                   id: 34
// YAML-NEXT:                   time_step: 1
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "#0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 229
// YAML-NEXT:                   time_step: 19
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$5"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 254
// YAML-NEXT:                   time_step: 20
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 2380001
// YAML-NEXT:                   time_step: 20
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 256
// YAML-NEXT:                   time_step: 21
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$6"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 227
// YAML-NEXT:                   time_step: 22
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$7"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 6
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 510002
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 7
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 82
// YAML-NEXT:                   time_step: 7
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 8
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 99
// YAML-NEXT:                   time_step: 8
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 9
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 121
// YAML-NEXT:                   time_step: 9
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 11
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 153
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 12
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 186
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 13
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 1750001
// YAML-NEXT:                   time_step: 13
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 171
// YAML-NEXT:                   time_step: 13
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 14
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ICMP_EQ"
// YAML-NEXT:                   id: 201
// YAML-NEXT:                   time_step: 14
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 15
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "NOT"
// YAML-NEXT:                   id: 213
// YAML-NEXT:                   time_step: 15
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$5"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$8"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 16
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 209
// YAML-NEXT:                   time_step: 16
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$6"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 2230000
// YAML-NEXT:                   time_step: 16
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 166
// YAML-NEXT:                   time_step: 16
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$7"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 2180001
// YAML-NEXT:                   time_step: 16
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "10"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 223
// YAML-NEXT:                   time_step: 17
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 2040003
// YAML-NEXT:                   time_step: 17
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 224
// YAML-NEXT:                   time_step: 18
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$5"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 230
// YAML-NEXT:                   time_step: 19
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 185
// YAML-NEXT:                   time_step: 20
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 231
// YAML-NEXT:                   time_step: 21
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$5"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   id: 261
// YAML-NEXT:                   time_step: 21
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 6
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 4
// YAML-NEXT:                   time_step: 6
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "arg2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 7
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 57
// YAML-NEXT:                   time_step: 7
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 8
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ZEXT"
// YAML-NEXT:                   id: 70
// YAML-NEXT:                   time_step: 8
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 9
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 83
// YAML-NEXT:                   time_step: 9
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 10
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 100
// YAML-NEXT:                   time_step: 10
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 11
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI_START"
// YAML-NEXT:                   id: 122
// YAML-NEXT:                   time_step: 11
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$2"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 12
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 157
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 14
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 175
// YAML-NEXT:                   time_step: 14
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 168
// YAML-NEXT:                   time_step: 14
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$4"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 15
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 208
// YAML-NEXT:                   time_step: 15
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 16
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 228
// YAML-NEXT:                   time_step: 16
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 3
// YAML-NEXT:       core_id: "13"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 15
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 2040001
// YAML-NEXT:                   time_step: 15
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 16
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 2240001
// YAML-NEXT:                   time_step: 16
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 3
// YAML-NEXT:       core_id: "14"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 2240002
// YAML-NEXT:                   time_step: 17
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 16
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 2040002
// YAML-NEXT:                   time_step: 16
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"


// ASM: # Compiled II: 17
// ASM: PE(0,0):
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [NORTH, RED] -> [EAST, RED] (t=17, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$1], [$3] -> [NORTH, RED] (t=19, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#0] -> [NORTH, RED] (t=4, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$2] (t=4, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$0] (t=21, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$1] (t=5, inv_iters=0)
// ASM-NEXT:   GRANT_PREDICATE, [EAST, RED], [$0] -> [NORTH, RED] (t=22, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [arg0] -> [$0] (t=8, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=8)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$1] -> [$0] (t=9, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=9)
// ASM-NEXT: {
// ASM-NEXT:   ZEXT, [$0] -> [NORTH, RED] (t=10, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=10)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [EAST, RED] (t=11, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=11)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$3] (t=13, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=13)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [NORTH, RED] (t=14, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$1] (t=14, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=14)
// ASM-NEXT: {
// ASM-NEXT:   NOT, [$2] -> [$0] (t=15, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=15)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [$0] -> [NORTH, RED] (t=16, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=16)
// ASM: PE(1,0):
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [$0] -> [$1] (t=17, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$0] (t=18, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   FMUL_FADD, [$0], [$1], [EAST, RED] -> [EAST, RED] (t=19, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$0] (t=19, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [WEST, RED] (t=20, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   PHI, [$0], [NORTH, RED] -> [$0] (t=21, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [WEST, RED] (t=21, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   RETURN_VOID, [$0] (t=22, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$1] (t=9, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=9)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [EAST, RED] (t=11, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=11)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$0] (t=12, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [EAST, RED] (t=12, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=12)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [WEST, RED] (t=13, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=13)
// ASM-NEXT: {
// ASM-NEXT:   GEP, [$0], [$1] -> [$0] (t=16, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=16)
// ASM: PE(2,0):
// ASM-NEXT: {
// ASM-NEXT:   LOAD, [NORTH, RED] -> [WEST, RED] (t=18, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$1] -> [NORTH, RED] (t=19, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$0] (t=19, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   STORE, [WEST, RED], [$0] (t=20, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [WEST, RED] (t=20, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [NORTH, RED] (t=12, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=12)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$0] (t=13, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$1] (t=13, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=13)
// ASM: PE(0,1):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [SOUTH, RED], [NORTH, RED] -> [EAST, RED] (t=17, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [arg1] -> [$0] (t=1, inv_iters=0)
// ASM-NEXT:   CTRL_MOV, [EAST, RED] -> [$1] (t=18, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_SGT, [$0], [#0] -> [$0] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   AND, [NORTH, RED], [$0] -> [$0], [SOUTH, RED] (t=3, inv_iters=0)
// ASM-NEXT:   CTRL_MOV, [SOUTH, RED] -> [$4] (t=20, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [$0] -> [SOUTH, RED], [NORTH, RED], [EAST, RED], [$0], [$2] (t=4, inv_iters=0)
// ASM-NEXT:   CTRL_MOV, [NORTH, RED] -> [$5] (t=21, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [SOUTH, RED], [$0] -> [$0], [EAST, RED] (t=5, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [$2] -> [NORTH, RED] (t=5, inv_iters=0)
// ASM-NEXT:   CTRL_MOV, [EAST, RED] -> [$6] (t=22, inv_iters=1)
// ASM-NEXT:   CTRL_MOV, [NORTH, RED] -> [$2] (t=22, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [SOUTH, RED] -> [EAST, RED], [$0] (t=6, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=6)
// ASM-NEXT: {
// ASM-NEXT:   CTRL_MOV, [NORTH, RED] -> [$7] (t=24, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=7)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$1] -> [$3], [$0] (t=9, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=9)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$5] -> [SOUTH, RED], [NORTH, RED], [EAST, RED] (t=10, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=10)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [SOUTH, RED], [$2] -> [$0] (t=11, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$1] (t=11, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=11)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$7] -> [$0] (t=12, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [SOUTH, RED] (t=12, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=12)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$4] -> [SOUTH, RED], [$0] (t=13, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$2] (t=13, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=13)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$1] -> [EAST, RED], [NORTH, RED] (t=14, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=14)
// ASM-NEXT: {
// ASM-NEXT:   GEP, [$2], [$3] -> [$0] (t=15, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [SOUTH, RED] -> [NORTH, RED] (t=15, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=15)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$6] -> [SOUTH, RED], [EAST, RED] (t=16, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=16)
// ASM: PE(1,1):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$4], [$0] -> [WEST, RED] (t=17, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$5] (t=17, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$0] (t=17, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_EQ, [$2], [$0] -> [$0], [$2], [$3] (t=18, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [SOUTH, RED] (t=18, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   NOT, [$0] -> [$4], [SOUTH, RED], [$0], [EAST, RED], [NORTH, RED] (t=19, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$2], [$3] -> [SOUTH, RED] (t=20, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [$4] -> [EAST, RED] (t=20, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [$0] -> [NORTH, RED] (t=20, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$5], [$7] -> [WEST, RED] (t=21, inv_iters=1)
// ASM-NEXT:   CTRL_MOV, [NORTH, RED] -> [$2] (t=21, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [EAST, RED], [WEST, RED] -> [$0] (t=5, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-NEXT: {
// ASM-NEXT:   ZEXT, [$0] -> [NORTH, RED] (t=6, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [EAST, RED] (t=6, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=6)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [WEST, RED], [$2] -> [EAST, RED], [$0] (t=7, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=7)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$1] -> [SOUTH, RED], [EAST, RED], [$0] (t=8, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=8)
// ASM-NEXT: {
// ASM-NEXT:   ADD, [$0], [#1] -> [$0], [$2] (t=9, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=9)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_EQ, [$0], [NORTH, RED] -> [$0], [$1], [EAST, RED], [$3], [WEST, RED], [SOUTH, RED], [NORTH, RED] (t=10, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=10)
// ASM-NEXT: {
// ASM-NEXT:   NOT, [$0] -> [$3], [$7], [$6], [SOUTH, RED], [NORTH, RED], [$5], [WEST, RED], [$4] (t=11, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$0] (t=11, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [$3] -> [EAST, RED] (t=11, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=11)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$1] -> [$0] (t=12, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [WEST, RED] (t=12, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [$6] -> [SOUTH, RED] (t=12, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [$5] -> [NORTH, RED] (t=12, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [$4] -> [EAST, RED] (t=12, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=12)
// ASM-NEXT: {
// ASM-NEXT:   ADD, [$0], [#1] -> [NORTH, RED], [$4] (t=13, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=13)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$2], [$3] -> [$1] (t=14, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$0] (t=14, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=14)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [NORTH, RED] -> [$0] (t=15, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [NORTH, RED] (t=15, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=15)
// ASM-NEXT: {
// ASM-NEXT:   ADD, [$0], [#1] -> [$2], [EAST, RED] (t=16, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$0] (t=16, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=16)
// ASM: PE(2,1):
// ASM-NEXT: {
// ASM-NEXT:   GEP, [$3], [$4] -> [SOUTH, RED], [$4] (t=17, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$3] (t=17, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$1] -> [NORTH, RED], [$0] (t=18, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [$4] -> [SOUTH, RED] (t=18, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$1] (t=18, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$1] -> [SOUTH, RED] (t=19, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$2], [WEST, RED] -> [NORTH, RED] (t=20, inv_iters=1)
// ASM-NEXT:   CTRL_MOV, [SOUTH, RED] -> [$6] (t=20, inv_iters=1)
// ASM-NEXT:   CTRL_MOV, [EAST, RED] -> [$5] (t=20, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [arg1] -> [WEST, RED] (t=4, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$0] (t=21, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$3], [$0] -> [$1] (t=22, inv_iters=1)
// ASM-NEXT:   CTRL_MOV, [NORTH, RED] -> [$3] (t=22, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$0] (t=7, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=7)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$2] (t=8, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=8)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$4] (t=9, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=9)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$1] -> [$0] (t=10, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=10)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$3] -> [WEST, RED], [$0] (t=11, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [NORTH, RED] (t=11, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=11)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$6] -> [$3], [SOUTH, RED], [$0] (t=12, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$1] (t=12, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=12)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$1] -> [NORTH, RED], [WEST, RED] (t=13, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [EAST, RED] (t=13, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [SOUTH, RED] -> [$1] (t=13, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=13)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$2], [$5] -> [EAST, RED], [$0] (t=16, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$2] (t=16, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=16)
// ASM: PE(3,1):
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$0] (t=17, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$1] -> [WEST, RED] (t=19, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$1] (t=14, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=14)
// ASM: PE(0,2):
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [arg2] -> [$0] (t=0, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_SGT, [$0], [#0] -> [$0] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   AND, [EAST, RED], [$0] -> [SOUTH, RED] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$2], [$1] -> [SOUTH, RED] (t=20, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   CTRL_MOV, [EAST, RED] -> [SOUTH, RED] (t=21, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [SOUTH, RED] -> [EAST, RED] (t=5, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [SOUTH, RED] -> [$0] (t=6, inv_iters=0)
// ASM-NEXT:   CTRL_MOV, [EAST, RED] -> [SOUTH, RED] (t=23, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=6)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [SOUTH, RED] -> [$2] (t=11, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=11)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [SOUTH, RED], [EAST, RED] -> [EAST, RED], [$1] (t=15, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=15)
// ASM-NEXT: {
// ASM-NEXT:   NOT, [$0] -> [SOUTH, RED] (t=16, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [SOUTH, RED] -> [$1] (t=16, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [$1] -> [EAST, RED] (t=16, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=16)
// ASM: PE(1,2):
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [arg0] -> [$0] (t=0, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$3] (t=17, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_SGT, [$0], [#0] -> [WEST, RED] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$4], [$5] -> [$2] (t=19, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$3], [SOUTH, RED] -> [WEST, RED] (t=20, inv_iters=1)
// ASM-NEXT:   CTRL_MOV, [EAST, RED] -> [SOUTH, RED] (t=20, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$6], [SOUTH, RED] -> [$0] (t=21, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$7], [$8] -> [WEST, RED] (t=22, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [EAST, RED] (t=6, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=6)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [SOUTH, RED], [$0] -> [$0] (t=7, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=7)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$2] -> [$0] (t=8, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=8)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$1] -> [SOUTH, RED], [$1], [$0] (t=9, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=9)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [SOUTH, RED] -> [$4], [$2] (t=11, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=11)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$1], [SOUTH, RED] -> [$1] (t=12, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=12)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [SOUTH, RED] -> [EAST, RED] (t=13, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$0] (t=13, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=13)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_EQ, [SOUTH, RED], [$0] -> [$0], [SOUTH, RED], [WEST, RED], [NORTH, RED], [$3], [EAST, RED] (t=14, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=14)
// ASM-NEXT: {
// ASM-NEXT:   NOT, [$0] -> [SOUTH, RED], [NORTH, RED], [$0], [$5], [EAST, RED], [$8] (t=15, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=15)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$2], [$3] -> [$6] (t=16, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [$0] -> [EAST, RED] (t=16, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [SOUTH, RED] -> [$7] (t=16, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [SOUTH, RED] (t=16, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=16)
// ASM: PE(2,2):
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$0] (t=17, inv_iters=1)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [SOUTH, RED] (t=17, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$5] (t=18, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [SOUTH, RED], [$0] -> [WEST, RED] (t=19, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$2], [$3] -> [$3] (t=20, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$4], [$5] -> [SOUTH, RED] (t=21, inv_iters=1)
// ASM-NEXT:   CTRL_MOV, [SOUTH, RED] -> [$2] (t=21, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [arg2] -> [$0] (t=6, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=6)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [WEST, RED] -> [$0] (t=7, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=7)
// ASM-NEXT: {
// ASM-NEXT:   ZEXT, [$0] -> [$0] (t=8, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=8)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$2] -> [$0] (t=9, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=9)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$1] -> [$0] (t=10, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=10)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [$3] -> [$2], [$0] (t=11, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=11)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [SOUTH, RED] -> [WEST, RED], [$1], [$0] (t=12, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=12)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$3] (t=14, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [SOUTH, RED] -> [$4] (t=14, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=14)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [WEST, RED] -> [SOUTH, RED] (t=15, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=15)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$1], [WEST, RED] -> [$1] (t=16, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=16)
// ASM: PE(1,3):
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [SOUTH, RED] -> [EAST, RED] (t=15, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=15)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [SOUTH, RED] -> [EAST, RED] (t=16, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=16)
// ASM: PE(2,3):
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [SOUTH, RED] (t=17, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [SOUTH, RED] (t=16, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=16)


