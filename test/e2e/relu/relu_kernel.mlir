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
// MAPPING: module attributes
// MAPPING: func.func @kernel
// MAPPING-SAME: compiled_ii = 5
// MAPPING-SAME: mapping_mode = "spatial-temporal"
// MAPPING-SAME: mapping_strategy = "heuristic"
// MAPPING-SAME: rec_mii = 5
// MAPPING-SAME: res_mii = 2
// MAPPING-SAME: x_tiles = 4
// MAPPING-SAME: y_tiles = 4
// MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = 0 : i32}> {dfg_id = 0 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %1 = neura.reserve {dfg_id = 1 : i32} : !neura.data<i32, i1>
// MAPPING-NEXT:     %2 = "neura.data_mov"(%0) {dfg_id = 4 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}, {id = 322 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 1 : i32}, {id = 322 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 2 : i32}, {id = 322 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 3 : i32}, {id = 322 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 4 : i32}, {id = 322 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 5 : i32}, {id = 322 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %3 = "neura.phi"(%1, %2) {dfg_id = 6 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %4 = neura.reserve {dfg_id = 2 : i32} : !neura.data<i32, i1>
// MAPPING-NEXT:     %5 = "neura.data_mov"(%0) {dfg_id = 5 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %6 = "neura.phi"(%4, %5) {dfg_id = 7 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 3 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %7 = "neura.data_mov"(%6) {dfg_id = 11 : i32, mapping_locs = [{id = 36 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %8 = "neura.cast"(%7) <{cast_type = "trunc"}> {dfg_id = 13 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i16, i1>
// MAPPING-NEXT:     %9 = "neura.data_mov"(%8) {dfg_id = 17 : i32, mapping_locs = [{id = 224 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i16, i1>) -> !neura.data<i16, i1>
// MAPPING-NEXT:     %10 = "neura.div"(%9) {dfg_id = 20 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 1 : i32}], rhs_value = 70 : i16} : (!neura.data<i16, i1>) -> !neura.data<i16, i1>
// MAPPING-NEXT:     %11 = "neura.data_mov"(%8) {dfg_id = 16 : i32, mapping_locs = [{id = 23 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i16, i1>) -> !neura.data<i16, i1>
// MAPPING-NEXT:     %12 = "neura.rem"(%11) {dfg_id = 19 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 2 : i32}], rhs_value = 70 : i16} : (!neura.data<i16, i1>) -> !neura.data<i16, i1>
// MAPPING-NEXT:     %13 = "neura.data_mov"(%10) {dfg_id = 23 : i32, mapping_locs = [{id = 23 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i16, i1>) -> !neura.data<i16, i1>
// MAPPING-NEXT:     %14 = neura.zext %13 {dfg_id = 26 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 3 : i32, y = 2 : i32}]} : !neura.data<i16, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %15 = "neura.data_mov"(%12) {dfg_id = 22 : i32, mapping_locs = [{id = 37 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i16, i1>) -> !neura.data<i16, i1>
// MAPPING-NEXT:     %16 = neura.zext %15 {dfg_id = 25 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 3 : i32, y = 3 : i32}]} : !neura.data<i16, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %17 = "neura.data_mov"(%14) {dfg_id = 32 : i32, mapping_locs = [{id = 37 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %18 = "neura.data_mov"(%16) {dfg_id = 30 : i32, mapping_locs = [{id = 480 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %19 = "neura.gep"(%17, %18) <{operandSegmentSizes = array<i32: 0, 2>}> {dfg_id = 36 : i32, lhs_value = "%arg4", mapping_locs = [{id = 15 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 3 : i32, y = 3 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %20 = "neura.data_mov"(%19) {dfg_id = 40 : i32, mapping_locs = [{id = 480 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %21 = "neura.load"(%20) {dfg_id = 41 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 3 : i32, y = 3 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %22 = "neura.data_mov"(%21) {dfg_id = 43 : i32, mapping_locs = [{id = 480 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %23 = "neura.icmp"(%22) <{cmpType = "sge"}> {dfg_id = 44 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 3 : i32, y = 3 : i32}], rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %24 = "neura.data_mov"(%23) {dfg_id = 45 : i32, mapping_locs = [{id = 46 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %25 = "neura.data_mov"(%21) {dfg_id = 42 : i32, mapping_locs = [{id = 46 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}, {id = 448 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %26 = "neura.data_mov"(%3) {dfg_id = 9 : i32, mapping_locs = [{id = 34 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %27 = "neura.sel"(%24, %25, %26) {dfg_id = 46 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 8 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<i1, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %28 = "neura.data_mov"(%14) {dfg_id = 31 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 31 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}, {id = 288 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 288 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %29 = "neura.data_mov"(%16) {dfg_id = 29 : i32, mapping_locs = [{id = 46 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 43 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}, {id = 42 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}, {id = 289 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %30 = "neura.gep"(%28, %29) <{operandSegmentSizes = array<i32: 0, 2>}> {dfg_id = 35 : i32, lhs_value = "%arg3", mapping_locs = [{id = 9 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %31 = "neura.data_mov"(%27) {dfg_id = 47 : i32, mapping_locs = [{id = 43 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %32 = "neura.data_mov"(%30) {dfg_id = 39 : i32, mapping_locs = [{id = 30 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     "neura.store"(%31, %32) {dfg_id = 48 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT:     %33 = "neura.data_mov"(%6) {dfg_id = 10 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %34 = "neura.add"(%33) {dfg_id = 12 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 2 : i32}], rhs_value = 1 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %35 = "neura.data_mov"(%34) {dfg_id = 15 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %36 = "neura.icmp"(%35) <{cmpType = "eq"}> {dfg_id = 18 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}], rhs_value = 4200 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %37 = "neura.data_mov"(%36) {dfg_id = 21 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %38 = "neura.not"(%37) {dfg_id = 24 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %39 = "neura.data_mov"(%34) {dfg_id = 14 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 35 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 320 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %40 = "neura.data_mov"(%38) {dfg_id = 28 : i32, mapping_locs = [{id = 321 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %41 = neura.grant_predicate %39, %40 {dfg_id = 34 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %41 -> %4 {dfg_id = 38 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}]} : !neura.data<i32, i1> !neura.data<i32, i1>
// MAPPING-NEXT:     %42 = "neura.data_mov"(%3) {dfg_id = 8 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 7 : i32}, {id = 192 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %43 = "neura.data_mov"(%38) {dfg_id = 27 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 193 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}, {id = 193 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}, {id = 193 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}, {id = 193 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %44 = neura.grant_predicate %42, %43 {dfg_id = 33 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 9 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %44 -> %1 {dfg_id = 37 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 9 : i32}, {id = 320 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}, {id = 320 : i32, index_per_ii = 1 : i32, invalid_iterations = 2 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}]} : !neura.data<i32, i1> !neura.data<i32, i1>
// MAPPING-NEXT:     "neura.return"() {dfg_id = 3 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 2 : i32}]} : () -> ()

// YAML:        compiled_ii: 5
// YAML:        instructions:
// YAML:        - opcode: "DATA_MOV"
// YAML:        - opcode: "CAST_TRUNC"
// YAML:        - opcode: "ICMP_EQ"
// YAML:        - opcode: "ICMP_SGE"

// ASM:      PE(2,1):
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$1]
// ASM-NEXT: } (t=5)

// RUN: mlir-neura-opt %t-kernel.mlir \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --view-op-graph 2>&1 | sed -n '/^digraph G {/,/^}$/p' > relu_kernel.dot
// RUN: dot -Tpng relu_kernel.dot -o relu_kernel.png
// RUN: dot -Tjson relu_kernel.dot -o relu_kernel.json
// RUN: FileCheck %s --input-file=relu_kernel.dot -check-prefix=DOT

// DOT: digraph G {
