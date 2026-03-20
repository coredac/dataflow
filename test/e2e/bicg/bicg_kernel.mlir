// Compile the C kernel to LLVM IR (let clang handle headers and macros).
// Use -I %S so local headers (bicg.h, polybench.h) are visible.
// RUN: clang -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -std=c11 \
// RUN:   -I %S/../../benchmark/CGRA-Bench/kernels/bicg -DSMALL_DATASET \
// RUN:   -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/bicg/bicg.c

// RUN: llvm-extract --rfunc=".*kernel.*" %t-kernel-full.ll -o %t-kernel-only.ll
// RUN: mlir-translate --import-llvm %t-kernel-only.ll -o %t-kernel.mlir

// Lower and map to the Neura accelerator, then generate code.
// Exact mapping (tiles, II, etc.) depends on the architecture/heuristics,
// so checks below focus on structural properties for stability.
// RUN: mlir-neura-opt %t-kernel.mlir \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-input-arg-to-const \
// RUN:   --fold-constant \
// RUN:   -o %t-before-canonicalize.mlir
// RUN: FileCheck %s --input-file=%t-before-canonicalize.mlir -check-prefix=BEFORE_CANONICALIZE

// RUN: mlir-neura-opt %t-kernel.mlir \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-input-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-live-in \
// RUN:   -o %t-after-canonicalize.mlir
// RUN: FileCheck %s --input-file=%t-after-canonicalize.mlir -check-prefix=AFTER_CANONICALIZE

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
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --architecture-spec=%S/../../arch_spec/architecture.yaml \
// RUN:   --generate-code -o %t-mapping.mlir
// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM


// BEFORE_CANONICALIZE: module attributes
// BEFORE_CANONICALIZE: func.func @kernel
// BEFORE_CANONICALIZE: %0 = "neura.constant"() <{value = "%arg0"}> : () -> i32
// BEFORE_CANONICALIZE: %1 = "neura.constant"() <{value = "%arg1"}> : () -> i32
// BEFORE_CANONICALIZE: %2 = "neura.constant"() <{value = "%arg3"}> : () -> !llvm.ptr
// BEFORE_CANONICALIZE: %3 = "neura.constant"() <{value = "%arg4"}> : () -> !llvm.ptr
// BEFORE_CANONICALIZE: %4 = "neura.constant"() <{value = 0 : i8}> : () -> i8
// BEFORE_CANONICALIZE: %5 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// BEFORE_CANONICALIZE: %6 = "neura.icmp"(%0) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (i32) -> i1
// BEFORE_CANONICALIZE: neura.cond_br %6 : i1 then to ^bb1 else to ^bb2
// BEFORE_CANONICALIZE: ^bb1:  // pred: ^bb0
// BEFORE_CANONICALIZE: %7 = neura.zext %0 : i32 -> i64
// BEFORE_CANONICALIZE: %8 = "neura.shl"(%7) {rhs_value = 3 : i64} : (i64) -> i64
// BEFORE_CANONICALIZE: "neura.memset"(%2, %4, %8) <{is_volatile = false}> : (!llvm.ptr, i8, i64) -> ()
// BEFORE_CANONICALIZE: %9 = "neura.icmp"(%1) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (i32) -> i1
// BEFORE_CANONICALIZE: neura.cond_br %9 : i1 then to ^bb4 else to ^bb8
// BEFORE_CANONICALIZE: ^bb2:  // pred: ^bb0
// BEFORE_CANONICALIZE: %10 = "neura.icmp"(%1) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (i32) -> i1
// BEFORE_CANONICALIZE: neura.cond_br %10 : i1 then to ^bb3 else to ^bb8
// BEFORE_CANONICALIZE: ^bb3:  // pred: ^bb2
// BEFORE_CANONICALIZE: %11 = neura.zext %1 : i32 -> i64
// BEFORE_CANONICALIZE: %12 = "neura.shl"(%11) {rhs_value = 3 : i64} : (i64) -> i64
// BEFORE_CANONICALIZE: "neura.memset"(%3, %4, %12) <{is_volatile = false}> : (!llvm.ptr, i8, i64) -> ()
// BEFORE_CANONICALIZE: neura.br to ^bb8
// BEFORE_CANONICALIZE: ^bb4:  // pred: ^bb1
// BEFORE_CANONICALIZE: %13 = neura.zext %1 : i32 -> i64
// BEFORE_CANONICALIZE: %14 = neura.zext %0 : i32 -> i64
// BEFORE_CANONICALIZE: neura.br %5 : i64 to ^bb5
// BEFORE_CANONICALIZE: ^bb5(%15: i64):  // 2 preds: ^bb4, ^bb7
// BEFORE_CANONICALIZE: %16 = "neura.gep"(%15) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg4"} : (i64) -> !llvm.ptr
// BEFORE_CANONICALIZE: "neura.store"(%16) {lhs_value = 0.000000e+00 : f64} : (!llvm.ptr) -> ()
// BEFORE_CANONICALIZE: %17 = "neura.gep"(%15) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg6"} : (i64) -> !llvm.ptr
// BEFORE_CANONICALIZE: neura.br %5 : i64 to ^bb6
// BEFORE_CANONICALIZE: ^bb6(%18: i64):  // 2 preds: ^bb5, ^bb6
// BEFORE_CANONICALIZE: %19 = "neura.gep"(%18) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg3"} : (i64) -> !llvm.ptr
// BEFORE_CANONICALIZE: %20 = "neura.load"(%19) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %21 = "neura.load"(%17) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %22 = "neura.gep"(%15, %18) <{operandSegmentSizes = array<i32: 0, 2>}> {lhs_value = "%arg2"} : (i64, i64) -> !llvm.ptr
// BEFORE_CANONICALIZE: %23 = "neura.load"(%22) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %24 = "neura.fmul_fadd"(%21, %23, %20) : (f64, f64, f64) -> f64
// BEFORE_CANONICALIZE: "neura.store"(%24, %19) : (f64, !llvm.ptr) -> ()
// BEFORE_CANONICALIZE: %25 = "neura.load"(%16) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %26 = "neura.load"(%22) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %27 = "neura.gep"(%18) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg5"} : (i64) -> !llvm.ptr
// BEFORE_CANONICALIZE: %28 = "neura.load"(%27) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %29 = "neura.fmul_fadd"(%26, %28, %25) : (f64, f64, f64) -> f64
// BEFORE_CANONICALIZE: "neura.store"(%29, %16) : (f64, !llvm.ptr) -> ()
// BEFORE_CANONICALIZE: %30 = "neura.add"(%18) {rhs_value = 1 : i64} : (i64) -> i64
// BEFORE_CANONICALIZE: %31 = "neura.icmp"(%30, %14) <{cmpType = "eq"}> : (i64, i64) -> i1
// BEFORE_CANONICALIZE: neura.cond_br %31 : i1 then to ^bb7 else %30 : i64 to ^bb6
// BEFORE_CANONICALIZE: ^bb7:  // pred: ^bb6
// BEFORE_CANONICALIZE: %32 = "neura.add"(%15) {rhs_value = 1 : i64} : (i64) -> i64
// BEFORE_CANONICALIZE: %33 = "neura.icmp"(%32, %13) <{cmpType = "eq"}> : (i64, i64) -> i1
// BEFORE_CANONICALIZE: neura.cond_br %33 : i1 then to ^bb8 else %32 : i64 to ^bb5
// BEFORE_CANONICALIZE: ^bb8:  // 4 preds: ^bb1, ^bb2, ^bb3, ^bb7
// BEFORE_CANONICALIZE: "neura.return"() : () -> ()

// AFTER_CANONICALIZE:        func.func @kernel
// AFTER_CANONICALIZE-NEXT: %0 = "neura.constant"() <{value = "%arg0"}> : () -> i32
// AFTER_CANONICALIZE-NEXT:     %1 = "neura.constant"() <{value = "%arg1"}> : () -> i32
// AFTER_CANONICALIZE-NEXT:     %2 = "neura.constant"() <{value = "%arg3"}> : () -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     %3 = "neura.constant"() <{value = "%arg4"}> : () -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     %4 = "neura.constant"() <{value = 0 : i8}> : () -> i8
// AFTER_CANONICALIZE-NEXT:     %5 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// AFTER_CANONICALIZE-NEXT:     %6 = "neura.icmp"(%0) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (i32) -> i1
// AFTER_CANONICALIZE-NEXT:     neura.cond_br %6 : i1 then %0, %2, %4, %1, %5 : i32, !llvm.ptr, i8, i32, i64 to ^bb1 else %1, %3, %4 : i32, !llvm.ptr, i8 to ^bb2
// AFTER_CANONICALIZE-NEXT:   ^bb1(%7: i32, %8: !llvm.ptr, %9: i8, %10: i32, %11: i64):  // pred: ^bb0
// AFTER_CANONICALIZE-NEXT:     %12 = neura.zext %7 : i32 -> i64
// AFTER_CANONICALIZE-NEXT:     %13 = "neura.shl"(%12) {rhs_value = 3 : i64} : (i64) -> i64
// AFTER_CANONICALIZE-NEXT:     "neura.memset"(%8, %9, %13) <{is_volatile = false}> : (!llvm.ptr, i8, i64) -> ()
// AFTER_CANONICALIZE-NEXT:     %14 = "neura.icmp"(%10) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (i32) -> i1
// AFTER_CANONICALIZE-NEXT:     neura.cond_br %14 : i1 then %10, %7, %11 : i32, i32, i64 to ^bb4 else to ^bb8
// AFTER_CANONICALIZE-NEXT:   ^bb2(%15: i32, %16: !llvm.ptr, %17: i8):  // pred: ^bb0
// AFTER_CANONICALIZE-NEXT:     %18 = "neura.icmp"(%15) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (i32) -> i1
// AFTER_CANONICALIZE-NEXT:     neura.cond_br %18 : i1 then %15, %16, %17 : i32, !llvm.ptr, i8 to ^bb3 else to ^bb8
// AFTER_CANONICALIZE-NEXT:   ^bb3(%19: i32, %20: !llvm.ptr, %21: i8):  // pred: ^bb2
// AFTER_CANONICALIZE-NEXT:     %22 = neura.zext %19 : i32 -> i64
// AFTER_CANONICALIZE-NEXT:     %23 = "neura.shl"(%22) {rhs_value = 3 : i64} : (i64) -> i64
// AFTER_CANONICALIZE-NEXT:     "neura.memset"(%20, %21, %23) <{is_volatile = false}> : (!llvm.ptr, i8, i64) -> ()
// AFTER_CANONICALIZE-NEXT:     neura.br to ^bb8
// AFTER_CANONICALIZE-NEXT:   ^bb4(%24: i32, %25: i32, %26: i64):  // pred: ^bb1
// AFTER_CANONICALIZE-NEXT:     %27 = neura.zext %24 : i32 -> i64
// AFTER_CANONICALIZE-NEXT:     %28 = neura.zext %25 : i32 -> i64
// AFTER_CANONICALIZE-NEXT:     neura.br %26, %26, %28, %27 : i64, i64, i64, i64 to ^bb5
// AFTER_CANONICALIZE-NEXT:   ^bb5(%29: i64, %30: i64, %31: i64, %32: i64):  // 2 preds: ^bb4, ^bb7
// AFTER_CANONICALIZE-NEXT:     %33 = "neura.gep"(%29) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg4"} : (i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     "neura.store"(%33) {lhs_value = 0.000000e+00 : f64} : (!llvm.ptr) -> ()
// AFTER_CANONICALIZE-NEXT:     %34 = "neura.gep"(%29) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg6"} : (i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     neura.br %30, %34, %29, %33, %31, %32, %30 : i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64 to ^bb6
// AFTER_CANONICALIZE-NEXT:   ^bb6(%35: i64, %36: !llvm.ptr, %37: i64, %38: !llvm.ptr, %39: i64, %40: i64, %41: i64):  // 2 preds: ^bb5, ^bb6
// AFTER_CANONICALIZE-NEXT:     %42 = "neura.gep"(%35) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg3"} : (i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     %43 = "neura.load"(%42) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %44 = "neura.load"(%36) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %45 = "neura.gep"(%37, %35) <{operandSegmentSizes = array<i32: 0, 2>}> {lhs_value = "%arg2"} : (i64, i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     %46 = "neura.load"(%45) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %47 = "neura.fmul_fadd"(%44, %46, %43) : (f64, f64, f64) -> f64
// AFTER_CANONICALIZE-NEXT:     "neura.store"(%47, %42) : (f64, !llvm.ptr) -> ()
// AFTER_CANONICALIZE-NEXT:     %48 = "neura.load"(%38) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %49 = "neura.load"(%45) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %50 = "neura.gep"(%35) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg5"} : (i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     %51 = "neura.load"(%50) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %52 = "neura.fmul_fadd"(%49, %51, %48) : (f64, f64, f64) -> f64
// AFTER_CANONICALIZE-NEXT:     "neura.store"(%52, %38) : (f64, !llvm.ptr) -> ()
// AFTER_CANONICALIZE-NEXT:     %53 = "neura.add"(%35) {rhs_value = 1 : i64} : (i64) -> i64
// AFTER_CANONICALIZE-NEXT:     %54 = "neura.icmp"(%53, %39) <{cmpType = "eq"}> : (i64, i64) -> i1
// AFTER_CANONICALIZE-NEXT:     neura.cond_br %54 : i1 then %37, %40, %41, %39 : i64, i64, i64, i64 to ^bb7 else %53, %36, %37, %38, %39, %40, %41 : i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64 to ^bb6
// AFTER_CANONICALIZE-NEXT:   ^bb7(%55: i64, %56: i64, %57: i64, %58: i64):  // pred: ^bb6
// AFTER_CANONICALIZE-NEXT:     %59 = "neura.add"(%55) {rhs_value = 1 : i64} : (i64) -> i64
// AFTER_CANONICALIZE-NEXT:     %60 = "neura.icmp"(%59, %56) <{cmpType = "eq"}> : (i64, i64) -> i1
// AFTER_CANONICALIZE-NEXT:     neura.cond_br %60 : i1 then to ^bb8 else %59, %57, %58, %56 : i64, i64, i64, i64 to ^bb5
// AFTER_CANONICALIZE-NEXT:   ^bb8:  // 4 preds: ^bb1, ^bb2, ^bb3, ^bb7
// AFTER_CANONICALIZE-NEXT:     "neura.return"() : () -> ()


// MAPPING:      func.func @kernel(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg4: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg5: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg6: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 12 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 9 : i32, res_mii = 6 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = "%arg0"}> {dfg_id = 0 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 1 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %1 = "neura.constant"() <{value = "%arg0"}> {dfg_id = 1 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %2 = "neura.grant_once"() <{constant_value = "%arg1"}> {dfg_id = 2 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 0 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %3 = "neura.grant_once"() <{constant_value = "%arg3"}> {dfg_id = 3 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 13 : i32, x = 1 : i32, y = 2 : i32}]} : () -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %4 = "neura.grant_once"() <{constant_value = "%arg4"}> {dfg_id = 4 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 12 : i32, x = 0 : i32, y = 3 : i32}]} : () -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %5 = "neura.grant_once"() <{constant_value = 0 : i8}> {dfg_id = 5 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 12 : i32, x = 0 : i32, y = 2 : i32}]} : () -> !neura.data<i8, i1>
// MAPPING-NEXT:     %6 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 6 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %7 = "neura.data_mov"(%1) {dfg_id = 20 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %8 = "neura.icmp"(%7) <{cmpType = "sgt"}> {dfg_id = 28 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 0 : i32}], rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %9 = "neura.data_mov"(%8) {dfg_id = 29 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %10 = "neura.grant_once"(%9) {dfg_id = 30 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %11 = "neura.data_mov"(%0) {dfg_id = 19 : i32, mapping_locs = [{id = 4000 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %12 = "neura.data_mov"(%10) {dfg_id = 36 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 4001 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %13 = neura.grant_predicate %11, %12 {dfg_id = 42 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %14 = "neura.data_mov"(%3) {dfg_id = 23 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 13 : i32}, {id = 160 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 14 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %15 = "neura.data_mov"(%10) {dfg_id = 35 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 2 : i32}, {id = 0 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}, {id = 0 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 4 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 163 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 6 : i32}, {id = 163 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 7 : i32}, {id = 163 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 8 : i32}, {id = 163 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 9 : i32}, {id = 163 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 10 : i32}, {id = 163 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 11 : i32}, {id = 163 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 12 : i32}, {id = 163 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 13 : i32}, {id = 163 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 14 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %16 = neura.grant_predicate %14, %15 {dfg_id = 41 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 15 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %17 = "neura.data_mov"(%5) {dfg_id = 25 : i32, mapping_locs = [{id = 24 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 12 : i32}, {id = 289 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i8, i1>) -> !neura.data<i8, i1>
// MAPPING-NEXT:     %18 = "neura.data_mov"(%10) {dfg_id = 34 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 2 : i32}, {id = 1 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 12 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 24 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 290 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 6 : i32}, {id = 290 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 7 : i32}, {id = 290 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 8 : i32}, {id = 290 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}, {id = 290 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 10 : i32}, {id = 290 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 11 : i32}, {id = 290 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 12 : i32}, {id = 290 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>


// YAML:      array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   compiled_ii: 12
// YAML-NEXT:   cores:
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "0"
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
// YAML-NEXT:                   id: 740001
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"


// ASM:      # Compiled II: 12
// ASM:      PE(0,0):
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [arg0] -> [$0] (t=0, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [NORTH, RED] (t=12, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_SGT, [$0], [#0] -> [$0] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [$0] -> [NORTH, RED], [$0], [$2], [EAST, RED], [$1] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [$2] -> [NORTH, RED] (t=3, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [$0] -> [EAST, RED] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [$0] -> [EAST, RED] (t=4, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$0] (t=16, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   RETURN_VOID, [$0] (t=17, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=5)
// ASM-NEXT: {
// ASM-NEXT:   NOT, [$1] -> [EAST, RED], [$0], [NORTH, RED] (t=9, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=9)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [$0] -> [NORTH, RED] (t=10, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=10)


// RUN: mlir-neura-opt %t-kernel.mlir --view-op-graph 2>&1 | sed -n '/^digraph G {/,/^}$/p' > bicg_kernel_original.dot
// RUN: dot -Tpng bicg_kernel_original.dot -o bicg_kernel_original.png
// RUN: dot -Tjson bicg_kernel_original.dot -o bicg_kernel_original.json
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
// RUN:   --view-op-graph 2>&1 | sed -n '/^digraph G {/,/^}$/p' > bicg_kernel.dot
// RUN: dot -Tpng bicg_kernel.dot -o bicg_kernel.png
// RUN: dot -Tjson bicg_kernel.dot -o bicg_kernel.json
// RUN: FileCheck %s --input-file=bicg_kernel.dot -check-prefix=DOT

// DOT: digraph G {
