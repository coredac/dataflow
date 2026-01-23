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
// AFTER_CANONICALIZE-NEXT:     neura.br %26, %26, %27 : i64, i64, i64 to ^bb5
// AFTER_CANONICALIZE-NEXT:   ^bb5(%29: i64, %30: i64, %31: i64):  // 2 preds: ^bb4, ^bb7
// AFTER_CANONICALIZE-NEXT:     %32 = "neura.gep"(%29) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg4"} : (i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     "neura.store"(%32) {lhs_value = 0.000000e+00 : f64} : (!llvm.ptr) -> ()
// AFTER_CANONICALIZE-NEXT:     %33 = "neura.gep"(%29) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg6"} : (i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     neura.br %30, %31, %30 : i64, i64, i64 to ^bb6
// AFTER_CANONICALIZE-NEXT:   ^bb6(%34: i64, %35: i64, %36: i64):  // 2 preds: ^bb5, ^bb6
// AFTER_CANONICALIZE-NEXT:     %37 = "neura.gep"(%34) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg3"} : (i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     %38 = "neura.load"(%37) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %39 = "neura.load"(%33) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %40 = "neura.gep"(%29, %34) <{operandSegmentSizes = array<i32: 0, 2>}> {lhs_value = "%arg2"} : (i64, i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     %41 = "neura.load"(%40) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %42 = "neura.fmul_fadd"(%39, %41, %38) : (f64, f64, f64) -> f64
// AFTER_CANONICALIZE-NEXT:     "neura.store"(%42, %37) : (f64, !llvm.ptr) -> ()
// AFTER_CANONICALIZE-NEXT:     %43 = "neura.load"(%32) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %44 = "neura.load"(%40) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %45 = "neura.gep"(%34) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg5"} : (i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     %46 = "neura.load"(%45) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %47 = "neura.fmul_fadd"(%44, %46, %43) : (f64, f64, f64) -> f64
// AFTER_CANONICALIZE-NEXT:     "neura.store"(%47, %32) : (f64, !llvm.ptr) -> ()
// AFTER_CANONICALIZE-NEXT:     %48 = "neura.add"(%34) {rhs_value = 1 : i64} : (i64) -> i64
// AFTER_CANONICALIZE-NEXT:     %49 = "neura.icmp"(%48, %28) <{cmpType = "eq"}> : (i64, i64) -> i1
// AFTER_CANONICALIZE-NEXT:     neura.cond_br %49 : i1 then %29, %35, %36 : i64, i64, i64 to ^bb7 else %48, %35, %36 : i64, i64, i64 to ^bb6
// AFTER_CANONICALIZE-NEXT:   ^bb7(%50: i64, %51: i64, %52: i64):  // pred: ^bb6
// AFTER_CANONICALIZE-NEXT:     %53 = "neura.add"(%50) {rhs_value = 1 : i64} : (i64) -> i64
// AFTER_CANONICALIZE-NEXT:     %54 = "neura.icmp"(%53, %51) <{cmpType = "eq"}> : (i64, i64) -> i1
// AFTER_CANONICALIZE-NEXT:     neura.cond_br %54 : i1 then to ^bb8 else %53, %52, %51 : i64, i64, i64 to ^bb5
// AFTER_CANONICALIZE-NEXT:   ^bb8:  // 4 preds: ^bb1, ^bb2, ^bb3, ^bb7
// AFTER_CANONICALIZE-NEXT:     "neura.return"() : () -> ()


// MAPPING:      func.func @kernel(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg4: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg5: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg6: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 12 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 9 : i32, res_mii = 5 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = "%arg0"}> {dfg_id = 0 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 3 : i32, y = 1 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %1 = "neura.constant"() <{value = "%arg0"}> {dfg_id = 1 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %2 = "neura.grant_once"() <{constant_value = "%arg1"}> {dfg_id = 2 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 0 : i32}]} : () -> !neura.data<i32, i1>
// MAPPING-NEXT:     %3 = "neura.grant_once"() <{constant_value = "%arg3"}> {dfg_id = 3 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 13 : i32, x = 0 : i32, y = 3 : i32}]} : () -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %4 = "neura.grant_once"() <{constant_value = "%arg4"}> {dfg_id = 4 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 12 : i32, x = 1 : i32, y = 2 : i32}]} : () -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %5 = "neura.grant_once"() <{constant_value = 0 : i8}> {dfg_id = 5 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 12 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<i8, i1>
// MAPPING-NEXT:     %6 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 6 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 1 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %7 = "neura.data_mov"(%1) {dfg_id = 15 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %8 = "neura.icmp"(%7) <{cmpType = "sgt"}> {dfg_id = 23 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 0 : i32}], rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %9 = "neura.data_mov"(%8) {dfg_id = 24 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %10 = "neura.grant_once"(%9) {dfg_id = 25 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %11 = "neura.data_mov"(%0) {dfg_id = 14 : i32, mapping_locs = [{id = 224 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %12 = "neura.data_mov"(%10) {dfg_id = 31 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 10 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 14 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 18 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %13 = neura.grant_predicate %11, %12 {dfg_id = 37 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 3 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %14 = "neura.data_mov"(%3) {dfg_id = 18 : i32, mapping_locs = [{id = 384 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %15 = "neura.data_mov"(%10) {dfg_id = 30 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 2 : i32}, {id = 1 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 12 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 26 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 385 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}, {id = 385 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}, {id = 385 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}, {id = 385 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 385 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}, {id = 385 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}, {id = 385 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 385 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %16 = neura.grant_predicate %14, %15 {dfg_id = 36 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 14 : i32, x = 0 : i32, y = 3 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %17 = "neura.data_mov"(%5) {dfg_id = 20 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 12 : i32}, {id = 31 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 13 : i32}]} : (!neura.data<i8, i1>) -> !neura.data<i8, i1>
// MAPPING-NEXT:     %18 = "neura.data_mov"(%10) {dfg_id = 29 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 2 : i32}, {id = 0 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 4 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 16 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 290 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 6 : i32}, {id = 290 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 7 : i32}, {id = 290 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 8 : i32}, {id = 290 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}, {id = 290 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 10 : i32}, {id = 290 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 11 : i32}, {id = 290 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 12 : i32}, {id = 290 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %19 = neura.grant_predicate %17, %18 {dfg_id = 35 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 14 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i8, i1>, !neura.data<i1, i1> -> !neura.data<i8, i1>
// MAPPING-NEXT:     %20 = "neura.data_mov"(%2) {dfg_id = 16 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 64 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %21 = "neura.data_mov"(%10) {dfg_id = 28 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 3 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %22 = neura.grant_predicate %20, %21 {dfg_id = 34 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 0 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %23 = "neura.data_mov"(%6) {dfg_id = 22 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %24 = "neura.data_mov"(%10) {dfg_id = 27 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 0 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %25 = neura.grant_predicate %23, %24 {dfg_id = 33 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %26 = "neura.data_mov"(%10) {dfg_id = 26 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 2 : i32}, {id = 1 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 3 : i32}, {id = 1 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}, {id = 1 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}, {id = 1 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}, {id = 1 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}, {id = 1 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %27 = "neura.not"(%26) {dfg_id = 32 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %28 = "neura.data_mov"(%2) {dfg_id = 17 : i32, mapping_locs = [{id = 96 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 8 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 5 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 2 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 0 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 0 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 0 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 0 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %29 = "neura.data_mov"(%27) {dfg_id = 40 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %30 = neura.grant_predicate %28, %29 {dfg_id = 50 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %31 = "neura.data_mov"(%4) {dfg_id = 19 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %32 = "neura.data_mov"(%27) {dfg_id = 39 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}, {id = 10 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}, {id = 16 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}, {id = 289 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %33 = neura.grant_predicate %31, %32 {dfg_id = 49 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 13 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %34 = "neura.data_mov"(%5) {dfg_id = 21 : i32, mapping_locs = [{id = 36 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 12 : i32}]} : (!neura.data<i8, i1>) -> !neura.data<i8, i1>
// MAPPING-NEXT:     %35 = "neura.data_mov"(%27) {dfg_id = 38 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}, {id = 3 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}, {id = 6 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}, {id = 9 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 12 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %36 = neura.grant_predicate %34, %35 {dfg_id = 48 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 13 : i32, x = 3 : i32, y = 1 : i32}]} : !neura.data<i8, i1>, !neura.data<i1, i1> -> !neura.data<i8, i1>
// MAPPING-NEXT:     %37 = "neura.data_mov"(%30) {dfg_id = 56 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %38 = "neura.icmp"(%37) <{cmpType = "sgt"}> {dfg_id = 62 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 0 : i32, y = 0 : i32}], rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %39 = "neura.data_mov"(%38) {dfg_id = 71 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}, {id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 12 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %40 = "neura.not"(%39) {dfg_id = 82 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 13 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %41 = "neura.data_mov"(%40) {dfg_id = 92 : i32, mapping_locs = [{id = 128 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %42 = "neura.data_mov"(%40) {dfg_id = 93 : i32, mapping_locs = [{id = 129 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %43 = neura.grant_predicate %41, %42 {dfg_id = 104 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 14 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
// MAPPING-NEXT:     %44 = "neura.data_mov"(%30) {dfg_id = 55 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}, {id = 128 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %45 = "neura.data_mov"(%38) {dfg_id = 70 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %46 = neura.grant_predicate %44, %45 {dfg_id = 81 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 12 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %47 = "neura.data_mov"(%33) {dfg_id = 54 : i32, mapping_locs = [{id = 27 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 13 : i32}, {id = 256 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 14 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %48 = "neura.data_mov"(%38) {dfg_id = 69 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}, {id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 1 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 13 : i32}, {id = 12 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 14 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %49 = neura.grant_predicate %47, %48 {dfg_id = 80 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 15 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %50 = "neura.data_mov"(%36) {dfg_id = 53 : i32, mapping_locs = [{id = 225 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}, {id = 225 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}]} : (!neura.data<i8, i1>) -> !neura.data<i8, i1>
// MAPPING-NEXT:     %51 = "neura.data_mov"(%38) {dfg_id = 68 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}, {id = 3 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 12 : i32}, {id = 6 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 13 : i32}, {id = 9 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 14 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %52 = neura.grant_predicate %50, %51 {dfg_id = 79 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 15 : i32, x = 3 : i32, y = 1 : i32}]} : !neura.data<i8, i1>, !neura.data<i1, i1> -> !neura.data<i8, i1>
// MAPPING-NEXT:     %53 = "neura.data_mov"(%46) {dfg_id = 91 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 12 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %54 = neura.zext %53 {dfg_id = 103 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 13 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<i32, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %55 = "neura.data_mov"(%54) {dfg_id = 110 : i32, mapping_locs = [{id = 256 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %56 = "neura.shl"(%55) {dfg_id = 123 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 14 : i32, x = 0 : i32, y = 2 : i32}], rhs_value = 3 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %57 = "neura.data_mov"(%49) {dfg_id = 90 : i32, mapping_locs = [{id = 24 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 15 : i32}, {id = 28 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 16 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %58 = "neura.data_mov"(%52) {dfg_id = 89 : i32, mapping_locs = [{id = 21 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 15 : i32}, {id = 20 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 16 : i32}]} : (!neura.data<i8, i1>) -> !neura.data<i8, i1>
// MAPPING-NEXT:     %59 = "neura.data_mov"(%56) {dfg_id = 133 : i32, mapping_locs = [{id = 24 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 14 : i32}, {id = 28 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 15 : i32}, {id = 320 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 16 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     "neura.memset"(%57, %58, %59) <{is_volatile = false}> {dfg_id = 146 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 17 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<i8, i1>, !neura.data<i64, i1>) -> ()
// MAPPING-NEXT:     %60 = "neura.data_mov"(%56) {dfg_id = 132 : i32, mapping_locs = [{id = 26 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 14 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.return_void %60 : !neura.data<i64, i1> {dfg_id = 145 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 15 : i32, x = 0 : i32, y = 3 : i32}]}
// MAPPING-NEXT:     %61 = "neura.data_mov"(%13) {dfg_id = 47 : i32, mapping_locs = [{id = 23 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 352 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 352 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 352 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}, {id = 352 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}, {id = 352 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}, {id = 352 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %62 = neura.zext %61 {dfg_id = 52 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 13 : i32, x = 3 : i32, y = 2 : i32}]} : !neura.data<i32, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %63 = "neura.data_mov"(%62) {dfg_id = 61 : i32, mapping_locs = [{id = 352 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %64 = "neura.shl"(%63) {dfg_id = 67 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 14 : i32, x = 3 : i32, y = 2 : i32}], rhs_value = 3 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %65 = "neura.data_mov"(%16) {dfg_id = 45 : i32, mapping_locs = [{id = 38 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 14 : i32}, {id = 41 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 15 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %66 = "neura.data_mov"(%19) {dfg_id = 44 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 14 : i32}, {id = 34 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 15 : i32}]} : (!neura.data<i8, i1>) -> !neura.data<i8, i1>
// MAPPING-NEXT:     %67 = "neura.data_mov"(%64) {dfg_id = 78 : i32, mapping_locs = [{id = 37 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 14 : i32}, {id = 46 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 15 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     "neura.memset"(%65, %66, %67) <{is_volatile = false}> {dfg_id = 88 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 16 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<i8, i1>, !neura.data<i64, i1>) -> ()
// MAPPING-NEXT:     %68 = "neura.data_mov"(%22) {dfg_id = 43 : i32, mapping_locs = [{id = 64 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %69 = "neura.icmp"(%68) <{cmpType = "sgt"}> {dfg_id = 51 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 0 : i32}], rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %70 = "neura.data_mov"(%69) {dfg_id = 60 : i32, mapping_locs = [{id = 67 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 5 : i32}, {id = 6 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 96 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 96 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 96 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}, {id = 96 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}, {id = 96 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %71 = "neura.not"(%70) {dfg_id = 66 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 12 : i32, x = 3 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %72 = "neura.data_mov"(%22) {dfg_id = 42 : i32, mapping_locs = [{id = 65 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}, {id = 65 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}, {id = 65 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %73 = "neura.data_mov"(%69) {dfg_id = 59 : i32, mapping_locs = [{id = 66 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 5 : i32}, {id = 66 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %74 = neura.grant_predicate %72, %73 {dfg_id = 65 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 0 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %75 = "neura.data_mov"(%13) {dfg_id = 46 : i32, mapping_locs = [{id = 224 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %76 = "neura.data_mov"(%69) {dfg_id = 58 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 

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
// YAML-NEXT:                   id: 710000
// YAML-NEXT:                   time_step: 12
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ICMP_SGT"
// YAML-NEXT:                   id: 23
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

// ASM:      # Compiled II: 12

// ASM:      PE(0,0):
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [arg0] -> [$0] (t=0, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [$0] -> [NORTH, RED] (t=12, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_SGT, [$0], [#0] -> [$0] (t=1, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [$1] -> [NORTH, RED] (t=13, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [$0] -> [NORTH, RED], [$3], [$2], [EAST, RED], [$0], [$1] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [$3] -> [NORTH, RED] (t=3, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [$2] -> [EAST, RED] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [NORTH, RED], [$0] -> [EAST, RED] (t=4, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=4)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$0] (t=6, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=6)
// ASM-NEXT: {
// ASM-NEXT:   NOT, [$1] -> [$1], [NORTH, RED], [EAST, RED] (t=9, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=9)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$1] -> [$0], [NORTH, RED] (t=10, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=10)
// ASM-NEXT: {
// ASM-NEXT:   ICMP_SGT, [$0], [#0] -> [$0], [NORTH, RED], [$1], [EAST, RED] (t=11, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=11)









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
