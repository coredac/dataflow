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
// RUN:   --promote-func-arg-to-const \
// RUN:   --fold-constant \
// RUN:   -o %t-before-canonicalize.mlir
// RUN: FileCheck %s --input-file=%t-before-canonicalize.mlir -check-prefix=BEFORE_CANONICALIZE

// RUN: mlir-neura-opt %t-kernel.mlir \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-live-in \
// RUN:   -o %t-after-canonicalize.mlir
// RUN: FileCheck %s --input-file=%t-after-canonicalize.mlir -check-prefix=AFTER_CANONICALIZE

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


// BEFORE_CANONICALIZE: module attributes
// BEFORE_CANONICALIZE: func.func @kernel
// BEFORE_CANONICALIZE: %0 = "neura.constant"() <{value = "%arg0"}> : () -> i32
// BEFORE_CANONICALIZE: %1 = "neura.constant"() <{value = "%arg1"}> : () -> i32
// BEFORE_CANONICALIZE: %2 = "neura.constant"() <{value = "%arg3"}> : () -> !llvm.ptr
// BEFORE_CANONICALIZE: %3 = "neura.constant"() <{value = "%arg4"}> : () -> !llvm.ptr
// BEFORE_CANONICALIZE: %4 = "neura.constant"() <{value = 3 : i64}> : () -> i64
// BEFORE_CANONICALIZE: %5 = "neura.constant"() <{value = 0 : i8}> : () -> i8
// BEFORE_CANONICALIZE: %6 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// BEFORE_CANONICALIZE: %7 = "neura.icmp"(%0) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (i32) -> i1
// BEFORE_CANONICALIZE: neura.cond_br %7 : i1 then to ^bb1 else to ^bb2
// BEFORE_CANONICALIZE: ^bb1:  // pred: ^bb0
// BEFORE_CANONICALIZE: %8 = neura.zext %0 : i32 -> i64
// BEFORE_CANONICALIZE: %9 = "neura.shl"(%8, %4) : (i64, i64) -> i64
// BEFORE_CANONICALIZE: "neura.memset"(%2, %5, %9) <{is_volatile = false}> : (!llvm.ptr, i8, i64) -> ()
// BEFORE_CANONICALIZE: %10 = "neura.icmp"(%1) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (i32) -> i1
// BEFORE_CANONICALIZE: neura.cond_br %10 : i1 then to ^bb4 else to ^bb8
// BEFORE_CANONICALIZE: ^bb2:  // pred: ^bb0
// BEFORE_CANONICALIZE: %11 = "neura.icmp"(%1) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (i32) -> i1
// BEFORE_CANONICALIZE: neura.cond_br %11 : i1 then to ^bb3 else to ^bb8
// BEFORE_CANONICALIZE: ^bb3:  // pred: ^bb2
// BEFORE_CANONICALIZE: %12 = neura.zext %1 : i32 -> i64
// BEFORE_CANONICALIZE: %13 = "neura.shl"(%12, %4) : (i64, i64) -> i64
// BEFORE_CANONICALIZE: "neura.memset"(%3, %5, %13) <{is_volatile = false}> : (!llvm.ptr, i8, i64) -> ()
// BEFORE_CANONICALIZE: neura.br to ^bb8
// BEFORE_CANONICALIZE: ^bb4:  // pred: ^bb1
// BEFORE_CANONICALIZE: %14 = neura.zext %1 : i32 -> i64
// BEFORE_CANONICALIZE: %15 = neura.zext %0 : i32 -> i64
// BEFORE_CANONICALIZE: neura.br %6 : i64 to ^bb5
// BEFORE_CANONICALIZE: ^bb5(%16: i64):  // 2 preds: ^bb4, ^bb7
// BEFORE_CANONICALIZE: %17 = "neura.gep"(%16) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg4"} : (i64) -> !llvm.ptr
// BEFORE_CANONICALIZE: "neura.store"(%17) {lhs_value = 0.000000e+00 : f64} : (!llvm.ptr) -> ()
// BEFORE_CANONICALIZE: %18 = "neura.gep"(%16) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg6"} : (i64) -> !llvm.ptr
// BEFORE_CANONICALIZE: neura.br %6 : i64 to ^bb6
// BEFORE_CANONICALIZE: ^bb6(%19: i64):  // 2 preds: ^bb5, ^bb6
// BEFORE_CANONICALIZE: %20 = "neura.gep"(%19) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg3"} : (i64) -> !llvm.ptr
// BEFORE_CANONICALIZE: %21 = "neura.load"(%20) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %22 = "neura.load"(%18) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %23 = "neura.gep"(%16, %19) <{operandSegmentSizes = array<i32: 0, 2>}> {lhs_value = "%arg2"} : (i64, i64) -> !llvm.ptr
// BEFORE_CANONICALIZE: %24 = "neura.load"(%23) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %25 = "neura.fmul_fadd"(%22, %24, %21) : (f64, f64, f64) -> f64
// BEFORE_CANONICALIZE: "neura.store"(%25, %20) : (f64, !llvm.ptr) -> ()
// BEFORE_CANONICALIZE: %26 = "neura.load"(%17) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %27 = "neura.load"(%23) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %28 = "neura.gep"(%19) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg5"} : (i64) -> !llvm.ptr
// BEFORE_CANONICALIZE: %29 = "neura.load"(%28) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %30 = "neura.fmul_fadd"(%27, %29, %26) : (f64, f64, f64) -> f64
// BEFORE_CANONICALIZE: "neura.store"(%30, %17) : (f64, !llvm.ptr) -> ()
// BEFORE_CANONICALIZE: %31 = "neura.add"(%19) {rhs_value = 1 : i64} : (i64) -> i64
// BEFORE_CANONICALIZE: %32 = "neura.icmp"(%31, %15) <{cmpType = "eq"}> : (i64, i64) -> i1
// BEFORE_CANONICALIZE: neura.cond_br %32 : i1 then to ^bb7 else %31 : i64 to ^bb6
// BEFORE_CANONICALIZE: ^bb7:  // pred: ^bb6
// BEFORE_CANONICALIZE: %33 = "neura.add"(%16) {rhs_value = 1 : i64} : (i64) -> i64
// BEFORE_CANONICALIZE: %34 = "neura.icmp"(%33, %14) <{cmpType = "eq"}> : (i64, i64) -> i1
// BEFORE_CANONICALIZE: neura.cond_br %34 : i1 then to ^bb8 else %33 : i64 to ^bb5
// BEFORE_CANONICALIZE: ^bb8:  // 4 preds: ^bb1, ^bb2, ^bb3, ^bb7
// BEFORE_CANONICALIZE: "neura.return"() : () -> ()

// AFTER_CANONICALIZE:        func.func @kernel(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg4: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg5: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg6: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// AFTER_CANONICALIZE-NEXT:     %0 = "neura.constant"() <{value = "%arg0"}> : () -> i32
// AFTER_CANONICALIZE-NEXT:     %1 = "neura.constant"() <{value = "%arg1"}> : () -> i32
// AFTER_CANONICALIZE-NEXT:     %2 = "neura.constant"() <{value = "%arg3"}> : () -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     %3 = "neura.constant"() <{value = "%arg4"}> : () -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     %4 = "neura.constant"() <{value = 3 : i64}> : () -> i64
// AFTER_CANONICALIZE-NEXT:     %5 = "neura.constant"() <{value = 0 : i8}> : () -> i8
// AFTER_CANONICALIZE-NEXT:     %6 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// AFTER_CANONICALIZE-NEXT:     %7 = "neura.icmp"(%0) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (i32) -> i1
// AFTER_CANONICALIZE-NEXT:     neura.cond_br %7 : i1 then %0, %4, %2, %5, %1, %6 : i32, i64, !llvm.ptr, i8, i32, i64 to ^bb1 else %1, %4, %3, %5 : i32, i64, !llvm.ptr, i8 to ^bb2
// AFTER_CANONICALIZE-NEXT:   ^bb1(%8: i32, %9: i64, %10: !llvm.ptr, %11: i8, %12: i32, %13: i64):  // pred: ^bb0
// AFTER_CANONICALIZE-NEXT:     %14 = neura.zext %8 : i32 -> i64
// AFTER_CANONICALIZE-NEXT:     %15 = "neura.shl"(%14, %9) : (i64, i64) -> i64
// AFTER_CANONICALIZE-NEXT:     "neura.memset"(%10, %11, %15) <{is_volatile = false}> : (!llvm.ptr, i8, i64) -> ()
// AFTER_CANONICALIZE-NEXT:     %16 = "neura.icmp"(%12) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (i32) -> i1
// AFTER_CANONICALIZE-NEXT:     neura.cond_br %16 : i1 then %12, %8, %13 : i32, i32, i64 to ^bb4 else to ^bb8
// AFTER_CANONICALIZE-NEXT:   ^bb2(%17: i32, %18: i64, %19: !llvm.ptr, %20: i8):  // pred: ^bb0
// AFTER_CANONICALIZE-NEXT:     %21 = "neura.icmp"(%17) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (i32) -> i1
// AFTER_CANONICALIZE-NEXT:     neura.cond_br %21 : i1 then %17, %18, %19, %20 : i32, i64, !llvm.ptr, i8 to ^bb3 else to ^bb8
// AFTER_CANONICALIZE-NEXT:   ^bb3(%22: i32, %23: i64, %24: !llvm.ptr, %25: i8):  // pred: ^bb2
// AFTER_CANONICALIZE-NEXT:     %26 = neura.zext %22 : i32 -> i64
// AFTER_CANONICALIZE-NEXT:     %27 = "neura.shl"(%26, %23) : (i64, i64) -> i64
// AFTER_CANONICALIZE-NEXT:     "neura.memset"(%24, %25, %27) <{is_volatile = false}> : (!llvm.ptr, i8, i64) -> ()
// AFTER_CANONICALIZE-NEXT:     neura.br to ^bb8
// AFTER_CANONICALIZE-NEXT:   ^bb4(%28: i32, %29: i32, %30: i64):  // pred: ^bb1
// AFTER_CANONICALIZE-NEXT:     %31 = neura.zext %28 : i32 -> i64
// AFTER_CANONICALIZE-NEXT:     %32 = neura.zext %29 : i32 -> i64
// AFTER_CANONICALIZE-NEXT:     neura.br %30, %30, %31 : i64, i64, i64 to ^bb5
// AFTER_CANONICALIZE-NEXT:   ^bb5(%33: i64, %34: i64, %35: i64):  // 2 preds: ^bb4, ^bb7
// AFTER_CANONICALIZE-NEXT:     %36 = "neura.gep"(%33) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg4"} : (i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     "neura.store"(%36) {lhs_value = 0.000000e+00 : f64} : (!llvm.ptr) -> ()
// AFTER_CANONICALIZE-NEXT:     %37 = "neura.gep"(%33) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg6"} : (i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     neura.br %34, %33, %35, %34 : i64, i64, i64, i64 to ^bb6
// AFTER_CANONICALIZE-NEXT:   ^bb6(%38: i64, %39: i64, %40: i64, %41: i64):  // 2 preds: ^bb5, ^bb6
// AFTER_CANONICALIZE-NEXT:     %42 = "neura.gep"(%38) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg3"} : (i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     %43 = "neura.load"(%42) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %44 = "neura.load"(%37) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %45 = "neura.gep"(%39, %38) <{operandSegmentSizes = array<i32: 0, 2>}> {lhs_value = "%arg2"} : (i64, i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     %46 = "neura.load"(%45) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %47 = "neura.fmul_fadd"(%44, %46, %43) : (f64, f64, f64) -> f64
// AFTER_CANONICALIZE-NEXT:     "neura.store"(%47, %42) : (f64, !llvm.ptr) -> ()
// AFTER_CANONICALIZE-NEXT:     %48 = "neura.load"(%36) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %49 = "neura.load"(%45) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %50 = "neura.gep"(%38) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg5"} : (i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     %51 = "neura.load"(%50) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %52 = "neura.fmul_fadd"(%49, %51, %48) : (f64, f64, f64) -> f64
// AFTER_CANONICALIZE-NEXT:     "neura.store"(%52, %36) : (f64, !llvm.ptr) -> ()
// AFTER_CANONICALIZE-NEXT:     %53 = "neura.add"(%38) {rhs_value = 1 : i64} : (i64) -> i64
// AFTER_CANONICALIZE-NEXT:     %54 = "neura.icmp"(%53, %32) <{cmpType = "eq"}> : (i64, i64) -> i1
// AFTER_CANONICALIZE-NEXT:     neura.cond_br %54 : i1 then %39, %40, %41 : i64, i64, i64 to ^bb7 else %53, %39, %40, %41 : i64, i64, i64, i64 to ^bb6
// AFTER_CANONICALIZE-NEXT:   ^bb7(%55: i64, %56: i64, %57: i64):  // pred: ^bb6
// AFTER_CANONICALIZE-NEXT:     %58 = "neura.add"(%55) {rhs_value = 1 : i64} : (i64) -> i64
// AFTER_CANONICALIZE-NEXT:     %59 = "neura.icmp"(%58, %56) <{cmpType = "eq"}> : (i64, i64) -> i1
// AFTER_CANONICALIZE-NEXT:     neura.cond_br %59 : i1 then to ^bb8 else %58, %57, %56 : i64, i64, i64 to ^bb5
// AFTER_CANONICALIZE-NEXT:   ^bb8:  // 4 preds: ^bb1, ^bb2, ^bb3, ^bb7
// AFTER_CANONICALIZE-NEXT:     "neura.return"() : () -> ()
// AFTER_CANONICALIZE-NEXT:   }

//MAPPING: func.func @kernel
//MAPPING-SAME: accelerator = "neura", dataflow_mode = "predicate"
//MAPPING-SAME: mapping_info = {compiled_ii = 12 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 9 : i32, res_mii = 5 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}

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
// YAML-NEXT:             - timestep: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CONSTANT"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"

// ASM:      PE(0,0):
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [#0] -> [EAST, RED]
// ASM-NEXT: } (t=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [] -> [EAST, RED], [NORTH, RED]
// ASM-NEXT: } (t=2)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$0]
// ASM-NEXT: } (t=4)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [WEST, RED] -> [$1]
// ASM-NEXT: } (t=5)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$1] -> [$0]
// ASM-NEXT: } (t=7)
// ASM-NEXT: {
// ASM-NEXT:   ZEXT, [$0] -> [NORTH, RED]
// ASM-NEXT: } (t=8)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [] -> [NORTH, RED]
// ASM-NEXT: } (t=11)
