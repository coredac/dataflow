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
// AFTER_CANONICALIZE-NEXT:     %0 = "neura.constant"() <{value = "%arg0"}> : () -> i32
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
// AFTER_CANONICALIZE-NEXT:     neura.br %30, %29, %31, %30 : i64, i64, i64, i64 to ^bb6
// AFTER_CANONICALIZE-NEXT:   ^bb6(%34: i64, %35: i64, %36: i64, %37: i64):  // 2 preds: ^bb5, ^bb6
// AFTER_CANONICALIZE-NEXT:     %38 = "neura.gep"(%34) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg3"} : (i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     %39 = "neura.load"(%38) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %40 = "neura.load"(%33) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %41 = "neura.gep"(%35, %34) <{operandSegmentSizes = array<i32: 0, 2>}> {lhs_value = "%arg2"} : (i64, i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     %42 = "neura.load"(%41) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %43 = "neura.fmul_fadd"(%40, %42, %39) : (f64, f64, f64) -> f64
// AFTER_CANONICALIZE-NEXT:     "neura.store"(%43, %38) : (f64, !llvm.ptr) -> ()
// AFTER_CANONICALIZE-NEXT:     %44 = "neura.load"(%32) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %45 = "neura.load"(%41) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %46 = "neura.gep"(%34) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg5"} : (i64) -> !llvm.ptr
// AFTER_CANONICALIZE-NEXT:     %47 = "neura.load"(%46) : (!llvm.ptr) -> f64
// AFTER_CANONICALIZE-NEXT:     %48 = "neura.fmul_fadd"(%45, %47, %44) : (f64, f64, f64) -> f64
// AFTER_CANONICALIZE-NEXT:     "neura.store"(%48, %32) : (f64, !llvm.ptr) -> ()
// AFTER_CANONICALIZE-NEXT:     %49 = "neura.add"(%34) {rhs_value = 1 : i64} : (i64) -> i64
// AFTER_CANONICALIZE-NEXT:     %50 = "neura.icmp"(%49, %28) <{cmpType = "eq"}> : (i64, i64) -> i1
// AFTER_CANONICALIZE-NEXT:     neura.cond_br %50 : i1 then %35, %36, %37 : i64, i64, i64 to ^bb7 else %49, %35, %36, %37 : i64, i64, i64, i64 to ^bb6
// AFTER_CANONICALIZE-NEXT:   ^bb7(%51: i64, %52: i64, %53: i64):  // pred: ^bb6
// AFTER_CANONICALIZE-NEXT:     %54 = "neura.add"(%51) {rhs_value = 1 : i64} : (i64) -> i64
// AFTER_CANONICALIZE-NEXT:     %55 = "neura.icmp"(%54, %52) <{cmpType = "eq"}> : (i64, i64) -> i1
// AFTER_CANONICALIZE-NEXT:     neura.cond_br %55 : i1 then to ^bb8 else %54, %53, %52 : i64, i64, i64 to ^bb5
// AFTER_CANONICALIZE-NEXT:   ^bb8:  // 4 preds: ^bb1, ^bb2, ^bb3, ^bb7
// AFTER_CANONICALIZE-NEXT:     "neura.return"() : () -> ()
// AFTER_CANONICALIZE-NEXT:   }

//MAPPING: func.func @kernel
//MAPPING-SAME: accelerator = "neura", dataflow_mode = "predicate"
//MAPPING-SAME: mapping_info = {compiled_ii = 11 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 9 : i32, res_mii = 5 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}

// YAML:      array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   compiled_ii: 11
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
// YAML-NEXT:                   id: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "arg0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 2
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "arg1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$3"
// YAML-NEXT:                       color: "RED"

// ASM:      # Compiled II: 11
// ASM:      PE(0,0):
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [arg0] -> [NORTH, RED]
// ASM-NEXT: } (t=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [arg1] -> [NORTH, RED], [$3]
// ASM-NEXT: } (t=2)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [$2]
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$0]
// ASM-NEXT: } (t=4)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [arg0] -> [NORTH, RED]
// ASM-NEXT: } (t=5)
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$1]
// ASM-NEXT: } (t=6)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [$1] -> [EAST, RED]
// ASM-NEXT: } (t=7)
