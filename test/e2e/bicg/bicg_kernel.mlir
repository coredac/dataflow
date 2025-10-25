// Compile the C kernel to LLVM IR (let clang handle headers and macros).
// Use -I %S so local headers (bicg.h, polybench.h) are visible.
// RUN: clang -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -std=c11 \
// RUN:   -I %S/../../benchmark/CGRA-Bench/kernels/bicg -DSMALL_DATASET \
// RUN:   -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/bicg/bicg.c

// Extract only the kernel function(s). PolyBench typically uses kernel_bicg,
// so a regex keeps this robust across name variants.
// RUN: llvm-extract --rfunc=".*kernel.*" %t-kernel-full.ll -o %t-kernel-only.ll

// Import the LLVM IR into MLIR (LLVM dialect).
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

// Check the IR before canonicalize-live-in to verify constant folding behavior
// RUN: FileCheck %s --input-file=%t-before-canonicalize.mlir -check-prefix=BEFORE_CANONICALIZE
// BEFORE_CANONICALIZE: module attributes
// BEFORE_CANONICALIZE: func.func @kernel
// BEFORE_CANONICALIZE: %0 = "neura.constant"() <{value = "%arg0"}> : () -> i32
// BEFORE_CANONICALIZE: %1 = "neura.constant"() <{value = "%arg1"}> : () -> i32
// BEFORE_CANONICALIZE: %2 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// BEFORE_CANONICALIZE: %3 = "neura.icmp"(%0) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (i32) -> i1
// BEFORE_CANONICALIZE: neura.cond_br %3 : i1 then to ^bb1 else to ^bb2
// BEFORE_CANONICALIZE: ^bb1:  // pred: ^bb0
// BEFORE_CANONICALIZE: %4 = "neura.constant"() <{value = 0 : i8}> : () -> i8
// BEFORE_CANONICALIZE: "neura.store"(%4) {rhs_value = "%arg3"} : (i8) -> ()
// BEFORE_CANONICALIZE: %5 = "neura.icmp"(%1) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (i32) -> i1
// BEFORE_CANONICALIZE: neura.cond_br %5 : i1 then to ^bb4 else to ^bb8
// BEFORE_CANONICALIZE: ^bb2:  // pred: ^bb0
// BEFORE_CANONICALIZE: %6 = "neura.icmp"(%1) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (i32) -> i1
// BEFORE_CANONICALIZE: neura.cond_br %6 : i1 then to ^bb3 else to ^bb8
// BEFORE_CANONICALIZE: ^bb3:  // pred: ^bb2
// BEFORE_CANONICALIZE: %7 = "neura.constant"() <{value = 0 : i8}> : () -> i8
// BEFORE_CANONICALIZE: "neura.store"(%7) {rhs_value = "%arg4"} : (i8) -> ()
// BEFORE_CANONICALIZE: neura.br to ^bb8
// BEFORE_CANONICALIZE: ^bb4:  // pred: ^bb1
// BEFORE_CANONICALIZE: %8 = neura.zext %1 : i32 -> i64
// BEFORE_CANONICALIZE: %9 = neura.zext %0 : i32 -> i64
// BEFORE_CANONICALIZE: neura.br %2 : i64 to ^bb5
// BEFORE_CANONICALIZE: ^bb5(%10: i64):  // 2 preds: ^bb4, ^bb7
// BEFORE_CANONICALIZE: %11 = "neura.gep"(%10) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg4"} : (i64) -> !llvm.ptr
// BEFORE_CANONICALIZE: "neura.store"(%11) {lhs_value = 0.000000e+00 : f64} : (!llvm.ptr) -> ()
// BEFORE_CANONICALIZE: %12 = "neura.gep"(%10) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg6"} : (i64) -> !llvm.ptr
// BEFORE_CANONICALIZE: neura.br %2 : i64 to ^bb6
// BEFORE_CANONICALIZE: ^bb6(%13: i64):  // 2 preds: ^bb5, ^bb6
// BEFORE_CANONICALIZE: %14 = "neura.gep"(%13) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg3"} : (i64) -> !llvm.ptr
// BEFORE_CANONICALIZE: %15 = "neura.load"(%14) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %16 = "neura.load"(%12) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %17 = "neura.gep"(%10, %13) <{operandSegmentSizes = array<i32: 0, 2>}> {lhs_value = "%arg2"} : (i64, i64) -> !llvm.ptr
// BEFORE_CANONICALIZE: %18 = "neura.load"(%17) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %19 = "neura.fmul_fadd"(%16, %18, %15) : (f64, f64, f64) -> f64
// BEFORE_CANONICALIZE: "neura.store"(%19, %14) : (f64, !llvm.ptr) -> ()
// BEFORE_CANONICALIZE: %20 = "neura.load"(%11) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %21 = "neura.load"(%17) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %22 = "neura.gep"(%13) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg5"} : (i64) -> !llvm.ptr
// BEFORE_CANONICALIZE: %23 = "neura.load"(%22) : (!llvm.ptr) -> f64
// BEFORE_CANONICALIZE: %24 = "neura.fmul_fadd"(%21, %23, %20) : (f64, f64, f64) -> f64
// BEFORE_CANONICALIZE: "neura.store"(%24, %11) : (f64, !llvm.ptr) -> ()
// BEFORE_CANONICALIZE: %25 = "neura.add"(%13) {rhs_value = 1 : i64} : (i64) -> i64
// BEFORE_CANONICALIZE: %26 = "neura.icmp"(%25, %9) <{cmpType = "eq"}> : (i64, i64) -> i1
// BEFORE_CANONICALIZE: neura.cond_br %26 : i1 then to ^bb7 else %25 : i64 to ^bb6
// BEFORE_CANONICALIZE: ^bb7:  // pred: ^bb6
// BEFORE_CANONICALIZE: %27 = "neura.add"(%10) {rhs_value = 1 : i64} : (i64) -> i64
// BEFORE_CANONICALIZE: %28 = "neura.icmp"(%27, %8) <{cmpType = "eq"}> : (i64, i64) -> i1
// BEFORE_CANONICALIZE: neura.cond_br %28 : i1 then to ^bb8 else %27 : i64 to ^bb5
// BEFORE_CANONICALIZE: ^bb8:  // 4 preds: ^bb1, ^bb2, ^bb3, ^bb7
// BEFORE_CANONICALIZE: "neura.return"() : () -> ()
// BEFORE_CANONICALIZE: }
// BEFORE_CANONICALIZE: }

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

// Sanity-check the mapped MLIR contains a module/func and neura ops.
// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
// MAPPING: module
// MAPPING: func.func
// MAPPING: neura.
// MAPPING: neura.return

// Verify the generated YAML/ASM artifacts look well-formed.
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML
// YAML: array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   compiled_ii: 12
// YAML-NEXT:   cores:
// YAML-NEXT:   - column: 0
// YAML-NEXT:     row: 0
// YAML-NEXT:     core_id: "0"
// YAML-NEXT:     entries:
// YAML-NEXT:     - entry_id: "entry0"
// YAML-NEXT:       instructions:
// YAML-NEXT:       - timestep: 0
// YAML-NEXT:         operations:
// YAML-NEXT:         - opcode: "CONSTANT"

// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM
// ASM:      PE(0,0):
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [#0] -> [EAST, RED]
// ASM-NEXT: } (t=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [] -> [EAST, RED]
// ASM-NEXT: } (t=2)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [#0] -> [NORTH, RED]
// ASM-NEXT: } (t=3)
