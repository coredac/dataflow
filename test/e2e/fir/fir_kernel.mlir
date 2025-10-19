// Compiles the original C kernel to mlir, then lowers it via Neura.
// RUN: clang++ -S -emit-llvm -O0 -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/fir/fir.cpp
// RUN: llvm-extract --func=_Z6kernelPfS_S_ %t-kernel-full.ll -o %t-kernel-only.ll
// RUN: mlir-translate --import-llvm %t-kernel-only.ll -o %t-kernel.mlir

// RUN: mlir-neura-opt %t-kernel.mlir \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --promote-func-arg-to-const \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --architecture-spec=../../arch_spec/architecture.yaml \
// RUN:   --generate-code -o %t-mapping.mlir 
// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM

#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, f128 = dense<128> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, i128 = dense<128> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, "dlti.stack_alignment" = 128 : i64, "dlti.endianness" = "little">, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.func @_Z6kernelPfS_S_(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(32 : i32) : i32
    %4 = llvm.mlir.constant(0 : i64) : i64
    %5 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %6 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %8 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %9 = llvm.alloca %0 x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %5 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg1, %6 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg2, %7 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %1, %9 {alignment = 4 : i64} : f32, !llvm.ptr
    llvm.store %2, %8 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb3
    %10 = llvm.load %8 {alignment = 4 : i64} : !llvm.ptr -> i32
    %11 = llvm.icmp "slt" %10, %3 : i32
    llvm.cond_br %11, ^bb2, ^bb4
  ^bb2:  // pred: ^bb1
    %12 = llvm.load %5 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %13 = llvm.load %8 {alignment = 4 : i64} : !llvm.ptr -> i32
    %14 = llvm.sext %13 : i32 to i64
    %15 = llvm.getelementptr inbounds %12[%14] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %16 = llvm.load %15 {alignment = 4 : i64} : !llvm.ptr -> f32
    %17 = llvm.load %7 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %18 = llvm.load %8 {alignment = 4 : i64} : !llvm.ptr -> i32
    %19 = llvm.sext %18 : i32 to i64
    %20 = llvm.getelementptr inbounds %17[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %21 = llvm.load %20 {alignment = 4 : i64} : !llvm.ptr -> f32
    %22 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> f32
    %23 = llvm.intr.fmuladd(%16, %21, %22) : (f32, f32, f32) -> f32
    llvm.store %23, %9 {alignment = 4 : i64} : f32, !llvm.ptr
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    %24 = llvm.load %8 {alignment = 4 : i64} : !llvm.ptr -> i32
    %25 = llvm.add %24, %0 overflow<nsw> : i32
    llvm.store %25, %8 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1 {loop_annotation = #loop_annotation}
  ^bb4:  // pred: ^bb1
    %26 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> f32
    %27 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %28 = llvm.getelementptr inbounds %27[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %26, %28 {alignment = 4 : i64} : f32, !llvm.ptr
    llvm.return
  }
}

// MAPPING: module
// MAPPING: func @_Z6kernelPfS_S_
// MAPPING: neura.constant
// MAPPING: neura.fmul_fadd
// MAPPING: neura.load
// MAPPING: neura.store

// YAML: instructions:
// YAML: - opcode: "CONSTANT"
// YAML: - opcode: "FMUL_FADD"
// YAML: - opcode: "LOAD"
// YAML: - opcode: "STORE"

// ASM: PE(0,0):
// ASM: CONSTANT
