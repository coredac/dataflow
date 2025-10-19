// Compiles the original C kernel to mlir, then lowers it via Neura.
// Got error when using -O3 -fno-vectorize -fno-slp-vectorize -mllvm -force-vector-width=1 
// Issue: https://github.com/coredac/dataflow/issues/164
// RUN: clang++ -S -emit-llvm -O2 -o %t-kernel-full.ll %S/../../benchmark/CGRA-Bench/kernels/histogram/histogram.cpp
// RUN: llvm-extract --rfunc=".*kernel.*" %t-kernel-full.ll -o %t-kernel-only.ll
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
// RUN:   --architecture-spec=%S/../../arch_spec/architecture.yaml \
// RUN:   --generate-code -o %t-mapping.mlir 
// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM

#loop_annotation = #llvm.loop_annotation<mustProgress = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "float", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc2, access_type = #tbaa_type_desc2, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, i128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f64 = dense<64> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, "dlti.stack_alignment" = 128 : i64, "dlti.endianness" = "little">, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.func local_unnamed_addr @_Z6kernelPfPi(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(-1.000000e+00 : f32) : f32
    %2 = llvm.mlir.constant(5.000000e+00 : f32) : f32
    %3 = llvm.mlir.constant(1.800000e+01 : f32) : f32
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.mlir.constant(2 : i64) : i64
    %7 = llvm.mlir.constant(20 : i64) : i64
    llvm.br ^bb1(%0 : i64)
  ^bb1(%8: i64):  // 2 preds: ^bb0, ^bb1
    %9 = llvm.getelementptr inbounds %arg0[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %10 = llvm.load %9 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %11 = llvm.fadd %10, %1 : f32
    %12 = llvm.fmul %11, %2 : f32
    %13 = llvm.fdiv %12, %3 : f32
    %14 = llvm.fptosi %13 : f32 to i32
    %15 = llvm.sext %14 : i32 to i64
    %16 = llvm.getelementptr inbounds %arg1[%15] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %17 = llvm.load %16 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i32
    %18 = llvm.add %17, %4 overflow<nsw> : i32
    llvm.store %18, %16 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : i32, !llvm.ptr
    %19 = llvm.or disjoint %8, %5 : i64
    %20 = llvm.getelementptr inbounds %arg0[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %21 = llvm.load %20 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %22 = llvm.fadd %21, %1 : f32
    %23 = llvm.fmul %22, %2 : f32
    %24 = llvm.fdiv %23, %3 : f32
    %25 = llvm.fptosi %24 : f32 to i32
    %26 = llvm.sext %25 : i32 to i64
    %27 = llvm.getelementptr inbounds %arg1[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %28 = llvm.load %27 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i32
    %29 = llvm.add %28, %4 overflow<nsw> : i32
    llvm.store %29, %27 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : i32, !llvm.ptr
    %30 = llvm.add %8, %6 overflow<nsw, nuw> : i64
    %31 = llvm.icmp "eq" %30, %7 : i64
    llvm.cond_br %31, ^bb2, ^bb1(%30 : i64) {loop_annotation = #loop_annotation}
  ^bb2:  // pred: ^bb1
    llvm.return
  }
}

// MAPPING: module
// MAPPING: func @_Z6kernelPfPi
// MAPPING: neura.constant
// MAPPING: neura.fdiv
// MAPPING: neura.cast

// YAML: instructions:
// YAML: - opcode: "CONSTANT"
// YAML: - opcode: "FDIV"
// YAML: - opcode: "CAST"

// ASM: PE(0,0):
// ASM: CONSTANT
