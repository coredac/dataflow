# RUN: clang++ -S -emit-llvm -O3 -fno-unroll-loops -fno-vectorize -o %t-kernel.ll kernel.cpp
# RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir
# RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --assign-accelerator \
# RUN:           --lower-llvm-to-neura \
# RUN:           --promote-func-arg-to-const \
# RUN:           --canonicalize-live-in \
# RUN:           --leverage-predicated-value \
# RUN:           --fold-constant \
# RUN:           --transform-ctrl-to-data-flow \
# RUN:           --fold-constant \
# RUN:           --fuse-pattern \
# RUN:           --view-op-graph \
# RUN:           --insert-data-mov %t-kernel.mlir | FileCheck %s --check-prefix=CHECK-FUSED

# RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --assign-accelerator \
# RUN:           --lower-llvm-to-neura \
# RUN:           --promote-func-arg-to-const \
# RUN:           --canonicalize-live-in \
# RUN:           --leverage-predicated-value \
# RUN:           --fold-constant \
# RUN:           --transform-ctrl-to-data-flow \
# RUN:           --fold-constant \
# RUN:           --fuse-pattern \
# RUN:           --insert-data-mov \
# RUN:           --map-to-accelerator="mapping-strategy=heuristic backtrack-config=customized" %t-kernel.mlir | FileCheck %s --check-prefix=CHECK-MAPPING

# CHECK-FUSED: func.func @_Z6kernelPA1024_iPiS1_S1_S1_
# CHECK-FUSED: accelerator = "neura"
# CHECK-FUSED-DAG: %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
# CHECK-FUSED-DAG: %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
# CHECK-FUSED-DAG: %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>

# CHECK-MAPPING: func.func @_Z6kernelPA1024_iPiS1_S1_S1_

# RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --verify-each=true --mlir-print-ir-after-failure \
# RUN:           --assign-accelerator \
# RUN:           --lower-llvm-to-neura \
# RUN:           --promote-func-arg-to-const \
# RUN:           --canonicalize-cast \
# RUN:           --canonicalize-live-in \
# RUN:           --leverage-predicated-value \
# RUN:           --fold-constant \
# RUN:           --transform-ctrl-to-data-flow \
# RUN:           --fold-constant \
# RUN:           --iter-merge-pattern="min-support=3 max-iter=4" %t-kernel.mlir | FileCheck %s --check-prefix=CHECK-ITER-MERGE-PATTERN

# CHECK-ITER-MERGE-PATTERN:         %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi"}> ({
# CHECK-ITER-MERGE-PATTERN:     ^bb0(%arg5: !neura.data<i64, i1>):
# CHECK-ITER-MERGE-PATTERN:         %61 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:         %62 = "neura.phi"(%arg5, %61) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:         "neura.yield"(%61, %62) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:     }) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
# CHECK-ITER-MERGE-PATTERN:        %15:3 = "neura.fused_op"(%14, %11#0, %13, %4) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi->fused_op:phi->fused_op:gep->load"}> ({
# CHECK-ITER-MERGE-PATTERN:     ^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
# CHECK-ITER-MERGE-PATTERN:         %61 = "neura.phi"(%arg5, %arg6) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:         %62 = "neura.phi"(%arg7, %arg8) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:         %63 = "neura.gep"(%62, %61) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:         %64 = "neura.load"(%63) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
# CHECK-ITER-MERGE-PATTERN:         "neura.yield"(%61, %62, %64) : (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:     }) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
# CHECK-ITER-MERGE-PATTERN:         %16:3 = "neura.fused_op"(%12, %2, %15#0) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi->fused_op:gep->load"}> ({
# CHECK-ITER-MERGE-PATTERN:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
# CHECK-ITER-MERGE-PATTERN:         %61 = "neura.phi"(%arg5, %arg6) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:         %62 = "neura.gep"(%61, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:         %63 = "neura.load"(%62) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
# CHECK-ITER-MERGE-PATTERN:         "neura.yield"(%61, %62, %63) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:     }) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)

# RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --verify-each=true --mlir-print-ir-after-failure \
# RUN:           --assign-accelerator \
# RUN:           --lower-llvm-to-neura \
# RUN:           --promote-func-arg-to-const \
# RUN:           --canonicalize-cast \
# RUN:           --canonicalize-live-in \
# RUN:           --leverage-predicated-value \
# RUN:           --fold-constant \
# RUN:           --transform-ctrl-to-data-flow \
# RUN:           --fold-constant \
# RUN:           --init-pattern %t-kernel.mlir | FileCheck %s --check-prefix=CHECK-INIT-PATTERN

# CHECK-INIT-PATTERN:    %21:2 = "neura.fused_op"(%16, %20) <{frequency = 6 : i64, pattern_id = 2 : i64, pattern_name = "gep->load"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>):
# CHECK-INIT-PATTERN:      %74 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      %75 = "neura.load"(%74) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%74, %75) : (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
# CHECK-INIT-PATTERN:    %22 = "neura.fused_op"(%18, %20) <{frequency = 6 : i64, pattern_id = 2 : i64, pattern_name = "gep->load"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>):
# CHECK-INIT-PATTERN:      %74 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      %75 = "neura.load"(%74) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%75) : (!neura.data<i32, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<i32, i1>

# RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --verify-each=true --mlir-print-ir-after-failure \
# RUN:           --assign-accelerator \
# RUN:           --lower-llvm-to-neura \
# RUN:           --promote-func-arg-to-const \
# RUN:           --canonicalize-cast \
# RUN:           --canonicalize-live-in \
# RUN:           --leverage-predicated-value \
# RUN:           --fold-constant \
# RUN:           --transform-ctrl-to-data-flow \
# RUN:           --fold-constant \
# RUN:           --iter-merge-pattern="min-support=3 max-iter=4" \
# RUN:           --insert-data-mov \
# RUN:           --map-to-accelerator="mapping-strategy=heuristic backtrack-config=simple" %t-kernel.mlir | FileCheck %s --check-prefix=CHECK-ITER-MERGE-PATTERN-MAPPING

// MAPPING: func.func @_Z6kernelPA1024_iPiS1_S1_S1_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// MAPPING-NEXT: %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// MAPPING-NEXT: %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
// MAPPING-NEXT: %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
// MAPPING-NEXT: %8 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT: %9 = "neura.data_mov"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %10 = "neura.phi"(%8, %9) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %11 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT: %12 = "neura.data_mov"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %13 = "neura.phi"(%11, %12) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %14 = neura.reserve : !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %15 = "neura.data_mov"(%3) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %16 = "neura.phi"(%14, %15) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %17 = neura.reserve : !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %18 = "neura.data_mov"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %19 = "neura.phi"(%17, %18) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %20 = neura.reserve : !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %21 = "neura.data_mov"(%1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %22 = "neura.phi"(%20, %21) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %23 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT: %24 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %25 = "neura.phi"(%23, %24) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %26 = neura.reserve : !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %27 = "neura.data_mov"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %28 = "neura.phi"(%26, %27) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %29 = neura.reserve : !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %30 = "neura.data_mov"(%4) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %31 = "neura.phi"(%29, %30) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %32 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT: %33 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %34 = "neura.phi"(%32, %33) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %35 = "neura.data_mov"(%28) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %36 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %38 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT: %39 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %40 = "neura.phi"(%38, %39) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %41 = neura.reserve : !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %42 = "neura.data_mov"(%28) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %43 = "neura.phi"(%41, %42) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %44 = neura.reserve : !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %45 = "neura.data_mov"(%31) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %46 = "neura.phi"(%44, %45) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %47 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT: %48 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %49 = "neura.phi"(%47, %48) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %50 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT: %51 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %52 = "neura.phi"(%50, %51) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %53 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT: %54 = "neura.data_mov"(%13) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %55 = "neura.phi"(%53, %54) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %56 = neura.reserve : !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %57 = "neura.data_mov"(%16) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %58 = "neura.phi"(%56, %57) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %59 = neura.reserve : !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %60 = "neura.data_mov"(%19) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %61 = "neura.phi"(%59, %60) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %62 = neura.reserve : !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %63 = "neura.data_mov"(%22) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %64 = "neura.phi"(%62, %63) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %65 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT: %66 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %67 = "neura.phi"(%65, %66) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %68 = "neura.data_mov"(%64) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %69 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %71 = "neura.data_mov"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %73 = "neura.data_mov"(%31) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %74 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
// MAPPING-NEXT: %76 = "neura.data_mov"(%61) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %77 = "neura.data_mov"(%49) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %78 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %80 = "neura.data_mov"(%79) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %82 = "neura.data_mov"(%81) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %83 = "neura.data_mov"(%75) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %84 = "neura.data_mov"(%72) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %86 = "neura.data_mov"(%85) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %87 = "neura.data_mov"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: "neura.store"(%86, %87) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT: %88 = "neura.data_mov"(%37) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %90 = "neura.data_mov"(%79) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %92 = "neura.data_mov"(%58) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %93 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
// MAPPING-NEXT: %95 = "neura.data_mov"(%94) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %96 = "neura.data_mov"(%91) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %97 = "neura.data_mov"(%89) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %99 = "neura.data_mov"(%98) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: %100 = "neura.data_mov"(%37) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: "neura.store"(%99, %100) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT: %101 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %102 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %104 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %105 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %107 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %109 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %110 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: neura.ctrl_mov %111 -> %65 : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT: %112 = "neura.data_mov"(%64) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %113 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: neura.ctrl_mov %114 -> %62 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %115 = "neura.data_mov"(%61) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %116 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %117 = neura.grant_predicate %115, %116 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: neura.ctrl_mov %117 -> %59 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %118 = "neura.data_mov"(%58) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %119 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %120 = neura.grant_predicate %118, %119 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: neura.ctrl_mov %120 -> %56 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %121 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %122 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: neura.ctrl_mov %123 -> %53 : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT: %124 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %125 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: neura.ctrl_mov %126 -> %50 : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT: %127 = "neura.data_mov"(%49) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %128 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %129 = neura.grant_predicate %127, %128 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: neura.ctrl_mov %129 -> %47 : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT: %130 = "neura.data_mov"(%46) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %131 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %132 = neura.grant_predicate %130, %131 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: neura.ctrl_mov %132 -> %44 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %133 = "neura.data_mov"(%43) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %134 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %135 = neura.grant_predicate %133, %134 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: neura.ctrl_mov %135 -> %41 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %136 = "neura.data_mov"(%40) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %137 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %138 = neura.grant_predicate %136, %137 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: neura.ctrl_mov %138 -> %38 : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT: %139 = "neura.data_mov"(%49) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %140 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: %142 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %143 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: %145 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %146 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: %148 = "neura.data_mov"(%46) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %149 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %151 = "neura.data_mov"(%43) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %152 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %154 = "neura.data_mov"(%40) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %155 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: %157 = "neura.data_mov"(%64) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %158 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %160 = "neura.data_mov"(%61) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %161 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %163 = "neura.data_mov"(%58) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %164 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %166 = "neura.data_mov"(%141) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %167 = "neura.data_mov"(%144) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %169 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %170 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %172 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %174 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %175 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %176 = neura.grant_predicate %174, %175 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: neura.ctrl_mov %176 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT: %177 = "neura.data_mov"(%150) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %178 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %179 = neura.grant_predicate %177, %178 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: neura.ctrl_mov %179 -> %29 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %180 = "neura.data_mov"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %181 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %182 = neura.grant_predicate %180, %181 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: neura.ctrl_mov %182 -> %26 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %183 = "neura.data_mov"(%156) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %184 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: neura.ctrl_mov %185 -> %23 : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT: %186 = "neura.data_mov"(%159) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %187 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %188 = neura.grant_predicate %186, %187 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: neura.ctrl_mov %188 -> %20 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %189 = "neura.data_mov"(%162) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %190 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %191 = neura.grant_predicate %189, %190 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: neura.ctrl_mov %191 -> %17 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %192 = "neura.data_mov"(%165) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %193 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %194 = neura.grant_predicate %192, %193 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: neura.ctrl_mov %194 -> %14 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: %195 = "neura.data_mov"(%144) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %196 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: neura.ctrl_mov %197 -> %11 : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT: %198 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %199 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: neura.ctrl_mov %200 -> %8 : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT: "neura.return"() : () -> ()
// MAPPING-NEXT: }

// YAML:      array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   cores:

# CHECK-ITER-MERGE-PATTERN-MAPPING: mapping_info = {compiled_ii = 12 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 8 : i32, res_mii = 3 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}