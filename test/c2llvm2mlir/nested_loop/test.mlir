// RUN: clang++ -S -emit-llvm kernel.cpp -o %t-kernel.ll
// RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir

// RUN: mlir-neura-opt --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov %t-kernel.mlir \
// RUN:   | FileCheck %s --check-prefix=CHECK-LLVM2NEURA

// RUN: mlir-neura-opt --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic backtrack-config=simple" \
// RUN:   --architecture-spec=../../arch_spec/architecture.yaml %t-kernel.mlir | FileCheck %s --check-prefix=CHECK-LLVM2NEURA-MAP

// CHECK-LLVM2NEURA: accelerator = "neura"
// CHECK-LLVM2NEURA: %25 = neura.alloca %24 : !neura.data<i32, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-LLVM2NEURA: %38 = neura.phi_start %37, %36 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CHECK-LLVM2NEURA: %182 = neura.sext %181 : !neura.data<i32, i1> -> !neura.data<i64, i1>
// CHECK-LLVM2NEURA: %201 = "neura.mul"(%199, %200) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>

// CHECK-LLVM2NEURA-MAP: func.func @_Z6kernelPiS_S_(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", frame_pointer = #llvm.framePointerKind<all>, linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 13 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 9 : i32, res_mii = 6 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, no_inline, no_unwind, optimize_none, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 0 : i64, visibility_ = 0 : i64} {