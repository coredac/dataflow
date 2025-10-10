# RUN: clang++ -S -emit-llvm -O3 -fno-unroll-loops -fno-vectorize -o %t-kernel.ll kernel.cpp
# RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir
# RUN: mlir-neura-opt --architecture-spec=%S/../arch_spec/architecture.yaml --assign-accelerator \
# RUN:           --lower-llvm-to-neura \
# RUN:           --canonicalize-live-in \
# RUN:           --leverage-predicated-value \
# RUN:           --fold-constant \
# RUN:           --transform-ctrl-to-data-flow \
# RUN:           --fold-constant \
# RUN:           --fuse-pattern \
# RUN:           --view-op-graph \
# RUN:           --insert-data-mov %t-kernel.mlir | FileCheck %s --check-prefix=CHECK-FUSED

# RUN: mlir-neura-opt --architecture-spec=%S/../arch_spec/architecture.yaml --assign-accelerator \
# RUN:           --lower-llvm-to-neura \
# RUN:           --canonicalize-live-in \
# RUN:           --leverage-predicated-value \
# RUN:           --fold-constant \
# RUN:           --transform-ctrl-to-data-flow \
# RUN:           --fold-constant \
# RUN:           --fuse-pattern \
# RUN:           --insert-data-mov \
# RUN:           --map-to-accelerator="mapping-strategy=heuristic backtrack-config=customized" %t-kernel.mlir | FileCheck %s --check-prefix=CHECK-MAPPING

# CHECK-FUSED: func.func
# CHECK-FUSED: accelerator = "neura"
# CHECK-FUSED: %102 = neura.load_indexed %100[%101 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
# CHECK-FUSED: %33 = "neura.mul_add"(%30, %31, %32) : (i32, i32, i32) -> i32
# CHECK-FUSED: %42 = "neura.mul_add"(%39, %40, %41) : (i32, i32, i32) -> i32

# CHECK-MAPPING: mapping_info = {compiled_ii = 18 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 9 : i32, res_mii = 5 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}
# CHECK-MAPPING: mapping_locs