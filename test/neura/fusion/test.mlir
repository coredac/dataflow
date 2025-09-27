# RUN: clang++ -S -emit-llvm -o kernel.ll kernel.cpp
# RUN: mlir-translate --import-llvm kernel.ll -o kernel.mlir
# RUN: mlir-neura-opt --assign-accelerator \
# RUN:           --lower-llvm-to-neura \
# RUN:           --canonicalize-live-in \
# RUN:           --leverage-predicated-value \
# RUN:           --fold-constant \
# RUN:           --transform-ctrl-to-data-flow \
# RUN:           --fold-constant \
# RUN:           --fuse-pattern \
# RUN:           --view-op-graph \
# RUN:           --insert-data-mov kernel.mlir | FileCheck %s --check-prefix=CHECK-FUSED

# CHECK-FUSED: func.func
# CHECK-FUSED: accelerator = "neura"
# CHECK-FUSED: %222 = neura.load_indexed %220[%221 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
# CHECK-FUSED: %231 = neura.load_indexed %229[%230 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
# CHECK-FUSED: %253 = "neura.mul_add"(%250, %251, %252) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
# CHECK-FUSED: neura.store_indexed %260 to %261[%262 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
