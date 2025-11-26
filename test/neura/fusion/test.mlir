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

# CHECK-MAPPING: mapping_info = {compiled_ii = 14 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 9 : i32, res_mii = 5 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}

# RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --verify-each=true --mlir-print-ir-after-failure \
# RUN:           --assign-accelerator \
# RUN:           --lower-llvm-to-neura \
# RUN:           --canonicalize-cast \
# RUN:           --canonicalize-live-in \
# RUN:           --leverage-predicated-value \
# RUN:           --fold-constant \
# RUN:           --transform-ctrl-to-data-flow \
# RUN:           --fold-constant \
# RUN:           --iter-merge-pattern="min-support=2 max-iter=10" %t-kernel.mlir| FileCheck %s --check-prefix=CHECK-ITER-MERGE-PATTERN

# CHECK-ITER-MERGE-PATTERN:     %12:2 = "neura.fused_op"(%8, %arg2, %11) <{frequency = 2 : i64, pattern_id = 13 : i64, pattern_name = "phi->gep"}> ({
# CHECK-ITER-MERGE-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !llvm.ptr, %arg7: !neura.data<i64, i1>):
# CHECK-ITER-MERGE-PATTERN:      %65 = "neura.phi"(%arg5, %arg6) : (!neura.data<!llvm.ptr, i1>, !llvm.ptr) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:      %66 = "neura.gep"(%65, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:      "neura.yield"(%65, %66) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>, !llvm.ptr, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
# CHECK-ITER-MERGE-PATTERN:    %16:2 = "neura.fused_op"(%9, %arg4, %15) <{frequency = 4 : i64, pattern_id = 15 : i64, pattern_name = "phi->phi"}> ({
# CHECK-ITER-MERGE-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !llvm.ptr, %arg7: !neura.data<!llvm.ptr, i1>):
# CHECK-ITER-MERGE-PATTERN:      %65 = "neura.phi"(%arg5, %arg6) : (!neura.data<!llvm.ptr, i1>, !llvm.ptr) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:      %66 = "neura.phi"(%arg7, %65) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:      "neura.yield"(%65, %66) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>, !llvm.ptr, !neura.data<!llvm.ptr, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
# CHECK-ITER-MERGE-PATTERN:        %20 = "neura.fused_op"(%1, %19) <{frequency = 2 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:grant_once->phi->phi"}> ({
# CHECK-ITER-MERGE-PATTERN:    ^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
# CHECK-ITER-MERGE-PATTERN:      %65 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:      %66 = "neura.phi"(%arg5, %65) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:      %67 = "neura.phi"(%arg6, %66) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:      "neura.yield"(%67) : (!neura.data<i64, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:    }) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:        %33 = "neura.fused_op"(%32, %31, %30#1) <{frequency = 2 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:load->mul->fused_op:load->add"}> ({
# CHECK-ITER-MERGE-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i32, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
# CHECK-ITER-MERGE-PATTERN:      %65 = "neura.load"(%arg5) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
# CHECK-ITER-MERGE-PATTERN:      %66 = "neura.mul"(%65, %arg6) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
# CHECK-ITER-MERGE-PATTERN:      %67 = "neura.load"(%arg7) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
# CHECK-ITER-MERGE-PATTERN:      %68 = "neura.add"(%66, %67) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
# CHECK-ITER-MERGE-PATTERN:      "neura.yield"(%68) : (!neura.data<i32, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>


# RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --verify-each=true --mlir-print-ir-after-failure \
# RUN:           --assign-accelerator \
# RUN:           --lower-llvm-to-neura \
# RUN:           --canonicalize-cast \
# RUN:           --canonicalize-live-in \
# RUN:           --leverage-predicated-value \
# RUN:           --fold-constant \
# RUN:           --transform-ctrl-to-data-flow \
# RUN:           --fold-constant \
# RUN:           --init-pattern %t-kernel.mlir | FileCheck %s --check-prefix=CHECK-INIT-PATTERN

# CHECK-INIT-PATTERN:         %2 = "neura.fused_op"(%1) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "grant_once->phi"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<i64, i1>):
# CHECK-INIT-PATTERN:      %69 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
# CHECK-INIT-PATTERN:      %70 = "neura.phi"(%arg5, %69) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%70) : (!neura.data<i64, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-INIT-PATTERN:    %14:2 = "neura.fused_op"(%10, %arg2, %13) <{frequency = 2 : i64, pattern_id = 13 : i64, pattern_name = "phi->gep"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !llvm.ptr, %arg7: !neura.data<i64, i1>):
# CHECK-INIT-PATTERN:      %69 = "neura.phi"(%arg5, %arg6) : (!neura.data<!llvm.ptr, i1>, !llvm.ptr) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      %70 = "neura.gep"(%69, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%69, %70) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>, !llvm.ptr, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
# CHECK-INIT-PATTERN:    %18:2 = "neura.fused_op"(%11, %arg4, %17) <{frequency = 4 : i64, pattern_id = 15 : i64, pattern_name = "phi->phi"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !llvm.ptr, %arg7: !neura.data<!llvm.ptr, i1>):
# CHECK-INIT-PATTERN:      %69 = "neura.phi"(%arg5, %arg6) : (!neura.data<!llvm.ptr, i1>, !llvm.ptr) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      %70 = "neura.phi"(%arg7, %69) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%69, %70) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>, !llvm.ptr, !neura.data<!llvm.ptr, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)