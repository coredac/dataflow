# RUN: clang++ -S -emit-llvm -O3 -fno-unroll-loops -fno-vectorize -o %t-kernel.ll kernel.cpp
# RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir
# RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --assign-accelerator \
# RUN:           --lower-llvm-to-neura \
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

# RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --verify-each=true --mlir-print-ir-after-failure \
# RUN:           --assign-accelerator \
# RUN:           --lower-llvm-to-neura \
# RUN:           --canonicalize-cast \
# RUN:           --canonicalize-live-in \
# RUN:           --leverage-predicated-value \
# RUN:           --fold-constant \
# RUN:           --transform-ctrl-to-data-flow \
# RUN:           --fold-constant \
# RUN:           --iter-merge-pattern="min-support=2 max-iter=10" %t-kernel.mlir | FileCheck %s --check-prefix=CHECK-ITER-MERGE-PATTERN

# CHECK-ITER-MERGE-PATTERN: %6:2 = "neura.fused_op"(%5) <{frequency = 8 : i64, pattern_id = 4 : i64, pattern_name = "grant_once->phi"}> ({
# CHECK-ITER-MERGE-PATTERN:   ^bb0(%arg5: !neura.data<i64, i1>):
# CHECK-ITER-MERGE-PATTERN:     %65 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:     %66 = "neura.phi"(%arg5, %65) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:     "neura.yield"(%65, %66) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:   }) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
# CHECK-ITER-MERGE-PATTERN: %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN: %8 = "neura.fused_op"(%7) <{frequency = 8 : i64, pattern_id = 4 : i64, pattern_name = "grant_once->phi"}> ({
# CHECK-ITER-MERGE-PATTERN:   ^bb0(%arg5: !neura.data<!llvm.ptr, i1>):
# CHECK-ITER-MERGE-PATTERN:     %65 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:     %66 = "neura.phi"(%arg5, %65) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:     "neura.yield"(%66) : (!neura.data<!llvm.ptr, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:   }) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:   %17 = "neura.fused_op"(%0, %16) <{frequency = 4 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:grant_once->phi->phi"}> ({
# CHECK-ITER-MERGE-PATTERN:     ^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
# CHECK-ITER-MERGE-PATTERN:       %65 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:       %66 = "neura.phi"(%arg5, %65) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:       %67 = "neura.phi"(%arg6, %66) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:       "neura.yield"(%67) : (!neura.data<i64, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:   }) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:   %18 = neura.reserve : !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:   %19 = "neura.fused_op"(%1, %18) <{frequency = 4 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:grant_once->phi->phi"}> ({
# CHECK-ITER-MERGE-PATTERN:     ^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
# CHECK-ITER-MERGE-PATTERN:       %65 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:       %66 = "neura.phi"(%arg5, %65) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:       %67 = "neura.phi"(%arg6, %66) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:       "neura.yield"(%67) : (!neura.data<i64, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:   }) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:   %20 = neura.reserve : !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:   %21 = "neura.fused_op"(%2, %20) <{frequency = 4 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:grant_once->phi->phi"}> ({
# CHECK-ITER-MERGE-PATTERN:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>):
# CHECK-ITER-MERGE-PATTERN:       %65 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       %66 = "neura.phi"(%arg5, %65) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       %67 = "neura.phi"(%arg6, %66) :
# CHECK-ITER-MERGE-PATTERN:       %65 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       %67 = "neura.phi"(%arg6, %66) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       "neura.yield"(%67) : (!neura.data<!llvm.ptr, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:   }) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:   %26 = neura.reserve : !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:   %27 = neura.reserve : !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:   %28 = neura.reserve : !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:   %29 = "neura.phi"(%28, %6#1) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:   %30:2 = "neura.fused_op"(%4, %27, %29) <{frequency = 2 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:grant_once->phi->fused_op:phi->gep"}> ({
# CHECK-ITER-MERGE-PATTERN:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
# CHECK-ITER-MERGE-PATTERN:       %65 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       %66 = "neura.phi"(%arg5, %65) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       %67 = "neura.phi"(%arg6, %66) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:      %66 = "neura.gep"(%arg7, %65, %arg8) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       "neura.yield"(%65, %66) : (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:    }) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>)
# CHECK-ITER-MERGE-PATTERN:   %34:2 = "neura.fused_op"(%22, %12, %31#1, %33) <{frequency = 2 : i64, pattern_id = 16 : i64, pattern_name = "phi->fused_op:load->fused_op:fused_op:load->mul->fused_op:load->add"}> ({
# CHECK-ITER-MERGE-PATTERN:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
# CHECK-ITER-MERGE-PATTERN:       %65 = "neura.phi"(%arg5, %arg6) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:      %66 = "neura.load"(%arg7) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
# CHECK-ITER-MERGE-PATTERN:       %68 = "neura.mul"(%67, %66) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
# CHECK-ITER-MERGE-PATTERN:       %69 = "neura.load"(%65) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
# CHECK-ITER-MERGE-PATTERN:       %70 = "neura.add"(%68, %69) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
# CHECK-ITER-MERGE-PATTERN:       "neura.yield"(%65, %70) : (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:   }) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
# CHECK-ITER-MERGE-PATTERN:   "neura.store"(%34#1, %34#0) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:   %35 = "neura.add"(%29, %19) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:   %36 = "neura.icmp"(%35, %17) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
# CHECK-ITER-MERGE-PATTERN:   %37 = "neura.not"(%36) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
# CHECK-ITER-MERGE-PATTERN:   %38 = neura.grant_predicate %35, %37 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:   neura.ctrl_mov %38 -> %28 : !neura.data<i64, i1> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:   %39 = neura.grant_predicate %30#0, %37 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:   %47:2 = "neura.fused_op"(%15, %11#0, %37) <{frequency = 8 : i64, pattern_id = 15 : i64, pattern_name = "phi->grant_predicate"}> ({
# CHECK-ITER-MERGE-PATTERN:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
# CHECK-ITER-MERGE-PATTERN:       %65 = "neura.phi"(%arg5, %arg6) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       %66 = neura.grant_predicate %65, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       "neura.yield"(%65, %66) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:   }) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
# CHECK-ITER-MERGE-PATTERN:   neura.ctrl_mov %47#1 -> %15 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:   %48:2 = "neura.fused_op"(%14, %8, %37) <{frequency = 8 : i64, pattern_id = 15 : i64, pattern_name = "phi->grant_predicate"}> ({
# CHECK-ITER-MERGE-PATTERN:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
# CHECK-ITER-MERGE-PATTERN:       %65 = "neura.phi"(%arg5, %arg6) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       %66 = neura.grant_predicate %65, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       "neura.yield"(%65, %66) : (!neura.data<!llvm.ptr, i1>, !neura.data
# CHECK-ITER-MERGE-PATTERN:   %57 = "neura.fused_op"(%47#0, %36, %55) <{frequency = 8 : i64, pattern_id = 6 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
# CHECK-ITER-MERGE-PATTERN:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
# CHECK-ITER-MERGE-PATTERN:       %65 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       %66 = neura.grant_predicate %65, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       "neura.yield"(%66) : (!neura.data<!llvm.ptr, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:   }) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:   neura.ctrl_mov %57 -> %9 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:   %58 = "neura.fused_op"(%48#0, %36, %55) <{frequency = 8 : i64, pattern_id = 6 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
# CHECK-ITER-MERGE-PATTERN:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
# CHECK-ITER-MERGE-PATTERN:       %65 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       %66 = neura.grant_predicate %65, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i
# CHECK-ITER-MERGE-PATTERN:      %65 = neura.grant_predicate %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:      %66 = neura.grant_predicate %65, %arg7 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:      "neura.yield"(%66) : (!neura.data<i64, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:    }) : (!neura.data<i64, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:    neura.ctrl_mov %59 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
# CHECK-ITER-MERGE-PATTERN:    %60 = "neura.fused_op"(%30#0, %36, %55) <{frequency = 8 : i64, pattern_id = 6 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
# CHECK-ITER-MERGE-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
# CHECK-ITER-MERGE-PATTERN:      %65 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:      %66 = neura.grant_predicate %65, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:      "neura.yield"(%66) : (!neura.data<!llvm.ptr, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:   neura.ctrl_mov %60 -> %4 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:   %61 = "neura.fused_op"(%25, %36, %55) <{frequency = 8 : i64, pattern_id = 6 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
# CHECK-ITER-MERGE-PATTERN:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
# CHECK-ITER-MERGE-PATTERN:       %65 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       %66 = neura.grant_predicate %65, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       "neura.yield"(%66) : (!neura.data<!llvm.ptr, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:   }) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:   neura.ctrl_mov %61 -> %3 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:   %62 = "neura.fused_op"(%21, %36, %55) <{frequency = 8 : i64, pattern_id = 6 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
# CHECK-ITER-MERGE-PATTERN:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
# CHECK-ITER-MERGE-PATTERN:       %65 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       %66 = neura.grant_predicate %65, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
# CHECK-ITER-MERGE-PATTERN:       "neura.yield"(%66) : (!neura.data<!llvm.ptr, i1>) -> ()
# CHECK-ITER-MERGE-PATTERN:   }) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
}

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

# CHECK-INIT-PATTERN:     %1 = "neura.fused_op"(%0) <{frequency = 8 : i64, pattern_id = 4 : i64, pattern_name = "grant_once->phi"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<i64, i1>):
# CHECK-INIT-PATTERN:      %78 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
# CHECK-INIT-PATTERN:      %79 = "neura.phi"(%arg5, %78) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%79) : (!neura.data<i64, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-INIT-PATTERN:    %2 = neura.reserve : !neura.data<i64, i1>
# CHECK-INIT-PATTERN:    %3 = "neura.fused_op"(%2) <{frequency = 8 : i64, pattern_id = 4 : i64, pattern_name = "grant_once->phi"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<i64, i1>):
# CHECK-INIT-PATTERN:      %78 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
# CHECK-INIT-PATTERN:      %79 = "neura.phi"(%arg5, %78) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%79) : (!neura.data<i64, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-INIT-PATTERN:    %5 = "neura.fused_op"(%4) <{frequency = 8 : i64, pattern_id = 4 : i64, pattern_name = "grant_once->phi"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>):
# CHECK-INIT-PATTERN:      %78 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      %79 = "neura.phi"(%arg5, %78) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%79) : (!neura.data<!llvm.ptr, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:    %6 = neura.reserve : !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:    %7 = "neura.fused_op"(%6) <{frequency = 8 : i64, pattern_id = 4 : i64, pattern_name = "grant_once->phi"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>):
# CHECK-INIT-PATTERN:      %78 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      %79 = "neura.phi"(%arg5, %78) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%79) : (!neura.data<!llvm.ptr, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:    %9 = "neura.fused_op"(%8) <{frequency = 8 : i64, pattern_id = 4 : i64, pattern_name = "grant_once->phi"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>):
# CHECK-INIT-PATTERN:      %78 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      %79 = "neura.phi"(%arg5, %78) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%79) : (!neura.data<!llvm.ptr, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:    %11:2 = "neura.fused_op"(%10) <{frequency = 8 : i64, pattern_id = 4 : i64, pattern_name = "grant_once->phi"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<i64, i1>):
# CHECK-INIT-PATTERN:      %78 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
# CHECK-INIT-PATTERN:      %79 = "neura.phi"(%arg5, %78) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%78, %79) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
# CHECK-INIT-PATTERN:    %13 = "neura.fused_op"(%12) <{frequency = 8 : i64, pattern_id = 4 : i64, pattern_name = "grant_once->phi"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>):
# CHECK-INIT-PATTERN:      %78 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      %79 = "neura.phi"(%arg5, %78) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%79) : (!neura.data<!llvm.ptr, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:    %15 = "neura.fused_op"(%14) <{frequency = 8 : i64, pattern_id = 4 : i64, pattern_name = "grant_once->phi"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>):
# CHECK-INIT-PATTERN:      %78 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      %79 = "neura.phi"(%arg5, %78) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%79) : (!neura.data<!llvm.ptr, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:    %17:2 = "neura.fused_op"(%16, %11#0, %15) <{frequency = 3 : i64, pattern_id = 14 : i64, pattern_name = "phi->gep"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
# CHECK-INIT-PATTERN:      %78 = "neura.phi"(%arg5, %arg6) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
# CHECK-INIT-PATTERN:      %79 = "neura.gep"(%arg7, %78) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%78, %79) : (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>)
# CHECK-INIT-PATTERN:        %39:2 = "neura.fused_op"(%36, %9, %38) <{frequency = 3 : i64, pattern_id = 14 : i64, pattern_name = "phi->gep"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
# CHECK-INIT-PATTERN:      %78 = "neura.phi"(%arg5, %arg6) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      %79 = "neura.gep"(%78, %arg7) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%78, %79) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
# CHECK-INIT-PATTERN:    %42 = "neura.fused_op"(%41, %40) <{frequency = 2 : i64, pattern_id = 10 : i64, pattern_name = "load->mul"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i32, i1>):
# CHECK-INIT-PATTERN:      %78 = "neura.load"(%arg5) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
# CHECK-INIT-PATTERN:      %79 = "neura.mul"(%78, %arg6) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%79) : (!neura.data<i32, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
# CHECK-INIT-PATTERN:    %43 = "neura.fused_op"(%39#1, %42) <{frequency = 2 : i64, pattern_id = 9 : i64, pattern_name = "load->add"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i32, i1>):
# CHECK-INIT-PATTERN:      %78 = "neura.load"(%arg5) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
# CHECK-INIT-PATTERN:      %79 = "neura.add"(%arg6, %78) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%79) : (!neura.data<i32, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
# CHECK-INIT-PATTERN:    "neura.store"(%43, %39#1) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
# CHECK-INIT-PATTERN:    %44 = "neura.load"(%41) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
# CHECK-INIT-PATTERN:    %45 = "neura.gep"(%27, %38) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
# CHECK-INIT-PATTERN:    %46 = "neura.fused_op"(%45, %44) <{frequency = 2 : i64, pattern_id = 10 : i64, pattern_name = "load->mul"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i32, i1>):
# CHECK-INIT-PATTERN:      %78 = "neura.load"(%arg5) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
# CHECK-INIT-PATTERN:      %79 = "neura.mul"(%78, %arg6) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%79) : (!neura.data<i32, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
# CHECK-INIT-PATTERN:    %47 = "neura.fused_op"(%29, %46) <{frequency = 2 : i64, pattern_id = 9 : i64, pattern_name = "load->add"}> ({
# CHECK-INIT-PATTERN:    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i32, i1>):
# CHECK-INIT-PATTERN:      %78 = "neura.load"(%arg5) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
# CHECK-INIT-PATTERN:      %79 = "neura.add"(%arg6, %78) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
# CHECK-INIT-PATTERN:      "neura.yield"(%79) : (!neura.data<i32, i1>) -> ()
# CHECK-INIT-PATTERN:    }) : (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>