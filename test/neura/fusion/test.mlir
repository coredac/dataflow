// RUN: clang++ -S -emit-llvm -O3 -fno-unroll-loops -fno-vectorize -o %t-kernel.ll kernel.cpp
// RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir
// RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --assign-accelerator \
// RUN:           --lower-llvm-to-neura \
// RUN:           --promote-func-arg-to-const \
// RUN:           --canonicalize-return \
// RUN:           --canonicalize-live-in \
// RUN:           --leverage-predicated-value \
// RUN:           --fold-constant \
// RUN:           --transform-ctrl-to-data-flow \
// RUN:           --fold-constant \
// RUN:           --fuse-pattern \
// RUN:           --view-op-graph \
// RUN:           --insert-data-mov %t-kernel.mlir \
// RUN: | FileCheck %s --check-prefix=CHECK-FUSED

// RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --assign-accelerator \
// RUN:           --lower-llvm-to-neura \
// RUN:           --promote-func-arg-to-const \
// RUN:           --canonicalize-return \
// RUN:           --canonicalize-live-in \
// RUN:           --leverage-predicated-value \
// RUN:           --fold-constant \
// RUN:           --transform-ctrl-to-data-flow \
// RUN:           --fold-constant \
// RUN:           --fuse-pattern \
// RUN:           --insert-data-mov \
// RUN:           --map-to-accelerator="mapping-strategy=heuristic backtrack-config=customized" %t-kernel.mlir | FileCheck %s --check-prefix=CHECK-MAPPING

// CHECK-FUSED: func.func @_Z6kernelPA1024_iPiS1_S1_S1_
// CHECK-FUSED: accelerator = "neura"
// CHECK-FUSED-DAG: %91 = neura.load_indexed %89[%90 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
// CHECK-FUSED-DAG: %82 = "neura.mul_add"(%79, %80, %81) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CHECK-FUSED-DAG: %95 = "neura.mul_add"(%92, %93, %94) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>

// CHECK-MAPPING: mapping_info = {compiled_ii = 12 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 9 : i32, res_mii = 5 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}

// RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --verify-each=true --mlir-print-ir-after-failure \
// RUN:           --assign-accelerator \
// RUN:           --lower-llvm-to-neura \
// RUN:           --promote-func-arg-to-const \
// RUN:           --canonicalize-cast \
// RUN:           --canonicalize-return \
// RUN:           --canonicalize-live-in \
// RUN:           --leverage-predicated-value \
// RUN:           --fold-constant \
// RUN:           --transform-ctrl-to-data-flow \
// RUN:           --fold-constant \
// RUN:           --iter-merge-pattern="min-support=3 max-iter=4" %t-kernel.mlir \
// RUN: | FileCheck %s --check-prefix=CHECK-ITER-MERGE-PATTERN

// CHECK-ITER-MERGE-PATTERN:      %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
// CHECK-ITER-MERGE-PATTERN-NEXT:     ^bb0(%arg5: !neura.data<i64, i1>):
// CHECK-ITER-MERGE-PATTERN-NEXT:       %61 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       %62 = neura.phi_start %61, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       neura.yield %61, %62 : !neura.data<i64, i1>, !neura.data<i64, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:     }) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
// CHECK-ITER-MERGE-PATTERN:      %16:2 = "neura.fused_op"(%4, %13, %15) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
// CHECK-ITER-MERGE-PATTERN-NEXT:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
// CHECK-ITER-MERGE-PATTERN-NEXT:       %61 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       %62 = "neura.gep"(%61, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       %63 = "neura.load"(%62) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       neura.yield %61, %63 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:     }) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
// CHECK-ITER-MERGE-PATTERN:     %17:3 = "neura.fused_op"(%2, %12, %15) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
// CHECK-ITER-MERGE-PATTERN-NEXT:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
// CHECK-ITER-MERGE-PATTERN-NEXT:       %61 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       %62 = "neura.gep"(%61, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       %63 = "neura.load"(%62) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       neura.yield %61, %62, %63 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:     }) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)

// RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --verify-each=true --mlir-print-ir-after-failure \
// RUN:           --assign-accelerator \
// RUN:           --lower-llvm-to-neura \
// RUN:           --promote-func-arg-to-const \
// RUN:           --canonicalize-cast \
// RUN:           --canonicalize-return \
// RUN:           --canonicalize-live-in \
// RUN:           --leverage-predicated-value \
// RUN:           --fold-constant \
// RUN:           --transform-ctrl-to-data-flow \
// RUN:           --fold-constant \
// RUN:           --init-pattern %t-kernel.mlir \
// RUN:           | FileCheck %s --check-prefix=CHECK-INIT-PATTERN

// CHECK-INIT-PATTERN:          %21:2 = "neura.fused_op"(%16, %20) <{frequency = 6 : i64, pattern_id = 2 : i64, pattern_name = "gep->load"}> ({
// CHECK-INIT-PATTERN-NEXT:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>):
// CHECK-INIT-PATTERN-NEXT:       %72 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-INIT-PATTERN-NEXT:       %73 = "neura.load"(%72) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// CHECK-INIT-PATTERN-NEXT:       neura.yield %72, %73 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
// CHECK-INIT-PATTERN-NEXT:     }) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
// CHECK-INIT-PATTERN-NEXT:     %22 = "neura.fused_op"(%18, %20) <{frequency = 6 : i64, pattern_id = 2 : i64, pattern_name = "gep->load"}> ({
// CHECK-INIT-PATTERN-NEXT:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>):
// CHECK-INIT-PATTERN-NEXT:       %72 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-INIT-PATTERN-NEXT:       %73 = "neura.load"(%72) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// CHECK-INIT-PATTERN-NEXT:       neura.yield %73 : !neura.data<i32, i1>
// CHECK-INIT-PATTERN-NEXT:     }) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<i32, i1>

// RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --verify-each=true --mlir-print-ir-after-failure \
// RUN:           --assign-accelerator \
// RUN:           --lower-llvm-to-neura \
// RUN:           --promote-func-arg-to-const \
// RUN:           --canonicalize-cast \
// RUN:           --canonicalize-return \
// RUN:           --canonicalize-live-in \
// RUN:           --leverage-predicated-value \
// RUN:           --fold-constant \
// RUN:           --transform-ctrl-to-data-flow \
// RUN:           --fold-constant \
// RUN:           --iter-merge-pattern="min-support=3 max-iter=4" \
// RUN:           --insert-data-mov \
// RUN:           --map-to-accelerator="mapping-strategy=heuristic backtrack-config=simple" %t-kernel.mlir | FileCheck %s --check-prefix=CHECK-ITER-MERGE-PATTERN-MAPPING

// CHECK-ITER-MERGE-PATTERN-MAPPING: mapping_info = {compiled_ii = 12 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 8 : i32, res_mii = 3 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}

// RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --verify-each=true --mlir-print-ir-after-failure \
// RUN:           --assign-accelerator \
// RUN:           --lower-llvm-to-neura \
// RUN:           --promote-func-arg-to-const \
// RUN:           --canonicalize-return \
// RUN:           --canonicalize-cast \
// RUN:           --canonicalize-live-in \
// RUN:           --leverage-predicated-value \
// RUN:           --fold-constant \
// RUN:           --transform-ctrl-to-data-flow \
// RUN:           --fold-constant \
// RUN:           --iter-merge-pattern="min-support=3 max-iter=4" \
// RUN:           --hardware-merge="output=hardware_config.json" %t-kernel.mlir | FileCheck %s --check-prefix=CHECK-HARDWARE-MERGE --input-file=hardware_config.json 

// CHECK-HARDWARE-MERGE: {
// CHECK-HARDWARE-MERGE:   "hardware_configuration": {
// CHECK-HARDWARE-MERGE:     "summary": {
// CHECK-HARDWARE-MERGE:       "total_templates": 3
// CHECK-HARDWARE-MERGE:     },
// CHECK-HARDWARE-MERGE:     "hardware_templates": [
// CHECK-HARDWARE-MERGE:       {
// CHECK-HARDWARE-MERGE:         "template_id": 0,
// CHECK-HARDWARE-MERGE:         "instance_count": 1,
// CHECK-HARDWARE-MERGE:         "supported_single_ops": ["neura.add", "neura.phi_start"],
// CHECK-HARDWARE-MERGE:         "supported_composite_ops": [
// CHECK-HARDWARE-MERGE:           {"pattern_id": 2, "name": "phi_start->add"}
// CHECK-HARDWARE-MERGE:         ],
// CHECK-HARDWARE-MERGE:         "slots": [
// CHECK-HARDWARE-MERGE:           {"slot_id": 0, "supported_ops": ["neura.phi_start"]},
// CHECK-HARDWARE-MERGE:           {"slot_id": 1, "supported_ops": ["neura.add"]}
// CHECK-HARDWARE-MERGE:         ],
// CHECK-HARDWARE-MERGE:         "slot_connections": {
// CHECK-HARDWARE-MERGE:           "connections": [{"from": 0, "to": 1}]
// CHECK-HARDWARE-MERGE:         },
// CHECK-HARDWARE-MERGE:         "pattern_execution_plans": [
// CHECK-HARDWARE-MERGE:           {
// CHECK-HARDWARE-MERGE:             "pattern_id": 2,
// CHECK-HARDWARE-MERGE:             "pattern_name": "phi_start->add",
// CHECK-HARDWARE-MERGE:             "slot_mapping": [0, 1],
// CHECK-HARDWARE-MERGE:             "execution_stages": [
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 0,
// CHECK-HARDWARE-MERGE:                 "parallel_slots": [0],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.phi_start"]
// CHECK-HARDWARE-MERGE:               },
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 1,
// CHECK-HARDWARE-MERGE:                 "parallel_slots": [1],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.add"]
// CHECK-HARDWARE-MERGE:               }
// CHECK-HARDWARE-MERGE:             ]
// CHECK-HARDWARE-MERGE:           }
// CHECK-HARDWARE-MERGE:         ]
// CHECK-HARDWARE-MERGE:       },
// CHECK-HARDWARE-MERGE:       {
// CHECK-HARDWARE-MERGE:         "template_id": 1,
// CHECK-HARDWARE-MERGE:         "instance_count": 1,
// CHECK-HARDWARE-MERGE:         "supported_single_ops": ["neura.grant_once", "neura.grant_predicate", "neura.not", "neura.store"],
// CHECK-HARDWARE-MERGE:         "supported_composite_ops": [
// CHECK-HARDWARE-MERGE:           {"pattern_id": 0, "name": "fused_op:fused_op:not->grant_predicate->grant_predicate->grant_predicate"}
// CHECK-HARDWARE-MERGE:         ],
// CHECK-HARDWARE-MERGE:         "slots": [
// CHECK-HARDWARE-MERGE:           {"slot_id": 0, "supported_ops": ["neura.not"]},
// CHECK-HARDWARE-MERGE:           {"slot_id": 1, "supported_ops": ["neura.grant_predicate"]},
// CHECK-HARDWARE-MERGE:           {"slot_id": 2, "supported_ops": ["neura.grant_predicate"]},
// CHECK-HARDWARE-MERGE:           {"slot_id": 3, "supported_ops": ["neura.grant_predicate"]}
// CHECK-HARDWARE-MERGE:         ],
// CHECK-HARDWARE-MERGE:         "slot_connections": {
// CHECK-HARDWARE-MERGE:           "connections": [{"from": 0, "to": 1}, {"from": 0, "to": 2}, {"from": 0, "to": 3}]
// CHECK-HARDWARE-MERGE:         },
// CHECK-HARDWARE-MERGE:         "pattern_execution_plans": [
// CHECK-HARDWARE-MERGE:           {
// CHECK-HARDWARE-MERGE:             "pattern_id": 0,
// CHECK-HARDWARE-MERGE:             "pattern_name": "fused_op:fused_op:not->grant_predicate->grant_predicate->grant_predicate",
// CHECK-HARDWARE-MERGE:             "slot_mapping": [0, 1, 2, 3],
// CHECK-HARDWARE-MERGE:             "execution_stages": [
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 0,
// CHECK-HARDWARE-MERGE:                 "parallel_slots": [0],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.not"]
// CHECK-HARDWARE-MERGE:               },
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 1,
// CHECK-HARDWARE-MERGE:                 "parallel_slots": [1, 2, 3],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.grant_predicate", "neura.grant_predicate", "neura.grant_predicate"]
// CHECK-HARDWARE-MERGE:               }
// CHECK-HARDWARE-MERGE:             ]
// CHECK-HARDWARE-MERGE:           }
// CHECK-HARDWARE-MERGE:         ]
// CHECK-HARDWARE-MERGE:       },
// CHECK-HARDWARE-MERGE:       {
// CHECK-HARDWARE-MERGE:         "template_id": 2,
// CHECK-HARDWARE-MERGE:         "instance_count": 1,
// CHECK-HARDWARE-MERGE:         "supported_single_ops": ["neura.grant_once", "neura.grant_predicate", "neura.phi_start"],
// CHECK-HARDWARE-MERGE:         "supported_composite_ops": [
// CHECK-HARDWARE-MERGE:           {"pattern_id": 1, "name": "grant_once->phi_start"}
// CHECK-HARDWARE-MERGE:         ],
// CHECK-HARDWARE-MERGE:         "slots": [
// CHECK-HARDWARE-MERGE:           {"slot_id": 0, "supported_ops": ["neura.grant_once"]},
// CHECK-HARDWARE-MERGE:           {"slot_id": 1, "supported_ops": ["neura.phi_start"]}
// CHECK-HARDWARE-MERGE:         ],
// CHECK-HARDWARE-MERGE:         "slot_connections": {
// CHECK-HARDWARE-MERGE:           "connections": [{"from": 0, "to": 1}]
// CHECK-HARDWARE-MERGE:         },
// CHECK-HARDWARE-MERGE:         "pattern_execution_plans": [
// CHECK-HARDWARE-MERGE:           {
// CHECK-HARDWARE-MERGE:             "pattern_id": 1,
// CHECK-HARDWARE-MERGE:             "pattern_name": "grant_once->phi_start",
// CHECK-HARDWARE-MERGE:             "slot_mapping": [0, 1],
// CHECK-HARDWARE-MERGE:             "execution_stages": [
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 0,
// CHECK-HARDWARE-MERGE:                 "parallel_slots": [0],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.grant_once"]
// CHECK-HARDWARE-MERGE:               },
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 1,
// CHECK-HARDWARE-MERGE:                 "parallel_slots": [1],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.phi_start"]
// CHECK-HARDWARE-MERGE:               }
// CHECK-HARDWARE-MERGE:             ]
// CHECK-HARDWARE-MERGE:           }
// CHECK-HARDWARE-MERGE:         ]
// CHECK-HARDWARE-MERGE:       }
// CHECK-HARDWARE-MERGE:     ]
// CHECK-HARDWARE-MERGE:   }
// CHECK-HARDWARE-MERGE: }
