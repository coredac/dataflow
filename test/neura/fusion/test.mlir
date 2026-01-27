// RUN: clang++ -S -emit-llvm -O3 -fno-unroll-loops -fno-vectorize -o %t-kernel.ll kernel.cpp
// RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir
// RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --assign-accelerator \
// RUN:           --lower-llvm-to-neura \
// RUN:           --promote-input-arg-to-const \
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
// RUN:           --promote-input-arg-to-const \
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
// RUN:           --promote-input-arg-to-const \
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
// CHECK-ITER-MERGE-PATTERN-NEXT:       neura.yield results(%61, %62 : !neura.data<i64, i1>, !neura.data<i64, i1>)
// CHECK-ITER-MERGE-PATTERN-NEXT:     }) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
// CHECK-ITER-MERGE-PATTERN:      %16:2 = "neura.fused_op"(%4, %13, %15) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
// CHECK-ITER-MERGE-PATTERN-NEXT:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
// CHECK-ITER-MERGE-PATTERN-NEXT:       %61 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       %62 = "neura.gep"(%61, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       %63 = "neura.load"(%62) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       neura.yield results(%61, %63 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
// CHECK-ITER-MERGE-PATTERN-NEXT:     }) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
// CHECK-ITER-MERGE-PATTERN:     %17:3 = "neura.fused_op"(%2, %12, %15) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
// CHECK-ITER-MERGE-PATTERN-NEXT:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
// CHECK-ITER-MERGE-PATTERN-NEXT:       %61 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       %62 = "neura.gep"(%61, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       %63 = "neura.load"(%62) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       neura.yield results(%61, %62, %63 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
// CHECK-ITER-MERGE-PATTERN-NEXT:     }) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)

// RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --verify-each=true --mlir-print-ir-after-failure \
// RUN:           --assign-accelerator \
// RUN:           --lower-llvm-to-neura \
// RUN:           --promote-input-arg-to-const \
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
// CHECK-INIT-PATTERN-NEXT:       neura.yield results(%72, %73 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
// CHECK-INIT-PATTERN-NEXT:     }) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
// CHECK-INIT-PATTERN-NEXT:     %22 = "neura.fused_op"(%18, %20) <{frequency = 6 : i64, pattern_id = 2 : i64, pattern_name = "gep->load"}> ({
// CHECK-INIT-PATTERN-NEXT:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>):
// CHECK-INIT-PATTERN-NEXT:       %72 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-INIT-PATTERN-NEXT:       %73 = "neura.load"(%72) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// CHECK-INIT-PATTERN-NEXT:       neura.yield results(%73 : !neura.data<i32, i1>)
// CHECK-INIT-PATTERN-NEXT:     }) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<i32, i1>

// RUN: mlir-neura-opt --architecture-spec=%S/../../arch_spec/architecture.yaml --verify-each=true --mlir-print-ir-after-failure \
// RUN:           --assign-accelerator \
// RUN:           --lower-llvm-to-neura \
// RUN:           --promote-input-arg-to-const \
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
// RUN:           --promote-input-arg-to-const \
// RUN:           --canonicalize-return \
// RUN:           --canonicalize-cast \
// RUN:           --canonicalize-live-in \
// RUN:           --leverage-predicated-value \
// RUN:           --fold-constant \
// RUN:           --transform-ctrl-to-data-flow \
// RUN:           --fold-constant \
// RUN:           --iter-merge-pattern="min-support=3 max-iter=4" \
// RUN:           --hardware-merge="output=hardware_config.json" %t-kernel.mlir
// RUN: FileCheck %s --input-file=hardware_config.json --check-prefix=CHECK-HARDWARE-MERGE

// CHECK-HARDWARE-MERGE: {
// CHECK-HARDWARE-MERGE:   "hardware_configuration": {
// CHECK-HARDWARE-MERGE:     "summary": {
// CHECK-HARDWARE-MERGE:       "total_templates": 3
// CHECK-HARDWARE-MERGE:     },
// CHECK-HARDWARE-MERGE:     "hardware_templates": [
// CHECK-HARDWARE-MERGE:       {
// CHECK-HARDWARE-MERGE:         "template_id": 0,
// CHECK-HARDWARE-MERGE:         "instance_count": 2,
// CHECK-HARDWARE-MERGE:         "supported_single_ops": ["neura.gep", "neura.grant_once", "neura.grant_predicate", "neura.load", "neura.phi_start", "neura.store"],
// CHECK-HARDWARE-MERGE:         "supported_composite_ops": [
// CHECK-HARDWARE-MERGE:           {"pattern_id": 10, "name": "phi_start->fused_op:gep->load"},
// CHECK-HARDWARE-MERGE:           {"pattern_id": 0, "name": "gep->load"},
// CHECK-HARDWARE-MERGE:           {"pattern_id": 11, "name": "phi_start->grant_predicate"}
// CHECK-HARDWARE-MERGE:         ],
// CHECK-HARDWARE-MERGE:         "functional_units": [
// CHECK-HARDWARE-MERGE:           {"fu_id": 0, "op_type": "neura.phi_start"},
// CHECK-HARDWARE-MERGE:           {"fu_id": 1, "op_type": "neura.gep"},
// CHECK-HARDWARE-MERGE:           {"fu_id": 2, "op_type": "neura.load"},
// CHECK-HARDWARE-MERGE:           {"fu_id": 3, "op_type": "neura.grant_predicate"}
// CHECK-HARDWARE-MERGE:         ],
// CHECK-HARDWARE-MERGE:         "fu_connections": [
// CHECK-HARDWARE-MERGE:           {"from_fu": 0, "to_fu": 1},
// CHECK-HARDWARE-MERGE:           {"from_fu": 0, "to_fu": 3},
// CHECK-HARDWARE-MERGE:           {"from_fu": 1, "to_fu": 2}
// CHECK-HARDWARE-MERGE:         ],
// CHECK-HARDWARE-MERGE:         "pattern_execution_plans": [
// CHECK-HARDWARE-MERGE:           {
// CHECK-HARDWARE-MERGE:             "pattern_id": 10,
// CHECK-HARDWARE-MERGE:             "pattern_name": "phi_start->fused_op:gep->load",
// CHECK-HARDWARE-MERGE:             "fu_mapping": [0, 1, 2],
// CHECK-HARDWARE-MERGE:             "execution_stages": [
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 0,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [0],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.phi_start"]
// CHECK-HARDWARE-MERGE:               },
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 1,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [1],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.gep"]
// CHECK-HARDWARE-MERGE:               },
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 2,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [2],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.load"]
// CHECK-HARDWARE-MERGE:               }
// CHECK-HARDWARE-MERGE:             ]
// CHECK-HARDWARE-MERGE:           },
// CHECK-HARDWARE-MERGE:           {
// CHECK-HARDWARE-MERGE:             "pattern_id": 0,
// CHECK-HARDWARE-MERGE:             "pattern_name": "gep->load",
// CHECK-HARDWARE-MERGE:             "fu_mapping": [1, 2],
// CHECK-HARDWARE-MERGE:             "execution_stages": [
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 0,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [1],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.gep"]
// CHECK-HARDWARE-MERGE:               },
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 1,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [2],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.load"]
// CHECK-HARDWARE-MERGE:               }
// CHECK-HARDWARE-MERGE:             ]
// CHECK-HARDWARE-MERGE:           },
// CHECK-HARDWARE-MERGE:           {
// CHECK-HARDWARE-MERGE:             "pattern_id": 11,
// CHECK-HARDWARE-MERGE:             "pattern_name": "phi_start->grant_predicate",
// CHECK-HARDWARE-MERGE:             "fu_mapping": [0, 3],
// CHECK-HARDWARE-MERGE:             "execution_stages": [
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 0,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [0],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.phi_start"]
// CHECK-HARDWARE-MERGE:               },
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 1,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [3],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.grant_predicate"]
// CHECK-HARDWARE-MERGE:               }
// CHECK-HARDWARE-MERGE:             ]
// CHECK-HARDWARE-MERGE:           }
// CHECK-HARDWARE-MERGE:         ]
// CHECK-HARDWARE-MERGE:       },
// CHECK-HARDWARE-MERGE:       {
// CHECK-HARDWARE-MERGE:         "template_id": 1,
// CHECK-HARDWARE-MERGE:         "instance_count": 3,
// CHECK-HARDWARE-MERGE:         "supported_single_ops": ["neura.grant_once", "neura.grant_predicate", "neura.icmp"],
// CHECK-HARDWARE-MERGE:         "supported_composite_ops": [
// CHECK-HARDWARE-MERGE:           {"pattern_id": 1, "name": "fused_op:icmp->grant_predicate->grant_predicate"},
// CHECK-HARDWARE-MERGE:           {"pattern_id": 3, "name": "icmp->grant_predicate"},
// CHECK-HARDWARE-MERGE:           {"pattern_id": 2, "name": "grant_predicate->grant_predicate"}
// CHECK-HARDWARE-MERGE:         ],
// CHECK-HARDWARE-MERGE:         "functional_units": [
// CHECK-HARDWARE-MERGE:           {"fu_id": 0, "op_type": "neura.icmp"},
// CHECK-HARDWARE-MERGE:           {"fu_id": 1, "op_type": "neura.grant_predicate"},
// CHECK-HARDWARE-MERGE:           {"fu_id": 2, "op_type": "neura.grant_predicate"}
// CHECK-HARDWARE-MERGE:         ],
// CHECK-HARDWARE-MERGE:         "fu_connections": [
// CHECK-HARDWARE-MERGE:           {"from_fu": 0, "to_fu": 1},
// CHECK-HARDWARE-MERGE:           {"from_fu": 1, "to_fu": 2}
// CHECK-HARDWARE-MERGE:         ],
// CHECK-HARDWARE-MERGE:         "pattern_execution_plans": [
// CHECK-HARDWARE-MERGE:           {
// CHECK-HARDWARE-MERGE:             "pattern_id": 1,
// CHECK-HARDWARE-MERGE:             "pattern_name": "fused_op:icmp->grant_predicate->grant_predicate",
// CHECK-HARDWARE-MERGE:             "fu_mapping": [0, 1, 2],
// CHECK-HARDWARE-MERGE:             "execution_stages": [
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 0,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [0],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.icmp"]
// CHECK-HARDWARE-MERGE:               },
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 1,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [1, 2],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.grant_predicate", "neura.grant_predicate"]
// CHECK-HARDWARE-MERGE:               }
// CHECK-HARDWARE-MERGE:             ]
// CHECK-HARDWARE-MERGE:           },
// CHECK-HARDWARE-MERGE:           {
// CHECK-HARDWARE-MERGE:             "pattern_id": 3,
// CHECK-HARDWARE-MERGE:             "pattern_name": "icmp->grant_predicate",
// CHECK-HARDWARE-MERGE:             "fu_mapping": [0, 1],
// CHECK-HARDWARE-MERGE:             "execution_stages": [
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 0,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [0],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.icmp"]
// CHECK-HARDWARE-MERGE:               },
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 1,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [1],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.grant_predicate"]
// CHECK-HARDWARE-MERGE:               }
// CHECK-HARDWARE-MERGE:             ]
// CHECK-HARDWARE-MERGE:           },
// CHECK-HARDWARE-MERGE:           {
// CHECK-HARDWARE-MERGE:             "pattern_id": 2,
// CHECK-HARDWARE-MERGE:             "pattern_name": "grant_predicate->grant_predicate",
// CHECK-HARDWARE-MERGE:             "fu_mapping": [1, 2],
// CHECK-HARDWARE-MERGE:             "execution_stages": [
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 0,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [1],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.grant_predicate"]
// CHECK-HARDWARE-MERGE:               },
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 1,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [2],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.grant_predicate"]
// CHECK-HARDWARE-MERGE:               }
// CHECK-HARDWARE-MERGE:             ]
// CHECK-HARDWARE-MERGE:           }
// CHECK-HARDWARE-MERGE:         ]
// CHECK-HARDWARE-MERGE:       },
// CHECK-HARDWARE-MERGE:       {
// CHECK-HARDWARE-MERGE:         "template_id": 2,
// CHECK-HARDWARE-MERGE:         "instance_count": 2,
// CHECK-HARDWARE-MERGE:         "supported_single_ops": ["neura.grant_once", "neura.grant_predicate", "neura.phi_start"],
// CHECK-HARDWARE-MERGE:         "supported_composite_ops": [
// CHECK-HARDWARE-MERGE:           {"pattern_id": 4, "name": "grant_once->fused_op:phi_start->phi_start"},
// CHECK-HARDWARE-MERGE:           {"pattern_id": 9, "name": "grant_once->phi_start"},
// CHECK-HARDWARE-MERGE:           {"pattern_id": 8, "name": "phi_start->phi_start"}
// CHECK-HARDWARE-MERGE:         ],
// CHECK-HARDWARE-MERGE:         "functional_units": [
// CHECK-HARDWARE-MERGE:           {"fu_id": 0, "op_type": "neura.grant_once"},
// CHECK-HARDWARE-MERGE:           {"fu_id": 1, "op_type": "neura.phi_start"},
// CHECK-HARDWARE-MERGE:           {"fu_id": 2, "op_type": "neura.phi_start"}
// CHECK-HARDWARE-MERGE:         ],
// CHECK-HARDWARE-MERGE:         "fu_connections": [
// CHECK-HARDWARE-MERGE:           {"from_fu": 0, "to_fu": 1},
// CHECK-HARDWARE-MERGE:           {"from_fu": 1, "to_fu": 2}
// CHECK-HARDWARE-MERGE:         ],
// CHECK-HARDWARE-MERGE:         "pattern_execution_plans": [
// CHECK-HARDWARE-MERGE:           {
// CHECK-HARDWARE-MERGE:             "pattern_id": 9,
// CHECK-HARDWARE-MERGE:             "pattern_name": "grant_once->phi_start",
// CHECK-HARDWARE-MERGE:             "fu_mapping": [0, 1],
// CHECK-HARDWARE-MERGE:             "execution_stages": [
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 0,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [0],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.grant_once"]
// CHECK-HARDWARE-MERGE:               },
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 1,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [1],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.phi_start"]
// CHECK-HARDWARE-MERGE:               }
// CHECK-HARDWARE-MERGE:             ]
// CHECK-HARDWARE-MERGE:           },
// CHECK-HARDWARE-MERGE:           {
// CHECK-HARDWARE-MERGE:             "pattern_id": 4,
// CHECK-HARDWARE-MERGE:             "pattern_name": "grant_once->fused_op:phi_start->phi_start",
// CHECK-HARDWARE-MERGE:             "fu_mapping": [0, 1, 2],
// CHECK-HARDWARE-MERGE:             "execution_stages": [
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 0,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [0],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.grant_once"]
// CHECK-HARDWARE-MERGE:               },
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 1,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [1],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.phi_start"]
// CHECK-HARDWARE-MERGE:               },
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 2,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [2],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.phi_start"]
// CHECK-HARDWARE-MERGE:               }
// CHECK-HARDWARE-MERGE:             ]
// CHECK-HARDWARE-MERGE:           },
// CHECK-HARDWARE-MERGE:           {
// CHECK-HARDWARE-MERGE:             "pattern_id": 8,
// CHECK-HARDWARE-MERGE:             "pattern_name": "phi_start->phi_start",
// CHECK-HARDWARE-MERGE:             "fu_mapping": [1, 2],
// CHECK-HARDWARE-MERGE:             "execution_stages": [
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 0,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [1],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.phi_start"]
// CHECK-HARDWARE-MERGE:               },
// CHECK-HARDWARE-MERGE:               {
// CHECK-HARDWARE-MERGE:                 "stage": 1,
// CHECK-HARDWARE-MERGE:                 "parallel_fus": [2],
// CHECK-HARDWARE-MERGE:                 "parallel_ops": ["neura.phi_start"]
// CHECK-HARDWARE-MERGE:               }
// CHECK-HARDWARE-MERGE:             ]
// CHECK-HARDWARE-MERGE:           }
// CHECK-HARDWARE-MERGE:         ]
// CHECK-HARDWARE-MERGE:       }
// CHECK-HARDWARE-MERGE:     ]
// CHECK-HARDWARE-MERGE:   }
// CHECK-HARDWARE-MERGE: }
