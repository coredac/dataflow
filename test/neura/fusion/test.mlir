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
// CHECK-FUSED-DAG: %102 = neura.load_indexed %100[%101 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
// CHECK-FUSED-DAG: %93 = "neura.mul_add"(%90, %91, %92) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CHECK-FUSED-DAG: %106 = "neura.mul_add"(%103, %104, %105) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>

// CHECK-MAPPING: /bin/sh: --: invalid option
// CHECK-MAPPING-NEXT: Usage:	/bin/sh [GNU long option] [option] ...
// CHECK-MAPPING-NEXT: 	/bin/sh [GNU long option] [option] script-file ...
// CHECK-MAPPING-NEXT: GNU long options:
// CHECK-MAPPING-NEXT: 	--debug
// CHECK-MAPPING-NEXT: 	--debugger
// CHECK-MAPPING-NEXT: 	--dump-po-strings
// CHECK-MAPPING-NEXT: 	--dump-strings
// CHECK-MAPPING-NEXT: 	--help
// CHECK-MAPPING-NEXT: 	--init-file
// CHECK-MAPPING-NEXT: 	--login
// CHECK-MAPPING-NEXT: 	--noediting
// CHECK-MAPPING-NEXT: 	--noprofile
// CHECK-MAPPING-NEXT: 	--norc
// CHECK-MAPPING-NEXT: 	--posix
// CHECK-MAPPING-NEXT: 	--pretty-print
// CHECK-MAPPING-NEXT: 	--rcfile
// CHECK-MAPPING-NEXT: 	--restricted
// CHECK-MAPPING-NEXT: 	--verbose
// CHECK-MAPPING-NEXT: 	--version
// CHECK-MAPPING-NEXT: Shell options:
// CHECK-MAPPING-NEXT: 	-ilrsD or -c command or -O shopt_option		(invocation only)
// CHECK-MAPPING-NEXT: 	-abefhkmnptuvxBCHP or -o option

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

// CHECK-ITER-MERGE-PATTERN:      %9:2 = "neura.fused_op"(%8) <{frequency = 4 : i64, pattern_id = 8 : i64, pattern_name = "grant_once->phi_start"}> ({
// CHECK-ITER-MERGE-PATTERN-NEXT:     ^bb0(%arg5: !neura.data<i64, i1>):
// CHECK-ITER-MERGE-PATTERN-NEXT:       %72 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       %73 = neura.phi_start %72, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       neura.yield results(%72, %73 : !neura.data<i64, i1>, !neura.data<i64, i1>)
// CHECK-ITER-MERGE-PATTERN-NEXT:     }) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
// CHECK-ITER-MERGE-PATTERN:      %38:2 = "neura.fused_op"(%9#1, %37, %26) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
// CHECK-ITER-MERGE-PATTERN-NEXT:     ^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
// CHECK-ITER-MERGE-PATTERN-NEXT:       %72 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       %73 = "neura.gep"(%arg7, %72) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       %74 = "neura.load"(%73) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       neura.yield results(%72, %74 : !neura.data<i64, i1>, !neura.data<i32, i1>)
// CHECK-ITER-MERGE-PATTERN-NEXT:     }) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
// CHECK-ITER-MERGE-PATTERN:      %39:2 = "neura.fused_op"(%32, %30, %38#0) <{frequency = 4 : i64, pattern_id = 0 : i64, pattern_name = "gep->load"}> ({
// CHECK-ITER-MERGE-PATTERN-NEXT:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
// CHECK-ITER-MERGE-PATTERN-NEXT:       %72 = "neura.gep"(%arg5, %arg6, %arg7) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       %73 = "neura.load"(%72) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// CHECK-ITER-MERGE-PATTERN-NEXT:       neura.yield results(%72, %73 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
// CHECK-ITER-MERGE-PATTERN-NEXT:     }) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)

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

// CHECK-INIT-PATTERN:          %45:2 = "neura.fused_op"(%38, %36, %44) <{frequency = 4 : i64, pattern_id = 2 : i64, pattern_name = "gep->load"}> ({
// CHECK-INIT-PATTERN-NEXT:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
// CHECK-INIT-PATTERN-NEXT:       %81 = "neura.gep"(%arg5, %arg6, %arg7) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-INIT-PATTERN-NEXT:       %82 = "neura.load"(%81) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// CHECK-INIT-PATTERN-NEXT:       neura.yield results(%81, %82 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
// CHECK-INIT-PATTERN-NEXT:     }) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
// CHECK-INIT-PATTERN-NEXT:     %46 = "neura.fused_op"(%32, %44) <{frequency = 4 : i64, pattern_id = 2 : i64, pattern_name = "gep->load"}> ({
// CHECK-INIT-PATTERN-NEXT:     ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>):
// CHECK-INIT-PATTERN-NEXT:       %81 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-INIT-PATTERN-NEXT:       %82 = "neura.load"(%81) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// CHECK-INIT-PATTERN-NEXT:       neura.yield results(%82 : !neura.data<i32, i1>)
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

// CHECK-ITER-MERGE-PATTERN-MAPPING: /bin/sh: --: invalid option
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: Usage:	/bin/sh [GNU long option] [option] ...
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	/bin/sh [GNU long option] [option] script-file ...
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: GNU long options:
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	--debug
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	--debugger
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	--dump-po-strings
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	--dump-strings
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	--help
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	--init-file
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	--login
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	--noediting
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	--noprofile
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	--norc
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	--posix
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	--pretty-print
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	--rcfile
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	--restricted
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	--verbose
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	--version
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: Shell options:
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	-ilrsD or -c command or -O shopt_option		(invocation only)
// CHECK-ITER-MERGE-PATTERN-MAPPING-NEXT: 	-abefhkmnptuvxBCHP or -o option

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
// CHECK-HARDWARE-MERGE-NEXT:   "hardware_configuration": {
// CHECK-HARDWARE-MERGE-NEXT:     "summary": {
// CHECK-HARDWARE-MERGE-NEXT:       "total_templates": 3
// CHECK-HARDWARE-MERGE-NEXT:     },
// CHECK-HARDWARE-MERGE-NEXT:     "hardware_templates": [
// CHECK-HARDWARE-MERGE-NEXT:       {
// CHECK-HARDWARE-MERGE-NEXT:         "template_id": 0,
// CHECK-HARDWARE-MERGE-NEXT:         "instance_count": 2,
// CHECK-HARDWARE-MERGE-NEXT:         "supported_single_ops": ["neura.gep", "neura.grant_once", "neura.grant_predicate", "neura.load", "neura.phi_start", "neura.store"],
// CHECK-HARDWARE-MERGE-NEXT:         "supported_composite_ops": [
// CHECK-HARDWARE-MERGE-NEXT:           {"pattern_id": 9, "name": "phi_start->fused_op:gep->load"},
// CHECK-HARDWARE-MERGE-NEXT:           {"pattern_id": 0, "name": "gep->load"},
// CHECK-HARDWARE-MERGE-NEXT:           {"pattern_id": 11, "name": "phi_start->grant_predicate"}
// CHECK-HARDWARE-MERGE-NEXT:         ],
// CHECK-HARDWARE-MERGE-NEXT:         "functional_units": [
// CHECK-HARDWARE-MERGE-NEXT:           {"fu_id": 0, "op_type": "neura.phi_start"},
// CHECK-HARDWARE-MERGE-NEXT:           {"fu_id": 1, "op_type": "neura.gep"},
// CHECK-HARDWARE-MERGE-NEXT:           {"fu_id": 2, "op_type": "neura.load"},
// CHECK-HARDWARE-MERGE-NEXT:           {"fu_id": 3, "op_type": "neura.grant_predicate"}
// CHECK-HARDWARE-MERGE-NEXT:         ],
// CHECK-HARDWARE-MERGE-NEXT:         "fu_connections": [
// CHECK-HARDWARE-MERGE-NEXT:           {"from_fu": 0, "to_fu": 1},
// CHECK-HARDWARE-MERGE-NEXT:           {"from_fu": 0, "to_fu": 3},
// CHECK-HARDWARE-MERGE-NEXT:           {"from_fu": 1, "to_fu": 2}
// CHECK-HARDWARE-MERGE-NEXT:         ],
// CHECK-HARDWARE-MERGE-NEXT:         "pattern_execution_plans": [
// CHECK-HARDWARE-MERGE-NEXT:           {
// CHECK-HARDWARE-MERGE-NEXT:             "pattern_id": 9,
// CHECK-HARDWARE-MERGE-NEXT:             "pattern_name": "phi_start->fused_op:gep->load",
// CHECK-HARDWARE-MERGE-NEXT:             "fu_mapping": [0, 1, 2],
// CHECK-HARDWARE-MERGE-NEXT:             "execution_stages": [
// CHECK-HARDWARE-MERGE-NEXT:               {
// CHECK-HARDWARE-MERGE-NEXT:                 "stage": 0,
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_fus": [0],
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_ops": ["neura.phi_start"]
// CHECK-HARDWARE-MERGE-NEXT:               },
// CHECK-HARDWARE-MERGE-NEXT:               {
// CHECK-HARDWARE-MERGE-NEXT:                 "stage": 1,
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_fus": [1],
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_ops": ["neura.gep"]
// CHECK-HARDWARE-MERGE-NEXT:               },
// CHECK-HARDWARE-MERGE-NEXT:               {
// CHECK-HARDWARE-MERGE-NEXT:                 "stage": 2,
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_fus": [2],
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_ops": ["neura.load"]
// CHECK-HARDWARE-MERGE-NEXT:               }
// CHECK-HARDWARE-MERGE-NEXT:             ]
// CHECK-HARDWARE-MERGE-NEXT:           },
// CHECK-HARDWARE-MERGE-NEXT:           {
// CHECK-HARDWARE-MERGE-NEXT:             "pattern_id": 0,
// CHECK-HARDWARE-MERGE-NEXT:             "pattern_name": "gep->load",
// CHECK-HARDWARE-MERGE-NEXT:             "fu_mapping": [1, 2],
// CHECK-HARDWARE-MERGE-NEXT:             "execution_stages": [
// CHECK-HARDWARE-MERGE-NEXT:               {
// CHECK-HARDWARE-MERGE-NEXT:                 "stage": 0,
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_fus": [1],
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_ops": ["neura.gep"]
// CHECK-HARDWARE-MERGE-NEXT:               },
// CHECK-HARDWARE-MERGE-NEXT:               {
// CHECK-HARDWARE-MERGE-NEXT:                 "stage": 1,
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_fus": [2],
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_ops": ["neura.load"]
// CHECK-HARDWARE-MERGE-NEXT:               }
// CHECK-HARDWARE-MERGE-NEXT:             ]
// CHECK-HARDWARE-MERGE-NEXT:           },
// CHECK-HARDWARE-MERGE-NEXT:           {
// CHECK-HARDWARE-MERGE-NEXT:             "pattern_id": 11,
// CHECK-HARDWARE-MERGE-NEXT:             "pattern_name": "phi_start->grant_predicate",
// CHECK-HARDWARE-MERGE-NEXT:             "fu_mapping": [0, 3],
// CHECK-HARDWARE-MERGE-NEXT:             "execution_stages": [
// CHECK-HARDWARE-MERGE-NEXT:               {
// CHECK-HARDWARE-MERGE-NEXT:                 "stage": 0,
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_fus": [0],
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_ops": ["neura.phi_start"]
// CHECK-HARDWARE-MERGE-NEXT:               },
// CHECK-HARDWARE-MERGE-NEXT:               {
// CHECK-HARDWARE-MERGE-NEXT:                 "stage": 1,
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_fus": [3],
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_ops": ["neura.grant_predicate"]
// CHECK-HARDWARE-MERGE-NEXT:               }
// CHECK-HARDWARE-MERGE-NEXT:             ]
// CHECK-HARDWARE-MERGE-NEXT:           }
// CHECK-HARDWARE-MERGE-NEXT:         ]
// CHECK-HARDWARE-MERGE-NEXT:       },
// CHECK-HARDWARE-MERGE-NEXT:       {
// CHECK-HARDWARE-MERGE-NEXT:         "template_id": 1,
// CHECK-HARDWARE-MERGE-NEXT:         "instance_count": 2,
// CHECK-HARDWARE-MERGE-NEXT:         "supported_single_ops": ["neura.grant_once", "neura.grant_predicate", "neura.icmp"],
// CHECK-HARDWARE-MERGE-NEXT:         "supported_composite_ops": [
// CHECK-HARDWARE-MERGE-NEXT:           {"pattern_id": 2, "name": "fused_op:icmp->grant_predicate->grant_predicate"},
// CHECK-HARDWARE-MERGE-NEXT:           {"pattern_id": 3, "name": "icmp->grant_predicate"}
// CHECK-HARDWARE-MERGE-NEXT:         ],
// CHECK-HARDWARE-MERGE-NEXT:         "functional_units": [
// CHECK-HARDWARE-MERGE-NEXT:           {"fu_id": 0, "op_type": "neura.icmp"},
// CHECK-HARDWARE-MERGE-NEXT:           {"fu_id": 1, "op_type": "neura.grant_predicate"},
// CHECK-HARDWARE-MERGE-NEXT:           {"fu_id": 2, "op_type": "neura.grant_predicate"}
// CHECK-HARDWARE-MERGE-NEXT:         ],
// CHECK-HARDWARE-MERGE-NEXT:         "fu_connections": [
// CHECK-HARDWARE-MERGE-NEXT:           {"from_fu": 0, "to_fu": 1},
// CHECK-HARDWARE-MERGE-NEXT:           {"from_fu": 0, "to_fu": 2}
// CHECK-HARDWARE-MERGE-NEXT:         ],
// CHECK-HARDWARE-MERGE-NEXT:         "pattern_execution_plans": [
// CHECK-HARDWARE-MERGE-NEXT:           {
// CHECK-HARDWARE-MERGE-NEXT:             "pattern_id": 2,
// CHECK-HARDWARE-MERGE-NEXT:             "pattern_name": "fused_op:icmp->grant_predicate->grant_predicate",
// CHECK-HARDWARE-MERGE-NEXT:             "fu_mapping": [0, 1, 2],
// CHECK-HARDWARE-MERGE-NEXT:             "execution_stages": [
// CHECK-HARDWARE-MERGE-NEXT:               {
// CHECK-HARDWARE-MERGE-NEXT:                 "stage": 0,
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_fus": [0],
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_ops": ["neura.icmp"]
// CHECK-HARDWARE-MERGE-NEXT:               },
// CHECK-HARDWARE-MERGE-NEXT:               {
// CHECK-HARDWARE-MERGE-NEXT:                 "stage": 1,
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_fus": [1, 2],
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_ops": ["neura.grant_predicate", "neura.grant_predicate"]
// CHECK-HARDWARE-MERGE-NEXT:               }
// CHECK-HARDWARE-MERGE-NEXT:             ]
// CHECK-HARDWARE-MERGE-NEXT:           },
// CHECK-HARDWARE-MERGE-NEXT:           {
// CHECK-HARDWARE-MERGE-NEXT:             "pattern_id": 3,
// CHECK-HARDWARE-MERGE-NEXT:             "pattern_name": "icmp->grant_predicate",
// CHECK-HARDWARE-MERGE-NEXT:             "fu_mapping": [0, 1],
// CHECK-HARDWARE-MERGE-NEXT:             "execution_stages": [
// CHECK-HARDWARE-MERGE-NEXT:               {
// CHECK-HARDWARE-MERGE-NEXT:                 "stage": 0,
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_fus": [0],
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_ops": ["neura.icmp"]
// CHECK-HARDWARE-MERGE-NEXT:               },
// CHECK-HARDWARE-MERGE-NEXT:               {
// CHECK-HARDWARE-MERGE-NEXT:                 "stage": 1,
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_fus": [1],
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_ops": ["neura.grant_predicate"]
// CHECK-HARDWARE-MERGE-NEXT:               }
// CHECK-HARDWARE-MERGE-NEXT:             ]
// CHECK-HARDWARE-MERGE-NEXT:           }
// CHECK-HARDWARE-MERGE-NEXT:         ]
// CHECK-HARDWARE-MERGE-NEXT:       },
// CHECK-HARDWARE-MERGE-NEXT:       {
// CHECK-HARDWARE-MERGE-NEXT:         "template_id": 2,
// CHECK-HARDWARE-MERGE-NEXT:         "instance_count": 1,
// CHECK-HARDWARE-MERGE-NEXT:         "supported_single_ops": ["neura.grant_once", "neura.grant_predicate", "neura.phi_start"],
// CHECK-HARDWARE-MERGE-NEXT:         "supported_composite_ops": [
// CHECK-HARDWARE-MERGE-NEXT:           {"pattern_id": 5, "name": "grant_once->fused_op:phi_start->phi_start"},
// CHECK-HARDWARE-MERGE-NEXT:           {"pattern_id": 8, "name": "grant_once->phi_start"}
// CHECK-HARDWARE-MERGE-NEXT:         ],
// CHECK-HARDWARE-MERGE-NEXT:         "functional_units": [
// CHECK-HARDWARE-MERGE-NEXT:           {"fu_id": 0, "op_type": "neura.grant_once"},
// CHECK-HARDWARE-MERGE-NEXT:           {"fu_id": 1, "op_type": "neura.phi_start"},
// CHECK-HARDWARE-MERGE-NEXT:           {"fu_id": 2, "op_type": "neura.phi_start"}
// CHECK-HARDWARE-MERGE-NEXT:         ],
// CHECK-HARDWARE-MERGE-NEXT:         "fu_connections": [
// CHECK-HARDWARE-MERGE-NEXT:           {"from_fu": 0, "to_fu": 1},
// CHECK-HARDWARE-MERGE-NEXT:           {"from_fu": 1, "to_fu": 2}
// CHECK-HARDWARE-MERGE-NEXT:         ],
// CHECK-HARDWARE-MERGE-NEXT:         "pattern_execution_plans": [
// CHECK-HARDWARE-MERGE-NEXT:           {
// CHECK-HARDWARE-MERGE-NEXT:             "pattern_id": 8,
// CHECK-HARDWARE-MERGE-NEXT:             "pattern_name": "grant_once->phi_start",
// CHECK-HARDWARE-MERGE-NEXT:             "fu_mapping": [0, 1],
// CHECK-HARDWARE-MERGE-NEXT:             "execution_stages": [
// CHECK-HARDWARE-MERGE-NEXT:               {
// CHECK-HARDWARE-MERGE-NEXT:                 "stage": 0,
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_fus": [0],
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_ops": ["neura.grant_once"]
// CHECK-HARDWARE-MERGE-NEXT:               },
// CHECK-HARDWARE-MERGE-NEXT:               {
// CHECK-HARDWARE-MERGE-NEXT:                 "stage": 1,
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_fus": [1],
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_ops": ["neura.phi_start"]
// CHECK-HARDWARE-MERGE-NEXT:               }
// CHECK-HARDWARE-MERGE-NEXT:             ]
// CHECK-HARDWARE-MERGE-NEXT:           },
// CHECK-HARDWARE-MERGE-NEXT:           {
// CHECK-HARDWARE-MERGE-NEXT:             "pattern_id": 5,
// CHECK-HARDWARE-MERGE-NEXT:             "pattern_name": "grant_once->fused_op:phi_start->phi_start",
// CHECK-HARDWARE-MERGE-NEXT:             "fu_mapping": [0, 1, 2],
// CHECK-HARDWARE-MERGE-NEXT:             "execution_stages": [
// CHECK-HARDWARE-MERGE-NEXT:               {
// CHECK-HARDWARE-MERGE-NEXT:                 "stage": 0,
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_fus": [0],
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_ops": ["neura.grant_once"]
// CHECK-HARDWARE-MERGE-NEXT:               },
// CHECK-HARDWARE-MERGE-NEXT:               {
// CHECK-HARDWARE-MERGE-NEXT:                 "stage": 1,
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_fus": [1],
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_ops": ["neura.phi_start"]
// CHECK-HARDWARE-MERGE-NEXT:               },
// CHECK-HARDWARE-MERGE-NEXT:               {
// CHECK-HARDWARE-MERGE-NEXT:                 "stage": 2,
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_fus": [2],
// CHECK-HARDWARE-MERGE-NEXT:                 "parallel_ops": ["neura.phi_start"]
// CHECK-HARDWARE-MERGE-NEXT:               }
// CHECK-HARDWARE-MERGE-NEXT:             ]
// CHECK-HARDWARE-MERGE-NEXT:           }
// CHECK-HARDWARE-MERGE-NEXT:         ]
// CHECK-HARDWARE-MERGE-NEXT:       }
// CHECK-HARDWARE-MERGE-NEXT:     ]
// CHECK-HARDWARE-MERGE-NEXT:   }
// CHECK-HARDWARE-MERGE-NEXT: }
