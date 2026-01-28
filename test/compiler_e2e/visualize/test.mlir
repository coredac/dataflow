// Test neura-compiler e2e pipeline
// RUN: neura-compiler --neura-conversion %s --architecture-spec=%S/../../arch_spec/architecture.yaml -o %t-mapping.mlir
// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml -check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm -check-prefix=ASM
// RUN: FileCheck %s --input-file=tmp-generated-dfg.yaml -check-prefix=DFG
// RUN: FileCheck %s --input-file=%S/opgraph.dot -check-prefix=CHECK-GRAPH

func.func @test_print_op_graph(%a: f32, %b: f32) -> f32 {
  %c = arith.constant 2.0 : f32
  %d = arith.addf %a, %b : f32
  %e = arith.mulf %d, %c : f32
  return %e : f32
}

// MAPPING:    module
// MAPPING:    func.func @test_print_op_graph(%arg0: f32, %arg1: f32) -> f32 attributes {accelerator = "neura", dataflow_mode = "predicate", mapping_info = {compiled_ii = 1 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 1 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}}
// MAPPING:    %0 = "neura.constant"() <{value = "%arg0"}> {dfg_id = 0 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<f32, i1>
// MAPPING:    %1 = "neura.data_mov"(%0) {dfg_id = 2 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING:    %2 = "neura.fadd"(%1) {dfg_id = 3 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 0 : i32}], rhs_value = "%arg1"} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING:    %3 = "neura.data_mov"(%2) {dfg_id = 4 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING:    %4 = "neura.fmul"(%3) {dfg_id = 5 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 0 : i32}], rhs_value = 2.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING:    %5 = "neura.data_mov"(%4) {dfg_id = 6 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING:    neura.return_value %5 : !neura.data<f32, i1> {dfg_id = 7 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 0 : i32}]}
// MAPPING:    neura.yield {dfg_id = 1 : i32}

// YAML: array_config:
// YAML:   columns: 4
// YAML:   rows: 4
// YAML:   compiled_ii: 1
// YAML:   cores:
// YAML:     - column: 0
// YAML:       row: 0
// YAML:       core_id: "0"
// YAML:       entries:
// YAML:         - entry_id: "entry0"
// YAML:           instructions:
// YAML:             - index_per_ii: 0
// YAML:               operations:
// YAML:                 - opcode: "CONSTANT"
// YAML:                   id: 0
// YAML:                   time_step: 0
// YAML:                   invalid_iterations: 0
// YAML:                   src_operands:
// YAML:                     - operand: "arg0"
// YAML:                   dst_operands:
// YAML:                     - operand: "EAST"
// YAML:     - column: 1
// YAML:       row: 0
// YAML:       core_id: "1"
// YAML:       entries:
// YAML:         - entry_id: "entry0"
// YAML:           instructions:
// YAML:             - index_per_ii: 0
// YAML:               operations:
// YAML:                 - opcode: "FADD"
// YAML:                   id: 3
// YAML:                   time_step: 1
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "WEST"
// YAML:                     - operand: "arg1"
// YAML:                   dst_operands:
// YAML:                     - operand: "EAST"
// YAML:     - column: 2
// YAML:       row: 0
// YAML:       core_id: "2"
// YAML:       entries:
// YAML:         - entry_id: "entry0"
// YAML:           instructions:
// YAML:             - index_per_ii: 0
// YAML:               operations:
// YAML:                 - opcode: "FMUL"
// YAML:                   id: 5
// YAML:                   time_step: 2
// YAML:                   invalid_iterations: 2
// YAML:                   src_operands:
// YAML:                     - operand: "WEST"
// YAML:                     - operand: "#2.000000"
// YAML:                   dst_operands:
// YAML:                     - operand: "EAST"
// YAML:     - column: 3
// YAML:       row: 0
// YAML:       core_id: "3"
// YAML:       entries:
// YAML:         - entry_id: "entry0"
// YAML:           instructions:
// YAML:             - index_per_ii: 0
// YAML:               operations:
// YAML:                 - opcode: "RETURN_VALUE"
// YAML:                   id: 7
// YAML:                   time_step: 3
// YAML:                   invalid_iterations: 3
// YAML:                   src_operands:
// YAML:                     - operand: "WEST"

// DFG: nodes:
// DFG:   - id: 0
// DFG:     opcode: "CONSTANT"
// DFG:     tile_x: 0
// DFG:     tile_y: 0
// DFG:     time_step: 0
// DFG:   - id: 3
// DFG:     opcode: "FADD"
// DFG:     tile_x: 1
// DFG:     tile_y: 0
// DFG:     time_step: 1
// DFG:   - id: 5
// DFG:     opcode: "FMUL"
// DFG:     tile_x: 2
// DFG:     tile_y: 0
// DFG:     time_step: 2
// DFG:   - id: 7
// DFG:     opcode: "RETURN_VALUE"
// DFG:     tile_x: 3
// DFG:     tile_y: 0
// DFG:     time_step: 3
// DFG: edges:
// DFG:   - from: 5
// DFG:     to: 7
// DFG:   - from: 3
// DFG:     to: 5
// DFG:   - from: 0
// DFG:     to: 3

// ASM:  Compiled II: 1
// ASM:  PE(0,0):
// ASM:  {
// ASM:    CONSTANT, [arg0] -> [EAST, RED] (t=0, inv_iters=0)
// ASM:  } (idx_per_ii=0)
// ASM:  PE(1,0):
// ASM:  {
// ASM:    FADD, [WEST, RED], [arg1] -> [EAST, RED] (t=1, inv_iters=1)
// ASM:  } (idx_per_ii=0)
// ASM:  PE(2,0):
// ASM:  {
// ASM:    FMUL, [WEST, RED], [#2.000000] -> [EAST, RED] (t=2, inv_iters=2)
// ASM:  } (idx_per_ii=0)
// ASM:  PE(3,0):
// ASM:  {
// ASM:    RETURN_VALUE, [WEST, RED] (t=3, inv_iters=3)
// ASM:  } (idx_per_ii=0)


// CHECK-GRAPH: digraph G
// CHECK-GRAPH: label = "neura.constant : (f32)\n\nvalue: 2.000000e+00 : f32", shape = ellipse, style = filled];
// CHECK-GRAPH: label = "neura.fadd : (f32)\n", shape = ellipse, style = filled];
// CHECK-GRAPH: label = "neura.fmul : (f32)\n", shape = ellipse, style = filled];
// CHECK-GRAPH: label = "neura.return : ()\n", shape = ellipse, style = filled];

// CHECK-GRAPH: digraph G
// CHECK-GRAPH: label = "neura.constant : (!neura.data<f32, i1>)\n\nvalue: \"%arg0\"", shape = ellipse, style = filled];
// CHECK-GRAPH: label = "neura.fadd : (!neura.data<f32, i1>)\n\nrhs_value: \"%arg1\"", shape = ellipse, style = filled];
// CHECK-GRAPH: label = "neura.fmul : (!neura.data<f32, i1>)\n\nrhs_value: 2.000000e+00 : f32", shape = ellipse, style = filled];
// CHECK-GRAPH: label = "neura.return : ()\n\nreturn_type: \"value\"", shape = ellipse, style = filled];

// CHECK-GRAPH: digraph G
// CHECK-GRAPH: label = "neura.constant : (!neura.data<f32, i1>)\n\nvalue: \"%arg0\"", shape = ellipse, style = filled];
// CHECK-GRAPH: label = "neura.data_mov : (!neura.data<f32, i1>)\n", shape = ellipse, style = filled];
// CHECK-GRAPH: label = "neura.fadd : (!neura.data<f32, i1>)\n\nrhs_value: \"%arg1\"", shape = ellipse, style = filled];
// CHECK-GRAPH: label = "neura.data_mov : (!neura.data<f32, i1>)\n", shape = ellipse, style = filled];
// CHECK-GRAPH: label = "neura.fmul : (!neura.data<f32, i1>)\n\nrhs_value: 2.000000e+00 : f32", shape = ellipse, style = filled];
// CHECK-GRAPH: label = "neura.data_mov : (!neura.data<f32, i1>)\n", shape = ellipse, style = filled];
// CHECK-GRAPH: label = "neura.return_value : ()\n", shape = ellipse, style = filled];
// CHECK-GRAPH: label = "neura.yield : ()\n", shape = ellipse, style = filled];
