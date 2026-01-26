// Test PrintOpGraphPass in neura-compiler
// RUN: neura-compiler --neura-conversion %s --architecture-spec=%S/../arch_spec/architecture.yaml
// RUN: FileCheck %s --input-file=%S/opgraph.dot -check-prefix=CHECK-GRAPH

func.func @test_print_op_graph(%a: f32, %b: f32) -> f32 {
  %c = arith.constant 2.0 : f32
  %d = arith.addf %a, %b : f32
  %e = arith.mulf %d, %c : f32
  return %e : f32
}

// CHECK-GRAPH: digraph G
// CHECK_GRAPH: label = "neura.constant : (!neura.data<f32, i1>)\n\ndfg_id: 0 : i32\nmapping_locs: [{id = 0 : i32, inde...\nvalue: \"%arg0\"", shape = ellipse, style = filled];
// CHECK_GRAPH: label = "neura.data_mov : (!neura.data<f32, i1>)\n\ndfg_id: 2 : i32\nmapping_locs: [{id = 0 : i32, inde...", shape = ellipse, style = filled];
// CHECK_GRAPH: label = "neura.fadd : (!neura.data<f32, i1>)\n\ndfg_id: 3 : i32\nmapping_locs: [{id = 1 : i32, inde...\nrhs_value: \"%arg1\"", shape = ellipse, style = filled];
// CHECK_GRAPH: label = "neura.data_mov : (!neura.data<f32, i1>)\n\ndfg_id: 4 : i32\nmapping_locs: [{id = 3 : i32, inde...", shape = ellipse, style = filled];
// CHECK_GRAPH: label = "neura.fmul : (!neura.data<f32, i1>)\n\ndfg_id: 5 : i32\nmapping_locs: [{id = 2 : i32, inde...\nrhs_value: 2.000000e+00 : f32", shape = ellipse, style = filled];
// CHECK_GRAPH: label = "neura.data_mov : (!neura.data<f32, i1>)\n\ndfg_id: 6 : i32\nmapping_locs: [{id = 6 : i32, inde...", shape = ellipse, style = filled];
// CHECK_GRAPH: label = "neura.return_value : ()\n\ndfg_id: 7 : i32\nmapping_locs: [{id = 3 : i32, inde...", shape = ellipse, style = filled];
// CHECK_GRAPH: label = "neura.yield : ()\n\ndfg_id: 1 : i32", shape = ellipse, style = filled];
