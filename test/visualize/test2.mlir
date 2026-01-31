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
// CHECK-GRAPH: label = "neura.yield : ()\n\noperandSegmentSizes: array<i32: 0, 0>", shape = ellipse, style = filled];
