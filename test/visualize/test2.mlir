// Test PrintOpGraphPass in neura-compiler
// RUN: neura-compiler --neura-conversion %s
// RUN: FileCheck %s --input-file=%S/opgraph.dot -check-prefix=CHECK-GRAPH

func.func @test_print_op_graph(%a: f32, %b: f32) -> f32 {
  %c = arith.constant 1.0 : f32
  %d = arith.addf %a, %b : f32
  %e = arith.mulf %d, %c : f32
  return %e : f32
}

// CHECK-GRAPH: digraph G
// CHECK-GRAPH: compound = true;
// CHECK-GRAPH: label = "func.func : ()\n\naccelerator: \"neura\"\nfunction_type: (f32, f32) -> f32\nsym_name: \"test_print_op_graph...";
// CHECK-GRAPH: label = "neura.fadd : (f32)\n"
// CHECK-GRAPH: label = "neura.return : ()\n"
// CHECK-GRAPH: digraph G
// CHECK-GRAPH: label = "func.func : ()\n\naccelerator: \"neura\"\nfunction_type: (f32, f32) -> f32\nsym_name: \"test_print_op_graph...";
// CHECK-GRAPH: label = "neura.constant : (!neura.data<f32, i1>)
// CHECK-GRAPH: label = "neura.fadd : (!neura.data<f32, i1>)\n"
// CHECK-GRAPH: digraph G
// CHECK-GRAPH: label = "func.func : ()\n\naccelerator: \"neura\"\nfunction_type: (f32, f32) -> f32\nsym_name: \"test_print_op_graph...";
// CHECK-GRAPH: label = "neura.constant : (!neura.data<f32, i1>)
// CHECK-GRAPH: label = "neura.data_mov : (!neura.data<f32, i1>)
// CHECK-GRAPH: label = "neura.fadd : (!neura.data<f32, i1>)\n"
