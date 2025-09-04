// Test 1: Verify dot generation for fused operations
// RUN: mlir-neura-opt --assign-accelerator --lower-arith-to-neura --fuse-pattern --view-op-graph %s 2>&1 | FileCheck %s --check-prefix=CHECK-DOT-GENERATION

func.func @test_dot_generation(%a: f32, %b: f32, %c: f32) -> f32 {
  %temp = arith.addf %a, %b : f32
  %res = arith.addf %temp, %c : f32
  // CHECK-DOT-GENERATION: digraph G
  // CHECK-DOT-GENERATION: label = "func.func : ()\n\naccelerator: \"neura\"\nfunction_type: (f32, f32, f32) -> f...\nsym_name: \"test_dot_generation...";
  // CHECK-DOT-GENERATION: label = "neura.fadd_fadd : (f32)\n"
  return %res : f32
}

