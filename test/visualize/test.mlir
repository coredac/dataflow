// Test cases for FusePatternsPass

// Test 1: Verify dot generation for fused operations
// RUN: mlir-neura-opt --assign-accelerator --lower-arith-to-neura --fuse-patterns --generate-dot %s | FileCheck %s --check-prefix=CHECK-DOT-GENERATION

func.func @test_dot_generation(%a: f32, %b: f32, %c: f32) -> f32 {
  %temp = arith.addf %a, %b : f32
  %res = arith.addf %temp, %c : f32
  // CHECK-DOT-GENERATION: // DOT: digraph
  // CHECK-DOT-GENERATION: // DOT: "fadd_fadd"
  return %res : f32
}

