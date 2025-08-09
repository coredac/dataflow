// Test cases for FusePatternsPass
// Tests both FAddFAdd and FMulFAdd fusion patterns

// Test 1: Basic FAddFAdd fusion (LHS fadd)
// RUN: mlir-neura-opt --assign-accelerator --lower-arith-to-neura --fuse-patterns %s | FileCheck %s --check-prefix=CHECK-FADD-FADD-LHS

func.func @test_fadd_fadd_lhs(%a: f32, %b: f32, %c: f32) -> f32 {
  %temp = arith.addf %a, %b : f32
  %res = arith.addf %temp, %c : f32
  // CHECK-FADD-FADD-LHS: neur1ielZH^1a.fadd_fadd
  return %res : f32
}

// Test 2: Basic FAddFAdd fusion (RHS fadd)
// RUN: mlir-neura-opt --assign-accelerator --lower-arith-to-neura --fuse-patterns %s | FileCheck %s --check-prefix=CHECK-FADD-FADD-RHS

func.func @test_fadd_fadd_rhs(%a: f32, %b: f32, %c: f32) -> f32 {
  %temp = arith.addf %b, %c : f32
  %res = arith.addf %a, %temp : f32
  // CHECK-FADD-FADD-RHS: neura.fadd_fadd
  return %res : f32
}

// Test 3: Basic FMulFAdd fusion (LHS fmul)
// RUN: mlir-neura-opt --assign-accelerator --lower-arith-to-neura --fuse-patterns %s | FileCheck %s --check-prefix=CHECK-FMUL-FADD-LHS

func.func @test_fmul_fadd_lhs(%a: f32, %b: f32, %c: f32) -> f32 {
  %temp = arith.mulf %a, %b : f32
  %res = arith.addf %temp, %c : f32
  // CHECK-FMUL-FADD-LHS: neura.fmul_fadd
  return %res : f32
}

// Test 4: Basic FMulFAdd fusion (RHS fmul)
// RUN: mlir-neura-opt --assign-accelerator --lower-arith-to-neura --fuse-patterns %s | FileCheck %s --check-prefix=CHECK-FMUL-FADD-RHS

func.func @test_fmul_fadd_rhs(%a: f32, %b: f32, %c: f32) -> f32 {
  %temp = arith.mulf %b, %c : f32
  %res = arith.addf %a, %temp : f32
  // CHECK-FMUL-FADD-RHS: neura.fmul_fadd
  return %res : f32
}

// Test 5: No fusion when first fadd has multiple uses
// RUN: mlir-neura-opt --assign-accelerator --lower-arith-to-neura --fuse-patterns %s | FileCheck %s --check-prefix=CHECK-NO-FUSION-MULTI-USE

func.func @test_no_fusion_multi_use(%a: f32, %b: f32, %c: f32) -> f32 {
  %temp = arith.addf %a, %b : f32
  %res1 = arith.addf %temp, %c : f32
  %res2 = arith.addf %temp, %c : f32  // Second use of %temp
  // CHECK-NO-FUSION-MULTI-USE-NOT: neura.fadd_fadd
  // CHECK-NO-FUSION-MULTI-USE: neura.fadd
  return %res1 : f32
}

// Test 6: No fusion when first fmul has multiple uses
// RUN: mlir-neura-opt --assign-accelerator --lower-arith-to-neura --fuse-patterns %s | FileCheck %s --check-prefix=CHECK-NO-FMUL-MULTI-USE

func.func @test_no_fmul_fusion_multi_use(%a: f32, %b: f32, %c: f32) -> f32 {
  %temp = arith.mulf %a, %b : f32
  %res1 = arith.addf %temp, %c : f32
  %res2 = arith.addf %temp, %c : f32  // Second use of %temp
  // CHECK-NO-FMUL-MULTI-USE-NOT: neura.fmul_fadd
  // CHECK-NO-FMUL-MULTI-USE: neura.fmul
  return %res1 : f32
}

// Test 7: Complex chain with multiple fusion opportunities
// RUN: mlir-neura-opt --assign-accelerator --lower-arith-to-neura --fuse-patterns %s | FileCheck %s --check-prefix=CHECK-COMPLEX-CHAIN

func.func @test_complex_chain(%a: f32, %b: f32, %c: f32, %d: f32) -> f32 {
  %temp1 = arith.addf %a, %b : f32
  %temp2 = arith.addf %temp1, %c : f32
  %temp3 = arith.mulf %temp2, %d : f32
  %res = arith.addf %temp3, %a : f32
  // CHECK-COMPLEX-CHAIN: neura.fadd_fadd
  // CHECK-COMPLEX-CHAIN: neura.fmul_fadd
  return %res : f32
}

// Test 8: No fusion when operands are not the expected operations
// RUN: mlir-neura-opt --assign-accelerator --lower-arith-to-neura --fuse-patterns %s | FileCheck %s --check-prefix=CHECK-NO-FUSION-WRONG-OPS

func.func @test_no_fusion_wrong_ops(%a: f32, %b: f32, %c: f32) -> f32 {
  %temp1 = arith.subf %a, %b : f32  // Using subf instead of addf
  %res = arith.addf %temp1, %c : f32
  // CHECK-NO-FUSION-WRONG-OPS-NOT: neura.fadd_fadd
  // CHECK-NO-FUSION-WRONG-OPS: neura.fsub
  // CHECK-NO-FUSION-WRONG-OPS: neura.fadd
  return %res : f32
}

// Test 9: Multiple fusion patterns in sequence
// RUN: mlir-neura-opt --assign-accelerator --lower-arith-to-neura --fuse-patterns %s | FileCheck %s --check-prefix=CHECK-MULTIPLE-FUSION

func.func @test_multiple_fusion(%a: f32, %b: f32, %c: f32, %d: f32) -> f32 {
  %temp1 = arith.addf %a, %b : f32
  %temp2 = arith.addf %temp1, %c : f32
  %temp3 = arith.mulf %temp2, %d : f32
  %res = arith.addf %temp3, %a : f32
  // CHECK-MULTIPLE-FUSION: neura.fadd_fadd
  // CHECK-MULTIPLE-FUSION: neura.fmul_fadd
  return %res : f32
}

