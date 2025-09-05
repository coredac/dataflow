// RUN: neura-interpreter %s --verbose | FileCheck %s

// (2.0 * 3.0) + 4.0 = 10.0
func.func @test_fmul_fadd_basic() -> f32 {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 3.0 : f32
  %c = arith.constant 4.0 : f32
  %res = "neura.fmul_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 10.000000
  return %res : f32
}

// (5.0 * (-2.0)) + 12.0 = 2.0
func.func @test_fmul_fadd_negative() -> f32 {
  %a = arith.constant 5.0 : f32
  %b = arith.constant -2.0 : f32
  %c = arith.constant 12.0 : f32
  %res = "neura.fmul_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 2.000000
  return %res : f32
}

// (0.0 * 5.0) + 6.0 = 6.0
func.func @test_fmul_fadd_zero() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 5.0 : f32
  %c = arith.constant 6.0 : f32
  %res = "neura.fmul_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 6.000000
  return %res : f32
}

// (1.5 * 2.0) + 3.5 = 6.5
func.func @test_fmul_fadd_decimal() -> f32 {
  %a = arith.constant 1.5 : f32
  %b = arith.constant 2.0 : f32
  %c = arith.constant 3.5 : f32
  %res = "neura.fmul_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 6.500000
  return %res : f32
}

// (4.0 * 2.0) + 1.0 = 9.0
func.func @test_fmul_fadd_with_valid_predicate() -> f32 {
  %a = arith.constant 4.0 : f32
  %b = arith.constant 2.0 : f32
  %c = arith.constant 1.0 : f32
  %pred = arith.constant 1 : i1  // predicate 为 true
  %pred_f32 = "neura.cast"(%pred) {cast_type = "bool2f"} : (i1) -> f32
  %res = "neura.fmul_fadd"(%a, %b, %c, %pred_f32) : (f32, f32, f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 9.000000
  return %res : f32
}

func.func @test_fmul_fadd_with_invalid_predicate() -> f32 {
  %a = arith.constant 3.0 : f32
  %b = arith.constant 3.0 : f32
  %c = arith.constant 3.0 : f32
  %pred = arith.constant 0 : i1
  %pred_f32 = "neura.cast"(%pred) {cast_type = "bool2f"} : (i1) -> f32
  %res = "neura.fmul_fadd"(%a, %b, %c, %pred_f32) : (f32, f32, f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : f32
}

func.func @test_fmul_fadd_with_invalid_input_predicate() -> f32 {
  %a = arith.constant 2.0 : f32
  %b = "neura.constant"() {value = 5.0 : f32, predicate = false} : () -> f32
  %c = arith.constant 3.0 : f32
  %res = "neura.fmul_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : f32
}