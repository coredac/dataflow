// RUN: neura-interpreter %s | FileCheck %s

func.func @test_vfmul_basic() -> vector<2xf32> {
  %a = "neura.constant"() {value = dense<[2.0, 3.0]> : vector<2xf32>} : () -> vector<2xf32>
  %b = "neura.constant"() {value = dense<[4.0, 5.0]> : vector<2xf32>} : () -> vector<2xf32>
  %res = "neura.vfmul"(%a, %b) : (vector<2xf32>, vector<2xf32>) -> vector<2xf32>
  
  // CHECK: [neura-interpreter] Executing neura.vfmul:
  // CHECK: LHS: vector size = 2, predicate = 1
  // CHECK: RHS: vector size = 2, predicate = 1
  // CHECK: Vector data:
  // CHECK: LHS: [2.000000e+00, 3.000000e+00]
  // CHECK: RHS: [4.000000e+00, 5.000000e+00]
  // CHECK: Result: [8.000000e+00, 1.500000e+01]
  // CHECK: Final result: vector size = 2, predicate = true
  // CHECK: [neura-interpreter] Output: [8.000000, 15.000000]
  
  return %res : vector<2xf32>
}

func.func @test_vfmul_with_valid_predicate() -> vector<3xf32> {
  %a = "neura.constant"() {value = dense<[6.0, 7.0, 8.0]> : vector<3xf32>} : () -> vector<3xf32>
  %b = "neura.constant"() {value = dense<[0.5, 2.0, 0.1]> : vector<3xf32>} : () -> vector<3xf32>
  %pred = arith.constant 1 : i1
  %pred_as_f32 = "neura.cast"(%pred) {cast_type = "bool2f"} : (i1) -> f32
  %res = "neura.vfmul"(%a, %b, %pred_as_f32) : (vector<3xf32>, vector<3xf32>, f32) -> vector<3xf32>
  
  // CHECK: [neura-interpreter] Executing neura.cast:
  // CHECK: Input: value = 1.000000e+00, predicate = 1
  // CHECK: Final result: value = 1.000000e+00, predicate = true
  
  // CHECK: [neura-interpreter] Executing neura.vfmul:
  // CHECK: LHS: vector size = 3, predicate = 1
  // CHECK: RHS: vector size = 3, predicate = 1
  // CHECK: Predicate: value = 1.000000e+00, predicate = 1
  // CHECK: Vector data:
  // CHECK: LHS: [6.000000e+00, 7.000000e+00, 8.000000e+00]
  // CHECK: RHS: [5.000000e-01, 2.000000e+00, 1.000000e-01]
  // CHECK: Result: [3.000000e+00, 1.400000e+01, 8.000000e-01]
  // CHECK: Final result: vector size = 3, predicate = true
  // CHECK: [neura-interpreter] Output: [3.000000, 14.000000, 0.800000]
  
  return %res : vector<3xf32>
}