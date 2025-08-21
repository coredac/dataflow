// RUN: neura-interpreter %s --verbose | FileCheck %s

func.func @test_grant_predicate() -> vector<4xf32> {
  %val = "neura.constant"() {
    value = dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf32>,
    predicate = true
  } : () -> vector<4xf32>

  %pred = arith.constant 1 : i1

  %res = "neura.grant_predicate"(%val, %pred) :
           (vector<4xf32>, i1) -> vector<4xf32>
  // CHECK: [neura-interpreter]  → Output: [1.000000, 2.000000, 3.000000, 4.000000]
  return %res : vector<4xf32>
}

func.func @test_grant_once() -> vector<2xf32> {
  %val = "neura.constant"() {
    value = dense<[5.5, 6.5]> : vector<2xf32>
  } : () -> vector<2xf32>

  %res = "neura.grant_once"(%val) : (vector<2xf32>) -> vector<2xf32>
  // CHECK: [neura-interpreter]  → Output: [5.500000, 6.500000]
  return %res : vector<2xf32>
}

func.func @test_grant_always() -> vector<3xf32> {
  %val = "neura.constant"() {
    value = dense<[10.0, 20.0, 30.0]> : vector<3xf32>,
    predicate = false
  } : () -> vector<3xf32>

  %res = "neura.grant_always"(%val) : (vector<3xf32>) -> vector<3xf32>

  // CHECK: [neura-interpreter]  → Output: [10.000000, 20.000000, 30.000000]
  return %res : vector<3xf32>
}

func.func @test_combined_grants() -> vector<4xf32> {
  %v = "neura.constant"() {
    value = dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf32>,
    predicate = true
  } : () -> vector<4xf32>

  %p0 = arith.constant 0 : i1
  %p1 = arith.constant 1 : i1

  %v_once = "neura.grant_once"(%v) : (vector<4xf32>) -> vector<4xf32>
  %v_pred_true = "neura.grant_predicate"(%v_once, %p1) : (vector<4xf32>, i1) -> vector<4xf32>
  %v_final = "neura.grant_always"(%v_pred_true) : (vector<4xf32>) -> vector<4xf32>

  // CHECK: [neura-interpreter]  → Output: [1.000000, 2.000000, 3.000000, 4.000000]
  return %v_final : vector<4xf32>
}

func.func @test_combined_grants_blocked() -> vector<4xf32> {
  %v = "neura.constant"() {
    value = dense<[5.0, 6.0, 7.0, 8.0]> : vector<4xf32>,
    predicate = true
  } : () -> vector<4xf32>

  %p0 = arith.constant 0 : i1
  %v_once = "neura.grant_once"(%v) : (vector<4xf32>) -> vector<4xf32>
  %v_pred = "neura.grant_predicate"(%v_once, %p0) : (vector<4xf32>, i1) -> vector<4xf32>
  %v_final = "neura.grant_always"(%v_pred) : (vector<4xf32>) -> vector<4xf32>
  // CHECK: [neura-interpreter]  → Output: [5.000000, 6.000000, 7.000000, 8.000000]
  return %v_final : vector<4xf32>
}