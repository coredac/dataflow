// RUN: neura-interpreter %s | FileCheck %s

func.func @test_vfmul_basic() -> vector<2xf32> {
  %a = "neura.constant"() {value = dense<[2.0, 3.0]> : vector<2xf32>} : () -> vector<2xf32>
  %b = "neura.constant"() {value = dense<[4.0, 5.0]> : vector<2xf32>} : () -> vector<2xf32>
  %res = "neura.vfmul"(%a, %b) : (vector<2xf32>, vector<2xf32>) -> vector<2xf32>
  // CHECK: "neura.constant"
  // CHECK-NEXT: "neura.constant"
  // CHECK-NEXT: "neura.vfmul"
  // CHECK-NEXT: Lhs: 2.000000 3.000000 
  // CHECK-NEXT: Rhs: 4.000000 5.000000 
  // CHECK-NEXT: Result: 8.000000 15.000000 
  // CHECK-NEXT: VFMul: vector size=2, predicate=true
  // CHECK-NEXT: DEBUG: Return value is vector: YES
  // CHECK-NEXT: [neura-interpreter] Output: [8.000000, 15.000000]
  return %res : vector<2xf32>
}

func.func @test_vfmul_with_valid_predicate() -> vector<3xf32> {
  %a = "neura.constant"() {value = dense<[6.0, 7.0, 8.0]> : vector<3xf32>} : () -> vector<3xf32>
  %b = "neura.constant"() {value = dense<[0.5, 2.0, 0.1]> : vector<3xf32>} : () -> vector<3xf32>
  %pred = arith.constant 1 : i1
  %pred_as_f32 = "neura.cast"(%pred) {cast_type = "bool2f"} : (i1) -> f32
  %res = "neura.vfmul"(%a, %b, %pred_as_f32) : (vector<3xf32>, vector<3xf32>, f32) -> vector<3xf32>
  // CHECK: "neura.constant"
  // CHECK-NEXT: "neura.constant"
  // CHECK-NEXT: "neura.cast"
  // CHECK-NEXT: "neura.vfmul"
  // CHECK-NEXT: Lhs: 6.000000 7.000000 8.000000 
  // CHECK-NEXT: Rhs: 0.500000 2.000000 0.100000 
  // CHECK-NEXT: Result: 3.000000 14.000000 0.800000 
  // CHECK-NEXT: VFMul: vector size=3, predicate=true
  // CHECK-NEXT: DEBUG: Return value is vector: YES
  // CHECK-NEXT: [neura-interpreter] Output: [3.000000, 14.000000, 0.800000]
  return %res : vector<3xf32>
}