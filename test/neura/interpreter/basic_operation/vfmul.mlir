// RUN: neura-interpreter %s --verbose | FileCheck %s

func.func @test_vfmul_basic() -> vector<2xf32> {
  %a = "neura.constant"() {value = dense<[2.0, 3.0]> : vector<2xf32>} : () -> vector<2xf32>
  %b = "neura.constant"() {value = dense<[4.0, 5.0]> : vector<2xf32>} : () -> vector<2xf32>
  %res = "neura.vfmul"(%a, %b) : (vector<2xf32>, vector<2xf32>) -> vector<2xf32>

  // CHECK: [neura-interpreter]  → Output: [8.000000, 15.000000]

  return %res : vector<2xf32>
}

func.func @test_vfmul_with_invalid_embed_predicate() -> vector<3xf32> {
  %a = "neura.constant"() {value = dense<[6.0, 7.0, 8.0]> : vector<3xf32>, predicate = false} : () -> vector<3xf32>
  %b = "neura.constant"() {value = dense<[0.5, 2.0, 0.1]> : vector<3xf32>, predicate = false} : () -> vector<3xf32>
  %res = "neura.vfmul"(%a, %b) : (vector<3xf32>, vector<3xf32>) -> vector<3xf32>

  // CHECK: [neura-interpreter]  → Output: [0.000000, 0.000000, 0.000000]

  return %res : vector<3xf32>
}