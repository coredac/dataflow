// RUN: neura-interpreter %s | FileCheck %s

module {
  func.func @test_phi() -> f32 {
    %true_path_val = "neura.constant"() <{
      value = 10.0 : f32,
      predicate = true
    }> : () -> f32

    %false_path_val = "neura.constant"() <{
      value = 20.0 : f32,
      predicate = false
    }> : () -> f32

    %result = "neura.phi"(%true_path_val, %false_path_val) : (f32, f32) -> f32

    return %result : f32
    // CHECK: [neura-interpreter] Output: 10.000000
  }
}