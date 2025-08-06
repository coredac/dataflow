// RUN: neura-interpreter %s | FileCheck %s

func.func @test_br_with_args() -> i32 {
  %0 = "neura.constant"() {value = 42 : i32} : () -> i32
  "neura.br"(%0) [^bb1] {operandSegmentSizes = array<i32: 1>} : (i32) -> ()

  ^bb1(%a: i32):
  // CHECK: [neura-interpreter]  → Output: 42.000000
  return %a : i32
}

func.func @test_br_with_multi_args() {
  %0 = "neura.constant"() {value = 42 : i32} : () -> i32
  %1 = "neura.constant"() {value = 1.0 : f32} : () -> f32
  "neura.br"(%0, %1) [^bb1] {operandSegmentSizes = array<i32: 2>} : (i32, f32) -> ()
  
  ^bb1(%a: i32, %b: f32):
  "neura.add"(%a, %a) : (i32, i32) -> i32
  // CHECK-NEXT: [neura-interpreter]  → Output: (void)
  return
}