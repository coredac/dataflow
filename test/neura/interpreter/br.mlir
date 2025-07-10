// RUN: neura-interpreter %s | FileCheck %s

func.func @test_br_with_args() -> i32 {
  %0 = "neura.constant"() {value = 42 : i32} : () -> i32
  // CHECK: [neura-interpreter] Executing neura.constant:
  "neura.br"(%0) [^bb1] {operandSegmentSizes = array<i32: 1>} : (i32) -> ()
  // CHECK-NEXT: [neura-interpreter] Executing neura.br:
  // CHECK-NEXT:   Target block: index 1
  // CHECK-NEXT:   Pass argument 0 to block parameter: value = 4.200000e+01
  // CHECK-NEXT:   Successfully jumped to block (index 1)
  // CHECK-NEXT:   Resetting operation index to start of block

  ^bb1(%a: i32):
  // CHECK-NEXT: [neura-interpreter] Output: 42.000000
  return %a : i32
}

func.func @test_br_with_multi_args() {
  %0 = "neura.constant"() {value = 42 : i32} : () -> i32
  // CHECK: [neura-interpreter] Executing neura.constant:
  %1 = "neura.constant"() {value = 1.0 : f32} : () -> f32
  // CHECK-NEXT: [neura-interpreter] Executing neura.constant:
  "neura.br"(%0, %1) [^bb1] {operandSegmentSizes = array<i32: 2>} : (i32, f32) -> ()
  // CHECK-NEXT: [neura-interpreter] Executing neura.br:
  // CHECK-NEXT:   Target block: index 1
  // CHECK-NEXT:   Pass argument 0 to block parameter: value = 4.200000e+01
  // CHECK-NEXT:   Pass argument 1 to block parameter: value = 1.000000e+00
  // CHECK-NEXT:   Successfully jumped to block (index 1)
  // CHECK-NEXT:   Resetting operation index to start of block

  ^bb1(%a: i32, %b: f32):
  "neura.add"(%a, %a) : (i32, i32) -> i32
  // CHECK-NEXT: [neura-interpreter] Executing neura.add:
  // CHECK-NEXT:   LHS: value = 4.200000e+01, predicate = 1
  // CHECK-NEXT:   RHS: value = 4.200000e+01, predicate = 1
  // CHECK-NEXT:   Result: value = 8.400000e+01, predicate = 1
  // CHECK-NEXT: [neura-interpreter] Output: (void)
  return
}