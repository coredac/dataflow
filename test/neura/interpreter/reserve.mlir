// RUN: neura-interpreter %s | FileCheck %s

func.func @test_reserve_basic() {
  %a = "neura.reserve"() : () -> (i32)
  // CHECK: [neura-interpreter] Executing neura.reserve:
  // CHECK-NEXT:   Created placeholder: %0 = neura.reserve : i32
  // CHECK-NEXT:     Initial value: 0.0f
  // CHECK-NEXT:     Initial predicate: false
  // CHECK-NEXT:     Type: i32
  // CHECK-NEXT: [neura-interpreter] Output: (void)
  return
}

func.func @test_reserve_multiple_types() {
  %a = "neura.reserve"() : () -> (i32)
  // CHECK: [neura-interpreter] Executing neura.reserve:
  // CHECK-NEXT:   Created placeholder: %0 = neura.reserve : i32
  // CHECK-NEXT:     Initial value: 0.0f
  // CHECK-NEXT:     Initial predicate: false
  // CHECK-NEXT:     Type: i32

  %b = "neura.reserve"() : () -> (f32)
  // CHECK: [neura-interpreter] Executing neura.reserve:
  // CHECK-NEXT:   Created placeholder: %1 = neura.reserve : f32
  // CHECK-NEXT:     Initial value: 0.0f
  // CHECK-NEXT:     Initial predicate: false
  // CHECK-NEXT:     Type: f32

  %c = "neura.reserve"() : () -> (tensor<4xf32>)
  // CHECK: [neura-interpreter] Executing neura.reserve:
  // CHECK-NEXT:   Created placeholder: %2 = neura.reserve : tensor<4xf32>
  // CHECK-NEXT:     Initial value: 0.0f
  // CHECK-NEXT:     Initial predicate: false
  // CHECK-NEXT:     Type: tensor<4xf32>
  // CHECK-NEXT: [neura-interpreter] Output: (void)
  return
}