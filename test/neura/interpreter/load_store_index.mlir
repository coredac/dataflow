// RUN: neura-interpreter %s | FileCheck %s

module {
  // Test case: float (f32) load/store with single index
  func.func @test_load_store_indexed() -> f32 {
    %val = "neura.constant"() { value = 42.0 } : () -> f32
    // CHECK: [neura-interpreter] Executing neura.constant:
    %base = "neura.constant"() { value = 100 } : () -> i32
    // CHECK: [neura-interpreter] Executing neura.constant:
    %offset = arith.constant 4 : i32

    "neura.store_indexed"(%val, %base, %offset) {
      operandSegmentSizes = array<i32: 1, 1, 1, 0>
    } : (f32, i32, i32) -> ()
    // CHECK: [neura-interpreter] Executing neura.store_indexed:
    // CHECK-NEXT:   StoreIndexed [addr = 104] <= val = 4.200000e+01 (predicate=true)

    %load = "neura.load_indexed"(%base, %offset) {
      operandSegmentSizes = array<i32: 1, 1, 0>
    } : (i32, i32) -> f32
    // CHECK: [neura-interpreter] Executing neura.load_indexed:
    // CHECK-NEXT:   LoadIndexed [addr = 104] => val = 4.200000e+01 (predicate=true)
    // CHECK-NEXT: [neura-interpreter] Output: 42.000000

    return %load : f32
  }

  // Test case: 32-bit integer (i32) load/store with single index
  func.func @test_i32() -> i32 {
    %val = "neura.constant"() { value = 66 } : () -> i32
    // CHECK: [neura-interpreter] Executing neura.constant:
    %base = "neura.constant"() { value = 200 } : () -> i32
    // CHECK: [neura-interpreter] Executing neura.constant:
    %offset = arith.constant 4 : i32

    "neura.store_indexed"(%val, %base, %offset) {
      operandSegmentSizes = array<i32: 1, 1, 1, 0>
    } : (i32, i32, i32) -> ()
    // CHECK: [neura-interpreter] Executing neura.store_indexed:
    // CHECK-NEXT:   StoreIndexed [addr = 204] <= val = 6.600000e+01 (predicate=true)

    %load = "neura.load_indexed"(%base, %offset) {
      operandSegmentSizes = array<i32: 1, 1, 0>
    } : (i32, i32) -> i32
    // CHECK: [neura-interpreter] Executing neura.load_indexed:
    // CHECK-NEXT:   LoadIndexed [addr = 204] => val = 6.600000e+01 (predicate=true)
    // CHECK-NEXT: [neura-interpreter] Output: 66.000000
  
    return %load : i32
  }

  // Test case: float (f32) load/store with multi-dimensional indexing (2 indices)
  func.func @test_multi_index() -> f32 {
    %base = "neura.constant"() { value = 500 } : () -> i32
    // CHECK: [neura-interpreter] Executing neura.constant:
    %i = arith.constant 2 : i32
    %j = arith.constant 3 : i32
    %stride = arith.constant 10 : i32

    %offset_i = "neura.fmul"(%i, %stride) : (i32, i32) -> i32
    // CHECK: [neura-interpreter] Executing neura.fmul:
    // CHECK-NEXT:   LHS: value = 2.000000e+00, predicate = 1
    // CHECK-NEXT:   RHS: value = 1.000000e+01, predicate = 1
    // CHECK-NEXT:   Result: value = 2.000000e+01, predicate = 1
    %offset = "neura.add"(%offset_i, %j) : (i32, i32) -> i32
    // CHECK: [neura-interpreter] Executing neura.add:
    // CHECK-NEXT:   LHS: value = 2.000000e+01, predicate = 1
    // CHECK-NEXT:   RHS: value = 3.000000e+00, predicate = 1
    // CHECK-NEXT:   Result: value = 2.300000e+01, predicate = 1

    %val = "neura.constant"() { value = 777.0 } : () -> f32
    // CHECK: [neura-interpreter] Executing neura.constant:

    "neura.store_indexed"(%val, %base, %i, %j) {
      operandSegmentSizes = array<i32: 1, 1, 2, 0>
    } : (f32, i32, i32, i32) -> ()
    // CHECK: [neura-interpreter] Executing neura.store_indexed:
    // CHECK-NEXT:   StoreIndexed [addr = 505] <= val = 7.770000e+02 (predicate=true)

    %load = "neura.load_indexed"(%base, %i, %j) {
      operandSegmentSizes = array<i32: 1, 2, 0>
    } : (i32, i32, i32) -> f32
    // CHECK: [neura-interpreter] Executing neura.load_indexed:
    // CHECK-NEXT:   LoadIndexed [addr = 505] => val = 7.770000e+02 (predicate=true)
    // CHECK-NEXT: [neura-interpreter] Output: 777.000000

    return %load : f32
  }
}