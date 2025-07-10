// RUN: neura-interpreter %s | FileCheck %s

func.func @test_ctrl_mov_basic() {
  %a = "neura.reserve"() : () -> (i32)
  %const = arith.constant 42 : i32
  
  "neura.ctrl_mov"(%const, %a) : (i32, i32) -> ()
  
  // CHECK: [neura-interpreter] Executing neura.ctrl_mov:
  // CHECK-NEXT:   Source: %c42_i32 = arith.constant 42 : i32
  // CHECK-NEXT:     Value: 4.200000e+01
  // CHECK-NEXT:     Predicate: true
  // CHECK-NEXT:   Target: %0 = neura.reserve : i32
  // CHECK-NEXT:     Old value: 0.000000e+00
  // CHECK-NEXT:     Old predicate: false
  // CHECK-NEXT:   Updated target placeholder:
  // CHECK-NEXT:     New value: 4.200000e+01
  // CHECK-NEXT:     New predicate: true
  // CHECK-NEXT: [neura-interpreter] Output: (void)
  
  return
}

func.func @test_ctrl_mov_chained() {
  %a = "neura.reserve"() : () -> (i32)
  %b = "neura.reserve"() : () -> (i32)
  %const = arith.constant 10 : i32

  "neura.ctrl_mov"(%const, %a) : (i32, i32) -> ()
  // CHECK: [neura-interpreter] Executing neura.ctrl_mov:
  // CHECK-NEXT:   Source: %c10_i32 = arith.constant 10 : i32
  // CHECK-NEXT:     Value: 1.000000e+01
  // CHECK-NEXT:     Predicate: true
  // CHECK-NEXT:   Target: %0 = neura.reserve : i32
  // CHECK-NEXT:     Old value: 0.000000e+00
  // CHECK-NEXT:     Old predicate: false
  // CHECK-NEXT:   Updated target placeholder:
  // CHECK-NEXT:     New value: 1.000000e+01
  // CHECK-NEXT:     New predicate: true

  "neura.ctrl_mov"(%a, %b) : (i32, i32) -> ()
  // CHECK: [neura-interpreter] Executing neura.ctrl_mov:
  // CHECK-NEXT:   Source: %0 = neura.reserve : i32
  // CHECK-NEXT:     Value: 1.000000e+01
  // CHECK-NEXT:     Predicate: true
  // CHECK-NEXT:   Target: %1 = neura.reserve : i32
  // CHECK-NEXT:     Old value: 0.000000e+00
  // CHECK-NEXT:     Old predicate: false
  // CHECK-NEXT:   Updated target placeholder:
  // CHECK-NEXT:     New value: 1.000000e+01
  // CHECK-NEXT:     New predicate: true
  // CHECK-NEXT: [neura-interpreter] Output: (void)

  return
}

func.func @test_ctrl_mov_vector() {
  %vec_reserve = "neura.reserve"() : () -> (vector<4xf32>)
  %vec_const = "neura.constant"() {value = dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf32>} : () -> vector<4xf32>

  "neura.ctrl_mov"(%vec_const, %vec_reserve) : (vector<4xf32>, vector<4xf32>) -> ()
  // CHECK: [neura-interpreter] Executing neura.ctrl_mov:
  // CHECK-NEXT:   Source: %1 = "neura.constant"() <{value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf32>}> : () -> vector<4xf32>
  // CHECK-NEXT:     Value: {{.*}}  
  // CHECK-NEXT:     Predicate: true
  // CHECK-NEXT:   Target: %0 = neura.reserve : vector<4xf32>
  // CHECK-NEXT:     Old value: 0.000000e+00
  // CHECK-NEXT:     Old predicate: false
  // CHECK-NEXT:   Updated target placeholder:
  // CHECK-NEXT:     New value: {{.*}}  
  // CHECK-NEXT:     New predicate: true
  // CHECK-NEXT: [neura-interpreter] Output: (void)

  return
}