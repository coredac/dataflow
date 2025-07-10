// 修正工具链为 neura-interpreter，匹配日志来源
// RUN: neura-interpreter %s | FileCheck %s

// ====== Equal comparison (eq) ======
func.func @test_fcmp_eq_true() -> i1 {
  %a = arith.constant 3.14 : f32
  %eq = "neura.fcmp"(%a, %a) {cmpType = "eq"} : (f32, f32) -> i1
  // 严格匹配日志缩进（2个空格）和前缀
  // CHECK: [neura-interpreter] Executing neura.fcmp:
  // CHECK-NEXT:  LHS: value = 3.140000e+00, predicate = 1
  // CHECK-NEXT:  RHS: value = 3.140000e+00, predicate = 1
  // CHECK-NEXT:  Comparison type: eq
  // CHECK-NEXT:  Comparison result: true
  // CHECK-NEXT:  Final result: value = 1.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 1.000000
  return %eq : i1  
}

func.func @test_fcmp_eq_false() -> i1 {
  %a = arith.constant 3.14 : f32
  %b = arith.constant 2.71 : f32
  %eq = "neura.fcmp"(%a, %b) {cmpType = "eq"} : (f32, f32) -> i1
  // 用 CHECK-NEXT 确保顺序严格一致
  // CHECK: [neura-interpreter] Executing neura.fcmp:
  // CHECK-NEXT:  LHS: value = 3.140000e+00, predicate = 1
  // CHECK-NEXT:  RHS: value = 2.710000e+00, predicate = 1
  // CHECK-NEXT:  Comparison type: eq
  // CHECK-NEXT:  Comparison result: false
  // CHECK-NEXT:  Final result: value = 0.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 0.000000
  return %eq : i1  
}

func.func @test_fcmp_ne_true() -> i1 {
  %a = arith.constant 3.14 : f32
  %b = arith.constant 2.71 : f32
  %ne = "neura.fcmp"(%a, %b) {cmpType = "ne"} : (f32, f32) -> i1
  // CHECK: [neura-interpreter] Executing neura.fcmp:
  // CHECK-NEXT:  LHS: value = 3.140000e+00, predicate = 1
  // CHECK-NEXT:  RHS: value = 2.710000e+00, predicate = 1
  // CHECK-NEXT:  Comparison type: ne
  // CHECK-NEXT:  Comparison result: true
  // CHECK-NEXT:  Final result: value = 1.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 1.000000
  return %ne : i1  
}

func.func @test_fcmp_ne_false() -> i1 {
  %a = arith.constant 3.14 : f32
  %ne = "neura.fcmp"(%a, %a) {cmpType = "ne"} : (f32, f32) -> i1
  // CHECK: [neura-interpreter] Executing neura.fcmp:
  // CHECK-NEXT:  LHS: value = 3.140000e+00, predicate = 1
  // CHECK-NEXT:  RHS: value = 3.140000e+00, predicate = 1
  // CHECK-NEXT:  Comparison type: ne
  // CHECK-NEXT:  Comparison result: false
  // CHECK-NEXT:  Final result: value = 0.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 0.000000
  return %ne : i1  
}

func.func @test_fcmp_lt_true() -> i1 {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 3.0 : f32
  %lt = "neura.fcmp"(%a, %b) {cmpType = "lt"} : (f32, f32) -> i1
  // CHECK: [neura-interpreter] Executing neura.fcmp:
  // CHECK-NEXT:  LHS: value = 2.000000e+00, predicate = 1
  // CHECK-NEXT:  RHS: value = 3.000000e+00, predicate = 1
  // CHECK-NEXT:  Comparison type: lt
  // CHECK-NEXT:  Comparison result: true
  // CHECK-NEXT:  Final result: value = 1.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 1.000000
  return %lt : i1  
}

func.func @test_fcmp_lt_false() -> i1 {
  %a = arith.constant 3.0 : f32
  %b = arith.constant 2.0 : f32
  %lt = "neura.fcmp"(%a, %b) {cmpType = "lt"} : (f32, f32) -> i1
  // CHECK: [neura-interpreter] Executing neura.fcmp:
  // CHECK-NEXT:  LHS: value = 3.000000e+00, predicate = 1
  // CHECK-NEXT:  RHS: value = 2.000000e+00, predicate = 1
  // CHECK-NEXT:  Comparison type: lt
  // CHECK-NEXT:  Comparison result: false
  // CHECK-NEXT:  Final result: value = 0.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 0.000000
  return %lt : i1  
}

func.func @test_fcmp_le_true() -> i1 {
  %a = arith.constant 2.5 : f32
  %b = arith.constant 2.5 : f32
  %le = "neura.fcmp"(%a, %b) {cmpType = "le"} : (f32, f32) -> i1
  // CHECK: [neura-interpreter] Executing neura.fcmp:
  // CHECK-NEXT:  LHS: value = 2.500000e+00, predicate = 1
  // CHECK-NEXT:  RHS: value = 2.500000e+00, predicate = 1
  // CHECK-NEXT:  Comparison type: le
  // CHECK-NEXT:  Comparison result: true
  // CHECK-NEXT:  Final result: value = 1.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 1.000000
  return %le : i1  
}

func.func @test_fcmp_le_false() -> i1 {
  %a = arith.constant 3.5 : f32
  %b = arith.constant 2.5 : f32
  %le = "neura.fcmp"(%a, %b) {cmpType = "le"} : (f32, f32) -> i1
  // CHECK: [neura-interpreter] Executing neura.fcmp:
  // CHECK-NEXT:  LHS: value = 3.500000e+00, predicate = 1
  // CHECK-NEXT:  RHS: value = 2.500000e+00, predicate = 1
  // CHECK-NEXT:  Comparison type: le
  // CHECK-NEXT:  Comparison result: false
  // CHECK-NEXT:  Final result: value = 0.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 0.000000
  return %le : i1  
}

func.func @test_fcmp_gt_true() -> i1 {
  %a = arith.constant 5.0 : f32
  %b = arith.constant 3.0 : f32
  %gt = "neura.fcmp"(%a, %b) {cmpType = "gt"} : (f32, f32) -> i1
  // CHECK: [neura-interpreter] Executing neura.fcmp:
  // CHECK-NEXT:  LHS: value = 5.000000e+00, predicate = 1
  // CHECK-NEXT:  RHS: value = 3.000000e+00, predicate = 1
  // CHECK-NEXT:  Comparison type: gt
  // CHECK-NEXT:  Comparison result: true
  // CHECK-NEXT:  Final result: value = 1.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 1.000000
  return %gt : i1  
}

func.func @test_fcmp_gt_false() -> i1 {
  %a = arith.constant 3.0 : f32
  %b = arith.constant 5.0 : f32
  %gt = "neura.fcmp"(%a, %b) {cmpType = "gt"} : (f32, f32) -> i1
  // CHECK: [neura-interpreter] Executing neura.fcmp:
  // CHECK-NEXT:  LHS: value = 3.000000e+00, predicate = 1
  // CHECK-NEXT:  RHS: value = 5.000000e+00, predicate = 1
  // CHECK-NEXT:  Comparison type: gt
  // CHECK-NEXT:  Comparison result: false
  // CHECK-NEXT:  Final result: value = 0.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 0.000000
  return %gt : i1  
}

func.func @test_fcmp_ge_true() -> i1 {
  %a = arith.constant 5.0 : f32
  %b = arith.constant 5.0 : f32
  %ge = "neura.fcmp"(%a, %b) {cmpType = "ge"} : (f32, f32) -> i1
  // CHECK: [neura-interpreter] Executing neura.fcmp:
  // CHECK-NEXT:  LHS: value = 5.000000e+00, predicate = 1
  // CHECK-NEXT:  RHS: value = 5.000000e+00, predicate = 1
  // CHECK-NEXT:  Comparison type: ge
  // CHECK-NEXT:  Comparison result: true
  // CHECK-NEXT:  Final result: value = 1.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 1.000000
  return %ge : i1  
}

func.func @test_fcmp_ge_false() -> i1 {
  %a = arith.constant 3.0 : f32
  %b = arith.constant 5.0 : f32
  %ge = "neura.fcmp"(%a, %b) {cmpType = "ge"} : (f32, f32) -> i1
  // CHECK: [neura-interpreter] Executing neura.fcmp:
  // CHECK-NEXT:  LHS: value = 3.000000e+00, predicate = 1
  // CHECK-NEXT:  RHS: value = 5.000000e+00, predicate = 1
  // CHECK-NEXT:  Comparison type: ge
  // CHECK-NEXT:  Comparison result: false
  // CHECK-NEXT:  Final result: value = 0.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 0.000000
  return %ge : i1  
}