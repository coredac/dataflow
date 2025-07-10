// RUN: neura-interpreter %s | FileCheck %s

func.func @test_sel_with_comparison() -> f32 {
  %a = arith.constant 5.0 : f32
  %b = arith.constant 3.0 : f32
  %cond = "neura.fcmp"(%a, %b) {cmpType = "gt"} : (f32, f32) -> i1
  // CHECK: [neura-interpreter] Executing neura.fcmp:
  // CHECK-NEXT:   LHS: value = 5.000000e+00, predicate = 1
  // CHECK-NEXT:   RHS: value = 3.000000e+00, predicate = 1
  // CHECK-NEXT:   Comparison type: gt
  // CHECK-NEXT:   Comparison result: true
  // CHECK-NEXT:   Final result: value = 1.000000e+00, predicate = true

  %true_val = arith.constant 100.0 : f32
  %false_val = arith.constant 200.0 : f32
  %res = "neura.sel"(%true_val, %false_val, %cond) : (f32, f32, i1) -> f32
  // CHECK: [neura-interpreter] Executing neura.sel:
  // CHECK-NEXT:   Condition: value = 1.000000e+00, predicate = 1
  // CHECK-NEXT:   If true:    value = 1.000000e+02, predicate = 1
  // CHECK-NEXT:   If false:   value = 2.000000e+02, predicate = 1
  // CHECK-NEXT:   Evaluated condition: true
  // CHECK-NEXT:   Selecting 'ifTrue' branch
  // CHECK-NEXT:   Final result: value = 1.000000e+02, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 100.000000

  return %res : f32
}

func.func @test_sel_with_comparison_false() -> f32 {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 3.0 : f32
  %cond = "neura.fcmp"(%a, %b) {cmpType = "gt"} : (f32, f32) -> i1
  // CHECK: [neura-interpreter] Executing neura.fcmp:
  // CHECK-NEXT:   LHS: value = 2.000000e+00, predicate = 1
  // CHECK-NEXT:   RHS: value = 3.000000e+00, predicate = 1
  // CHECK-NEXT:   Comparison type: gt
  // CHECK-NEXT:   Comparison result: false
  // CHECK-NEXT:   Final result: value = 0.000000e+00, predicate = true

  %true_val = arith.constant 100.0 : f32
  %false_val = arith.constant 200.0 : f32
  %res = "neura.sel"(%true_val, %false_val, %cond) : (f32, f32, i1) -> f32
  // CHECK: [neura-interpreter] Executing neura.sel:
  // CHECK-NEXT:   Condition: value = 0.000000e+00, predicate = 1
  // CHECK-NEXT:   If true:    value = 1.000000e+02, predicate = 1
  // CHECK-NEXT:   If false:   value = 2.000000e+02, predicate = 1
  // CHECK-NEXT:   Evaluated condition: false
  // CHECK-NEXT:   Selecting 'ifFalse' branch
  // CHECK-NEXT:   Final result: value = 2.000000e+02, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 200.000000

  return %res : f32
}

func.func @test_sel_nested_with_comparison() -> f32 {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 3.0 : f32
  %cond1 = "neura.fcmp"(%a, %b) {cmpType = "gt"} : (f32, f32) -> i1
  // CHECK: [neura-interpreter] Executing neura.fcmp:
  // CHECK-NEXT:   LHS: value = 2.000000e+00, predicate = 1
  // CHECK-NEXT:   RHS: value = 3.000000e+00, predicate = 1
  // CHECK-NEXT:   Comparison type: gt
  // CHECK-NEXT:   Comparison result: false
  // CHECK-NEXT:   Final result: value = 0.000000e+00, predicate = true

  %true_val1 = arith.constant 100.0 : f32
  %false_val1 = arith.constant 200.0 : f32
  %sel1 = "neura.sel"(%true_val1, %false_val1, %cond1) : (f32, f32, i1) -> f32
  // CHECK: [neura-interpreter] Executing neura.sel:
  // CHECK-NEXT:   Condition: value = 0.000000e+00, predicate = 1
  // CHECK-NEXT:   If true:    value = 1.000000e+02, predicate = 1
  // CHECK-NEXT:   If false:   value = 2.000000e+02, predicate = 1
  // CHECK-NEXT:   Evaluated condition: false
  // CHECK-NEXT:   Selecting 'ifFalse' branch
  // CHECK-NEXT:   Final result: value = 2.000000e+02, predicate = true

  %c = arith.constant 5.0 : f32
  %d = arith.constant 1.0 : f32
  %cond2 = "neura.fcmp"(%c, %d) {cmpType = "gt"} : (f32, f32) -> i1
  // CHECK: [neura-interpreter] Executing neura.fcmp:
  // CHECK-NEXT:   LHS: value = 5.000000e+00, predicate = 1
  // CHECK-NEXT:   RHS: value = 1.000000e+00, predicate = 1
  // CHECK-NEXT:   Comparison type: gt
  // CHECK-NEXT:   Comparison result: true
  // CHECK-NEXT:   Final result: value = 1.000000e+00, predicate = true

  %true_val2 = arith.constant 300.0 : f32
  %res = "neura.sel"(%true_val2, %sel1, %cond2) : (f32, f32, i1) -> f32
  // CHECK: [neura-interpreter] Executing neura.sel:
  // CHECK-NEXT:   Condition: value = 1.000000e+00, predicate = 1
  // CHECK-NEXT:   If true:    value = 3.000000e+02, predicate = 1
  // CHECK-NEXT:   If false:   value = 2.000000e+02, predicate = 1
  // CHECK-NEXT:   Evaluated condition: true
  // CHECK-NEXT:   Selecting 'ifTrue' branch
  // CHECK-NEXT:   Final result: value = 3.000000e+02, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 300.000000

  return %res : f32
}