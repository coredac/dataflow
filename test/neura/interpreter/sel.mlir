// RUN: neura-interpreter %s | FileCheck %s

func.func @test_sel_with_comparison() -> f32 {
  %a = arith.constant 5.0 : f32
  %b = arith.constant 3.0 : f32
  %cond = "neura.fcmp"(%a, %b) {cmpType = "gt"} : (f32, f32) -> i1
  %true_val = arith.constant 100.0 : f32
  %false_val = arith.constant 200.0 : f32
  %res = "neura.sel"(%true_val, %false_val, %cond) : (f32, f32, i1) -> f32

  // CHECK: neura.fcmp
  // CHECK-NEXT: neura.sel
  // CHECK-NEXT: [neura-interpreter] Output: 100.000000

  return %res : f32
}

func.func @test_sel_with_comparison_false() -> f32 {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 3.0 : f32
  %cond = "neura.fcmp"(%a, %b) {cmpType = "gt"} : (f32, f32) -> i1
  %true_val = arith.constant 100.0 : f32
  %false_val = arith.constant 200.0 : f32
  %res = "neura.sel"(%true_val, %false_val, %cond) : (f32, f32, i1) -> f32

  // CHECK: neura.fcmp
  // CHECK-NEXT: neura.sel
  // CHECK-NEXT: [neura-interpreter] Output: 200.000000

  return %res : f32
}

func.func @test_sel_nested_with_comparison() -> f32 {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 3.0 : f32
  %cond1 = "neura.fcmp"(%a, %b) {cmpType = "gt"} : (f32, f32) -> i1

  %true_val1 = arith.constant 100.0 : f32
  %false_val1 = arith.constant 200.0 : f32
  %sel1 = "neura.sel"(%true_val1, %false_val1, %cond1) : (f32, f32, i1) -> f32

  %c = arith.constant 5.0 : f32
  %d = arith.constant 1.0 : f32
  %cond2 = "neura.fcmp"(%c, %d) {cmpType = "gt"} : (f32, f32) -> i1

  %true_val2 = arith.constant 300.0 : f32
  // 第二个sel选true_val2或sel1结果
  %res = "neura.sel"(%true_val2, %sel1, %cond2) : (f32, f32, i1) -> f32

  // CHECK: neura.fcmp
  // CHECK-NEXT: neura.sel
  // CHECK-NEXT: neura.fcmp
  // CHECK-NEXT: neura.sel
  // CHECK-NEXT: [neura-interpreter] Output: 300.000000

  return %res : f32
}
