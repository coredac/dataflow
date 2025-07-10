// RUN: neura-interpreter %s | FileCheck %s

func.func @test_cond_br_true() {
  %cond = arith.constant 1 : i1 
  "neura.cond_br"(%cond) [^bb1, ^bb2] {operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i1) -> ()
  // CHECK: [neura-interpreter] Executing neura.cond_br:
  // CHECK-NEXT:   Condition value: 1.000000e+00, condition validity: 1
  // CHECK-NEXT:   Condition result: true -> targeting block (index 1)
  // CHECK-NEXT:   Successfully jumped to true block (index 1)
  // CHECK-NEXT: [neura-interpreter] Output: (void)
^bb1:
  return
^bb2:
  return
}

func.func @test_cond_br_false() {
  %cond = arith.constant 0 : i1  
  "neura.cond_br"(%cond) [^bb1, ^bb2] {operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i1) -> ()
  // CHECK: [neura-interpreter] Executing neura.cond_br:
  // CHECK-NEXT:   Condition value: 0.000000e+00, condition validity: 1
  // CHECK-NEXT:   Condition result: false -> targeting block (index 2)
  // CHECK-NEXT:   Successfully jumped to false block (index 2)
  // CHECK-NEXT: [neura-interpreter] Output: (void)
^bb1:
  return
^bb2:
  return
}

func.func @test_cond_br_with_valid_predicate() {
  %cond = arith.constant 1 : i1
  %pred = arith.constant 1 : i32  
  "neura.cond_br"(%cond, %pred) [^bb1, ^bb2] {operandSegmentSizes = array<i32: 1, 1, 0, 0>} : (i1, i32) -> ()
  // CHECK: [neura-interpreter] Executing neura.cond_br:
  // CHECK-NEXT:   Condition value: 1.000000e+00, condition validity: 1
  // CHECK-NEXT:   Predicate operand: value = 1.000000e+00, validity = 1
  // CHECK-NEXT:   Condition result: true -> targeting block (index 1)
  // CHECK-NEXT:   Successfully jumped to true block (index 1)
  // CHECK-NEXT: [neura-interpreter] Output: (void)
^bb1:
  return
^bb2:
  return
}

func.func @test_nested_cond_br() {
  %cond1 = arith.constant 1 : i1  
  "neura.cond_br"(%cond1) [^bb1, ^bb2] {operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i1) -> ()
  // CHECK: [neura-interpreter] Executing neura.cond_br:
  // CHECK-NEXT:   Condition value: 1.000000e+00, condition validity: 1
  // CHECK-NEXT:   Condition result: true -> targeting block (index 1)
  // CHECK-NEXT:   Successfully jumped to true block (index 1)

^bb1:
  %cond2 = arith.constant 0 : i1 
  "neura.cond_br"(%cond2) [^bb3, ^bb4] {operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i1) -> ()
  // CHECK: [neura-interpreter] Executing neura.cond_br:
  // CHECK-NEXT:   Condition value: 0.000000e+00, condition validity: 1
  // CHECK-NEXT:   Condition result: false -> targeting block (index 4)
  // CHECK-NEXT:   Successfully jumped to false block (index 4)
  // CHECK-NEXT: [neura-interpreter] Output: (void)

^bb2:
  return

^bb3:
  return

^bb4:
  return
}