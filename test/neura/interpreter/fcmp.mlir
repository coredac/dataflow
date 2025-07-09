// RUN: mlir-neura-opt %s | FileCheck %s

// ====== Equal comparison (eq) ======
// Positive case: Equal floats
func.func @test_fcmp_eq_true() -> i1 {
  %a = arith.constant 3.14 : f32
  %eq = "neura.fcmp"(%a, %a) {cmpType = "eq"} : (f32, f32) -> i1
  // CHECK: neura.fcmp {{.*}} "eq"
  return %eq : i1  // Expected: 1 (true)
}

// Negative case: Unequal floats
func.func @test_fcmp_eq_false() -> i1 {
  %a = arith.constant 3.14 : f32
  %b = arith.constant 2.71 : f32
  %eq = "neura.fcmp"(%a, %b) {cmpType = "eq"} : (f32, f32) -> i1
  // CHECK: neura.fcmp {{.*}} "eq"
  return %eq : i1  // Expected: 0 (false)
}

// ====== Not equal comparison (ne) ======
// Positive case: Unequal floats
func.func @test_fcmp_ne_true() -> i1 {
  %a = arith.constant 3.14 : f32
  %b = arith.constant 2.71 : f32
  %ne = "neura.fcmp"(%a, %b) {cmpType = "ne"} : (f32, f32) -> i1
  // CHECK: neura.fcmp {{.*}} "ne"
  return %ne : i1  // Expected: 1 (true)
}

// Negative case: Equal floats
func.func @test_fcmp_ne_false() -> i1 {
  %a = arith.constant 3.14 : f32
  %ne = "neura.fcmp"(%a, %a) {cmpType = "ne"} : (f32, f32) -> i1
  // CHECK: neura.fcmp {{.*}} "ne"
  return %ne : i1  // Expected: 0 (false)
}

// ====== Less than (lt) ======
// Positive case: lhs < rhs
func.func @test_fcmp_lt_true() -> i1 {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 3.0 : f32
  %lt = "neura.fcmp"(%a, %b) {cmpType = "lt"} : (f32, f32) -> i1
  // CHECK: neura.fcmp {{.*}} "lt"
  return %lt : i1  // Expected: 1 (true)
}

// Negative case: lhs >= rhs
func.func @test_fcmp_lt_false() -> i1 {
  %a = arith.constant 3.0 : f32
  %b = arith.constant 2.0 : f32
  %lt = "neura.fcmp"(%a, %b) {cmpType = "lt"} : (f32, f32) -> i1
  // CHECK: neura.fcmp {{.*}} "lt"
  return %lt : i1  // Expected: 0 (false)
}

// ====== Less than or equal (le) ======
// Positive case: lhs <= rhs
func.func @test_fcmp_le_true() -> i1 {
  %a = arith.constant 2.5 : f32
  %b = arith.constant 2.5 : f32
  %le = "neura.fcmp"(%a, %b) {cmpType = "le"} : (f32, f32) -> i1
  // CHECK: neura.fcmp {{.*}} "le"
  return %le : i1  // Expected: 1 (true)
}

// Negative case: lhs > rhs
func.func @test_fcmp_le_false() -> i1 {
  %a = arith.constant 3.5 : f32
  %b = arith.constant 2.5 : f32
  %le = "neura.fcmp"(%a, %b) {cmpType = "le"} : (f32, f32) -> i1
  // CHECK: neura.fcmp {{.*}} "le"
  return %le : i1  // Expected: 0 (false)
}

// ====== Greater than (gt) ======
// Positive case: lhs > rhs
func.func @test_fcmp_gt_true() -> i1 {
  %a = arith.constant 5.0 : f32
  %b = arith.constant 3.0 : f32
  %gt = "neura.fcmp"(%a, %b) {cmpType = "gt"} : (f32, f32) -> i1
  // CHECK: neura.fcmp {{.*}} "gt"
  return %gt : i1  // Expected: 1 (true)
}

// Negative case: lhs <= rhs
func.func @test_fcmp_gt_false() -> i1 {
  %a = arith.constant 3.0 : f32
  %b = arith.constant 5.0 : f32
  %gt = "neura.fcmp"(%a, %b) {cmpType = "gt"} : (f32, f32) -> i1
  // CHECK: neura.fcmp {{.*}} "gt"
  return %gt : i1  // Expected: 0 (false)
}

// ====== Greater than or equal (ge) ======
// Positive case: lhs >= rhs
func.func @test_fcmp_ge_true() -> i1 {
  %a = arith.constant 5.0 : f32
  %b = arith.constant 5.0 : f32
  %ge = "neura.fcmp"(%a, %b) {cmpType = "ge"} : (f32, f32) -> i1
  // CHECK: neura.fcmp {{.*}} "ge"
  return %ge : i1  // Expected: 1 (true)
}

// Negative case: lhs < rhs
func.func @test_fcmp_ge_false() -> i1 {
  %a = arith.constant 3.0 : f32
  %b = arith.constant 5.0 : f32
  %ge = "neura.fcmp"(%a, %b) {cmpType = "ge"} : (f32, f32) -> i1
  // CHECK: neura.fcmp {{.*}} "ge"
  return %ge : i1  // Expected: 0 (false)
}
