// RUN: neura-interpreter %s --verbose | FileCheck %s

// Test basic subtraction with positive result
func.func @test_sub_positive() -> i32 {
  %a = arith.constant 200 : i32
  %b = arith.constant 50 : i32
  %res = "neura.sub"(%a, %b) : (i32, i32) -> i32

  // CHECK: [neura-interpreter]  → Output: 150.000000
  return %res : i32
}

// Test subtraction with negative result
func.func @test_sub_negative() -> i32 {
  %a = arith.constant 50 : i32
  %b = arith.constant 200 : i32
  %res = "neura.sub"(%a, %b) : (i32, i32) -> i32

  // CHECK: [neura-interpreter]  → Output: -150.000000
  return %res : i32
}

// TODO: Remove tests with predicate input because we plan to remove the predicate attribute in
// https://github.com/coredac/dataflow/issues/116

// Test subtraction with predicate=true
// func.func @test_sub_with_embed_predicate_true() -> i32 {
//   %a = "neura.constant"() {value = 300 : i32, predicate = true} : () -> i32
//   %b = "neura.constant"() {value = 100 : i32, predicate = true} : () -> i32
//   %res = "neura.sub"(%a, %b) : (i32, i32) -> i32
//   // [neura-interpreter]  → Output: 200.000000
//   return %res : i32
// }


// Test subtraction with predicate=false
// func.func @test_sub_with_embed_predicate_false() -> i32 {
//   %a = "neura.constant"() {value = 500 : i32, predicate = false} : () -> i32
//   %b = "neura.constant"() {value = 200 : i32, predicate = false} : () -> i32
//   %res = "neura.sub"(%a, %b) : (i32, i32) -> i32
//   // [neura-interpreter]  → Output: 0.000000
//   return %res : i32
// }