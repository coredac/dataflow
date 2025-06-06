// RUN: neura-interpreter %s | FileCheck %s

module {
  func.func @test() -> !neura.data<f32, i1> {
    %arg0 = "neura.constant"() <{value = 9.0 : f32, predicate = true}> : () -> !neura.data<f32, i1>
    %cst = "neura.constant"() <{value = 2.0 : f32, predicate = false}> : () -> !neura.data<f32, i1>
    %res = "neura.fadd"(%arg0, %cst) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
    return %res : !neura.data<f32, i1>
    // CHECK: Output: 11.000000 (predicate=false)
  }
}
