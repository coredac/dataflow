module {
  func.func @test(%arg0: i64) -> f32 attributes {accelerator = "neura"} {
    %0 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
    %1 = "neura.constant"() <{value = 1.000000e+00 : f32}> : () -> !neura.data<f32, i1>
    %2 = "neura.constant"() <{value = 2.000000e+00 : f32}> : () -> !neura.data<f32, i1>
    %3 = "neura.constant"() <{value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
    %4 = "neura.constant"() <{value = 4.000000e+00 : f32}> : () -> !neura.data<f32, i1>
    %5 = "neura.icmp"(%arg0, %0) <{cmpType = "eq"}> : (i64, !neura.data<i64, i1>) -> !neura.data<i1, i1>
    neura.cond_br %5 : !neura.data<i1, i1> then %3, %4 : !neura.data<f32, i1>, !neura.data<f32, i1> to ^bb2 else to ^bb1
  ^bb1:  // pred: ^bb0
    %6 = "neura.fadd"(%1, %2) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
    neura.br %6 : !neura.data<f32, i1> to ^bb3
  ^bb2(%7: !neura.data<f32, i1>, %8: !neura.data<f32, i1>):  // pred: ^bb0
    %9 = "neura.fmul"(%7, %8) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
    neura.br %9 : !neura.data<f32, i1> to ^bb3
  ^bb3(%10: !neura.data<f32, i1>):  // 2 preds: ^bb1, ^bb2
    "neura.return"(%10) : (!neura.data<f32, i1>) -> ()
  }
}

