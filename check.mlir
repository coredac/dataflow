module {
  func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
    %0 = "neura.constant"() <{value = 10 : i64}> : () -> !neura.data<i64, i1>
    %1 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
    %2 = "neura.constant"() <{value = 1 : i64}> : () -> !neura.data<i64, i1>
    %3 = "neura.constant"() <{value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
    %4 = "neura.constant"() <{value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
    neura.br %1, %4 : !neura.data<i64, i1>, !neura.data<f32, i1> to ^bb1
  ^bb1(%5: !neura.data<i64, i1>, %6: !neura.data<f32, i1>):  // 2 preds: ^bb0, ^bb1
    %7 = "neura.fadd"(%6, %3) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
    %8 = "neura.add"(%5, %2) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %9 = "neura.icmp"(%8, %0) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
    neura.cond_br %9 : !neura.data<i1, i1> then %8, %7 : !neura.data<i64, i1>, !neura.data<f32, i1> to ^bb1 else %7 : !neura.data<f32, i1> to ^bb2
  ^bb2(%10: !neura.data<f32, i1>):  // pred: ^bb1
    "neura.return"(%10) : (!neura.data<f32, i1>) -> ()
  }
}

