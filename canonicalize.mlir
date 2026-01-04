module {
  func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
    %0 = "neura.constant"() <{value = 0 : i64}> : () -> i64
    %1 = "neura.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    neura.br %0, %1 : i64, f32 to ^bb1
  ^bb1(%2: i64, %3: f32):  // 2 preds: ^bb0, ^bb1
    %4 = "neura.fadd"(%3) {rhs_value = 3.000000e+00 : f32} : (f32) -> f32
    %5 = "neura.add"(%2) {rhs_value = 1 : i64} : (i64) -> i64
    %6 = "neura.icmp"(%5) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (i64) -> i1
    neura.cond_br %6 : i1 then %5, %4 : i64, f32 to ^bb1 else %4 : f32 to ^bb2
  ^bb2(%7: f32):  // pred: ^bb1
    "neura.return"(%7) : (f32) -> ()
  }
}

