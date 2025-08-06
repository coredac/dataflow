module {
  func.func @test() -> f32 {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %0 = "neura.fadd"(%cst, %cst_0) : (f32, f32) -> f32
    return %0 : f32
  }
}

