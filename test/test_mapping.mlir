module {
  func.func @loop_test() -> f32 {
    %0 = llvm.mlir.constant(10 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.mlir.constant(3.000000e+00 : f32) : f32
    %4 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    llvm.br ^bb1(%1, %4 : i64, f32)
  ^bb1(%5: i64, %6: f32):  // 2 preds: ^bb0, ^bb1
    %7 = llvm.fadd %6, %3 : f32
    %8 = llvm.add %5, %2 : i64
    %9 = llvm.icmp "slt" %8, %0 : i64
    llvm.cond_br %9, ^bb1(%8, %7 : i64, f32), ^bb2(%7 : f32)
  ^bb2(%10: f32):  // pred: ^bb1
    return %10 : f32
  }
}

