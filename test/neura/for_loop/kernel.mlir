#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module attributes {dlti.dl_spec = #dlti.dl_spec<i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.mlir.global external @input(dense<1.000000e+00> : tensor<32xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x f32>
  llvm.mlir.global external @output(dense<0.000000e+00> : tensor<32xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x f32>
  llvm.mlir.global external @coefficients(dense<[2.500000e-01, 1.500000e+00, 3.750000e+00, -2.250000e+00, 5.000000e-01, 7.500000e-01, -3.000000e+00, 1.250000e+00, 2.500000e-01, 1.500000e+00, 3.750000e+00, -2.250000e+00, 5.000000e-01, 7.500000e-01, -3.000000e+00, 1.250000e+00, 2.500000e-01, 1.500000e+00, 3.750000e+00, -2.250000e+00, 5.000000e-01, 7.500000e-01, -3.000000e+00, 1.250000e+00, 2.500000e-01, 1.500000e+00, 3.750000e+00, -2.250000e+00, 5.000000e-01, 7.500000e-01, -3.000000e+00, 1.250000e+00]> : tensor<32xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x f32>
  llvm.mlir.global private unnamed_addr constant @".str"("output: %f\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @main() -> (i32 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<all>, no_inline, optimize_none, passthrough = ["mustprogress", "norecurse", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.addressof @input : !llvm.ptr
    %3 = llvm.mlir.addressof @output : !llvm.ptr
    %4 = llvm.mlir.addressof @coefficients : !llvm.ptr
    %5 = llvm.mlir.addressof @".str" : !llvm.ptr
    %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %6 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.call @_Z6kernelPfS_S_(%2, %3, %4) {no_unwind} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %7 = llvm.load %3 {alignment = 16 : i64} : !llvm.ptr -> f32
    %8 = llvm.fpext %7 : f32 to f64
    %9 = llvm.call @printf(%5, %8) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    llvm.return %1 : i32
  }
  llvm.func @_Z6kernelPfS_S_(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(32 : i32) : i32
    %3 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %8 = llvm.alloca %0 x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %3 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg1, %4 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg2, %5 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %1, %7 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %1, %6 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb3
    %9 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    %10 = llvm.icmp "slt" %9, %2 : i32
    llvm.cond_br %10, ^bb2, ^bb4
  ^bb2:  // pred: ^bb1
    %11 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %12 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    %13 = llvm.sext %12 : i32 to i64
    %14 = llvm.getelementptr inbounds %11[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %15 = llvm.load %14 {alignment = 4 : i64} : !llvm.ptr -> f32
    %16 = llvm.load %5 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %17 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    %18 = llvm.sext %17 : i32 to i64
    %19 = llvm.getelementptr inbounds %16[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %20 = llvm.load %19 {alignment = 4 : i64} : !llvm.ptr -> f32
    %21 = llvm.fmul %15, %20 : f32
    llvm.store %21, %8 {alignment = 4 : i64} : f32, !llvm.ptr
    %22 = llvm.load %8 {alignment = 4 : i64} : !llvm.ptr -> f32
    %23 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %24 = llvm.load %7 {alignment = 4 : i64} : !llvm.ptr -> i32
    %25 = llvm.sext %24 : i32 to i64
    %26 = llvm.getelementptr inbounds %23[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %27 = llvm.load %26 {alignment = 4 : i64} : !llvm.ptr -> f32
    %28 = llvm.fadd %27, %22 : f32
    llvm.store %28, %26 {alignment = 4 : i64} : f32, !llvm.ptr
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    %29 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    %30 = llvm.add %29, %0 overflow<nsw> : i32
    llvm.store %30, %6 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1 {loop_annotation = #loop_annotation}
  ^bb4:  // pred: ^bb1
    llvm.return
  }
  llvm.func @printf(!llvm.ptr {llvm.noundef}, ...) -> i32 attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
}
