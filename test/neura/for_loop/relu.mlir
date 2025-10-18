#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module attributes {dlti.dl_spec = #dlti.dl_spec<f80 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.stack_alignment" = 128 : i64, "dlti.endianness" = "little">, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.mlir.global external @input(dense<[1, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20, -21, 22, -23, 24, -25, 26, -27, 28, -29, 30, -31]> : tensor<32xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x i32>
  llvm.mlir.global external @output(dense<0> : tensor<32xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x i32>
  llvm.mlir.global private unnamed_addr constant @".str"("output[%d] = %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @main() -> (i32 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<all>, no_inline, optimize_none, passthrough = ["mustprogress", "norecurse", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(32 : i32) : i32
    %3 = llvm.mlir.addressof @input : !llvm.ptr
    %4 = llvm.mlir.addressof @output : !llvm.ptr
    %5 = llvm.mlir.constant(0 : i64) : i64
    %6 = llvm.mlir.addressof @".str" : !llvm.ptr
    %7 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %8 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %9 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %7 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %1, %8 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb3
    %10 = llvm.load %8 {alignment = 4 : i64} : !llvm.ptr -> i32
    %11 = llvm.icmp "slt" %10, %2 : i32
    llvm.cond_br %11, ^bb2, ^bb4
  ^bb2:  // pred: ^bb1
    %12 = llvm.load %8 {alignment = 4 : i64} : !llvm.ptr -> i32
    %13 = llvm.sext %12 : i32 to i64
    %14 = llvm.getelementptr inbounds %4[%5, %13] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<32 x i32>
    llvm.store %1, %14 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    %15 = llvm.load %8 {alignment = 4 : i64} : !llvm.ptr -> i32
    %16 = llvm.add %15, %0 overflow<nsw> : i32
    llvm.store %16, %8 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1 {loop_annotation = #loop_annotation}
  ^bb4:  // pred: ^bb1
    llvm.call @_Z6kernelPiS_(%3, %4) {no_unwind} : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.store %1, %9 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb4, ^bb7
    %17 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> i32
    %18 = llvm.icmp "slt" %17, %2 : i32
    llvm.cond_br %18, ^bb6, ^bb8
  ^bb6:  // pred: ^bb5
    %19 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> i32
    %20 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> i32
    %21 = llvm.sext %20 : i32 to i64
    %22 = llvm.getelementptr inbounds %4[%5, %21] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<32 x i32>
    %23 = llvm.load %22 {alignment = 4 : i64} : !llvm.ptr -> i32
    %24 = llvm.call @printf(%6, %19, %23) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32
    llvm.br ^bb7
  ^bb7:  // pred: ^bb6
    %25 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> i32
    %26 = llvm.add %25, %0 overflow<nsw> : i32
    llvm.store %26, %9 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb5 {loop_annotation = #loop_annotation}
  ^bb8:  // pred: ^bb5
    llvm.return %1 : i32
  }
  llvm.func @_Z6kernelPiS_(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(32 : i32) : i32
    %3 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %3 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg1, %4 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %1, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb6
    %6 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %7 = llvm.icmp "slt" %6, %2 : i32
    llvm.cond_br %7, ^bb2, ^bb7
  ^bb2:  // pred: ^bb1
    %8 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %9 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %10 = llvm.sext %9 : i32 to i64
    %11 = llvm.getelementptr inbounds %8[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %12 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %13 = llvm.icmp "sgt" %12, %1 : i32
    llvm.cond_br %13, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %14 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %15 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %16 = llvm.sext %15 : i32 to i64
    %17 = llvm.getelementptr inbounds %14[%16] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %18 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %19 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %20 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %21 = llvm.sext %20 : i32 to i64
    %22 = llvm.getelementptr inbounds %19[%21] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %23 = llvm.load %22 {alignment = 4 : i64} : !llvm.ptr -> i32
    %24 = llvm.add %23, %18 overflow<nsw> : i32
    llvm.store %24, %22 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    %25 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %26 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %27 = llvm.sext %26 : i32 to i64
    %28 = llvm.getelementptr inbounds %25[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %29 = llvm.load %28 {alignment = 4 : i64} : !llvm.ptr -> i32
    %30 = llvm.add %29, %1 overflow<nsw> : i32
    llvm.store %30, %28 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    llvm.br ^bb6
  ^bb6:  // pred: ^bb5
    %31 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %32 = llvm.add %31, %0 overflow<nsw> : i32
    llvm.store %32, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1 {loop_annotation = #loop_annotation}
  ^bb7:  // pred: ^bb1
    llvm.return
  }
  llvm.func @printf(!llvm.ptr {llvm.noundef}, ...) -> i32 attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
}
