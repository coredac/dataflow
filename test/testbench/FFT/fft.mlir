#loop_annotation = #llvm.loop_annotation<mustProgress = true>
#loop_vectorize = #llvm.loop_vectorize<disable = false, scalableEnable = false, width = 4 : i32>
#loop_annotation1 = #llvm.loop_annotation<vectorize = #loop_vectorize, mustProgress = true>
module attributes {dlti.dl_spec = #dlti.dl_spec<i64 = dense<64> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f128 = dense<128> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, f80 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.mlir.global external @data_real(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.mlir.global external @data_imag(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.mlir.global external @coef_real(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.mlir.global external @coef_imag(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.func @main() -> (i32 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<all>, no_inline, optimize_none, passthrough = ["mustprogress", "norecurse", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(256 : i32) : i32
    %3 = llvm.mlir.addressof @data_real : !llvm.ptr
    %4 = llvm.mlir.addressof @data_imag : !llvm.ptr
    %5 = llvm.mlir.addressof @coef_real : !llvm.ptr
    %6 = llvm.mlir.addressof @coef_imag : !llvm.ptr
    %7 = llvm.mlir.constant(0 : i64) : i64
    %8 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %9 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %10 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %11 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %12 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %10 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %1, %12 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %1, %11 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb3
    %13 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %14 = llvm.icmp "slt" %13, %2 : i32
    llvm.cond_br %14, ^bb2, ^bb4
  ^bb2:  // pred: ^bb1
    %15 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %16 = llvm.sext %15 : i32 to i64
    %17 = llvm.getelementptr inbounds %4[%7, %16] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
    llvm.store %8, %17 {alignment = 4 : i64} : f32, !llvm.ptr
    %18 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %19 = llvm.sext %18 : i32 to i64
    %20 = llvm.getelementptr inbounds %5[%7, %19] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
    llvm.store %9, %20 {alignment = 4 : i64} : f32, !llvm.ptr
    %21 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %22 = llvm.sext %21 : i32 to i64
    %23 = llvm.getelementptr inbounds %6[%7, %22] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
    llvm.store %9, %23 {alignment = 4 : i64} : f32, !llvm.ptr
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    %24 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %25 = llvm.add %24, %0 overflow<nsw> : i32
    llvm.store %25, %11 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1 {loop_annotation = #loop_annotation}
  ^bb4:  // pred: ^bb1
    llvm.call @_Z6kernelPfS_S_S_(%3, %4, %5, %6) {no_unwind} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return %1 : i32
  }
  llvm.func @_Z6kernelPfS_S_S_(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}, %arg3: !llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(128 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(8 : i32) : i32
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %6 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %8 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %9 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %10 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %11 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %12 = llvm.alloca %0 x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %13 = llvm.alloca %0 x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %14 = llvm.alloca %0 x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %15 = llvm.alloca %0 x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %16 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %17 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %5 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg1, %6 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg2, %7 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg3, %8 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %0, %16 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %1, %17 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %2, %9 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb11
    %18 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> i32
    %19 = llvm.icmp "slt" %18, %3 : i32
    llvm.cond_br %19, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.store %2, %10 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb2, ^bb9
    %20 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> i32
    %21 = llvm.load %16 {alignment = 4 : i64} : !llvm.ptr -> i32
    %22 = llvm.icmp "slt" %20, %21 : i32
    llvm.cond_br %22, ^bb4, ^bb10
  ^bb4:  // pred: ^bb3
    %23 = llvm.load %7 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %24 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> i32
    %25 = llvm.shl %0, %24 : i32
    %26 = llvm.sub %25, %0 overflow<nsw> : i32
    %27 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> i32
    %28 = llvm.add %26, %27 overflow<nsw> : i32
    %29 = llvm.sext %28 : i32 to i64
    %30 = llvm.getelementptr inbounds %23[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %31 = llvm.load %30 {alignment = 4 : i64} : !llvm.ptr -> f32
    llvm.store %31, %14 {alignment = 4 : i64} : f32, !llvm.ptr
    %32 = llvm.load %8 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %33 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> i32
    %34 = llvm.shl %0, %33 : i32
    %35 = llvm.sub %34, %0 overflow<nsw> : i32
    %36 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> i32
    %37 = llvm.add %35, %36 overflow<nsw> : i32
    %38 = llvm.sext %37 : i32 to i64
    %39 = llvm.getelementptr inbounds %32[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %40 = llvm.load %39 {alignment = 4 : i64} : !llvm.ptr -> f32
    llvm.store %40, %15 {alignment = 4 : i64} : f32, !llvm.ptr
    llvm.store %2, %11 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb4, ^bb7
    %41 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %42 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %43 = llvm.icmp "slt" %41, %42 : i32
    llvm.cond_br %43, ^bb6, ^bb8
  ^bb6:  // pred: ^bb5
    %44 = llvm.load %14 {alignment = 4 : i64} : !llvm.ptr -> f32
    %45 = llvm.load %5 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %46 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> i32
    %47 = llvm.mul %4, %46 overflow<nsw> : i32
    %48 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %49 = llvm.mul %47, %48 overflow<nsw> : i32
    %50 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %51 = llvm.add %49, %50 overflow<nsw> : i32
    %52 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %53 = llvm.add %51, %52 overflow<nsw> : i32
    %54 = llvm.sext %53 : i32 to i64
    %55 = llvm.getelementptr inbounds %45[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %56 = llvm.load %55 {alignment = 4 : i64} : !llvm.ptr -> f32
    %57 = llvm.load %15 {alignment = 4 : i64} : !llvm.ptr -> f32
    %58 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %59 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> i32
    %60 = llvm.mul %4, %59 overflow<nsw> : i32
    %61 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %62 = llvm.mul %60, %61 overflow<nsw> : i32
    %63 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %64 = llvm.add %62, %63 overflow<nsw> : i32
    %65 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %66 = llvm.add %64, %65 overflow<nsw> : i32
    %67 = llvm.sext %66 : i32 to i64
    %68 = llvm.getelementptr inbounds %58[%67] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %69 = llvm.load %68 {alignment = 4 : i64} : !llvm.ptr -> f32
    %70 = llvm.fmul %57, %69 : f32
    %71 = llvm.fneg %70 : f32
    %72 = llvm.intr.fmuladd(%44, %56, %71) : (f32, f32, f32) -> f32
    llvm.store %72, %12 {alignment = 4 : i64} : f32, !llvm.ptr
    %73 = llvm.load %15 {alignment = 4 : i64} : !llvm.ptr -> f32
    %74 = llvm.load %5 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %75 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> i32
    %76 = llvm.mul %4, %75 overflow<nsw> : i32
    %77 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %78 = llvm.mul %76, %77 overflow<nsw> : i32
    %79 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %80 = llvm.add %78, %79 overflow<nsw> : i32
    %81 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %82 = llvm.add %80, %81 overflow<nsw> : i32
    %83 = llvm.sext %82 : i32 to i64
    %84 = llvm.getelementptr inbounds %74[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %85 = llvm.load %84 {alignment = 4 : i64} : !llvm.ptr -> f32
    %86 = llvm.load %14 {alignment = 4 : i64} : !llvm.ptr -> f32
    %87 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %88 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> i32
    %89 = llvm.mul %4, %88 overflow<nsw> : i32
    %90 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %91 = llvm.mul %89, %90 overflow<nsw> : i32
    %92 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %93 = llvm.add %91, %92 overflow<nsw> : i32
    %94 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %95 = llvm.add %93, %94 overflow<nsw> : i32
    %96 = llvm.sext %95 : i32 to i64
    %97 = llvm.getelementptr inbounds %87[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %98 = llvm.load %97 {alignment = 4 : i64} : !llvm.ptr -> f32
    %99 = llvm.fmul %86, %98 : f32
    %100 = llvm.intr.fmuladd(%73, %85, %99) : (f32, f32, f32) -> f32
    llvm.store %100, %13 {alignment = 4 : i64} : f32, !llvm.ptr
    %101 = llvm.load %5 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %102 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> i32
    %103 = llvm.mul %4, %102 overflow<nsw> : i32
    %104 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %105 = llvm.mul %103, %104 overflow<nsw> : i32
    %106 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %107 = llvm.add %105, %106 overflow<nsw> : i32
    %108 = llvm.sext %107 : i32 to i64
    %109 = llvm.getelementptr inbounds %101[%108] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %110 = llvm.load %109 {alignment = 4 : i64} : !llvm.ptr -> f32
    %111 = llvm.load %12 {alignment = 4 : i64} : !llvm.ptr -> f32
    %112 = llvm.fsub %110, %111 : f32
    %113 = llvm.load %5 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %114 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> i32
    %115 = llvm.mul %4, %114 overflow<nsw> : i32
    %116 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %117 = llvm.mul %115, %116 overflow<nsw> : i32
    %118 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %119 = llvm.add %117, %118 overflow<nsw> : i32
    %120 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %121 = llvm.add %119, %120 overflow<nsw> : i32
    %122 = llvm.sext %121 : i32 to i64
    %123 = llvm.getelementptr inbounds %113[%122] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %112, %123 {alignment = 4 : i64} : f32, !llvm.ptr
    %124 = llvm.load %12 {alignment = 4 : i64} : !llvm.ptr -> f32
    %125 = llvm.load %5 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %126 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> i32
    %127 = llvm.mul %4, %126 overflow<nsw> : i32
    %128 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %129 = llvm.mul %127, %128 overflow<nsw> : i32
    %130 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %131 = llvm.add %129, %130 overflow<nsw> : i32
    %132 = llvm.sext %131 : i32 to i64
    %133 = llvm.getelementptr inbounds %125[%132] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %134 = llvm.load %133 {alignment = 4 : i64} : !llvm.ptr -> f32
    %135 = llvm.fadd %134, %124 : f32
    llvm.store %135, %133 {alignment = 4 : i64} : f32, !llvm.ptr
    %136 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %137 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> i32
    %138 = llvm.mul %4, %137 overflow<nsw> : i32
    %139 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %140 = llvm.mul %138, %139 overflow<nsw> : i32
    %141 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %142 = llvm.add %140, %141 overflow<nsw> : i32
    %143 = llvm.sext %142 : i32 to i64
    %144 = llvm.getelementptr inbounds %136[%143] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %145 = llvm.load %144 {alignment = 4 : i64} : !llvm.ptr -> f32
    %146 = llvm.load %13 {alignment = 4 : i64} : !llvm.ptr -> f32
    %147 = llvm.fsub %145, %146 : f32
    %148 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %149 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> i32
    %150 = llvm.mul %4, %149 overflow<nsw> : i32
    %151 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %152 = llvm.mul %150, %151 overflow<nsw> : i32
    %153 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %154 = llvm.add %152, %153 overflow<nsw> : i32
    %155 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %156 = llvm.add %154, %155 overflow<nsw> : i32
    %157 = llvm.sext %156 : i32 to i64
    %158 = llvm.getelementptr inbounds %148[%157] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %147, %158 {alignment = 4 : i64} : f32, !llvm.ptr
    %159 = llvm.load %13 {alignment = 4 : i64} : !llvm.ptr -> f32
    %160 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %161 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> i32
    %162 = llvm.mul %4, %161 overflow<nsw> : i32
    %163 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %164 = llvm.mul %162, %163 overflow<nsw> : i32
    %165 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %166 = llvm.add %164, %165 overflow<nsw> : i32
    %167 = llvm.sext %166 : i32 to i64
    %168 = llvm.getelementptr inbounds %160[%167] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %169 = llvm.load %168 {alignment = 4 : i64} : !llvm.ptr -> f32
    %170 = llvm.fadd %169, %159 : f32
    llvm.store %170, %168 {alignment = 4 : i64} : f32, !llvm.ptr
    llvm.br ^bb7
  ^bb7:  // pred: ^bb6
    %171 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %172 = llvm.add %171, %0 overflow<nsw> : i32
    llvm.store %172, %11 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb5 {loop_annotation = #loop_annotation1}
  ^bb8:  // pred: ^bb5
    llvm.br ^bb9
  ^bb9:  // pred: ^bb8
    %173 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> i32
    %174 = llvm.add %173, %0 overflow<nsw> : i32
    llvm.store %174, %10 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb3 {loop_annotation = #loop_annotation}
  ^bb10:  // pred: ^bb3
    %175 = llvm.load %16 {alignment = 4 : i64} : !llvm.ptr -> i32
    %176 = llvm.shl %175, %0 : i32
    llvm.store %176, %16 {alignment = 4 : i64} : i32, !llvm.ptr
    %177 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %178 = llvm.ashr %177, %0 : i32
    llvm.store %178, %17 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb11
  ^bb11:  // pred: ^bb10
    %179 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> i32
    %180 = llvm.add %179, %0 overflow<nsw> : i32
    llvm.store %180, %9 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1 {loop_annotation = #loop_annotation}
  ^bb12:  // pred: ^bb1
    llvm.return
  }
}
