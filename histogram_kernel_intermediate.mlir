module attributes {dlti.dl_spec = #dlti.dl_spec<f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, i128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f64 = dense<64> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, "dlti.stack_alignment" = 128 : i64, "dlti.endianness" = "little">, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  func.func @_Z6kernelPfPi(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = "neura.constant"() <{value = 0 : i64}> : () -> i64
    %1 = "neura.constant"() <{value = -1.000000e+00 : f32}> : () -> f32
    %2 = "neura.constant"() <{value = 5.000000e+00 : f32}> : () -> f32
    %3 = "neura.constant"() <{value = 1.800000e+01 : f32}> : () -> f32
    %4 = "neura.constant"() <{value = 1 : i32}> : () -> i32
    %5 = "neura.constant"() <{value = 1 : i64}> : () -> i64
    %6 = "neura.constant"() <{value = 2 : i64}> : () -> i64
    %7 = "neura.constant"() <{value = 20 : i64}> : () -> i64
    neura.br %0, %arg0, %1, %2, %3, %arg1, %4, %5, %6, %7 : i64, !llvm.ptr, f32, f32, f32, !llvm.ptr, i32, i64, i64, i64 to ^bb1
  ^bb1(%8: i64, %9: !llvm.ptr, %10: f32, %11: f32, %12: f32, %13: !llvm.ptr, %14: i32, %15: i64, %16: i64, %17: i64):  // 2 preds: ^bb0, ^bb1
    %18 = "neura.gep"(%9, %8) : (!llvm.ptr, i64) -> !llvm.ptr
    %19 = "neura.load"(%18) : (!llvm.ptr) -> f32
    %20 = "neura.fadd"(%19, %10) : (f32, f32) -> f32
    %21 = "neura.fmul"(%20, %11) : (f32, f32) -> f32
    %22 = "neura.fdiv"(%21, %12) : (f32, f32) -> f32
    %23 = "neura.cast"(%22) <{cast_type = "fptosi"}> : (f32) -> i32
    %24 = neura.sext %23 : i32 -> i64
    %25 = "neura.gep"(%13, %24) : (!llvm.ptr, i64) -> !llvm.ptr
    %26 = "neura.load"(%25) : (!llvm.ptr) -> i32
    %27 = "neura.add"(%26, %14) : (i32, i32) -> i32
    "neura.store"(%27, %25) : (i32, !llvm.ptr) -> ()
    %28 = "neura.or"(%8, %15) : (i64, i64) -> i64
    %29 = "neura.gep"(%9, %28) : (!llvm.ptr, i64) -> !llvm.ptr
    %30 = "neura.load"(%29) : (!llvm.ptr) -> f32
    %31 = "neura.fadd"(%30, %10) : (f32, f32) -> f32
    %32 = "neura.fmul"(%31, %11) : (f32, f32) -> f32
    %33 = "neura.fdiv"(%32, %12) : (f32, f32) -> f32
    %34 = "neura.cast"(%33) <{cast_type = "fptosi"}> : (f32) -> i32
    %35 = neura.sext %34 : i32 -> i64
    %36 = "neura.gep"(%13, %35) : (!llvm.ptr, i64) -> !llvm.ptr
    %37 = "neura.load"(%36) : (!llvm.ptr) -> i32
    %38 = "neura.add"(%37, %14) : (i32, i32) -> i32
    "neura.store"(%38, %36) : (i32, !llvm.ptr) -> ()
    %39 = "neura.add"(%8, %16) : (i64, i64) -> i64
    %40 = "neura.icmp"(%39, %17) <{cmpType = "eq"}> : (i64, i64) -> i1
    neura.cond_br %40 : i1 then to ^bb2 else %39, %9, %10, %11, %12, %13, %14, %15, %16, %17 : i64, !llvm.ptr, f32, f32, f32, !llvm.ptr, i32, i64, i64, i64 to ^bb1
  ^bb2:  // pred: ^bb1
    "neura.return"() : () -> ()
  }
}

