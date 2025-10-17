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
    neura.br %0 : i64 to ^bb1
  ^bb1(%8: i64):  // 2 preds: ^bb0, ^bb1
    %9 = "neura.gep"(%arg0, %8) : (!llvm.ptr, i64) -> !llvm.ptr
    %10 = "neura.load"(%9) : (!llvm.ptr) -> f32
    %11 = "neura.fadd"(%10, %1) : (f32, f32) -> f32
    %12 = "neura.fmul"(%11, %2) : (f32, f32) -> f32
    %13 = llvm.fdiv %12, %3 : f32
    %14 = llvm.fptosi %13 : f32 to i32
    %15 = neura.sext %14 : i32 -> i64
    %16 = "neura.gep"(%arg1, %15) : (!llvm.ptr, i64) -> !llvm.ptr
    %17 = "neura.load"(%16) : (!llvm.ptr) -> i32
    %18 = "neura.add"(%17, %4) : (i32, i32) -> i32
    "neura.store"(%18, %16) : (i32, !llvm.ptr) -> ()
    %19 = "neura.or"(%8, %5) : (i64, i64) -> i64
    %20 = "neura.gep"(%arg0, %19) : (!llvm.ptr, i64) -> !llvm.ptr
    %21 = "neura.load"(%20) : (!llvm.ptr) -> f32
    %22 = "neura.fadd"(%21, %1) : (f32, f32) -> f32
    %23 = "neura.fmul"(%22, %2) : (f32, f32) -> f32
    %24 = llvm.fdiv %23, %3 : f32
    %25 = llvm.fptosi %24 : f32 to i32
    %26 = neura.sext %25 : i32 -> i64
    %27 = "neura.gep"(%arg1, %26) : (!llvm.ptr, i64) -> !llvm.ptr
    %28 = "neura.load"(%27) : (!llvm.ptr) -> i32
    %29 = "neura.add"(%28, %4) : (i32, i32) -> i32
    "neura.store"(%29, %27) : (i32, !llvm.ptr) -> ()
    %30 = "neura.add"(%8, %6) : (i64, i64) -> i64
    %31 = "neura.icmp"(%30, %7) <{cmpType = "eq"}> : (i64, i64) -> i1
    neura.cond_br %31 : i1 then to ^bb2 else %30 : i64 to ^bb1
  ^bb2:  // pred: ^bb1
    "neura.return"() : () -> ()
  }
}

