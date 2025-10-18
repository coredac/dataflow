module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, f128 = dense<128> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, f80 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.mlir.global external local_unnamed_addr @data_real(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.mlir.global external local_unnamed_addr @data_imag(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.mlir.global external local_unnamed_addr @coef_real(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.mlir.global external local_unnamed_addr @coef_imag(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {memory_effects = #llvm.memory_effects<other = readwrite, argMem = none, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.addressof @data_real : !llvm.ptr
    %1 = llvm.mlir.poison : vector<4xf32>
    %2 = llvm.mlir.addressof @coef_imag : !llvm.ptr
    %3 = llvm.mlir.addressof @coef_real : !llvm.ptr
    %4 = llvm.mlir.addressof @data_imag : !llvm.ptr
    %5 = "neura.constant"() <{value = 0 : i64}> : () -> i64
    %6 = "neura.constant"() <{value = 16 : i64}> : () -> i64
    %7 = "neura.constant"() <{value = dense<1.000000e+00> : vector<4xf32>}> : () -> vector<4xf32>
    %8 = "neura.constant"() <{value = dense<2.000000e+00> : vector<4xf32>}> : () -> vector<4xf32>
    %9 = "neura.constant"() <{value = 8 : i64}> : () -> i64
    %10 = "neura.constant"() <{value = 256 : i64}> : () -> i64
    %11 = "neura.constant"() <{value = 128 : i32}> : () -> i32
    %12 = "neura.constant"() <{value = 1 : i32}> : () -> i32
    %13 = "neura.constant"() <{value = 0 : i32}> : () -> i32
    %14 = "neura.constant"() <{value = -1 : i32}> : () -> i32
    %15 = "neura.constant"() <{value = 4 : i32}> : () -> i32
    %16 = "neura.constant"() <{value = 252 : i64}> : () -> i64
    %17 = "neura.constant"() <{value = 1 : i64}> : () -> i64
    %18 = "neura.constant"() <{value = 4 : i64}> : () -> i64
    %19 = "neura.constant"() <{value = 8 : i32}> : () -> i32
    neura.br %5 : i64 to ^bb1
  ^bb1(%20: i64):  // 2 preds: ^bb0, ^bb1
    %21 = "neura.constant"() <{value = 0 : i32}> : () -> index
    %22 = "neura.gep"(%4, %21, %20) : (!llvm.ptr, index, i64) -> !llvm.ptr
    %23 = "neura.constant"() <{value = 16 : i32}> : () -> index
    %24 = "neura.gep"(%22, %23) : (!llvm.ptr, index) -> !llvm.ptr
    "neura.store"(%7, %22) : (vector<4xf32>, !llvm.ptr) -> ()
    "neura.store"(%7, %24) : (vector<4xf32>, !llvm.ptr) -> ()
    %25 = "neura.constant"() <{value = 0 : i32}> : () -> index
    %26 = "neura.gep"(%3, %25, %20) : (!llvm.ptr, index, i64) -> !llvm.ptr
    %27 = "neura.constant"() <{value = 16 : i32}> : () -> index
    %28 = "neura.gep"(%26, %27) : (!llvm.ptr, index) -> !llvm.ptr
    "neura.store"(%8, %26) : (vector<4xf32>, !llvm.ptr) -> ()
    "neura.store"(%8, %28) : (vector<4xf32>, !llvm.ptr) -> ()
    %29 = "neura.constant"() <{value = 0 : i32}> : () -> index
    %30 = "neura.gep"(%2, %29, %20) : (!llvm.ptr, index, i64) -> !llvm.ptr
    %31 = "neura.constant"() <{value = 16 : i32}> : () -> index
    %32 = "neura.gep"(%30, %31) : (!llvm.ptr, index) -> !llvm.ptr
    "neura.store"(%8, %30) : (vector<4xf32>, !llvm.ptr) -> ()
    "neura.store"(%8, %32) : (vector<4xf32>, !llvm.ptr) -> ()
    %33 = "neura.or"(%20, %9) : (i64, i64) -> i64
    %34 = "neura.constant"() <{value = 0 : i32}> : () -> index
    %35 = "neura.gep"(%4, %34, %33) : (!llvm.ptr, index, i64) -> !llvm.ptr
    %36 = "neura.constant"() <{value = 16 : i32}> : () -> index
    %37 = "neura.gep"(%35, %36) : (!llvm.ptr, index) -> !llvm.ptr
    "neura.store"(%7, %35) : (vector<4xf32>, !llvm.ptr) -> ()
    "neura.store"(%7, %37) : (vector<4xf32>, !llvm.ptr) -> ()
    %38 = "neura.constant"() <{value = 0 : i32}> : () -> index
    %39 = "neura.gep"(%3, %38, %33) : (!llvm.ptr, index, i64) -> !llvm.ptr
    %40 = "neura.constant"() <{value = 16 : i32}> : () -> index
    %41 = "neura.gep"(%39, %40) : (!llvm.ptr, index) -> !llvm.ptr
    "neura.store"(%8, %39) : (vector<4xf32>, !llvm.ptr) -> ()
    "neura.store"(%8, %41) : (vector<4xf32>, !llvm.ptr) -> ()
    %42 = "neura.constant"() <{value = 0 : i32}> : () -> index
    %43 = "neura.gep"(%2, %42, %33) : (!llvm.ptr, index, i64) -> !llvm.ptr
    %44 = "neura.constant"() <{value = 16 : i32}> : () -> index
    %45 = "neura.gep"(%43, %44) : (!llvm.ptr, index) -> !llvm.ptr
    "neura.store"(%8, %43) : (vector<4xf32>, !llvm.ptr) -> ()
    "neura.store"(%8, %45) : (vector<4xf32>, !llvm.ptr) -> ()
    %46 = "neura.add"(%20, %6) : (i64, i64) -> i64
    %47 = "neura.icmp"(%46, %10) <{cmpType = "eq"}> : (i64, i64) -> i1
    neura.cond_br %47 : i1 then %11, %12, %13 : i32, i32, i32 to ^bb2 else %46 : i64 to ^bb1
  ^bb2(%48: i32, %49: i32, %50: i32):  // 2 preds: ^bb1, ^bb10
    %51 = "neura.shl"(%14, %50) : (i32, i32) -> i32
    %52 = llvm.xor %51, %14 : i32
    %53 = neura.zext %48 : i32 -> i64
    %54 = neura.zext %52 : i32 -> i64
    %55 = neura.zext %49 : i32 -> i64
    %56 = "neura.icmp"(%48, %15) <{cmpType = "ult"}> : (i32, i32) -> i1
    %57 = llvm.and %53, %16 : i64
    %58 = "neura.icmp"(%57, %53) <{cmpType = "eq"}> : (i64, i64) -> i1
    neura.br %5 : i64 to ^bb3
  ^bb3(%59: i64):  // 2 preds: ^bb2, ^bb9
    %60 = "neura.add"(%59, %54) : (i64, i64) -> i64
    %61 = "neura.gep"(%3, %60) : (!llvm.ptr, i64) -> !llvm.ptr
    %62 = "neura.load"(%61) : (!llvm.ptr) -> f32
    %63 = "neura.gep"(%2, %60) : (!llvm.ptr, i64) -> !llvm.ptr
    %64 = "neura.load"(%63) : (!llvm.ptr) -> f32
    %65 = "neura.shl"(%59, %17) : (i64, i64) -> i64
    %66 = "neura.mul"(%65, %53) : (i64, i64) -> i64
    %67 = "neura.add"(%66, %53) : (i64, i64) -> i64
    neura.cond_br %56 : i1 then %5 : i64 to ^bb7 else to ^bb4
  ^bb4:  // pred: ^bb3
    %68 = llvm.insertelement %64, %1[%5 : i64] : vector<4xf32>
    %69 = llvm.shufflevector %68, %1 [0, 0, 0, 0] : vector<4xf32> 
    %70 = llvm.insertelement %62, %1[%5 : i64] : vector<4xf32>
    %71 = llvm.shufflevector %70, %1 [0, 0, 0, 0] : vector<4xf32> 
    neura.br %5 : i64 to ^bb5
  ^bb5(%72: i64):  // 2 preds: ^bb4, ^bb5
    %73 = "neura.add"(%67, %72) : (i64, i64) -> i64
    %74 = "neura.gep"(%0, %73) : (!llvm.ptr, i64) -> !llvm.ptr
    %75 = "neura.load"(%74) : (!llvm.ptr) -> vector<4xf32>
    %76 = "neura.gep"(%4, %73) : (!llvm.ptr, i64) -> !llvm.ptr
    %77 = "neura.load"(%76) : (!llvm.ptr) -> vector<4xf32>
    %78 = llvm.fneg %77 : vector<4xf32>
    %79 = "neura.vfmul"(%69, %78) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %80 = llvm.intr.fmuladd(%71, %75, %79) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %81 = "neura.vfmul"(%71, %77) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %82 = llvm.intr.fmuladd(%69, %75, %81) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %83 = "neura.add"(%72, %66) : (i64, i64) -> i64
    %84 = "neura.gep"(%0, %83) : (!llvm.ptr, i64) -> !llvm.ptr
    %85 = "neura.load"(%84) : (!llvm.ptr) -> vector<4xf32>
    %86 = llvm.fsub %85, %80 : vector<4xf32>
    "neura.store"(%86, %74) : (vector<4xf32>, !llvm.ptr) -> ()
    %87 = llvm.fadd %85, %80 : vector<4xf32>
    "neura.store"(%87, %84) : (vector<4xf32>, !llvm.ptr) -> ()
    %88 = "neura.gep"(%4, %83) : (!llvm.ptr, i64) -> !llvm.ptr
    %89 = "neura.load"(%88) : (!llvm.ptr) -> vector<4xf32>
    %90 = llvm.fsub %89, %82 : vector<4xf32>
    "neura.store"(%90, %76) : (vector<4xf32>, !llvm.ptr) -> ()
    %91 = llvm.fadd %82, %89 : vector<4xf32>
    "neura.store"(%91, %88) : (vector<4xf32>, !llvm.ptr) -> ()
    %92 = "neura.add"(%72, %18) : (i64, i64) -> i64
    %93 = "neura.icmp"(%92, %57) <{cmpType = "eq"}> : (i64, i64) -> i1
    neura.cond_br %93 : i1 then to ^bb6 else %92 : i64 to ^bb5
  ^bb6:  // pred: ^bb5
    neura.cond_br %58 : i1 then to ^bb9 else %57 : i64 to ^bb7
  ^bb7(%94: i64):  // 2 preds: ^bb3, ^bb6
    neura.br %94 : i64 to ^bb8
  ^bb8(%95: i64):  // 2 preds: ^bb7, ^bb8
    %96 = "neura.add"(%67, %95) : (i64, i64) -> i64
    %97 = "neura.gep"(%0, %96) : (!llvm.ptr, i64) -> !llvm.ptr
    %98 = "neura.load"(%97) : (!llvm.ptr) -> f32
    %99 = "neura.gep"(%4, %96) : (!llvm.ptr, i64) -> !llvm.ptr
    %100 = "neura.load"(%99) : (!llvm.ptr) -> f32
    %101 = llvm.fneg %100 : f32
    %102 = "neura.fmul"(%64, %101) : (f32, f32) -> f32
    %103 = llvm.intr.fmuladd(%62, %98, %102) : (f32, f32, f32) -> f32
    %104 = "neura.fmul"(%62, %100) : (f32, f32) -> f32
    %105 = llvm.intr.fmuladd(%64, %98, %104) : (f32, f32, f32) -> f32
    %106 = "neura.add"(%95, %66) : (i64, i64) -> i64
    %107 = "neura.gep"(%0, %106) : (!llvm.ptr, i64) -> !llvm.ptr
    %108 = "neura.load"(%107) : (!llvm.ptr) -> f32
    %109 = "neura.fsub"(%108, %103) : (f32, f32) -> f32
    "neura.store"(%109, %97) : (f32, !llvm.ptr) -> ()
    %110 = "neura.fadd"(%108, %103) : (f32, f32) -> f32
    "neura.store"(%110, %107) : (f32, !llvm.ptr) -> ()
    %111 = "neura.gep"(%4, %106) : (!llvm.ptr, i64) -> !llvm.ptr
    %112 = "neura.load"(%111) : (!llvm.ptr) -> f32
    %113 = "neura.fsub"(%112, %105) : (f32, f32) -> f32
    "neura.store"(%113, %99) : (f32, !llvm.ptr) -> ()
    %114 = "neura.fadd"(%105, %112) : (f32, f32) -> f32
    "neura.store"(%114, %111) : (f32, !llvm.ptr) -> ()
    %115 = "neura.add"(%95, %17) : (i64, i64) -> i64
    %116 = "neura.icmp"(%115, %53) <{cmpType = "eq"}> : (i64, i64) -> i1
    neura.cond_br %116 : i1 then to ^bb9 else %115 : i64 to ^bb8
  ^bb9:  // 2 preds: ^bb6, ^bb8
    %117 = "neura.add"(%59, %17) : (i64, i64) -> i64
    %118 = "neura.icmp"(%117, %55) <{cmpType = "eq"}> : (i64, i64) -> i1
    neura.cond_br %118 : i1 then to ^bb10 else %117 : i64 to ^bb3
  ^bb10:  // pred: ^bb9
    %119 = "neura.shl"(%49, %12) : (i32, i32) -> i32
    %120 = llvm.lshr %48, %12 : i32
    %121 = "neura.add"(%50, %12) : (i32, i32) -> i32
    %122 = "neura.icmp"(%121, %19) <{cmpType = "eq"}> : (i32, i32) -> i1
    neura.cond_br %122 : i1 then to ^bb11 else %120, %119, %121 : i32, i32, i32 to ^bb2
  ^bb11:  // pred: ^bb10
    "neura.return"(%13) : (i32) -> ()
  }
  func.func @_Z6kernelPfS_S_S_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = llvm.mlir.poison : vector<4xf32>
    %1 = "neura.constant"() <{value = 128 : i32}> : () -> i32
    %2 = "neura.constant"() <{value = 1 : i32}> : () -> i32
    %3 = "neura.constant"() <{value = 0 : i32}> : () -> i32
    %4 = "neura.constant"() <{value = -1 : i32}> : () -> i32
    %5 = "neura.constant"() <{value = 3 : i64}> : () -> i64
    %6 = "neura.constant"() <{value = -4 : i64}> : () -> i64
    %7 = "neura.constant"() <{value = 2 : i64}> : () -> i64
    %8 = "neura.constant"() <{value = 4 : i32}> : () -> i32
    %9 = "neura.constant"() <{value = 252 : i64}> : () -> i64
    %10 = "neura.constant"() <{value = 0 : i64}> : () -> i64
    %11 = "neura.constant"() <{value = 1 : i64}> : () -> i64
    %12 = "neura.constant"() <{value = true}> : () -> i1
    %13 = "neura.constant"() <{value = 4 : i64}> : () -> i64
    %14 = "neura.constant"() <{value = 8 : i32}> : () -> i32
    neura.br %1, %2, %3 : i32, i32, i32 to ^bb1
  ^bb1(%15: i32, %16: i32, %17: i32):  // 2 preds: ^bb0, ^bb9
    %18 = "neura.shl"(%4, %17) : (i32, i32) -> i32
    %19 = llvm.xor %18, %4 : i32
    %20 = neura.zext %15 : i32 -> i64
    %21 = neura.zext %19 : i32 -> i64
    %22 = neura.zext %16 : i32 -> i64
    %23 = "neura.shl"(%22, %5) : (i64, i64) -> i64
    %24 = "neura.add"(%23, %6) : (i64, i64) -> i64
    %25 = "neura.mul"(%24, %20) : (i64, i64) -> i64
    %26 = "neura.gep"(%arg0, %25) : (!llvm.ptr, i64) -> !llvm.ptr
    %27 = "neura.gep"(%arg1, %25) : (!llvm.ptr, i64) -> !llvm.ptr
    %28 = "neura.shl"(%20, %7) : (i64, i64) -> i64
    %29 = "neura.gep"(%arg1, %28) : (!llvm.ptr, i64) -> !llvm.ptr
    %30 = "neura.shl"(%22, %5) : (i64, i64) -> i64
    %31 = "neura.mul"(%30, %20) : (i64, i64) -> i64
    %32 = "neura.gep"(%arg1, %31) : (!llvm.ptr, i64) -> !llvm.ptr
    %33 = "neura.gep"(%arg0, %28) : (!llvm.ptr, i64) -> !llvm.ptr
    %34 = "neura.gep"(%arg0, %31) : (!llvm.ptr, i64) -> !llvm.ptr
    %35 = "neura.icmp"(%15, %8) <{cmpType = "ult"}> : (i32, i32) -> i1
    %36 = "neura.icmp"(%arg0, %27) <{cmpType = "ult"}> : (!llvm.ptr, !llvm.ptr) -> i1
    %37 = "neura.icmp"(%arg1, %26) <{cmpType = "ult"}> : (!llvm.ptr, !llvm.ptr) -> i1
    %38 = llvm.and %36, %37 : i1
    %39 = "neura.icmp"(%arg0, %32) <{cmpType = "ult"}> : (!llvm.ptr, !llvm.ptr) -> i1
    %40 = "neura.icmp"(%29, %26) <{cmpType = "ult"}> : (!llvm.ptr, !llvm.ptr) -> i1
    %41 = llvm.and %39, %40 : i1
    %42 = "neura.or"(%38, %41) : (i1, i1) -> i1
    %43 = "neura.icmp"(%33, %27) <{cmpType = "ult"}> : (!llvm.ptr, !llvm.ptr) -> i1
    %44 = "neura.icmp"(%arg1, %34) <{cmpType = "ult"}> : (!llvm.ptr, !llvm.ptr) -> i1
    %45 = llvm.and %43, %44 : i1
    %46 = "neura.or"(%42, %45) : (i1, i1) -> i1
    %47 = "neura.icmp"(%33, %32) <{cmpType = "ult"}> : (!llvm.ptr, !llvm.ptr) -> i1
    %48 = "neura.icmp"(%29, %34) <{cmpType = "ult"}> : (!llvm.ptr, !llvm.ptr) -> i1
    %49 = llvm.and %47, %48 : i1
    %50 = "neura.or"(%46, %49) : (i1, i1) -> i1
    %51 = llvm.and %20, %9 : i64
    %52 = "neura.icmp"(%51, %20) <{cmpType = "eq"}> : (i64, i64) -> i1
    neura.br %10 : i64 to ^bb2
  ^bb2(%53: i64):  // 2 preds: ^bb1, ^bb8
    %54 = "neura.add"(%53, %21) : (i64, i64) -> i64
    %55 = "neura.gep"(%arg2, %54) : (!llvm.ptr, i64) -> !llvm.ptr
    %56 = "neura.load"(%55) : (!llvm.ptr) -> f32
    %57 = "neura.gep"(%arg3, %54) : (!llvm.ptr, i64) -> !llvm.ptr
    %58 = "neura.load"(%57) : (!llvm.ptr) -> f32
    %59 = "neura.shl"(%53, %11) : (i64, i64) -> i64
    %60 = "neura.mul"(%59, %20) : (i64, i64) -> i64
    %61 = "neura.add"(%60, %20) : (i64, i64) -> i64
    %62 = llvm.select %35, %12, %50 : i1, i1
    neura.cond_br %62 : i1 then %10 : i64 to ^bb6 else to ^bb3
  ^bb3:  // pred: ^bb2
    %63 = llvm.insertelement %58, %0[%10 : i64] : vector<4xf32>
    %64 = llvm.shufflevector %63, %0 [0, 0, 0, 0] : vector<4xf32> 
    %65 = llvm.insertelement %56, %0[%10 : i64] : vector<4xf32>
    %66 = llvm.shufflevector %65, %0 [0, 0, 0, 0] : vector<4xf32> 
    neura.br %10 : i64 to ^bb4
  ^bb4(%67: i64):  // 2 preds: ^bb3, ^bb4
    %68 = "neura.add"(%61, %67) : (i64, i64) -> i64
    %69 = "neura.gep"(%arg0, %68) : (!llvm.ptr, i64) -> !llvm.ptr
    %70 = "neura.load"(%69) : (!llvm.ptr) -> vector<4xf32>
    %71 = "neura.gep"(%arg1, %68) : (!llvm.ptr, i64) -> !llvm.ptr
    %72 = "neura.load"(%71) : (!llvm.ptr) -> vector<4xf32>
    %73 = llvm.fneg %72 : vector<4xf32>
    %74 = "neura.vfmul"(%64, %73) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %75 = llvm.intr.fmuladd(%66, %70, %74) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %76 = "neura.vfmul"(%66, %72) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %77 = llvm.intr.fmuladd(%64, %70, %76) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %78 = "neura.add"(%67, %60) : (i64, i64) -> i64
    %79 = "neura.gep"(%arg0, %78) : (!llvm.ptr, i64) -> !llvm.ptr
    %80 = "neura.load"(%79) : (!llvm.ptr) -> vector<4xf32>
    %81 = llvm.fsub %80, %75 : vector<4xf32>
    "neura.store"(%81, %69) : (vector<4xf32>, !llvm.ptr) -> ()
    %82 = llvm.fadd %80, %75 : vector<4xf32>
    "neura.store"(%82, %79) : (vector<4xf32>, !llvm.ptr) -> ()
    %83 = "neura.gep"(%arg1, %78) : (!llvm.ptr, i64) -> !llvm.ptr
    %84 = "neura.load"(%83) : (!llvm.ptr) -> vector<4xf32>
    %85 = llvm.fsub %84, %77 : vector<4xf32>
    "neura.store"(%85, %71) : (vector<4xf32>, !llvm.ptr) -> ()
    %86 = llvm.fadd %77, %84 : vector<4xf32>
    "neura.store"(%86, %83) : (vector<4xf32>, !llvm.ptr) -> ()
    %87 = "neura.add"(%67, %13) : (i64, i64) -> i64
    %88 = "neura.icmp"(%87, %51) <{cmpType = "eq"}> : (i64, i64) -> i1
    neura.cond_br %88 : i1 then to ^bb5 else %87 : i64 to ^bb4
  ^bb5:  // pred: ^bb4
    neura.cond_br %52 : i1 then to ^bb8 else %51 : i64 to ^bb6
  ^bb6(%89: i64):  // 2 preds: ^bb2, ^bb5
    neura.br %89 : i64 to ^bb7
  ^bb7(%90: i64):  // 2 preds: ^bb6, ^bb7
    %91 = "neura.add"(%61, %90) : (i64, i64) -> i64
    %92 = "neura.gep"(%arg0, %91) : (!llvm.ptr, i64) -> !llvm.ptr
    %93 = "neura.load"(%92) : (!llvm.ptr) -> f32
    %94 = "neura.gep"(%arg1, %91) : (!llvm.ptr, i64) -> !llvm.ptr
    %95 = "neura.load"(%94) : (!llvm.ptr) -> f32
    %96 = llvm.fneg %95 : f32
    %97 = "neura.fmul"(%58, %96) : (f32, f32) -> f32
    %98 = llvm.intr.fmuladd(%56, %93, %97) : (f32, f32, f32) -> f32
    %99 = "neura.fmul"(%56, %95) : (f32, f32) -> f32
    %100 = llvm.intr.fmuladd(%58, %93, %99) : (f32, f32, f32) -> f32
    %101 = "neura.add"(%90, %60) : (i64, i64) -> i64
    %102 = "neura.gep"(%arg0, %101) : (!llvm.ptr, i64) -> !llvm.ptr
    %103 = "neura.load"(%102) : (!llvm.ptr) -> f32
    %104 = "neura.fsub"(%103, %98) : (f32, f32) -> f32
    "neura.store"(%104, %92) : (f32, !llvm.ptr) -> ()
    %105 = "neura.fadd"(%103, %98) : (f32, f32) -> f32
    "neura.store"(%105, %102) : (f32, !llvm.ptr) -> ()
    %106 = "neura.gep"(%arg1, %101) : (!llvm.ptr, i64) -> !llvm.ptr
    %107 = "neura.load"(%106) : (!llvm.ptr) -> f32
    %108 = "neura.fsub"(%107, %100) : (f32, f32) -> f32
    "neura.store"(%108, %94) : (f32, !llvm.ptr) -> ()
    %109 = "neura.fadd"(%100, %107) : (f32, f32) -> f32
    "neura.store"(%109, %106) : (f32, !llvm.ptr) -> ()
    %110 = "neura.add"(%90, %11) : (i64, i64) -> i64
    %111 = "neura.icmp"(%110, %20) <{cmpType = "eq"}> : (i64, i64) -> i1
    neura.cond_br %111 : i1 then to ^bb8 else %110 : i64 to ^bb7
  ^bb8:  // 2 preds: ^bb5, ^bb7
    %112 = "neura.add"(%53, %11) : (i64, i64) -> i64
    %113 = "neura.icmp"(%112, %22) <{cmpType = "eq"}> : (i64, i64) -> i1
    neura.cond_br %113 : i1 then to ^bb9 else %112 : i64 to ^bb2
  ^bb9:  // pred: ^bb8
    %114 = "neura.shl"(%16, %2) : (i32, i32) -> i32
    %115 = llvm.lshr %15, %2 : i32
    %116 = "neura.add"(%17, %2) : (i32, i32) -> i32
    %117 = "neura.icmp"(%116, %14) <{cmpType = "eq"}> : (i32, i32) -> i1
    neura.cond_br %117 : i1 then to ^bb10 else %115, %114, %116 : i32, i32, i32 to ^bb1
  ^bb10:  // pred: ^bb9
    "neura.return"() : () -> ()
  }
}

