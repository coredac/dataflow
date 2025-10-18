module attributes {dlti.dl_spec = #dlti.dl_spec<i64 = dense<64> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, f80 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, "dlti.stack_alignment" = 128 : i64, "dlti.endianness" = "little">, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.mlir.global external local_unnamed_addr @data_real(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.mlir.global external local_unnamed_addr @data_imag(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.mlir.global external local_unnamed_addr @coef_real(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.mlir.global external local_unnamed_addr @coef_imag(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {memory_effects = #llvm.memory_effects<other = readwrite, argMem = none, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.poison : vector<4xf32>
    %1 = llvm.mlir.addressof @coef_imag : !llvm.ptr
    %2 = llvm.mlir.addressof @coef_real : !llvm.ptr
    %3 = llvm.mlir.addressof @data_imag : !llvm.ptr
    %4 = llvm.mlir.addressof @data_real : !llvm.ptr
    %5 = "neura.constant"() <{value = 0 : i64}> : () -> i64
    %6 = "neura.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
    %7 = "neura.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
    %8 = "neura.constant"() <{value = 1 : i64}> : () -> i64
    %9 = "neura.constant"() <{value = 256 : i64}> : () -> i64
    %10 = "neura.constant"() <{value = 128 : i32}> : () -> i32
    %11 = "neura.constant"() <{value = 1 : i32}> : () -> i32
    %12 = "neura.constant"() <{value = 0 : i32}> : () -> i32
    %13 = "neura.constant"() <{value = -1 : i32}> : () -> i32
    %14 = "neura.constant"() <{value = 4 : i32}> : () -> i32
    %15 = "neura.constant"() <{value = 252 : i64}> : () -> i64
    %16 = "neura.constant"() <{value = 4 : i64}> : () -> i64
    %17 = "neura.constant"() <{value = 8 : i32}> : () -> i32
    neura.br %5 : i64 to ^bb1
  ^bb1(%18: i64):  // 2 preds: ^bb0, ^bb1
    %19 = llvm.trunc %18 overflow<nsw, nuw> : i64 to i32
    %20 = llvm.uitofp nneg %19 : i32 to f32
    %21 = "neura.constant"() <{value = 0 : i32}> : () -> index
    %22 = "neura.gep"(%4, %21, %18) : (!llvm.ptr, index, i64) -> !llvm.ptr
    "neura.store"(%20, %22) : (f32, !llvm.ptr) -> ()
    %23 = "neura.constant"() <{value = 0 : i32}> : () -> index
    %24 = "neura.gep"(%3, %23, %18) : (!llvm.ptr, index, i64) -> !llvm.ptr
    "neura.store"(%6, %24) : (f32, !llvm.ptr) -> ()
    %25 = "neura.constant"() <{value = 0 : i32}> : () -> index
    %26 = "neura.gep"(%2, %25, %18) : (!llvm.ptr, index, i64) -> !llvm.ptr
    "neura.store"(%7, %26) : (f32, !llvm.ptr) -> ()
    %27 = "neura.constant"() <{value = 0 : i32}> : () -> index
    %28 = "neura.gep"(%1, %27, %18) : (!llvm.ptr, index, i64) -> !llvm.ptr
    "neura.store"(%7, %28) : (f32, !llvm.ptr) -> ()
    %29 = "neura.add"(%18, %8) : (i64, i64) -> i64
    %30 = "neura.icmp"(%29, %9) <{cmpType = "eq"}> : (i64, i64) -> i1
    neura.cond_br %30 : i1 then %10, %11, %12 : i32, i32, i32 to ^bb2 else %29 : i64 to ^bb1
  ^bb2(%31: i32, %32: i32, %33: i32):  // 2 preds: ^bb1, ^bb10
    %34 = "neura.shl"(%13, %33) : (i32, i32) -> i32
    %35 = llvm.xor %34, %13 : i32
    %36 = neura.zext %31 : i32 -> i64
    %37 = neura.zext %35 : i32 -> i64
    %38 = neura.zext %32 : i32 -> i64
    %39 = "neura.icmp"(%31, %14) <{cmpType = "ult"}> : (i32, i32) -> i1
    %40 = llvm.and %36, %15 : i64
    %41 = "neura.icmp"(%40, %36) <{cmpType = "eq"}> : (i64, i64) -> i1
    neura.br %5 : i64 to ^bb3
  ^bb3(%42: i64):  // 2 preds: ^bb2, ^bb9
    %43 = "neura.add"(%42, %37) : (i64, i64) -> i64
    %44 = "neura.gep"(%2, %43) : (!llvm.ptr, i64) -> !llvm.ptr
    %45 = "neura.load"(%44) : (!llvm.ptr) -> f32
    %46 = "neura.gep"(%1, %43) : (!llvm.ptr, i64) -> !llvm.ptr
    %47 = "neura.load"(%46) : (!llvm.ptr) -> f32
    %48 = "neura.shl"(%42, %8) : (i64, i64) -> i64
    %49 = "neura.mul"(%48, %36) : (i64, i64) -> i64
    %50 = "neura.add"(%49, %36) : (i64, i64) -> i64
    neura.cond_br %39 : i1 then %5 : i64 to ^bb7 else to ^bb4
  ^bb4:  // pred: ^bb3
    %51 = llvm.insertelement %47, %0[%5 : i64] : vector<4xf32>
    %52 = llvm.shufflevector %51, %0 [0, 0, 0, 0] : vector<4xf32> 
    %53 = llvm.insertelement %45, %0[%5 : i64] : vector<4xf32>
    %54 = llvm.shufflevector %53, %0 [0, 0, 0, 0] : vector<4xf32> 
    neura.br %5 : i64 to ^bb5
  ^bb5(%55: i64):  // 2 preds: ^bb4, ^bb5
    %56 = "neura.add"(%50, %55) : (i64, i64) -> i64
    %57 = "neura.gep"(%4, %56) : (!llvm.ptr, i64) -> !llvm.ptr
    %58 = "neura.load"(%57) : (!llvm.ptr) -> vector<4xf32>
    %59 = "neura.gep"(%3, %56) : (!llvm.ptr, i64) -> !llvm.ptr
    %60 = "neura.load"(%59) : (!llvm.ptr) -> vector<4xf32>
    %61 = llvm.fneg %60 : vector<4xf32>
    %62 = "neura.vfmul"(%52, %61) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %63 = llvm.intr.fmuladd(%54, %58, %62) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %64 = "neura.vfmul"(%54, %60) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %65 = llvm.intr.fmuladd(%52, %58, %64) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %66 = "neura.add"(%55, %49) : (i64, i64) -> i64
    %67 = "neura.gep"(%4, %66) : (!llvm.ptr, i64) -> !llvm.ptr
    %68 = "neura.load"(%67) : (!llvm.ptr) -> vector<4xf32>
    %69 = llvm.fsub %68, %63 : vector<4xf32>
    "neura.store"(%69, %57) : (vector<4xf32>, !llvm.ptr) -> ()
    %70 = llvm.fadd %68, %63 : vector<4xf32>
    "neura.store"(%70, %67) : (vector<4xf32>, !llvm.ptr) -> ()
    %71 = "neura.gep"(%3, %66) : (!llvm.ptr, i64) -> !llvm.ptr
    %72 = "neura.load"(%71) : (!llvm.ptr) -> vector<4xf32>
    %73 = llvm.fsub %72, %65 : vector<4xf32>
    "neura.store"(%73, %59) : (vector<4xf32>, !llvm.ptr) -> ()
    %74 = llvm.fadd %65, %72 : vector<4xf32>
    "neura.store"(%74, %71) : (vector<4xf32>, !llvm.ptr) -> ()
    %75 = "neura.add"(%55, %16) : (i64, i64) -> i64
    %76 = "neura.icmp"(%75, %40) <{cmpType = "eq"}> : (i64, i64) -> i1
    neura.cond_br %76 : i1 then to ^bb6 else %75 : i64 to ^bb5
  ^bb6:  // pred: ^bb5
    neura.cond_br %41 : i1 then to ^bb9 else %40 : i64 to ^bb7
  ^bb7(%77: i64):  // 2 preds: ^bb3, ^bb6
    neura.br %77 : i64 to ^bb8
  ^bb8(%78: i64):  // 2 preds: ^bb7, ^bb8
    %79 = "neura.add"(%50, %78) : (i64, i64) -> i64
    %80 = "neura.gep"(%4, %79) : (!llvm.ptr, i64) -> !llvm.ptr
    %81 = "neura.load"(%80) : (!llvm.ptr) -> f32
    %82 = "neura.gep"(%3, %79) : (!llvm.ptr, i64) -> !llvm.ptr
    %83 = "neura.load"(%82) : (!llvm.ptr) -> f32
    %84 = llvm.fneg %83 : f32
    %85 = "neura.fmul"(%47, %84) : (f32, f32) -> f32
    %86 = llvm.intr.fmuladd(%45, %81, %85) : (f32, f32, f32) -> f32
    %87 = "neura.fmul"(%45, %83) : (f32, f32) -> f32
    %88 = llvm.intr.fmuladd(%47, %81, %87) : (f32, f32, f32) -> f32
    %89 = "neura.add"(%78, %49) : (i64, i64) -> i64
    %90 = "neura.gep"(%4, %89) : (!llvm.ptr, i64) -> !llvm.ptr
    %91 = "neura.load"(%90) : (!llvm.ptr) -> f32
    %92 = "neura.fsub"(%91, %86) : (f32, f32) -> f32
    "neura.store"(%92, %80) : (f32, !llvm.ptr) -> ()
    %93 = "neura.fadd"(%91, %86) : (f32, f32) -> f32
    "neura.store"(%93, %90) : (f32, !llvm.ptr) -> ()
    %94 = "neura.gep"(%3, %89) : (!llvm.ptr, i64) -> !llvm.ptr
    %95 = "neura.load"(%94) : (!llvm.ptr) -> f32
    %96 = "neura.fsub"(%95, %88) : (f32, f32) -> f32
    "neura.store"(%96, %82) : (f32, !llvm.ptr) -> ()
    %97 = "neura.fadd"(%88, %95) : (f32, f32) -> f32
    "neura.store"(%97, %94) : (f32, !llvm.ptr) -> ()
    %98 = "neura.add"(%78, %8) : (i64, i64) -> i64
    %99 = "neura.icmp"(%98, %36) <{cmpType = "eq"}> : (i64, i64) -> i1
    neura.cond_br %99 : i1 then to ^bb9 else %98 : i64 to ^bb8
  ^bb9:  // 2 preds: ^bb6, ^bb8
    %100 = "neura.add"(%42, %8) : (i64, i64) -> i64
    %101 = "neura.icmp"(%100, %38) <{cmpType = "eq"}> : (i64, i64) -> i1
    neura.cond_br %101 : i1 then to ^bb10 else %100 : i64 to ^bb3
  ^bb10:  // pred: ^bb9
    %102 = "neura.shl"(%32, %11) : (i32, i32) -> i32
    %103 = llvm.lshr %31, %11 : i32
    %104 = "neura.add"(%33, %11) : (i32, i32) -> i32
    %105 = "neura.icmp"(%104, %17) <{cmpType = "eq"}> : (i32, i32) -> i1
    neura.cond_br %105 : i1 then to ^bb11 else %103, %102, %104 : i32, i32, i32 to ^bb2
  ^bb11:  // pred: ^bb10
    "neura.return"(%12) : (i32) -> ()
  }
  llvm.func local_unnamed_addr @_Z6kernelPfS_S_S_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {accelerator = "neura", memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
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

