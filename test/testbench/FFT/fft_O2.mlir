#alias_scope_domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "LVerDomain">
#loop_annotation = #llvm.loop_annotation<mustProgress = true>
#loop_unroll = #llvm.loop_unroll<runtimeDisable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#alias_scope = #llvm.alias_scope<id = distinct[1]<>, domain = #alias_scope_domain>
#alias_scope1 = #llvm.alias_scope<id = distinct[2]<>, domain = #alias_scope_domain>
#alias_scope2 = #llvm.alias_scope<id = distinct[3]<>, domain = #alias_scope_domain>
#alias_scope3 = #llvm.alias_scope<id = distinct[4]<>, domain = #alias_scope_domain>
#loop_annotation1 = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true, isVectorized = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "float", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, f128 = dense<128> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, f80 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.mlir.global external local_unnamed_addr @data_real(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.mlir.global external local_unnamed_addr @data_imag(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.mlir.global external local_unnamed_addr @coef_real(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.mlir.global external local_unnamed_addr @coef_imag(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {memory_effects = #llvm.memory_effects<other = readwrite, argMem = none, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.addressof @data_imag : !llvm.ptr
    %2 = llvm.mlir.constant(16 : i64) : i64
    %3 = llvm.mlir.constant(dense<1.000000e+00> : vector<4xf32>) : vector<4xf32>
    %4 = llvm.mlir.addressof @coef_real : !llvm.ptr
    %5 = llvm.mlir.constant(dense<2.000000e+00> : vector<4xf32>) : vector<4xf32>
    %6 = llvm.mlir.addressof @coef_imag : !llvm.ptr
    %7 = llvm.mlir.constant(8 : i64) : i64
    %8 = llvm.mlir.constant(256 : i64) : i64
    %9 = llvm.mlir.constant(128 : i32) : i32
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.mlir.constant(0 : i32) : i32
    %12 = llvm.mlir.constant(-1 : i32) : i32
    %13 = llvm.mlir.constant(4 : i32) : i32
    %14 = llvm.mlir.constant(252 : i64) : i64
    %15 = llvm.mlir.constant(1 : i64) : i64
    %16 = llvm.mlir.poison : vector<4xf32>
    %17 = llvm.mlir.addressof @data_real : !llvm.ptr
    %18 = llvm.mlir.constant(4 : i64) : i64
    %19 = llvm.mlir.constant(8 : i32) : i32
    llvm.br ^bb1(%0 : i64)
  ^bb1(%20: i64):  // 2 preds: ^bb0, ^bb1
    %21 = llvm.getelementptr inbounds %1[%0, %20] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
    %22 = llvm.getelementptr inbounds %21[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %3, %21 {alignment = 16 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    llvm.store %3, %22 {alignment = 16 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %23 = llvm.getelementptr inbounds %4[%0, %20] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
    %24 = llvm.getelementptr inbounds %23[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %5, %23 {alignment = 16 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    llvm.store %5, %24 {alignment = 16 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %25 = llvm.getelementptr inbounds %6[%0, %20] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
    %26 = llvm.getelementptr inbounds %25[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %5, %25 {alignment = 16 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    llvm.store %5, %26 {alignment = 16 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %27 = llvm.or disjoint %20, %7 : i64
    %28 = llvm.getelementptr inbounds %1[%0, %27] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
    %29 = llvm.getelementptr inbounds %28[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %3, %28 {alignment = 16 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    llvm.store %3, %29 {alignment = 16 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %30 = llvm.getelementptr inbounds %4[%0, %27] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
    %31 = llvm.getelementptr inbounds %30[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %5, %30 {alignment = 16 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    llvm.store %5, %31 {alignment = 16 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %32 = llvm.getelementptr inbounds %6[%0, %27] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
    %33 = llvm.getelementptr inbounds %32[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %5, %32 {alignment = 16 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    llvm.store %5, %33 {alignment = 16 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %34 = llvm.add %20, %2 overflow<nsw, nuw> : i64
    %35 = llvm.icmp "eq" %34, %8 : i64
    llvm.cond_br %35, ^bb2(%9, %10, %11 : i32, i32, i32), ^bb1(%34 : i64) {loop_annotation = #loop_annotation1}
  ^bb2(%36: i32, %37: i32, %38: i32):  // 2 preds: ^bb1, ^bb10
    %39 = llvm.shl %12, %38 overflow<nsw> : i32
    %40 = llvm.xor %39, %12 : i32
    %41 = llvm.zext nneg %36 : i32 to i64
    %42 = llvm.zext nneg %40 : i32 to i64
    %43 = llvm.zext %37 : i32 to i64
    %44 = llvm.icmp "ult" %36, %13 : i32
    %45 = llvm.and %41, %14 : i64
    %46 = llvm.icmp "eq" %45, %41 : i64
    llvm.br ^bb3(%0 : i64)
  ^bb3(%47: i64):  // 2 preds: ^bb2, ^bb9
    %48 = llvm.add %47, %42 overflow<nsw, nuw> : i64
    %49 = llvm.getelementptr inbounds %4[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %50 = llvm.load %49 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %51 = llvm.getelementptr inbounds %6[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %52 = llvm.load %51 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %53 = llvm.shl %47, %15 overflow<nsw, nuw> : i64
    %54 = llvm.mul %53, %41 overflow<nsw, nuw> : i64
    %55 = llvm.add %54, %41 overflow<nsw, nuw> : i64
    llvm.cond_br %44, ^bb7(%0 : i64), ^bb4
  ^bb4:  // pred: ^bb3
    %56 = llvm.insertelement %52, %16[%0 : i64] : vector<4xf32>
    %57 = llvm.shufflevector %56, %16 [0, 0, 0, 0] : vector<4xf32> 
    %58 = llvm.insertelement %50, %16[%0 : i64] : vector<4xf32>
    %59 = llvm.shufflevector %58, %16 [0, 0, 0, 0] : vector<4xf32> 
    llvm.br ^bb5(%0 : i64)
  ^bb5(%60: i64):  // 2 preds: ^bb4, ^bb5
    %61 = llvm.add %55, %60 overflow<nsw, nuw> : i64
    %62 = llvm.getelementptr inbounds %17[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %63 = llvm.load %62 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xf32>
    %64 = llvm.getelementptr inbounds %1[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %65 = llvm.load %64 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xf32>
    %66 = llvm.fneg %65 : vector<4xf32>
    %67 = llvm.fmul %57, %66 : vector<4xf32>
    %68 = llvm.intr.fmuladd(%59, %63, %67) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %69 = llvm.fmul %59, %65 : vector<4xf32>
    %70 = llvm.intr.fmuladd(%57, %63, %69) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %71 = llvm.add %60, %54 overflow<nsw, nuw> : i64
    %72 = llvm.getelementptr inbounds %17[%71] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %73 = llvm.load %72 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xf32>
    %74 = llvm.fsub %73, %68 : vector<4xf32>
    llvm.store %74, %62 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %75 = llvm.fadd %73, %68 : vector<4xf32>
    llvm.store %75, %72 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %76 = llvm.getelementptr inbounds %1[%71] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %77 = llvm.load %76 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xf32>
    %78 = llvm.fsub %77, %70 : vector<4xf32>
    llvm.store %78, %64 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %79 = llvm.fadd %70, %77 : vector<4xf32>
    llvm.store %79, %76 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %80 = llvm.add %60, %18 overflow<nuw> : i64
    %81 = llvm.icmp "eq" %80, %45 : i64
    llvm.cond_br %81, ^bb6, ^bb5(%80 : i64)
  ^bb6:  // pred: ^bb5
    llvm.cond_br %46, ^bb9, ^bb7(%45 : i64)
  ^bb7(%82: i64):  // 2 preds: ^bb3, ^bb6
    llvm.br ^bb8(%82 : i64)
  ^bb8(%83: i64):  // 2 preds: ^bb7, ^bb8
    %84 = llvm.add %55, %83 overflow<nsw, nuw> : i64
    %85 = llvm.getelementptr inbounds %17[%84] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %86 = llvm.load %85 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %87 = llvm.getelementptr inbounds %1[%84] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %88 = llvm.load %87 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %89 = llvm.fneg %88 : f32
    %90 = llvm.fmul %52, %89 : f32
    %91 = llvm.intr.fmuladd(%50, %86, %90) : (f32, f32, f32) -> f32
    %92 = llvm.fmul %50, %88 : f32
    %93 = llvm.intr.fmuladd(%52, %86, %92) : (f32, f32, f32) -> f32
    %94 = llvm.add %83, %54 overflow<nsw, nuw> : i64
    %95 = llvm.getelementptr inbounds %17[%94] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %96 = llvm.load %95 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %97 = llvm.fsub %96, %91 : f32
    llvm.store %97, %85 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %98 = llvm.fadd %96, %91 : f32
    llvm.store %98, %95 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %99 = llvm.getelementptr inbounds %1[%94] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %100 = llvm.load %99 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %101 = llvm.fsub %100, %93 : f32
    llvm.store %101, %87 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %102 = llvm.fadd %93, %100 : f32
    llvm.store %102, %99 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %103 = llvm.add %83, %15 overflow<nsw, nuw> : i64
    %104 = llvm.icmp "eq" %103, %41 : i64
    llvm.cond_br %104, ^bb9, ^bb8(%103 : i64)
  ^bb9:  // 2 preds: ^bb6, ^bb8
    %105 = llvm.add %47, %15 overflow<nsw, nuw> : i64
    %106 = llvm.icmp "eq" %105, %43 : i64
    llvm.cond_br %106, ^bb10, ^bb3(%105 : i64) {loop_annotation = #loop_annotation}
  ^bb10:  // pred: ^bb9
    %107 = llvm.shl %37, %10 : i32
    %108 = llvm.lshr %36, %10 : i32
    %109 = llvm.add %38, %10 overflow<nsw, nuw> : i32
    %110 = llvm.icmp "eq" %109, %19 : i32
    llvm.cond_br %110, ^bb11, ^bb2(%108, %107, %109 : i32, i32, i32) {loop_annotation = #loop_annotation}
  ^bb11:  // pred: ^bb10
    llvm.return %11 : i32
  }
  llvm.func local_unnamed_addr @_Z6kernelPfS_S_S_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(128 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(-1 : i32) : i32
    %4 = llvm.mlir.constant(3 : i64) : i64
    %5 = llvm.mlir.constant(-4 : i64) : i64
    %6 = llvm.mlir.constant(2 : i64) : i64
    %7 = llvm.mlir.constant(4 : i32) : i32
    %8 = llvm.mlir.constant(252 : i64) : i64
    %9 = llvm.mlir.constant(0 : i64) : i64
    %10 = llvm.mlir.constant(1 : i64) : i64
    %11 = llvm.mlir.constant(true) : i1
    %12 = llvm.mlir.poison : vector<4xf32>
    %13 = llvm.mlir.constant(4 : i64) : i64
    %14 = llvm.mlir.constant(8 : i32) : i32
    llvm.br ^bb1(%0, %1, %2 : i32, i32, i32)
  ^bb1(%15: i32, %16: i32, %17: i32):  // 2 preds: ^bb0, ^bb9
    %18 = llvm.shl %3, %17 overflow<nsw> : i32
    %19 = llvm.xor %18, %3 : i32
    %20 = llvm.zext nneg %15 : i32 to i64
    %21 = llvm.zext nneg %19 : i32 to i64
    %22 = llvm.zext %16 : i32 to i64
    %23 = llvm.shl %22, %4 overflow<nsw, nuw> : i64
    %24 = llvm.add %23, %5 overflow<nsw> : i64
    %25 = llvm.mul %24, %20 overflow<nsw> : i64
    %26 = llvm.getelementptr %arg0[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %27 = llvm.getelementptr %arg1[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %28 = llvm.shl %20, %6 overflow<nsw, nuw> : i64
    %29 = llvm.getelementptr %arg1[%28] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %30 = llvm.shl %22, %4 overflow<nsw, nuw> : i64
    %31 = llvm.mul %30, %20 overflow<nsw, nuw> : i64
    %32 = llvm.getelementptr %arg1[%31] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %33 = llvm.getelementptr %arg0[%28] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %34 = llvm.getelementptr %arg0[%31] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %35 = llvm.icmp "ult" %15, %7 : i32
    %36 = llvm.icmp "ult" %arg0, %27 : !llvm.ptr
    %37 = llvm.icmp "ult" %arg1, %26 : !llvm.ptr
    %38 = llvm.and %36, %37 : i1
    %39 = llvm.icmp "ult" %arg0, %32 : !llvm.ptr
    %40 = llvm.icmp "ult" %29, %26 : !llvm.ptr
    %41 = llvm.and %39, %40 : i1
    %42 = llvm.or %38, %41 : i1
    %43 = llvm.icmp "ult" %33, %27 : !llvm.ptr
    %44 = llvm.icmp "ult" %arg1, %34 : !llvm.ptr
    %45 = llvm.and %43, %44 : i1
    %46 = llvm.or %42, %45 : i1
    %47 = llvm.icmp "ult" %33, %32 : !llvm.ptr
    %48 = llvm.icmp "ult" %29, %34 : !llvm.ptr
    %49 = llvm.and %47, %48 : i1
    %50 = llvm.or %46, %49 : i1
    %51 = llvm.and %20, %8 : i64
    %52 = llvm.icmp "eq" %51, %20 : i64
    llvm.br ^bb2(%9 : i64)
  ^bb2(%53: i64):  // 2 preds: ^bb1, ^bb8
    %54 = llvm.add %53, %21 overflow<nsw, nuw> : i64
    %55 = llvm.getelementptr inbounds %arg2[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %56 = llvm.load %55 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %57 = llvm.getelementptr inbounds %arg3[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %58 = llvm.load %57 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %59 = llvm.shl %53, %10 overflow<nsw, nuw> : i64
    %60 = llvm.mul %59, %20 overflow<nsw, nuw> : i64
    %61 = llvm.add %60, %20 overflow<nsw, nuw> : i64
    %62 = llvm.select %35, %11, %50 : i1, i1
    llvm.cond_br %62, ^bb6(%9 : i64), ^bb3
  ^bb3:  // pred: ^bb2
    %63 = llvm.insertelement %58, %12[%9 : i64] : vector<4xf32>
    %64 = llvm.shufflevector %63, %12 [0, 0, 0, 0] : vector<4xf32> 
    %65 = llvm.insertelement %56, %12[%9 : i64] : vector<4xf32>
    %66 = llvm.shufflevector %65, %12 [0, 0, 0, 0] : vector<4xf32> 
    llvm.br ^bb4(%9 : i64)
  ^bb4(%67: i64):  // 2 preds: ^bb3, ^bb4
    %68 = llvm.add %61, %67 overflow<nsw, nuw> : i64
    %69 = llvm.getelementptr inbounds %arg0[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %70 = llvm.load %69 {alias_scopes = [#alias_scope], alignment = 4 : i64, noalias_scopes = [#alias_scope1, #alias_scope2], tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xf32>
    %71 = llvm.getelementptr inbounds %arg1[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %72 = llvm.load %71 {alias_scopes = [#alias_scope2], alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xf32>
    %73 = llvm.fneg %72 : vector<4xf32>
    %74 = llvm.fmul %64, %73 : vector<4xf32>
    %75 = llvm.intr.fmuladd(%66, %70, %74) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %76 = llvm.fmul %66, %72 : vector<4xf32>
    %77 = llvm.intr.fmuladd(%64, %70, %76) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %78 = llvm.add %67, %60 overflow<nsw, nuw> : i64
    %79 = llvm.getelementptr inbounds %arg0[%78] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %80 = llvm.load %79 {alias_scopes = [#alias_scope3], alignment = 4 : i64, noalias_scopes = [#alias_scope1, #alias_scope2], tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xf32>
    %81 = llvm.fsub %80, %75 : vector<4xf32>
    llvm.store %81, %69 {alias_scopes = [#alias_scope], alignment = 4 : i64, noalias_scopes = [#alias_scope1, #alias_scope2], tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %82 = llvm.fadd %80, %75 : vector<4xf32>
    llvm.store %82, %79 {alias_scopes = [#alias_scope3], alignment = 4 : i64, noalias_scopes = [#alias_scope1, #alias_scope2], tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %83 = llvm.getelementptr inbounds %arg1[%78] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %84 = llvm.load %83 {alias_scopes = [#alias_scope1], alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xf32>
    %85 = llvm.fsub %84, %77 : vector<4xf32>
    llvm.store %85, %71 {alias_scopes = [#alias_scope2], alignment = 4 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %86 = llvm.fadd %77, %84 : vector<4xf32>
    llvm.store %86, %83 {alias_scopes = [#alias_scope1], alignment = 4 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %87 = llvm.add %67, %13 overflow<nuw> : i64
    %88 = llvm.icmp "eq" %87, %51 : i64
    llvm.cond_br %88, ^bb5, ^bb4(%87 : i64)
  ^bb5:  // pred: ^bb4
    llvm.cond_br %52, ^bb8, ^bb6(%51 : i64)
  ^bb6(%89: i64):  // 2 preds: ^bb2, ^bb5
    llvm.br ^bb7(%89 : i64)
  ^bb7(%90: i64):  // 2 preds: ^bb6, ^bb7
    %91 = llvm.add %61, %90 overflow<nsw, nuw> : i64
    %92 = llvm.getelementptr inbounds %arg0[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %93 = llvm.load %92 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %94 = llvm.getelementptr inbounds %arg1[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %95 = llvm.load %94 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %96 = llvm.fneg %95 : f32
    %97 = llvm.fmul %58, %96 : f32
    %98 = llvm.intr.fmuladd(%56, %93, %97) : (f32, f32, f32) -> f32
    %99 = llvm.fmul %56, %95 : f32
    %100 = llvm.intr.fmuladd(%58, %93, %99) : (f32, f32, f32) -> f32
    %101 = llvm.add %90, %60 overflow<nsw, nuw> : i64
    %102 = llvm.getelementptr inbounds %arg0[%101] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %103 = llvm.load %102 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %104 = llvm.fsub %103, %98 : f32
    llvm.store %104, %92 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %105 = llvm.fadd %103, %98 : f32
    llvm.store %105, %102 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %106 = llvm.getelementptr inbounds %arg1[%101] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %107 = llvm.load %106 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %108 = llvm.fsub %107, %100 : f32
    llvm.store %108, %94 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %109 = llvm.fadd %100, %107 : f32
    llvm.store %109, %106 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %110 = llvm.add %90, %10 overflow<nsw, nuw> : i64
    %111 = llvm.icmp "eq" %110, %20 : i64
    llvm.cond_br %111, ^bb8, ^bb7(%110 : i64)
  ^bb8:  // 2 preds: ^bb5, ^bb7
    %112 = llvm.add %53, %10 overflow<nsw, nuw> : i64
    %113 = llvm.icmp "eq" %112, %22 : i64
    llvm.cond_br %113, ^bb9, ^bb2(%112 : i64) {loop_annotation = #loop_annotation}
  ^bb9:  // pred: ^bb8
    %114 = llvm.shl %16, %1 : i32
    %115 = llvm.lshr %15, %1 : i32
    %116 = llvm.add %17, %1 overflow<nsw, nuw> : i32
    %117 = llvm.icmp "eq" %116, %14 : i32
    llvm.cond_br %117, ^bb10, ^bb1(%115, %114, %116 : i32, i32, i32) {loop_annotation = #loop_annotation}
  ^bb10:  // pred: ^bb9
    llvm.return
  }
}
