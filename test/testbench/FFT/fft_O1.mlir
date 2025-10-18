#alias_scope_domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "LVerDomain">
#loop_unroll = #llvm.loop_unroll<disable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#alias_scope = #llvm.alias_scope<id = distinct[1]<>, domain = #alias_scope_domain>
#alias_scope1 = #llvm.alias_scope<id = distinct[2]<>, domain = #alias_scope_domain>
#alias_scope2 = #llvm.alias_scope<id = distinct[3]<>, domain = #alias_scope_domain>
#alias_scope3 = #llvm.alias_scope<id = distinct[4]<>, domain = #alias_scope_domain>
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "float", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f64 = dense<64> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, "dlti.stack_alignment" = 128 : i64, "dlti.endianness" = "little">, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.mlir.global external local_unnamed_addr @data_real(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.mlir.global external local_unnamed_addr @data_imag(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.mlir.global external local_unnamed_addr @coef_real(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.mlir.global external local_unnamed_addr @coef_imag(dense<0.000000e+00> : tensor<256xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<256 x f32>
  llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {memory_effects = #llvm.memory_effects<other = readwrite, argMem = none, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.addressof @data_imag : !llvm.ptr
    %2 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %3 = llvm.mlir.addressof @coef_real : !llvm.ptr
    %4 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %5 = llvm.mlir.addressof @coef_imag : !llvm.ptr
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.mlir.constant(256 : i64) : i64
    %8 = llvm.mlir.constant(128 : i32) : i32
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.mlir.constant(0 : i32) : i32
    %11 = llvm.mlir.constant(-1 : i32) : i32
    %12 = llvm.mlir.constant(4 : i32) : i32
    %13 = llvm.mlir.constant(252 : i64) : i64
    %14 = llvm.mlir.poison : vector<4xf32>
    %15 = llvm.mlir.addressof @data_real : !llvm.ptr
    %16 = llvm.mlir.constant(4 : i64) : i64
    %17 = llvm.mlir.constant(8 : i32) : i32
    llvm.br ^bb1(%0 : i64)
  ^bb1(%18: i64):  // 2 preds: ^bb0, ^bb1
    %19 = llvm.getelementptr inbounds %1[%0, %18] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
    llvm.store %2, %19 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %20 = llvm.getelementptr inbounds %3[%0, %18] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
    llvm.store %4, %20 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %21 = llvm.getelementptr inbounds %5[%0, %18] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
    llvm.store %4, %21 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %22 = llvm.add %18, %6 overflow<nsw, nuw> : i64
    %23 = llvm.icmp "eq" %22, %7 : i64
    llvm.cond_br %23, ^bb2(%8, %9, %10 : i32, i32, i32), ^bb1(%22 : i64) {loop_annotation = #loop_annotation}
  ^bb2(%24: i32, %25: i32, %26: i32):  // 2 preds: ^bb1, ^bb10
    %27 = llvm.shl %11, %26 overflow<nsw> : i32
    %28 = llvm.xor %27, %11 : i32
    %29 = llvm.zext nneg %24 : i32 to i64
    %30 = llvm.zext nneg %28 : i32 to i64
    %31 = llvm.zext %25 : i32 to i64
    %32 = llvm.icmp "ult" %24, %12 : i32
    %33 = llvm.and %29, %13 : i64
    %34 = llvm.icmp "eq" %33, %29 : i64
    llvm.br ^bb3(%0 : i64)
  ^bb3(%35: i64):  // 2 preds: ^bb2, ^bb9
    %36 = llvm.add %35, %30 overflow<nsw, nuw> : i64
    %37 = llvm.getelementptr inbounds %3[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %38 = llvm.load %37 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %39 = llvm.getelementptr inbounds %5[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %40 = llvm.load %39 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %41 = llvm.shl %35, %6 overflow<nsw, nuw> : i64
    %42 = llvm.mul %41, %29 overflow<nsw, nuw> : i64
    %43 = llvm.add %42, %29 overflow<nsw, nuw> : i64
    llvm.cond_br %32, ^bb7(%0 : i64), ^bb4
  ^bb4:  // pred: ^bb3
    %44 = llvm.insertelement %40, %14[%0 : i64] : vector<4xf32>
    %45 = llvm.shufflevector %44, %14 [0, 0, 0, 0] : vector<4xf32> 
    %46 = llvm.insertelement %38, %14[%0 : i64] : vector<4xf32>
    %47 = llvm.shufflevector %46, %14 [0, 0, 0, 0] : vector<4xf32> 
    llvm.br ^bb5(%0 : i64)
  ^bb5(%48: i64):  // 2 preds: ^bb4, ^bb5
    %49 = llvm.add %43, %48 overflow<nsw, nuw> : i64
    %50 = llvm.getelementptr inbounds %15[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %51 = llvm.load %50 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xf32>
    %52 = llvm.getelementptr inbounds %1[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %53 = llvm.load %52 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xf32>
    %54 = llvm.fneg %53 : vector<4xf32>
    %55 = llvm.fmul %45, %54 : vector<4xf32>
    %56 = llvm.intr.fmuladd(%47, %51, %55) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %57 = llvm.fmul %47, %53 : vector<4xf32>
    %58 = llvm.intr.fmuladd(%45, %51, %57) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %59 = llvm.add %48, %42 overflow<nsw, nuw> : i64
    %60 = llvm.getelementptr inbounds %15[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %61 = llvm.load %60 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xf32>
    %62 = llvm.fsub %61, %56 : vector<4xf32>
    llvm.store %62, %50 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %63 = llvm.fadd %61, %56 : vector<4xf32>
    llvm.store %63, %60 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %64 = llvm.getelementptr inbounds %1[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %65 = llvm.load %64 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xf32>
    %66 = llvm.fsub %65, %58 : vector<4xf32>
    llvm.store %66, %52 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %67 = llvm.fadd %58, %65 : vector<4xf32>
    llvm.store %67, %64 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %68 = llvm.add %48, %16 overflow<nuw> : i64
    %69 = llvm.icmp "eq" %68, %33 : i64
    llvm.cond_br %69, ^bb6, ^bb5(%68 : i64)
  ^bb6:  // pred: ^bb5
    llvm.cond_br %34, ^bb9, ^bb7(%33 : i64)
  ^bb7(%70: i64):  // 2 preds: ^bb3, ^bb6
    llvm.br ^bb8(%70 : i64)
  ^bb8(%71: i64):  // 2 preds: ^bb7, ^bb8
    %72 = llvm.add %43, %71 overflow<nsw, nuw> : i64
    %73 = llvm.getelementptr inbounds %15[%72] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %74 = llvm.load %73 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %75 = llvm.getelementptr inbounds %1[%72] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %76 = llvm.load %75 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %77 = llvm.fneg %76 : f32
    %78 = llvm.fmul %40, %77 : f32
    %79 = llvm.intr.fmuladd(%38, %74, %78) : (f32, f32, f32) -> f32
    %80 = llvm.fmul %38, %76 : f32
    %81 = llvm.intr.fmuladd(%40, %74, %80) : (f32, f32, f32) -> f32
    %82 = llvm.add %71, %42 overflow<nsw, nuw> : i64
    %83 = llvm.getelementptr inbounds %15[%82] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %84 = llvm.load %83 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %85 = llvm.fsub %84, %79 : f32
    llvm.store %85, %73 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %86 = llvm.fadd %84, %79 : f32
    llvm.store %86, %83 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %87 = llvm.getelementptr inbounds %1[%82] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %88 = llvm.load %87 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %89 = llvm.fsub %88, %81 : f32
    llvm.store %89, %75 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %90 = llvm.fadd %81, %88 : f32
    llvm.store %90, %87 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %91 = llvm.add %71, %6 overflow<nsw, nuw> : i64
    %92 = llvm.icmp "eq" %91, %29 : i64
    llvm.cond_br %92, ^bb9, ^bb8(%91 : i64)
  ^bb9:  // 2 preds: ^bb6, ^bb8
    %93 = llvm.add %35, %6 overflow<nsw, nuw> : i64
    %94 = llvm.icmp "eq" %93, %31 : i64
    llvm.cond_br %94, ^bb10, ^bb3(%93 : i64) {loop_annotation = #loop_annotation}
  ^bb10:  // pred: ^bb9
    %95 = llvm.shl %25, %9 : i32
    %96 = llvm.lshr %24, %9 : i32
    %97 = llvm.add %26, %9 overflow<nsw, nuw> : i32
    %98 = llvm.icmp "eq" %97, %17 : i32
    llvm.cond_br %98, ^bb11, ^bb2(%96, %95, %97 : i32, i32, i32) {loop_annotation = #loop_annotation}
  ^bb11:  // pred: ^bb10
    llvm.return %10 : i32
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
    %21 = llvm.zext nneg %15 : i32 to i64
    %22 = llvm.zext nneg %19 : i32 to i64
    %23 = llvm.zext %16 : i32 to i64
    %24 = llvm.shl %23, %4 overflow<nsw, nuw> : i64
    %25 = llvm.add %24, %5 overflow<nsw> : i64
    %26 = llvm.mul %25, %20 overflow<nsw> : i64
    %27 = llvm.getelementptr %arg0[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %28 = llvm.getelementptr %arg1[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %29 = llvm.shl %20, %6 overflow<nsw, nuw> : i64
    %30 = llvm.getelementptr %arg1[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %31 = llvm.shl %23, %4 overflow<nsw, nuw> : i64
    %32 = llvm.mul %31, %20 overflow<nsw, nuw> : i64
    %33 = llvm.getelementptr %arg1[%32] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %34 = llvm.getelementptr %arg0[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %35 = llvm.getelementptr %arg0[%32] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %36 = llvm.zext nneg %15 : i32 to i64
    %37 = llvm.icmp "ult" %15, %7 : i32
    %38 = llvm.icmp "ult" %arg0, %28 : !llvm.ptr
    %39 = llvm.icmp "ult" %arg1, %27 : !llvm.ptr
    %40 = llvm.and %38, %39 : i1
    %41 = llvm.icmp "ult" %arg0, %33 : !llvm.ptr
    %42 = llvm.icmp "ult" %30, %27 : !llvm.ptr
    %43 = llvm.and %41, %42 : i1
    %44 = llvm.or %40, %43 : i1
    %45 = llvm.icmp "ult" %34, %28 : !llvm.ptr
    %46 = llvm.icmp "ult" %arg1, %35 : !llvm.ptr
    %47 = llvm.and %45, %46 : i1
    %48 = llvm.or %44, %47 : i1
    %49 = llvm.icmp "ult" %34, %33 : !llvm.ptr
    %50 = llvm.icmp "ult" %30, %35 : !llvm.ptr
    %51 = llvm.and %49, %50 : i1
    %52 = llvm.or %48, %51 : i1
    %53 = llvm.and %20, %8 : i64
    %54 = llvm.icmp "eq" %53, %20 : i64
    llvm.br ^bb2(%9 : i64)
  ^bb2(%55: i64):  // 2 preds: ^bb1, ^bb8
    %56 = llvm.add %55, %22 overflow<nsw, nuw> : i64
    %57 = llvm.getelementptr inbounds %arg2[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %58 = llvm.load %57 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %59 = llvm.getelementptr inbounds %arg3[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %60 = llvm.load %59 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %61 = llvm.shl %55, %10 overflow<nsw, nuw> : i64
    %62 = llvm.mul %61, %20 overflow<nsw, nuw> : i64
    %63 = llvm.add %62, %21 overflow<nsw, nuw> : i64
    %64 = llvm.select %37, %11, %52 : i1, i1
    llvm.cond_br %64, ^bb6(%9 : i64), ^bb3
  ^bb3:  // pred: ^bb2
    %65 = llvm.insertelement %60, %12[%9 : i64] : vector<4xf32>
    %66 = llvm.shufflevector %65, %12 [0, 0, 0, 0] : vector<4xf32> 
    %67 = llvm.insertelement %58, %12[%9 : i64] : vector<4xf32>
    %68 = llvm.shufflevector %67, %12 [0, 0, 0, 0] : vector<4xf32> 
    llvm.br ^bb4(%9 : i64)
  ^bb4(%69: i64):  // 2 preds: ^bb3, ^bb4
    %70 = llvm.add %63, %69 overflow<nsw, nuw> : i64
    %71 = llvm.getelementptr inbounds %arg0[%70] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %72 = llvm.load %71 {alias_scopes = [#alias_scope], alignment = 4 : i64, noalias_scopes = [#alias_scope1, #alias_scope2], tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xf32>
    %73 = llvm.getelementptr inbounds %arg1[%70] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %74 = llvm.load %73 {alias_scopes = [#alias_scope2], alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xf32>
    %75 = llvm.fneg %74 : vector<4xf32>
    %76 = llvm.fmul %66, %75 : vector<4xf32>
    %77 = llvm.intr.fmuladd(%68, %72, %76) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %78 = llvm.fmul %68, %74 : vector<4xf32>
    %79 = llvm.intr.fmuladd(%66, %72, %78) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %80 = llvm.add %69, %62 overflow<nsw, nuw> : i64
    %81 = llvm.getelementptr inbounds %arg0[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %82 = llvm.load %81 {alias_scopes = [#alias_scope3], alignment = 4 : i64, noalias_scopes = [#alias_scope1, #alias_scope2], tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xf32>
    %83 = llvm.fsub %82, %77 : vector<4xf32>
    llvm.store %83, %71 {alias_scopes = [#alias_scope], alignment = 4 : i64, noalias_scopes = [#alias_scope1, #alias_scope2], tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %84 = llvm.fadd %82, %77 : vector<4xf32>
    llvm.store %84, %81 {alias_scopes = [#alias_scope3], alignment = 4 : i64, noalias_scopes = [#alias_scope1, #alias_scope2], tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %85 = llvm.getelementptr inbounds %arg1[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %86 = llvm.load %85 {alias_scopes = [#alias_scope1], alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xf32>
    %87 = llvm.fsub %86, %79 : vector<4xf32>
    llvm.store %87, %73 {alias_scopes = [#alias_scope2], alignment = 4 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %88 = llvm.fadd %79, %86 : vector<4xf32>
    llvm.store %88, %85 {alias_scopes = [#alias_scope1], alignment = 4 : i64, tbaa = [#tbaa_tag]} : vector<4xf32>, !llvm.ptr
    %89 = llvm.add %69, %13 overflow<nuw> : i64
    %90 = llvm.icmp "eq" %89, %53 : i64
    llvm.cond_br %90, ^bb5, ^bb4(%89 : i64)
  ^bb5:  // pred: ^bb4
    llvm.cond_br %54, ^bb8, ^bb6(%53 : i64)
  ^bb6(%91: i64):  // 2 preds: ^bb2, ^bb5
    llvm.br ^bb7(%91 : i64)
  ^bb7(%92: i64):  // 2 preds: ^bb6, ^bb7
    %93 = llvm.add %63, %92 overflow<nsw, nuw> : i64
    %94 = llvm.getelementptr inbounds %arg0[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %95 = llvm.load %94 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %96 = llvm.getelementptr inbounds %arg1[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %97 = llvm.load %96 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %98 = llvm.fneg %97 : f32
    %99 = llvm.fmul %60, %98 : f32
    %100 = llvm.intr.fmuladd(%58, %95, %99) : (f32, f32, f32) -> f32
    %101 = llvm.fmul %58, %97 : f32
    %102 = llvm.intr.fmuladd(%60, %95, %101) : (f32, f32, f32) -> f32
    %103 = llvm.add %92, %62 overflow<nsw, nuw> : i64
    %104 = llvm.getelementptr inbounds %arg0[%103] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %105 = llvm.load %104 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %106 = llvm.fsub %105, %100 : f32
    llvm.store %106, %94 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %107 = llvm.fadd %105, %100 : f32
    llvm.store %107, %104 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %108 = llvm.getelementptr inbounds %arg1[%103] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %109 = llvm.load %108 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %110 = llvm.fsub %109, %102 : f32
    llvm.store %110, %96 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %111 = llvm.fadd %102, %109 : f32
    llvm.store %111, %108 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    %112 = llvm.add %92, %10 overflow<nsw, nuw> : i64
    %113 = llvm.icmp "eq" %112, %36 : i64
    llvm.cond_br %113, ^bb8, ^bb7(%112 : i64)
  ^bb8:  // 2 preds: ^bb5, ^bb7
    %114 = llvm.add %55, %10 overflow<nsw, nuw> : i64
    %115 = llvm.icmp "eq" %114, %23 : i64
    llvm.cond_br %115, ^bb9, ^bb2(%114 : i64) {loop_annotation = #loop_annotation}
  ^bb9:  // pred: ^bb8
    %116 = llvm.shl %16, %1 : i32
    %117 = llvm.lshr %15, %1 : i32
    %118 = llvm.add %17, %1 overflow<nsw, nuw> : i32
    %119 = llvm.icmp "eq" %118, %14 : i32
    llvm.cond_br %119, ^bb10, ^bb1(%117, %116, %118 : i32, i32, i32) {loop_annotation = #loop_annotation}
  ^bb10:  // pred: ^bb9
    llvm.return
  }
}
