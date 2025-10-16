#loop_unroll = #llvm.loop_unroll<disable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "float", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<i64 = dense<64> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, i32 = dense<32> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, f80 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.func local_unnamed_addr @_Z6kernelPfS_S_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {approx_func_fp_math = true, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], "no-builtin-fma", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87", "-amx-avx512", "-avx10.1-256", "-avx10.1-512", "-avx10.2-256", "-avx10.2-512", "-avx512bf16", "-avx512bitalg", "-avx512bw", "-avx512cd", "-avx512dq", "-avx512f", "-avx512fp16", "-avx512ifma", "-avx512vbmi", "-avx512vbmi2", "-avx512vl", "-avx512vnni", "-avx512vp2intersect", "-avx512vpopcntdq", "-fma"]>, tune_cpu = "generic", unsafe_fp_math = true} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.mlir.constant(32 : i64) : i64
    llvm.br ^bb1(%0, %1 : i64, f32)
  ^bb1(%4: i64, %5: f32):  // 2 preds: ^bb0, ^bb1
    %6 = llvm.getelementptr inbounds %arg0[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %7 = llvm.load %6 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %8 = llvm.getelementptr inbounds %arg2[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %9 = llvm.load %8 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
    %10 = llvm.fmul %9, %7 {fastmathFlags = #llvm.fastmath<fast>} : f32
    %11 = llvm.fadd %10, %5 {fastmathFlags = #llvm.fastmath<fast>} : f32
    %12 = llvm.add %4, %2 overflow<nsw, nuw> : i64
    %13 = llvm.icmp "eq" %12, %3 : i64
    llvm.cond_br %13, ^bb2, ^bb1(%12, %11 : i64, f32) {loop_annotation = #loop_annotation}
  ^bb2:  // pred: ^bb1
    llvm.store %11, %arg1 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
    llvm.return
  }
}
