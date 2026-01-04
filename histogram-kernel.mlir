#loop_unroll = #llvm.loop_unroll<disable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, f16 = dense<16> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, i128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.func local_unnamed_addr @_Z6kernelPiS_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(5 : i32) : i32
    %2 = llvm.mlir.constant(-5 : i32) : i32
    %3 = llvm.mlir.constant(18 : i32) : i32
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.mlir.constant(20 : i64) : i64
    llvm.br ^bb1(%0 : i64)
  ^bb1(%7: i64):  // 2 preds: ^bb0, ^bb1
    %8 = llvm.getelementptr inbounds %arg0[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %9 = llvm.load %8 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %10 = llvm.mul %9, %1 : i32
    %11 = llvm.add %10, %2 : i32
    %12 = llvm.sdiv %11, %3 : i32
    %13 = llvm.sext %12 : i32 to i64
    %14 = llvm.getelementptr inbounds %arg1[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %15 = llvm.load %14 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %16 = llvm.add %15, %4 overflow<nsw> : i32
    llvm.store %16, %14 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %17 = llvm.add %7, %5 overflow<nsw, nuw> : i64
    %18 = llvm.icmp "eq" %17, %6 : i64
    llvm.cond_br %18, ^bb2, ^bb1(%17 : i64) {loop_annotation = #loop_annotation}
  ^bb2:  // pred: ^bb1
    llvm.return
  }
}
