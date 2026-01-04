#loop_vectorize = #llvm.loop_vectorize<width = 1 : i32>
#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#loop_annotation = #llvm.loop_annotation<vectorize = #loop_vectorize, mustProgress = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, f64 = dense<64> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f128 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.func local_unnamed_addr @kernel(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg4: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(70 : i16) : i16
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(4200 : i32) : i32
    llvm.br ^bb1(%0 : i32)
  ^bb1(%4: i32):  // 2 preds: ^bb0, ^bb1
    %5 = llvm.trunc %4 overflow<nuw> : i32 to i16
    %6 = llvm.udiv %5, %1 : i16
    %7 = llvm.urem %5, %1 : i16
    %8 = llvm.zext nneg %6 : i16 to i64
    %9 = llvm.zext nneg %7 : i16 to i64
    %10 = llvm.getelementptr inbounds %arg4[%8, %9] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<70 x i32>
    %11 = llvm.load %10 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %12 = llvm.intr.smax(%11, %0) : (i32, i32) -> i32
    %13 = llvm.getelementptr inbounds %arg3[%8, %9] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<70 x i32>
    llvm.store %12, %13 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %14 = llvm.add %4, %2 overflow<nsw, nuw> : i32
    %15 = llvm.icmp "eq" %14, %3 : i32
    llvm.cond_br %15, ^bb2, ^bb1(%14 : i32) {loop_annotation = #loop_annotation}
  ^bb2:  // pred: ^bb1
    llvm.return
  }
}
