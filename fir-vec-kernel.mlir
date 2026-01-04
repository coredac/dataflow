#loop_unroll = #llvm.loop_unroll<disable = true, runtimeDisable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true, isVectorized = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, "dlti.stack_alignment" = 128 : i64, "dlti.endianness" = "little">, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.func local_unnamed_addr @_Z6kernelPiS_S_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", will_return} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(dense<0> : vector<4xi32>) : vector<4xi32>
    %3 = llvm.mlir.constant(4 : i64) : i64
    %4 = llvm.mlir.constant(32 : i64) : i64
    llvm.br ^bb1(%0, %2 : i64, vector<4xi32>)
  ^bb1(%5: i64, %6: vector<4xi32>):  // 2 preds: ^bb0, ^bb1
    %7 = llvm.getelementptr inbounds %arg0[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %8 = llvm.load %7 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xi32>
    %9 = llvm.getelementptr inbounds %arg2[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %10 = llvm.load %9 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> vector<4xi32>
    %11 = llvm.mul %10, %8 overflow<nsw> : vector<4xi32>
    %12 = llvm.add %11, %6 : vector<4xi32>
    %13 = llvm.add %5, %3 overflow<nuw> : i64
    %14 = llvm.icmp "eq" %13, %4 : i64
    llvm.cond_br %14, ^bb2, ^bb1(%13, %12 : i64, vector<4xi32>) {loop_annotation = #loop_annotation}
  ^bb2:  // pred: ^bb1
    %15 = "llvm.intr.vector.reduce.add"(%12) : (vector<4xi32>) -> i32
    llvm.return %15 : i32
  }
}
