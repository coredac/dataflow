#loop_unroll = #llvm.loop_unroll<disable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "double", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, f80 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, f16 = dense<16> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f64 = dense<64> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, "dlti.stack_alignment" = 128 : i64, "dlti.endianness" = "little">, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.func local_unnamed_addr @kernel(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg4: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg5: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg6: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(3 : i64) : i64
    %2 = llvm.mlir.constant(0 : i8) : i8
    %3 = llvm.mlir.constant(0 : i64) : i64
    %4 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.icmp "sgt" %arg0, %0 : i32
    llvm.cond_br %6, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %7 = llvm.zext nneg %arg0 : i32 to i64
    %8 = llvm.shl %7, %1 overflow<nsw, nuw> : i64
    "llvm.intr.memset"(%arg3, %2, %8) <{isVolatile = false, tbaa = [#tbaa_tag]}> : (!llvm.ptr, i8, i64) -> ()
    %9 = llvm.icmp "sgt" %arg1, %0 : i32
    llvm.cond_br %9, ^bb4, ^bb8
  ^bb2:  // pred: ^bb0
    %10 = llvm.icmp "sgt" %arg1, %0 : i32
    llvm.cond_br %10, ^bb3, ^bb8
  ^bb3:  // pred: ^bb2
    %11 = llvm.zext nneg %arg1 : i32 to i64
    %12 = llvm.shl %11, %1 overflow<nsw, nuw> : i64
    "llvm.intr.memset"(%arg4, %2, %12) <{isVolatile = false, tbaa = [#tbaa_tag]}> : (!llvm.ptr, i8, i64) -> ()
    llvm.br ^bb8
  ^bb4:  // pred: ^bb1
    %13 = llvm.zext nneg %arg1 : i32 to i64
    %14 = llvm.zext nneg %arg0 : i32 to i64
    llvm.br ^bb5(%3 : i64)
  ^bb5(%15: i64):  // 2 preds: ^bb4, ^bb7
    %16 = llvm.getelementptr inbounds %arg4[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %4, %16 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
    %17 = llvm.getelementptr inbounds %arg6[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.br ^bb6(%3 : i64)
  ^bb6(%18: i64):  // 2 preds: ^bb5, ^bb6
    %19 = llvm.getelementptr inbounds %arg3[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %20 = llvm.load %19 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
    %21 = llvm.load %17 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
    %22 = llvm.getelementptr inbounds %arg2[%15, %18] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<116 x f64>
    %23 = llvm.load %22 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
    %24 = llvm.intr.fmuladd(%21, %23, %20) : (f64, f64, f64) -> f64
    llvm.store %24, %19 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
    %25 = llvm.load %16 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
    %26 = llvm.load %22 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
    %27 = llvm.getelementptr inbounds %arg5[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %28 = llvm.load %27 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
    %29 = llvm.intr.fmuladd(%26, %28, %25) : (f64, f64, f64) -> f64
    llvm.store %29, %16 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
    %30 = llvm.add %18, %5 overflow<nsw, nuw> : i64
    %31 = llvm.icmp "eq" %30, %14 : i64
    llvm.cond_br %31, ^bb7, ^bb6(%30 : i64) {loop_annotation = #loop_annotation}
  ^bb7:  // pred: ^bb6
    %32 = llvm.add %15, %5 overflow<nsw, nuw> : i64
    %33 = llvm.icmp "eq" %32, %13 : i64
    llvm.cond_br %33, ^bb8, ^bb5(%32 : i64) {loop_annotation = #loop_annotation}
  ^bb8:  // 4 preds: ^bb1, ^bb2, ^bb3, ^bb7
    llvm.return
  }
}
