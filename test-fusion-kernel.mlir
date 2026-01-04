#loop_unroll = #llvm.loop_unroll<disable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.mlir.global external local_unnamed_addr @A(dense<0> : tensor<1024x1024xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<1024 x array<1024 x i32>>
  llvm.mlir.global external local_unnamed_addr @s(dense<0> : tensor<1024xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<1024 x i32>
  llvm.mlir.global external local_unnamed_addr @q(dense<0> : tensor<1024xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<1024 x i32>
  llvm.mlir.global external local_unnamed_addr @p(dense<0> : tensor<1024xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<1024 x i32>
  llvm.mlir.global external local_unnamed_addr @r(dense<0> : tensor<1024xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<1024 x i32>
  llvm.func local_unnamed_addr @_Z6kernelPA1024_iPiS1_S1_S1_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(1024 : i64) : i64
    llvm.br ^bb1(%0 : i64)
  ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb3
    %4 = llvm.getelementptr inbounds %arg4[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %5 = llvm.getelementptr inbounds %arg2[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.br ^bb2(%0 : i64)
  ^bb2(%6: i64):  // 2 preds: ^bb1, ^bb2
    %7 = llvm.getelementptr inbounds %arg1[%6] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %8 = llvm.load %7 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %9 = llvm.load %4 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %10 = llvm.getelementptr inbounds %arg0[%3, %6] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<1024 x i32>
    %11 = llvm.load %10 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %12 = llvm.mul %11, %9 overflow<nsw> : i32
    %13 = llvm.add %12, %8 overflow<nsw> : i32
    llvm.store %13, %7 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %14 = llvm.load %5 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %15 = llvm.load %10 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %16 = llvm.getelementptr inbounds %arg3[%6] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %17 = llvm.load %16 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %18 = llvm.mul %17, %15 overflow<nsw> : i32
    %19 = llvm.add %18, %14 overflow<nsw> : i32
    llvm.store %19, %5 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %20 = llvm.add %6, %1 overflow<nsw, nuw> : i64
    %21 = llvm.icmp "eq" %20, %2 : i64
    llvm.cond_br %21, ^bb3, ^bb2(%20 : i64) {loop_annotation = #loop_annotation}
  ^bb3:  // pred: ^bb2
    %22 = llvm.add %3, %1 overflow<nsw, nuw> : i64
    %23 = llvm.icmp "eq" %22, %2 : i64
    llvm.cond_br %23, ^bb4, ^bb1(%22 : i64) {loop_annotation = #loop_annotation}
  ^bb4:  // pred: ^bb3
    llvm.return
  }
  llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {memory_effects = #llvm.memory_effects<other = readwrite, argMem = none, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.addressof @r : !llvm.ptr
    %2 = llvm.mlir.addressof @q : !llvm.ptr
    %3 = llvm.mlir.addressof @s : !llvm.ptr
    %4 = llvm.mlir.addressof @A : !llvm.ptr
    %5 = llvm.mlir.addressof @p : !llvm.ptr
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.mlir.constant(1024 : i64) : i64
    %8 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb1(%0 : i64)
  ^bb1(%9: i64):  // 2 preds: ^bb0, ^bb3
    %10 = llvm.getelementptr inbounds %1[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %11 = llvm.getelementptr inbounds %2[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %12 = llvm.load %10 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %13 = llvm.load %11 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    llvm.br ^bb2(%13, %0 : i32, i64)
  ^bb2(%14: i32, %15: i64):  // 2 preds: ^bb1, ^bb2
    %16 = llvm.getelementptr inbounds %3[%15] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %17 = llvm.load %16 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %18 = llvm.getelementptr inbounds %4[%9, %15] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<1024 x i32>
    %19 = llvm.load %18 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %20 = llvm.mul %19, %12 overflow<nsw> : i32
    %21 = llvm.add %20, %17 overflow<nsw> : i32
    llvm.store %21, %16 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %22 = llvm.getelementptr inbounds %5[%15] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %23 = llvm.load %22 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %24 = llvm.mul %23, %19 overflow<nsw> : i32
    %25 = llvm.add %24, %14 overflow<nsw> : i32
    %26 = llvm.add %15, %6 overflow<nsw, nuw> : i64
    %27 = llvm.icmp "eq" %26, %7 : i64
    llvm.cond_br %27, ^bb3, ^bb2(%25, %26 : i32, i64) {loop_annotation = #loop_annotation}
  ^bb3:  // pred: ^bb2
    llvm.store %25, %11 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %28 = llvm.add %9, %6 overflow<nsw, nuw> : i64
    %29 = llvm.icmp "eq" %28, %7 : i64
    llvm.cond_br %29, ^bb4, ^bb1(%28 : i64) {loop_annotation = #loop_annotation}
  ^bb4:  // pred: ^bb3
    llvm.return %8 : i32
  }
}
