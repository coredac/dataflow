; ModuleID = 'fft.cpp'
source_filename = "fft.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@data_real = dso_local global [256 x float] zeroinitializer, align 16
@data_imag = dso_local global [256 x float] zeroinitializer, align 16
@coef_real = dso_local global [256 x float] zeroinitializer, align 16
@coef_imag = dso_local global [256 x float] zeroinitializer, align 16

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 0, ptr %j, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 256
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, ptr %i, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [256 x float], ptr @data_imag, i64 0, i64 %idxprom
  store float 1.000000e+00, ptr %arrayidx, align 4
  %2 = load i32, ptr %i, align 4
  %idxprom1 = sext i32 %2 to i64
  %arrayidx2 = getelementptr inbounds [256 x float], ptr @coef_real, i64 0, i64 %idxprom1
  store float 2.000000e+00, ptr %arrayidx2, align 4
  %3 = load i32, ptr %i, align 4
  %idxprom3 = sext i32 %3 to i64
  %arrayidx4 = getelementptr inbounds [256 x float], ptr @coef_imag, i64 0, i64 %idxprom3
  store float 2.000000e+00, ptr %arrayidx4, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %4 = load i32, ptr %i, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !6

for.end:                                          ; preds = %for.cond
  call void @_Z6kernelPfS_S_S_(ptr noundef @data_real, ptr noundef @data_imag, ptr noundef @coef_real, ptr noundef @coef_imag)
  ret i32 0
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z6kernelPfS_S_S_(ptr noundef %data_real, ptr noundef %data_imag, ptr noundef %coef_real, ptr noundef %coef_imag) #1 {
entry:
  %data_real.addr = alloca ptr, align 8
  %data_imag.addr = alloca ptr, align 8
  %coef_real.addr = alloca ptr, align 8
  %coef_imag.addr = alloca ptr, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  %temp_real = alloca float, align 4
  %temp_imag = alloca float, align 4
  %Wr = alloca float, align 4
  %Wi = alloca float, align 4
  %groupsPerStage = alloca i32, align 4
  %buttersPerGroup = alloca i32, align 4
  store ptr %data_real, ptr %data_real.addr, align 8
  store ptr %data_imag, ptr %data_imag.addr, align 8
  store ptr %coef_real, ptr %coef_real.addr, align 8
  store ptr %coef_imag, ptr %coef_imag.addr, align 8
  store i32 1, ptr %groupsPerStage, align 4
  store i32 128, ptr %buttersPerGroup, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc79, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 8
  br i1 %cmp, label %for.body, label %for.end81

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc75, %for.body
  %1 = load i32, ptr %j, align 4
  %2 = load i32, ptr %groupsPerStage, align 4
  %cmp2 = icmp slt i32 %1, %2
  br i1 %cmp2, label %for.body3, label %for.end77

for.body3:                                        ; preds = %for.cond1
  %3 = load ptr, ptr %coef_real.addr, align 8
  %4 = load i32, ptr %i, align 4
  %shl = shl i32 1, %4
  %sub = sub nsw i32 %shl, 1
  %5 = load i32, ptr %j, align 4
  %add = add nsw i32 %sub, %5
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds float, ptr %3, i64 %idxprom
  %6 = load float, ptr %arrayidx, align 4
  store float %6, ptr %Wr, align 4
  %7 = load ptr, ptr %coef_imag.addr, align 8
  %8 = load i32, ptr %i, align 4
  %shl4 = shl i32 1, %8
  %sub5 = sub nsw i32 %shl4, 1
  %9 = load i32, ptr %j, align 4
  %add6 = add nsw i32 %sub5, %9
  %idxprom7 = sext i32 %add6 to i64
  %arrayidx8 = getelementptr inbounds float, ptr %7, i64 %idxprom7
  %10 = load float, ptr %arrayidx8, align 4
  store float %10, ptr %Wi, align 4
  store i32 0, ptr %k, align 4
  br label %for.cond9

for.cond9:                                        ; preds = %for.inc, %for.body3
  %11 = load i32, ptr %k, align 4
  %12 = load i32, ptr %buttersPerGroup, align 4
  %cmp10 = icmp slt i32 %11, %12
  br i1 %cmp10, label %for.body11, label %for.end

for.body11:                                       ; preds = %for.cond9
  %13 = load float, ptr %Wr, align 4
  %14 = load ptr, ptr %data_real.addr, align 8
  %15 = load i32, ptr %j, align 4
  %mul = mul nsw i32 2, %15
  %16 = load i32, ptr %buttersPerGroup, align 4
  %mul12 = mul nsw i32 %mul, %16
  %17 = load i32, ptr %buttersPerGroup, align 4
  %add13 = add nsw i32 %mul12, %17
  %18 = load i32, ptr %k, align 4
  %add14 = add nsw i32 %add13, %18
  %idxprom15 = sext i32 %add14 to i64
  %arrayidx16 = getelementptr inbounds float, ptr %14, i64 %idxprom15
  %19 = load float, ptr %arrayidx16, align 4
  %20 = load float, ptr %Wi, align 4
  %21 = load ptr, ptr %data_imag.addr, align 8
  %22 = load i32, ptr %j, align 4
  %mul18 = mul nsw i32 2, %22
  %23 = load i32, ptr %buttersPerGroup, align 4
  %mul19 = mul nsw i32 %mul18, %23
  %24 = load i32, ptr %buttersPerGroup, align 4
  %add20 = add nsw i32 %mul19, %24
  %25 = load i32, ptr %k, align 4
  %add21 = add nsw i32 %add20, %25
  %idxprom22 = sext i32 %add21 to i64
  %arrayidx23 = getelementptr inbounds float, ptr %21, i64 %idxprom22
  %26 = load float, ptr %arrayidx23, align 4
  %mul24 = fmul float %20, %26
  %neg = fneg float %mul24
  %27 = call float @llvm.fmuladd.f32(float %13, float %19, float %neg)
  store float %27, ptr %temp_real, align 4
  %28 = load float, ptr %Wi, align 4
  %29 = load ptr, ptr %data_real.addr, align 8
  %30 = load i32, ptr %j, align 4
  %mul25 = mul nsw i32 2, %30
  %31 = load i32, ptr %buttersPerGroup, align 4
  %mul26 = mul nsw i32 %mul25, %31
  %32 = load i32, ptr %buttersPerGroup, align 4
  %add27 = add nsw i32 %mul26, %32
  %33 = load i32, ptr %k, align 4
  %add28 = add nsw i32 %add27, %33
  %idxprom29 = sext i32 %add28 to i64
  %arrayidx30 = getelementptr inbounds float, ptr %29, i64 %idxprom29
  %34 = load float, ptr %arrayidx30, align 4
  %35 = load float, ptr %Wr, align 4
  %36 = load ptr, ptr %data_imag.addr, align 8
  %37 = load i32, ptr %j, align 4
  %mul32 = mul nsw i32 2, %37
  %38 = load i32, ptr %buttersPerGroup, align 4
  %mul33 = mul nsw i32 %mul32, %38
  %39 = load i32, ptr %buttersPerGroup, align 4
  %add34 = add nsw i32 %mul33, %39
  %40 = load i32, ptr %k, align 4
  %add35 = add nsw i32 %add34, %40
  %idxprom36 = sext i32 %add35 to i64
  %arrayidx37 = getelementptr inbounds float, ptr %36, i64 %idxprom36
  %41 = load float, ptr %arrayidx37, align 4
  %mul38 = fmul float %35, %41
  %42 = call float @llvm.fmuladd.f32(float %28, float %34, float %mul38)
  store float %42, ptr %temp_imag, align 4
  %43 = load ptr, ptr %data_real.addr, align 8
  %44 = load i32, ptr %j, align 4
  %mul39 = mul nsw i32 2, %44
  %45 = load i32, ptr %buttersPerGroup, align 4
  %mul40 = mul nsw i32 %mul39, %45
  %46 = load i32, ptr %k, align 4
  %add41 = add nsw i32 %mul40, %46
  %idxprom42 = sext i32 %add41 to i64
  %arrayidx43 = getelementptr inbounds float, ptr %43, i64 %idxprom42
  %47 = load float, ptr %arrayidx43, align 4
  %48 = load float, ptr %temp_real, align 4
  %sub44 = fsub float %47, %48
  %49 = load ptr, ptr %data_real.addr, align 8
  %50 = load i32, ptr %j, align 4
  %mul45 = mul nsw i32 2, %50
  %51 = load i32, ptr %buttersPerGroup, align 4
  %mul46 = mul nsw i32 %mul45, %51
  %52 = load i32, ptr %buttersPerGroup, align 4
  %add47 = add nsw i32 %mul46, %52
  %53 = load i32, ptr %k, align 4
  %add48 = add nsw i32 %add47, %53
  %idxprom49 = sext i32 %add48 to i64
  %arrayidx50 = getelementptr inbounds float, ptr %49, i64 %idxprom49
  store float %sub44, ptr %arrayidx50, align 4
  %54 = load float, ptr %temp_real, align 4
  %55 = load ptr, ptr %data_real.addr, align 8
  %56 = load i32, ptr %j, align 4
  %mul51 = mul nsw i32 2, %56
  %57 = load i32, ptr %buttersPerGroup, align 4
  %mul52 = mul nsw i32 %mul51, %57
  %58 = load i32, ptr %k, align 4
  %add53 = add nsw i32 %mul52, %58
  %idxprom54 = sext i32 %add53 to i64
  %arrayidx55 = getelementptr inbounds float, ptr %55, i64 %idxprom54
  %59 = load float, ptr %arrayidx55, align 4
  %add56 = fadd float %59, %54
  store float %add56, ptr %arrayidx55, align 4
  %60 = load ptr, ptr %data_imag.addr, align 8
  %61 = load i32, ptr %j, align 4
  %mul57 = mul nsw i32 2, %61
  %62 = load i32, ptr %buttersPerGroup, align 4
  %mul58 = mul nsw i32 %mul57, %62
  %63 = load i32, ptr %k, align 4
  %add59 = add nsw i32 %mul58, %63
  %idxprom60 = sext i32 %add59 to i64
  %arrayidx61 = getelementptr inbounds float, ptr %60, i64 %idxprom60
  %64 = load float, ptr %arrayidx61, align 4
  %65 = load float, ptr %temp_imag, align 4
  %sub62 = fsub float %64, %65
  %66 = load ptr, ptr %data_imag.addr, align 8
  %67 = load i32, ptr %j, align 4
  %mul63 = mul nsw i32 2, %67
  %68 = load i32, ptr %buttersPerGroup, align 4
  %mul64 = mul nsw i32 %mul63, %68
  %69 = load i32, ptr %buttersPerGroup, align 4
  %add65 = add nsw i32 %mul64, %69
  %70 = load i32, ptr %k, align 4
  %add66 = add nsw i32 %add65, %70
  %idxprom67 = sext i32 %add66 to i64
  %arrayidx68 = getelementptr inbounds float, ptr %66, i64 %idxprom67
  store float %sub62, ptr %arrayidx68, align 4
  %71 = load float, ptr %temp_imag, align 4
  %72 = load ptr, ptr %data_imag.addr, align 8
  %73 = load i32, ptr %j, align 4
  %mul69 = mul nsw i32 2, %73
  %74 = load i32, ptr %buttersPerGroup, align 4
  %mul70 = mul nsw i32 %mul69, %74
  %75 = load i32, ptr %k, align 4
  %add71 = add nsw i32 %mul70, %75
  %idxprom72 = sext i32 %add71 to i64
  %arrayidx73 = getelementptr inbounds float, ptr %72, i64 %idxprom72
  %76 = load float, ptr %arrayidx73, align 4
  %add74 = fadd float %76, %71
  store float %add74, ptr %arrayidx73, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body11
  %77 = load i32, ptr %k, align 4
  %inc = add nsw i32 %77, 1
  store i32 %inc, ptr %k, align 4
  br label %for.cond9, !llvm.loop !8

for.end:                                          ; preds = %for.cond9
  br label %for.inc75

for.inc75:                                        ; preds = %for.end
  %78 = load i32, ptr %j, align 4
  %inc76 = add nsw i32 %78, 1
  store i32 %inc76, ptr %j, align 4
  br label %for.cond1, !llvm.loop !16

for.end77:                                        ; preds = %for.cond1
  %79 = load i32, ptr %groupsPerStage, align 4
  %shl78 = shl i32 %79, 1
  store i32 %shl78, ptr %groupsPerStage, align 4
  %80 = load i32, ptr %buttersPerGroup, align 4
  %shr = ashr i32 %80, 1
  store i32 %shr, ptr %buttersPerGroup, align 4
  br label %for.inc79

for.inc79:                                        ; preds = %for.end77
  %81 = load i32, ptr %i, align 4
  %inc80 = add nsw i32 %81, 1
  store i32 %inc80, ptr %i, align 4
  br label %for.cond, !llvm.loop !17

for.end81:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #2

attributes #0 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7, !9, !10, !11, !12}
!9 = !{!"llvm.loop.vectorize.width", i32 4}
!10 = !{!"llvm.loop.vectorize.scalable.enable", i1 false}
!11 = !{!"llvm.loop.vectorize.enable", i1 true}
!12 = !{!"llvm.loop.vectorize.followup_all", !13}
!13 = distinct !{!13, !7, !14, !15}
!14 = !{!"llvm.loop.isvectorized"}
!15 = !{!"llvm.loop.unroll.count", i32 4}
!16 = distinct !{!16, !7}
!17 = distinct !{!17, !7}
