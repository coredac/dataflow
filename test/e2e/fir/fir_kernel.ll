; ModuleID = 'test/e2e/fir/fir_kernel.cpp'
source_filename = "test/e2e/fir/fir_kernel.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z6kernelPfS_S_(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  %8 = alloca float, align 4
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %2, ptr %6, align 8
  store float 0.000000e+00, ptr %8, align 4
  store i32 0, ptr %7, align 4
  br label %9

9:                                                ; preds = %25, %3
  %10 = load i32, ptr %7, align 4
  %11 = icmp slt i32 %10, 32
  br i1 %11, label %12, label %28

12:                                               ; preds = %9
  %13 = load ptr, ptr %4, align 8
  %14 = load i32, ptr %7, align 4
  %15 = sext i32 %14 to i64
  %16 = getelementptr inbounds float, ptr %13, i64 %15
  %17 = load float, ptr %16, align 4
  %18 = load ptr, ptr %6, align 8
  %19 = load i32, ptr %7, align 4
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds float, ptr %18, i64 %20
  %22 = load float, ptr %21, align 4
  %23 = load float, ptr %8, align 4
  %24 = call float @llvm.fmuladd.f32(float %17, float %22, float %23)
  store float %24, ptr %8, align 4
  br label %25

25:                                               ; preds = %12
  %26 = load i32, ptr %7, align 4
  %27 = add nsw i32 %26, 1
  store i32 %27, ptr %7, align 4
  br label %9, !llvm.loop !6

28:                                               ; preds = %9
  %29 = load float, ptr %8, align 4
  %30 = load ptr, ptr %5, align 8
  %31 = getelementptr inbounds float, ptr %30, i64 0
  store float %29, ptr %31, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #1

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

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
