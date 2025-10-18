; ModuleID = 'histogram_kernel.cpp'
source_filename = "histogram_kernel.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @_Z6kernelPfPi(ptr nocapture noundef readonly %0, ptr nocapture noundef %1) local_unnamed_addr #0 {
  br label %3

3:                                                ; preds = %3, %2
  %4 = phi i64 [ 0, %2 ], [ %26, %3 ]
  %5 = getelementptr inbounds nuw float, ptr %0, i64 %4
  %6 = load float, ptr %5, align 4, !tbaa !5
  %7 = fadd float %6, -1.000000e+00
  %8 = fmul float %7, 5.000000e+00
  %9 = fdiv float %8, 1.800000e+01
  %10 = fptosi float %9 to i32
  %11 = sext i32 %10 to i64
  %12 = getelementptr inbounds i32, ptr %1, i64 %11
  %13 = load i32, ptr %12, align 4, !tbaa !9
  %14 = add nsw i32 %13, 1
  store i32 %14, ptr %12, align 4, !tbaa !9
  %15 = or disjoint i64 %4, 1
  %16 = getelementptr inbounds nuw float, ptr %0, i64 %15
  %17 = load float, ptr %16, align 4, !tbaa !5
  %18 = fadd float %17, -1.000000e+00
  %19 = fmul float %18, 5.000000e+00
  %20 = fdiv float %19, 1.800000e+01
  %21 = fptosi float %20 to i32
  %22 = sext i32 %21 to i64
  %23 = getelementptr inbounds i32, ptr %1, i64 %22
  %24 = load i32, ptr %23, align 4, !tbaa !9
  %25 = add nsw i32 %24, 1
  store i32 %25, ptr %23, align 4, !tbaa !9
  %26 = add nuw nsw i64 %4, 2
  %27 = icmp eq i64 %26, 20
  br i1 %27, label %28, label %3, !llvm.loop !11

28:                                               ; preds = %3
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"float", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !7, i64 0}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
