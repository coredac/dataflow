; ModuleID = 'kernel.cpp'
source_filename = "kernel.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = dso_local local_unnamed_addr global [1024 x [1024 x i32]] zeroinitializer, align 16
@s = dso_local local_unnamed_addr global [1024 x i32] zeroinitializer, align 16
@q = dso_local local_unnamed_addr global [1024 x i32] zeroinitializer, align 16
@p = dso_local local_unnamed_addr global [1024 x i32] zeroinitializer, align 16
@r = dso_local local_unnamed_addr global [1024 x i32] zeroinitializer, align 16

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @_Z6kernelPA1024_iPiS1_S1_S1_(ptr nocapture noundef readonly %0, ptr nocapture noundef %1, ptr nocapture noundef %2, ptr nocapture noundef readonly %3, ptr nocapture noundef readonly %4) local_unnamed_addr #0 {
  br label %6

6:                                                ; preds = %5, %27
  %7 = phi i64 [ 0, %5 ], [ %28, %27 ]
  %8 = getelementptr inbounds nuw i32, ptr %4, i64 %7
  %9 = getelementptr inbounds nuw i32, ptr %2, i64 %7
  br label %10

10:                                               ; preds = %6, %10
  %11 = phi i64 [ 0, %6 ], [ %25, %10 ]
  %12 = getelementptr inbounds nuw i32, ptr %1, i64 %11
  %13 = load i32, ptr %12, align 4, !tbaa !5
  %14 = load i32, ptr %8, align 4, !tbaa !5
  %15 = getelementptr inbounds nuw [1024 x i32], ptr %0, i64 %7, i64 %11
  %16 = load i32, ptr %15, align 4, !tbaa !5
  %17 = mul nsw i32 %16, %14
  %18 = add nsw i32 %17, %13
  store i32 %18, ptr %12, align 4, !tbaa !5
  %19 = load i32, ptr %9, align 4, !tbaa !5
  %20 = load i32, ptr %15, align 4, !tbaa !5
  %21 = getelementptr inbounds nuw i32, ptr %3, i64 %11
  %22 = load i32, ptr %21, align 4, !tbaa !5
  %23 = mul nsw i32 %22, %20
  %24 = add nsw i32 %23, %19
  store i32 %24, ptr %9, align 4, !tbaa !5
  %25 = add nuw nsw i64 %11, 1
  %26 = icmp eq i64 %25, 1024
  br i1 %26, label %27, label %10, !llvm.loop !9

27:                                               ; preds = %10
  %28 = add nuw nsw i64 %7, 1
  %29 = icmp eq i64 %28, 1024
  br i1 %29, label %30, label %6, !llvm.loop !12

30:                                               ; preds = %27
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  br label %1

1:                                                ; preds = %22, %0
  %2 = phi i64 [ 0, %0 ], [ %23, %22 ]
  %3 = getelementptr inbounds nuw i32, ptr @r, i64 %2
  %4 = getelementptr inbounds nuw i32, ptr @q, i64 %2
  %5 = load i32, ptr %3, align 4, !tbaa !5
  %6 = load i32, ptr %4, align 4, !tbaa !5
  br label %7

7:                                                ; preds = %7, %1
  %8 = phi i32 [ %6, %1 ], [ %19, %7 ]
  %9 = phi i64 [ 0, %1 ], [ %20, %7 ]
  %10 = getelementptr inbounds nuw i32, ptr @s, i64 %9
  %11 = load i32, ptr %10, align 4, !tbaa !5
  %12 = getelementptr inbounds nuw [1024 x i32], ptr @A, i64 %2, i64 %9
  %13 = load i32, ptr %12, align 4, !tbaa !5
  %14 = mul nsw i32 %13, %5
  %15 = add nsw i32 %14, %11
  store i32 %15, ptr %10, align 4, !tbaa !5
  %16 = getelementptr inbounds nuw i32, ptr @p, i64 %9
  %17 = load i32, ptr %16, align 4, !tbaa !5
  %18 = mul nsw i32 %17, %13
  %19 = add nsw i32 %18, %8
  %20 = add nuw nsw i64 %9, 1
  %21 = icmp eq i64 %20, 1024
  br i1 %21, label %22, label %7, !llvm.loop !9

22:                                               ; preds = %7
  store i32 %19, ptr %4, align 4, !tbaa !5
  %23 = add nuw nsw i64 %2, 1
  %24 = icmp eq i64 %23, 1024
  br i1 %24, label %25, label %1, !llvm.loop !12

25:                                               ; preds = %22
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = distinct !{!9, !10, !11}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.unroll.disable"}
!12 = distinct !{!12, !10, !11}
