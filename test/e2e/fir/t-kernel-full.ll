; ModuleID = '../../benchmark/CGRA-Bench/kernels/fir/fir_int.cpp'
source_filename = "../../benchmark/CGRA-Bench/kernels/fir/fir_int.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@input = dso_local local_unnamed_addr global [32 x i32] zeroinitializer, align 16
@output = dso_local local_unnamed_addr global [32 x i32] zeroinitializer, align 16
@coefficients = dso_local local_unnamed_addr global [32 x i32] [i32 0, i32 1, i32 3, i32 -2, i32 0, i32 0, i32 -3, i32 1, i32 0, i32 1, i32 3, i32 -2, i32 0, i32 0, i32 -3, i32 1, i32 0, i32 1, i32 3, i32 -2, i32 0, i32 0, i32 -3, i32 1, i32 0, i32 1, i32 3, i32 -2, i32 0, i32 0, i32 -3, i32 1], align 16

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %10, %1 ]
  %3 = phi <4 x i32> [ zeroinitializer, %0 ], [ %9, %1 ]
  %4 = getelementptr inbounds nuw i32, ptr @input, i64 %2
  %5 = load <4 x i32>, ptr %4, align 16, !tbaa !5
  %6 = getelementptr inbounds nuw i32, ptr @coefficients, i64 %2
  %7 = load <4 x i32>, ptr %6, align 16, !tbaa !5
  %8 = mul nsw <4 x i32> %7, %5
  %9 = add <4 x i32> %8, %3
  %10 = add nuw i64 %2, 4
  %11 = icmp eq i64 %10, 32
  br i1 %11, label %12, label %1, !llvm.loop !9

12:                                               ; preds = %1
  %13 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %9)
  store i32 %13, ptr @output, align 16, !tbaa !5
  ret i32 0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local noundef i32 @_Z6kernelPiS_S_(ptr nocapture noundef readonly %0, ptr nocapture noundef readnone %1, ptr nocapture noundef readonly %2) local_unnamed_addr #1 {
  br label %4

4:                                                ; preds = %4, %3
  %5 = phi i64 [ 0, %3 ], [ %13, %4 ]
  %6 = phi <4 x i32> [ zeroinitializer, %3 ], [ %12, %4 ]
  %7 = getelementptr inbounds nuw i32, ptr %0, i64 %5
  %8 = load <4 x i32>, ptr %7, align 4, !tbaa !5
  %9 = getelementptr inbounds nuw i32, ptr %2, i64 %5
  %10 = load <4 x i32>, ptr %9, align 4, !tbaa !5
  %11 = mul nsw <4 x i32> %10, %8
  %12 = add <4 x i32> %11, %6
  %13 = add nuw i64 %5, 4
  %14 = icmp eq i64 %13, 32
  br i1 %14, label %15, label %4, !llvm.loop !14

15:                                               ; preds = %4
  %16 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %12)
  ret i32 %16
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

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
!9 = distinct !{!9, !10, !11, !12, !13}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.unroll.disable"}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = !{!"llvm.loop.unroll.runtime.disable"}
!14 = distinct !{!14, !10, !11, !12, !13}
