; ModuleID = 'test/benchmark/CGRA-Bench/kernels/fir/fir_int.cpp'
source_filename = "test/benchmark/CGRA-Bench/kernels/fir/fir_int.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@input = dso_local local_unnamed_addr global [32 x i32] zeroinitializer, align 16
@output = dso_local local_unnamed_addr global [32 x i32] zeroinitializer, align 16
@coefficients = dso_local local_unnamed_addr global [32 x i32] [i32 0, i32 1, i32 3, i32 -2, i32 0, i32 0, i32 -3, i32 1, i32 0, i32 1, i32 3, i32 -2, i32 0, i32 0, i32 -3, i32 1, i32 0, i32 1, i32 3, i32 -2, i32 0, i32 0, i32 -3, i32 1, i32 0, i32 1, i32 3, i32 -2, i32 0, i32 0, i32 -3, i32 1], align 16

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.phi = phi <4 x i32> [ zeroinitializer, %entry ], [ %3, %vector.body ]
  %0 = getelementptr inbounds nuw i32, ptr @input, i64 %index
  %wide.load = load <4 x i32>, ptr %0, align 16, !tbaa !5
  %1 = getelementptr inbounds nuw i32, ptr @coefficients, i64 %index
  %wide.load1 = load <4 x i32>, ptr %1, align 16, !tbaa !5
  %2 = mul nsw <4 x i32> %wide.load1, %wide.load
  %3 = add <4 x i32> %2, %vec.phi
  %index.next = add nuw i64 %index, 4
  %4 = icmp eq i64 %index.next, 32
  br i1 %4, label %middle.block, label %vector.body, !llvm.loop !9

middle.block:                                     ; preds = %vector.body
  %5 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %3)
  store i32 %5, ptr @output, align 16, !tbaa !5
  ret i32 0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local noundef i32 @_Z6kernelPiS_S_(ptr nocapture noundef readonly %input, ptr nocapture noundef readnone %output, ptr nocapture noundef readonly %coefficient) local_unnamed_addr #1 {
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.phi = phi <4 x i32> [ zeroinitializer, %entry ], [ %3, %vector.body ]
  %0 = getelementptr inbounds nuw i32, ptr %input, i64 %index
  %wide.load = load <4 x i32>, ptr %0, align 4, !tbaa !5
  %1 = getelementptr inbounds nuw i32, ptr %coefficient, i64 %index
  %wide.load10 = load <4 x i32>, ptr %1, align 4, !tbaa !5
  %2 = mul nsw <4 x i32> %wide.load10, %wide.load
  %3 = add <4 x i32> %2, %vec.phi
  %index.next = add nuw i64 %index, 4
  %4 = icmp eq i64 %index.next, 32
  br i1 %4, label %middle.block, label %vector.body, !llvm.loop !14

middle.block:                                     ; preds = %vector.body
  %5 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %3)
  ret i32 %5
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
