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
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %entry
  %indvars.iv.i = phi i64 [ 0, %entry ], [ %indvars.iv.next.i, %for.body.i ]
  %sum.08.i = phi i32 [ 0, %entry ], [ %add.i, %for.body.i ]
  %arrayidx.i = getelementptr inbounds nuw i32, ptr @input, i64 %indvars.iv.i
  %0 = load i32, ptr %arrayidx.i, align 4, !tbaa !5
  %arrayidx2.i = getelementptr inbounds nuw i32, ptr @coefficients, i64 %indvars.iv.i
  %1 = load i32, ptr %arrayidx2.i, align 4, !tbaa !5
  %mul.i = mul nsw i32 %1, %0
  %add.i = add nsw i32 %mul.i, %sum.08.i
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, 32
  br i1 %exitcond.not.i, label %_Z6kernelPiS_S_.exit, label %for.body.i, !llvm.loop !9

_Z6kernelPiS_S_.exit:                             ; preds = %for.body.i
  store i32 %add.i, ptr @output, align 16, !tbaa !5
  ret i32 0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local noundef i32 @_Z6kernelPiS_S_(ptr nocapture noundef readonly %input, ptr nocapture noundef readnone %output, ptr nocapture noundef readonly %coefficient) local_unnamed_addr #1 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %sum.08 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds nuw i32, ptr %input, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4, !tbaa !5
  %arrayidx2 = getelementptr inbounds nuw i32, ptr %coefficient, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx2, align 4, !tbaa !5
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, %sum.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 32
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !9

for.end:                                          ; preds = %for.body
  ret i32 %add
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

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
