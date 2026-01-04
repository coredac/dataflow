; ModuleID = 'test/neura/fusion/kernel.cpp'
source_filename = "test/neura/fusion/kernel.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = dso_local local_unnamed_addr global [1024 x [1024 x i32]] zeroinitializer, align 16
@s = dso_local local_unnamed_addr global [1024 x i32] zeroinitializer, align 16
@q = dso_local local_unnamed_addr global [1024 x i32] zeroinitializer, align 16
@p = dso_local local_unnamed_addr global [1024 x i32] zeroinitializer, align 16
@r = dso_local local_unnamed_addr global [1024 x i32] zeroinitializer, align 16

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @_Z6kernelPA1024_iPiS1_S1_S1_(ptr nocapture noundef readonly %A, ptr nocapture noundef %s, ptr nocapture noundef %q, ptr nocapture noundef readonly %p, ptr nocapture noundef readonly %r) local_unnamed_addr #0 {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry, %for.inc24
  %indvars.iv45 = phi i64 [ 0, %entry ], [ %indvars.iv.next46, %for.inc24 ]
  %arrayidx5 = getelementptr inbounds nuw i32, ptr %r, i64 %indvars.iv45
  %arrayidx13 = getelementptr inbounds nuw i32, ptr %q, i64 %indvars.iv45
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.body3
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx = getelementptr inbounds nuw i32, ptr %s, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4, !tbaa !5
  %1 = load i32, ptr %arrayidx5, align 4, !tbaa !5
  %arrayidx9 = getelementptr inbounds nuw [1024 x i32], ptr %A, i64 %indvars.iv45, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx9, align 4, !tbaa !5
  %mul = mul nsw i32 %2, %1
  %add = add nsw i32 %mul, %0
  store i32 %add, ptr %arrayidx, align 4, !tbaa !5
  %3 = load i32, ptr %arrayidx13, align 4, !tbaa !5
  %4 = load i32, ptr %arrayidx9, align 4, !tbaa !5
  %arrayidx19 = getelementptr inbounds nuw i32, ptr %p, i64 %indvars.iv
  %5 = load i32, ptr %arrayidx19, align 4, !tbaa !5
  %mul20 = mul nsw i32 %5, %4
  %add21 = add nsw i32 %mul20, %3
  store i32 %add21, ptr %arrayidx13, align 4, !tbaa !5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.inc24, label %for.body3, !llvm.loop !9

for.inc24:                                        ; preds = %for.body3
  %indvars.iv.next46 = add nuw nsw i64 %indvars.iv45, 1
  %exitcond48.not = icmp eq i64 %indvars.iv.next46, 1024
  br i1 %exitcond48.not, label %for.end26, label %for.cond1.preheader, !llvm.loop !12

for.end26:                                        ; preds = %for.inc24
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
entry:
  br label %for.cond1.preheader.i

for.cond1.preheader.i:                            ; preds = %for.inc24.i, %entry
  %indvars.iv45.i = phi i64 [ 0, %entry ], [ %indvars.iv.next46.i, %for.inc24.i ]
  %arrayidx5.i = getelementptr inbounds nuw i32, ptr @r, i64 %indvars.iv45.i
  %arrayidx13.i = getelementptr inbounds nuw i32, ptr @q, i64 %indvars.iv45.i
  %0 = load i32, ptr %arrayidx5.i, align 4, !tbaa !5
  %arrayidx13.i.promoted = load i32, ptr %arrayidx13.i, align 4, !tbaa !5
  br label %for.body3.i

for.body3.i:                                      ; preds = %for.body3.i, %for.cond1.preheader.i
  %add21.i1 = phi i32 [ %arrayidx13.i.promoted, %for.cond1.preheader.i ], [ %add21.i, %for.body3.i ]
  %indvars.iv.i = phi i64 [ 0, %for.cond1.preheader.i ], [ %indvars.iv.next.i, %for.body3.i ]
  %arrayidx.i = getelementptr inbounds nuw i32, ptr @s, i64 %indvars.iv.i
  %1 = load i32, ptr %arrayidx.i, align 4, !tbaa !5
  %arrayidx9.i = getelementptr inbounds nuw [1024 x i32], ptr @A, i64 %indvars.iv45.i, i64 %indvars.iv.i
  %2 = load i32, ptr %arrayidx9.i, align 4, !tbaa !5
  %mul.i = mul nsw i32 %2, %0
  %add.i = add nsw i32 %mul.i, %1
  store i32 %add.i, ptr %arrayidx.i, align 4, !tbaa !5
  %arrayidx19.i = getelementptr inbounds nuw i32, ptr @p, i64 %indvars.iv.i
  %3 = load i32, ptr %arrayidx19.i, align 4, !tbaa !5
  %mul20.i = mul nsw i32 %3, %2
  %add21.i = add nsw i32 %mul20.i, %add21.i1
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, 1024
  br i1 %exitcond.not.i, label %for.inc24.i, label %for.body3.i, !llvm.loop !9

for.inc24.i:                                      ; preds = %for.body3.i
  store i32 %add21.i, ptr %arrayidx13.i, align 4, !tbaa !5
  %indvars.iv.next46.i = add nuw nsw i64 %indvars.iv45.i, 1
  %exitcond48.not.i = icmp eq i64 %indvars.iv.next46.i, 1024
  br i1 %exitcond48.not.i, label %_Z6kernelPA1024_iPiS1_S1_S1_.exit, label %for.cond1.preheader.i, !llvm.loop !12

_Z6kernelPA1024_iPiS1_S1_S1_.exit:                ; preds = %for.inc24.i
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
