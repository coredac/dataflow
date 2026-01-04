; ModuleID = 'test/benchmark/CGRA-Bench/kernels/histogram/histogram_int.cpp'
source_filename = "test/benchmark/CGRA-Bench/kernels/histogram/histogram_int.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@input_data = dso_local local_unnamed_addr global [20 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 19], align 16
@histogram = dso_local local_unnamed_addr global [5 x i32] zeroinitializer, align 16
@.str = private unnamed_addr constant [27 x i8] c"DATA_LEN %d BUCKET_LEN %d\0A\00", align 1
@.str.1 = private unnamed_addr constant [8 x i8] c"len %d\0A\00", align 1
@.str.2 = private unnamed_addr constant [8 x i8] c"min %d\0A\00", align 1
@.str.3 = private unnamed_addr constant [8 x i8] c"del %d\0A\00", align 1
@.str.4 = private unnamed_addr constant [4 x i8] c"%d \00", align 1

; Function Attrs: mustprogress nofree norecurse nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %argc, ptr nocapture noundef readnone %argv) local_unnamed_addr #0 {
entry:
  %call = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 20, i32 noundef 5)
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %entry
  %indvars.iv.i = phi i64 [ 0, %entry ], [ %indvars.iv.next.i, %for.body.i ]
  %arrayidx.i = getelementptr inbounds nuw i32, ptr @input_data, i64 %indvars.iv.i
  %0 = load i32, ptr %arrayidx.i, align 4, !tbaa !5
  %1 = mul i32 %0, 5
  %mul.i = add i32 %1, -5
  %div.i = sdiv i32 %mul.i, 18
  %idxprom2.i = sext i32 %div.i to i64
  %arrayidx3.i = getelementptr inbounds i32, ptr @histogram, i64 %idxprom2.i
  %2 = load i32, ptr %arrayidx3.i, align 4, !tbaa !5
  %inc.i = add nsw i32 %2, 1
  store i32 %inc.i, ptr %arrayidx3.i, align 4, !tbaa !5
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, 20
  br i1 %exitcond.not.i, label %_Z6kernelPiS_.exit, label %for.body.i, !llvm.loop !9

_Z6kernelPiS_.exit:                               ; preds = %for.body.i
  %call.i = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 5)
  %call1.i = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 1)
  %call2.i = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef 18)
  br label %for.body.i1

for.body.i1:                                      ; preds = %for.body.i1, %_Z6kernelPiS_.exit
  %indvars.iv.i2 = phi i64 [ 0, %_Z6kernelPiS_.exit ], [ %indvars.iv.next.i4, %for.body.i1 ]
  %arrayidx.i3 = getelementptr inbounds nuw [5 x i32], ptr @histogram, i64 0, i64 %indvars.iv.i2
  %3 = load i32, ptr %arrayidx.i3, align 4, !tbaa !5
  %call3.i = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %3)
  %indvars.iv.next.i4 = add nuw nsw i64 %indvars.iv.i2, 1
  %exitcond.not.i5 = icmp eq i64 %indvars.iv.next.i4, 5
  br i1 %exitcond.not.i5, label %_Z6outputv.exit, label %for.body.i1, !llvm.loop !12

_Z6outputv.exit:                                  ; preds = %for.body.i1
  %putchar.i = tail call i32 @putchar(i32 10)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @_Z6kernelPiS_(ptr nocapture noundef readonly %input, ptr nocapture noundef %histogram) local_unnamed_addr #2 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds nuw i32, ptr %input, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4, !tbaa !5
  %1 = mul i32 %0, 5
  %mul = add i32 %1, -5
  %div = sdiv i32 %mul, 18
  %idxprom2 = sext i32 %div to i64
  %arrayidx3 = getelementptr inbounds i32, ptr %histogram, i64 %idxprom2
  %2 = load i32, ptr %arrayidx3, align 4, !tbaa !5
  %inc = add nsw i32 %2, 1
  store i32 %inc, ptr %arrayidx3, align 4, !tbaa !5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 20
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !9

for.end:                                          ; preds = %for.body
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @_Z6outputv() local_unnamed_addr #3 {
entry:
  %call = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 5)
  %call1 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 1)
  %call2 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef 18)
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %putchar = tail call i32 @putchar(i32 10)
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds nuw [5 x i32], ptr @histogram, i64 0, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4, !tbaa !5
  %call3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %0)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 5
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !12
}

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #4

attributes #0 = { mustprogress nofree norecurse nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { mustprogress nofree nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { nofree nounwind }

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
