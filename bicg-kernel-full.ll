; ModuleID = 'test/benchmark/CGRA-Bench/kernels/bicg/bicg.c'
source_filename = "test/benchmark/CGRA-Bench/kernels/bicg/bicg.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@stderr = external local_unnamed_addr global ptr, align 8
@.str.1 = private unnamed_addr constant [23 x i8] c"==BEGIN DUMP_ARRAYS==\0A\00", align 1
@.str.2 = private unnamed_addr constant [15 x i8] c"begin dump: %s\00", align 1
@.str.3 = private unnamed_addr constant [2 x i8] c"s\00", align 1
@.str.5 = private unnamed_addr constant [8 x i8] c"%0.2lf \00", align 1
@.str.6 = private unnamed_addr constant [17 x i8] c"\0Aend   dump: %s\0A\00", align 1
@.str.7 = private unnamed_addr constant [2 x i8] c"q\00", align 1
@.str.8 = private unnamed_addr constant [23 x i8] c"==END   DUMP_ARRAYS==\0A\00", align 1

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @kernel(i32 noundef %m, i32 noundef %n, ptr nocapture noundef readonly %A, ptr nocapture noundef %s, ptr nocapture noundef %q, ptr nocapture noundef readonly %p, ptr nocapture noundef readonly %r) local_unnamed_addr #0 {
entry:
  %cmp57 = icmp sgt i32 %m, 0
  br i1 %cmp57, label %for.cond1.preheader, label %for.cond1.preheader.thread

for.cond1.preheader:                              ; preds = %entry
  %0 = zext nneg i32 %m to i64
  %1 = shl nuw nsw i64 %0, 3
  tail call void @llvm.memset.p0.i64(ptr align 8 %s, i8 0, i64 %1, i1 false), !tbaa !5
  %cmp261 = icmp sgt i32 %n, 0
  br i1 %cmp261, label %for.body3.us.preheader, label %for.end34

for.cond1.preheader.thread:                       ; preds = %entry
  %cmp26172 = icmp sgt i32 %n, 0
  br i1 %cmp26172, label %for.body3.preheader, label %for.end34

for.body3.preheader:                              ; preds = %for.cond1.preheader.thread
  %2 = zext nneg i32 %n to i64
  %3 = shl nuw nsw i64 %2, 3
  tail call void @llvm.memset.p0.i64(ptr align 8 %q, i8 0, i64 %3, i1 false), !tbaa !5
  br label %for.end34

for.body3.us.preheader:                           ; preds = %for.cond1.preheader
  %wide.trip.count70 = zext nneg i32 %n to i64
  %wide.trip.count = zext nneg i32 %m to i64
  br label %for.body3.us

for.body3.us:                                     ; preds = %for.body3.us.preheader, %for.cond6.for.inc32_crit_edge.us
  %indvars.iv67 = phi i64 [ 0, %for.body3.us.preheader ], [ %indvars.iv.next68, %for.cond6.for.inc32_crit_edge.us ]
  %arrayidx5.us = getelementptr inbounds nuw double, ptr %q, i64 %indvars.iv67
  store double 0.000000e+00, ptr %arrayidx5.us, align 8, !tbaa !5
  %arrayidx12.us = getelementptr inbounds nuw double, ptr %r, i64 %indvars.iv67
  br label %for.body8.us

for.body8.us:                                     ; preds = %for.body3.us, %for.body8.us
  %indvars.iv = phi i64 [ 0, %for.body3.us ], [ %indvars.iv.next, %for.body8.us ]
  %arrayidx10.us = getelementptr inbounds nuw double, ptr %s, i64 %indvars.iv
  %4 = load double, ptr %arrayidx10.us, align 8, !tbaa !5
  %5 = load double, ptr %arrayidx12.us, align 8, !tbaa !5
  %arrayidx16.us = getelementptr inbounds nuw [116 x double], ptr %A, i64 %indvars.iv67, i64 %indvars.iv
  %6 = load double, ptr %arrayidx16.us, align 8, !tbaa !5
  %7 = tail call double @llvm.fmuladd.f64(double %5, double %6, double %4)
  store double %7, ptr %arrayidx10.us, align 8, !tbaa !5
  %8 = load double, ptr %arrayidx5.us, align 8, !tbaa !5
  %9 = load double, ptr %arrayidx16.us, align 8, !tbaa !5
  %arrayidx26.us = getelementptr inbounds nuw double, ptr %p, i64 %indvars.iv
  %10 = load double, ptr %arrayidx26.us, align 8, !tbaa !5
  %11 = tail call double @llvm.fmuladd.f64(double %9, double %10, double %8)
  store double %11, ptr %arrayidx5.us, align 8, !tbaa !5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond6.for.inc32_crit_edge.us, label %for.body8.us, !llvm.loop !9

for.cond6.for.inc32_crit_edge.us:                 ; preds = %for.body8.us
  %indvars.iv.next68 = add nuw nsw i64 %indvars.iv67, 1
  %exitcond71.not = icmp eq i64 %indvars.iv.next68, %wide.trip.count70
  br i1 %exitcond71.not, label %for.end34, label %for.body3.us, !llvm.loop !12

for.end34:                                        ; preds = %for.cond6.for.inc32_crit_edge.us, %for.cond1.preheader.thread, %for.body3.preheader, %for.cond1.preheader
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %argc, ptr nocapture noundef readonly %argv) local_unnamed_addr #2 {
entry:
  %call = tail call ptr @polybench_alloc_data(i64 noundef 14384, i32 noundef 8) #9
  %call1 = tail call ptr @polybench_alloc_data(i64 noundef 116, i32 noundef 8) #9
  %call2 = tail call ptr @polybench_alloc_data(i64 noundef 124, i32 noundef 8) #9
  %call3 = tail call ptr @polybench_alloc_data(i64 noundef 116, i32 noundef 8) #9
  %call4 = tail call ptr @polybench_alloc_data(i64 noundef 124, i32 noundef 8) #9
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %entry
  %indvars.iv.i = phi i64 [ 0, %entry ], [ %indvars.iv.next.i, %for.body.i ]
  %0 = trunc nuw nsw i64 %indvars.iv.i to i32
  %conv.i = uitofp nneg i32 %0 to double
  %div.i = fdiv double %conv.i, 1.160000e+02
  %arrayidx.i = getelementptr inbounds nuw double, ptr %call3, i64 %indvars.iv.i
  store double %div.i, ptr %arrayidx.i, align 8, !tbaa !5
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, 116
  br i1 %exitcond.not.i, label %for.body5.i, label %for.body.i, !llvm.loop !13

for.body5.i:                                      ; preds = %for.body.i, %for.inc27.i
  %indvars.iv58.i = phi i64 [ %indvars.iv.next59.i, %for.inc27.i ], [ 0, %for.body.i ]
  %1 = trunc nuw nsw i64 %indvars.iv58.i to i32
  %conv7.i = uitofp nneg i32 %1 to double
  %div9.i = fdiv double %conv7.i, 1.240000e+02
  %arrayidx11.i = getelementptr inbounds nuw double, ptr %call4, i64 %indvars.iv58.i
  store double %div9.i, ptr %arrayidx11.i, align 8, !tbaa !5
  br label %for.body15.i

for.body15.i:                                     ; preds = %for.body15.i, %for.body5.i
  %indvars.iv53.i = phi i64 [ 0, %for.body5.i ], [ %indvars.iv.next54.i, %for.body15.i ]
  %indvars.iv.next54.i = add nuw nsw i64 %indvars.iv53.i, 1
  %2 = mul nuw nsw i64 %indvars.iv.next54.i, %indvars.iv58.i
  %3 = trunc nuw nsw i64 %2 to i32
  %rem16.i = urem i32 %3, 124
  %conv17.i = uitofp nneg i32 %rem16.i to double
  %div19.i = fdiv double %conv17.i, 1.240000e+02
  %arrayidx23.i = getelementptr inbounds nuw [116 x double], ptr %call, i64 %indvars.iv58.i, i64 %indvars.iv53.i
  store double %div19.i, ptr %arrayidx23.i, align 8, !tbaa !5
  %exitcond57.not.i = icmp eq i64 %indvars.iv.next54.i, 116
  br i1 %exitcond57.not.i, label %for.inc27.i, label %for.body15.i, !llvm.loop !14

for.inc27.i:                                      ; preds = %for.body15.i
  %indvars.iv.next59.i = add nuw nsw i64 %indvars.iv58.i, 1
  %exitcond61.not.i = icmp eq i64 %indvars.iv.next59.i, 124
  br i1 %exitcond61.not.i, label %init_array.exit, label %for.body5.i, !llvm.loop !15

init_array.exit:                                  ; preds = %for.inc27.i
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(928) %call1, i8 0, i64 928, i1 false), !tbaa !5
  br label %for.body3.us.i

for.body3.us.i:                                   ; preds = %for.cond6.for.inc32_crit_edge.us.i, %init_array.exit
  %indvars.iv67.i = phi i64 [ 0, %init_array.exit ], [ %indvars.iv.next68.i, %for.cond6.for.inc32_crit_edge.us.i ]
  %arrayidx5.us.i = getelementptr inbounds nuw double, ptr %call2, i64 %indvars.iv67.i
  store double 0.000000e+00, ptr %arrayidx5.us.i, align 8, !tbaa !5
  %arrayidx12.us.i = getelementptr inbounds nuw double, ptr %call4, i64 %indvars.iv67.i
  br label %for.body8.us.i

for.body8.us.i:                                   ; preds = %for.body8.us.i, %for.body3.us.i
  %indvars.iv.i29 = phi i64 [ 0, %for.body3.us.i ], [ %indvars.iv.next.i30, %for.body8.us.i ]
  %arrayidx10.us.i = getelementptr inbounds nuw double, ptr %call1, i64 %indvars.iv.i29
  %4 = load double, ptr %arrayidx10.us.i, align 8, !tbaa !5
  %5 = load double, ptr %arrayidx12.us.i, align 8, !tbaa !5
  %arrayidx16.us.i = getelementptr inbounds nuw [116 x double], ptr %call, i64 %indvars.iv67.i, i64 %indvars.iv.i29
  %6 = load double, ptr %arrayidx16.us.i, align 8, !tbaa !5
  %7 = tail call double @llvm.fmuladd.f64(double %5, double %6, double %4)
  store double %7, ptr %arrayidx10.us.i, align 8, !tbaa !5
  %8 = load double, ptr %arrayidx5.us.i, align 8, !tbaa !5
  %9 = load double, ptr %arrayidx16.us.i, align 8, !tbaa !5
  %arrayidx26.us.i = getelementptr inbounds nuw double, ptr %call3, i64 %indvars.iv.i29
  %10 = load double, ptr %arrayidx26.us.i, align 8, !tbaa !5
  %11 = tail call double @llvm.fmuladd.f64(double %9, double %10, double %8)
  store double %11, ptr %arrayidx5.us.i, align 8, !tbaa !5
  %indvars.iv.next.i30 = add nuw nsw i64 %indvars.iv.i29, 1
  %exitcond.not.i31 = icmp eq i64 %indvars.iv.next.i30, 116
  br i1 %exitcond.not.i31, label %for.cond6.for.inc32_crit_edge.us.i, label %for.body8.us.i, !llvm.loop !9

for.cond6.for.inc32_crit_edge.us.i:               ; preds = %for.body8.us.i
  %indvars.iv.next68.i = add nuw nsw i64 %indvars.iv67.i, 1
  %exitcond71.not.i = icmp eq i64 %indvars.iv.next68.i, 124
  br i1 %exitcond71.not.i, label %kernel.exit, label %for.body3.us.i, !llvm.loop !12

kernel.exit:                                      ; preds = %for.cond6.for.inc32_crit_edge.us.i
  %cmp = icmp sgt i32 %argc, 42
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %kernel.exit
  %12 = load ptr, ptr %argv, align 8, !tbaa !16
  %strcmpload = load i8, ptr %12, align 1
  %tobool.not = icmp eq i8 %strcmpload, 0
  br i1 %tobool.not, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  tail call fastcc void @print_array(ptr noundef nonnull %call1, ptr noundef nonnull %call2)
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %kernel.exit
  tail call void @free(ptr noundef nonnull %call) #9
  tail call void @free(ptr noundef nonnull %call1) #9
  tail call void @free(ptr noundef nonnull %call2) #9
  tail call void @free(ptr noundef nonnull %call3) #9
  tail call void @free(ptr noundef nonnull %call4) #9
  ret i32 0
}

declare ptr @polybench_alloc_data(i64 noundef, i32 noundef) local_unnamed_addr #3

; Function Attrs: cold nofree nounwind uwtable
define internal fastcc void @print_array(ptr nocapture noundef readonly %s, ptr nocapture noundef readonly %q) unnamed_addr #4 {
entry:
  %0 = load ptr, ptr @stderr, align 8, !tbaa !19
  %1 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 22, i64 1, ptr %0) #10
  %2 = load ptr, ptr @stderr, align 8, !tbaa !19
  %call1 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %2, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.3) #11
  br label %for.body

for.body:                                         ; preds = %entry, %if.end
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %if.end ]
  %rem.lhs.trunc = trunc i64 %indvars.iv to i8
  %rem31 = urem i8 %rem.lhs.trunc, 20
  %cmp2 = icmp eq i8 %rem31, 0
  br i1 %cmp2, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %3 = load ptr, ptr @stderr, align 8, !tbaa !19
  %fputc30 = tail call i32 @fputc(i32 10, ptr %3)
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %4 = load ptr, ptr @stderr, align 8, !tbaa !19
  %arrayidx = getelementptr inbounds nuw double, ptr %s, i64 %indvars.iv
  %5 = load double, ptr %arrayidx, align 8, !tbaa !5
  %call4 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %4, ptr noundef nonnull @.str.5, double noundef %5) #11
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 116
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !21

for.end:                                          ; preds = %if.end
  %6 = load ptr, ptr @stderr, align 8, !tbaa !19
  %call5 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %6, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.3) #11
  %7 = load ptr, ptr @stderr, align 8, !tbaa !19
  %call6 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %7, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.7) #11
  br label %for.body9

for.body9:                                        ; preds = %for.end, %if.end14
  %indvars.iv36 = phi i64 [ 0, %for.end ], [ %indvars.iv.next37, %if.end14 ]
  %rem10.lhs.trunc = trunc i64 %indvars.iv36 to i8
  %rem1032 = urem i8 %rem10.lhs.trunc, 20
  %cmp11 = icmp eq i8 %rem1032, 0
  br i1 %cmp11, label %if.then12, label %if.end14

if.then12:                                        ; preds = %for.body9
  %8 = load ptr, ptr @stderr, align 8, !tbaa !19
  %fputc = tail call i32 @fputc(i32 10, ptr %8)
  br label %if.end14

if.end14:                                         ; preds = %if.then12, %for.body9
  %9 = load ptr, ptr @stderr, align 8, !tbaa !19
  %arrayidx16 = getelementptr inbounds nuw double, ptr %q, i64 %indvars.iv36
  %10 = load double, ptr %arrayidx16, align 8, !tbaa !5
  %call17 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %9, ptr noundef nonnull @.str.5, double noundef %10) #11
  %indvars.iv.next37 = add nuw nsw i64 %indvars.iv36, 1
  %exitcond39.not = icmp eq i64 %indvars.iv.next37, 124
  br i1 %exitcond39.not, label %for.end20, label %for.body9, !llvm.loop !22

for.end20:                                        ; preds = %if.end14
  %11 = load ptr, ptr @stderr, align 8, !tbaa !19
  %call21 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %11, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.7) #11
  %12 = load ptr, ptr @stderr, align 8, !tbaa !19
  %13 = tail call i64 @fwrite(ptr nonnull @.str.8, i64 22, i64 1, ptr %12) #10
  ret void
}

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr nocapture noundef) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr nocapture noundef, ptr nocapture noundef readonly, ...) local_unnamed_addr #6

; Function Attrs: nofree nounwind
declare noundef i64 @fwrite(ptr nocapture noundef, i64 noundef, i64 noundef, ptr nocapture noundef) local_unnamed_addr #7

; Function Attrs: nofree nounwind
declare noundef i32 @fputc(i32 noundef, ptr nocapture noundef) local_unnamed_addr #7

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #8

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { cold nofree nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { nofree nounwind }
attributes #8 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #9 = { nounwind }
attributes #10 = { cold }
attributes #11 = { cold nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"double", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10, !11}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.unroll.disable"}
!12 = distinct !{!12, !10, !11}
!13 = distinct !{!13, !10, !11}
!14 = distinct !{!14, !10, !11}
!15 = distinct !{!15, !10, !11}
!16 = !{!17, !17, i64 0}
!17 = !{!"p1 omnipotent char", !18, i64 0}
!18 = !{!"any pointer", !7, i64 0}
!19 = !{!20, !20, i64 0}
!20 = !{!"p1 _ZTS8_IO_FILE", !18, i64 0}
!21 = distinct !{!21, !10, !11}
!22 = distinct !{!22, !10, !11}
