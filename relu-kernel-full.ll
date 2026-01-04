; ModuleID = 'test/benchmark/CGRA-Bench/kernels/relu/relu.c'
source_filename = "test/benchmark/CGRA-Bench/kernels/relu/relu.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@stderr = external local_unnamed_addr global ptr, align 8
@.str.1 = private unnamed_addr constant [23 x i8] c"==BEGIN DUMP_ARRAYS==\0A\00", align 1
@.str.2 = private unnamed_addr constant [15 x i8] c"begin dump: %s\00", align 1
@.str.3 = private unnamed_addr constant [2 x i8] c"C\00", align 1
@.str.5 = private unnamed_addr constant [4 x i8] c"%d \00", align 1
@.str.6 = private unnamed_addr constant [17 x i8] c"\0Aend   dump: %s\0A\00", align 1
@.str.7 = private unnamed_addr constant [23 x i8] c"==END   DUMP_ARRAYS==\0A\00", align 1

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @kernel(i32 noundef %ni, i32 noundef %nj, i32 noundef %nk, ptr nocapture noundef writeonly %C, ptr nocapture noundef readonly %A, ptr nocapture noundef readnone %B) local_unnamed_addr #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %x.029 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %div.lhs.trunc = trunc nuw i32 %x.029 to i16
  %div27 = udiv i16 %div.lhs.trunc, 70
  %rem28 = urem i16 %div.lhs.trunc, 70
  %idxprom = zext nneg i16 %div27 to i64
  %idxprom1 = zext nneg i16 %rem28 to i64
  %arrayidx2 = getelementptr inbounds nuw [70 x i32], ptr %A, i64 %idxprom, i64 %idxprom1
  %0 = load i32, ptr %arrayidx2, align 4, !tbaa !5
  %spec.select = tail call i32 @llvm.smax.i32(i32 %0, i32 0)
  %1 = getelementptr inbounds nuw [70 x i32], ptr %C, i64 %idxprom, i64 %idxprom1
  store i32 %spec.select, ptr %1, align 4, !tbaa !5
  %inc = add nuw nsw i32 %x.029, 1
  %exitcond.not = icmp eq i32 %inc, 4200
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !9

for.end:                                          ; preds = %for.body
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %argc, ptr nocapture noundef readonly %argv) local_unnamed_addr #1 {
entry:
  %call = tail call ptr @polybench_alloc_data(i64 noundef 4200, i32 noundef 4) #9
  %call1 = tail call ptr @polybench_alloc_data(i64 noundef 4200, i32 noundef 4) #9
  %call2 = tail call ptr @polybench_alloc_data(i64 noundef 4200, i32 noundef 4) #9
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(16800) %call, i8 0, i64 16800, i1 false), !tbaa !5
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(16800) %call1, i8 0, i64 16800, i1 false), !tbaa !5
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(16800) %call2, i8 0, i64 16800, i1 false), !tbaa !5
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %entry
  %x.029.i = phi i32 [ 0, %entry ], [ %inc.i, %for.body.i ]
  %div.lhs.trunc.i = trunc nuw i32 %x.029.i to i16
  %div27.i = udiv i16 %div.lhs.trunc.i, 70
  %rem28.i = urem i16 %div.lhs.trunc.i, 70
  %idxprom.i = zext nneg i16 %div27.i to i64
  %idxprom1.i = zext nneg i16 %rem28.i to i64
  %arrayidx2.i = getelementptr inbounds nuw [70 x i32], ptr %call1, i64 %idxprom.i, i64 %idxprom1.i
  %0 = load i32, ptr %arrayidx2.i, align 4, !tbaa !5
  %spec.select.i = tail call i32 @llvm.smax.i32(i32 %0, i32 0)
  %1 = getelementptr inbounds nuw [70 x i32], ptr %call, i64 %idxprom.i, i64 %idxprom1.i
  store i32 %spec.select.i, ptr %1, align 4, !tbaa !5
  %inc.i = add nuw nsw i32 %x.029.i, 1
  %exitcond.not.i = icmp eq i32 %inc.i, 4200
  br i1 %exitcond.not.i, label %kernel.exit, label %for.body.i, !llvm.loop !9

kernel.exit:                                      ; preds = %for.body.i
  %cmp = icmp sgt i32 %argc, 42
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %kernel.exit
  %2 = load ptr, ptr %argv, align 8, !tbaa !16
  %strcmpload = load i8, ptr %2, align 1
  %tobool.not = icmp eq i8 %strcmpload, 0
  br i1 %tobool.not, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  tail call fastcc void @print_array(ptr noundef nonnull %call)
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %kernel.exit
  tail call void @free(ptr noundef nonnull %call) #9
  tail call void @free(ptr noundef nonnull %call1) #9
  tail call void @free(ptr noundef %call2) #9
  ret i32 0
}

declare ptr @polybench_alloc_data(i64 noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: cold nofree nounwind uwtable
define internal fastcc void @print_array(ptr nocapture noundef readonly %C) unnamed_addr #3 {
entry:
  %0 = load ptr, ptr @stderr, align 8, !tbaa !19
  %1 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 22, i64 1, ptr %0) #10
  %2 = load ptr, ptr @stderr, align 8, !tbaa !19
  %call1 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %2, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.3) #11
  br label %for.cond2.preheader

for.cond2.preheader:                              ; preds = %entry, %for.inc10
  %indvars.iv26 = phi i64 [ 0, %entry ], [ %indvars.iv.next27, %for.inc10 ]
  %3 = mul nuw nsw i64 %indvars.iv26, 60
  br label %for.body4

for.body4:                                        ; preds = %for.cond2.preheader, %if.end
  %indvars.iv = phi i64 [ 0, %for.cond2.preheader ], [ %indvars.iv.next, %if.end ]
  %4 = add nuw nsw i64 %indvars.iv, %3
  %5 = trunc nuw nsw i64 %4 to i32
  %rem = urem i32 %5, 20
  %cmp5 = icmp eq i32 %rem, 0
  br i1 %cmp5, label %if.then, label %if.end

if.then:                                          ; preds = %for.body4
  %6 = load ptr, ptr @stderr, align 8, !tbaa !19
  %fputc = tail call i32 @fputc(i32 10, ptr %6)
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body4
  %7 = load ptr, ptr @stderr, align 8, !tbaa !19
  %arrayidx8 = getelementptr inbounds nuw [70 x i32], ptr %C, i64 %indvars.iv26, i64 %indvars.iv
  %8 = load i32, ptr %arrayidx8, align 4, !tbaa !5
  %call9 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %7, ptr noundef nonnull @.str.5, i32 noundef %8) #11
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 70
  br i1 %exitcond.not, label %for.inc10, label %for.body4, !llvm.loop !21

for.inc10:                                        ; preds = %if.end
  %indvars.iv.next27 = add nuw nsw i64 %indvars.iv26, 1
  %exitcond30.not = icmp eq i64 %indvars.iv.next27, 60
  br i1 %exitcond30.not, label %for.end12, label %for.cond2.preheader, !llvm.loop !23

for.end12:                                        ; preds = %for.inc10
  %9 = load ptr, ptr @stderr, align 8, !tbaa !19
  %call13 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %9, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.3) #11
  %10 = load ptr, ptr @stderr, align 8, !tbaa !19
  %11 = tail call i64 @fwrite(ptr nonnull @.str.7, i64 22, i64 1, ptr %10) #10
  ret void
}

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr nocapture noundef) local_unnamed_addr #4

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr nocapture noundef, ptr nocapture noundef readonly, ...) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare noundef i64 @fwrite(ptr nocapture noundef, i64 noundef, i64 noundef, ptr nocapture noundef) local_unnamed_addr #6

; Function Attrs: nofree nounwind
declare noundef i32 @fputc(i32 noundef, ptr nocapture noundef) local_unnamed_addr #6

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #7

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #8

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { cold nofree nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { nofree nounwind }
attributes #7 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
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
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10, !11, !12}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.vectorize.width", i32 1}
!12 = !{!"llvm.loop.vectorize.followup_all", !13}
!13 = distinct !{!13, !10, !14, !15}
!14 = !{!"llvm.loop.isvectorized"}
!15 = !{!"llvm.loop.unroll.count", i32 4}
!16 = !{!17, !17, i64 0}
!17 = !{!"p1 omnipotent char", !18, i64 0}
!18 = !{!"any pointer", !7, i64 0}
!19 = !{!20, !20, i64 0}
!20 = !{!"p1 _ZTS8_IO_FILE", !18, i64 0}
!21 = distinct !{!21, !10, !22}
!22 = !{!"llvm.loop.unroll.disable"}
!23 = distinct !{!23, !10, !22}
