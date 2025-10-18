; ModuleID = 'fft.cpp'
source_filename = "fft.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@data_real = dso_local local_unnamed_addr global [256 x float] zeroinitializer, align 16
@data_imag = dso_local local_unnamed_addr global [256 x float] zeroinitializer, align 16
@coef_real = dso_local local_unnamed_addr global [256 x float] zeroinitializer, align 16
@coef_imag = dso_local local_unnamed_addr global [256 x float] zeroinitializer, align 16

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds nuw [256 x float], ptr @data_imag, i64 0, i64 %indvars.iv
  store float 1.000000e+00, ptr %arrayidx, align 4, !tbaa !5
  %arrayidx2 = getelementptr inbounds nuw [256 x float], ptr @coef_real, i64 0, i64 %indvars.iv
  store float 2.000000e+00, ptr %arrayidx2, align 4, !tbaa !5
  %arrayidx4 = getelementptr inbounds nuw [256 x float], ptr @coef_imag, i64 0, i64 %indvars.iv
  store float 2.000000e+00, ptr %arrayidx4, align 4, !tbaa !5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 256
  br i1 %exitcond.not, label %for.cond1.preheader.i, label %for.body, !llvm.loop !9

for.cond1.preheader.i:                            ; preds = %for.body, %for.end77.i
  %buttersPerGroup.0145.i = phi i32 [ %shr.i, %for.end77.i ], [ 128, %for.body ]
  %groupsPerStage.0144.i = phi i32 [ %shl78.i, %for.end77.i ], [ 1, %for.body ]
  %i.0143.i = phi i32 [ %inc80.i, %for.end77.i ], [ 0, %for.body ]
  %notmask.i = shl nsw i32 -1, %i.0143.i
  %sub.i = xor i32 %notmask.i, -1
  %0 = zext nneg i32 %buttersPerGroup.0145.i to i64
  %1 = zext nneg i32 %sub.i to i64
  %wide.trip.count156.i = zext i32 %groupsPerStage.0144.i to i64
  %min.iters.check = icmp samesign ult i32 %buttersPerGroup.0145.i, 4
  %n.vec = and i64 %0, 252
  %cmp.n = icmp eq i64 %n.vec, %0
  br label %for.body3.i

for.body3.i:                                      ; preds = %for.inc75.i, %for.cond1.preheader.i
  %indvars.iv149.i = phi i64 [ 0, %for.cond1.preheader.i ], [ %indvars.iv.next150.i, %for.inc75.i ]
  %2 = add nuw nsw i64 %indvars.iv149.i, %1
  %arrayidx.i = getelementptr inbounds nuw float, ptr @coef_real, i64 %2
  %3 = load float, ptr %arrayidx.i, align 4, !tbaa !5
  %arrayidx8.i = getelementptr inbounds nuw float, ptr @coef_imag, i64 %2
  %4 = load float, ptr %arrayidx8.i, align 4, !tbaa !5
  %5 = shl nuw nsw i64 %indvars.iv149.i, 1
  %6 = mul nuw nsw i64 %5, %0
  %7 = add nuw nsw i64 %6, %0
  br i1 %min.iters.check, label %for.body11.i.preheader, label %vector.ph

vector.ph:                                        ; preds = %for.body3.i
  %broadcast.splatinsert = insertelement <4 x float> poison, float %4, i64 0
  %broadcast.splat = shufflevector <4 x float> %broadcast.splatinsert, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert12 = insertelement <4 x float> poison, float %3, i64 0
  %broadcast.splat13 = shufflevector <4 x float> %broadcast.splatinsert12, <4 x float> poison, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %8 = add nuw nsw i64 %7, %index
  %9 = getelementptr inbounds nuw float, ptr @data_real, i64 %8
  %wide.load = load <4 x float>, ptr %9, align 4, !tbaa !5
  %10 = getelementptr inbounds nuw float, ptr @data_imag, i64 %8
  %wide.load11 = load <4 x float>, ptr %10, align 4, !tbaa !5
  %11 = fneg <4 x float> %wide.load11
  %12 = fmul <4 x float> %broadcast.splat, %11
  %13 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat13, <4 x float> %wide.load, <4 x float> %12)
  %14 = fmul <4 x float> %broadcast.splat13, %wide.load11
  %15 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat, <4 x float> %wide.load, <4 x float> %14)
  %16 = add nuw nsw i64 %index, %6
  %17 = getelementptr inbounds nuw float, ptr @data_real, i64 %16
  %wide.load14 = load <4 x float>, ptr %17, align 8, !tbaa !5
  %18 = fsub <4 x float> %wide.load14, %13
  store <4 x float> %18, ptr %9, align 4, !tbaa !5
  %19 = fadd <4 x float> %wide.load14, %13
  store <4 x float> %19, ptr %17, align 8, !tbaa !5
  %20 = getelementptr inbounds nuw float, ptr @data_imag, i64 %16
  %wide.load15 = load <4 x float>, ptr %20, align 8, !tbaa !5
  %21 = fsub <4 x float> %wide.load15, %15
  store <4 x float> %21, ptr %10, align 4, !tbaa !5
  %22 = fadd <4 x float> %15, %wide.load15
  store <4 x float> %22, ptr %20, align 8, !tbaa !5
  %index.next = add nuw i64 %index, 4
  %23 = icmp eq i64 %index.next, %n.vec
  br i1 %23, label %middle.block, label %vector.body, !llvm.loop !12

middle.block:                                     ; preds = %vector.body
  br i1 %cmp.n, label %for.inc75.i, label %for.body11.i.preheader

for.body11.i.preheader:                           ; preds = %for.body3.i, %middle.block
  %indvars.iv.i.ph = phi i64 [ 0, %for.body3.i ], [ %n.vec, %middle.block ]
  br label %for.body11.i

for.body11.i:                                     ; preds = %for.body11.i.preheader, %for.body11.i
  %indvars.iv.i = phi i64 [ %indvars.iv.next.i, %for.body11.i ], [ %indvars.iv.i.ph, %for.body11.i.preheader ]
  %24 = add nuw nsw i64 %7, %indvars.iv.i
  %arrayidx16.i = getelementptr inbounds nuw float, ptr @data_real, i64 %24
  %25 = load float, ptr %arrayidx16.i, align 4, !tbaa !5
  %arrayidx23.i = getelementptr inbounds nuw float, ptr @data_imag, i64 %24
  %26 = load float, ptr %arrayidx23.i, align 4, !tbaa !5
  %27 = fneg float %26
  %neg.i = fmul float %4, %27
  %28 = tail call float @llvm.fmuladd.f32(float %3, float %25, float %neg.i)
  %mul38.i = fmul float %3, %26
  %29 = tail call float @llvm.fmuladd.f32(float %4, float %25, float %mul38.i)
  %30 = add nuw nsw i64 %indvars.iv.i, %6
  %arrayidx43.i = getelementptr inbounds nuw float, ptr @data_real, i64 %30
  %31 = load float, ptr %arrayidx43.i, align 4, !tbaa !5
  %sub44.i = fsub float %31, %28
  store float %sub44.i, ptr %arrayidx16.i, align 4, !tbaa !5
  %add56.i = fadd float %31, %28
  store float %add56.i, ptr %arrayidx43.i, align 4, !tbaa !5
  %arrayidx61.i = getelementptr inbounds nuw float, ptr @data_imag, i64 %30
  %32 = load float, ptr %arrayidx61.i, align 4, !tbaa !5
  %sub62.i = fsub float %32, %29
  store float %sub62.i, ptr %arrayidx23.i, align 4, !tbaa !5
  %add74.i = fadd float %29, %32
  store float %add74.i, ptr %arrayidx61.i, align 4, !tbaa !5
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, %0
  br i1 %exitcond.not.i, label %for.inc75.i, label %for.body11.i, !llvm.loop !17

for.inc75.i:                                      ; preds = %for.body11.i, %middle.block
  %indvars.iv.next150.i = add nuw nsw i64 %indvars.iv149.i, 1
  %exitcond157.not.i = icmp eq i64 %indvars.iv.next150.i, %wide.trip.count156.i
  br i1 %exitcond157.not.i, label %for.end77.i, label %for.body3.i, !llvm.loop !18

for.end77.i:                                      ; preds = %for.inc75.i
  %shl78.i = shl i32 %groupsPerStage.0144.i, 1
  %shr.i = lshr i32 %buttersPerGroup.0145.i, 1
  %inc80.i = add nuw nsw i32 %i.0143.i, 1
  %exitcond158.not.i = icmp eq i32 %inc80.i, 8
  br i1 %exitcond158.not.i, label %_Z6kernelPfS_S_S_.exit, label %for.cond1.preheader.i, !llvm.loop !19

_Z6kernelPfS_S_S_.exit:                           ; preds = %for.end77.i
  ret i32 0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @_Z6kernelPfS_S_S_(ptr nocapture noundef %data_real, ptr nocapture noundef %data_imag, ptr nocapture noundef readonly %coef_real, ptr nocapture noundef readonly %coef_imag) local_unnamed_addr #1 {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry, %for.end77
  %buttersPerGroup.0145 = phi i32 [ 128, %entry ], [ %shr, %for.end77 ]
  %groupsPerStage.0144 = phi i32 [ 1, %entry ], [ %shl78, %for.end77 ]
  %i.0143 = phi i32 [ 0, %entry ], [ %inc80, %for.end77 ]
  %notmask = shl nsw i32 -1, %i.0143
  %sub = xor i32 %notmask, -1
  %0 = zext nneg i32 %buttersPerGroup.0145 to i64
  %1 = zext nneg i32 %buttersPerGroup.0145 to i64
  %2 = zext nneg i32 %sub to i64
  %wide.trip.count156 = zext i32 %groupsPerStage.0144 to i64
  %3 = shl nuw nsw i64 %wide.trip.count156, 3
  %4 = add nsw i64 %3, -4
  %5 = mul nsw i64 %4, %0
  %scevgep = getelementptr i8, ptr %data_real, i64 %5
  %scevgep159 = getelementptr i8, ptr %data_imag, i64 %5
  %6 = shl nuw nsw i64 %0, 2
  %scevgep160 = getelementptr nuw i8, ptr %data_imag, i64 %6
  %7 = shl nuw nsw i64 %wide.trip.count156, 3
  %8 = mul nuw nsw i64 %7, %0
  %scevgep161 = getelementptr i8, ptr %data_imag, i64 %8
  %scevgep162 = getelementptr nuw i8, ptr %data_real, i64 %6
  %scevgep163 = getelementptr i8, ptr %data_real, i64 %8
  %wide.trip.count = zext nneg i32 %buttersPerGroup.0145 to i64
  %min.iters.check = icmp samesign ult i32 %buttersPerGroup.0145, 4
  %bound0 = icmp ult ptr %data_real, %scevgep159
  %bound1 = icmp ult ptr %data_imag, %scevgep
  %found.conflict = and i1 %bound0, %bound1
  %bound0164 = icmp ult ptr %data_real, %scevgep161
  %bound1165 = icmp ult ptr %scevgep160, %scevgep
  %found.conflict166 = and i1 %bound0164, %bound1165
  %conflict.rdx = or i1 %found.conflict, %found.conflict166
  %bound0167 = icmp ult ptr %scevgep162, %scevgep159
  %bound1168 = icmp ult ptr %data_imag, %scevgep163
  %found.conflict169 = and i1 %bound0167, %bound1168
  %conflict.rdx170 = or i1 %conflict.rdx, %found.conflict169
  %bound0171 = icmp ult ptr %scevgep162, %scevgep161
  %bound1172 = icmp ult ptr %scevgep160, %scevgep163
  %found.conflict173 = and i1 %bound0171, %bound1172
  %conflict.rdx174 = or i1 %conflict.rdx170, %found.conflict173
  %n.vec = and i64 %0, 252
  %cmp.n = icmp eq i64 %n.vec, %0
  br label %for.body3

for.body3:                                        ; preds = %for.cond1.preheader, %for.inc75
  %indvars.iv149 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next150, %for.inc75 ]
  %9 = add nuw nsw i64 %indvars.iv149, %2
  %arrayidx = getelementptr inbounds nuw float, ptr %coef_real, i64 %9
  %10 = load float, ptr %arrayidx, align 4, !tbaa !5
  %arrayidx8 = getelementptr inbounds nuw float, ptr %coef_imag, i64 %9
  %11 = load float, ptr %arrayidx8, align 4, !tbaa !5
  %12 = shl nuw nsw i64 %indvars.iv149, 1
  %13 = mul nuw nsw i64 %12, %0
  %14 = add nuw nsw i64 %13, %1
  %brmerge = select i1 %min.iters.check, i1 true, i1 %conflict.rdx174
  br i1 %brmerge, label %for.body11.preheader, label %vector.ph

vector.ph:                                        ; preds = %for.body3
  %broadcast.splatinsert = insertelement <4 x float> poison, float %11, i64 0
  %broadcast.splat = shufflevector <4 x float> %broadcast.splatinsert, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert176 = insertelement <4 x float> poison, float %10, i64 0
  %broadcast.splat177 = shufflevector <4 x float> %broadcast.splatinsert176, <4 x float> poison, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %15 = add nuw nsw i64 %14, %index
  %16 = getelementptr inbounds nuw float, ptr %data_real, i64 %15
  %wide.load = load <4 x float>, ptr %16, align 4, !tbaa !5, !alias.scope !20, !noalias !23
  %17 = getelementptr inbounds nuw float, ptr %data_imag, i64 %15
  %wide.load175 = load <4 x float>, ptr %17, align 4, !tbaa !5, !alias.scope !26
  %18 = fneg <4 x float> %wide.load175
  %19 = fmul <4 x float> %broadcast.splat, %18
  %20 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat177, <4 x float> %wide.load, <4 x float> %19)
  %21 = fmul <4 x float> %broadcast.splat177, %wide.load175
  %22 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat, <4 x float> %wide.load, <4 x float> %21)
  %23 = add nuw nsw i64 %index, %13
  %24 = getelementptr inbounds nuw float, ptr %data_real, i64 %23
  %wide.load178 = load <4 x float>, ptr %24, align 4, !tbaa !5, !alias.scope !27, !noalias !23
  %25 = fsub <4 x float> %wide.load178, %20
  store <4 x float> %25, ptr %16, align 4, !tbaa !5, !alias.scope !20, !noalias !23
  %26 = fadd <4 x float> %wide.load178, %20
  store <4 x float> %26, ptr %24, align 4, !tbaa !5, !alias.scope !27, !noalias !23
  %27 = getelementptr inbounds nuw float, ptr %data_imag, i64 %23
  %wide.load179 = load <4 x float>, ptr %27, align 4, !tbaa !5, !alias.scope !29
  %28 = fsub <4 x float> %wide.load179, %22
  store <4 x float> %28, ptr %17, align 4, !tbaa !5, !alias.scope !26
  %29 = fadd <4 x float> %22, %wide.load179
  store <4 x float> %29, ptr %27, align 4, !tbaa !5, !alias.scope !29
  %index.next = add nuw i64 %index, 4
  %30 = icmp eq i64 %index.next, %n.vec
  br i1 %30, label %middle.block, label %vector.body, !llvm.loop !30

middle.block:                                     ; preds = %vector.body
  br i1 %cmp.n, label %for.inc75, label %for.body11.preheader

for.body11.preheader:                             ; preds = %for.body3, %middle.block
  %indvars.iv.ph = phi i64 [ 0, %for.body3 ], [ %n.vec, %middle.block ]
  br label %for.body11

for.body11:                                       ; preds = %for.body11.preheader, %for.body11
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body11 ], [ %indvars.iv.ph, %for.body11.preheader ]
  %31 = add nuw nsw i64 %14, %indvars.iv
  %arrayidx16 = getelementptr inbounds nuw float, ptr %data_real, i64 %31
  %32 = load float, ptr %arrayidx16, align 4, !tbaa !5
  %arrayidx23 = getelementptr inbounds nuw float, ptr %data_imag, i64 %31
  %33 = load float, ptr %arrayidx23, align 4, !tbaa !5
  %34 = fneg float %33
  %neg = fmul float %11, %34
  %35 = tail call float @llvm.fmuladd.f32(float %10, float %32, float %neg)
  %mul38 = fmul float %10, %33
  %36 = tail call float @llvm.fmuladd.f32(float %11, float %32, float %mul38)
  %37 = add nuw nsw i64 %indvars.iv, %13
  %arrayidx43 = getelementptr inbounds nuw float, ptr %data_real, i64 %37
  %38 = load float, ptr %arrayidx43, align 4, !tbaa !5
  %sub44 = fsub float %38, %35
  store float %sub44, ptr %arrayidx16, align 4, !tbaa !5
  %add56 = fadd float %38, %35
  store float %add56, ptr %arrayidx43, align 4, !tbaa !5
  %arrayidx61 = getelementptr inbounds nuw float, ptr %data_imag, i64 %37
  %39 = load float, ptr %arrayidx61, align 4, !tbaa !5
  %sub62 = fsub float %39, %36
  store float %sub62, ptr %arrayidx23, align 4, !tbaa !5
  %add74 = fadd float %36, %39
  store float %add74, ptr %arrayidx61, align 4, !tbaa !5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.inc75, label %for.body11, !llvm.loop !31

for.inc75:                                        ; preds = %for.body11, %middle.block
  %indvars.iv.next150 = add nuw nsw i64 %indvars.iv149, 1
  %exitcond157.not = icmp eq i64 %indvars.iv.next150, %wide.trip.count156
  br i1 %exitcond157.not, label %for.end77, label %for.body3, !llvm.loop !18

for.end77:                                        ; preds = %for.inc75
  %shl78 = shl i32 %groupsPerStage.0144, 1
  %shr = lshr i32 %buttersPerGroup.0145, 1
  %inc80 = add nuw nsw i32 %i.0143, 1
  %exitcond158.not = icmp eq i32 %inc80, 8
  br i1 %exitcond158.not, label %for.end81, label %for.cond1.preheader, !llvm.loop !19

for.end81:                                        ; preds = %for.end77
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

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
!9 = distinct !{!9, !10, !11}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.unroll.disable"}
!12 = distinct !{!12, !13, !16}
!13 = distinct !{!13, !10, !14, !15}
!14 = !{!"llvm.loop.isvectorized"}
!15 = !{!"llvm.loop.unroll.count", i32 4}
!16 = !{!"llvm.loop.unroll.runtime.disable"}
!17 = distinct !{!17, !13}
!18 = distinct !{!18, !10, !11}
!19 = distinct !{!19, !10, !11}
!20 = !{!21}
!21 = distinct !{!21, !22}
!22 = distinct !{!22, !"LVerDomain"}
!23 = !{!24, !25}
!24 = distinct !{!24, !22}
!25 = distinct !{!25, !22}
!26 = !{!25}
!27 = !{!28}
!28 = distinct !{!28, !22}
!29 = !{!24}
!30 = distinct !{!30, !13, !16}
!31 = distinct !{!31, !13}
