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
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next.1, %vector.body ]
  %0 = getelementptr inbounds nuw [256 x float], ptr @data_imag, i64 0, i64 %index
  %1 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store <4 x float> splat (float 1.000000e+00), ptr %0, align 16, !tbaa !5
  store <4 x float> splat (float 1.000000e+00), ptr %1, align 16, !tbaa !5
  %2 = getelementptr inbounds nuw [256 x float], ptr @coef_real, i64 0, i64 %index
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store <4 x float> splat (float 2.000000e+00), ptr %2, align 16, !tbaa !5
  store <4 x float> splat (float 2.000000e+00), ptr %3, align 16, !tbaa !5
  %4 = getelementptr inbounds nuw [256 x float], ptr @coef_imag, i64 0, i64 %index
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store <4 x float> splat (float 2.000000e+00), ptr %4, align 16, !tbaa !5
  store <4 x float> splat (float 2.000000e+00), ptr %5, align 16, !tbaa !5
  %index.next = or disjoint i64 %index, 8
  %6 = getelementptr inbounds nuw [256 x float], ptr @data_imag, i64 0, i64 %index.next
  %7 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store <4 x float> splat (float 1.000000e+00), ptr %6, align 16, !tbaa !5
  store <4 x float> splat (float 1.000000e+00), ptr %7, align 16, !tbaa !5
  %8 = getelementptr inbounds nuw [256 x float], ptr @coef_real, i64 0, i64 %index.next
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 16
  store <4 x float> splat (float 2.000000e+00), ptr %8, align 16, !tbaa !5
  store <4 x float> splat (float 2.000000e+00), ptr %9, align 16, !tbaa !5
  %10 = getelementptr inbounds nuw [256 x float], ptr @coef_imag, i64 0, i64 %index.next
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 16
  store <4 x float> splat (float 2.000000e+00), ptr %10, align 16, !tbaa !5
  store <4 x float> splat (float 2.000000e+00), ptr %11, align 16, !tbaa !5
  %index.next.1 = add nuw nsw i64 %index, 16
  %12 = icmp eq i64 %index.next.1, 256
  br i1 %12, label %for.body3.lr.ph.i, label %vector.body, !llvm.loop !9

for.body3.lr.ph.i:                                ; preds = %vector.body, %for.end77.i
  %buttersPerGroup.0145.i = phi i32 [ %shr.i, %for.end77.i ], [ 128, %vector.body ]
  %groupsPerStage.0144.i = phi i32 [ %shl78.i, %for.end77.i ], [ 1, %vector.body ]
  %i.0143.i = phi i32 [ %inc80.i, %for.end77.i ], [ 0, %vector.body ]
  %notmask.i = shl nsw i32 -1, %i.0143.i
  %sub.i = xor i32 %notmask.i, -1
  %13 = zext nneg i32 %buttersPerGroup.0145.i to i64
  %14 = zext nneg i32 %sub.i to i64
  %wide.trip.count156.i = zext i32 %groupsPerStage.0144.i to i64
  %min.iters.check = icmp samesign ult i32 %buttersPerGroup.0145.i, 4
  %n.vec = and i64 %13, 252
  %cmp.n = icmp eq i64 %n.vec, %13
  br label %for.body11.lr.ph.i

for.body11.lr.ph.i:                               ; preds = %for.inc75.i, %for.body3.lr.ph.i
  %indvars.iv149.i = phi i64 [ 0, %for.body3.lr.ph.i ], [ %indvars.iv.next150.i, %for.inc75.i ]
  %15 = add nuw nsw i64 %indvars.iv149.i, %14
  %arrayidx.i = getelementptr inbounds nuw float, ptr @coef_real, i64 %15
  %16 = load float, ptr %arrayidx.i, align 4, !tbaa !5
  %arrayidx8.i = getelementptr inbounds nuw float, ptr @coef_imag, i64 %15
  %17 = load float, ptr %arrayidx8.i, align 4, !tbaa !5
  %18 = shl nuw nsw i64 %indvars.iv149.i, 1
  %19 = mul nuw nsw i64 %18, %13
  %20 = add nuw nsw i64 %19, %13
  br i1 %min.iters.check, label %for.body11.i.preheader, label %vector.ph13

vector.ph13:                                      ; preds = %for.body11.lr.ph.i
  %broadcast.splatinsert = insertelement <4 x float> poison, float %17, i64 0
  %broadcast.splat = shufflevector <4 x float> %broadcast.splatinsert, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert17 = insertelement <4 x float> poison, float %16, i64 0
  %broadcast.splat18 = shufflevector <4 x float> %broadcast.splatinsert17, <4 x float> poison, <4 x i32> zeroinitializer
  br label %vector.body14

vector.body14:                                    ; preds = %vector.body14, %vector.ph13
  %index15 = phi i64 [ 0, %vector.ph13 ], [ %index.next21, %vector.body14 ]
  %21 = add nuw nsw i64 %20, %index15
  %22 = getelementptr inbounds nuw float, ptr @data_real, i64 %21
  %wide.load = load <4 x float>, ptr %22, align 4, !tbaa !5
  %23 = getelementptr inbounds nuw float, ptr @data_imag, i64 %21
  %wide.load16 = load <4 x float>, ptr %23, align 4, !tbaa !5
  %24 = fneg <4 x float> %wide.load16
  %25 = fmul <4 x float> %broadcast.splat, %24
  %26 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat18, <4 x float> %wide.load, <4 x float> %25)
  %27 = fmul <4 x float> %broadcast.splat18, %wide.load16
  %28 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat, <4 x float> %wide.load, <4 x float> %27)
  %29 = add nuw nsw i64 %index15, %19
  %30 = getelementptr inbounds nuw float, ptr @data_real, i64 %29
  %wide.load19 = load <4 x float>, ptr %30, align 8, !tbaa !5
  %31 = fsub <4 x float> %wide.load19, %26
  store <4 x float> %31, ptr %22, align 4, !tbaa !5
  %32 = fadd <4 x float> %wide.load19, %26
  store <4 x float> %32, ptr %30, align 8, !tbaa !5
  %33 = getelementptr inbounds nuw float, ptr @data_imag, i64 %29
  %wide.load20 = load <4 x float>, ptr %33, align 8, !tbaa !5
  %34 = fsub <4 x float> %wide.load20, %28
  store <4 x float> %34, ptr %23, align 4, !tbaa !5
  %35 = fadd <4 x float> %28, %wide.load20
  store <4 x float> %35, ptr %33, align 8, !tbaa !5
  %index.next21 = add nuw i64 %index15, 4
  %36 = icmp eq i64 %index.next21, %n.vec
  br i1 %36, label %middle.block11, label %vector.body14, !llvm.loop !13

middle.block11:                                   ; preds = %vector.body14
  br i1 %cmp.n, label %for.inc75.i, label %for.body11.i.preheader

for.body11.i.preheader:                           ; preds = %for.body11.lr.ph.i, %middle.block11
  %indvars.iv.i.ph = phi i64 [ 0, %for.body11.lr.ph.i ], [ %n.vec, %middle.block11 ]
  br label %for.body11.i

for.body11.i:                                     ; preds = %for.body11.i.preheader, %for.body11.i
  %indvars.iv.i = phi i64 [ %indvars.iv.next.i, %for.body11.i ], [ %indvars.iv.i.ph, %for.body11.i.preheader ]
  %37 = add nuw nsw i64 %20, %indvars.iv.i
  %arrayidx16.i = getelementptr inbounds nuw float, ptr @data_real, i64 %37
  %38 = load float, ptr %arrayidx16.i, align 4, !tbaa !5
  %arrayidx23.i = getelementptr inbounds nuw float, ptr @data_imag, i64 %37
  %39 = load float, ptr %arrayidx23.i, align 4, !tbaa !5
  %40 = fneg float %39
  %neg.i = fmul float %17, %40
  %41 = tail call float @llvm.fmuladd.f32(float %16, float %38, float %neg.i)
  %mul38.i = fmul float %16, %39
  %42 = tail call float @llvm.fmuladd.f32(float %17, float %38, float %mul38.i)
  %43 = add nuw nsw i64 %indvars.iv.i, %19
  %arrayidx43.i = getelementptr inbounds nuw float, ptr @data_real, i64 %43
  %44 = load float, ptr %arrayidx43.i, align 4, !tbaa !5
  %sub44.i = fsub float %44, %41
  store float %sub44.i, ptr %arrayidx16.i, align 4, !tbaa !5
  %add56.i = fadd float %44, %41
  store float %add56.i, ptr %arrayidx43.i, align 4, !tbaa !5
  %arrayidx61.i = getelementptr inbounds nuw float, ptr @data_imag, i64 %43
  %45 = load float, ptr %arrayidx61.i, align 4, !tbaa !5
  %sub62.i = fsub float %45, %42
  store float %sub62.i, ptr %arrayidx23.i, align 4, !tbaa !5
  %add74.i = fadd float %42, %45
  store float %add74.i, ptr %arrayidx61.i, align 4, !tbaa !5
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, %13
  br i1 %exitcond.not.i, label %for.inc75.i, label %for.body11.i, !llvm.loop !17

for.inc75.i:                                      ; preds = %for.body11.i, %middle.block11
  %indvars.iv.next150.i = add nuw nsw i64 %indvars.iv149.i, 1
  %exitcond157.not.i = icmp eq i64 %indvars.iv.next150.i, %wide.trip.count156.i
  br i1 %exitcond157.not.i, label %for.end77.i, label %for.body11.lr.ph.i, !llvm.loop !18

for.end77.i:                                      ; preds = %for.inc75.i
  %shl78.i = shl i32 %groupsPerStage.0144.i, 1
  %shr.i = lshr i32 %buttersPerGroup.0145.i, 1
  %inc80.i = add nuw nsw i32 %i.0143.i, 1
  %exitcond158.not.i = icmp eq i32 %inc80.i, 8
  br i1 %exitcond158.not.i, label %_Z6kernelPfS_S_S_.exit, label %for.body3.lr.ph.i, !llvm.loop !19

_Z6kernelPfS_S_S_.exit:                           ; preds = %for.end77.i
  ret i32 0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @_Z6kernelPfS_S_S_(ptr nocapture noundef %data_real, ptr nocapture noundef %data_imag, ptr nocapture noundef readonly %coef_real, ptr nocapture noundef readonly %coef_imag) local_unnamed_addr #1 {
entry:
  br label %for.body3.lr.ph

for.body3.lr.ph:                                  ; preds = %for.end77, %entry
  %buttersPerGroup.0145 = phi i32 [ 128, %entry ], [ %shr, %for.end77 ]
  %groupsPerStage.0144 = phi i32 [ 1, %entry ], [ %shl78, %for.end77 ]
  %i.0143 = phi i32 [ 0, %entry ], [ %inc80, %for.end77 ]
  %notmask = shl nsw i32 -1, %i.0143
  %sub = xor i32 %notmask, -1
  %0 = zext nneg i32 %buttersPerGroup.0145 to i64
  %1 = zext nneg i32 %sub to i64
  %wide.trip.count156 = zext i32 %groupsPerStage.0144 to i64
  %2 = shl nuw nsw i64 %wide.trip.count156, 3
  %3 = add nsw i64 %2, -4
  %4 = mul nsw i64 %3, %0
  %scevgep = getelementptr i8, ptr %data_real, i64 %4
  %scevgep159 = getelementptr i8, ptr %data_imag, i64 %4
  %5 = shl nuw nsw i64 %0, 2
  %scevgep160 = getelementptr nuw i8, ptr %data_imag, i64 %5
  %6 = shl nuw nsw i64 %wide.trip.count156, 3
  %7 = mul nuw nsw i64 %6, %0
  %scevgep161 = getelementptr i8, ptr %data_imag, i64 %7
  %scevgep162 = getelementptr nuw i8, ptr %data_real, i64 %5
  %scevgep163 = getelementptr i8, ptr %data_real, i64 %7
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
  br label %for.body11.lr.ph

for.body11.lr.ph:                                 ; preds = %for.inc75, %for.body3.lr.ph
  %indvars.iv149 = phi i64 [ 0, %for.body3.lr.ph ], [ %indvars.iv.next150, %for.inc75 ]
  %8 = add nuw nsw i64 %indvars.iv149, %1
  %arrayidx = getelementptr inbounds nuw float, ptr %coef_real, i64 %8
  %9 = load float, ptr %arrayidx, align 4, !tbaa !5
  %arrayidx8 = getelementptr inbounds nuw float, ptr %coef_imag, i64 %8
  %10 = load float, ptr %arrayidx8, align 4, !tbaa !5
  %11 = shl nuw nsw i64 %indvars.iv149, 1
  %12 = mul nuw nsw i64 %11, %0
  %13 = add nuw nsw i64 %12, %0
  %brmerge = select i1 %min.iters.check, i1 true, i1 %conflict.rdx174
  br i1 %brmerge, label %for.body11.preheader, label %vector.ph

vector.ph:                                        ; preds = %for.body11.lr.ph
  %broadcast.splatinsert = insertelement <4 x float> poison, float %10, i64 0
  %broadcast.splat = shufflevector <4 x float> %broadcast.splatinsert, <4 x float> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert176 = insertelement <4 x float> poison, float %9, i64 0
  %broadcast.splat177 = shufflevector <4 x float> %broadcast.splatinsert176, <4 x float> poison, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %14 = add nuw nsw i64 %13, %index
  %15 = getelementptr inbounds nuw float, ptr %data_real, i64 %14
  %wide.load = load <4 x float>, ptr %15, align 4, !tbaa !5, !alias.scope !20, !noalias !23
  %16 = getelementptr inbounds nuw float, ptr %data_imag, i64 %14
  %wide.load175 = load <4 x float>, ptr %16, align 4, !tbaa !5, !alias.scope !26
  %17 = fneg <4 x float> %wide.load175
  %18 = fmul <4 x float> %broadcast.splat, %17
  %19 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat177, <4 x float> %wide.load, <4 x float> %18)
  %20 = fmul <4 x float> %broadcast.splat177, %wide.load175
  %21 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %broadcast.splat, <4 x float> %wide.load, <4 x float> %20)
  %22 = add nuw nsw i64 %index, %12
  %23 = getelementptr inbounds nuw float, ptr %data_real, i64 %22
  %wide.load178 = load <4 x float>, ptr %23, align 4, !tbaa !5, !alias.scope !27, !noalias !23
  %24 = fsub <4 x float> %wide.load178, %19
  store <4 x float> %24, ptr %15, align 4, !tbaa !5, !alias.scope !20, !noalias !23
  %25 = fadd <4 x float> %wide.load178, %19
  store <4 x float> %25, ptr %23, align 4, !tbaa !5, !alias.scope !27, !noalias !23
  %26 = getelementptr inbounds nuw float, ptr %data_imag, i64 %22
  %wide.load179 = load <4 x float>, ptr %26, align 4, !tbaa !5, !alias.scope !29
  %27 = fsub <4 x float> %wide.load179, %21
  store <4 x float> %27, ptr %16, align 4, !tbaa !5, !alias.scope !26
  %28 = fadd <4 x float> %21, %wide.load179
  store <4 x float> %28, ptr %26, align 4, !tbaa !5, !alias.scope !29
  %index.next = add nuw i64 %index, 4
  %29 = icmp eq i64 %index.next, %n.vec
  br i1 %29, label %middle.block, label %vector.body, !llvm.loop !30

middle.block:                                     ; preds = %vector.body
  br i1 %cmp.n, label %for.inc75, label %for.body11.preheader

for.body11.preheader:                             ; preds = %for.body11.lr.ph, %middle.block
  %indvars.iv.ph = phi i64 [ 0, %for.body11.lr.ph ], [ %n.vec, %middle.block ]
  br label %for.body11

for.body11:                                       ; preds = %for.body11.preheader, %for.body11
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body11 ], [ %indvars.iv.ph, %for.body11.preheader ]
  %30 = add nuw nsw i64 %13, %indvars.iv
  %arrayidx16 = getelementptr inbounds nuw float, ptr %data_real, i64 %30
  %31 = load float, ptr %arrayidx16, align 4, !tbaa !5
  %arrayidx23 = getelementptr inbounds nuw float, ptr %data_imag, i64 %30
  %32 = load float, ptr %arrayidx23, align 4, !tbaa !5
  %33 = fneg float %32
  %neg = fmul float %10, %33
  %34 = tail call float @llvm.fmuladd.f32(float %9, float %31, float %neg)
  %mul38 = fmul float %9, %32
  %35 = tail call float @llvm.fmuladd.f32(float %10, float %31, float %mul38)
  %36 = add nuw nsw i64 %indvars.iv, %12
  %arrayidx43 = getelementptr inbounds nuw float, ptr %data_real, i64 %36
  %37 = load float, ptr %arrayidx43, align 4, !tbaa !5
  %sub44 = fsub float %37, %34
  store float %sub44, ptr %arrayidx16, align 4, !tbaa !5
  %add56 = fadd float %37, %34
  store float %add56, ptr %arrayidx43, align 4, !tbaa !5
  %arrayidx61 = getelementptr inbounds nuw float, ptr %data_imag, i64 %36
  %38 = load float, ptr %arrayidx61, align 4, !tbaa !5
  %sub62 = fsub float %38, %35
  store float %sub62, ptr %arrayidx23, align 4, !tbaa !5
  %add74 = fadd float %35, %38
  store float %add74, ptr %arrayidx61, align 4, !tbaa !5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %0
  br i1 %exitcond.not, label %for.inc75, label %for.body11, !llvm.loop !31

for.inc75:                                        ; preds = %for.body11, %middle.block
  %indvars.iv.next150 = add nuw nsw i64 %indvars.iv149, 1
  %exitcond157.not = icmp eq i64 %indvars.iv.next150, %wide.trip.count156
  br i1 %exitcond157.not, label %for.end77, label %for.body11.lr.ph, !llvm.loop !18

for.end77:                                        ; preds = %for.inc75
  %shl78 = shl i32 %groupsPerStage.0144, 1
  %shr = lshr i32 %buttersPerGroup.0145, 1
  %inc80 = add nuw nsw i32 %i.0143, 1
  %exitcond158.not = icmp eq i32 %inc80, 8
  br i1 %exitcond158.not, label %for.end81, label %for.body3.lr.ph, !llvm.loop !19

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
!9 = distinct !{!9, !10, !11, !12}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.isvectorized", i32 1}
!12 = !{!"llvm.loop.unroll.runtime.disable"}
!13 = distinct !{!13, !14, !12}
!14 = distinct !{!14, !10, !15, !16}
!15 = !{!"llvm.loop.isvectorized"}
!16 = !{!"llvm.loop.unroll.count", i32 4}
!17 = distinct !{!17, !14}
!18 = distinct !{!18, !10}
!19 = distinct !{!19, !10}
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
!30 = distinct !{!30, !14, !12}
!31 = distinct !{!31, !14}
