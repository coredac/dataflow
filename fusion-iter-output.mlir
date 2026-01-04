[DEBUG] Recurrence cycle (length 4):
  %61 = neura.reserve : !neura.data<i64, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %106 = "neura.data_mov"(%97#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %106 -> %61 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 4):
  %61 = neura.reserve : !neura.data<i64, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %106 = "neura.data_mov"(%97#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %106 -> %61 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 3):
  %61 = neura.reserve : !neura.data<i64, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %96 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %106 = "neura.data_mov"(%97#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %106 -> %61 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 2):
  %55 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %57 = "neura.fused_op"(%56, %9, %55) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %60 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %115 = neura.grant_predicate %60, %100 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %115 -> %55 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 2):
  %49 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %54 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %116 = neura.grant_predicate %54, %101 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %116 -> %49 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 5):
  %49 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %52 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %101 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %116 = neura.grant_predicate %54, %101 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %116 -> %49 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 2):
  %43 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %48 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %117 = neura.grant_predicate %48, %102 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %117 -> %43 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 6):
  %43 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %102 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %117 = neura.grant_predicate %48, %102 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %117 -> %43 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 6):
  %43 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %102 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %117 = neura.grant_predicate %48, %102 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %117 -> %43 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 5):
  %43 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %96 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %102 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %117 = neura.grant_predicate %48, %102 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %117 -> %43 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 2):
  %38 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %40 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %118 = neura.grant_predicate %40, %103 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %118 -> %38 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %38 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %41 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %103 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %118 = neura.grant_predicate %40, %103 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %118 -> %38 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 4):
  %38 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %42 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %103 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %118 = neura.grant_predicate %40, %103 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %118 -> %38 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 4):
  %38 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %41 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %96 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %103 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %118 = neura.grant_predicate %40, %103 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %118 -> %38 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 2):
  %33 = neura.reserve : !neura.data<i64, i1>
  %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %36 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %119 = neura.grant_predicate %36, %104 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %119 -> %33 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 4):
  %33 = neura.reserve : !neura.data<i64, i1>
  %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %37 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %104 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %119 = neura.grant_predicate %36, %104 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %119 -> %33 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 2):
  %32 = neura.reserve : !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %68 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %120 = neura.grant_predicate %68, %105 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %120 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 4):
  %32 = neura.reserve : !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %105 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %120 = neura.grant_predicate %68, %105 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %120 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 1):
  %31 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %112:2 = "neura.fused_op"(%22, %31, %99) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
  %114 = "neura.data_mov"(%112#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %114 -> %31 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 1):
  %30 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %109:2 = "neura.fused_op"(%26, %30, %98) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
  %111 = "neura.data_mov"(%109#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %111 -> %30 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 1):
  %29 = neura.reserve : !neura.data<i64, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %107 = "neura.data_mov"(%97#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %107 -> %29 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %17 = neura.reserve : !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %138 = "neura.data_mov"(%129#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %138 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %17 = neura.reserve : !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %138 = "neura.data_mov"(%129#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %138 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %17 = neura.reserve : !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %138 = "neura.data_mov"(%129#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %138 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %17 = neura.reserve : !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %138 = "neura.data_mov"(%129#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %138 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %17 = neura.reserve : !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %138 = "neura.data_mov"(%129#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %138 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %17 = neura.reserve : !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %138 = "neura.data_mov"(%129#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %138 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %17 = neura.reserve : !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %138 = "neura.data_mov"(%129#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %138 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %17 = neura.reserve : !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %138 = "neura.data_mov"(%129#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %138 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %17 = neura.reserve : !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %138 = "neura.data_mov"(%129#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %138 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %17 = neura.reserve : !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %138 = "neura.data_mov"(%129#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %138 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %17 = neura.reserve : !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %138 = "neura.data_mov"(%129#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %138 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %17 = neura.reserve : !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %138 = "neura.data_mov"(%129#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %138 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 3):
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %22 = "neura.data_mov"(%19#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %112:2 = "neura.fused_op"(%22, %31, %99) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
  %113 = "neura.data_mov"(%112#0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 6):
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %99 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %112:2 = "neura.fused_op"(%22, %31, %99) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
  %113 = "neura.data_mov"(%112#0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 6):
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %99 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %112:2 = "neura.fused_op"(%22, %31, %99) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
  %113 = "neura.data_mov"(%112#0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 4):
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %88 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 4):
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %88 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 7):
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %132 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 7):
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %132 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 7):
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %132 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 7):
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %132 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 7):
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %132 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 7):
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %132 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 6):
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %132 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 6):
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %132 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 6):
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %132 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 6):
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %132 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 6):
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %132 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 6):
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %132 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 3):
  %15 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %25:3 = "neura.fused_op"(%24, %15, %20) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.gep"(%152, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %154 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %26 = "neura.data_mov"(%25#0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %109:2 = "neura.fused_op"(%26, %30, %98) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
  %110 = "neura.data_mov"(%109#0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %145 = "neura.fused_op"(%110, %89, %133) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %146 = "neura.data_mov"(%145) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %146 -> %15 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 7):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %14 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 7):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %12 = "neura.data_mov"(%11#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 7):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %14 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 7):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %12 = "neura.data_mov"(%11#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 7):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %14 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 7):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %12 = "neura.data_mov"(%11#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 7):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %14 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 7):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %12 = "neura.data_mov"(%11#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 7):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %14 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 7):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %12 = "neura.data_mov"(%11#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 7):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %14 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 7):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %12 = "neura.data_mov"(%11#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %14 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %12 = "neura.data_mov"(%11#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %14 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %12 = "neura.data_mov"(%11#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %14 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %96 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 3):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %13 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %14 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %12 = "neura.data_mov"(%11#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %14 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %12 = "neura.data_mov"(%11#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %14 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %12 = "neura.data_mov"(%11#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %14 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %10 = neura.reserve : !neura.data<i64, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %12 = "neura.data_mov"(%11#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 2):
  %9 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %57 = "neura.fused_op"(%56, %9, %55) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %59 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %143 = "neura.fused_op"(%59, %90, %134) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %144 = "neura.data_mov"(%143) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %144 -> %9 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 2):
  %8 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %53 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %142 = "neura.data_mov"(%141) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %142 -> %8 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 4):
  %8 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %52 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %91 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %142 = "neura.data_mov"(%141) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %142 -> %8 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 7):
  %8 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %52 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %135 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %142 = "neura.data_mov"(%141) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %142 -> %8 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 7):
  %8 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %52 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %135 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %142 = "neura.data_mov"(%141) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %142 -> %8 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 7):
  %8 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %52 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %135 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %142 = "neura.data_mov"(%141) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %142 -> %8 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 6):
  %8 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %52 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %135 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %142 = "neura.data_mov"(%141) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %142 -> %8 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 6):
  %8 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %52 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %135 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %142 = "neura.data_mov"(%141) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %142 -> %8 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 6):
  %8 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %52 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %135 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %142 = "neura.data_mov"(%141) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %142 -> %8 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 2):
  %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %47 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 5):
  %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %92 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 5):
  %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %92 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 8):
  %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %136 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 8):
  %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %136 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 8):
  %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %136 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 8):
  %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %136 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 8):
  %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %136 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 8):
  %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %136 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 7):
  %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %136 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 7):
  %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %136 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 6):
  %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %96 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %136 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 7):
  %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %136 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 7):
  %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %136 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 7):
  %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %136 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 7):
  %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %136 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 4):
  %6 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %41 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %94 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 3):
  %6 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %42 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %94 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 7):
  %6 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %41 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %131 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %6 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %42 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %131 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 7):
  %6 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %41 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %131 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %6 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %42 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %131 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 7):
  %6 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %41 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %131 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %6 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %42 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %131 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %6 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %41 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %131 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %6 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %42 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %131 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %6 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %41 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %96 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %131 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %6 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %41 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %131 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %6 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %42 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %131 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %6 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %41 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %131 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %6 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %42 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %131 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 3):
  %5 = neura.reserve : !neura.data<i64, i1>
  %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %35 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %150 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %151 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 4):
  %5 = neura.reserve : !neura.data<i64, i1>
  %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %37 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %150 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %151 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %5 = neura.reserve : !neura.data<i64, i1>
  %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %37 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %130 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %151 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %5 = neura.reserve : !neura.data<i64, i1>
  %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %37 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %130 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %151 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %5 = neura.reserve : !neura.data<i64, i1>
  %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %35 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %130 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %151 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %5 = neura.reserve : !neura.data<i64, i1>
  %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %37 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %130 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %151 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %5 = neura.reserve : !neura.data<i64, i1>
  %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %37 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %130 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %151 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %5 = neura.reserve : !neura.data<i64, i1>
  %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %37 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %130 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %151 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %5 = neura.reserve : !neura.data<i64, i1>
  %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %37 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %130 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %151 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Longest recurrence cycle (length 8):
%7 = neura.reserve : !neura.data<!llvm.ptr, i1>
%45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
%46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
%62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
%64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
%83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
%84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
%85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
%93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
%122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
%123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
%125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
%126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
%129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
%136 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
%139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
%140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = "neura.gep"(%152, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = "neura.load"(%152) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %153 : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %153 : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %153 : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %153 : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %153 : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %153 : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %153 : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %153 : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %152, %153, %154 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %154 : !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %154 : !neura.data<i64, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Skipping op inside fused_op: neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %5 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %6 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %8 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %9 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %10 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %15 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %17 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %29 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %30 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %31 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %32 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %33 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %38 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %43 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %49 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %55 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %61 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.yield
[MapToAcceleratorPass] Topologically sorted op: %50 = "neura.data_mov"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %56 = "neura.data_mov"(%1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %24 = "neura.data_mov"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %44 = "neura.data_mov"(%3) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %18 = "neura.data_mov"(%4) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
[MapToAcceleratorPass] Topologically sorted op: %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %57 = "neura.fused_op"(%56, %9, %55) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %12 = "neura.data_mov"(%11#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %14 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %13 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %37 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %36 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %35 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %42 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %41 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %40 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %54 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %53 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %52 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %60 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %59 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %58 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %48 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %47 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[MapToAcceleratorPass] Topologically sorted op: %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
[MapToAcceleratorPass] Topologically sorted op: %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %20 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %22 = "neura.data_mov"(%19#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %65 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %66 = "neura.data_mov"(%62#1) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %25:3 = "neura.fused_op"(%24, %15, %20) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.gep"(%152, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %154 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[MapToAcceleratorPass] Topologically sorted op: %72:2 = "neura.fused_op"(%58, %65) <{frequency = 6 : i64, pattern_id = 0 : i64, pattern_name = "gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.load"(%152) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[MapToAcceleratorPass] Topologically sorted op: %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[MapToAcceleratorPass] Topologically sorted op: %26 = "neura.data_mov"(%25#0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %27 = "neura.data_mov"(%25#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %28 = "neura.data_mov"(%25#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %73 = "neura.data_mov"(%72#0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %74 = "neura.data_mov"(%72#1) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %96 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %68 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %70 = "neura.data_mov"(%67#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %71 = "neura.data_mov"(%67#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
[MapToAcceleratorPass] Topologically sorted op: %77 = "neura.load"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %75 = "neura.add"(%71, %74) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %92 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %91 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %90 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %89 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %88 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %94 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %78 = "neura.data_mov"(%77) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %76 = "neura.data_mov"(%75) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
[MapToAcceleratorPass] Topologically sorted op: %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %79 = "neura.mul"(%66, %78) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: "neura.store"(%76, %73) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
[MapToAcceleratorPass] Topologically sorted op: %150 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %105 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %104 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %103 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %102 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %101 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %100 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %99 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %98 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %106 = "neura.data_mov"(%97#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %107 = "neura.data_mov"(%97#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %80 = "neura.data_mov"(%79) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %120 = neura.grant_predicate %68, %105 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %119 = neura.grant_predicate %36, %104 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %118 = neura.grant_predicate %40, %103 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %117 = neura.grant_predicate %48, %102 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %116 = neura.grant_predicate %54, %101 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %115 = neura.grant_predicate %60, %100 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %112:2 = "neura.fused_op"(%22, %31, %99) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
[MapToAcceleratorPass] Topologically sorted op: %109:2 = "neura.fused_op"(%26, %30, %98) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %106 -> %61 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %107 -> %29 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
[MapToAcceleratorPass] Topologically sorted op: %81 = "neura.add"(%80, %28) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %120 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %119 -> %33 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %118 -> %38 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %117 -> %43 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %116 -> %49 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %115 -> %55 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %113 = "neura.data_mov"(%112#0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %114 = "neura.data_mov"(%112#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %110 = "neura.data_mov"(%109#0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %111 = "neura.data_mov"(%109#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %127 = "neura.data_mov"(%125#1) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %82 = "neura.data_mov"(%81) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %114 -> %31 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %111 -> %30 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
[MapToAcceleratorPass] Topologically sorted op: neura.return_void %127 : !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: "neura.store"(%82, %27) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
[MapToAcceleratorPass] Topologically sorted op: %136 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %135 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %134 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %133 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %132 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %131 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %130 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %138 = "neura.data_mov"(%129#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %143 = "neura.fused_op"(%59, %90, %134) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %145 = "neura.fused_op"(%110, %89, %133) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %138 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %142 = "neura.data_mov"(%141) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %144 = "neura.data_mov"(%143) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %146 = "neura.data_mov"(%145) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %151 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %142 -> %8 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %144 -> %9 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %146 -> %15 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] ALAP Bucket Level 0: 5 ops
  %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
  %7 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %10 = neura.reserve : !neura.data<i64, i1>
  %43 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %44 = "neura.data_mov"(%3) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] ALAP Bucket Level 1: 19 ops
  %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
  %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
  %6 = neura.reserve : !neura.data<i64, i1>
  %8 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %17 = neura.reserve : !neura.data<i64, i1>
  %38 = neura.reserve : !neura.data<i64, i1>
  %49 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %61 = neura.reserve : !neura.data<i64, i1>
  %50 = "neura.data_mov"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %18 = "neura.data_mov"(%4) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
  %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %12 = "neura.data_mov"(%11#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %14 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %13 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %48 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %47 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] ALAP Bucket Level 2: 18 ops
  %5 = neura.reserve : !neura.data<i64, i1>
  %32 = neura.reserve : !neura.data<i64, i1>
  %33 = neura.reserve : !neura.data<i64, i1>
  %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %42 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %41 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %40 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %54 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %53 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %52 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
  %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %22 = "neura.data_mov"(%19#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 3: 10 ops
  %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %37 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %36 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %35 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %96 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %68 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 4: 14 ops
  %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
  %9 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %55 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %56 = "neura.data_mov"(%1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %70 = "neura.data_mov"(%67#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %92 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %91 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %88 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %94 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 5: 19 ops
  %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
  %15 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %29 = neura.reserve : !neura.data<i64, i1>
  %24 = "neura.data_mov"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %57 = "neura.fused_op"(%56, %9, %55) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %60 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %59 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %58 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %20 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %65 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %66 = "neura.data_mov"(%62#1) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %77 = "neura.load"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %78 = "neura.data_mov"(%77) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %150 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 6: 25 ops
  %30 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %31 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %25:3 = "neura.fused_op"(%24, %15, %20) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.gep"(%152, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %154 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %72:2 = "neura.fused_op"(%58, %65) <{frequency = 6 : i64, pattern_id = 0 : i64, pattern_name = "gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.load"(%152) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
  %26 = "neura.data_mov"(%25#0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %28 = "neura.data_mov"(%25#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %74 = "neura.data_mov"(%72#1) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %71 = "neura.data_mov"(%67#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %79 = "neura.mul"(%66, %78) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  %105 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %104 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %103 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %102 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %101 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %99 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %98 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %106 = "neura.data_mov"(%97#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %107 = "neura.data_mov"(%97#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %80 = "neura.data_mov"(%79) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.ctrl_mov %106 -> %61 : !neura.data<i64, i1> !neura.data<i64, i1>
  neura.ctrl_mov %107 -> %29 : !neura.data<i64, i1> !neura.data<i64, i1>
  %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
  %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] ALAP Bucket Level 7: 40 ops
  %27 = "neura.data_mov"(%25#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %73 = "neura.data_mov"(%72#0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %75 = "neura.add"(%71, %74) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  %90 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %89 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %76 = "neura.data_mov"(%75) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %100 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %120 = neura.grant_predicate %68, %105 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %119 = neura.grant_predicate %36, %104 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %118 = neura.grant_predicate %40, %103 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %117 = neura.grant_predicate %48, %102 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %116 = neura.grant_predicate %54, %101 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %112:2 = "neura.fused_op"(%22, %31, %99) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
  %109:2 = "neura.fused_op"(%26, %30, %98) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
  %81 = "neura.add"(%80, %28) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.ctrl_mov %120 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1>
  neura.ctrl_mov %119 -> %33 : !neura.data<i64, i1> !neura.data<i64, i1>
  neura.ctrl_mov %118 -> %38 : !neura.data<i64, i1> !neura.data<i64, i1>
  neura.ctrl_mov %117 -> %43 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %116 -> %49 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  %113 = "neura.data_mov"(%112#0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %114 = "neura.data_mov"(%112#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %110 = "neura.data_mov"(%109#0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %111 = "neura.data_mov"(%109#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %127 = "neura.data_mov"(%125#1) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %82 = "neura.data_mov"(%81) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.ctrl_mov %114 -> %31 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %111 -> %30 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
  %136 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %135 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %134 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %133 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %132 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %131 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %130 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %138 = "neura.data_mov"(%129#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
  neura.ctrl_mov %138 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 8: 25 ops
  neura.yield
  "neura.store"(%76, %73) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
  %115 = neura.grant_predicate %60, %100 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %115 -> %55 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  neura.return_void %127 : !neura.data<i1, i1>
  "neura.store"(%82, %27) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
  %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %143 = "neura.fused_op"(%59, %90, %134) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %145 = "neura.fused_op"(%110, %89, %133) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
  %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %142 = "neura.data_mov"(%141) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %144 = "neura.data_mov"(%143) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %146 = "neura.data_mov"(%145) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1>
  neura.ctrl_mov %151 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
  neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %142 -> %8 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %144 -> %9 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %146 -> %15 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] ALAP sorted op: %7 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %10 = neura.reserve : !neura.data<i64, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %43 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %44 = "neura.data_mov"(%3) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>) (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %6 = neura.reserve : !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %8 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %16 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %17 = neura.reserve : !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %38 = neura.reserve : !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %49 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %61 = neura.reserve : !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %50 = "neura.data_mov"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %18 = "neura.data_mov"(%4) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %12 = "neura.data_mov"(%11#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %14 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %13 = "neura.data_mov"(%11#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %48 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %47 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>) (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %5 = neura.reserve : !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %32 = neura.reserve : !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %33 = neura.reserve : !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %42 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %41 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %40 = "neura.data_mov"(%39) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %54 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %53 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %52 = "neura.data_mov"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %21 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %22 = "neura.data_mov"(%19#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %23 = "neura.data_mov"(%19#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %64 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %63 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %37 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %36 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %35 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %96 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %84 = "neura.data_mov"(%83) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %69 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %68 = "neura.data_mov"(%67#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %9 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %55 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %56 = "neura.data_mov"(%1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %70 = "neura.data_mov"(%67#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %92 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %91 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %88 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %87 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %86 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %93 = "neura.data_mov"(%85#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %95 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %94 = "neura.data_mov"(%85#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %57 = "neura.fused_op"(%56, %9, %55) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %15 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %29 = neura.reserve : !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %24 = "neura.data_mov"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %60 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %59 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %58 = "neura.data_mov"(%57) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %20 = "neura.data_mov"(%19#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %65 = "neura.data_mov"(%62#0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %66 = "neura.data_mov"(%62#1) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %77 = "neura.load"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %78 = "neura.data_mov"(%77) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %150 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %124 = "neura.data_mov"(%121) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %128 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %123 = "neura.data_mov"(%122) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %25:3 = "neura.fused_op"(%24, %15, %20) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.gep"(%152, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %154 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %72:2 = "neura.fused_op"(%58, %65) <{frequency = 6 : i64, pattern_id = 0 : i64, pattern_name = "gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.load"(%152) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>) (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %79 = "neura.mul"(%66, %78) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %30 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %31 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %26 = "neura.data_mov"(%25#0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %28 = "neura.data_mov"(%25#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %74 = "neura.data_mov"(%72#1) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %71 = "neura.data_mov"(%67#2) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %105 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %104 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %103 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %102 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %101 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %99 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %98 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %106 = "neura.data_mov"(%97#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %107 = "neura.data_mov"(%97#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %108 = "neura.data_mov"(%97#3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %80 = "neura.data_mov"(%79) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %106 -> %61 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %107 -> %29 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %126 = "neura.data_mov"(%125#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %112:2 = "neura.fused_op"(%22, %31, %99) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %109:2 = "neura.fused_op"(%26, %30, %98) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %75 = "neura.add"(%71, %74) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %120 = neura.grant_predicate %68, %105 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %119 = neura.grant_predicate %36, %104 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %118 = neura.grant_predicate %40, %103 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %117 = neura.grant_predicate %48, %102 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %116 = neura.grant_predicate %54, %101 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %81 = "neura.add"(%80, %28) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %27 = "neura.data_mov"(%25#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %73 = "neura.data_mov"(%72#0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %90 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %89 = "neura.data_mov"(%85#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %76 = "neura.data_mov"(%75) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %100 = "neura.data_mov"(%97#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %120 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %119 -> %33 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %118 -> %38 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %117 -> %43 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %116 -> %49 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %113 = "neura.data_mov"(%112#0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %114 = "neura.data_mov"(%112#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %110 = "neura.data_mov"(%109#0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %111 = "neura.data_mov"(%109#1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %127 = "neura.data_mov"(%125#1) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %82 = "neura.data_mov"(%81) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %114 -> %31 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %111 -> %30 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %136 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %135 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %134 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %133 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %132 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %131 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %130 = "neura.data_mov"(%129#0) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %137 = "neura.data_mov"(%129#1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %138 = "neura.data_mov"(%129#2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %137 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %138 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %143 = "neura.fused_op"(%59, %90, %134) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %145 = "neura.fused_op"(%110, %89, %133) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %115 = neura.grant_predicate %60, %100 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: "neura.store"(%76, %73) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %115 -> %55 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: "neura.store"(%82, %27) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %140 = "neura.data_mov"(%139) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %142 = "neura.data_mov"(%141) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %144 = "neura.data_mov"(%143) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %146 = "neura.data_mov"(%145) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %148 = "neura.data_mov"(%147) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %149 -> %6 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %151 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %140 -> %7 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %142 -> %8 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %144 -> %9 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %146 -> %15 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %148 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: neura.return_void %127 : !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: neura.yield (ALAP level: 8)
---------------------------------------------------------
[HeuristicMapping] Starting mapping with 175 operations.
Configuration: MAX Backtrack Depth = 1, MAX Candidate Locations = 1
[HeuristicMapping] Filtered 130 non-materialized operations, 45 operations require physical mapping.
[HeuristicMapping] Materialized operations list:
0 %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1> (level: 0)
1 %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (level: 1)
2 %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>) (level: 1)
3 %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1> (level: 1)
4 %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1> (level: 1)
5 %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 2)
6 %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>) (level: 2)
7 %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (level: 2)
8 %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 2)
9 %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 3)
10 %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 3)
11 %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 3)
12 %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (level: 4)
13 %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1> (level: 4)
14 %57 = "neura.fused_op"(%56, %9, %55) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (level: 5)
15 %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
16 %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 5)
17 %77 = "neura.load"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 5)
18 %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1> (level: 5)
19 %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (level: 6)
20 %25:3 = "neura.fused_op"(%24, %15, %20) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.gep"(%152, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %154 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 6)
21 %72:2 = "neura.fused_op"(%58, %65) <{frequency = 6 : i64, pattern_id = 0 : i64, pattern_name = "gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.load"(%152) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 6)
22 %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>) (level: 6)
23 %79 = "neura.mul"(%66, %78) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 6)
24 %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (level: 7)
25 %112:2 = "neura.fused_op"(%22, %31, %99) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) (level: 7)
26 %109:2 = "neura.fused_op"(%26, %30, %98) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) (level: 7)
27 %75 = "neura.add"(%71, %74) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 7)
28 %120 = neura.grant_predicate %68, %105 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 7)
29 %119 = neura.grant_predicate %36, %104 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 7)
30 %118 = neura.grant_predicate %40, %103 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 7)
31 %117 = neura.grant_predicate %48, %102 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
32 %116 = neura.grant_predicate %54, %101 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
33 %81 = "neura.add"(%80, %28) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 7)
34 %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
35 %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
36 %143 = "neura.fused_op"(%59, %90, %134) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
37 %145 = "neura.fused_op"(%110, %89, %133) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
38 %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
39 %115 = neura.grant_predicate %60, %100 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
40 %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 8)
41 %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 8)
42 "neura.store"(%76, %73) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 8)
43 "neura.store"(%82, %27) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 8)
44 neura.return_void %127 : !neura.data<i1, i1> (level: 8)
[HeuristicMapping] Found 128 candidate locations for operation: %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=0
[HeuristicMapping] Successfully mapped operation %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 110 candidate locations for operation: %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=1
[tryRouteDataMove] Routing from Tile#10 @t=0 to Tile#10 @t=1
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 126 candidate locations for operation: %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=1
[HeuristicMapping] Successfully mapped operation %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] Found 125 candidate locations for operation: %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=1
[HeuristicMapping] Successfully mapped operation %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 124 candidate locations for operation: %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=1
[HeuristicMapping] Successfully mapped operation %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 95 candidate locations for operation: %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=2
[tryRouteDataMove] Routing from Tile#5 @t=1 to Tile#5 @t=2
[tryRouteDataMove] Successfully routed on same tile using Register #160
[tryRouteDataMove] Routing from Tile#4 @t=1 to Tile#5 @t=2
[HeuristicMapping] Successfully mapped operation %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Found 99 candidate locations for operation: %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=2
[tryRouteDataMove] Routing from Tile#5 @t=1 to Tile#9 @t=2
[tryRouteDataMove] Routing from Tile#10 @t=1 to Tile#9 @t=2
[HeuristicMapping] Successfully mapped operation %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Found 104 candidate locations for operation: %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=2
[tryRouteDataMove] Routing from Tile#9 @t=1 to Tile#10 @t=2
[HeuristicMapping] Successfully mapped operation %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 120 candidate locations for operation: %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=2
[HeuristicMapping] Successfully mapped operation %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 91 candidate locations for operation: %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=3
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#9 @t=3
[tryRouteDataMove] Routing from Tile#10 @t=2 to Tile#9 @t=3
[tryRouteDataMove] Routing from Tile#9 @t=2 to Tile#9 @t=3
[tryRouteDataMove] Successfully routed on same tile using Register #288
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#9 @t=3
[tryRouteDataMove] Cannot find routing path from Tile#5 @t=2 to Tile#9 @t=3
[HeuristicMapping] Failed to map operation %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) to candidate location 1/1
[HeuristicMapping] Found 91 candidate locations for operation: %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] All 1 locations for 9 tried, backtracking...
[HeuristicMapping] Backtracking to operation 8 (depth = 1).
[HeuristicMapping] Found 120 candidate locations for operation: %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] All 1 locations for 8 tried, backtracking...
[HeuristicMapping] Backtracking to operation 7 (depth = 2).
[HeuristicMapping] Max backtrack depth exceeded: 2 > 1.
---------------------------------------------------------
[HeuristicMapping] Starting mapping with 175 operations.
Configuration: MAX Backtrack Depth = 1, MAX Candidate Locations = 1
[HeuristicMapping] Filtered 130 non-materialized operations, 45 operations require physical mapping.
[HeuristicMapping] Materialized operations list:
0 %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1> (level: 0)
1 %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (level: 1)
2 %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>) (level: 1)
3 %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1> (level: 1)
4 %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1> (level: 1)
5 %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 2)
6 %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>) (level: 2)
7 %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (level: 2)
8 %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 2)
9 %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 3)
10 %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 3)
11 %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 3)
12 %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (level: 4)
13 %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1> (level: 4)
14 %57 = "neura.fused_op"(%56, %9, %55) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (level: 5)
15 %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
16 %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 5)
17 %77 = "neura.load"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 5)
18 %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1> (level: 5)
19 %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (level: 6)
20 %25:3 = "neura.fused_op"(%24, %15, %20) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.gep"(%152, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %154 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 6)
21 %72:2 = "neura.fused_op"(%58, %65) <{frequency = 6 : i64, pattern_id = 0 : i64, pattern_name = "gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.load"(%152) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 6)
22 %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>) (level: 6)
23 %79 = "neura.mul"(%66, %78) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 6)
24 %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (level: 7)
25 %112:2 = "neura.fused_op"(%22, %31, %99) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) (level: 7)
26 %109:2 = "neura.fused_op"(%26, %30, %98) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) (level: 7)
27 %75 = "neura.add"(%71, %74) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 7)
28 %120 = neura.grant_predicate %68, %105 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 7)
29 %119 = neura.grant_predicate %36, %104 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 7)
30 %118 = neura.grant_predicate %40, %103 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 7)
31 %117 = neura.grant_predicate %48, %102 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
32 %116 = neura.grant_predicate %54, %101 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
33 %81 = "neura.add"(%80, %28) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 7)
34 %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
35 %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
36 %143 = "neura.fused_op"(%59, %90, %134) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
37 %145 = "neura.fused_op"(%110, %89, %133) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
38 %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
39 %115 = neura.grant_predicate %60, %100 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
40 %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 8)
41 %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 8)
42 "neura.store"(%76, %73) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 8)
43 "neura.store"(%82, %27) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 8)
44 neura.return_void %127 : !neura.data<i1, i1> (level: 8)
[HeuristicMapping] Found 144 candidate locations for operation: %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=0
[HeuristicMapping] Successfully mapped operation %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 126 candidate locations for operation: %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=1
[tryRouteDataMove] Routing from Tile#10 @t=0 to Tile#10 @t=1
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 142 candidate locations for operation: %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=1
[HeuristicMapping] Successfully mapped operation %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] Found 141 candidate locations for operation: %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=1
[HeuristicMapping] Successfully mapped operation %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 140 candidate locations for operation: %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=1
[HeuristicMapping] Successfully mapped operation %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 100 candidate locations for operation: %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=3
[tryRouteDataMove] Routing from Tile#6 @t=1 to Tile#5 @t=3
[tryRouteDataMove] Routing from Tile#0 @t=1 to Tile#5 @t=3
[HeuristicMapping] Successfully mapped operation %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Found 114 candidate locations for operation: %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=2
[tryRouteDataMove] Routing from Tile#6 @t=1 to Tile#6 @t=2
[tryRouteDataMove] Successfully routed on same tile using Register #192
[tryRouteDataMove] Routing from Tile#10 @t=1 to Tile#6 @t=2
[HeuristicMapping] Successfully mapped operation %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Found 120 candidate locations for operation: %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=2
[tryRouteDataMove] Routing from Tile#5 @t=1 to Tile#5 @t=2
[tryRouteDataMove] Successfully routed on same tile using Register #160
[HeuristicMapping] Successfully mapped operation %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 136 candidate locations for operation: %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=2
[HeuristicMapping] Successfully mapped operation %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 118 candidate locations for operation: %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=4
[tryRouteDataMove] Routing from Tile#5 @t=3 to Tile#5 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #160
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#5 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #161
[tryRouteDataMove] Routing from Tile#6 @t=2 to Tile#5 @t=4
[tryRouteDataMove] Routing from Tile#5 @t=3 to Tile#5 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #163
[HeuristicMapping] Successfully mapped operation %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Found 134 candidate locations for operation: %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=3
[HeuristicMapping] Successfully mapped operation %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 109 candidate locations for operation: %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=3
[tryRouteDataMove] Routing from Tile#6 @t=2 to Tile#10 @t=3
[tryRouteDataMove] Routing from Tile#9 @t=2 to Tile#10 @t=3
[HeuristicMapping] Successfully mapped operation %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 112 candidate locations for operation: %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=5
[tryRouteDataMove] Routing from Tile#10 @t=3 to Tile#9 @t=5
[tryRouteDataMove] Routing from Tile#6 @t=3 to Tile#9 @t=5
[tryRouteDataMove] Routing from Tile#5 @t=4 to Tile#9 @t=5
[tryRouteDataMove] Cannot find routing path from Tile#5 @t=4 to Tile#9 @t=5
[HeuristicMapping] Failed to map operation %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) to candidate location 1/1
[HeuristicMapping] Found 112 candidate locations for operation: %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] All 1 locations for 12 tried, backtracking...
[HeuristicMapping] Backtracking to operation 11 (depth = 1).
[HeuristicMapping] Found 109 candidate locations for operation: %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] All 1 locations for 11 tried, backtracking...
[HeuristicMapping] Backtracking to operation 10 (depth = 2).
[HeuristicMapping] Max backtrack depth exceeded: 2 > 1.
---------------------------------------------------------
[HeuristicMapping] Starting mapping with 175 operations.
Configuration: MAX Backtrack Depth = 1, MAX Candidate Locations = 1
[HeuristicMapping] Filtered 130 non-materialized operations, 45 operations require physical mapping.
[HeuristicMapping] Materialized operations list:
0 %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1> (level: 0)
1 %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (level: 1)
2 %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>) (level: 1)
3 %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1> (level: 1)
4 %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1> (level: 1)
5 %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 2)
6 %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>) (level: 2)
7 %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (level: 2)
8 %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 2)
9 %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 3)
10 %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 3)
11 %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 3)
12 %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (level: 4)
13 %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1> (level: 4)
14 %57 = "neura.fused_op"(%56, %9, %55) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (level: 5)
15 %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
16 %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 5)
17 %77 = "neura.load"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 5)
18 %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1> (level: 5)
19 %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (level: 6)
20 %25:3 = "neura.fused_op"(%24, %15, %20) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.gep"(%152, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %154 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 6)
21 %72:2 = "neura.fused_op"(%58, %65) <{frequency = 6 : i64, pattern_id = 0 : i64, pattern_name = "gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.load"(%152) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 6)
22 %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>) (level: 6)
23 %79 = "neura.mul"(%66, %78) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 6)
24 %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (level: 7)
25 %112:2 = "neura.fused_op"(%22, %31, %99) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) (level: 7)
26 %109:2 = "neura.fused_op"(%26, %30, %98) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) (level: 7)
27 %75 = "neura.add"(%71, %74) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 7)
28 %120 = neura.grant_predicate %68, %105 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 7)
29 %119 = neura.grant_predicate %36, %104 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 7)
30 %118 = neura.grant_predicate %40, %103 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 7)
31 %117 = neura.grant_predicate %48, %102 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
32 %116 = neura.grant_predicate %54, %101 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
33 %81 = "neura.add"(%80, %28) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 7)
34 %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
35 %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
36 %143 = "neura.fused_op"(%59, %90, %134) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
37 %145 = "neura.fused_op"(%110, %89, %133) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
38 %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
39 %115 = neura.grant_predicate %60, %100 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
40 %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 8)
41 %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 8)
42 "neura.store"(%76, %73) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 8)
43 "neura.store"(%82, %27) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 8)
44 neura.return_void %127 : !neura.data<i1, i1> (level: 8)
[HeuristicMapping] Found 160 candidate locations for operation: %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=0
[HeuristicMapping] Successfully mapped operation %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 126 candidate locations for operation: %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=1
[tryRouteDataMove] Routing from Tile#0 @t=0 to Tile#4 @t=1
[HeuristicMapping] Successfully mapped operation %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 158 candidate locations for operation: %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=1
[HeuristicMapping] Successfully mapped operation %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] Found 157 candidate locations for operation: %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=1
[HeuristicMapping] Successfully mapped operation %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 156 candidate locations for operation: %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=1
[HeuristicMapping] Successfully mapped operation %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 116 candidate locations for operation: %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=3
[tryRouteDataMove] Routing from Tile#9 @t=1 to Tile#5 @t=3
[tryRouteDataMove] Routing from Tile#0 @t=1 to Tile#5 @t=3
[HeuristicMapping] Successfully mapped operation %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Found 124 candidate locations for operation: %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=2
[tryRouteDataMove] Routing from Tile#9 @t=1 to Tile#8 @t=2
[tryRouteDataMove] Routing from Tile#4 @t=1 to Tile#8 @t=2
[HeuristicMapping] Successfully mapped operation %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Found 137 candidate locations for operation: %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=2
[tryRouteDataMove] Routing from Tile#10 @t=1 to Tile#10 @t=2
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 152 candidate locations for operation: %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=2
[HeuristicMapping] Successfully mapped operation %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 127 candidate locations for operation: %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=4
[tryRouteDataMove] Routing from Tile#5 @t=3 to Tile#9 @t=4
[tryRouteDataMove] Routing from Tile#10 @t=2 to Tile#9 @t=4
[tryRouteDataMove] Routing from Tile#8 @t=2 to Tile#9 @t=4
[tryRouteDataMove] Routing from Tile#5 @t=3 to Tile#9 @t=4
[tryRouteDataMove] Cannot find routing path from Tile#5 @t=3 to Tile#9 @t=4
[HeuristicMapping] Failed to map operation %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) to candidate location 1/1
[HeuristicMapping] Found 127 candidate locations for operation: %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] All 1 locations for 9 tried, backtracking...
[HeuristicMapping] Backtracking to operation 8 (depth = 1).
[HeuristicMapping] Found 152 candidate locations for operation: %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] All 1 locations for 8 tried, backtracking...
[HeuristicMapping] Backtracking to operation 7 (depth = 2).
[HeuristicMapping] Max backtrack depth exceeded: 2 > 1.
---------------------------------------------------------
[HeuristicMapping] Starting mapping with 175 operations.
Configuration: MAX Backtrack Depth = 1, MAX Candidate Locations = 1
[HeuristicMapping] Filtered 130 non-materialized operations, 45 operations require physical mapping.
[HeuristicMapping] Materialized operations list:
0 %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1> (level: 0)
1 %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (level: 1)
2 %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>) (level: 1)
3 %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1> (level: 1)
4 %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1> (level: 1)
5 %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 2)
6 %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>) (level: 2)
7 %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (level: 2)
8 %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 2)
9 %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 3)
10 %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 3)
11 %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 3)
12 %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (level: 4)
13 %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1> (level: 4)
14 %57 = "neura.fused_op"(%56, %9, %55) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (level: 5)
15 %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
16 %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 5)
17 %77 = "neura.load"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 5)
18 %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1> (level: 5)
19 %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (level: 6)
20 %25:3 = "neura.fused_op"(%24, %15, %20) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.gep"(%152, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %154 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 6)
21 %72:2 = "neura.fused_op"(%58, %65) <{frequency = 6 : i64, pattern_id = 0 : i64, pattern_name = "gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.load"(%152) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 6)
22 %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>) (level: 6)
23 %79 = "neura.mul"(%66, %78) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 6)
24 %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (level: 7)
25 %112:2 = "neura.fused_op"(%22, %31, %99) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) (level: 7)
26 %109:2 = "neura.fused_op"(%26, %30, %98) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) (level: 7)
27 %75 = "neura.add"(%71, %74) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 7)
28 %120 = neura.grant_predicate %68, %105 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 7)
29 %119 = neura.grant_predicate %36, %104 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 7)
30 %118 = neura.grant_predicate %40, %103 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 7)
31 %117 = neura.grant_predicate %48, %102 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
32 %116 = neura.grant_predicate %54, %101 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
33 %81 = "neura.add"(%80, %28) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 7)
34 %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
35 %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
36 %143 = "neura.fused_op"(%59, %90, %134) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
37 %145 = "neura.fused_op"(%110, %89, %133) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
38 %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
39 %115 = neura.grant_predicate %60, %100 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
40 %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 8)
41 %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 8)
42 "neura.store"(%76, %73) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 8)
43 "neura.store"(%82, %27) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 8)
44 neura.return_void %127 : !neura.data<i1, i1> (level: 8)
[HeuristicMapping] Found 176 candidate locations for operation: %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=0
[HeuristicMapping] Successfully mapped operation %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 142 candidate locations for operation: %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=1
[tryRouteDataMove] Routing from Tile#0 @t=0 to Tile#0 @t=1
[tryRouteDataMove] Successfully routed on same tile using Register #0
[HeuristicMapping] Successfully mapped operation %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 174 candidate locations for operation: %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=1
[HeuristicMapping] Successfully mapped operation %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] Found 173 candidate locations for operation: %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=1
[HeuristicMapping] Successfully mapped operation %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 172 candidate locations for operation: %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=1
[HeuristicMapping] Successfully mapped operation %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 143 candidate locations for operation: %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=2
[tryRouteDataMove] Routing from Tile#5 @t=1 to Tile#6 @t=2
[tryRouteDataMove] Routing from Tile#2 @t=1 to Tile#6 @t=2
[HeuristicMapping] Successfully mapped operation %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Found 137 candidate locations for operation: %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=2
[tryRouteDataMove] Routing from Tile#5 @t=1 to Tile#4 @t=2
[tryRouteDataMove] Routing from Tile#0 @t=1 to Tile#4 @t=2
[HeuristicMapping] Successfully mapped operation %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Found 145 candidate locations for operation: %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=2
[tryRouteDataMove] Routing from Tile#4 @t=1 to Tile#5 @t=2
[HeuristicMapping] Successfully mapped operation %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 168 candidate locations for operation: %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=2
[HeuristicMapping] Successfully mapped operation %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 135 candidate locations for operation: %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=3
[tryRouteDataMove] Routing from Tile#6 @t=2 to Tile#5 @t=3
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#5 @t=3
[tryRouteDataMove] Successfully routed on same tile using Register #160
[tryRouteDataMove] Routing from Tile#4 @t=2 to Tile#5 @t=3
[tryRouteDataMove] Routing from Tile#6 @t=2 to Tile#5 @t=3
[tryRouteDataMove] Cannot find routing path from Tile#6 @t=2 to Tile#5 @t=3
[HeuristicMapping] Failed to map operation %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) to candidate location 1/1
[HeuristicMapping] Found 135 candidate locations for operation: %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] All 1 locations for 9 tried, backtracking...
[HeuristicMapping] Backtracking to operation 8 (depth = 1).
[HeuristicMapping] Found 168 candidate locations for operation: %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] All 1 locations for 8 tried, backtracking...
[HeuristicMapping] Backtracking to operation 7 (depth = 2).
[HeuristicMapping] Max backtrack depth exceeded: 2 > 1.
---------------------------------------------------------
[HeuristicMapping] Starting mapping with 175 operations.
Configuration: MAX Backtrack Depth = 1, MAX Candidate Locations = 1
[HeuristicMapping] Filtered 130 non-materialized operations, 45 operations require physical mapping.
[HeuristicMapping] Materialized operations list:
0 %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1> (level: 0)
1 %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (level: 1)
2 %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>) (level: 1)
3 %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1> (level: 1)
4 %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1> (level: 1)
5 %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 2)
6 %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>) (level: 2)
7 %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (level: 2)
8 %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 2)
9 %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 3)
10 %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 3)
11 %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 3)
12 %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (level: 4)
13 %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1> (level: 4)
14 %57 = "neura.fused_op"(%56, %9, %55) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (level: 5)
15 %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
16 %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 5)
17 %77 = "neura.load"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 5)
18 %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1> (level: 5)
19 %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (level: 6)
20 %25:3 = "neura.fused_op"(%24, %15, %20) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.gep"(%152, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %154 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 6)
21 %72:2 = "neura.fused_op"(%58, %65) <{frequency = 6 : i64, pattern_id = 0 : i64, pattern_name = "gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.load"(%152) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>) (level: 6)
22 %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>) (level: 6)
23 %79 = "neura.mul"(%66, %78) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 6)
24 %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) (level: 7)
25 %112:2 = "neura.fused_op"(%22, %31, %99) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) (level: 7)
26 %109:2 = "neura.fused_op"(%26, %30, %98) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) (level: 7)
27 %75 = "neura.add"(%71, %74) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 7)
28 %120 = neura.grant_predicate %68, %105 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 7)
29 %119 = neura.grant_predicate %36, %104 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 7)
30 %118 = neura.grant_predicate %40, %103 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 7)
31 %117 = neura.grant_predicate %48, %102 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
32 %116 = neura.grant_predicate %54, %101 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
33 %81 = "neura.add"(%80, %28) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 7)
34 %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
35 %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
36 %143 = "neura.fused_op"(%59, %90, %134) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
37 %145 = "neura.fused_op"(%110, %89, %133) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
38 %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1> (level: 8)
39 %115 = neura.grant_predicate %60, %100 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
40 %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 8)
41 %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 8)
42 "neura.store"(%76, %73) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 8)
43 "neura.store"(%82, %27) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 8)
44 neura.return_void %127 : !neura.data<i1, i1> (level: 8)
[HeuristicMapping] Found 192 candidate locations for operation: %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=0
[HeuristicMapping] Successfully mapped operation %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 158 candidate locations for operation: %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=1
[tryRouteDataMove] Routing from Tile#0 @t=0 to Tile#4 @t=1
[HeuristicMapping] Successfully mapped operation %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 190 candidate locations for operation: %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=1
[HeuristicMapping] Successfully mapped operation %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] Found 189 candidate locations for operation: %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=1
[HeuristicMapping] Successfully mapped operation %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 188 candidate locations for operation: %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#7 @t=1
[HeuristicMapping] Successfully mapped operation %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 159 candidate locations for operation: %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=2
[tryRouteDataMove] Routing from Tile#10 @t=1 to Tile#6 @t=2
[tryRouteDataMove] Routing from Tile#7 @t=1 to Tile#6 @t=2
[HeuristicMapping] Successfully mapped operation %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = neura.phi_start %arg7, %arg8 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %155 = "neura.load"(%154) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Found 153 candidate locations for operation: %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=3
[tryRouteDataMove] Routing from Tile#10 @t=1 to Tile#6 @t=3
[tryRouteDataMove] Routing from Tile#4 @t=1 to Tile#6 @t=3
[HeuristicMapping] Successfully mapped operation %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Found 154 candidate locations for operation: %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=2
[tryRouteDataMove] Routing from Tile#0 @t=1 to Tile#1 @t=2
[HeuristicMapping] Successfully mapped operation %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 184 candidate locations for operation: %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=2
[HeuristicMapping] Successfully mapped operation %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 161 candidate locations for operation: %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=4
[tryRouteDataMove] Routing from Tile#6 @t=2 to Tile#6 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #192
[tryRouteDataMove] Routing from Tile#1 @t=2 to Tile#6 @t=4
[tryRouteDataMove] Routing from Tile#6 @t=3 to Tile#6 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #193
[tryRouteDataMove] Routing from Tile#6 @t=2 to Tile#6 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #194
[HeuristicMapping] Successfully mapped operation %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %155 = "neura.mul"(%154, %arg9) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Found 182 candidate locations for operation: %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=3
[HeuristicMapping] Successfully mapped operation %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %153 = neura.phi_start %152, %arg5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %153, %arg6 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  neura.yield %154 : !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 163 candidate locations for operation: %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=4
[tryRouteDataMove] Routing from Tile#6 @t=3 to Tile#5 @t=4
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#5 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #160
[HeuristicMapping] Successfully mapped operation %83 = "neura.add"(%64, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 156 candidate locations for operation: %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=5
[tryRouteDataMove] Routing from Tile#5 @t=4 to Tile#5 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #160
[tryRouteDataMove] Routing from Tile#10 @t=3 to Tile#5 @t=5
[tryRouteDataMove] Routing from Tile#6 @t=4 to Tile#5 @t=5
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#5 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #161
[HeuristicMapping] Successfully mapped operation %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg8, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] Found 179 candidate locations for operation: %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=4
[HeuristicMapping] Successfully mapped operation %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 154 candidate locations for operation: %57 = "neura.fused_op"(%56, %9, %55) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=5
[tryRouteDataMove] Routing from Tile#2 @t=4 to Tile#6 @t=5
[HeuristicMapping] Successfully mapped operation %57 = "neura.fused_op"(%56, %9, %55) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.phi_start %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 159 candidate locations for operation: %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=6
[tryRouteDataMove] Routing from Tile#10 @t=3 to Tile#6 @t=6
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#6 @t=6
[HeuristicMapping] Successfully mapped operation %121 = neura.grant_predicate %35, %87 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 158 candidate locations for operation: %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=6
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#5 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #160
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#5 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #161
[HeuristicMapping] Successfully mapped operation %122 = "neura.add"(%93, %95) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 157 candidate locations for operation: %77 = "neura.load"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=5
[tryRouteDataMove] Routing from Tile#6 @t=4 to Tile#10 @t=5
[HeuristicMapping] Successfully mapped operation %77 = "neura.load"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 174 candidate locations for operation: %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=5
[HeuristicMapping] Successfully mapped operation %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 155 candidate locations for operation: %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=6
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#9 @t=6
[tryRouteDataMove] Routing from Tile#5 @t=4 to Tile#9 @t=6
[tryRouteDataMove] Routing from Tile#10 @t=1 to Tile#9 @t=6
[HeuristicMapping] Successfully mapped operation %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.phi_start %arg7, %arg8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %155 = neura.grant_predicate %154, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %arg5 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] Found 149 candidate locations for operation: %25:3 = "neura.fused_op"(%24, %15, %20) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.gep"(%152, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %154 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=6
[tryRouteDataMove] Routing from Tile#4 @t=5 to Tile#4 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #128
[tryRouteDataMove] Routing from Tile#6 @t=2 to Tile#4 @t=6
[HeuristicMapping] Successfully mapped operation %25:3 = "neura.fused_op"(%24, %15, %20) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.gep"(%152, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.load"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153, %154 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Found 156 candidate locations for operation: %72:2 = "neura.fused_op"(%58, %65) <{frequency = 6 : i64, pattern_id = 0 : i64, pattern_name = "gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.load"(%152) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=7
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#6 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #193
[tryRouteDataMove] Routing from Tile#6 @t=3 to Tile#6 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #195
[HeuristicMapping] Successfully mapped operation %72:2 = "neura.fused_op"(%58, %65) <{frequency = 6 : i64, pattern_id = 0 : i64, pattern_name = "gep->load"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %153 = "neura.load"(%152) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
[HeuristicMapping] Found 146 candidate locations for operation: %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=6 to Tile#5 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #160
[tryRouteDataMove] Routing from Tile#6 @t=6 to Tile#5 @t=7
[HeuristicMapping] Successfully mapped operation %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
  %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %152, %152 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1>
}) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
[HeuristicMapping] Found 154 candidate locations for operation: %79 = "neura.mul"(%66, %78) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=6
[tryRouteDataMove] Routing from Tile#6 @t=3 to Tile#10 @t=6
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#10 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #321
[HeuristicMapping] Successfully mapped operation %79 = "neura.mul"(%66, %78) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 151 candidate locations for operation: %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=8
[tryRouteDataMove] Routing from Tile#5 @t=7 to Tile#5 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #160
[tryRouteDataMove] Routing from Tile#9 @t=6 to Tile#5 @t=8
[tryRouteDataMove] Routing from Tile#5 @t=6 to Tile#5 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #162
[HeuristicMapping] Successfully mapped operation %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
  %152 = "neura.not"(%arg5) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %153 = neura.grant_predicate %arg6, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %154 = neura.grant_predicate %arg7, %152 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
}) : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
[HeuristicMapping] Found 150 candidate locations for operation: %112:2 = "neura.fused_op"(%22, %31, %99) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=7
[tryRouteDataMove] Routing from Tile#6 @t=2 to Tile#10 @t=7
[tryRouteDataMove] Routing from Tile#9 @t=6 to Tile#10 @t=7
[HeuristicMapping] Successfully mapped operation %112:2 = "neura.fused_op"(%22, %31, %99) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
[HeuristicMapping] Found 135 candidate locations for operation: %109:2 = "neura.fused_op"(%26, %30, %98) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=6 to Tile#8 @t=7
[tryRouteDataMove] Routing from Tile#9 @t=6 to Tile#8 @t=7
[HeuristicMapping] Successfully mapped operation %109:2 = "neura.fused_op"(%26, %30, %98) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.phi_start %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
[HeuristicMapping] Found 148 candidate locations for operation: %75 = "neura.add"(%71, %74) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=8
[tryRouteDataMove] Routing from Tile#6 @t=4 to Tile#6 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #194
[tryRouteDataMove] Routing from Tile#6 @t=7 to Tile#6 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #192
[HeuristicMapping] Successfully mapped operation %75 = "neura.add"(%71, %74) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 82 candidate locations for operation: %120 = neura.grant_predicate %68, %105 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=7
[tryRouteDataMove] Routing from Tile#6 @t=4 to Tile#9 @t=7
[tryRouteDataMove] Routing from Tile#9 @t=6 to Tile#9 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #288
[tryRouteDataMove] Routing from Tile#9 @t=7 to Tile#6 @t=16
[HeuristicMapping] Successfully mapped operation %120 = neura.grant_predicate %68, %105 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 75 candidate locations for operation: %119 = neura.grant_predicate %36, %104 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=8
[tryRouteDataMove] Routing from Tile#10 @t=3 to Tile#9 @t=8
[tryRouteDataMove] Routing from Tile#9 @t=6 to Tile#9 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #289
[tryRouteDataMove] Routing from Tile#9 @t=8 to Tile#10 @t=15
[HeuristicMapping] Successfully mapped operation %119 = neura.grant_predicate %36, %104 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 60 candidate locations for operation: %118 = neura.grant_predicate %40, %103 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=9
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#5 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #163
[tryRouteDataMove] Routing from Tile#9 @t=6 to Tile#5 @t=9
[tryRouteDataMove] Routing from Tile#5 @t=9 to Tile#5 @t=14
[tryRouteDataMove] Successfully routed on same tile using Register #160
[HeuristicMapping] Successfully mapped operation %118 = neura.grant_predicate %40, %103 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 33 candidate locations for operation: %117 = neura.grant_predicate %48, %102 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=8
[tryRouteDataMove] Routing from Tile#4 @t=1 to Tile#8 @t=8
[tryRouteDataMove] Routing from Tile#9 @t=6 to Tile#8 @t=8
[tryRouteDataMove] Routing from Tile#8 @t=8 to Tile#4 @t=13
[HeuristicMapping] Successfully mapped operation %117 = neura.grant_predicate %48, %102 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 43 candidate locations for operation: %116 = neura.grant_predicate %54, %101 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=9
[tryRouteDataMove] Routing from Tile#1 @t=2 to Tile#9 @t=9
[tryRouteDataMove] Routing from Tile#9 @t=6 to Tile#9 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #294
[tryRouteDataMove] Routing from Tile#9 @t=9 to Tile#1 @t=14
[HeuristicMapping] Successfully mapped operation %116 = neura.grant_predicate %54, %101 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 130 candidate locations for operation: %81 = "neura.add"(%80, %28) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=9
[tryRouteDataMove] Routing from Tile#10 @t=6 to Tile#8 @t=9
[tryRouteDataMove] Routing from Tile#4 @t=6 to Tile#8 @t=9
[HeuristicMapping] Successfully mapped operation %81 = "neura.add"(%80, %28) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 142 candidate locations for operation: %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=9
[tryRouteDataMove] Routing from Tile#4 @t=1 to Tile#4 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #130
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#4 @t=9
[tryRouteDataMove] Routing from Tile#5 @t=8 to Tile#4 @t=9
[HeuristicMapping] Successfully mapped operation %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 141 candidate locations for operation: %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=9
[tryRouteDataMove] Routing from Tile#1 @t=2 to Tile#1 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #32
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#1 @t=9
[tryRouteDataMove] Routing from Tile#5 @t=8 to Tile#1 @t=9
[HeuristicMapping] Successfully mapped operation %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 139 candidate locations for operation: %143 = "neura.fused_op"(%59, %90, %134) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=9
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#6 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #196
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#6 @t=9
[tryRouteDataMove] Routing from Tile#5 @t=8 to Tile#6 @t=9
[HeuristicMapping] Successfully mapped operation %143 = "neura.fused_op"(%59, %90, %134) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 135 candidate locations for operation: %145 = "neura.fused_op"(%110, %89, %133) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=10
[tryRouteDataMove] Routing from Tile#8 @t=7 to Tile#9 @t=10
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#9 @t=10
[tryRouteDataMove] Routing from Tile#5 @t=8 to Tile#9 @t=10
[HeuristicMapping] Successfully mapped operation %145 = "neura.fused_op"(%110, %89, %133) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 127 candidate locations for operation: %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=10
[tryRouteDataMove] Routing from Tile#10 @t=7 to Tile#6 @t=10
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#6 @t=10
[tryRouteDataMove] Routing from Tile#5 @t=8 to Tile#6 @t=10
[HeuristicMapping] Successfully mapped operation %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
  %152 = neura.grant_predicate %arg5, %arg6 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %152, %arg7 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.yield %153 : !neura.data<!llvm.ptr, i1>
}) : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 81 candidate locations for operation: %115 = neura.grant_predicate %60, %100 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=9
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#10 @t=9
[tryRouteDataMove] Routing from Tile#9 @t=6 to Tile#10 @t=9
[tryRouteDataMove] Routing from Tile#10 @t=9 to Tile#6 @t=17
[HeuristicMapping] Successfully mapped operation %115 = neura.grant_predicate %60, %100 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 27 candidate locations for operation: %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=10
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#5 @t=10
[tryRouteDataMove] Successfully routed on same tile using Register #166
[tryRouteDataMove] Routing from Tile#5 @t=8 to Tile#5 @t=10
[tryRouteDataMove] Successfully routed on same tile using Register #162
[tryRouteDataMove] Routing from Tile#5 @t=10 to Tile#5 @t=14
[tryRouteDataMove] Successfully routed on same tile using Register #161
[HeuristicMapping] Successfully mapped operation %149 = neura.grant_predicate %94, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 38 candidate locations for operation: %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=11
[tryRouteDataMove] Routing from Tile#6 @t=6 to Tile#6 @t=11
[tryRouteDataMove] Successfully routed on same tile using Register #198
[tryRouteDataMove] Routing from Tile#5 @t=8 to Tile#6 @t=11
[tryRouteDataMove] Routing from Tile#6 @t=11 to Tile#10 @t=15
[HeuristicMapping] Successfully mapped operation %151 = neura.grant_predicate %150, %130 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 137 candidate locations for operation: "neura.store"(%76, %73) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=9
[tryRouteDataMove] Routing from Tile#6 @t=8 to Tile#2 @t=9
[tryRouteDataMove] Routing from Tile#6 @t=7 to Tile#2 @t=9
[HeuristicMapping] Successfully mapped operation "neura.store"(%76, %73) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
[HeuristicMapping] Found 127 candidate locations for operation: "neura.store"(%82, %27) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=10
[tryRouteDataMove] Routing from Tile#8 @t=9 to Tile#4 @t=10
[tryRouteDataMove] Routing from Tile#4 @t=6 to Tile#4 @t=10
[tryRouteDataMove] Successfully routed on same tile using Register #131
[HeuristicMapping] Successfully mapped operation "neura.store"(%82, %27) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
[HeuristicMapping] Found 121 candidate locations for operation: neura.return_void %127 : !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=8
[tryRouteDataMove] Routing from Tile#5 @t=7 to Tile#1 @t=8
[HeuristicMapping] Successfully mapped operation neura.return_void %127 : !neura.data<i1, i1>
[HeuristicMapping] Successfully mapped all 45 operations.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.mlir.global external local_unnamed_addr @A(dense<0> : tensor<1024x1024xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<1024 x array<1024 x i32>>
  llvm.mlir.global external local_unnamed_addr @s(dense<0> : tensor<1024xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<1024 x i32>
  llvm.mlir.global external local_unnamed_addr @q(dense<0> : tensor<1024xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<1024 x i32>
  llvm.mlir.global external local_unnamed_addr @p(dense<0> : tensor<1024xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<1024 x i32>
  llvm.mlir.global external local_unnamed_addr @r(dense<0> : tensor<1024xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<1024 x i32>
  func.func @_Z6kernelPA1024_iPiS1_S1_S1_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 12 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 8 : i32, res_mii = 3 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = "neura.grant_once"() <{constant_value = "%arg0"}> {dfg_id = 0 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<!llvm.ptr, i1>
    %1 = "neura.grant_once"() <{constant_value = "%arg1"}> {dfg_id = 1 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 0 : i32}]} : () -> !neura.data<!llvm.ptr, i1>
    %2 = "neura.grant_once"() <{constant_value = "%arg2"}> {dfg_id = 2 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 0 : i32, y = 1 : i32}]} : () -> !neura.data<!llvm.ptr, i1>
    %3 = "neura.grant_once"() <{constant_value = "%arg3"}> {dfg_id = 3 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<!llvm.ptr, i1>
    %4 = "neura.grant_once"() <{constant_value = "%arg4"}> {dfg_id = 4 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 3 : i32, y = 1 : i32}]} : () -> !neura.data<!llvm.ptr, i1>
    %5 = neura.reserve {dfg_id = 5 : i32} : !neura.data<i64, i1>
    %6 = neura.reserve {dfg_id = 6 : i32} : !neura.data<i64, i1>
    %7 = neura.reserve {dfg_id = 7 : i32} : !neura.data<!llvm.ptr, i1>
    %8 = neura.reserve {dfg_id = 8 : i32} : !neura.data<!llvm.ptr, i1>
    %9 = neura.reserve {dfg_id = 9 : i32} : !neura.data<!llvm.ptr, i1>
    %10 = neura.reserve {dfg_id = 10 : i32} : !neura.data<i64, i1>
    %11:2 = "neura.fused_op"(%10) <{frequency = 4 : i64, pattern_id = 9 : i64, pattern_name = "grant_once->phi_start"}> ({
    ^bb0(%arg5: !neura.data<i64, i1>):
      %152 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 11 : i32} : () -> !neura.data<i64, i1>
      %153 = neura.phi_start %152, %arg5 {dfg_id = 55 : i32} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
      neura.yield %152, %153 : !neura.data<i64, i1>, !neura.data<i64, i1> {dfg_id = 89 : i32}
    }) {dfg_id = 54 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>) -> (!neura.data<i64, i1>, !neura.data<i64, i1>)
    %12 = "neura.data_mov"(%11#0) {dfg_id = 86 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %13 = "neura.data_mov"(%11#1) {dfg_id = 88 : i32, mapping_locs = [{id = 31 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}, {id = 289 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 2 : i32}, {id = 289 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 3 : i32}, {id = 289 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}, {id = 289 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %14 = "neura.data_mov"(%11#1) {dfg_id = 87 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}, {id = 33 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %15 = neura.reserve {dfg_id = 12 : i32} : !neura.data<!llvm.ptr, i1>
    %16 = neura.reserve {dfg_id = 13 : i32} : !neura.data<!llvm.ptr, i1>
    %17 = neura.reserve {dfg_id = 14 : i32} : !neura.data<i64, i1>
    %18 = "neura.data_mov"(%4) {dfg_id = 53 : i32, mapping_locs = [{id = 21 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %19:3 = "neura.fused_op"(%12, %17, %18, %16) <{frequency = 3 : i64, pattern_id = 6 : i64, pattern_name = "phi_start->fused_op:phi_start->fused_op:gep->load"}> ({
    ^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<!llvm.ptr, i1>):
      %152 = neura.phi_start %arg5, %arg6 {dfg_id = 15 : i32} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
      %153 = neura.phi_start %arg7, %arg8 {dfg_id = 16 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
      %154 = "neura.gep"(%153, %152) <{operandSegmentSizes = array<i32: 1, 1>}> {dfg_id = 56 : i32} : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
      %155 = "neura.load"(%154) {dfg_id = 90 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
      neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1> {dfg_id = 127 : i32}
    }) {dfg_id = 126 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
    %20 = "neura.data_mov"(%19#0) {dfg_id = 135 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 13 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 129 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}, {id = 129 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %21 = "neura.data_mov"(%19#0) {dfg_id = 134 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 192 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %22 = "neura.data_mov"(%19#1) {dfg_id = 136 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 322 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 3 : i32}, {id = 322 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 4 : i32}, {id = 322 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 5 : i32}, {id = 322 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %23 = "neura.data_mov"(%19#2) {dfg_id = 137 : i32, mapping_locs = [{id = 194 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 2 : i32}, {id = 194 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %24 = "neura.data_mov"(%2) {dfg_id = 51 : i32, mapping_locs = [{id = 128 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %25:3 = "neura.fused_op"(%24, %15, %20) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i64, i1>):
      %152 = neura.phi_start %arg5, %arg6 {dfg_id = 17 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
      %153 = "neura.gep"(%152, %arg7) <{operandSegmentSizes = array<i32: 1, 1>}> {dfg_id = 57 : i32} : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
      %154 = "neura.load"(%153) {dfg_id = 91 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
      neura.yield %152, %153, %154 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1> {dfg_id = 128 : i32}
    }) {dfg_id = 143 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
    %26 = "neura.data_mov"(%25#0) {dfg_id = 147 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %27 = "neura.data_mov"(%25#1) {dfg_id = 148 : i32, mapping_locs = [{id = 131 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 6 : i32}, {id = 131 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 7 : i32}, {id = 131 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 8 : i32}, {id = 131 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %28 = "neura.data_mov"(%25#2) {dfg_id = 149 : i32, mapping_locs = [{id = 128 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 12 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 256 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %29 = neura.reserve {dfg_id = 18 : i32} : !neura.data<i64, i1>
    %30 = neura.reserve {dfg_id = 19 : i32} : !neura.data<!llvm.ptr, i1>
    %31 = neura.reserve {dfg_id = 20 : i32} : !neura.data<!llvm.ptr, i1>
    %32 = neura.reserve {dfg_id = 21 : i32} : !neura.data<i64, i1>
    %33 = neura.reserve {dfg_id = 22 : i32} : !neura.data<i64, i1>
    %34 = "neura.fused_op"(%5, %33) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
    ^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
      %152 = "neura.grant_once"() <{constant_value = 1024 : i64}> {dfg_id = 23 : i32} : () -> !neura.data<i64, i1>
      %153 = neura.phi_start %152, %arg5 {dfg_id = 59 : i32} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
      %154 = neura.phi_start %153, %arg6 {dfg_id = 95 : i32} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
      neura.yield %154 : !neura.data<i64, i1> {dfg_id = 129 : i32}
    }) {dfg_id = 58 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %35 = "neura.data_mov"(%34) {dfg_id = 94 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 192 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 192 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %36 = "neura.data_mov"(%34) {dfg_id = 93 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}, {id = 31 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 290 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 5 : i32}, {id = 290 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 6 : i32}, {id = 290 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %37 = "neura.data_mov"(%34) {dfg_id = 92 : i32, mapping_locs = [{id = 31 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 29 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %38 = neura.reserve {dfg_id = 24 : i32} : !neura.data<i64, i1>
    %39 = "neura.fused_op"(%6, %38) <{frequency = 5 : i64, pattern_id = 5 : i64, pattern_name = "grant_once->fused_op:phi_start->phi_start"}> ({
    ^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
      %152 = "neura.grant_once"() <{constant_value = 1 : i64}> {dfg_id = 25 : i32} : () -> !neura.data<i64, i1>
      %153 = neura.phi_start %152, %arg5 {dfg_id = 61 : i32} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
      %154 = neura.phi_start %153, %arg6 {dfg_id = 99 : i32} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
      neura.yield %154 : !neura.data<i64, i1> {dfg_id = 130 : i32}
    }) {dfg_id = 60 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %40 = "neura.data_mov"(%39) {dfg_id = 98 : i32, mapping_locs = [{id = 163 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 2 : i32}, {id = 163 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 3 : i32}, {id = 163 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 4 : i32}, {id = 163 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 5 : i32}, {id = 163 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 6 : i32}, {id = 163 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 7 : i32}, {id = 163 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %41 = "neura.data_mov"(%39) {dfg_id = 97 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 160 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %42 = "neura.data_mov"(%39) {dfg_id = 96 : i32, mapping_locs = [{id = 161 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 2 : i32}, {id = 161 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 3 : i32}, {id = 161 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %43 = neura.reserve {dfg_id = 26 : i32} : !neura.data<!llvm.ptr, i1>
    %44 = "neura.data_mov"(%3) {dfg_id = 52 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %45 = "neura.fused_op"(%44, %7, %43) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
      %152 = neura.phi_start %arg5, %arg6 {dfg_id = 27 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
      %153 = neura.phi_start %152, %arg7 {dfg_id = 62 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
      neura.yield %153 : !neura.data<!llvm.ptr, i1> {dfg_id = 100 : i32}
    }) {dfg_id = 85 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %46 = "neura.data_mov"(%45) {dfg_id = 125 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}, {id = 14 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %47 = "neura.data_mov"(%45) {dfg_id = 124 : i32, mapping_locs = [{id = 130 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 1 : i32}, {id = 130 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 2 : i32}, {id = 130 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 3 : i32}, {id = 130 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 4 : i32}, {id = 130 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 5 : i32}, {id = 130 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 6 : i32}, {id = 130 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 7 : i32}, {id = 130 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %48 = "neura.data_mov"(%45) {dfg_id = 123 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}, {id = 256 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 256 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}, {id = 256 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 256 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}, {id = 256 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 256 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %49 = neura.reserve {dfg_id = 28 : i32} : !neura.data<!llvm.ptr, i1>
    %50 = "neura.data_mov"(%0) {dfg_id = 49 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %51 = "neura.fused_op"(%50, %8, %49) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
      %152 = neura.phi_start %arg5, %arg6 {dfg_id = 29 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
      %153 = neura.phi_start %152, %arg7 {dfg_id = 63 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
      neura.yield %153 : !neura.data<!llvm.ptr, i1> {dfg_id = 101 : i32}
    }) {dfg_id = 83 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %52 = "neura.data_mov"(%51) {dfg_id = 119 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 7 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %53 = "neura.data_mov"(%51) {dfg_id = 118 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 32 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}, {id = 32 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 32 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}, {id = 32 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 32 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 32 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %54 = "neura.data_mov"(%51) {dfg_id = 117 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 16 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 293 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 4 : i32}, {id = 293 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 5 : i32}, {id = 293 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 6 : i32}, {id = 293 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 7 : i32}, {id = 293 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %55 = neura.reserve {dfg_id = 30 : i32} : !neura.data<!llvm.ptr, i1>
    %56 = "neura.data_mov"(%1) {dfg_id = 50 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %57 = "neura.fused_op"(%56, %9, %55) <{frequency = 10 : i64, pattern_id = 8 : i64, pattern_name = "phi_start->phi_start"}> ({
    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
      %152 = neura.phi_start %arg5, %arg6 {dfg_id = 31 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
      %153 = neura.phi_start %152, %arg7 {dfg_id = 64 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
      neura.yield %153 : !neura.data<!llvm.ptr, i1> {dfg_id = 102 : i32}
    }) {dfg_id = 84 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %58 = "neura.data_mov"(%57) {dfg_id = 122 : i32, mapping_locs = [{id = 193 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}, {id = 193 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %59 = "neura.data_mov"(%57) {dfg_id = 121 : i32, mapping_locs = [{id = 196 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 5 : i32}, {id = 196 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 6 : i32}, {id = 196 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 7 : i32}, {id = 196 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %60 = "neura.data_mov"(%57) {dfg_id = 120 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 320 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 320 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 320 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %61 = neura.reserve {dfg_id = 32 : i32} : !neura.data<i64, i1>
    %62:2 = "neura.fused_op"(%14, %61, %46) <{frequency = 8 : i64, pattern_id = 10 : i64, pattern_name = "phi_start->fused_op:gep->load"}> ({
    ^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>):
      %152 = neura.phi_start %arg5, %arg6 {dfg_id = 33 : i32} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
      %153 = "neura.gep"(%arg7, %152) <{operandSegmentSizes = array<i32: 1, 1>}> {dfg_id = 65 : i32} : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
      %154 = "neura.load"(%153) {dfg_id = 103 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
      neura.yield %152, %154 : !neura.data<i64, i1>, !neura.data<i32, i1> {dfg_id = 131 : i32}
    }) {dfg_id = 133 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>) -> (!neura.data<i64, i1>, !neura.data<i32, i1>)
    %63 = "neura.data_mov"(%62#0) {dfg_id = 141 : i32, mapping_locs = [{id = 193 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %64 = "neura.data_mov"(%62#0) {dfg_id = 140 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %65 = "neura.data_mov"(%62#0) {dfg_id = 139 : i32, mapping_locs = [{id = 195 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 3 : i32}, {id = 195 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 4 : i32}, {id = 195 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 5 : i32}, {id = 195 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %66 = "neura.data_mov"(%62#1) {dfg_id = 142 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 320 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 320 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %67:3 = "neura.fused_op"(%21, %32, %52, %63, %23) <{frequency = 3 : i64, pattern_id = 3 : i64, pattern_name = "fused_op:phi_start->fused_op:gep->load->mul"}> ({
    ^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<!llvm.ptr, i1>, %arg8: !neura.data<i64, i1>, %arg9: !neura.data<i32, i1>):
      %152 = neura.phi_start %arg5, %arg6 {dfg_id = 34 : i32} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
      %153 = "neura.gep"(%arg7, %152, %arg8) <{operandSegmentSizes = array<i32: 1, 2>}> {dfg_id = 66 : i32} : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
      %154 = "neura.load"(%153) {dfg_id = 104 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
      %155 = "neura.mul"(%154, %arg9) {dfg_id = 132 : i32} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
      neura.yield %152, %153, %155 : !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1> {dfg_id = 138 : i32}
    }) {dfg_id = 146 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i32, i1>) -> (!neura.data<i64, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
    %68 = "neura.data_mov"(%67#0) {dfg_id = 155 : i32, mapping_locs = [{id = 193 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}, {id = 17 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 16 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %69 = "neura.data_mov"(%67#0) {dfg_id = 154 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %70 = "neura.data_mov"(%67#1) {dfg_id = 156 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %71 = "neura.data_mov"(%67#2) {dfg_id = 157 : i32, mapping_locs = [{id = 194 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 4 : i32}, {id = 194 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 5 : i32}, {id = 194 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 6 : i32}, {id = 194 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %72:2 = "neura.fused_op"(%58, %65) <{frequency = 6 : i64, pattern_id = 0 : i64, pattern_name = "gep->load"}> ({
    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i64, i1>):
      %152 = "neura.gep"(%arg5, %arg6) <{operandSegmentSizes = array<i32: 1, 1>}> {dfg_id = 35 : i32} : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
      %153 = "neura.load"(%152) {dfg_id = 67 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
      neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<i32, i1> {dfg_id = 105 : i32}
    }) {dfg_id = 144 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<i32, i1>)
    %73 = "neura.data_mov"(%72#0) {dfg_id = 150 : i32, mapping_locs = [{id = 19 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 64 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %74 = "neura.data_mov"(%72#1) {dfg_id = 151 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %75 = "neura.add"(%71, %74) {dfg_id = 160 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    %76 = "neura.data_mov"(%75) {dfg_id = 172 : i32, mapping_locs = [{id = 19 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    "neura.store"(%76, %73) {dfg_id = 177 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
    %77 = "neura.load"(%70) {dfg_id = 159 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
    %78 = "neura.data_mov"(%77) {dfg_id = 171 : i32, mapping_locs = [{id = 321 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %79 = "neura.mul"(%66, %78) {dfg_id = 176 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    %80 = "neura.data_mov"(%79) {dfg_id = 193 : i32, mapping_locs = [{id = 31 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 288 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 27 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %81 = "neura.add"(%80, %28) {dfg_id = 205 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    %82 = "neura.data_mov"(%81) {dfg_id = 218 : i32, mapping_locs = [{id = 25 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    "neura.store"(%82, %27) {dfg_id = 223 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
    %83 = "neura.add"(%64, %41) {dfg_id = 145 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %84 = "neura.data_mov"(%83) {dfg_id = 153 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %85:3 = "neura.fused_op"(%84, %37, %69, %42) <{frequency = 3 : i64, pattern_id = 1 : i64, pattern_name = "fused_op:icmp->grant_predicate->grant_predicate"}> ({
    ^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
      %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> {dfg_id = 36 : i32} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
      %153 = neura.grant_predicate %arg7, %152 {dfg_id = 69 : i32} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
      %154 = neura.grant_predicate %arg8, %152 {dfg_id = 68 : i32} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
      neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1> {dfg_id = 106 : i32}
    }) {dfg_id = 158 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
    %86 = "neura.data_mov"(%85#0) {dfg_id = 167 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %87 = "neura.data_mov"(%85#0) {dfg_id = 166 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %88 = "neura.data_mov"(%85#0) {dfg_id = 165 : i32, mapping_locs = [{id = 165 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 5 : i32}, {id = 164 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 6 : i32}, {id = 14 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 194 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 8 : i32}, {id = 194 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %89 = "neura.data_mov"(%85#0) {dfg_id = 164 : i32, mapping_locs = [{id = 164 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 5 : i32}, {id = 161 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}, {id = 16 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 289 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}, {id = 289 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %90 = "neura.data_mov"(%85#0) {dfg_id = 163 : i32, mapping_locs = [{id = 162 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 5 : i32}, {id = 14 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 193 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}, {id = 193 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %91 = "neura.data_mov"(%85#0) {dfg_id = 162 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 33 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}, {id = 33 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}, {id = 33 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %92 = "neura.data_mov"(%85#0) {dfg_id = 161 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 129 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}, {id = 129 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}, {id = 129 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %93 = "neura.data_mov"(%85#1) {dfg_id = 168 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %94 = "neura.data_mov"(%85#2) {dfg_id = 170 : i32, mapping_locs = [{id = 166 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 5 : i32}, {id = 166 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 6 : i32}, {id = 166 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 7 : i32}, {id = 166 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 8 : i32}, {id = 166 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %95 = "neura.data_mov"(%85#2) {dfg_id = 169 : i32, mapping_locs = [{id = 161 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %96 = "neura.data_mov"(%83) {dfg_id = 152 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 288 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %97:4 = "neura.fused_op"(%86, %96, %13, %29) <{frequency = 7 : i64, pattern_id = 2 : i64, pattern_name = "fused_op:fused_op:not->grant_predicate->fused_op:phi_start->grant_predicate->grant_predicate"}> ({
    ^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>, %arg8: !neura.data<i64, i1>):
      %152 = "neura.not"(%arg5) {dfg_id = 37 : i32} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
      %153 = neura.grant_predicate %arg6, %152 {dfg_id = 70 : i32} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
      %154 = neura.phi_start %arg7, %arg8 {dfg_id = 38 : i32} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
      %155 = neura.grant_predicate %154, %152 {dfg_id = 72 : i32} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
      %156 = neura.grant_predicate %154, %arg5 {dfg_id = 71 : i32} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
      neura.yield %152, %153, %155, %156 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1> {dfg_id = 107 : i32}
    }) {dfg_id = 174 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
    %98 = "neura.data_mov"(%97#0) {dfg_id = 187 : i32, mapping_locs = [{id = 27 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %99 = "neura.data_mov"(%97#0) {dfg_id = 186 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %100 = "neura.data_mov"(%97#0) {dfg_id = 185 : i32, mapping_locs = [{id = 30 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 41 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 45 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %101 = "neura.data_mov"(%97#0) {dfg_id = 184 : i32, mapping_locs = [{id = 294 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 6 : i32}, {id = 294 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 7 : i32}, {id = 294 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %102 = "neura.data_mov"(%97#0) {dfg_id = 183 : i32, mapping_locs = [{id = 292 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 6 : i32}, {id = 27 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %103 = "neura.data_mov"(%97#0) {dfg_id = 182 : i32, mapping_locs = [{id = 291 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 6 : i32}, {id = 29 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 160 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %104 = "neura.data_mov"(%97#0) {dfg_id = 181 : i32, mapping_locs = [{id = 289 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}, {id = 289 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %105 = "neura.data_mov"(%97#0) {dfg_id = 180 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %106 = "neura.data_mov"(%97#1) {dfg_id = 188 : i32} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %107 = "neura.data_mov"(%97#2) {dfg_id = 189 : i32} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %108 = "neura.data_mov"(%97#3) {dfg_id = 190 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 161 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %109:2 = "neura.fused_op"(%26, %30, %98) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
      %152 = neura.phi_start %arg5, %arg6 {dfg_id = 39 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
      %153 = neura.grant_predicate %152, %arg7 {dfg_id = 73 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
      neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> {dfg_id = 108 : i32}
    }) {dfg_id = 201 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
    %110 = "neura.data_mov"(%109#0) {dfg_id = 214 : i32, mapping_locs = [{id = 24 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 288 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 288 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %111 = "neura.data_mov"(%109#1) {dfg_id = 215 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %112:2 = "neura.fused_op"(%22, %31, %99) <{frequency = 5 : i64, pattern_id = 11 : i64, pattern_name = "phi_start->grant_predicate"}> ({
    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<!llvm.ptr, i1>, %arg7: !neura.data<i1, i1>):
      %152 = neura.phi_start %arg5, %arg6 {dfg_id = 40 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
      %153 = neura.grant_predicate %152, %arg7 {dfg_id = 74 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
      neura.yield %152, %153 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> {dfg_id = 109 : i32}
    }) {dfg_id = 200 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>) -> (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>)
    %113 = "neura.data_mov"(%112#0) {dfg_id = 212 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 192 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 192 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %114 = "neura.data_mov"(%112#1) {dfg_id = 213 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %106 -> %61 {dfg_id = 202 : i32} : !neura.data<i64, i1> !neura.data<i64, i1>
    %115 = neura.grant_predicate %60, %100 {dfg_id = 199 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %115 -> %55 {dfg_id = 211 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}, {id = 197 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 10 : i32}, {id = 197 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 11 : i32}, {id = 197 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 12 : i32}, {id = 197 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 13 : i32}, {id = 197 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 14 : i32}, {id = 197 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 15 : i32}, {id = 197 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 16 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    %116 = neura.grant_predicate %54, %101 {dfg_id = 198 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %116 -> %49 {dfg_id = 210 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}, {id = 15 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}, {id = 32 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}, {id = 32 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}, {id = 32 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    %117 = neura.grant_predicate %48, %102 {dfg_id = 197 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %117 -> %43 {dfg_id = 209 : i32, mapping_locs = [{id = 25 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 128 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}, {id = 128 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}, {id = 128 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}, {id = 128 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    %118 = neura.grant_predicate %40, %103 {dfg_id = 196 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %118 -> %38 {dfg_id = 208 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}, {id = 160 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}, {id = 160 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}, {id = 160 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}, {id = 160 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %119 = neura.grant_predicate %36, %104 {dfg_id = 195 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %119 -> %33 {dfg_id = 207 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 321 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 321 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}, {id = 321 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}, {id = 321 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 321 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}, {id = 321 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %120 = neura.grant_predicate %68, %105 {dfg_id = 194 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %120 -> %32 {dfg_id = 206 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 33 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 196 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 9 : i32}, {id = 196 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 10 : i32}, {id = 196 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 11 : i32}, {id = 196 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 12 : i32}, {id = 196 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 13 : i32}, {id = 196 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 14 : i32}, {id = 196 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 15 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    neura.ctrl_mov %114 -> %31 {dfg_id = 219 : i32} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %111 -> %30 {dfg_id = 220 : i32} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %107 -> %29 {dfg_id = 203 : i32} : !neura.data<i64, i1> !neura.data<i64, i1>
    %121 = neura.grant_predicate %35, %87 {dfg_id = 173 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %122 = "neura.add"(%93, %95) {dfg_id = 175 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %123 = "neura.data_mov"(%122) {dfg_id = 192 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %124 = "neura.data_mov"(%121) {dfg_id = 179 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %125:2 = "neura.fused_op"(%123, %124) <{frequency = 11 : i64, pattern_id = 3 : i64, pattern_name = "icmp->grant_predicate"}> ({
    ^bb0(%arg5: !neura.data<i64, i1>, %arg6: !neura.data<i64, i1>):
      %152 = "neura.icmp"(%arg5, %arg6) <{cmpType = "eq"}> {dfg_id = 41 : i32} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
      %153 = neura.grant_predicate %152, %152 {dfg_id = 75 : i32} : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
      neura.yield %152, %153 : !neura.data<i1, i1>, !neura.data<i1, i1> {dfg_id = 110 : i32}
    }) {dfg_id = 204 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i1, i1>)
    %126 = "neura.data_mov"(%125#0) {dfg_id = 216 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %127 = "neura.data_mov"(%125#1) {dfg_id = 217 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %128 = "neura.data_mov"(%122) {dfg_id = 191 : i32, mapping_locs = [{id = 162 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 6 : i32}, {id = 162 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %129:3 = "neura.fused_op"(%126, %108, %128) <{frequency = 12 : i64, pattern_id = 4 : i64, pattern_name = "fused_op:not->grant_predicate->grant_predicate"}> ({
    ^bb0(%arg5: !neura.data<i1, i1>, %arg6: !neura.data<i64, i1>, %arg7: !neura.data<i64, i1>):
      %152 = "neura.not"(%arg5) {dfg_id = 42 : i32} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
      %153 = neura.grant_predicate %arg6, %152 {dfg_id = 77 : i32} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
      %154 = neura.grant_predicate %arg7, %152 {dfg_id = 76 : i32} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
      neura.yield %152, %153, %154 : !neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1> {dfg_id = 111 : i32}
    }) {dfg_id = 221 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> (!neura.data<i1, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>)
    %130 = "neura.data_mov"(%129#0) {dfg_id = 230 : i32, mapping_locs = [{id = 164 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 8 : i32}, {id = 161 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 14 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %131 = "neura.data_mov"(%129#0) {dfg_id = 229 : i32, mapping_locs = [{id = 162 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 8 : i32}, {id = 162 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %132 = "neura.data_mov"(%129#0) {dfg_id = 228 : i32, mapping_locs = [{id = 161 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}, {id = 14 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %133 = "neura.data_mov"(%129#0) {dfg_id = 227 : i32, mapping_locs = [{id = 16 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 290 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %134 = "neura.data_mov"(%129#0) {dfg_id = 226 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %135 = "neura.data_mov"(%129#0) {dfg_id = 225 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %136 = "neura.data_mov"(%129#0) {dfg_id = 224 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %137 = "neura.data_mov"(%129#1) {dfg_id = 231 : i32} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %138 = "neura.data_mov"(%129#2) {dfg_id = 232 : i32} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %139 = "neura.fused_op"(%47, %92, %136) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
      %152 = neura.grant_predicate %arg5, %arg6 {dfg_id = 43 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
      %153 = neura.grant_predicate %152, %arg7 {dfg_id = 78 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
      neura.yield %153 : !neura.data<!llvm.ptr, i1> {dfg_id = 112 : i32}
    }) {dfg_id = 233 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
    %140 = "neura.data_mov"(%139) {dfg_id = 242 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %141 = "neura.fused_op"(%53, %91, %135) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
      %152 = neura.grant_predicate %arg5, %arg6 {dfg_id = 44 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
      %153 = neura.grant_predicate %152, %arg7 {dfg_id = 79 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
      neura.yield %153 : !neura.data<!llvm.ptr, i1> {dfg_id = 113 : i32}
    }) {dfg_id = 234 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
    %142 = "neura.data_mov"(%141) {dfg_id = 243 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %143 = "neura.fused_op"(%59, %90, %134) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
      %152 = neura.grant_predicate %arg5, %arg6 {dfg_id = 45 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
      %153 = neura.grant_predicate %152, %arg7 {dfg_id = 80 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
      neura.yield %153 : !neura.data<!llvm.ptr, i1> {dfg_id = 114 : i32}
    }) {dfg_id = 235 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
    %144 = "neura.data_mov"(%143) {dfg_id = 244 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %145 = "neura.fused_op"(%110, %89, %133) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
      %152 = neura.grant_predicate %arg5, %arg6 {dfg_id = 46 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
      %153 = neura.grant_predicate %152, %arg7 {dfg_id = 81 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
      neura.yield %153 : !neura.data<!llvm.ptr, i1> {dfg_id = 115 : i32}
    }) {dfg_id = 236 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
    %146 = "neura.data_mov"(%145) {dfg_id = 245 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %147 = "neura.fused_op"(%113, %88, %132) <{frequency = 8 : i64, pattern_id = 2 : i64, pattern_name = "grant_predicate->grant_predicate"}> ({
    ^bb0(%arg5: !neura.data<!llvm.ptr, i1>, %arg6: !neura.data<i1, i1>, %arg7: !neura.data<i1, i1>):
      %152 = neura.grant_predicate %arg5, %arg6 {dfg_id = 47 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
      %153 = neura.grant_predicate %152, %arg7 {dfg_id = 82 : i32} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
      neura.yield %153 : !neura.data<!llvm.ptr, i1> {dfg_id = 116 : i32}
    }) {dfg_id = 237 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<!llvm.ptr, i1>
    %148 = "neura.data_mov"(%147) {dfg_id = 246 : i32} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %138 -> %17 {dfg_id = 241 : i32} : !neura.data<i64, i1> !neura.data<i64, i1>
    neura.ctrl_mov %148 -> %16 {dfg_id = 253 : i32} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %146 -> %15 {dfg_id = 252 : i32} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %137 -> %10 {dfg_id = 240 : i32} : !neura.data<i64, i1> !neura.data<i64, i1>
    neura.ctrl_mov %144 -> %9 {dfg_id = 251 : i32} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %142 -> %8 {dfg_id = 250 : i32} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %140 -> %7 {dfg_id = 249 : i32} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    %149 = neura.grant_predicate %94, %131 {dfg_id = 238 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %149 -> %6 {dfg_id = 247 : i32, mapping_locs = [{id = 161 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}, {id = 161 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}, {id = 161 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 161 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %150 = "neura.data_mov"(%121) {dfg_id = 178 : i32, mapping_locs = [{id = 198 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 6 : i32}, {id = 198 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 7 : i32}, {id = 198 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 8 : i32}, {id = 198 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 9 : i32}, {id = 198 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %151 = neura.grant_predicate %150, %130 {dfg_id = 239 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %151 -> %5 {dfg_id = 248 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}, {id = 322 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 12 : i32}, {id = 322 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 13 : i32}, {id = 322 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 14 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    neura.return_void %127 : !neura.data<i1, i1> {dfg_id = 222 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 0 : i32}]}
    neura.yield {dfg_id = 48 : i32}
  }
  llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {memory_effects = #llvm.memory_effects<other = readwrite, argMem = none, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.addressof @p : !llvm.ptr
    %1 = llvm.mlir.addressof @A : !llvm.ptr
    %2 = llvm.mlir.addressof @s : !llvm.ptr
    %3 = llvm.mlir.addressof @q : !llvm.ptr
    %4 = llvm.mlir.addressof @r : !llvm.ptr
    %5 = "neura.constant"() <{value = 0 : i64}> : () -> i64
    %6 = "neura.constant"() <{value = 0 : i32}> : () -> i32
    %7 = "neura.data_mov"(%5) : (i64) -> i64
    neura.br %7 : i64 to ^bb1
  ^bb1(%8: i64):  // 2 preds: ^bb0, ^bb3
    %9 = "neura.data_mov"(%4) : (!llvm.ptr) -> !llvm.ptr
    %10 = "neura.data_mov"(%8) : (i64) -> i64
    %11 = "neura.gep"(%9, %10) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
    %12 = "neura.data_mov"(%3) : (!llvm.ptr) -> !llvm.ptr
    %13 = "neura.data_mov"(%8) : (i64) -> i64
    %14 = "neura.gep"(%12, %13) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
    %15 = "neura.data_mov"(%11) : (!llvm.ptr) -> !llvm.ptr
    %16 = "neura.load"(%15) : (!llvm.ptr) -> i32
    %17 = "neura.data_mov"(%14) : (!llvm.ptr) -> !llvm.ptr
    %18 = "neura.load"(%17) : (!llvm.ptr) -> i32
    %19 = "neura.data_mov"(%18) : (i32) -> i32
    %20 = "neura.data_mov"(%5) : (i64) -> i64
    neura.br %19, %20 : i32, i64 to ^bb2
  ^bb2(%21: i32, %22: i64):  // 2 preds: ^bb1, ^bb2
    %23 = "neura.data_mov"(%2) : (!llvm.ptr) -> !llvm.ptr
    %24 = "neura.data_mov"(%22) : (i64) -> i64
    %25 = "neura.gep"(%23, %24) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
    %26 = "neura.data_mov"(%25) : (!llvm.ptr) -> !llvm.ptr
    %27 = "neura.load"(%26) : (!llvm.ptr) -> i32
    %28 = "neura.data_mov"(%1) : (!llvm.ptr) -> !llvm.ptr
    %29 = "neura.data_mov"(%8) : (i64) -> i64
    %30 = "neura.data_mov"(%22) : (i64) -> i64
    %31 = "neura.gep"(%28, %29, %30) <{operandSegmentSizes = array<i32: 1, 2>}> : (!llvm.ptr, i64, i64) -> !llvm.ptr
    %32 = "neura.data_mov"(%31) : (!llvm.ptr) -> !llvm.ptr
    %33 = "neura.load"(%32) : (!llvm.ptr) -> i32
    %34 = "neura.data_mov"(%33) : (i32) -> i32
    %35 = "neura.data_mov"(%16) : (i32) -> i32
    %36 = "neura.mul"(%34, %35) : (i32, i32) -> i32
    %37 = "neura.data_mov"(%36) : (i32) -> i32
    %38 = "neura.data_mov"(%27) : (i32) -> i32
    %39 = "neura.add"(%37, %38) : (i32, i32) -> i32
    %40 = "neura.data_mov"(%39) : (i32) -> i32
    %41 = "neura.data_mov"(%25) : (!llvm.ptr) -> !llvm.ptr
    "neura.store"(%40, %41) : (i32, !llvm.ptr) -> ()
    %42 = "neura.data_mov"(%0) : (!llvm.ptr) -> !llvm.ptr
    %43 = "neura.data_mov"(%22) : (i64) -> i64
    %44 = "neura.gep"(%42, %43) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
    %45 = "neura.data_mov"(%44) : (!llvm.ptr) -> !llvm.ptr
    %46 = "neura.load"(%45) : (!llvm.ptr) -> i32
    %47 = "neura.data_mov"(%46) : (i32) -> i32
    %48 = "neura.data_mov"(%33) : (i32) -> i32
    %49 = "neura.mul"(%47, %48) : (i32, i32) -> i32
    %50 = "neura.data_mov"(%49) : (i32) -> i32
    %51 = "neura.data_mov"(%21) : (i32) -> i32
    %52 = "neura.add"(%50, %51) : (i32, i32) -> i32
    %53 = "neura.data_mov"(%22) : (i64) -> i64
    %54 = "neura.add"(%53) {rhs_value = 1 : i64} : (i64) -> i64
    %55 = "neura.data_mov"(%54) : (i64) -> i64
    %56 = "neura.icmp"(%55) <{cmpType = "eq"}> {rhs_value = 1024 : i64} : (i64) -> i1
    %57 = "neura.data_mov"(%56) : (i1) -> i1
    %58 = "neura.data_mov"(%52) : (i32) -> i32
    %59 = "neura.data_mov"(%54) : (i64) -> i64
    neura.cond_br %57 : i1 then to ^bb3 else %58, %59 : i32, i64 to ^bb2
  ^bb3:  // pred: ^bb2
    %60 = "neura.data_mov"(%52) : (i32) -> i32
    %61 = "neura.data_mov"(%14) : (!llvm.ptr) -> !llvm.ptr
    "neura.store"(%60, %61) : (i32, !llvm.ptr) -> ()
    %62 = "neura.data_mov"(%8) : (i64) -> i64
    %63 = "neura.add"(%62) {rhs_value = 1 : i64} : (i64) -> i64
    %64 = "neura.data_mov"(%63) : (i64) -> i64
    %65 = "neura.icmp"(%64) <{cmpType = "eq"}> {rhs_value = 1024 : i64} : (i64) -> i1
    %66 = "neura.data_mov"(%65) : (i1) -> i1
    %67 = "neura.data_mov"(%63) : (i64) -> i64
    neura.cond_br %66 : i1 then to ^bb4 else %67 : i64 to ^bb1
  ^bb4:  // pred: ^bb3
    %68 = "neura.data_mov"(%6) : (i32) -> i32
    "neura.return"(%68) : (i32) -> ()
  }
}

