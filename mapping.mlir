[DEBUG] Recurrence cycle (length 3):
  %5 = neura.reserve : !neura.data<i64, i1>
  %7 = neura.phi_start %6, %5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %10 = "neura.data_mov"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %11 = "neura.add"(%10) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %14 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %16 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 4):
  %5 = neura.reserve : !neura.data<i64, i1>
  %7 = neura.phi_start %6, %5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %10 = "neura.data_mov"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %11 = "neura.add"(%10) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %12 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %13 = "neura.icmp"(%12) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
  %15 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %16 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 3):
  %2 = neura.reserve : !neura.data<f32, i1>
  %4 = neura.phi_start %3, %2 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
  %8 = "neura.data_mov"(%4) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %9 = "neura.fadd"(%8) {rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %17 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %19 = neura.grant_predicate %17, %18 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
  neura.ctrl_mov %19 -> %2 : !neura.data<f32, i1> !neura.data<f32, i1>
[MapToAcceleratorPass] Longest recurrence cycle (length 4):
%5 = neura.reserve : !neura.data<i64, i1>
%7 = neura.phi_start %6, %5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
%10 = "neura.data_mov"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
%11 = "neura.add"(%10) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
%12 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
%13 = "neura.icmp"(%12) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
%15 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
%16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
neura.ctrl_mov %16 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %1 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %2 = neura.reserve : !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %5 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.yield
[MapToAcceleratorPass] Topologically sorted op: %6 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %3 = "neura.data_mov"(%1) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %7 = neura.phi_start %6, %5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %4 = neura.phi_start %3, %2 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %10 = "neura.data_mov"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %8 = "neura.data_mov"(%4) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %11 = "neura.add"(%10) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %9 = "neura.fadd"(%8) {rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %14 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %12 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %22 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %17 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %13 = "neura.icmp"(%12) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %20 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %18 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %15 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %21 = "neura.not"(%20) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %19 = neura.grant_predicate %17, %18 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %23 = "neura.data_mov"(%21) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %19 -> %2 : !neura.data<f32, i1> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %16 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %24 = neura.grant_predicate %22, %23 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %25 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.return_value %25 : !neura.data<f32, i1>
[MapToAcceleratorPass] ALAP Bucket Level 0: 3 ops
  %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %5 = neura.reserve : !neura.data<i64, i1>
  %6 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 1: 2 ops
  %7 = neura.phi_start %6, %5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %10 = "neura.data_mov"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 2: 6 ops
  %1 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
  %2 = neura.reserve : !neura.data<f32, i1>
  %3 = "neura.data_mov"(%1) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %11 = "neura.add"(%10) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %14 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %12 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 3: 5 ops
  %4 = neura.phi_start %3, %2 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
  %8 = "neura.data_mov"(%4) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %13 = "neura.icmp"(%12) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
  %20 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %15 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] ALAP Bucket Level 4: 7 ops
  %9 = "neura.fadd"(%8) {rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %22 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %17 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %21 = "neura.not"(%20) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %23 = "neura.data_mov"(%21) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  neura.ctrl_mov %16 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 5: 3 ops
  %18 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %24 = neura.grant_predicate %22, %23 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
  %25 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] ALAP Bucket Level 6: 4 ops
  neura.yield
  %19 = neura.grant_predicate %17, %18 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
  neura.ctrl_mov %19 -> %2 : !neura.data<f32, i1> !neura.data<f32, i1>
  neura.return_value %25 : !neura.data<f32, i1>
[MapToAcceleratorPass] ALAP sorted op: %5 = neura.reserve : !neura.data<i64, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %6 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %7 = neura.phi_start %6, %5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %10 = "neura.data_mov"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %11 = "neura.add"(%10) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %2 = neura.reserve : !neura.data<f32, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %3 = "neura.data_mov"(%1) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %14 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %12 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %1 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %13 = "neura.icmp"(%12) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %4 = neura.phi_start %3, %2 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %8 = "neura.data_mov"(%4) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %20 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %15 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %9 = "neura.fadd"(%8) {rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %22 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %17 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %21 = "neura.not"(%20) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %23 = "neura.data_mov"(%21) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %16 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %24 = neura.grant_predicate %22, %23 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %18 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %25 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %19 = neura.grant_predicate %17, %18 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %19 -> %2 : !neura.data<f32, i1> !neura.data<f32, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: neura.return_value %25 : !neura.data<f32, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: neura.yield (ALAP level: 6)
---------------------------------------------------------
[HeuristicMapping] Starting mapping with 30 operations.
Configuration: MAX Backtrack Depth = 1, MAX Candidate Locations = 1
[HeuristicMapping] Filtered 18 non-materialized operations, 12 operations require physical mapping.
[HeuristicMapping] Materialized operations list:
0 %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1> (level: 0)
1 %7 = neura.phi_start %6, %5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 1)
2 %11 = "neura.add"(%10) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 2)
3 %1 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1> (level: 2)
4 %13 = "neura.icmp"(%12) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1> (level: 3)
5 %4 = neura.phi_start %3, %2 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1> (level: 3)
6 %9 = "neura.fadd"(%8) {rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (level: 4)
7 %16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 4)
8 %21 = "neura.not"(%20) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (level: 4)
9 %24 = neura.grant_predicate %22, %23 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1> (level: 5)
10 %19 = neura.grant_predicate %17, %18 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1> (level: 6)
11 neura.return_value %25 : !neura.data<f32, i1> (level: 6)
[HeuristicMapping] Found 64 candidate locations for operation: %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=0
[HeuristicMapping] Successfully mapped operation %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 31 candidate locations for operation: %7 = neura.phi_start %6, %5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=1
[tryRouteDataMove] Routing from Tile#0 @t=0 to Tile#1 @t=1
[HeuristicMapping] Successfully mapped operation %7 = neura.phi_start %6, %5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 37 candidate locations for operation: %11 = "neura.add"(%10) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=2
[tryRouteDataMove] Routing from Tile#1 @t=1 to Tile#1 @t=2
[tryRouteDataMove] Successfully routed on same tile using Register #32
[HeuristicMapping] Successfully mapped operation %11 = "neura.add"(%10) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 61 candidate locations for operation: %1 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=2
[HeuristicMapping] Successfully mapped operation %1 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
[HeuristicMapping] Found 35 candidate locations for operation: %13 = "neura.icmp"(%12) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=3
[tryRouteDataMove] Routing from Tile#1 @t=2 to Tile#1 @t=3
[tryRouteDataMove] Successfully routed on same tile using Register #32
[HeuristicMapping] Successfully mapped operation %13 = "neura.icmp"(%12) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 27 candidate locations for operation: %4 = neura.phi_start %3, %2 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=3
[tryRouteDataMove] Routing from Tile#0 @t=2 to Tile#0 @t=3
[tryRouteDataMove] Successfully routed on same tile using Register #0
[HeuristicMapping] Successfully mapped operation %4 = neura.phi_start %3, %2 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
[HeuristicMapping] Found 26 candidate locations for operation: %9 = "neura.fadd"(%8) {rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=4
[tryRouteDataMove] Routing from Tile#0 @t=3 to Tile#4 @t=4
[HeuristicMapping] Successfully mapped operation %9 = "neura.fadd"(%8) {rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[HeuristicMapping] Found 3 candidate locations for operation: %16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=4
[tryRouteDataMove] Routing from Tile#1 @t=2 to Tile#1 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #33
[tryRouteDataMove] Routing from Tile#1 @t=3 to Tile#1 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #32
[tryRouteDataMove] Routing from Tile#1 @t=4 to Tile#1 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #32
[HeuristicMapping] Successfully mapped operation %16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 32 candidate locations for operation: %21 = "neura.not"(%20) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=4
[tryRouteDataMove] Routing from Tile#1 @t=3 to Tile#2 @t=4
[HeuristicMapping] Successfully mapped operation %21 = "neura.not"(%20) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 19 candidate locations for operation: %24 = neura.grant_predicate %22, %23 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=6
[tryRouteDataMove] Routing from Tile#4 @t=4 to Tile#5 @t=6
[tryRouteDataMove] Routing from Tile#2 @t=4 to Tile#5 @t=6
[HeuristicMapping] Successfully mapped operation %24 = neura.grant_predicate %22, %23 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
[HeuristicMapping] Found 1 candidate locations for operation: %19 = neura.grant_predicate %17, %18 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=6
[tryRouteDataMove] Routing from Tile#4 @t=4 to Tile#4 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #128
[tryRouteDataMove] Routing from Tile#1 @t=3 to Tile#4 @t=6
[tryRouteDataMove] Routing from Tile#4 @t=6 to Tile#0 @t=7
[HeuristicMapping] Successfully mapped operation %19 = neura.grant_predicate %17, %18 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
[HeuristicMapping] Found 37 candidate locations for operation: neura.return_value %25 : !neura.data<f32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=6 to Tile#5 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #160
[HeuristicMapping] Successfully mapped operation neura.return_value %25 : !neura.data<f32, i1>
[HeuristicMapping] Successfully mapped all 12 operations.
module {
  func.func @loop_test() -> f32 attributes {accelerator = "neura", dataflow_mode = "predicate", mapping_info = {compiled_ii = 4 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 4 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
    %0 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 0 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
    %1 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> {dfg_id = 1 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<f32, i1>
    %2 = neura.reserve {dfg_id = 2 : i32} : !neura.data<f32, i1>
    %3 = "neura.data_mov"(%1) {dfg_id = 6 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %4 = neura.phi_start %3, %2 {dfg_id = 8 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
    %5 = neura.reserve {dfg_id = 3 : i32} : !neura.data<i64, i1>
    %6 = "neura.data_mov"(%0) {dfg_id = 5 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %7 = neura.phi_start %6, %5 {dfg_id = 7 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
    %8 = "neura.data_mov"(%4) {dfg_id = 10 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %9 = "neura.fadd"(%8) {dfg_id = 12 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 1 : i32}], rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %10 = "neura.data_mov"(%7) {dfg_id = 9 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %11 = "neura.add"(%10) {dfg_id = 11 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 0 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %12 = "neura.data_mov"(%11) {dfg_id = 14 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %13 = "neura.icmp"(%12) <{cmpType = "slt"}> {dfg_id = 17 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 0 : i32}], rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
    %14 = "neura.data_mov"(%11) {dfg_id = 13 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 2 : i32}, {id = 33 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %15 = "neura.data_mov"(%13) {dfg_id = 20 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %16 = neura.grant_predicate %14, %15 {dfg_id = 23 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %16 -> %5 {dfg_id = 26 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %17 = "neura.data_mov"(%9) {dfg_id = 16 : i32, mapping_locs = [{id = 128 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 128 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %18 = "neura.data_mov"(%13) {dfg_id = 19 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 4 : i32}, {id = 129 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %19 = neura.grant_predicate %17, %18 {dfg_id = 22 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
    neura.ctrl_mov %19 -> %2 {dfg_id = 25 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
    %20 = "neura.data_mov"(%13) {dfg_id = 18 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %21 = "neura.not"(%20) {dfg_id = 21 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %22 = "neura.data_mov"(%9) {dfg_id = 15 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 4 : i32}, {id = 160 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %23 = "neura.data_mov"(%21) {dfg_id = 24 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 4 : i32}, {id = 4 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %24 = neura.grant_predicate %22, %23 {dfg_id = 27 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
    %25 = "neura.data_mov"(%24) {dfg_id = 28 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    neura.return_value %25 : !neura.data<f32, i1> {dfg_id = 29 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 1 : i32}]}
    neura.yield {dfg_id = 4 : i32}
  }
}

