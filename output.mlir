[mlir-neura-opt] Architecture specification file: test/arch_spec/architecture.yaml
[ctrl2data] Processing neura.return operation...
"neura.return"(%7) : (f32) -> ()
[ctrl2data] Marking neura.return with value...
[ctrl2data] Function is not void, no further action needed.
[ctrl2data] Asserting live-out values dominated by block arguments
[ctrl2data] All live-out values are dominated by block arguments or live-in values.
[ctrl2data] ReturnOp found: "neura.return"(%9) {return_type = "value"} : (!neura.data<f32, i1>) -> ()
[ctrl2data] Asserting live-out values dominated by block arguments
[ctrl2data] All live-out values are dominated by block arguments or live-in values.
[ctrl2data] Converting neura.return operations to return_void/value...
[ctrl2data] All neura.return operations converted successfully.
[ctrl2data] Set dataflow mode to predicate for function: loop_test
[ctrl2data] Converting phi operations to phi_start...
[MapToAcceleratorPass] Starting mapping pass...
[MapToAcceleratorPass] Using Mapping Mode: spatial-temporal
[MapToAcceleratorPass] Unknown YAML root key: extensions
[MapToAcceleratorPass] Unknown YAML root key: simulator
Collecting recurrence cycles from back edge: parent_op %16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>->%5 = neura.reserve : !neura.data<i64, i1>
Collecting recurrence cycles from back edge: parent_op %19 = neura.grant_predicate %17, %18 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>->%2 = neura.reserve : !neura.data<f32, i1>
[calculateResMii] Total operations: 13
Collecting recurrence cycles from back edge: parent_op %16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>->%5 = neura.reserve : !neura.data<i64, i1>
Collecting recurrence cycles from back edge: parent_op %19 = neura.grant_predicate %17, %18 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>->%2 = neura.reserve : !neura.data<f32, i1>
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
[MapToAcceleratorPass] ALAP sorted op: %23 = "neura.data_mov"(%21)[MapToAcceleratorPass] Start mapping with target II of 4
[calculateAward] Operation: %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>; Producers: 0
[DEBUG] Schedule op %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1> onto loc: tile#0 @t=0
[calculateAward] Operation: %7 = neura.phi_start %6, %5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>; Producers: 1
[DEBUG] Schedule op %7 = neura.phi_start %6, %5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> onto loc: tile#1 @t=1
Processing operand: %6 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
Reserving route for operation: %6 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
Path: link#0 @t=0 
[DEBUG] Successfully routed data move: %6 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> from tile#0 @t=0 to tile#1 @t=1
Processing operand: %5 = neura.reserve : !neura.data<i64, i1>
[calculateAward] Operation: %11 = "neura.add"(%10) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>; Producers: 1
[DEBUG] Schedule op %11 = "neura.add"(%10) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1> onto loc: tile#1 @t=2
Processing operand: %10 = "neura.data_mov"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
Reserving route for operation: %10 = "neura.data_mov"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
Path: register#32 @t=1 
[DEBUG] Successfully routed data move: %10 = "neura.data_mov"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> from tile#1 @t=1 to tile#1 @t=2
[calculateAward] Operation: %1 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>; Producers: 0
 : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
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
[HeuristicMapping] Trying cand[DEBUG] Schedule op %1 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1> onto loc: tile#0 @t=2
[calculateAward] Operation: %13 = "neura.icmp"(%12) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>; Producers: 1
[DEBUG] Schedule op %13 = "neura.icmp"(%12) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1> onto loc: tile#1 @t=3
Processing operand: %12 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
Reserving route for operation: %12 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
Path: register#32 @t=2 
[DEBUG] Successfully routed data move: %12 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> from tile#1 @t=2 to tile#1 @t=3
[calculateAward] Operation: %4 = neura.phi_start %3, %2 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>; Producers: 1
[DEBUG] Schedule op %4 = neura.phi_start %3, %2 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1> onto loc: tile#0 @t=3
Processing operand: %3 = "neura.data_mov"(%1) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
Reserving route for operation: %3 = "neura.data_mov"(%1) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
Path: register#0 @t=2 
[DEBUG] Successfully routed data move: %3 = "neura.data_mov"(%1) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> from tile#0 @t=2 to tile#0 @t=3
Processing operand: %2 = neura.reserve : !neura.data<f32, i1>
[calculateAward] Operation: %9 = "neura.fadd"(%8) {rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>; Producers: 1
[DEBUG] Schedule op %9 = "neura.fadd"(%8) {rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1> onto loc: tile#4 @t=4
Processing operand: %8 = "neura.data_mov"(%4) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
Reserving route for operation: %8 = "neura.data_mov"(%4) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
Path: link#1 @t=3 
[DEBUG] Successfully routed data move: %8 = "neura.data_mov"(%4) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> from tile#0 @t=3 to tile#4 @t=4
[calculateAward] Operation: %16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>; Producers: 2
[DEBUG] Schedule op %16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> onto loc: tile#1 @t=4
Processing operand: %14 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
Reserving route for operation: %14 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
Path: register#33 @t=2 register#33 @t=3 
[DEBUG] Successfully routed data move: %14 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> from tile#1 @t=2 to tile#1 @t=4
Processing operand: %15 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
Reserving route for operation: %15 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
Path: register#32 @t=3 
[DEBUG] Successfully routed data move: %15 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> from tile#1 @t=3 to tile#1 @t=4
[DEBUG] Found ctrl_mov user: neura.ctrl_mov %16 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
[tryRouteBackwardMove] src_loc: tile#1 @t=4, dst_loc: tile#1 @t=1
Reserving route for operation: neura.ctrl_mov %16 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
Path: register#32 @t=4 
[DEBUG] Successfully routed ctrl_mov: neura.ctrl_mov %16 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1> to tile#1 @t=1
[calculateAward] Operation: %21 = "neura.not"(%20) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>; Producers: 1
[DEBUG] Schedule op %21 = "neura.not"(%20) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> onto loc: tile#2 @t=4
Processing operand: %20 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
Reserving route for operation: %20 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
Path: link#3 @t=3 
[DEBUG] Successfully routed data move: %20 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> from tile#1 @t=3 to tile#2 @t=4
[calculateAward] Operation: %24 = neura.grant_predicate %22, %23 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>; Producers: 2
[DEBUG] Schedule op %24 = neura.grant_predicate %22, %23 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1> onto loc: tile#5 @t=6
Processing operand: %22 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
Reserving route for operation: %22 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
Path: link#10 @t=4 register#160 @t=5 
[DEBUG] Successfully routed data move: %22 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> from tile#4 @t=4 to tile#5 @t=6
Processing operand: %23 = "neura.data_mov"(%21) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
Reserving route for operation: %23 = "neura.data_mov"(%21) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
Path: link#5 @t=4 link#4 @t=5 
[DEBUG] Successfully routed data move: %23 = "neura.data_mov"(%21) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> from tile#2 @t=4 to tile#5 @t=6
[calculateAward] Operation: %19 = neura.grant_predicate %17, %18 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>; Producers: 2
[DEBUG] Schedule op %19 = neura.grant_predicate %17, %18 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1> onto loc: tile#4 @t=6
Processing operand: %17 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
Reserving route for operation: %17 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
Path: register#128 @t=4 register#128 @t=5 
[DEBUG] Successfully routed data move: %17 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> from tile#4 @t=4 to tile#4 @t=6
Processing operand: %18 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
Reserving route for operation: %18 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
Path: link#2 @t=3 link#1 @t=4 register#129 @t=5 
[DEBUG] Successfully routed data move: %18 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> from tile#1 @t=3 to tile#4 @t=6
[DEBUG] Found ctrl_mov user: neura.ctrl_mov %19 -> %2 : !neura.data<f32, i1> !neura.data<f32, i1>
[tryRouteBackwardMove] src_loc: tile#4 @t=6, dst_loc: tile#0 @t=3
Reserving route for operation: neura.ctrl_mov %19 -> %2 : !neura.data<f32, i1> !neura.data<f32, i1>
Path: link#11 @t=6 
[DEBUG] Successfully routed ctrl_mov: neura.ctrl_mov %19 -> %2 : !neura.data<f32, i1> !neura.data<f32, i1> to tile#0 @t=3
[calculateAward] Operation: neura.return_value %25 : !neura.data<f32, i1>; Producers: 1
[DEBUG] Schedule op neura.return_value %25 : !neura.data<f32, i1> onto loc: tile#5 @t=7
Processing operand: %25 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
idate 1/1 at tile#0 @t=2
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
[Reserving route for operation: %25 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
Path: register#160 @t=6 
[DEBUG] Successfully routed data move: %25 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> from tile#5 @t=6 to tile#5 @t=7
=== MappingState: Resource Allocation Table ===
II = 4

Tile     | t%4=0                               | t%4=1                               | t%4=2                               | t%4=3                               | 
---------+-------------------------------------+-------------------------------------+-------------------------------------+-------------------------------------+
Tile#0   | %0 = grant_once                     |                                     | %1 = grant_once                     | %4 = phi_start(%3, %2)              | 
Tile#1   | %16 = grant_predicate(%14, %15) ... | %7 = phi_start(%6, %5)              | %11 = add(%10)                      | %13 = icmp(%12)                     | 
Tile#2   | %21 = not(%20) (t=4)                |                                     |                                     |                                     | 
Tile#4   | %9 = fadd(%8) (t=4)                 |                                     | %19 = grant_predicate(%17, %18) ... |                                     | 
Tile#5   |                                     |                                     | %24 = grant_predicate(%22, %23) ... | return_value(%25) (t=7)             | 

=== Legend ===
- Table shows operations mapped to tiles (modulo II scheduling)
- Column headers: t%II=X means time slot X (t=X, X+II, X+2*II, ...)
- Operations with (t=Y) annotation are scheduled at actual time step Y
- Operations without annotation are scheduled at t=0 to t=3
=== End ===
Collecting recurrence cycles from back edge: parent_op %16 = neura.grant_predicate %14, %15 {mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>->%5 = neura.reserve : !neura.data<i64, i1>
Collecting recurrence cycles from back edge: parent_op %19 = neura.grant_predicate %17, %18 {mapping_locs = [{id = 4 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>->%2 = neura.reserve : !neura.data<f32, i1>
[MapToAcceleratorPass] Assigned dfg_id=0 to %0 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 0 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
[MapToAcceleratorPass] Assigned dfg_id=1 to %1 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> {dfg_id = 1 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<f32, i1>
[MapToAcceleratorPass] Assigned dfg_id=2 to %2 = neura.reserve {dfg_id = 2 : i32} : !neura.data<f32, i1>
[MapToAcceleratorPass] Assigned dfg_id=3 to %5 = neura.reserve {dfg_id = 3 : i32} : !neura.data<i64, i1>
[MapToAcceleratorPass] Assigned dfg_id=4 to neura.yield {dfg_id = 4 : i32}
[MapToAcceleratorPass] Assigned dfg_id=5 to %6 = "neura.data_mov"(%0) {dfg_id = 5 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Assigned dfg_id=6 to %3 = "neura.data_mov"(%1) {dfg_id = 6 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Assigned dfg_id=7 to %7 = neura.phi_start %6, %5 {dfg_id = 7 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Assigned dfg_id=8 to %4 = neura.phi_start %3, %2 {dfg_id = 8 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
[MapToAcceleratorPass] Assigned dfg_id=9 to %10 = "neura.data_mov"(%7) {dfg_id = 9 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Assigned dfg_id=10 to %8 = "neura.data_mov"(%4) {dfg_id = 10 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Assigned dfg_id=11 to %11 = "neura.add"(%10) {dfg_id = 11 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 0 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Assigned dfg_id=12 to %9 = "neura.fadd"(%8) {dfg_id = 12 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 1 : i32}], rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Assigned dfg_id=13 to %14 = "neura.data_mov"(%11) {dfg_id = 13 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 2 : i32}, {id = 33 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Assigned dfg_id=14 to %12 = "neura.data_mov"(%11) {dfg_id = 14 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Assigned dfg_id=15 to %22 = "neura.data_mov"(%9) {dfg_id = 15 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 4 : i32}, {id = 160 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Assigned dfg_id=16 to %17 = "neura.data_mov"(%9) {dfg_id = 16 : i32, mapping_locs = [{id = 128 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 128 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Assigned dfg_id=17 to %13 = "neura.icmp"(%12) <{cmpType = "slt"}> {dfg_id = 17 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 0 : i32}], rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Assigned dfg_id=18 to %20 = "neura.data_mov"(%13) {dfg_id = 18 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Assigned dfg_id=19 to %18 = "neura.data_mov"(%13) {dfg_id = 19 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 4 : i32}, {id = 129 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Assigned dfg_id=20 to %15 = "neura.data_mov"(%13) {dfg_id = 20 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Assigned dfg_id=21 to %21 = "neura.not"(%20) {dfg_id = 21 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Assigned dfg_id=22 to %19 = neura.grant_predicate %17, %18 {dfg_id = 22 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
[MapToAcceleratorPass] Assigned dfg_id=23 to %16 = neura.grant_predicate %14, %15 {dfg_id = 23 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Assigned dfg_id=24 to %23 = "neura.data_mov"(%21) {dfg_id = 24 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 4 : i32}, {id = 4 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Assigned dfg_id=25 to neura.ctrl_mov %19 -> %2 {dfg_id = 25 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
[MapToAcceleratorPass] Assigned dfg_id=26 to neura.ctrl_mov %16 -> %5 {dfg_id = 26 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Assigned dfg_id=27 to %24 = neura.grant_predicate %22, %23 {dfg_id = 27 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
[MapToAcceleratorPass] Assigned dfg_id=28 to %25 = "neura.data_mov"(%24) {dfg_id = 28 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Assigned dfg_id=29 to neura.return_value %25 : !neura.data<f32, i1> {dfg_id = 29 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 1 : i32}]}
[MapToAcceleratorPass] Assigned 30 dfg_id(s) in total
tryRouteDataMove] Successfully routed on same tile using Register #160
[HeuristicMapping] Successfully mapped operation neura.return_value %25 : !neura.data<f32, i1>
[HeuristicMapping] Successfully mapped all 12 operations.
[generate-code] DFG (SSA-based) emitted: nodes=17, edges=19 -> tmp-generated-dfg.dot, tmp-generated-dfg.yaml
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

