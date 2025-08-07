[DEBUG] Recurrence cycle (length 3):
  %27 = neura.reserve : !neura.data<i64, i1>
  %29 = "neura.phi"(%27, %28) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %33 = "neura.data_mov"(%29) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %35 = "neura.add"(%33, %34) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %39 = "neura.data_mov"(%35) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %41 = neura.grant_predicate %39, %40 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %41 -> %27 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 4):
  %27 = neura.reserve : !neura.data<i64, i1>
  %29 = "neura.phi"(%27, %28) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %33 = "neura.data_mov"(%29) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %35 = "neura.add"(%33, %34) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %36 = "neura.data_mov"(%35) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %38 = "neura.icmp"(%36, %37) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %40 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %41 = neura.grant_predicate %39, %40 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %41 -> %27 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 3):
  %24 = neura.reserve : !neura.data<f32, i1>
  %26 = "neura.phi"(%24, %25) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
  %30 = "neura.data_mov"(%26) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %32 = "neura.fadd"(%30, %31) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
  %42 = "neura.data_mov"(%32) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %44 = neura.grant_predicate %42, %43 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
  neura.ctrl_mov %44 -> %24 : !neura.data<f32, i1> !neura.data<f32, i1>
[DEBUG] Recurrence cycle (length 4):
  %18 = neura.reserve : !neura.data<i64, i1>
  %20 = "neura.phi"(%18, %19) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %34 = "neura.data_mov"(%20) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %35 = "neura.add"(%33, %34) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %36 = "neura.data_mov"(%35) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %38 = "neura.icmp"(%36, %37) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %49 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %50 = neura.grant_predicate %48, %49 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %50 -> %18 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 3):
  %15 = neura.reserve : !neura.data<i64, i1>
  %17 = "neura.phi"(%15, %16) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %37 = "neura.data_mov"(%17) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %38 = "neura.icmp"(%36, %37) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %52 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %53 = neura.grant_predicate %51, %52 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %53 -> %15 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Longest recurrence cycle (length 4):
%27 = neura.reserve : !neura.data<i64, i1>
%29 = "neura.phi"(%27, %28) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
%33 = "neura.data_mov"(%29) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
%35 = "neura.add"(%33, %34) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
%36 = "neura.data_mov"(%35) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
%38 = "neura.icmp"(%36, %37) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
%40 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
%41 = neura.grant_predicate %39, %40 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
neura.ctrl_mov %41 -> %27 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> : () -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %3 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %6 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %9 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %12 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %15 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %18 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %21 = neura.reserve : !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %24 = neura.reserve : !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %27 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %1 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %4 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %7 = "neura.data_mov"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %10 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %13 = "neura.data_mov"(%12) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %2 = "neura.grant_once"(%1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %5 = "neura.grant_once"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %8 = "neura.grant_once"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %11 = "neura.grant_once"(%10) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %14 = "neura.grant_once"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %51 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %16 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %28 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %48 = "neura.data_mov"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %19 = "neura.data_mov"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %45 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %22 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %25 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %17 = "neura.phi"(%15, %16) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %29 = "neura.phi"(%27, %28) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %20 = "neura.phi"(%18, %19) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %23 = "neura.phi"(%21, %22) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %26 = "neura.phi"(%24, %25) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %37 = "neura.data_mov"(%17) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %33 = "neura.data_mov"(%29) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %34 = "neura.data_mov"(%20) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %31 = "neura.data_mov"(%23) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %30 = "neura.data_mov"(%26) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %35 = "neura.add"(%33, %34) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %32 = "neura.fadd"(%30, %31) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %39 = "neura.data_mov"(%35) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %36 = "neura.data_mov"(%35) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %56 = "neura.data_mov"(%32) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %42 = "neura.data_mov"(%32) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %38 = "neura.icmp"(%36, %37) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %54 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %52 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %49 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %46 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %43 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %40 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %55 = "neura.not"(%54) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %53 = neura.grant_predicate %51, %52 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %50 = neura.grant_predicate %48, %49 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %47 = neura.grant_predicate %45, %46 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %44 = neura.grant_predicate %42, %43 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %41 = neura.grant_predicate %39, %40 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %57 = "neura.data_mov"(%55) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %53 -> %15 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %50 -> %18 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %47 -> %21 : !neura.data<f32, i1> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %44 -> %24 : !neura.data<f32, i1> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %41 -> %27 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %58 = neura.grant_predicate %56, %57 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: %59 = "neura.data_mov"(%58) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] Topologically sorted op: "neura.return"(%59) : (!neura.data<f32, i1>) -> ()
[MapToAcceleratorPass] ALAP Bucket Level 0: 4 ops
  %3 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
  %6 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
  %4 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %7 = "neura.data_mov"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 1: 8 ops
  %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> : () -> !neura.data<i64, i1>
  %18 = neura.reserve : !neura.data<i64, i1>
  %27 = neura.reserve : !neura.data<i64, i1>
  %1 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %5 = "neura.grant_once"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %8 = "neura.grant_once"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %28 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %19 = "neura.data_mov"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 2: 11 ops
  %9 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
  %12 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
  %15 = neura.reserve : !neura.data<i64, i1>
  %10 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %13 = "neura.data_mov"(%12) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %2 = "neura.grant_once"(%1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %16 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %29 = "neura.phi"(%27, %28) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %20 = "neura.phi"(%18, %19) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %33 = "neura.data_mov"(%29) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %34 = "neura.data_mov"(%20) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 3: 11 ops
  %21 = neura.reserve : !neura.data<f32, i1>
  %24 = neura.reserve : !neura.data<f32, i1>
  %11 = "neura.grant_once"(%10) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %14 = "neura.grant_once"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %22 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %25 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %17 = "neura.phi"(%15, %16) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %37 = "neura.data_mov"(%17) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %35 = "neura.add"(%33, %34) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %39 = "neura.data_mov"(%35) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %36 = "neura.data_mov"(%35) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 4: 9 ops
  %23 = "neura.phi"(%21, %22) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
  %26 = "neura.phi"(%24, %25) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
  %31 = "neura.data_mov"(%23) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %30 = "neura.data_mov"(%26) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %38 = "neura.icmp"(%36, %37) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %54 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %52 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %49 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %40 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] ALAP Bucket Level 5: 7 ops
  %32 = "neura.fadd"(%30, %31) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
  %56 = "neura.data_mov"(%32) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %42 = "neura.data_mov"(%32) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %55 = "neura.not"(%54) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %41 = neura.grant_predicate %39, %40 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %57 = "neura.data_mov"(%55) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  neura.ctrl_mov %41 -> %27 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 6: 7 ops
  %51 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %48 = "neura.data_mov"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %45 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %46 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %43 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %58 = neura.grant_predicate %56, %57 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
  %59 = "neura.data_mov"(%58) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] ALAP Bucket Level 7: 9 ops
  %53 = neura.grant_predicate %51, %52 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %50 = neura.grant_predicate %48, %49 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %47 = neura.grant_predicate %45, %46 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
  %44 = neura.grant_predicate %42, %43 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
  neura.ctrl_mov %53 -> %15 : !neura.data<i64, i1> !neura.data<i64, i1>
  neura.ctrl_mov %50 -> %18 : !neura.data<i64, i1> !neura.data<i64, i1>
  neura.ctrl_mov %47 -> %21 : !neura.data<f32, i1> !neura.data<f32, i1>
  neura.ctrl_mov %44 -> %24 : !neura.data<f32, i1> !neura.data<f32, i1>
  "neura.return"(%59) : (!neura.data<f32, i1>) -> ()
[MapToAcceleratorPass] ALAP sorted op: %3 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %6 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %4 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %7 = "neura.data_mov"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> : () -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %18 = neura.reserve : !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %27 = neura.reserve : !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %1 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %5 = "neura.grant_once"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %8 = "neura.grant_once"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %28 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %19 = "neura.data_mov"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %9 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %12 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %15 = neura.reserve : !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %10 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %13 = "neura.data_mov"(%12) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %2 = "neura.grant_once"(%1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %16 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %29 = "neura.phi"(%27, %28) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %20 = "neura.phi"(%18, %19) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %33 = "neura.data_mov"(%29) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %34 = "neura.data_mov"(%20) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %21 = neura.reserve : !neura.data<f32, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %24 = neura.reserve : !neura.data<f32, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %11 = "neura.grant_once"(%10) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %14 = "neura.grant_once"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %22 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %25 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %17 = "neura.phi"(%15, %16) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %37 = "neura.data_mov"(%17) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %35 = "neura.add"(%33, %34) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %39 = "neura.data_mov"(%35) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %36 = "neura.data_mov"(%35) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %23 = "neura.phi"(%21, %22) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %26 = "neura.phi"(%24, %25) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %31 = "neura.data_mov"(%23) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %30 = "neura.data_mov"(%26) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %38 = "neura.icmp"(%36, %37) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %54 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %52 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %49 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %40 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %32 = "neura.fadd"(%30, %31) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %56 = "neura.data_mov"(%32) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %42 = "neura.data_mov"(%32) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %55 = "neura.not"(%54) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %41 = neura.grant_predicate %39, %40 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %57 = "neura.data_mov"(%55) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %41 -> %27 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %51 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %48 = "neura.data_mov"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %45 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %46 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %43 = "neura.data_mov"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %58 = neura.grant_predicate %56, %57 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %59 = "neura.data_mov"(%58) : (!neura.data<f32, i1>) -> !neura.data<f32, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %53 = neura.grant_predicate %51, %52 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %50 = neura.grant_predicate %48, %49 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %47 = neura.grant_predicate %45, %46 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %44 = neura.grant_predicate %42, %43 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %53 -> %15 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %50 -> %18 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %47 -> %21 : !neura.data<f32, i1> !neura.data<f32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %44 -> %24 : !neura.data<f32, i1> !neura.data<f32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: "neura.return"(%59) : (!neura.data<f32, i1>) -> () (ALAP level: 7)
module {
  func.func @loop_test() -> f32 attributes {CompiledII = 6 : i32, RecMII = 4 : i32, ResMII = 2 : i32, accelerator = "neura"} {
    %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
    %1 = "neura.data_mov"(%0) {mapping_locs = [{id = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %2 = "neura.grant_once"(%1) {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %3 = "neura.constant"() <{predicate = true, value = 0 : i64}> {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
    %4 = "neura.data_mov"(%3) {mapping_locs = [{id = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %5 = "neura.grant_once"(%4) {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %6 = "neura.constant"() <{predicate = true, value = 1 : i64}> {mapping_locs = [{id = 2 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
    %7 = "neura.data_mov"(%6) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %8 = "neura.grant_once"(%7) {mapping_locs = [{id = 2 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %9 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 3 : i32}]} : () -> !neura.data<f32, i1>
    %10 = "neura.data_mov"(%9) {mapping_locs = [{id = 45 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %11 = "neura.grant_once"(%10) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %12 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> {mapping_locs = [{id = 11 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<f32, i1>
    %13 = "neura.data_mov"(%12) {mapping_locs = [{id = 36 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %14 = "neura.grant_once"(%13) {mapping_locs = [{id = 7 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 1 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %15 = neura.reserve : !neura.data<i64, i1>
    %16 = "neura.data_mov"(%2) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %17 = "neura.phi"(%15, %16) {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %18 = neura.reserve : !neura.data<i64, i1>
    %19 = "neura.data_mov"(%8) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %20 = "neura.phi"(%18, %19) {mapping_locs = [{id = 2 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %21 = neura.reserve : !neura.data<f32, i1>
    %22 = "neura.data_mov"(%11) {mapping_locs = [{id = 34 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %23 = "neura.phi"(%21, %22) {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
    %24 = neura.reserve : !neura.data<f32, i1>
    %25 = "neura.data_mov"(%14) {mapping_locs = [{id = 21 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %26 = "neura.phi"(%24, %25) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
    %27 = neura.reserve : !neura.data<i64, i1>
    %28 = "neura.data_mov"(%5) {mapping_locs = [{id = 4 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %29 = "neura.phi"(%27, %28) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %30 = "neura.data_mov"(%26) {mapping_locs = [{id = 20 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %31 = "neura.data_mov"(%23) {mapping_locs = [{id = 45 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %32 = "neura.fadd"(%30, %31) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
    %33 = "neura.data_mov"(%29) {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %34 = "neura.data_mov"(%20) {mapping_locs = [{id = 7 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %35 = "neura.add"(%33, %34) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %36 = "neura.data_mov"(%35) {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %37 = "neura.data_mov"(%17) {mapping_locs = [{id = 4 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %38 = "neura.icmp"(%36, %37) <{cmpType = "slt"}> {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
    %39 = "neura.data_mov"(%35) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %40 = "neura.data_mov"(%38) {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %41 = neura.grant_predicate %39, %40 {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %41 -> %27 {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 5 : i32}, {id = 20 : i32, resource = "register", time_step = 6 : i32}, {id = 20 : i32, resource = "register", time_step = 7 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %42 = "neura.data_mov"(%32) {mapping_locs = [{id = 33 : i32, resource = "link", time_step = 5 : i32}, {id = 17 : i32, resource = "link", time_step = 6 : i32}, {id = 21 : i32, resource = "register", time_step = 7 : i32}, {id = 21 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %43 = "neura.data_mov"(%38) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %44 = neura.grant_predicate %42, %43 {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
    neura.ctrl_mov %44 -> %24 {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 9 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
    %45 = "neura.data_mov"(%11) {mapping_locs = [{id = 31 : i32, resource = "link", time_step = 3 : i32}, {id = 27 : i32, resource = "link", time_step = 4 : i32}, {id = 32 : i32, resource = "register", time_step = 5 : i32}, {id = 32 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %46 = "neura.data_mov"(%38) {mapping_locs = [{id = 13 : i32, resource = "link", time_step = 4 : i32}, {id = 12 : i32, resource = "link", time_step = 5 : i32}, {id = 33 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %47 = neura.grant_predicate %45, %46 {mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 7 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
    neura.ctrl_mov %47 -> %21 {mapping_locs = [{id = 24 : i32, resource = "link", time_step = 7 : i32}, {id = 30 : i32, resource = "link", time_step = 8 : i32}, {id = 41 : i32, resource = "link", time_step = 9 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
    %48 = "neura.data_mov"(%8) {mapping_locs = [{id = 7 : i32, resource = "link", time_step = 1 : i32}, {id = 24 : i32, resource = "register", time_step = 2 : i32}, {id = 24 : i32, resource = "register", time_step = 3 : i32}, {id = 24 : i32, resource = "register", time_step = 4 : i32}, {id = 24 : i32, resource = "register", time_step = 5 : i32}, {id = 24 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %49 = "neura.data_mov"(%38) {mapping_locs = [{id = 16 : i32, resource = "link", time_step = 4 : i32}, {id = 28 : i32, resource = "link", time_step = 5 : i32}, {id = 33 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %50 = neura.grant_predicate %48, %49 {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %50 -> %18 {mapping_locs = [{id = 19 : i32, resource = "link", time_step = 7 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %51 = "neura.data_mov"(%2) {mapping_locs = [{id = 4 : i32, resource = "link", time_step = 2 : i32}, {id = 21 : i32, resource = "register", time_step = 3 : i32}, {id = 21 : i32, resource = "register", time_step = 4 : i32}, {id = 21 : i32, resource = "register", time_step = 5 : i32}, {id = 21 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %52 = "neura.data_mov"(%38) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %53 = neura.grant_predicate %51, %52 {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %53 -> %15 {mapping_locs = [{id = 15 : i32, resource = "link", time_step = 7 : i32}, {id = 4 : i32, resource = "register", time_step = 8 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %54 = "neura.data_mov"(%38) {mapping_locs = [{id = 15 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %55 = "neura.not"(%54) {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %56 = "neura.data_mov"(%32) {mapping_locs = [{id = 31 : i32, resource = "link", time_step = 5 : i32}, {id = 36 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %57 = "neura.data_mov"(%55) {mapping_locs = [{id = 4 : i32, resource = "link", time_step = 5 : i32}, {id = 16 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %58 = neura.grant_predicate %56, %57 {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
    %59 = "neura.data_mov"(%58) {mapping_locs = [{id = 30 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    "neura.return"(%59) {mapping_locs = [{id = 13 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 3 : i32}]} : (!neura.data<f32, i1>) -> ()
  }
}

