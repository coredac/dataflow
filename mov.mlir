module {
  func.func @loop_test() -> f32 attributes {accelerator = "neura", dataflow_mode = "predicate"} {
    %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
    %1 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
    %2 = neura.reserve : !neura.data<f32, i1>
    %3 = "neura.data_mov"(%1) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %4 = neura.phi_start %3, %2 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
    %5 = neura.reserve : !neura.data<i64, i1>
    %6 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %7 = neura.phi_start %6, %5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
    %8 = "neura.data_mov"(%4) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %9 = "neura.fadd"(%8) {rhs_value = 3.000000e+00 : f32} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %10 = "neura.data_mov"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %11 = "neura.add"(%10) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %12 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %13 = "neura.icmp"(%12) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
    %14 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %15 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %16 = neura.grant_predicate %14, %15 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %16 -> %5 : !neura.data<i64, i1> !neura.data<i64, i1>
    %17 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %18 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %19 = neura.grant_predicate %17, %18 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
    neura.ctrl_mov %19 -> %2 : !neura.data<f32, i1> !neura.data<f32, i1>
    %20 = "neura.data_mov"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %21 = "neura.not"(%20) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %22 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %23 = "neura.data_mov"(%21) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %24 = neura.grant_predicate %22, %23 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
    %25 = "neura.data_mov"(%24) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    neura.return_value %25 : !neura.data<f32, i1>
    neura.yield
  }
}

