// RUN: neura-interpreter %s --verbose --dataflow | FileCheck %s

func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
  %0 = "neura.grant_once"() <{constant_value = 10 : i64}> : () -> !neura.data<i64, i1>
  %1 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %2 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %3 = "neura.grant_once"() <{constant_value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
  %4 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
  %5 = "neura.reserve"() : () -> (!neura.data<i64, i1>)
  %6 = "neura.phi"(%5, %0) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %7 = "neura.reserve"() : () -> (!neura.data<i64, i1>)
  %8 = "neura.phi"(%7, %2) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %9 = "neura.reserve"() : () -> (!neura.data<f32, i1>)
  %10 = "neura.phi"(%9, %3) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
  %11 = "neura.reserve"() : () -> (!neura.data<f32, i1>)
  %12 = "neura.phi"(%11, %4) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
  %13 = "neura.reserve"() : () -> (!neura.data<i64, i1>)
  %14 = "neura.phi"(%13, %1) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %15 = "neura.fadd"(%12, %10) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
  %16 = "neura.add"(%14, %8) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %17 = "neura.icmp"(%16, %6) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %18 = "neura.grant_predicate"(%16, %17) : (!neura.data<i64, i1>, !neura.data<i1, i1>) -> !neura.data<i64, i1>
  "neura.ctrl_mov"(%18, %13) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> ()
  %19 = "neura.grant_predicate"(%15, %17) : (!neura.data<f32, i1>, !neura.data<i1, i1>) -> !neura.data<f32, i1>
  "neura.ctrl_mov"(%19, %11) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> ()
  %20 = "neura.grant_predicate"(%3, %17) : (!neura.data<f32, i1>, !neura.data<i1, i1>) -> !neura.data<f32, i1>
  "neura.ctrl_mov"(%20, %9) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> ()
  %21 = "neura.grant_predicate"(%2, %17) : (!neura.data<i64, i1>, !neura.data<i1, i1>) -> !neura.data<i64, i1>
  "neura.ctrl_mov"(%21, %7) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> ()
  %22 = "neura.grant_predicate"(%0, %17) : (!neura.data<i64, i1>, !neura.data<i1, i1>) -> !neura.data<i64, i1>
  "neura.ctrl_mov"(%22, %5) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> ()
  %23 = "neura.not"(%17) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %24 = "neura.grant_predicate"(%15, %23) : (!neura.data<f32, i1>, !neura.data<i1, i1>) -> !neura.data<f32, i1>

  // CHECK: [neura-interpreter]  │  └─30.000000, [pred = 1] 

  "neura.return"(%24) : (!neura.data<f32, i1>) -> ()
}