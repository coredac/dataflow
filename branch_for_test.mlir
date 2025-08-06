[DEBUG] Recurrence cycle (length 3):
  %18 = neura.reserve : !neura.data<i64, i1>
  %20 = "neura.phi"(%18, %19) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %24 = "neura.data_mov"(%20) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %26 = "neura.add"(%24, %25) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %30 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %32 = neura.grant_predicate %30, %31 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %32 -> %18 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 4):
  %18 = neura.reserve : !neura.data<i64, i1>
  %20 = "neura.phi"(%18, %19) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %24 = "neura.data_mov"(%20) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %26 = "neura.add"(%24, %25) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %27 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %29 = "neura.icmp"(%27, %28) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %31 = "neura.data_mov"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %32 = neura.grant_predicate %30, %31 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %32 -> %18 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 3):
  %15 = neura.reserve : !neura.data<f32, i1>
  %17 = "neura.phi"(%15, %16) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
  %21 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %23 = "neura.fadd"(%21, %22) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
  %33 = "neura.data_mov"(%23) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  %35 = neura.grant_predicate %33, %34 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
  neura.ctrl_mov %35 -> %15 : !neura.data<f32, i1> !neura.data<f32, i1>
[MapToAcceleratorPass] Longest recurrence cycle (length 4):
%18 = neura.reserve : !neura.data<i64, i1>
%20 = "neura.phi"(%18, %19) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
%24 = "neura.data_mov"(%20) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
%26 = "neura.add"(%24, %25) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
%27 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
%29 = "neura.icmp"(%27, %28) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
%31 = "neura.data_mov"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
%32 = neura.grant_predicate %30, %31 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
neura.ctrl_mov %32 -> %18 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> : () -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %3 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %6 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %9 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: %12 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: %15 = neura.reserve : !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: %18 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %1 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %4 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %7 = "neura.data_mov"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %10 = "neura.data_mov"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: %13 = "neura.data_mov"(%12) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: %2 = "neura.grant_always"(%1) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %5 = "neura.grant_once"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %8 = "neura.grant_always"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %11 = "neura.grant_always"(%10) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: %14 = "neura.grant_once"(%13) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: %28 = "neura.data_mov"(%2) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %19 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %25 = "neura.data_mov"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %22 = "neura.data_mov"(%11) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: %16 = "neura.data_mov"(%14) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: %20 = "neura.phi"(%18, %19) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %17 = "neura.phi"(%15, %16) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: %24 = "neura.data_mov"(%20) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %21 = "neura.data_mov"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: %26 = "neura.add"(%24, %25) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %23 = "neura.fadd"(%21, %22) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: %30 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %27 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %38 = "neura.data_mov"(%23) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: %33 = "neura.data_mov"(%23) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: %29 = "neura.icmp"(%27, %28) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] sorted op: %36 = "neura.data_mov"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] sorted op: %34 = "neura.data_mov"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] sorted op: %31 = "neura.data_mov"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] sorted op: %37 = "neura.not"(%36) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] sorted op: %35 = neura.grant_predicate %33, %34 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: %32 = neura.grant_predicate %30, %31 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %39 = "neura.data_mov"(%37) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] sorted op: neura.ctrl_mov %35 -> %15 : !neura.data<f32, i1> !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: neura.ctrl_mov %32 -> %18 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] sorted op: %40 = neura.grant_predicate %38, %39 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: %41 = "neura.data_mov"(%40) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
[MapToAcceleratorPass] sorted op: "neura.return"(%41) : (!neura.data<f32, i1>) -> ()
module {
  func.func @loop_test() -> f32 attributes {CompiledII = 4 : i32, RecMII = 4 : i32, ResMII = 1 : i32, accelerator = "neura"} {
    %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 1 : i32}]} : () -> !neura.data<i64, i1>
    %1 = "neura.data_mov"(%0) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %2 = "neura.grant_always"(%1) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %3 = "neura.constant"() <{predicate = true, value = 0 : i64}> {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
    %4 = "neura.data_mov"(%3) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %5 = "neura.grant_once"(%4) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %6 = "neura.constant"() <{predicate = true, value = 1 : i64}> {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 1 : i32}]} : () -> !neura.data<i64, i1>
    %7 = "neura.data_mov"(%6) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %8 = "neura.grant_always"(%7) {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %9 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 2 : i32}]} : () -> !neura.data<f32, i1>
    %10 = "neura.data_mov"(%9) {mapping_locs = []} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %11 = "neura.grant_always"(%10) {mapping_locs = [{id =   10 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %12 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 2 : i32}]} : () -> !neura.data<f32, i1>
    %13 = "neura.data_mov"(%12) {mapping_locs = [{id = 33 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %14 = "neura.grant_once"(%13) {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %15 = neura.reserve : !neura.data<f32, i1>
    %16 = "neura.data_mov"(%14) {mapping_locs = []} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %17 = "neura.phi"(%15, %16) {mapping_locs = [{id = 9 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
    %18 = neura.reserve : !neura.data<i64, i1>
    %19 = "neura.data_mov"(%5) {mapping_locs = [{id = 19 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %20 = "neura.phi"(%18, %19) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %21 = "neura.data_mov"(%17) {mapping_locs = [{id = 28 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %22 = "neura.data_mov"(%11) {mapping_locs = [{id = 32 : i32, resource = "link", time_step = 2 : i32}, {id = 44 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %23 = "neura.fadd"(%21, %22) {mapping_locs = [{id = 13 : i32, resource = "tile", time_step = 4 : i32, x = 3 : i32, y = 1 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
    %24 = "neura.data_mov"(%20) {mapping_locs = [{id = 15 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %25 = "neura.data_mov"(%8) {mapping_locs = [{id = 29 : i32, resource = "link", time_step = 1 : i32}, {id = 24 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %26 = "neura.add"(%24, %25) {mapping_locs = [{id = 4 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %27 = "neura.data_mov"(%26) {mapping_locs = [{id = 11 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %28 = "neura.data_mov"(%2) {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 1 : i32}, {id = 29 : i32, resource = "link", time_step = 2 : i32}, {id = 29 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %29 = "neura.icmp"(%27, %28) <{cmpType = "slt"}> {mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
    %30 = "neura.data_mov"(%26) {mapping_locs = []} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %31 = "neura.data_mov"(%29) {mapping_locs = [{id = 24 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %32 = neura.grant_predicate %30, %31 {mapping_locs = [{id = 4 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %32 -> %18 {mapping_locs = [{id = 12 : i32, resource = "link", time_step = 5 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %33 = "neura.data_mov"(%23) {mapping_locs = [{id = 41 : i32, resource = "link", time_step = 4 : i32}, {id = 38 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %34 = "neura.data_mov"(%29) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %35 = neura.grant_predicate %33, %34 {mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 0 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
    neura.ctrl_mov %35 -> %15 {mapping_locs = [{id = 26 : i32, resource = "link", time_step = 6 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
    %36 = "neura.data_mov"(%29) {mapping_locs = []} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %37 = "neura.not"(%36) {mapping_locs = [{id = 8 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %38 = "neura.data_mov"(%23) {mapping_locs = []} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %39 = "neura.data_mov"(%37) {mapping_locs = [{id = 25 : i32, resource = "link", time_step = 5 : i32}, {id = 39 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %40 = neura.grant_predicate %38, %39 {mapping_locs = [{id = 13 : i32, resource = "tile", time_step = 7 : i32, x = 3 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
    %41 = "neura.data_mov"(%40) {mapping_locs = [{id = 42 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    "neura.return"(%41) {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 8 : i32, x = 3 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>) -> ()
  }
}

