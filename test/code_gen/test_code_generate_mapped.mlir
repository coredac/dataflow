module {
  func.func @loop_test() -> f32 attributes {accelerator = "neura", dataflow_mode = "predicate", mapping_info = {compiled_ii = 6 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 4 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
    %0 = "neura.constant"() <{value = 10 : i64}> {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
    %1 = "neura.data_mov"(%0) {mapping_locs = [{id = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %2 = "neura.grant_once"(%1) {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %3 = "neura.constant"() <{value = 0 : i64}> {mapping_locs = [{id = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
    %4 = "neura.data_mov"(%3) {mapping_locs = [{id = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %5 = "neura.grant_once"(%4) {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %6 = "neura.constant"() <{value = 1 : i64}> {mapping_locs = [{id = 2 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
    %7 = "neura.data_mov"(%6) {mapping_locs = [{id = 64 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %8 = "neura.grant_once"(%7) {mapping_locs = [{id = 2 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %9 = "neura.constant"() <{value = 3.000000e+00 : f32}> {mapping_locs = [{id = 14 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 3 : i32}]} : () -> !neura.data<f32, i1>
    %10 = "neura.data_mov"(%9) {mapping_locs = [{id = 45 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %11 = "neura.grant_once"(%10) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %12 = "neura.constant"() <{value = 0.000000e+00 : f32}> {mapping_locs = [{id = 11 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<f32, i1>
    %13 = "neura.data_mov"(%12) {mapping_locs = [{id = 36 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %14 = "neura.grant_once"(%13) {mapping_locs = [{id = 7 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 1 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %15 = neura.reserve : !neura.data<i64, i1>
    %16 = "neura.data_mov"(%2) {mapping_locs = [{id = 32 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %17 = "neura.phi"(%15, %16) {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %18 = neura.reserve : !neura.data<i64, i1>
    %19 = "neura.data_mov"(%8) {mapping_locs = [{id = 64 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %20 = "neura.phi"(%18, %19) {mapping_locs = [{id = 2 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %21 = neura.reserve : !neura.data<f32, i1>
    %22 = "neura.data_mov"(%11) {mapping_locs = [{id = 320 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %23 = "neura.phi"(%21, %22) {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
    %24 = neura.reserve : !neura.data<f32, i1>
    %25 = "neura.data_mov"(%14) {mapping_locs = [{id = 21 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %26 = "neura.phi"(%24, %25) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
    %27 = neura.reserve : !neura.data<i64, i1>
    %28 = "neura.data_mov"(%5) {mapping_locs = [{id = 4 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %29 = "neura.phi"(%27, %28) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %30 = "neura.data_mov"(%26) {mapping_locs = [{id = 192 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %31 = "neura.data_mov"(%23) {mapping_locs = [{id = 33 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %32 = "neura.fadd"(%30, %31) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
    %33 = "neura.data_mov"(%29) {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %34 = "neura.data_mov"(%20) {mapping_locs = [{id = 7 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %35 = "neura.add"(%33, %34) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %36 = "neura.data_mov"(%35) {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %37 = "neura.data_mov"(%17) {mapping_locs = [{id = 4 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %38 = "neura.icmp"(%36, %37) <{cmpType = "slt"}> {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
    %39 = "neura.data_mov"(%35) {mapping_locs = [{id = 193 : i32, resource = "register", time_step = 3 : i32}, {id = 193 : i32, resource = "register", time_step = 4 : i32}, {id = 193 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %40 = "neura.data_mov"(%38) {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 4 : i32}, {id = 192 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %41 = neura.grant_predicate %39, %40 {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %41 -> %27 {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 6 : i32}, {id = 160 : i32, resource = "register", time_step = 7 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %42 = "neura.data_mov"(%32) {mapping_locs = [{id = 194 : i32, resource = "register", time_step = 5 : i32}, {id = 194 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %43 = "neura.data_mov"(%38) {mapping_locs = [{id = 16 : i32, resource = "link", time_step = 4 : i32}, {id = 28 : i32, resource = "link", time_step = 5 : i32}, {id = 33 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %44 = neura.grant_predicate %42, %43 {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
    neura.ctrl_mov %44 -> %24 {mapping_locs = [{id = 192 : i32, resource = "register", time_step = 7 : i32}, {id = 192 : i32, resource = "register", time_step = 8 : i32}, {id = 192 : i32, resource = "register", time_step = 9 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
    %45 = "neura.data_mov"(%23) {mapping_locs = [{id = 31 : i32, resource = "link", time_step = 4 : i32}, {id = 29 : i32, resource = "link", time_step = 5 : i32}, {id = 160 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %46 = "neura.data_mov"(%38) {mapping_locs = [{id = 162 : i32, resource = "register", time_step = 4 : i32}, {id = 162 : i32, resource = "register", time_step = 5 : i32}, {id = 162 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %47 = neura.grant_predicate %45, %46 {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
    neura.ctrl_mov %47 -> %21 {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 7 : i32}, {id = 20 : i32, resource = "link", time_step = 8 : i32}, {id = 321 : i32, resource = "register", time_step = 9 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
    %48 = "neura.data_mov"(%20) {mapping_locs = [{id = 5 : i32, resource = "link", time_step = 2 : i32}, {id = 33 : i32, resource = "register", time_step = 3 : i32}, {id = 4 : i32, resource = "link", time_step = 4 : i32}, {id = 160 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %49 = "neura.data_mov"(%38) {mapping_locs = [{id = 161 : i32, resource = "register", time_step = 4 : i32}, {id = 161 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %50 = neura.grant_predicate %48, %49 {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %50 -> %18 {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 6 : i32}, {id = 19 : i32, resource = "link", time_step = 7 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %51 = "neura.data_mov"(%17) {mapping_locs = [{id = 32 : i32, resource = "register", time_step = 3 : i32}, {id = 32 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %52 = "neura.data_mov"(%38) {mapping_locs = [{id = 15 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %53 = neura.grant_predicate %51, %52 {mapping_locs = [{id = 1 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %53 -> %15 {mapping_locs = [{id = 33 : i32, resource = "register", time_step = 5 : i32}, {id = 33 : i32, resource = "register", time_step = 6 : i32}, {id = 33 : i32, resource = "register", time_step = 7 : i32}, {id = 33 : i32, resource = "register", time_step = 8 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %54 = "neura.data_mov"(%38) {mapping_locs = [{id = 160 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %55 = "neura.not"(%54) {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %56 = "neura.data_mov"(%32) {mapping_locs = [{id = 18 : i32, resource = "link", time_step = 5 : i32}, {id = 224 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    %57 = "neura.data_mov"(%55) {mapping_locs = [{id = 14 : i32, resource = "link", time_step = 5 : i32}, {id = 18 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %58 = neura.grant_predicate %56, %57 {mapping_locs = [{id = 7 : i32, resource = "tile", time_step = 7 : i32, x = 3 : i32, y = 1 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
    %59 = "neura.data_mov"(%58) {mapping_locs = [{id = 22 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    "neura.return"(%59) {mapping_locs = [{id = 3 : i32, resource = "tile", time_step = 8 : i32, x = 3 : i32, y = 0 : i32}]} : (!neura.data<f32, i1>) -> ()
  }
}

