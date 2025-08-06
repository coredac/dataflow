module {
  func.func @test() -> f32 attributes {CompiledII = 1 : i32, ResMII = 1 : i32, accelerator = "neura"} {
    %cst = arith.constant {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 1 : i32}]} 1.000000e+00 : f32
    %cst_0 = arith.constant {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 2 : i32}]} 2.000000e+00 : f32
    %0 = "neura.data_mov"(%cst) {mapping_locs = [{id = 16 : i32, resource = "link", time_step = 0 : i32}]} : (f32) -> f32
    %1 = "neura.data_mov"(%cst_0) {mapping_locs = [{id = 31 : i32, resource = "link", time_step = 0 : i32}]} : (f32) -> f32
    %2 = "neura.fadd"(%0, %1) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 2 : i32}]} : (f32, f32) -> !neura.data<f32, i1>
    %3 = "neura.data_mov"(%2) {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
    "neura.return"(%3) {mapping_locs = [{id = 2 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>) -> ()
  }
}

