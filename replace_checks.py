
import sys

target_file = 'test/code_gen/test_code_generate.mlir'
with open(target_file, 'r') as f:
    content = f.read()

placeholder = '// MAPPING: func.func @loop_test() -> f32 attributes {accelerator = "neura", dataflow_mode = "predicate", mapping_info = {compiled_ii = 4 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 4 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {'

with open('code_gen_checks.txt', 'r') as f:
    new_checks = f.read().strip()

# Add the closing brace which was stripped or needs to be there
if not new_checks.endswith('}'):
    # extracted mlir usually ends with }
    pass

new_content = content.replace(placeholder, new_checks)

with open(target_file, 'w') as f:
    f.write(new_content)
