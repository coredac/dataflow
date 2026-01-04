
import sys

files = [
    {
        'target': 'test/mapping_quality/branch_for.mlir',
        'checks': 'branch_for_checks.txt',
        'placeholder': '// MAPPING: func.func @loop_test()\n// MAPPING-SAME: accelerator = "neura"\n// MAPPING-SAME: mapping_info = {compiled_ii = 4 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 4 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}'
    },
    {
        'target': 'test/neura/ctrl/branch_for.mlir',
        'checks': 'branch_for_checks.txt',
        'placeholder': '// MAPPING: func.func @loop_test()\n// MAPPING-SAME: accelerator = "neura"\n// MAPPING-SAME: mapping_info = {compiled_ii = 4 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 4 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}'
    }
]

for item in files:
    try:
        with open(item['target'], 'r') as f:
            content = f.read()
            
        with open(item['checks'], 'r') as f:
            new_checks = f.read().strip()
            
        if item['placeholder'] in content:
            new_content = content.replace(item['placeholder'], new_checks)
            with open(item['target'], 'w') as f:
                f.write(new_content)
            print(f"Updated {item['target']}")
        else:
            print(f"Placeholder not found in {item['target']}")
            
    except Exception as e:
        print(f"Error processing {item['target']}: {e}")
