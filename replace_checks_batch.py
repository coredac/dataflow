
import sys

files = [
    {
        'target': 'test/e2e/fir/fir_kernel.mlir',
        'checks': 'fir_checks.txt',
        'placeholder': '// MAPPING: func.func @_Z6kernelPiS_S_\n// MAPPING-SAME: accelerator = "neura"\n// MAPPING-SAME: dataflow_mode = "predicate"\n// MAPPING-SAME: mapping_info = {compiled_ii = 5 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 5 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}'
    },
    {
        'target': 'test/e2e/histogram/histogram_kernel.mlir',
        'checks': 'histogram_checks.txt',
        'placeholder': '// MAPPING: func.func @_Z6kernelPiS_\n// MAPPING-SAME: accelerator = "neura"\n// MAPPING-SAME: dataflow_mode = "predicate"\n// MAPPING-SAME: mapping_info = {compiled_ii = 5 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 5 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}'
    },
    {
        'target': 'test/e2e/relu/relu_kernel.mlir',
        'checks': 'relu_checks.txt',
        'placeholder': '// MAPPING: func.func @kernel\n// MAPPING-SAME: accelerator = "neura"\n// MAPPING-SAME: dataflow_mode = "predicate"\n// MAPPING-SAME: mapping_info = {compiled_ii = 5 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 5 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}'
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
