import os
path = "/home/x/shiran/Project/dataflow/test/mapping_quality/branch_for.mlir"
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
for l in lines:
    if "// MAPPING:" in l and "func.func" in l:
        new_lines.append("// MAPPING:      func.func @loop_test() -> f32 attributes {accelerator = \"neura\", dataflow_mode = \"predicate\"\n")
    else:
        new_lines.append(l)

with open(path, 'w') as f:
    f.writelines(new_lines)
