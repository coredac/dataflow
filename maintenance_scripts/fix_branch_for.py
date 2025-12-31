import subprocess
import os
import re

PROJECT_ROOT = "/home/x/shiran/Project/dataflow"
test_path = os.path.join(PROJECT_ROOT, "test/mapping_quality/branch_for.mlir")

with open(test_path, 'r') as f:
    lines = f.readlines()

# Get the actual output
subprocess.run("/home/x/shiran/Project/dataflow/build/tools/mlir-neura-opt/mlir-neura-opt " + test_path + " --assign-accelerator --lower-llvm-to-neura --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic backtrack-config=customized\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code", shell=True, cwd=PROJECT_ROOT)

with open("tmp-generated-instructions.asm", 'r') as f:
    asm_lines = f.read().splitlines()

# Replace ASM block
start = -1
for i, l in enumerate(lines):
    if "// ASM:" in l:
        start = i
        break

if start != -1:
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if not (lines[j].strip().startswith("// ASM") or lines[j].strip().startswith("//ASM") or lines[j].strip() == "//"):
            end = j
            break
    
    new_asm = []
    for idx, al in enumerate(asm_lines):
        if not al.strip(): continue
        if idx == 0 or al.startswith("PE("):
            new_asm.append(f"// ASM:      {al.strip()}\n")
        else:
            new_asm.append(f"// ASM-NEXT: {al.strip()}\n")
    
    lines = lines[:start] + new_asm + lines[end:]

with open(test_path, 'w') as f:
    f.writelines(lines)
