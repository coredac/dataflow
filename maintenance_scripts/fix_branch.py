import os
import re
import subprocess

PROJECT_ROOT = "/home/x/shiran/Project/dataflow"
NEURA_OPT = f"{PROJECT_ROOT}/build/tools/mlir-neura-opt/mlir-neura-opt"
test_path = os.path.join(PROJECT_ROOT, "test/mapping_quality/branch_for.mlir")

cmd = f"{NEURA_OPT} {test_path} --assign-accelerator --lower-llvm-to-neura --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic backtrack-config=customized\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

res = subprocess.run(cmd + " 2>&1", shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
output = res.stdout

# Find func.func block
# We want the signature to be exactly what's printed
match = re.search(r'(func\.func @loop_test.*?\{)', output, re.DOTALL)
if match:
    sig = match.group(1)
    # Get the rest of the function until the last }
    rest = output[match.end():]
    # This is a bit risky if there are nested braces, but usually fine for linear IR
    # Let's just take the lines until the next top-level }
    body = []
    brace_count = 1
    for line in rest.splitlines():
        body.append(line)
        brace_count += line.count('{')
        brace_count -= line.count('}')
        if brace_count == 0:
            break
    func_lines = [sig] + body
else:
    print("Could not find func.func")
    exit(1)

# Clean func_lines from debug
func_lines = [l for l in func_lines if not l.strip().startswith("[DEBUG]")]

with open(test_path, 'r') as f:
    lines = f.readlines()

# Replace MAPPING
s = -1
for i, l in enumerate(lines):
    if "// MAPPING:" in l:
        s = i
        break
if s != -1:
    e = len(lines)
    for j in range(s + 1, len(lines)):
        if not (lines[j].strip().startswith("// MAPPING") or lines[j].strip().startswith("//MAPPING") or lines[j].strip() == "//"):
            e = j
            break
    
    new_mapping = []
    for i, fl in enumerate(func_lines):
        fl = fl.strip()
        if not fl: continue
        if i == 0:
            new_mapping.append(f"// MAPPING:      {fl}\n")
        else:
            new_mapping.append(f"// MAPPING-NEXT: {fl}\n")
    
    lines = lines[:s] + new_mapping + lines[e:]

with open(test_path, 'w') as f:
    f.writelines(lines)
