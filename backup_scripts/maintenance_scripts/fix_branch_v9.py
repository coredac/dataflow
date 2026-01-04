import os
import subprocess

PROJECT_ROOT = "/home/x/shiran/Project/dataflow"
test_path = os.path.join(PROJECT_ROOT, "test/mapping_quality/branch_for.mlir")

# Step 1: Run the MAPPING command
cmd = "/home/x/shiran/Project/dataflow/build/tools/mlir-neura-opt/mlir-neura-opt " + test_path + " --assign-accelerator --lower-llvm-to-neura --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic backtrack-config=customized\" --architecture-spec=test/arch_spec/architecture.yaml 2>/dev/null"
res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
stdout = res.stdout.splitlines()

# Step 2: Extract func.func
func_lines = []
in_func = False
for l in stdout:
    if "func.func" in l and "{" in l: in_func = True
    if in_func:
        func_lines.append(l.strip())
        if l.strip() == "}": break

# Step 3: Run the YAML/ASM command
cmd2 = "/home/x/shiran/Project/dataflow/build/tools/mlir-neura-opt/mlir-neura-opt " + test_path + " --assign-accelerator --lower-llvm-to-neura --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic backtrack-config=customized\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code 2>/dev/null"
subprocess.run(cmd2, shell=True, cwd=PROJECT_ROOT)

with open("tmp-generated-instructions.yaml", 'r') as f:
    yaml_lines = f.read().splitlines()[:25]
with open("tmp-generated-instructions.asm", 'r') as f:
    asm_lines = f.read().splitlines()[:25]

# Step 4: Update the file
with open(test_path, 'r') as f:
    orig_lines = f.readlines()

def update_sec(p_lines, p_prefix, p_content):
    s = -1
    for i, l in enumerate(p_lines):
        if f"// {p_prefix}:" in l:
            s = i
            break
    if s == -1: return p_lines
    e = len(p_lines)
    for j in range(s + 1, len(p_lines)):
        cl = p_lines[j].strip()
        if not (cl.startswith(f"// {p_prefix}") or cl.startswith(f"//{p_prefix}") or cl == "//"):
            e = j
            break
    
    new_block = []
    for i, el in enumerate(p_content):
        # Truncate at }} to be safe
        if p_prefix == "MAPPING" and "}}" in el:
            el = el[:el.find("}}")+1] # Match until the first } of }}
        
        pref = p_prefix if (i == 0 or el.startswith("PE(") or el.startswith("array_config:")) else f"{p_prefix}-NEXT"
        new_block.append(f"// {pref}:      {el}\n")
    return p_lines[:s] + new_block + p_lines[e:]

final = update_sec(orig_lines, "MAPPING", func_lines)
final = update_sec(final, "YAML", yaml_lines)
final = update_sec(final, "ASM", asm_lines)

with open(test_path, 'w') as f:
    f.writelines(final)
