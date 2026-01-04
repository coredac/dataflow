import os
import subprocess
import re

LLVM_BIN = "/home/x/shiran/llvm-project/build/bin"
PROJECT_ROOT = "/home/x/shiran/Project/dataflow"
NEURA_TOOLS_BIN = f"/home/x/shiran/Project/dataflow/build/tools/mlir-neura-opt"
os.environ["PATH"] = f"{LLVM_BIN}:{NEURA_TOOLS_BIN}:{os.environ['PATH']}"

def update_block(lines, prefix, content_lines):
    start_idx = -1
    for i, line in enumerate(lines):
        if f"// {prefix}:" in line:
            start_idx = i
            break
    if start_idx == -1: return lines

    end_idx = len(lines)
    for j in range(start_idx + 1, len(lines)):
        if not (lines[j].strip().startswith(f"// {prefix}:") or lines[j].strip().startswith(f"// {prefix}-NEXT:")):
            if lines[j].strip() != "" and "// RUN:" not in lines[j]:
                end_idx = j
                break

    new_block = []
    for k, l in enumerate(content_lines):
        l = l.rstrip()
        if k == 0:
            new_block.append(f"// {prefix}:      {l}\n")
        else:
            new_block.append(f"// {prefix}-NEXT: {l.strip()}\n")
    return lines[:start_idx] + new_block + lines[end_idx:]

def process_test(test_rel, prefixes):
    abs_path = os.path.join(PROJECT_ROOT, test_rel)
    print(f"Processing {test_rel}...")
    
    with open(abs_path, 'r') as f:
        lines = f.readlines()
    
    # Run all RUN commands in sequence
    cmds = []
    curr = ""
    for line in lines:
        if line.startswith("// RUN:"):
            part = line.split("// RUN:")[1].strip()
            if curr: curr += " " + part
            else: curr = part
            if not curr.endswith("\\"):
                cmd = curr.replace("%s", abs_path).replace("%S", os.path.dirname(abs_path)).replace("%t", "/tmp/t")
                # Fix relative paths
                cmd = cmd.replace("../../arch_spec/architecture.yaml", os.path.join(PROJECT_ROOT, "test/arch_spec/architecture.yaml"))
                cmd = cmd.replace("../arch_spec/architecture.yaml", os.path.join(PROJECT_ROOT, "test/arch_spec/architecture.yaml"))
                cmd = cmd.replace("../../benchmark/", os.path.join(PROJECT_ROOT, "test/benchmark/"))
                cmds.append(cmd)
                curr = ""
            else:
                curr = curr[:-1]

    subprocess.run("rm -f tmp-generated-instructions.yaml tmp-generated-instructions.asm", shell=True)
    
    mapping_ir = ""
    for cmd in cmds:
        if "FileCheck" in cmd: continue
        print(f"  Running: {cmd}")
        # Capture the output of the last non-FileCheck command in case it's the one we need
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        # If the command produces a file like %t-mapping.mlir, we'll use that
        if "-o /tmp/t-mapping.mlir" in cmd or "-o /tmp/t.tmp-mapping.mlir" in cmd:
             pass # Will read file later
        else:
             mapping_ir = res.stdout

    # Read mapping IR if it went to a file
    mapping_file = "/tmp/t-mapping.mlir"
    if not os.path.exists(mapping_file):
        mapping_file = "/tmp/t.tmp-mapping.mlir"
    
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            mapping_ir = f.read()

    if mapping_ir:
        match = re.search(r'(func\.func @.*?\{.*?\})', mapping_ir, re.DOTALL)
        if match:
             func_body = match.group(1).splitlines()
             for prefix in prefixes:
                 if prefix in ["MAPPING", "FUSE-MAPPING"]:
                     lines = update_block(lines, prefix, func_body)
                     print(f"  Updated {prefix}")

    if os.path.exists("tmp-generated-instructions.yaml"):
        with open("tmp-generated-instructions.yaml", 'r') as f:
             lines = update_block(lines, "YAML", f.readlines())
             print("  Updated YAML")
    if os.path.exists("tmp-generated-instructions.asm"):
        with open("tmp-generated-instructions.asm", 'r') as f:
             lines = update_block(lines, "ASM", f.readlines())
             print("  Updated ASM")

    with open(abs_path, 'w') as f:
        f.writelines(lines)

# (Path, [Prefixes])
tasks = [
    ("test/controflow_fuse/simple_loop_reduction/simple_loop_reduction.mlir", ["FUSE-MAPPING"]),
    ("test/e2e/fir/fir_kernel.mlir", ["MAPPING"]),
    ("test/e2e/fir/fir_kernel_vec.mlir", ["MAPPING"]),
    ("test/mapping_quality/branch_for.mlir", ["MAPPING"]),
]

for path, prefs in tasks:
    process_test(path, prefs)

