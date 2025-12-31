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
        l = lines[j].strip()
        if not (l.startswith(f"// {prefix}") or l.startswith(f"//{prefix}")):
             if l == "" or l.startswith("// RUN:"):
                 end_idx = j
                 break
             if l.startswith("//"):
                 if re.match(r'// [A-Z0-9_\-]+:', l):
                     end_idx = j
                     break
             else:
                 end_idx = j
                 break

    new_block = []
    for k, l in enumerate(content_lines):
        l = l.rstrip()
        if not l.strip(): continue 
        if k == 0:
            new_block.append(f"// {prefix}:      {l}\n")
        else:
            new_block.append(f"// {prefix}-NEXT: {l.strip()}\n")
    return lines[:start_idx] + new_block + lines[end_idx:]

def process_test(test_rel):
    abs_path = os.path.join(PROJECT_ROOT, test_rel)
    if not os.path.exists(abs_path): return
    print(f"Processing {test_rel}...")
    
    with open(abs_path, 'r') as f:
        lines = f.readlines()
    
    cmds = []
    curr = ""
    for line in lines:
        if line.startswith("// RUN:"):
            parts = line.split("// RUN:")[1].strip()
            if curr: curr += " " + parts
            else: curr = parts
            if not curr.endswith("\\"):
                cmd = curr.replace("%s", abs_path).replace("%S", os.path.dirname(abs_path)).replace("%t", "/tmp/t")
                # Fix the ../ stuff
                cmd = cmd.replace("../../arch_spec/architecture.yaml", os.path.join(PROJECT_ROOT, "test/arch_spec/architecture.yaml"))
                cmd = cmd.replace("../arch_spec/architecture.yaml", os.path.join(PROJECT_ROOT, "test/arch_spec/architecture.yaml"))
                cmd = cmd.replace("../../benchmark/", os.path.join(PROJECT_ROOT, "test/benchmark/"))
                cmds.append(cmd)
                curr = ""
            else:
                curr = curr[:-1]

    subprocess.run("rm -f tmp-generated-instructions.yaml tmp-generated-instructions.asm", shell=True)
    subprocess.run("rm -f /tmp/t*", shell=True)
    
    for cmd in cmds:
        if "FileCheck" in cmd: continue
        print(f"  Running: {cmd}")
        subprocess.run(cmd, shell=True, capture_output=True, text=True)

    mapping_ir = ""
    candidates = ["/tmp/t-mapping.mlir", "/tmp/t.tmp-mapping.mlir", "/tmp/t.mlir"]
    for c in candidates:
        if os.path.exists(c):
            with open(c, 'r') as f:
                mapping_ir = f.read()
            break
    
    if not mapping_ir:
        for cmd in reversed(cmds):
            if "mlir-neura-opt" in cmd and "--generate-code" not in cmd:
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                mapping_ir = res.stdout
                break

    if mapping_ir:
        match = re.search(r'(func\.func @.*?\{.*?\})', mapping_ir, re.DOTALL)
        if match:
            lines = update_block(lines, "MAPPING", match.group(1).splitlines())

    if os.path.exists("tmp-generated-instructions.yaml"):
        with open("tmp-generated-instructions.yaml", 'r') as f:
             lines = update_block(lines, "YAML", f.readlines())
    if os.path.exists("tmp-generated-instructions.asm"):
        with open("tmp-generated-instructions.asm", 'r') as f:
             lines = update_block(lines, "ASM", f.readlines())

    final_lines = []
    for line in lines:
        if re.search(r'//\s+ASM-NEXT:\s+PE\(', line):
            final_lines.append(line.replace("ASM-NEXT", "ASM"))
        elif re.search(r'//\s+YAML-NEXT:\s+array_config:', line):
            final_lines.append(line.replace("YAML-NEXT", "YAML"))
        elif re.match(r'// \w+(-NEXT)?:$', line.strip()):
            continue
        else:
            final_lines.append(line)
    
    with open(abs_path, 'w') as f:
        f.writelines(final_lines)

for t in ["test/e2e/fir/fir_kernel.mlir", "test/e2e/fir/fir_kernel_vec.mlir", "test/mapping_quality/branch_for.mlir"]:
    process_test(t)
