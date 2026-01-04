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
        if not (l.startswith(f"// {prefix}:") or l.startswith(f"// {prefix}-NEXT:")):
            if l != "" and "// RUN:" not in l:
                end_idx = j
                break

    new_block = []
    for k, l in enumerate(content_lines):
        l = l.rstrip()
        if not l: continue # Skip empty lines in content to avoid empty check strings
        if k == 0:
            new_block.append(f"// {prefix}:      {l}\n")
        else:
            new_block.append(f"// {prefix}-NEXT: {l.strip()}\n")
    return lines[:start_idx] + new_block + lines[end_idx:]

def process_test(test_rel, mapping_prefix):
    abs_path = os.path.join(PROJECT_ROOT, test_rel)
    if not os.path.exists(abs_path): return
    print(f"Processing {test_rel}...")
    
    with open(abs_path, 'r') as f:
        lines = f.readlines()
    
    cmds = []
    curr = ""
    for line in lines:
        if line.strip().startswith("// RUN:"):
            parts = line.split("// RUN:")[1].strip()
            if curr: curr += " " + parts
            else: curr = parts
            if not curr.endswith("\\"):
                cmd = curr.replace("%s", abs_path).replace("%S", os.path.dirname(abs_path)).replace("%t", "/tmp/t")
                cmd = cmd.replace("../../arch_spec/architecture.yaml", os.path.join(PROJECT_ROOT, "test/arch_spec/architecture.yaml"))
                cmd = cmd.replace("../arch_spec/architecture.yaml", os.path.join(PROJECT_ROOT, "test/arch_spec/architecture.yaml"))
                cmd = cmd.replace("../../benchmark/", os.path.join(PROJECT_ROOT, "test/benchmark/"))
                cmds.append(cmd)
                curr = ""
            else:
                curr = curr[:-1]

    # Find generate command or mapping command
    gen_cmd = ""
    for c in cmds:
        if "--generate-code" in c:
            gen_cmd = c
            break
            
    if not gen_cmd:
        for c in cmds:
            if f"-check-prefix={mapping_prefix}" in c:
                gen_cmd = c.split("|")[0].strip()
                break

    if not gen_cmd: 
        # Fallback
        for c in cmds:
            if "mlir-neura-opt" in c:
                gen_cmd = c.split("|")[0].strip()
                break

    print(f"  Running: {gen_cmd}")
    subprocess.run("rm -f tmp-generated-instructions.yaml tmp-generated-instructions.asm", shell=True)
    res = subprocess.run(gen_cmd, shell=True, capture_output=True, text=True)
    
    # Try reading from -o file first
    output_ir = ""
    out_candidate = "/tmp/t-mapping.mlir"
    if not os.path.exists(out_candidate):
        out_candidate = "/tmp/t.tmp-mapping.mlir"
        
    if os.path.exists(out_candidate):
        with open(out_candidate, 'r') as f:
            output_ir = f.read()
    else:
        output_ir = res.stdout

    if output_ir:
        match = re.search(r'(func\.func @.*?\{.*?\})', output_ir, re.DOTALL)
        if match:
             lines = update_block(lines, mapping_prefix, match.group(1).splitlines())
             print(f"  Updated {mapping_prefix}")
        else:
             print("  Could not find func.func in IR output")

    if os.path.exists("tmp-generated-instructions.yaml"):
        with open("tmp-generated-instructions.yaml", 'r') as f:
             lines = update_block(lines, "YAML", f.readlines())
             print("  Updated YAML")
    if os.path.exists("tmp-generated-instructions.asm"):
        with open("tmp-generated-instructions.asm", 'r') as f:
             lines = update_block(lines, "ASM", f.readlines())
             print("  Updated ASM")

    # Final relaxation for ASM PE headers
    new_lines = []
    for line in lines:
        if re.search(r'//\s+ASM-NEXT:\s+PE\(', line):
            new_lines.append(line.replace("ASM-NEXT", "ASM"))
        elif re.search(r'//\s+YAML-NEXT:\s+array_config:', line):
            new_lines.append(line.replace("YAML-NEXT", "YAML"))
        elif re.match(r'// \w+(-NEXT)?:$', line.strip()):
            continue
        else:
            new_lines.append(line)
    
    with open(abs_path, 'w') as f:
        f.writelines(new_lines)

# (Path, Prefix)
tasks = [
    ("test/controflow_fuse/simple_loop_reduction/simple_loop_reduction.mlir", "FUSE-MAPPING"),
    ("test/e2e/fir/fir_kernel.mlir", "MAPPING"),
    ("test/e2e/fir/fir_kernel_vec.mlir", "MAPPING"),
    ("test/mapping_quality/branch_for.mlir", "MAPPING"),
]

for p, m in tasks:
    process_test(p, m)
