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
        if not (l.startswith(f"// {prefix}:") or l.startswith(f"// {prefix}-NEXT:") or l.startswith(f"// {prefix}-SAME:")):
             if l == "" or l.startswith("// RUN:") or (l.startswith("//") and ":" in l):
                 end_idx = j
                 break
    
    new_block = []
    for k, l in enumerate(content_lines):
        l_clean = l.rstrip()
        if not l_clean.strip(): continue
        if k == 0:
            new_block.append(f"// {prefix}:      {l_clean}\n")
        else:
            new_block.append(f"// {prefix}-NEXT: {l_clean.strip()}\n")
    return lines[:start_idx] + new_block + lines[end_idx:]

def filter_noise(text):
    res = []
    for l in text.splitlines():
        if any(x in l for x in ["[DEBUG]", "[MapToAccelerator", "Collecting recurrence", "[calculateResMii]"]):
            continue
        res.append(l)
    return "\n".join(res)

def extract_body(text):
    res = []
    found = False
    for l in text.splitlines():
        if "func.func @" in l:
            found = True
        if found:
            res.append(l)
    return res if res else None

def process(test_rel):
    abs_path = os.path.join(PROJECT_ROOT, test_rel)
    if not os.path.exists(abs_path): return
    print(f"Processing {test_rel}...")
    
    with open(abs_path, 'r') as f:
        lines = f.readlines()
    
    # Identify prefixes to update
    prefixes = []
    if any("// MAPPING:" in l for l in lines): prefixes.append("MAPPING")
    if any("// FUSE-MAPPING:" in l for l in lines): prefixes.append("FUSE-MAPPING")
    if any("// YAML:" in l for l in lines): prefixes.append("YAML")
    if any("// ASM:" in l for l in lines): prefixes.append("ASM")
    
    if not prefixes:
        print("  No prefixes found.")
        return

    cmds = []
    curr = ""
    for line in lines:
        if line.startswith("// RUN:"):
            part = line.split("// RUN:")[1].strip()
            if curr: curr += " " + part
            else: curr = part
            if not curr.endswith("\\"):
                cmd = curr.replace("%s", abs_path).replace("%S", os.path.dirname(abs_path)).replace("%t", "/tmp/t")
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
        subprocess.run(cmd, shell=True, capture_output=True)

    mapping_ir = ""
    for c in ["/tmp/t-mapping.mlir", "/tmp/t.tmp-mapping.mlir", "/tmp/t.mlir"]:
        if os.path.exists(c):
            with open(c, 'r') as f: mapping_ir = f.read()
            break
    
    if not mapping_ir:
        for cmd in reversed(cmds):
            if "mlir-neura-opt" in cmd and "--generate-code" not in cmd:
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                mapping_ir = res.stdout
                break

    if mapping_ir:
        mapping_ir = filter_noise(mapping_ir)
        body = extract_body(mapping_ir)
        if body:
            for p in ["MAPPING", "FUSE-MAPPING"]:
                if p in prefixes:
                    lines = update_block(lines, p, body)
                    print(f"  Updated {p}")

    if "YAML" in prefixes and os.path.exists("tmp-generated-instructions.yaml"):
        with open("tmp-generated-instructions.yaml", 'r') as f:
            lines = update_block(lines, "YAML", f.readlines())
            print("  Updated YAML")
    if "ASM" in prefixes and os.path.exists("tmp-generated-instructions.asm"):
        with open("tmp-generated-instructions.asm", 'r') as f:
            lines = update_block(lines, "ASM", f.readlines())
            print("  Updated ASM")

    # Final Relaxation for PE lines
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

files = [
    "test/neura/ctrl/branch_for.mlir",
    "test/neura/steer_ctrl/loop_with_return_value.mlir",
    "test/neura/for_loop/relu_test.mlir",
    "test/e2e/fir/fir_kernel.mlir",
    "test/e2e/fir/fir_kernel_vec.mlir",
    "test/e2e/histogram/histogram_kernel.mlir",
    "test/e2e/relu/relu_kernel.mlir",
    "test/e2e/bicg/bicg_kernel.mlir",
    "test/controflow_fuse/perfect_nested/perfect_nested.mlir",
    "test/mapping_quality/branch_for.mlir",
    "test/code_gen/test_code_generate.mlir",
    "test/controflow_fuse/simple_loop/simple_loop.mlir",
    "test/controflow_fuse/simple_loop_reduction/simple_loop_reduction.mlir",
    "test/mapping_quality/tiny_loop.mlir"
]

for f in files:
    process(f)
