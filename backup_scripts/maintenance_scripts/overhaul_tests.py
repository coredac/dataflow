import os
import re
import subprocess

PROJECT_ROOT = "/home/x/shiran/Project/dataflow"
os.environ["PATH"] = f"{PROJECT_ROOT}/build/tools/mlir-neura-opt:{os.environ['PATH']}"

def update_test(rel_path):
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    print(f"Overhauling {rel_path}...")
    
    with open(abs_path, 'r') as f:
        orig_content = f.read()
    lines = orig_content.splitlines()

    cmds = []
    curr = ""
    for l in lines:
        if l.startswith("// RUN:"):
            p = l.split("// RUN:")[1].strip()
            if curr: curr += " " + p
            else: curr = p
            if not curr.endswith("\\"):
                c = curr.replace("%s", abs_path).replace("%S", os.path.dirname(abs_path)).replace("%t", "/tmp/t")
                c = re.sub(r'(\.\./)+arch_spec/architecture.yaml', f"{PROJECT_ROOT}/test/arch_spec/architecture.yaml", c)
                c = re.sub(r'(\.\./)+benchmark/', f"{PROJECT_ROOT}/test/benchmark/", c)
                cmds.append(c)
                curr = ""
            else: curr = curr[:-1]

    subprocess.run("rm -f /tmp/t* tmp-generated-instructions.*", shell=True)
    for c in cmds:
        if "FileCheck" in c: continue
        subprocess.run(c, shell=True)

    mapping_ir = ""
    for cand in ["/tmp/t-mapping.mlir", "/tmp/t.tmp-mapping.mlir", "/tmp/t.mlir"]:
        if os.path.exists(cand):
             with open(cand, 'r') as f: mapping_ir = f.read()
             break
    if not mapping_ir:
        for c in reversed(cmds):
            if "mlir-neura-opt" in c and "--generate-code" not in c:
                res = subprocess.run(c, shell=True, capture_output=True, text=True)
                mapping_ir = res.stdout
                break

    def replace_block(p_lines, p_prefix, p_new_lines):
        if not p_new_lines: return p_lines
        s = -1
        for i, l in enumerate(p_lines):
            if f"// {p_prefix}:" in l:
                s = i
                break
        if s == -1: return p_lines
        
        e = len(p_lines)
        for j in range(s + 1, len(p_lines)):
            l_strip = p_lines[j].strip()
            if not (l_strip.startswith(f"// {p_prefix}") or l_strip.startswith(f"//{p_prefix}") or l_strip == "//" or l_strip == ""):
                e = j
                break
        
        new_block = []
        for idx, nl in enumerate(p_new_lines):
            nl = nl.rstrip()
            if not nl.strip(): continue
            if idx == 0 or nl.startswith("PE(") or nl.startswith("array_config:"):
                 new_block.append(f"// {p_prefix}:      {nl}")
            else:
                 new_block.append(f"// {p_prefix}-NEXT: {nl.strip()}")
        return p_lines[:s] + new_block + p_lines[e:]

    final_lines = list(lines)
    if mapping_ir:
        ir_match = re.search(r'(func\.func @.*?\{.*?\})', mapping_ir, re.DOTALL)
        if ir_match:
            final_lines = replace_block(final_lines, "MAPPING", ir_match.group(1).splitlines())

    if os.path.exists("tmp-generated-instructions.yaml"):
        with open("tmp-generated-instructions.yaml", 'r') as f:
            final_lines = replace_block(final_lines, "YAML", f.read().splitlines())
    if os.path.exists("tmp-generated-instructions.asm"):
        with open("tmp-generated-instructions.asm", 'r') as f:
            final_lines = replace_block(final_lines, "ASM", f.read().splitlines())

    with open(abs_path, 'w') as f:
        f.write("\n".join(final_lines) + "\n")

targets = [
    "test/e2e/fir/fir_kernel.mlir",
    "test/e2e/fir/fir_kernel_vec.mlir",
    "test/mapping_quality/branch_for.mlir"
]
for t in targets:
    update_test(t)
