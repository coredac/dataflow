import os
import re
import subprocess

PROJECT_ROOT = "/home/x/shiran/Project/dataflow"
NEURA_OPT = f"{PROJECT_ROOT}/build/tools/mlir-neura-opt/mlir-neura-opt"

def is_debug(l):
    s = l.strip()
    return s.startswith("[DEBUG]") or s.startswith("[MapToAccelerator") or \
           s.startswith("Collecting recurrence") or s.startswith("[calculateResMii]") or \
           s.startswith("Recurrence cycle")

def get_mapping_data(raw_output):
    start_idx = -1
    for i, l in enumerate(raw_output):
        if "func.func" in l and "{" in l:
            start_idx = i
            break
    if start_idx == -1: return []

    res = []
    # Use CHECK for every line to be resilient against debug info between lines
    # but the user said "remove next of the next line"
    
    brace_count = 0
    in_func = False
    for i in range(start_idx, len(raw_output)):
        l = raw_output[i]
        s = l.strip()
        if not s: continue
        if is_debug(l): continue
        
        if "func.func" in s:
            in_func = True
            # Truncate at }} to avoid FileCheck special char issue
            if "}}" in s:
                s = s[:s.rfind("}}")] # Stop before the closing braces
            res.append(("MAPPING", s))
        else:
            res.append(("MAPPING-NEXT", s))
        
        brace_count += l.count('{')
        brace_count -= l.count('}')
        if brace_count <= 0 and in_func:
            break
    return res

def get_meta_data(lines, limit=25):
    res = []
    for l in lines:
        s = l.strip()
        if not s or is_debug(l): continue
        res.append(s)
        if len(res) >= limit: break
    return res

def update_test(rel_path, cmd):
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    print(f"Updating {rel_path}...")
    subprocess.run("rm -f tmp-generated-instructions.*", shell=True, cwd=PROJECT_ROOT)
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
    mapping_data = get_mapping_data(res.stdout.splitlines())
    
    yaml_lines = []
    if os.path.exists("tmp-generated-instructions.yaml"):
        with open("tmp-generated-instructions.yaml", 'r') as f: yaml_lines = f.read().splitlines()
    yaml_data = get_meta_data(yaml_lines)

    asm_lines = []
    if os.path.exists("tmp-generated-instructions.asm"):
        with open("tmp-generated-instructions.asm", 'r') as f: asm_lines = f.read().splitlines()
    asm_data = get_meta_data(asm_lines)

    with open(abs_path, 'r') as f: lines = f.readlines()

    def replace_block(p_lines, p_prefix, p_content, is_mapping=False):
        if not p_content: return p_lines
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
        for i, item in enumerate(p_content):
            if is_mapping:
                pref, val = item
                new_block.append(f"// {pref}:      {val}\n")
            else:
                # Meta: use NEXT for everything except the first or anchors
                if i == 0 or any(x in item for x in ["PE(", "core_id:", "instructions:"]):
                    new_block.append(f"// {p_prefix}:      {item}\n")
                else:
                    new_block.append(f"// {p_prefix}-NEXT: {item}\n")
        return p_lines[:s] + new_block + p_lines[e:]

    lines = replace_block(lines, "MAPPING", mapping_data, True)
    lines = replace_block(lines, "YAML", yaml_data)
    lines = replace_block(lines, "ASM", asm_data)
    with open(abs_path, 'w') as f: f.writelines(lines)

fir_cmd = f"{NEURA_OPT} /tmp/fir_kernel.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-cast --promote-func-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"
fir_vec_cmd = f"{NEURA_OPT} /tmp/fir_kernel_vec.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-cast --promote-func-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"
branch_cmd = f"{NEURA_OPT} test/mapping_quality/branch_for.mlir --assign-accelerator --lower-llvm-to-neura --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic backtrack-config=customized\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

update_test("test/e2e/fir/fir_kernel.mlir", fir_cmd)
update_test("test/e2e/fir/fir_kernel_vec.mlir", fir_vec_cmd)
update_test("test/mapping_quality/branch_for.mlir", branch_cmd)
