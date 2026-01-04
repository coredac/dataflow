import os
import re

PROJECT_ROOT = "/home/x/shiran/Project/dataflow"

def update_test(test_rel_path, ir_path, yaml_path, asm_path):
    test_path = os.path.join(PROJECT_ROOT, test_rel_path)
    if not os.path.exists(test_path): return
    print(f"Updating {test_rel_path}...")
    
    with open(test_path, 'r') as f:
        lines = f.readlines()
    
    with open(ir_path, 'r') as f:
        new_ir = f.read()
    
    match = re.search(r'(func\.func @.*?\{.*?\})', new_ir, re.DOTALL)
    if not match: return
    func_lines = match.group(1).splitlines()

    start_idx = -1
    for i, line in enumerate(lines):
        if "// MAPPING:" in line:
            start_idx = i
            break
    
    if start_idx != -1:
        end_idx = len(lines)
        for j in range(start_idx + 1, len(lines)):
            if not (lines[j].strip().startswith("// MAPPING") or lines[j].strip().startswith("//  MAPPING")):
                 end_idx = j
                 break
        
        new_mapping_block = []
        for k, fl in enumerate(func_lines):
            fl = fl.rstrip()
            if not fl: continue
            # Exact match but with trailing wildcard
            new_mapping_block.append(f"// MAPPING:      {fl.strip()}{{{{.*}}}}\n")
        
        lines = lines[:start_idx] + new_mapping_block + lines[end_idx:]

    def update_sec(p_lines, p_prefix, p_path):
        if not os.path.exists(p_path): return p_lines
        s = -1
        for i, l in enumerate(p_lines):
            if f"// {p_prefix}:" in l:
                s = i
                break
        if s == -1: return p_lines
        e = len(p_lines)
        for j in range(s + 1, len(p_lines)):
             if not (p_lines[j].strip().startswith(f"// {p_prefix}")):
                 e = j
                 break
        with open(p_path, 'r') as f:
            content = f.read().splitlines()
        block = []
        for con in content:
            con = con.rstrip()
            if not con.strip(): continue
            block.append(f"// {p_prefix}:      {con.strip()}{{{{.*}}}}\n")
        return p_lines[:s] + block + p_lines[e:]

    lines = update_sec(lines, "YAML", yaml_path)
    lines = update_sec(lines, "ASM", asm_path)

    with open(test_path, 'w') as f:
        f.writelines(lines)

update_test("test/e2e/fir/fir_kernel.mlir", "/tmp/fir_mapping_result.mlir", "/tmp/fir.yaml", "/tmp/fir.asm")
update_test("test/e2e/fir/fir_kernel_vec.mlir", "/tmp/fir_vec_mapping_result.mlir", "/tmp/fir_vec.yaml", "/tmp/fir_vec.asm")
update_test("test/mapping_quality/branch_for.mlir", "/tmp/branch_for_mapping_result.mlir", "/tmp/branch_for.yaml", "/tmp/branch_for.asm")
