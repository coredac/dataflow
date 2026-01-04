import os
import re
import subprocess

PROJECT_ROOT = "/home/x/shiran/Project/dataflow"
NEURA_OPT = f"{PROJECT_ROOT}/build/tools/mlir-neura-opt/mlir-neura-opt"

def is_debug(l):
    s = l.strip()
    if not s: return False
    # Common patterns for debug/info in this tool
    if s.startswith("[DEBUG]") or s.startswith("[MapToAccelerator") or \
       s.startswith("Collecting recurrence") or s.startswith("[calculateResMii]") or \
       s.startswith("Dumping DFG") or s.startswith("[generate-code]") or \
       s.startswith("Recurrence cycle"):
        return True
    # The output in the logs shows indented lines under "Recurrence cycle" are also part of debug
    # but they might look like IR. However, they are usually inside a block.
    return False

def get_mapping_block(raw_output):
    # We want to extract the func.func and its body, 
    # but we must be careful to handle interspersed debug lines.
    
    # First, find where func.func starts
    start_idx = -1
    for i, l in enumerate(raw_output):
        if "func.func" in l and "{" in l:
            start_idx = i
            break
    
    if start_idx == -1: return []

    res = []
    # Use MAPPING for the first line
    res.append( ("MAPPING", raw_output[start_idx].strip()) )
    
    skip_next = False
    in_func = True
    brace_count = raw_output[start_idx].count('{') - raw_output[start_idx].count('}')
    
    for i in range(start_idx + 1, len(raw_output)):
        l = raw_output[i]
        s = l.strip()
        if not s: continue
        
        # Check if debug
        if is_debug(l):
            skip_next = True
            continue
        
        # Determine prefix
        # If it's a structural line or we skipped something, use MAPPING
        prefix = "MAPPING" if skip_next else "MAPPING-NEXT"
        res.append( (prefix, s) )
        skip_next = False
        
        brace_count += l.count('{')
        brace_count -= l.count('}')
        if brace_count <= 0:
            break
            
    return res

def process_meta(lines, limit=25):
    res = []
    skip_next = False
    count = 0
    for l in lines:
        s = l.strip()
        if not s: continue
        if is_debug(l):
            skip_next = True
            continue
        
        prefix = "CHECK" # generic placeholder
        res.append( (skip_next, s) )
        skip_next = False
        count += 1
        if count >= limit: break
    return res

def update_test(rel_path, cmd):
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    print(f"Updating {rel_path}...")
    
    subprocess.run("rm -f tmp-generated-instructions.*", shell=True, cwd=PROJECT_ROOT)
    res = subprocess.run(cmd + " 2>&1", shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
    raw_lines = res.stdout.splitlines()

    mapping_data = get_mapping_block(raw_lines)
    
    yaml_lines_raw = []
    if os.path.exists(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.yaml")):
        with open(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.yaml"), 'r') as f:
            yaml_lines_raw = f.read().splitlines()
    yaml_data = process_meta(yaml_lines_raw)

    asm_lines_raw = []
    if os.path.exists(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.asm")):
        with open(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.asm"), 'r') as f:
            asm_lines_raw = f.read().splitlines()
    asm_data = process_meta(asm_lines_raw)

    with open(abs_path, 'r') as f:
        file_lines = f.readlines()

    def replace_sec(p_lines, p_prefix, p_data, is_meta=False):
        if not p_data: return p_lines
        start = -1
        for i, l in enumerate(p_lines):
            if f"// {p_prefix}:" in l:
                start = i
                break
        if start == -1: return p_lines
        
        end = len(p_lines)
        for j in range(start + 1, len(p_lines)):
            cl = p_lines[j].strip()
            if not (cl.startswith(f"// {p_prefix}") or cl.startswith(f"//{p_prefix}") or cl == "//"):
                end = j
                break
        
        new_block = []
        for i, item in enumerate(p_data):
            if is_meta:
                skip, s = item
                # For meta, use literal NEXT as usual unless skip is true
                # Or if it's a structural anchor
                anchor = any(x in s for x in ["PE(", "core_id:", "instructions:", "array_config:", "column:"])
                if i == 0 or skip or anchor:
                    new_block.append(f"// {p_prefix}:      {s}\n")
                else:
                    new_block.append(f"// {p_prefix}-NEXT: {s}\n")
            else:
                pref, s = item
                new_block.append(f"// {pref}:      {s}\n")
        
        return p_lines[:start] + new_block + p_lines[end:]

    file_lines = replace_sec(file_lines, "MAPPING", mapping_data)
    file_lines = replace_sec(file_lines, "YAML", yaml_data, is_meta=True)
    file_lines = replace_sec(file_lines, "ASM", asm_data, is_meta=True)

    with open(abs_path, 'w') as f:
        f.writelines(file_lines)

# Commands
fir_cmd = f"{NEURA_OPT} /tmp/fir_kernel.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-cast --promote-func-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

fir_vec_cmd = f"{NEURA_OPT} /tmp/fir_kernel_vec.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-cast --promote-func-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

branch_cmd = f"{NEURA_OPT} test/mapping_quality/branch_for.mlir --assign-accelerator --lower-llvm-to-neura --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic backtrack-config=customized\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

update_test("test/e2e/fir/fir_kernel.mlir", fir_cmd)
update_test("test/e2e/fir/fir_kernel_vec.mlir", fir_vec_cmd)
update_test("test/mapping_quality/branch_for.mlir", branch_cmd)
