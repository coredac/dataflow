import os
import re
import subprocess

PROJECT_ROOT = "/home/x/shiran/Project/dataflow"
NEURA_OPT = f"{PROJECT_ROOT}/build/tools/mlir-neura-opt/mlir-neura-opt"

def is_debug(l):
    s = l.strip()
    if not s: return False
    # More comprehensive debug detection
    if s.startswith("[DEBUG]") or s.startswith("[MapToAccelerator") or \
       s.startswith("Collecting recurrence") or s.startswith("[calculateResMii]") or \
       s.startswith("Dumping DFG") or s.startswith("[generate-code]") or \
       s.startswith("Recurrence cycle"):
        return True
    return False

def get_mapping_data(raw_output):
    # Find func.func
    start_idx = -1
    for i, l in enumerate(raw_output):
        if "func.func" in l and "{" in l:
            start_idx = i
            break
    if start_idx == -1: return []

    res = []
    skipped_last = False # To handle the very first line if there was debug before it
    if start_idx > 0: skipped_last = True # Assume there was something before it

    brace_count = 0
    for i in range(start_idx, len(raw_output)):
        l = raw_output[i]
        s = l.strip()
        if not s: continue
        
        if is_debug(l):
            skipped_last = True
            continue
        
        # Determine prefix
        # If the line contains }} or other problematic chars for FileCheck, 
        # let's be safe. But the user said no non-deterministic.
        # I will just use the literal string.
        prefix = "MAPPING" if (skipped_last or i == start_idx) else "MAPPING-NEXT"
        
        # Special handling for func.func line to avoid {{ or }} issues
        if "func.func" in s:
            # Replace {{ with { { and }} with } } in the EXPECATION? 
            # No, then it won't match literal {{ }}.
            # But the IR usually doesn't have {{ }} unless it's a vector or attribute.
            # Attributes have {key = value}.
            # If we have }} at the end, let's just make it a CHECK-SAME to be safe.
            if "}}" in s:
                parts = s.split("}")
                res.append((prefix, parts[0] + "}"))
                for p in parts[1:-1]:
                    res.append(("MAPPING-SAME", p + "}"))
                if parts[-1]:
                    res.append(("MAPPING-SAME", parts[-1]))
            else:
                res.append((prefix, s))
        else:
            res.append((prefix, s))
        
        skipped_last = False
        brace_count += l.count('{')
        brace_count -= l.count('}')
        if i >= start_idx and brace_count <= 0:
            break
    return res

def get_meta_data(lines, limit=25):
    res = []
    skipped = False
    count = 0
    for l in lines:
        s = l.strip()
        if not s: continue
        if is_debug(l):
            skipped = True
            continue
        res.append((skipped, s))
        skipped = False
        count += 1
        if count >= limit: break
    return res

def update_test(rel_path, cmd):
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    print(f"Updating {rel_path}...")
    
    subprocess.run("rm -f tmp-generated-instructions.*", shell=True, cwd=PROJECT_ROOT)
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
    # Use stdout only for IR
    raw_lines = res.stdout.splitlines()
    
    mapping_data = get_mapping_data(raw_lines)
    
    yaml_lines = []
    if os.path.exists(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.yaml")):
        with open(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.yaml"), 'r') as f:
            yaml_lines = f.read().splitlines()
    yaml_data = get_meta_data(yaml_lines)

    asm_lines = []
    if os.path.exists(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.asm")):
        with open(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.asm"), 'r') as f:
            asm_lines = f.read().splitlines()
    asm_data = get_meta_data(asm_lines)

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
                # Structural anchors or skipped debug info use CHECK
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

# Commands (use stdout only)
fir_cmd = f"{NEURA_OPT} /tmp/fir_kernel.mlir --assign-accelerator ... --generate-code" 
# Wait, I need the full commands
fir_cmd = f"{NEURA_OPT} /tmp/fir_kernel.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-cast --promote-func-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

fir_vec_cmd = f"{NEURA_OPT} /tmp/fir_kernel_vec.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-cast --promote-func-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

branch_cmd = f"{NEURA_OPT} test/mapping_quality/branch_for.mlir --assign-accelerator --lower-llvm-to-neura --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic backtrack-config=customized\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

update_test("test/e2e/fir/fir_kernel.mlir", fir_cmd)
update_test("test/e2e/fir/fir_kernel_vec.mlir", fir_vec_cmd)
update_test("test/mapping_quality/branch_for.mlir", branch_cmd)
