import os
import re
import subprocess

PROJECT_ROOT = "/home/x/shiran/Project/dataflow"
NEURA_OPT = f"{PROJECT_ROOT}/build/tools/mlir-neura-opt/mlir-neura-opt"

def update_test(rel_path, cmd):
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    print(f"Finalizing {rel_path}...")
    
    subprocess.run("rm -f tmp-generated-instructions.*", shell=True, cwd=PROJECT_ROOT)
    res = subprocess.run(cmd + " 2>&1", shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
    
    raw_lines = res.stdout.splitlines()
    mapping_lines = []
    in_func = False
    last_was_skipped = False
    first_func_line = True

    for l in raw_lines:
        if "func.func" in l and "{" in l:
            in_func = True
        
        if in_func:
            stripped = l.strip()
            if not stripped: continue
            if stripped.startswith("[DEBUG]") or "Recurrence cycle" in stripped or "Assigned" in stripped:
                last_was_skipped = True
                continue
            
            if first_func_line:
                mapping_lines.append(f"// MAPPING:      {{{{.*}}}}{stripped}")
                first_func_line = False
            elif last_was_skipped:
                mapping_lines.append(f"// MAPPING:      {{{{.*}}}}{stripped}")
            else:
                # User's logic: if we skip debug, next should NOT be NEXT
                mapping_lines.append(f"// MAPPING-NEXT: {{{{.*}}}}{stripped}")
            
            last_was_skipped = False
            if stripped == "}":
                break

    def get_clean_sec(path, limit=30):
        if not os.path.exists(path): return []
        with open(path, 'r') as f:
            lines = f.read().splitlines()
        res = []
        skip_next = False
        for l in lines:
            s = l.strip()
            if not s: continue
            if s.startswith("[DEBUG]") or "Recurrence cycle" in s:
                skip_next = True
                continue
            res.append((s, skip_next))
            skip_next = False
            if len(res) >= limit: break
        return res

    yaml_meta = get_clean_sec(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.yaml"))
    asm_meta = get_clean_sec(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.asm"))

    with open(abs_path, 'r') as f:
        file_lines = f.readlines()

    def replace_block(p_lines, p_prefix, p_new_data, is_meta=False):
        if not p_new_data: return p_lines
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
        for i, item in enumerate(p_new_data):
            if is_meta:
                s, skip = item
                if i == 0 or skip or s.startswith("PE(") or s.startswith("array_config:") or s.startswith("core_id:"):
                    new_block.append(f"// {p_prefix}:      {s}\n")
                else:
                    new_block.append(f"// {p_prefix}-NEXT: {s}\n")
            else:
                # mapping_lines are already formatted
                new_block.append(item + "\n")
        
        return p_lines[:start] + new_block + p_lines[end:]

    final_lines = replace_block(file_lines, "MAPPING", mapping_lines)
    final_lines = replace_block(final_lines, "YAML", yaml_meta, is_meta=True)
    final_lines = replace_block(final_lines, "ASM", asm_meta, is_meta=True)

    with open(abs_path, 'w') as f:
        f.writelines(final_lines)

# Commands
fir_kernel_cmd = f"{NEURA_OPT} /tmp/fir_kernel.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-cast --promote-func-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

fir_kernel_vec_cmd = f"{NEURA_OPT} /tmp/fir_kernel_vec.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-cast --promote-func-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

branch_for_cmd = f"{NEURA_OPT} test/mapping_quality/branch_for.mlir --assign-accelerator --lower-llvm-to-neura --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic backtrack-config=customized\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

update_test("test/e2e/fir/fir_kernel.mlir", fir_kernel_cmd)
update_test("test/e2e/fir/fir_kernel_vec.mlir", fir_kernel_vec_cmd)
update_test("test/mapping_quality/branch_for.mlir", branch_for_cmd)
