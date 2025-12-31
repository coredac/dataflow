import os
import re
import subprocess

PROJECT_ROOT = "/home/x/shiran/Project/dataflow"
NEURA_OPT = f"{PROJECT_ROOT}/build/tools/mlir-neura-opt/mlir-neura-opt"

def clean_output(text):
    lines = text.splitlines()
    cleaned = []
    for l in lines:
        stripped = l.strip()
        if not stripped: continue
        # Filter typical debug/info noise
        if (stripped.startswith("[") and "]" in stripped) or \
           stripped.startswith("Collecting recurrence") or \
           stripped.startswith("  %") or \
           stripped.startswith("  neura.") or \
           stripped.startswith("  \"neura.") or \
           stripped.startswith("Dumping DFG"):
            # Wait, I shouldn't filter indent-2 lines if they are part of the IR.
            # But the debug info for recurrence cycles also starts with indents.
            # A better way: only keep lines that are valid IR or YAML/ASM.
            if stripped.startswith("[DEBUG]"): continue
            if "Recurrence cycle" in stripped: continue
            # If it's the IR, it should be within func.func.
            cleaned.append(l)
        else:
            cleaned.append(l)
    
    # Re-refinement: just remove lines starting with [DEBUG] or [MapToAccelerator
    refined = []
    for l in lines:
        if l.strip().startswith("[DEBUG]") or l.strip().startswith("[MapToAccelerator") or \
           l.strip().startswith("Collecting recurrence") or l.strip().startswith("[calculateResMii]"):
            continue
        refined.append(l)
    return refined

def update_test(rel_path, cmd):
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    print(f"Updating {rel_path}...")
    
    subprocess.run("rm -f tmp-generated-instructions.*", shell=True, cwd=PROJECT_ROOT)
    # Redirect stderr to stdout to catch all potential output
    res = subprocess.run(cmd + " 2>&1", shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
    
    clean_lines = clean_output(res.stdout)
    clean_text = "\n".join(clean_lines)

    # Extract MAPPING (func.func)
    match = re.search(r'(func\.func @.*?\{.*?\})', clean_text, re.DOTALL)
    if not match:
        match = re.search(r'(func\.func.*?\{.*?\})', clean_text, re.DOTALL)
    
    mapping_lines = []
    if match:
        mapping_lines = match.group(1).splitlines()

    yaml_lines = []
    if os.path.exists(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.yaml")):
        with open(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.yaml"), 'r') as f:
            yaml_lines = clean_output(f.read())[:25]
    
    asm_lines = []
    if os.path.exists(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.asm")):
        with open(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.asm"), 'r') as f:
            asm_lines = clean_output(f.read())[:25]

    with open(abs_path, 'r') as f:
        file_lines = f.readlines()

    def replace_block(p_lines, p_prefix, p_new_lines):
        if not p_new_lines: return p_lines
        start = -1
        for i, l in enumerate(p_lines):
            if f"// {p_prefix}:" in l:
                start = i
                break
        if start == -1: return p_lines
        
        end = len(p_lines)
        for j in range(start + 1, len(p_lines)):
            l_strip = p_lines[j].strip()
            if not (l_strip.startswith(f"// {p_prefix}") or l_strip.startswith(f"//{p_prefix}") or l_strip == "//" or l_strip == ""):
                end = j
                break
        
        new_block = []
        for i, nl in enumerate(p_new_lines):
            nl_strip = nl.strip()
            if not nl_strip: continue
            if i == 0:
                new_block.append(f"// {p_prefix}:      {nl_strip}\n")
            else:
                # Use plain prefix for structural items to avoid NEXT issues
                # Especially if we skip lines
                if p_prefix in ["ASM", "YAML"] and ("PE(" in nl_strip or "core_id:" in nl_strip or "column:" in nl_strip or "array_config:" in nl_strip):
                    new_block.append(f"// {p_prefix}:      {nl_strip}\n")
                else:
                    new_block.append(f"// {p_prefix}-NEXT: {nl_strip}\n")
        return p_lines[:start] + new_block + p_lines[end:]

    final_lines = replace_block(file_lines, "MAPPING", mapping_lines)
    final_lines = replace_block(final_lines, "YAML", yaml_lines)
    final_lines = replace_block(final_lines, "ASM", asm_lines)

    with open(abs_path, 'w') as f:
        f.writelines(final_lines)

# Commands
fir_kernel_cmd = f"{NEURA_OPT} /tmp/fir_kernel.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-cast --promote-func-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

fir_kernel_vec_cmd = f"{NEURA_OPT} /tmp/fir_kernel_vec.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-cast --promote-func-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

branch_for_cmd = f"{NEURA_OPT} test/mapping_quality/branch_for.mlir --assign-accelerator --lower-llvm-to-neura --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic backtrack-config=customized\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

update_test("test/e2e/fir/fir_kernel.mlir", fir_kernel_cmd)
update_test("test/e2e/fir/fir_kernel_vec.mlir", fir_kernel_vec_cmd)
update_test("test/mapping_quality/branch_for.mlir", branch_for_cmd)
