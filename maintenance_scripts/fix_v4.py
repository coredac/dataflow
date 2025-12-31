import os
import re
import subprocess

PROJECT_ROOT = "/home/x/shiran/Project/dataflow"
NEURA_OPT = f"{PROJECT_ROOT}/build/tools/mlir-neura-opt/mlir-neura-opt"

def get_clean_output(cmd):
    # Execute command and separate stdout/stderr if possible, 
    # but some tools might print debug to stdout.
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
    stdout_lines = res.stdout.splitlines()
    cleaned = []
    for l in stdout_lines:
        s = l.strip()
        if not s: continue
        if s.startswith("[DEBUG]") or s.startswith("[MapToAccelerator") or \
           s.startswith("Collecting recurrence") or s.startswith("[calculateResMii]") or \
           s.startswith("Dumping DFG") or s.startswith("[generate-code]"):
            continue
        cleaned.append(l)
    return cleaned

def update_test(rel_path, cmd):
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    print(f"Updating {rel_path}...")
    
    subprocess.run("rm -f tmp-generated-instructions.*", shell=True, cwd=PROJECT_ROOT)
    
    # Get IR lines
    ir_lines_all = get_clean_output(cmd)
    ir_text = "\n".join(ir_lines_all)
    
    # Extract func.func
    match = re.search(r'(func\.func @.*?\{.*?\})', ir_text, re.DOTALL)
    if not match:
        match = re.search(r'(func\.func.*?\{.*?\})', ir_text, re.DOTALL)
    
    mapping_lines = []
    if match:
        mapping_lines = match.group(1).splitlines()
    else:
        print(f"  Warning: No func.func found in output of {rel_path}")

    # Get YAML/ASM
    yaml_lines_raw = []
    if os.path.exists(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.yaml")):
        with open(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.yaml"), 'r') as f:
            yaml_lines_raw = f.read().splitlines()
    
    asm_lines_raw = []
    if os.path.exists(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.asm")):
        with open(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.asm"), 'r') as f:
            asm_lines_raw = f.read().splitlines()

    # Apply 25-line limit and basic cleaning
    yaml_lines = [l for l in yaml_lines_raw if not l.strip().startswith("[DEBUG]")][:25]
    asm_lines = [l for l in asm_lines_raw if not l.strip().startswith("[DEBUG]")][:25]

    with open(abs_path, 'r') as f:
        file_lines = f.readlines()

    def replace_block(p_lines, p_prefix, p_content):
        if not p_content: return p_lines
        start = -1
        for i, l in enumerate(p_lines):
            if f"// {p_prefix}:" in l:
                start = i
                break
        if start == -1: return p_lines
        
        # Determine the end of the existing block
        end = len(p_lines)
        for j in range(start + 1, len(p_lines)):
            cl = p_lines[j].strip()
            # If it's not a comment or a comment with a different prefix, it's the end.
            # But allow blank comment lines "//"
            if not (cl.startswith(f"// {p_prefix}") or cl.startswith(f"//{p_prefix}") or cl == "//"):
                end = j
                break
        
        new_block = []
        for i, line in enumerate(p_content):
            s = line.strip()
            if not s: continue
            if i == 0:
                new_block.append(f"// {p_prefix}:      {s}\n")
            else:
                # Use plain prefix for structural lines in YAML/ASM to be safe with FileCheck line matching
                if p_prefix in ["YAML", "ASM"] and any(x in s for x in ["PE(", "core_id:", "instructions:", "column:", "array_config:"]):
                    new_block.append(f"// {p_prefix}:      {s}\n")
                else:
                    new_block.append(f"// {p_prefix}-NEXT: {s}\n")
        
        return p_lines[:start] + new_block + p_lines[end:]

    file_lines = replace_block(file_lines, "MAPPING", mapping_lines)
    file_lines = replace_block(file_lines, "YAML", yaml_lines)
    file_lines = replace_block(file_lines, "ASM", asm_lines)

    with open(abs_path, 'w') as f:
        f.writelines(file_lines)

# Prepare inputs
subprocess.run("clang++ -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -o /tmp/fir.ll test/benchmark/CGRA-Bench/kernels/fir/fir_int.cpp", shell=True, cwd=PROJECT_ROOT)
subprocess.run("llvm-extract --rfunc=\".*kernel.*\" /tmp/fir.ll -o /tmp/fir_kernel.ll", shell=True, cwd=PROJECT_ROOT)
subprocess.run("mlir-translate --import-llvm /tmp/fir_kernel.ll -o /tmp/fir_kernel.mlir", shell=True, cwd=PROJECT_ROOT)

subprocess.run("clang++ -S -emit-llvm -O3 -fno-unroll-loops -o /tmp/fir-vec.ll test/benchmark/CGRA-Bench/kernels/fir/fir_int.cpp", shell=True, cwd=PROJECT_ROOT)
subprocess.run("llvm-extract --rfunc=\".*kernel.*\" /tmp/fir-vec.ll -o /tmp/fir_kernel_vec.ll", shell=True, cwd=PROJECT_ROOT)
subprocess.run("mlir-translate --import-llvm /tmp/fir_kernel_vec.ll -o /tmp/fir_kernel_vec.mlir", shell=True, cwd=PROJECT_ROOT)

# Commands
fir_cmd = f"{NEURA_OPT} /tmp/fir_kernel.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-cast --promote-func-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

fir_vec_cmd = f"{NEURA_OPT} /tmp/fir_kernel_vec.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-cast --promote-func-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

branch_cmd = f"{NEURA_OPT} test/mapping_quality/branch_for.mlir --assign-accelerator --lower-llvm-to-neura --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic backtrack-config=customized\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

update_test("test/e2e/fir/fir_kernel.mlir", fir_cmd)
update_test("test/e2e/fir/fir_kernel_vec.mlir", fir_vec_cmd)
update_test("test/mapping_quality/branch_for.mlir", branch_cmd)
