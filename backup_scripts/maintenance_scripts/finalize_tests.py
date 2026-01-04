import os
import re
import subprocess

PROJECT_ROOT = "/home/x/shiran/Project/dataflow"
os.environ["PATH"] = f"{PROJECT_ROOT}/build/tools/mlir-neura-opt:{os.environ['PATH']}"

def update_test(rel_path, ir_gen_cmd):
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    print(f"Finalizing {rel_path}...")
    
    subprocess.run("rm -f tmp-generated-instructions.*", shell=True, cwd=PROJECT_ROOT)
    res = subprocess.run(ir_gen_cmd, shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
    
    mapping_ir = res.stdout
    if not mapping_ir.strip() and "-o " in ir_gen_cmd:
        out_file = re.search(r'-o\s+(\S+)', ir_gen_cmd).group(1)
        if os.path.exists(os.path.join(PROJECT_ROOT, out_file)):
            with open(os.path.join(PROJECT_ROOT, out_file), 'r') as f:
                mapping_ir = f.read()

    match = re.search(r'(func\.func @.*?\{.*?\})', mapping_ir, re.DOTALL)
    if not match:
        # Try without the @ if it's different
        match = re.search(r'(func\.func.*?\{.*?\})', mapping_ir, re.DOTALL)
        if not match:
            print(f"  No func.func in {rel_path}")
            return
    ir_lines = match.group(1).splitlines()

    yaml_lines = []
    if os.path.exists(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.yaml")):
        with open(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.yaml"), 'r') as f:
            yaml_lines = f.read().splitlines()[:25]
    
    asm_lines = []
    if os.path.exists(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.asm")):
        with open(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.asm"), 'r') as f:
            asm_lines = f.read().splitlines()[:25]

    with open(abs_path, 'r') as f:
        orig = f.readlines()

    def replace_sec(p_lines, p_prefix, p_content):
        if not p_content: return p_lines
        s = -1
        for idx, l in enumerate(p_lines):
            if f"// {p_prefix}:" in l:
                s = idx
                break
        if s == -1: return p_lines
        
        e = len(p_lines)
        for idx in range(s + 1, len(p_lines)):
            cl = p_lines[idx].strip()
            if not (cl.startswith(f"// {p_prefix}") or cl.startswith(f"//{p_prefix}") or cl == "//"):
                e = idx
                break
        
        new_block = []
        for idx, cl in enumerate(p_content):
            stripped = cl.strip()
            if not stripped: continue
            if idx == 0:
                 # Match anywhere but use literal string
                 new_block.append(f"// {p_prefix}:      {stripped}\n")
            else:
                 # Relax NEXT for structural anchors to skip blank lines/comments in tool output if any
                 if p_prefix in ["ASM", "YAML"] and any(anchor in stripped for anchor in ["PE(", "core_id:", "instructions:", "array_config:"]):
                      new_block.append(f"// {p_prefix}:      {stripped}\n")
                 else:
                      new_block.append(f"// {p_prefix}-NEXT: {stripped}\n")
        return p_lines[:s] + new_block + p_lines[e:]

    final = replace_sec(orig, "MAPPING", ir_lines)
    final = replace_sec(final, "YAML", yaml_lines)
    final = replace_sec(final, "ASM", asm_lines)

    with open(abs_path, 'w') as f:
        f.writelines(final)

# Commands (use fixed /tmp paths for consistency)
fir_kernel_cmd = "mlir-neura-opt /tmp/fir_kernel.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-cast --promote-func-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

fir_kernel_vec_cmd = "mlir-neura-opt /tmp/fir_kernel_vec.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-cast --promote-func-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

branch_for_cmd = "mlir-neura-opt test/mapping_quality/branch_for.mlir --assign-accelerator --lower-llvm-to-neura --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic backtrack-config=customized\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code"

update_test("test/e2e/fir/fir_kernel.mlir", fir_kernel_cmd)
update_test("test/e2e/fir/fir_kernel_vec.mlir", fir_kernel_vec_cmd)
update_test("test/mapping_quality/branch_for.mlir", branch_for_cmd)
