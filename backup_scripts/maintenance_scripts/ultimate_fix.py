import os
import re
import subprocess

PROJECT_ROOT = "/home/x/shiran/Project/dataflow"
NEURA_TOOLS_BIN = f"{PROJECT_ROOT}/build/tools/mlir-neura-opt"
LLVM_BIN = "/home/x/shiran/llvm-project/build/bin"
os.environ["PATH"] = f"{LLVM_BIN}:{NEURA_TOOLS_BIN}:{os.environ['PATH']}"

def update_test(rel_path):
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    print(f"Processing {rel_path}...")
    
    with open(abs_path, 'r') as f:
        orig_content = f.read()
    
    # Re-run commands to get fresh output
    cmds = []
    curr = ""
    for l in orig_content.splitlines():
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

    # Get IR
    mapping_ir = ""
    for cand in ["/tmp/t-mapping.mlir", "/tmp/t.tmp-mapping.mlir", "/tmp/t.mlir"]:
        if os.path.exists(cand):
            with open(cand, 'r') as f: mapping_ir = f.read()
            break
    
    if not mapping_ir:
        # Try running it to stdout
        for c in reversed(cmds):
            if "mlir-neura-opt" in c and "--generate-code" not in c:
                res = subprocess.run(c, shell=True, capture_output=True, text=True)
                mapping_ir = res.stdout
                break

    lines = orig_content.splitlines()

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
            # Stop if we hit RUN or another prefix
            if l_strip.startswith("// RUN:") or (l_strip.startswith("//") and ":" in l_strip and not l_strip.startswith(f"//{p_prefix}") and not l_strip.startswith(f"// {p_prefix}")):
                end = j
                break
            if not (l_strip.startswith(f"// {p_prefix}") or l_strip.startswith(f"//{p_prefix}") or l_strip == "//"):
                 # This might be the end of the block
                 # But some tests have blank // lines, we should skip them if they are between our prefix lines
                 # Wait, let's look for the next thing that is NOT our prefix
                 pass
        
        # Refined end detection: stop at first line that is NOT a comment with our prefix
        for j in range(start + 1, len(p_lines)):
            l_strip = p_lines[j].strip()
            if not (l_strip.startswith(f"// {p_prefix}") or l_strip.startswith(f"//{p_prefix}") or l_strip == "//" or l_strip == ""):
                end = j
                break
        
        new_block = []
        for idx, nl in enumerate(p_new_lines):
            nl = nl.rstrip()
            if not nl.strip(): continue
            if idx == 0:
                # For MAPPING, if it's fir_kernel_vec, keep 'module'?
                # Actually, let's just use the func line
                new_block.append(f"// {p_prefix}:      {nl}")
            else:
                new_block.append(f"// {p_prefix}-NEXT: {nl.strip()}")
        return p_lines[:start] + new_block + p_lines[end:]

    if mapping_ir:
        ir_match = re.search(r'(func\.func @.*?\{.*?\})', mapping_ir, re.DOTALL)
        if ir_match:
            lines = replace_block(lines, "MAPPING", ir_match.group(1).splitlines())

    if os.path.exists("tmp-generated-instructions.yaml"):
        with open("tmp-generated-instructions.yaml", 'r') as f:
            lines = replace_block(lines, "YAML", f.read().splitlines())
    if os.path.exists("tmp-generated-instructions.asm"):
        with open("tmp-generated-instructions.asm", 'r') as f:
            lines = replace_block(lines, "ASM", f.read().splitlines())

    with open(abs_path, 'w') as f:
        f.write("\n".join(lines) + "\n")

for t in ["test/e2e/fir/fir_kernel.mlir", "test/e2e/fir/fir_kernel_vec.mlir", "test/mapping_quality/branch_for.mlir"]:
    update_test(t)
