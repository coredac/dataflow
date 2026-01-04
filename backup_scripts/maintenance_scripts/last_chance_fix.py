import os, subprocess, re

PROJECT_ROOT = "/home/x/shiran/Project/dataflow"
NEURA_TOOLS_BIN = f"{PROJECT_ROOT}/build/tools/mlir-neura-opt"
LLVM_BIN = "/home/x/shiran/llvm-project/build/bin"
os.environ["PATH"] = f"{LLVM_BIN}:{NEURA_TOOLS_BIN}:{os.environ['PATH']}"

def update_test(rel_path):
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(abs_path): return
    print(f"Fixing {rel_path}...")
    
    with open(abs_path, 'r') as f:
        lines = f.read().splitlines()
    
    cmds = []
    curr = ""
    for l in lines:
        if l.startswith("// RUN:"):
            p = l.split("// RUN:")[1].strip()
            if curr: curr += " " + p
            else: curr = p
            if not curr.endswith("\\"):
                c = curr.replace("%s", abs_path).replace("%S", os.path.dirname(abs_path)).replace("%t", "/tmp/t")
                c = c.replace("../../arch_spec/architecture.yaml", os.path.join(PROJECT_ROOT, "test/arch_spec/architecture.yaml"))
                c = c.replace("../arch_spec/architecture.yaml", os.path.join(PROJECT_ROOT, "test/arch_spec/architecture.yaml"))
                c = c.replace("../../benchmark/", os.path.join(PROJECT_ROOT, "test/benchmark/"))
                cmds.append(c)
                curr = ""
            else: curr = curr[:-1]

    subprocess.run("rm -f /tmp/t*", shell=True)
    subprocess.run("rm -f tmp-generated-instructions.*", shell=True)
    for c in cmds:
        if "FileCheck" in c: continue
        subprocess.run(c, shell=True)
    
    res_ir = ""
    for cand in ["/tmp/t-mapping.mlir", "/tmp/t.tmp-mapping.mlir", "/tmp/t.mlir"]:
        if os.path.exists(cand):
            with open(cand, 'r') as f: res_ir = f.read()
            break
    if not res_ir:
         for c in reversed(cmds):
             if "mlir-neura-opt" in c and "--generate-code" not in c:
                 r = subprocess.run(c, shell=True, capture_output=True, text=True)
                 res_ir = r.stdout
                 break
    
    new_lines = list(lines)
    
    def replace_sec(p_lines, p_prefix, p_new_content_lines):
        s = -1
        for i, l in enumerate(p_lines):
            if f"// {p_prefix}:" in l:
                s = i
                break
        if s == -1: return p_lines
        e = len(p_lines)
        for j in range(s+1, len(p_lines)):
            cl = p_lines[j].strip()
            if not (cl.startswith(f"// {p_prefix}") or cl.startswith(f"//{p_prefix}")):
                 if cl == "" or cl.startswith("// RUN:") or (cl.startswith("//") and ":" in cl):
                     e = j
                     break
        block = []
        for idx, con in enumerate(p_new_content_lines):
            con = con.rstrip()
            if not con.strip(): continue
            if idx == 0: block.append(f"// {p_prefix}:      {con}")
            else: block.append(f"// {p_prefix}-NEXT: {con.lstrip()}")
        return p_lines[:s] + block + p_lines[e:]

    if res_ir:
        ir_lines = []
        found = False
        for il in res_ir.splitlines():
             if any(x in il for x in ["[DEBUG]", "[MapToAccelerator", "Collecting recurrence", "[calculateResMii]"]):
                 continue
             if "func.func @" in il: found = True
             if found: ir_lines.append(il)
        if ir_lines:
             if ir_lines[-1].strip() == "}": ir_lines = ir_lines[:-1]
             for pref in ["MAPPING", "FUSE-MAPPING"]:
                 new_lines = replace_sec(new_lines, pref, ir_lines)

    if os.path.exists("tmp-generated-instructions.yaml"):
        with open("tmp-generated-instructions.yaml", 'r') as f:
            new_lines = replace_sec(new_lines, "YAML", f.readlines())
    if os.path.exists("tmp-generated-instructions.asm"):
        with open("tmp-generated-instructions.asm", 'r') as f:
            new_lines = replace_sec(new_lines, "ASM", f.readlines())

    save_lines = []
    for nl in new_lines:
        if re.search(r'//\s+ASM-NEXT:\s+PE\(', nl):
            nl = nl.replace("ASM-NEXT", "ASM")
        elif re.search(r'//\s+YAML-NEXT:\s+array_config:', nl):
            nl = nl.replace("YAML-NEXT", "YAML")
        elif re.match(r'// \w+(-NEXT)?:$', nl.strip()):
            continue
        save_lines.append(nl)
    
    with open(abs_path, 'w') as f:
        f.write("\n".join(save_lines) + "\n")

update_test("test/e2e/fir/fir_kernel.mlir")
update_test("test/e2e/fir/fir_kernel_vec.mlir")
update_test("test/mapping_quality/branch_for.mlir")
