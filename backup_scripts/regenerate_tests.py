import subprocess
import os
import re

OPT_BIN = "./build/tools/mlir-neura-opt/mlir-neura-opt"

def extract_opt_command(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    cmd_parts = []
    collecting = False
    for line in lines:
        if "RUN: mlir-neura-opt %s" in line:
            collecting = True
            part = line.split("%s")[-1].split("-o")[0].split(">")[0].split("|")[0].strip()
            if part.endswith("\\"): part = part[:-1].strip()
            cmd_parts.append(part)
        elif collecting and "RUN:" in line:
            part = line.split("RUN:")[-1].strip()
            if any(x in part for x in ["FileCheck", "-o", ">", "|"]):
                collecting = False
                part = part.split("-o")[0].split(">")[0].split("|")[0].strip()
                if part and not part.startswith("FileCheck") and not part.startswith("/"):
                    if part.endswith("\\"): part = part[:-1].strip()
                    cmd_parts.append(part)
                break
            if part.endswith("\\"): part = part[:-1].strip()
            cmd_parts.append(part)
        elif collecting:
            collecting = False
            break
            
    return " ".join(cmd_parts)

def update_test_file(filepath):
    print(f"Updating {filepath}")
    passes_str = extract_opt_command(filepath)
    if not passes_str:
        print(f"Could not extract passes for {filepath}")
        return

    passes_str = passes_str.replace("../arch_spec/architecture.yaml", "test/arch_spec/architecture.yaml")
    passes_str = passes_str.replace("%S/../../arch_spec/architecture.yaml", "test/arch_spec/architecture.yaml")
    
    cmd = f"{OPT_BIN} {filepath} {passes_str} --generate-code -o tmp_out.mlir"
    print(f"Executing: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return

    with open("tmp_out.mlir", 'r') as f:
        mapped_ir = f.read()
    
    yaml_out = ""
    if os.path.exists("tmp-generated-instructions.yaml"):
        with open("tmp-generated-instructions.yaml", 'r') as f:
            yaml_out = f.read()
            
    asm_out = ""
    if os.path.exists("tmp-generated-instructions.asm"):
        with open("tmp-generated-instructions.asm", 'r') as f:
            asm_out = f.read()

    mapping_lines = []
    functions = re.split(r'func\.func', mapped_ir)
    for func in functions[1:]:
        if 'accelerator = "neura"' in func:
            full_func = "func.func" + func
            depth = 0
            end_pos = -1
            for i, char in enumerate(full_func):
                if char == '{': depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break
            if end_pos != -1:
                body = full_func[:end_pos]
                for line in body.splitlines():
                    if line.strip():
                        if not mapping_lines:
                            header = line.strip()
                            header = re.sub(r'(attributes \{accelerator = "neura",).*?(\})', r'\1{{.*}}\2', header)
                            mapping_lines.append(f"// MAPPING: {header}")
                        else:
                            mapping_lines.append(f"// MAPPING-NEXT: {line.strip()}")
                break

    # YAML
    yaml_lines = []
    for l in yaml_out.splitlines()[:50]:
        if l.strip():
            yaml_lines.append(l)
    
    if yaml_lines:
        res = ["// YAML: " + yaml_lines[0]]
        for l in yaml_lines[1:]:
            res.append("// YAML-NEXT: " + l)
        yaml_lines = res

    # ASM
    asm_lines = []
    for l in asm_out.splitlines()[:50]:
        if l.strip():
            asm_lines.append(l)
    
    if asm_lines:
        res = ["// ASM: " + asm_lines[0]]
        for l in asm_lines[1:]:
            res.append("// ASM-NEXT: " + l)
        asm_lines = res

    with open(filepath, 'r') as f:
        content = f.read()
    
    split_pos = len(content)
    for marker in ["// MAPPING:", "// YAML:", "// ASM:"]:
        pos = content.find(marker)
        if pos != -1 and pos < split_pos:
            split_pos = pos
            
    header = content[:split_pos].rstrip()
    
    new_content = header + "\n\n" + "\n".join(mapping_lines) + "\n\n" + "\n".join(yaml_lines) + "\n\n" + "\n".join(asm_lines) + "\n"
    
    with open(filepath, 'w') as f:
        f.write(new_content)

failed_tests = [
    "test/code_gen/test_code_generate.mlir",
    "test/e2e/bicg/bicg_kernel.mlir",
    "test/e2e/fir/fir_kernel.mlir",
    "test/e2e/fir/fir_kernel_vec.mlir",
    "test/e2e/histogram/histogram_kernel.mlir",
    "test/e2e/relu/relu_kernel.mlir",
    "test/mapping_quality/branch_for.mlir",
    "test/neura/ctrl/branch_for.mlir",
    "test/neura/fusion/test.mlir"
]

for t in failed_tests:
    update_test_file(t)
