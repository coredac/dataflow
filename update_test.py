import sys
import subprocess
import os
import re

def update_file(test_file):
    print(f"Updating {test_file}...")
    
    with open(test_file, 'r') as f:
        lines = f.readlines()
    
    build_dir = "/home/x/shiran/Project/dataflow/build"
    mlir_neura_opt = f"{build_dir}/tools/mlir-neura-opt/mlir-neura-opt"
    arch_spec = "/home/x/shiran/Project/dataflow/test/arch_spec/architecture.yaml"
    
    # Temporary files
    tmp_mapping = "/tmp/tmp_mapping.mlir"
    tmp_yaml = "tmp-generated-instructions.yaml"
    tmp_asm = "tmp-generated-instructions.asm"
    
    # Clean up old tmp files
    if os.path.exists(tmp_yaml): os.remove(tmp_yaml)
    if os.path.exists(tmp_asm): os.remove(tmp_asm)

    # Some tests need specific pipelines. We'll try to guess based on the file content or path.
    # Default pipeline for many tests:
    cmd = [
        mlir_neura_opt, test_file,
        "--assign-accelerator",
        "--lower-llvm-to-neura",
        "--promote-func-arg-to-const",
        "--fold-constant",
        "--canonicalize-return",
        "--canonicalize-live-in",
        "--leverage-predicated-value",
        "--transform-ctrl-to-data-flow",
        "--fold-constant",
        "--insert-data-mov",
        f"--map-to-accelerator=mapping-strategy=heuristic",
        f"--architecture-spec={arch_spec}",
        "--generate-code",
        "-o", tmp_mapping
    ]

    # Special cases
    if "branch_for.mlir" in test_file and "mapping_quality" in test_file:
        cmd[12] = "--map-to-accelerator=mapping-strategy=heuristic backtrack-config=simple"
    
    if "e2e" in test_file or "honor_arch" in test_file or "c2llvm2mlir" in test_file:
        # These usually start from C, but we can try to run neura-opt on the .mlir if it exists or was generated.
        # For now, let's assume we are running on the .mlir file itself if it's already in the repo.
        pass

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        # If it failed because it's not a Neura IR yet (e.g. LLVM dialect), this script won't work directly.
        return

    # 1. Extract MAPPING lines
    with open(tmp_mapping, 'r') as f:
        new_mapping_content = f.read()
    
    match = re.search(r'func.func @.*attributes \{.*mapping_info = \{.*\}.*\} \{', new_mapping_content)
    mapping_check_lines = []
    if match:
        func_start = match.start()
        func_body_lines = new_mapping_content[func_start:].splitlines()[:100] # Increased to 100
        for i, line in enumerate(func_body_lines):
            prefix = "// MAPPING:" if i == 0 else "// MAPPING-NEXT:"
            l = line.strip()
            if l == "}" or (i > 0 and "func.func" in line): 
                break
            mapping_check_lines.append(f"{prefix} " + l)
    else:
        # Fallback for simple mapping checks without full generation
        print("Detailed mapping_info not found with curly brace match, trying simpler match.")
        match = re.search(r'func.func @.*attributes \{.*mapping_info = \{.*\}', new_mapping_content)
        if match:
             mapping_check_lines.append("// MAPPING: " + match.group(0).strip())

    # 2. Extract YAML lines
    yaml_check_lines = []
    if os.path.exists(tmp_yaml):
        with open(tmp_yaml, 'r') as f:
            y_lines = f.readlines()[:60]
            for i, line in enumerate(y_lines):
                prefix = "// YAML:" if i == 0 else "// YAML-NEXT:"
                yaml_check_lines.append(f"{prefix} {line.strip()}")
    
    # 3. Extract ASM lines
    asm_check_lines = []
    if os.path.exists(tmp_asm):
        with open(tmp_asm, 'r') as f:
            a_lines = f.readlines()[:60]
            for i, line in enumerate(a_lines):
                prefix = "// ASM:" if i == 0 else "// ASM-NEXT:"
                if "PE(" in line: prefix = "// ASM:"
                asm_check_lines.append(f"{prefix} {line.strip()}")

    # Replace in original file
    new_file_lines = []
    added_mapping = False
    added_yaml = False
    added_asm = False
    
    # If no MAPPING/YAML/ASM exists, we might need to append them.
    # But usually they are there.
    
    for line in lines:
        if line.startswith("// MAPPING"):
            if not added_mapping and mapping_check_lines:
                new_file_lines.extend([l + "\n" for l in mapping_check_lines])
                new_file_lines.append("\n")
                added_mapping = True
            continue
        if line.startswith("// YAML"):
            if not added_yaml and yaml_check_lines:
                new_file_lines.extend([l + "\n" for l in yaml_check_lines])
                new_file_lines.append("\n")
                added_yaml = True
            continue
        if line.startswith("// ASM"):
            if not added_asm and asm_check_lines:
                new_file_lines.extend([l + "\n" for l in asm_check_lines])
                new_file_lines.append("\n")
                added_asm = True
            continue
        new_file_lines.append(line)

    with open(test_file, 'w') as f:
        f.writelines(new_file_lines)
    print(f"Updated {test_file}")

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        update_file(arg)
