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
    mlir_translate = f"/home/x/shiran/llvm-project/build/bin/mlir-translate"
    clang = f"/home/x/shiran/llvm-project/build/bin/clang"
    clang_plus = f"/home/x/shiran/llvm-project/build/bin/clang++"
    arch_spec = "/home/x/shiran/Project/dataflow/test/arch_spec/architecture.yaml"
    
    # Temporary files
    tmp_ll_full = "/tmp/tmp_full.ll"
    tmp_ll_only = "/tmp/tmp_only.ll"
    tmp_mlir = "/tmp/tmp_kernel.mlir"
    tmp_mapping = "/tmp/tmp_mapping.mlir"
    tmp_yaml = "tmp-generated-instructions.yaml"
    tmp_asm = "tmp-generated-instructions.asm"
    
    # Clean up
    for f in [tmp_ll_full, tmp_ll_only, tmp_mlir, tmp_mapping, tmp_yaml, tmp_asm]:
        if os.path.exists(f): 
            try: os.remove(f)
            except: pass

    is_e2e = "e2e" in test_file or "honor_arch" in test_file
    
    if is_e2e:
        # Need to compile C/C++ first
        if "fir" in test_file:
            cpp_src = "/home/x/shiran/Project/dataflow/test/benchmark/CGRA-Bench/kernels/fir/fir_int.cpp"
            subprocess.run([clang_plus, "-S", "-emit-llvm", "-O3", "-fno-vectorize", "-fno-unroll-loops", "-o", tmp_ll_full, cpp_src])
        elif "relu" in test_file:
            c_src = "/home/x/shiran/Project/dataflow/test/benchmark/CGRA-Bench/kernels/relu/relu.c"
            subprocess.run([clang, "-S", "-emit-llvm", "-O3", "-fno-vectorize", "-fno-unroll-loops", "-std=c11", "-I", os.path.dirname(c_src), "-DSMALL_DATASET", "-o", tmp_ll_full, c_src])
        elif "bicg" in test_file:
            c_src = "/home/x/shiran/Project/dataflow/test/benchmark/CGRA-Bench/kernels/bicg/bicg.c"
            subprocess.run([clang, "-S", "-emit-llvm", "-O3", "-fno-vectorize", "-fno-unroll-loops", "-std=c11", "-I", os.path.dirname(c_src), "-DSMALL_DATASET", "-o", tmp_ll_full, c_src])
        elif "histogram" in test_file:
            cpp_src = "/home/x/shiran/Project/dataflow/test/benchmark/CGRA-Bench/kernels/histogram/histogram_int.cpp"
            subprocess.run([clang_plus, "-S", "-emit-llvm", "-O3", "-fno-vectorize", "-fno-unroll-loops", "-o", tmp_ll_full, cpp_src])
        else:
             print(f"Unsupported e2e test: {test_file}")
             return

        # Fix LLVM 20 compatibility for LLVM 16 llvm-extract
        if os.path.exists(tmp_ll_full):
            with open(tmp_ll_full, 'r') as f:
                content = f.read()
            content = content.replace(" nuw ", " ")
            content = content.replace(" nneg ", " ")
            with open(tmp_ll_full, 'w') as f:
                f.write(content)
            
            subprocess.run(["llvm-extract", '--rfunc=.*kernel.*', tmp_ll_full, "-o", tmp_ll_only])
            subprocess.run([mlir_translate, "--import-llvm", tmp_ll_only, "-o", tmp_mlir])
            input_mlir = tmp_mlir
        else:
            print(f"Failed to generate {tmp_ll_full}")
            return
    else:
        input_mlir = test_file

    # Now run neura-opt
    cmd = [
        mlir_neura_opt, input_mlir,
        "--assign-accelerator",
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
    if "e2e" in test_file:
        # e2e tests have --lower-llvm-to-neura instead of some other passes
        cmd = [
            mlir_neura_opt, input_mlir,
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

    if "branch_for.mlir" in test_file and "mapping_quality" in test_file:
        for i, arg in enumerate(cmd):
            if "--map-to-accelerator" in arg:
                cmd[i] = "--map-to-accelerator=mapping-strategy=heuristic backtrack-config=simple"

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        return

    # Extract sections
    def get_lines(filename, prefix_base, limit=60):
        if not os.path.exists(filename): return []
        with open(filename, 'r') as f:
            content_lines = f.readlines()[:limit]
        res = []
        for i, line in enumerate(content_lines):
            p = f"// {prefix_base}:" if i == 0 else f"// {prefix_base}-NEXT:"
            if prefix_base == "ASM" and "PE(" in line: p = "// ASM:"
            res.append(f"{p} {line.strip()}")
        return res

    # For mapping, we need to find the function in tmp_mapping
    mapping_lines = []
    with open(tmp_mapping, 'r') as f:
        mapping_content = f.read()
    match = re.search(r'func.func @.*attributes \{.*mapping_info = \{.*\}.*\} \{', mapping_content)
    if match:
        start = match.start()
        body = mapping_content[start:].splitlines()[:60]
        for i, line in enumerate(body):
            p = "// MAPPING:" if i == 0 else "// MAPPING-NEXT:"
            if line.strip() == "}":
                mapping_lines.append(f"{p} " + line.strip())
                break
            mapping_lines.append(f"{p} " + line.strip())
    
    yaml_lines = get_lines(tmp_yaml, "YAML")
    asm_lines = get_lines(tmp_asm, "ASM")

    # Reconstruct original file
    new_file_lines = []
    added = {"MAPPING": False, "YAML": False, "ASM": False}
    source_lines = {"MAPPING": mapping_lines, "YAML": yaml_lines, "ASM": asm_lines}
    
    for line in lines:
        handled = False
        for key in ["MAPPING", "YAML", "ASM"]:
            if line.startswith(f"// {key}"):
                if not added[key]:
                    new_file_lines.extend([l + "\n" for l in source_lines[key]])
                    new_file_lines.append("\n")
                    added[key] = True
                handled = True
                break
        if not handled:
            new_file_lines.append(line)

    with open(test_file, 'w') as f:
        f.writelines(new_file_lines)
    print(f"Updated {test_file}")

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        update_file(arg)
