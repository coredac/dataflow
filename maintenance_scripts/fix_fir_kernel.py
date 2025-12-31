import os
import re
import subprocess

PROJECT_ROOT = "/home/x/shiran/Project/dataflow"
test_path = os.path.join(PROJECT_ROOT, "test/e2e/fir/fir_kernel.mlir")

# Generate exact output
subprocess.run("clang++ -S -emit-llvm -O3 -fno-vectorize -fno-unroll-loops -o /tmp/fir.ll test/benchmark/CGRA-Bench/kernels/fir/fir_int.cpp", shell=True, cwd=PROJECT_ROOT)
subprocess.run("llvm-extract --rfunc=\".*kernel.*\" /tmp/fir.ll -o /tmp/fir_kernel.ll", shell=True, cwd=PROJECT_ROOT)
subprocess.run("mlir-translate --import-llvm /tmp/fir_kernel.ll -o /tmp/fir_kernel.mlir", shell=True, cwd=PROJECT_ROOT)
subprocess.run("/home/x/shiran/Project/dataflow/build/tools/mlir-neura-opt/mlir-neura-opt /tmp/fir_kernel.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-cast --promote-func-arg-to-const --fold-constant --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant --insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic\" --architecture-spec=test/arch_spec/architecture.yaml --generate-code -o /tmp/fir_fixed.mlir", shell=True, cwd=PROJECT_ROOT)

with open("/tmp/fir_fixed.mlir", 'r') as f:
    ir = f.read()
match = re.search(r'(func\.func @.*?\{.*?\})', ir, re.DOTALL)
ir_lines = match.group(1).splitlines()

with open(test_path, 'r') as f:
    lines = f.readlines()

# Find and replace MAPPING block
s = -1
for i, l in enumerate(lines):
    if "// MAPPING:" in l:
        s = i
        break
e = len(lines)
for j in range(s + 1, len(lines)):
    if not (lines[j].strip().startswith("// MAPPING") or lines[j].strip().startswith("//MAPPING") or lines[j].strip() == "//"):
        e = j
        break

new_block = []
for k, l in enumerate(ir_lines):
    if not l.strip(): continue
    if k == 0: new_block.append(f"// MAPPING:      {l.strip()}\n")
    else: new_block.append(f"// MAPPING-NEXT: {l.strip()}\n")

lines = lines[:s] + new_block + lines[e:]

with open(test_path, 'w') as f:
    f.writelines(lines)
