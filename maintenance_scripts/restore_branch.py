import os

path = "/home/x/shiran/Project/dataflow/test/mapping_quality/branch_for.mlir"
with open("/tmp/branch_header.txt", 'r') as f:
    header = f.read().splitlines()[0].strip()
with open("/tmp/branch_yaml.txt", 'r') as f:
    yaml = f.read().splitlines()
with open("/tmp/branch_asm.txt", 'r') as f:
    asm = f.read().splitlines()

# Clean header from debug info if any
if "func.func" in header:
    header = header[header.find("func.func"):]

with open(path, 'r') as f:
    lines = f.readlines()

# Insert RUN lines back at line 44 (after MOV check)
run_lines = [
    "\n",
    "// RUN: mlir-neura-opt %s \\\n",
    "// RUN:   --assign-accelerator \\\n",
    "// RUN:   --lower-llvm-to-neura \\\n",
    "// RUN:   --fold-constant \\\n",
    "// RUN:   --canonicalize-live-in \\\n",
    "// RUN:   --leverage-predicated-value \\\n",
    "// RUN:   --transform-ctrl-to-data-flow \\\n",
    "// RUN:   --fold-constant \\\n",
    "// RUN:   --insert-data-mov \\\n",
    "// RUN:   --map-to-accelerator=\"mapping-strategy=heuristic backtrack-config=customized\" \\\n",
    "// RUN:   --architecture-spec=../arch_spec/architecture.yaml \\\n",
    "// RUN:   | FileCheck %s -check-prefix=MAPPING\n",
    "\n",
    "// RUN: mlir-neura-opt %s \\\n",
    "// RUN:   --assign-accelerator \\\n",
    "// RUN:   --lower-llvm-to-neura \\\n",
    "// RUN:   --canonicalize-live-in \\\n",
    "// RUN:   --leverage-predicated-value \\\n",
    "// RUN:   --transform-ctrl-to-data-flow \\\n",
    "// RUN:   --fold-constant \\\n",
    "// RUN:   --insert-data-mov \\\n",
    "// RUN:   --map-to-accelerator=\"mapping-strategy=heuristic backtrack-config=customized\" \\\n",
    "// RUN:   --architecture-spec=../arch_spec/architecture.yaml \\\n",
    "// RUN:   --generate-code \n",
    "// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml -check-prefix=YAML\n",
    "// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM\n"
]

# Current lines up to 43
new_lines = lines[:43] + run_lines + lines[43:]

# Append MAPPING, YAML, ASM at the end
new_lines.append("\n")
# MAPPING: just the header as per user's refined feedback (no body check)
new_lines.append(f"// MAPPING:        {header}\n")
new_lines.append("\n")

# YAML
for i, l in enumerate(yaml):
    l = l.strip()
    if not l: continue
    pref = "YAML" if (i == 0 or "array_config" in l or "column:" in l or "core_id:" in l) else "YAML-NEXT"
    new_lines.append(f"// {pref}:      {l}\n")
new_lines.append("\n")

# ASM
for i, l in enumerate(asm):
    l = l.strip()
    if not l: continue
    pref = "ASM" if (i == 0 or "PE(" in l) else "ASM-NEXT"
    new_lines.append(f"// {pref}:      {l}\n")

with open(path, 'w') as f:
    f.writelines(new_lines)
