import os
import subprocess

PROJECT_ROOT = "/home/x/shiran/Project/dataflow"
test_path = os.path.join(PROJECT_ROOT, "test/mapping_quality/branch_for.mlir")

# Step 1: Run the command to get the latest MAPPING IR
cmd = (
    f"{PROJECT_ROOT}/build/tools/mlir-neura-opt/mlir-neura-opt {test_path} "
    "--assign-accelerator --lower-llvm-to-neura --canonicalize-live-in "
    "--leverage-predicated-value --transform-ctrl-to-data-flow --fold-constant "
    "--insert-data-mov --map-to-accelerator=\"mapping-strategy=heuristic backtrack-config=customized\" "
    "--architecture-spec=test/arch_spec/architecture.yaml 2>/dev/null"
)
res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
stdout = res.stdout.splitlines()

# Extract the function header
header = ""
for l in stdout:
    if "func.func @loop_test" in l:
        header = l.strip()
        break

# Step 2: Run the command to get YAML/ASM
cmd_gen = cmd + " --generate-code"
subprocess.run(f"rm -f tmp-generated-instructions.*", shell=True, cwd=PROJECT_ROOT)
subprocess.run(cmd_gen, shell=True, cwd=PROJECT_ROOT)

yaml_lines = []
if os.path.exists(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.yaml")):
    with open(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.yaml"), 'r') as f:
        yaml_lines = f.read().splitlines()[:25]

asm_lines = []
if os.path.exists(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.asm")):
    with open(os.path.join(PROJECT_ROOT, "tmp-generated-instructions.asm"), 'r') as f:
        asm_lines = f.read().splitlines()[:25]

# Step 3: Read current file and prepare replacement
with open(test_path, 'r') as f:
    lines = f.readlines()

# Filter out old MAPPING/YAML/ASM sections and RUN lines
# Find first RUN line related to MAPPING/YAML/ASM
first_bad_run = -1
for i, l in enumerate(lines):
    if "| FileCheck %s -check-prefix=MAPPING" in l:
        first_bad_run = i
        break

if first_bad_run != -1:
    # Go back to the preceding comment block
    start_del = first_bad_run
    while start_del > 0 and (lines[start_del-1].strip().startswith("// RUN:") or lines[start_del-1].strip() == ""):
        start_del -= 1
    # Actually, let's just find the exact chunk I inserted
    # I'll just look for line 44 as a fixed anchor for this specific task
    # (Since I know I put them there)
    # But a safer way is to find the function definition and delete everything after it that starts with // MAPPING: etc.
    pass

# Let's just rewrite the file content between line 44 and the end
# Based on Step 3168 state:
# Lines 1-43 are basic RUNs.
# Line 44 is empty.
# Line 45 is func.func.
# I'll rebuild from Step 3168 baseline.

# Re-read from a "cleanish" version if possible, or just slice.
# Lines 1-43 are good.
# Lines 45-78 are the function and CHECK/CANONICALIZE/CTRL2DATA/FUSE/MOV checks.
# I will append the NEW RUN lines and NEW checks at the end of the file.

fixed_lines = []
for l in lines:
    if "// RUN:" in l and any(x in l for x in ["-check-prefix=MAPPING", "-check-prefix=YAML", "-check-prefix=ASM"]):
        continue
    if any(l.strip().startswith(f"// {p}") for p in ["MAPPING", "YAML", "ASM"]):
        continue
    fixed_lines.append(l)

# Add RUN lines
new_run_lines = [
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
    "// RUN:   | grep -v \"\\[DEBUG\\]\" | FileCheck %s -check-prefix=MAPPING\n",
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

# Insert after existing RUNs (around line 43)
found_pos = -1
for i, l in enumerate(fixed_lines):
    if "func.func" in l:
        found_pos = i
        break

if found_pos != -1:
    final_lines = fixed_lines[:found_pos] + new_run_lines + fixed_lines[found_pos:]
else:
    final_lines = fixed_lines + new_run_lines

# Append checks
final_lines.append("\n")
final_lines.append(f"// MAPPING:        {header}\n")
final_lines.append("\n")

final_lines.append("// YAML:      array_config:\n")
for i, l in enumerate(yaml_lines[1:]):
    final_lines.append(f"// YAML-NEXT: {l}\n")
final_lines.append("\n")

final_lines.append(f"// ASM:      {asm_lines[0]}\n")
for i, l in enumerate(asm_lines[1:]):
    final_lines.append(f"// ASM-NEXT: {l}\n")

with open(test_path, 'w') as f:
    f.writelines(final_lines)
