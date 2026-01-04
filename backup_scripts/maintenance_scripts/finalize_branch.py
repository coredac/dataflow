import os

path = "/home/x/shiran/Project/dataflow/test/mapping_quality/branch_for.mlir"

mapping_header = 'func.func @loop_test() -> f32 attributes {accelerator = "neura", dataflow_mode = "predicate", mapping_info = {compiled_ii = 4 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 4 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {'

yaml_content = [
    "array_config:",
    "  columns: 4",
    "  rows: 4",
    "  compiled_ii: 4",
    "  cores:",
    "    - column: 0",
    "      row: 0",
    "      core_id: \"0\"",
    "      entries:",
    "        - entry_id: \"entry0\"",
    "          instructions:",
    "            - index_per_ii: 0",
    "              operations:",
    "                - opcode: \"GRANT_ONCE\"",
    "                  id: 1",
    "                  time_step: 0",
    "                  invalid_iterations: 0",
    "                  src_operands:",
    "                    - operand: \"#0\"",
    "                      color: \"RED\"",
    "                  dst_operands:",
    "                    - operand: \"EAST\"",
    "                      color: \"RED\"",
    "    - column: 1"
]

asm_content = [
    "# Compiled II: 4",
    "",
    "PE(0,0):",
    "{",
    "  GRANT_ONCE, [#0] -> [EAST, RED] (t=0, inv_iters=0)",
    "} (idx_per_ii=0)",
    "",
    "PE(1,0):",
    "{",
    "  PHI_START, [WEST, RED], [EAST, RED] -> [NORTH, RED] (t=1, inv_iters=0)",
    "} (idx_per_ii=1)",
    "{",
    "  DATA_MOV, [NORTH, RED] -> [EAST, RED] (t=3, inv_iters=0)",
    "} (idx_per_ii=3)",
    "",
    "PE(2,0):",
    "{",
    "  GRANT_PREDICATE, [WEST, RED], [NORTH, RED] -> [WEST, RED] (t=4, inv_iters=1)",
    "} (idx_per_ii=0)",
    "",
    "PE(3,0):",
    "{",
    "  GRANT_ONCE, [#10] -> [NORTH, RED] (t=1, inv_iters=0)",
    "} (idx_per_ii=1)"
]

with open(path, 'r') as f:
    lines = f.readlines()

# Revert RUN lines to simple form
new_lines = []
for l in lines:
    if "grep -v" in l:
        l = l.replace('| grep -v "\\[DEBUG\\]" ', "")
    if l.strip().startswith("// MAPPING") or l.strip().startswith("// YAML") or l.strip().startswith("// ASM"):
        continue
    new_lines.append(l)

# Append sections
new_lines.append("\n")
# Use MAPPING: without NEXT to skip debug info at the start
new_lines.append(f"// MAPPING:        {mapping_header}\n")
new_lines.append("\n")

for i, val in enumerate(yaml_content):
    pref = "YAML" if (i == 0 or "column:" in val or "core_id:" in val or "array_config" in val) else "YAML-NEXT"
    new_lines.append(f"// {pref}:      {val}\n")
new_lines.append("\n")

for i, val in enumerate(asm_content):
    if not val.strip():
        # Blank line in ASM output? Let's just use ASM: to be safe or skip
        new_lines.append(f"// ASM:\n")
        continue
    pref = "ASM" if (i == 0 or "PE(" in val) else "ASM-NEXT"
    new_lines.append(f"// {pref}:      {val}\n")

with open(path, 'w') as f:
    f.writelines(new_lines)
