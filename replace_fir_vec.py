
import sys

# For fir_kernel_vec.mlir
# The text block to replace is not just a single string but lines that we can identify.
# We will read file, find start and end of MAPPING block, and replace it.

target = 'test/e2e/fir/fir_kernel_vec.mlir'
with open(target, 'r') as f:
    lines = f.readlines()

checks_file = 'fir_vec_checks.txt'
with open(checks_file, 'r') as f:
    new_checks = f.read().strip()

# find range
start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if '// MAPPING:      func.func @_Z6kernelPiS_S_(' in line:
        start_idx = i
        break

if start_idx != -1:
    # Look for the end of the MAPPING block.
    # It seems to end at neura.return_value line, or before the empty line before YAML.
    for i in range(start_idx, len(lines)):
        if line.strip() == '' and i > start_idx + 5: # Assuming mapping block is somewhat long
             # Actually the current block is short, ends at line 32. Line 33 is empty.
             if lines[i].strip() == '':
                 end_idx = i
                 break
        if '// MAPPING:      neura.return_value' in lines[i]:
             # This is the last line of the simplified block
             pass
    
    if end_idx == -1:
        # scan a bit further?
        # Just find the empty line after start_idx
        for i in range(start_idx, len(lines)):
            if not lines[i].strip():
                end_idx = i
                break

if start_idx != -1 and end_idx != -1:
    new_lines = lines[:start_idx] + [new_checks + '\n'] + lines[end_idx:]
    with open(target, 'w') as f:
        f.writelines(new_lines)
    print(f"Updated {target}")
else:
    print(f"Could not find block in {target}")
