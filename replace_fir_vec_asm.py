
import sys

target = 'test/e2e/fir/fir_kernel_vec.mlir'
with open(target, 'r') as f:
    lines = f.readlines()

asm_checks_file = 'fir_vec_asm_checks.txt'
with open(asm_checks_file, 'r') as f:
    new_asm_checks = f.read().strip()

# find range for ASM block
start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if '// ASM:      # Compiled II: 5' in line:
        start_idx = i
        break

if start_idx != -1:
    # Use end of file since ASM is usually at the end
    end_idx = len(lines)
    
    new_lines = lines[:start_idx] + [new_asm_checks + '\n']
    with open(target, 'w') as f:
        f.writelines(new_lines)
    print(f"Updated {target}")
else:
    print(f"Could not find ASM block in {target}")
