
import sys

def convert_to_checks(input_file, check_prefix="MAPPING"):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    output = []
    first = True
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if first:
            output.append(f"// {check_prefix}: {line}")
            first = False
        else:
            output.append(f"// {check_prefix}-NEXT: {line}")
    return "\n".join(output)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)
    
    print(convert_to_checks(sys.argv[1]))
