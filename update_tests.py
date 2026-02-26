import re, sys

tests = [
    "irregular-loop/irregular-loop",
    "parallel-nested/parallel-nested",
    "multi-nested/multi-nested",
    "resnet/simple_resnet_tosa"
]

for t in tests:
    base = t.split("/")[0]
    fname = t.split("/")[1]
    test_file = f"/home/x/shiran/Project/dataflow/test/multi-cgra/taskflow/{t}.mlir"
    out_file = f"/home/x/shiran/Project/dataflow/test/multi-cgra/taskflow/{base}/Output/{fname}.mlir.tmp.resopt.mlir"
    
    try:
        with open(out_file, "r") as f:
            out_content = f.read()
    except FileNotFoundError:
        print(f"Skipping {t}, output file not found")
        continue

    # Extract tile map
    map_match = re.search(r'tile_occupation_map = "([^"]+)"', out_content)
    if not map_match:
        print(f"No tile map found in {t}")
        continue
    tile_map = map_match.group(1)
    
    # Extract attributes for all tasks
    task_attrs = []
    for line in out_content.split('\n'):
        if 'taskflow.task' in line and 'cgra_count =' in line:
            m = re.search(r'{(cgra_count = [^}]+)}', line)
            if m:
                task_attrs.append(m.group(1))

    # Read test file
    with open(test_file, "r") as f:
        test_content = f.read()
        
    # Replace test content
    # First, replace the tile map check. It might not exist, so we will put it right after the first `RESOPT` block start.
    
    # Replace individual task attrs
    out_lines = []
    attr_idx = 0
    in_resopt = False
    added_tile_map = False
    
    for line in test_content.split('\n'):
        if line.strip() == '// RESOPT:      func.func @':
            # This handles irregular-loop if we added it
            continue
        if 'RESOPT-SAME: tile_occupation_map' in line:
            continue
            
        if line.startswith('// RESOPT:') and not added_tile_map:
            # First RESOPT line found
            out_lines.append('// RESOPT:      func.func @')
            out_lines.append(f'// RESOPT-SAME: tile_occupation_map = "{tile_map}"')
            added_tile_map = True
            
        if line.startswith('// RESOPT-SAME: {cgra_count'):
            if attr_idx < len(task_attrs):
                out_lines.append(f'// RESOPT-SAME: {{{task_attrs[attr_idx]}}}')
                attr_idx += 1
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)
            
    # Also append the visual map at the very end
    out_lines.append('')
    out_lines.append('// Tile Occupation Map:')
    for ml in tile_map.split('\\0A'):
        out_lines.append(f'// {ml}')
        
    with open(test_file, "w") as f:
        f.write('\n'.join(out_lines))
    print(f"Updated {t}")
