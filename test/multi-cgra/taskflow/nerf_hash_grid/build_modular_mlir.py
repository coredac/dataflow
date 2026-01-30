#!/cluster/home/tangyz/.conda/envs/torch-mlir-env/bin/python
"""Build modular MLIR from NeRF PyTorch components.

This script compiles individual NeRF components (ray sampler, hash encoder,
MLP) into separate MLIR modules and merges them into a single modular MLIR
file with a top-level orchestrator function.

Features:
  - Automatic function signature extraction
  - Signature-based top-level function generation
  - MLIR verification with mlir-opt
  - Command-line output path specification
"""

import argparse
import os
import re
import subprocess
import sys

import torch
import torch_mlir

from nerf_components import HashGridEncoder
from nerf_components import NeRFMLP
from nerf_components import RaySampler


def compile_single_module(module, inputs, module_name):
  """Compiles a single PyTorch module to Linalg MLIR.

  Args:
    module: PyTorch module to compile.
    inputs: Tuple of input tensors for tracing.
    module_name: Name for the module (used in debug output).

  Returns:
    MLIR string representation, or None if compilation fails.
  """
  print(f'\nCompiling module: {module_name}')
  print('-' * 70)
  print(f'  Input shapes: {[x.shape for x in inputs]}')

  try:
    mlir_module = torch_mlir.compile(
        module,
        inputs,
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
        use_tracing=True)

    mlir_str = str(mlir_module)

    # Save debug file.
    debug_file = f'{module_name}_module.mlir'
    with open(debug_file, 'w') as f:
      f.write(mlir_str)

    print(f'  ✓ Compilation successful: {debug_file}')
    print(f'    Size: {len(mlir_str):,} characters')

    return mlir_str

  except Exception as e:
    print(f'  ✗ Compilation failed: {e}')
    import traceback
    traceback.print_exc()
    return None


def extract_function_signature(mlir_str):
  """Extracts function signature from MLIR.

  Args:
    mlir_str: MLIR string containing a @forward function.

  Returns:
    Tuple of (input_types, output_types, full_signature_string).
    Returns (None, None, None) if extraction fails.
  """
  # Match function signature:
  # func.func @forward(%arg0: type0, ...) -> (type_out0, ...)
  pattern = r'func\.func @forward\((.*?)\)\s*->\s*\(([^)]+)\)'
  match = re.search(pattern, mlir_str, re.DOTALL)

  if not match:
    # Try single return value: -> type
    pattern = r'func\.func @forward\((.*?)\)\s*->\s*([^\s{]+)'
    match = re.search(pattern, mlir_str, re.DOTALL)
    if not match:
      print('    ⚠ Cannot extract function signature')
      return None, None, None

    inputs_str = match.group(1).strip()
    outputs_str = match.group(2).strip()
    output_types = [outputs_str]
  else:
    inputs_str = match.group(1).strip()
    outputs_str = match.group(2).strip()
    output_types = [t.strip() for t in outputs_str.split(',') if t.strip()]

  # Extract input types.
  input_types = []
  for param in inputs_str.split(','):
    if ':' in param:
      type_part = param.split(':', 1)[1].strip()
      input_types.append(type_part)

  full_signature = f"({inputs_str}) -> ({', '.join(output_types)})"

  return input_types, output_types, full_signature


def extract_and_rename_function(mlir_str, new_name):
  """Extracts @forward function and renames it.

  Args:
    mlir_str: MLIR string containing the function.
    new_name: New name for the function.

  Returns:
    Renamed function as string, or None if extraction fails.
  """
  lines = mlir_str.split('\n')
  func_lines = []
  brace_count = 0
  in_function = False

  for line in lines:
    if 'func.func @forward(' in line:
      in_function = True
      # Rename function
      line = line.replace('func.func @forward',
                          f'func.func @{new_name}')

    if in_function:
      func_lines.append(line)
      brace_count += line.count('{')
      brace_count -= line.count('}')

      if brace_count == 0 and len(func_lines) > 1:
        break

  return '\n'.join(func_lines) if func_lines else None


def collect_map_definitions(mlir_str):
  """Collects all affine_map definitions from MLIR.

  Args:
    mlir_str: MLIR string.

  Returns:
    List of tuples (map_name, map_definition) where map_name is like 'map'
    or 'map1' and map_definition is the full affine_map expression.
  """
  maps = []
  for line in mlir_str.split('\n'):
    if line.startswith('#map'):
      # Parse: #map = affine_map<...>
      # or:    #map1 = affine_map<...>
      match = re.match(r'#(map\d*)\s*=\s*(.+)', line)
      if match:
        map_name = match.group(1)
        map_def = match.group(2).strip()
        maps.append((map_name, map_def))
  return maps


def build_global_map_definitions(maps_list1, maps_list2, maps_list3):
  """Builds global map definitions and renaming mappings for each module.

  Args:
    maps_list1: List of (map_name, map_def) tuples from module 1.
    maps_list2: List of (map_name, map_def) tuples from module 2.
    maps_list3: List of (map_name, map_def) tuples from module 3.

  Returns:
    Tuple of (global_map_lines, rename_map1, rename_map2, rename_map3) where:
    - global_map_lines: List of global map definition strings.
    - rename_mapX: Dict mapping old map name to new global map name for module X.
  """
  # Track unique map definitions and assign global names.
  unique_maps = {}  # map_def -> global_name
  global_map_lines = []
  global_counter = 0

  # Process all maps from all modules.
  all_module_maps = [
      ('module1', maps_list1),
      ('module2', maps_list2),
      ('module3', maps_list3),
  ]

  rename_maps = [{}, {}, {}]  # One dict per module.

  for module_idx, (module_name, maps_list) in enumerate(all_module_maps):
    for old_name, map_def in maps_list:
      if map_def not in unique_maps:
        # New unique map definition - assign global name.
        if global_counter == 0:
          global_name = 'map'
        else:
          global_name = f'map{global_counter}'
        global_counter += 1

        unique_maps[map_def] = global_name
        global_map_lines.append(f'#{global_name} = {map_def}')

      # Record the renaming: old_name -> global_name.
      global_name = unique_maps[map_def]
      rename_maps[module_idx][old_name] = global_name

  return global_map_lines, rename_maps[0], rename_maps[1], rename_maps[2]


def rename_maps_in_function(func_str, rename_map):
  """Renames map references in a function body.

  Args:
    func_str: Function definition as string.
    rename_map: Dict mapping old map names to new map names.

  Returns:
    Function string with renamed map references.
  """
  # Use a callback function for atomic replacements to avoid chaining
  def replace_callback(match):
    map_name = match.group(1)  # Capture the map name without '#'
    return '#' + rename_map.get(map_name, map_name)
  
  # Build pattern that matches any of the old map names
  # Sort by length (descending) to match longer names first (e.g., map10 before map1)
  sorted_names = sorted(rename_map.keys(), key=len, reverse=True)
  if not sorted_names:
    return func_str
  
  # Create pattern: #(map10|map1|map|...)(?=\W|$)
  pattern = r'#(' + '|'.join(re.escape(name) for name in sorted_names) + r')(?=\W|$)'
  
  # Replace all matches in a single pass (atomic operation)
  result = re.sub(pattern, replace_callback, func_str)
  
  return result


def build_wrapper_function(sig1, sig2, sig3):
  """Generates top-level orchestrator function based on signatures.

  Args:
    sig1: Ray sampler signature (input_types, output_types, full_sig).
    sig2: Hash encoder signature.
    sig3: NeRF MLP signature.

  Returns:
    Top-level function as string.
  """
  in1, out1, _ = sig1
  in2, out2, _ = sig2
  in3, out3, _ = sig3

  # Validate type compatibility.
  print('\nValidating type compatibility:')
  print(f'  ray_sampler output: {out1}')
  print(f'  hash_encoder input: {in2}')
  print(f'  hash_encoder output: {out2}')
  print(f'  nerf_mlp input: {in3}')
  print(f'  nerf_mlp output: {out3}')

  if len(out1) != 1 or len(in2) != 1:
    print('  ⚠ Warning: ray_sampler → hash_encoder type mismatch')
  if len(out2) != 1 or len(in3) < 1:
    print('  ⚠ Warning: hash_encoder → nerf_mlp type mismatch')

  # Generate top-level function.
  # Inputs: Same as ray_sampler.
  # Outputs: Same as nerf_mlp.
  wrapper_inputs = ', '.join([f'%arg{i}: {t}' for i, t in enumerate(in1)])
  wrapper_outputs = ', '.join(out3)

  wrapper = f'''  func.func @nerf_forward({wrapper_inputs}) 
                       -> ({wrapper_outputs}) {{
    // ================================================
    // Task 1: Ray Sampling
    // ================================================
    %positions = func.call @ray_sampler_func({', '.join([f'%arg{i}' for i in range(len(in1))])}) 
                 : ({', '.join(in1)}) -> {out1[0]}
    
    // ================================================
    // Task 2: Hash Encoding
    // ================================================
    %encoded = func.call @hash_encoder_func(%positions)
               : ({out1[0]}) -> {out2[0]}
    
    // ================================================
    // Task 3: MLP Inference
    // ================================================
'''

  # Handle MLP's multiple inputs (encoded + view_dirs).
  if len(in3) == 2:
    wrapper += f'''    %density, %rgb = func.call @nerf_mlp_func(%encoded, %arg{len(in1)-1})
                         : ({out2[0]}, {in1[-1]}) -> ({', '.join(out3)})
    
    return %density, %rgb : {', '.join(out3)}
  }}
'''
  else:
    wrapper += f'''    %result = func.call @nerf_mlp_func(%encoded)
                      : ({out2[0]}) -> ({', '.join(out3)})
    
    return %result : {', '.join(out3)}
  }}
'''

  return wrapper


def merge_mlir_modules(mlir1, mlir2, mlir3):
  """Merges three MLIR modules into a single modular MLIR file.

  Args:
    mlir1: MLIR string for ray sampler.
    mlir2: MLIR string for hash encoder.
    mlir3: MLIR string for NeRF MLP.

  Returns:
    Merged MLIR string, or None if merging fails.
  """
  print('\n' + '=' * 70)
  print('Merging Modules')
  print('=' * 70)

  # Extract signatures.
  print('\nExtracting function signatures...')
  sig1 = extract_function_signature(mlir1)
  sig2 = extract_function_signature(mlir2)
  sig3 = extract_function_signature(mlir3)

  if None in [sig1[0], sig2[0], sig3[0]]:
    print('✗ Failed to extract function signatures')
    return None

  print('  ✓ Signature extraction successful')

  # Extract function definitions.
  print('\nExtracting function definitions...')
  func1 = extract_and_rename_function(mlir1, 'ray_sampler_func')
  func2 = extract_and_rename_function(mlir2, 'hash_encoder_func')
  func3 = extract_and_rename_function(mlir3, 'nerf_mlp_func')

  if not all([func1, func2, func3]):
    print('✗ Failed to extract function definitions')
    return None

  print('  ✓ Function extraction successful')

  # Collect and rename all map definitions.
  print('\nCollecting affine_map definitions...')
  maps1 = collect_map_definitions(mlir1)
  maps2 = collect_map_definitions(mlir2)
  maps3 = collect_map_definitions(mlir3)
  
  print(f'  Module 1: {len(maps1)} maps')
  print(f'  Module 2: {len(maps2)} maps')
  print(f'  Module 3: {len(maps3)} maps')

  # Build global map definitions and rename mappings.
  print('\nBuilding global map definitions with renaming...')
  global_map_lines, rename_map1, rename_map2, rename_map3 = \
      build_global_map_definitions(maps1, maps2, maps3)
  
  print(f'  ✓ Created {len(global_map_lines)} unique global map definitions')
  
  # Rename map references in each function.
  print('\nRenaming map references in functions...')
  func1 = rename_maps_in_function(func1, rename_map1)
  func2 = rename_maps_in_function(func2, rename_map2)
  func3 = rename_maps_in_function(func3, rename_map3)
  print('  ✓ Map references renamed successfully')

  # Generate top-level function.
  print('\nGenerating top-level function...')
  wrapper = build_wrapper_function(sig1, sig2, sig3)
  print('  ✓ Top-level function generation successful')

  # Assemble final MLIR.
  merged = '\n'.join(global_map_lines) + '\n' if global_map_lines else ''
  merged += 'module {\n'
  merged += ('  ml_program.global private mutable @global_seed'
             '(dense<0> : tensor<i64>) : tensor<i64>\n\n')
  merged += '  // ============================================\n'
  merged += '  // Module 1: Ray Sampler\n'
  merged += '  // ============================================\n'
  merged += indent_mlir(func1, 2) + '\n\n'
  merged += '  // ============================================\n'
  merged += '  // Module 2: Hash Grid Encoder\n'
  merged += '  // ============================================\n'
  merged += indent_mlir(func2, 2) + '\n\n'
  merged += '  // ============================================\n'
  merged += '  // Module 3: NeRF MLP\n'
  merged += '  // ============================================\n'
  merged += indent_mlir(func3, 2) + '\n\n'
  merged += '  // ============================================\n'
  merged += '  // Top-level Function (Auto-generated)\n'
  merged += '  // ============================================\n'
  merged += wrapper + '\n'
  merged += '}\n'

  return merged


def fix_tensor_expand_shape_syntax(mlir_str):
  """Fixes tensor.expand_shape syntax for LLVM 20+ compatibility.
  
  Converts old syntax:
    %x = tensor.expand_shape %y [[0, 1]] : tensor<16xf32> into tensor<1x16xf32>
  
  To new syntax:
    %x = tensor.expand_shape %y [[0, 1]] output_shape [1, 16] : tensor<16xf32> into tensor<1x16xf32>
  
  Args:
    mlir_str: MLIR string to fix.
  
  Returns:
    Fixed MLIR string.
  """
  lines = mlir_str.split('\n')
  fixed_lines = []
  
  for line in lines:
    # Match tensor.expand_shape pattern
    # Pattern: tensor.expand_shape %var [[...]] : tensor<...> into tensor<shape>
    match = re.search(
        r'(.*tensor\.expand_shape\s+%\S+\s+\[\[.*?\]\])\s*:\s*(tensor<[^>]+>)\s+into\s+tensor<([^>]+)>',
        line
    )
    
    if match:
      prefix = match.group(1)  # Everything before ':'
      input_type = match.group(2)  # tensor<16xf32>
      output_shape = match.group(3)  # 1x16xf32
      
      # Extract shape dimensions from output_shape
      # Remove type suffix (e.g., 'xf32', 'xi64')
      shape_str = re.sub(r'x[a-z]\w+$', '', output_shape)
      # Split by 'x' to get dimensions
      dims = shape_str.split('x')
      
      # Build output_shape attribute
      output_shape_attr = f"output_shape [{', '.join(dims)}]"
      
      # Reconstruct the line with output_shape attribute
      fixed_line = f"{prefix} {output_shape_attr} : {input_type} into tensor<{output_shape}>"
      
      # Preserve any trailing content (like comments)
      trailing = line[match.end():]
      fixed_line += trailing
      
      fixed_lines.append(fixed_line)
    else:
      # No match, keep original line
      fixed_lines.append(line)
  
  return '\n'.join(fixed_lines)


def indent_mlir(mlir_str, spaces):
  """Adds indentation to MLIR string.

  Args:
    mlir_str: MLIR string to indent.
    spaces: Number of spaces for indentation.

  Returns:
    Indented MLIR string.
  """
  lines = mlir_str.split('\n')
  indent = ' ' * spaces
  return '\n'.join(indent + line if line.strip() else line for line in lines)


def verify_mlir(mlir_file):
  """Verifies MLIR file using mlir-opt.

  Args:
    mlir_file: Path to MLIR file to verify.

  Returns:
    True if verification succeeds, False otherwise.
  """
  print('\nVerifying MLIR file...')

  mlir_opt = '../../../../../build/tools/mlir-neura-opt/mlir-neura-opt'

  if not os.path.exists(mlir_opt):
    print('  ⚠ mlir-neura-opt not found, skipping verification')
    return True

  result = subprocess.run(
      [mlir_opt, mlir_file, '--verify-each=true', '-o', '/dev/null'],
      capture_output=True,
      text=True)

  if result.returncode == 0:
    print('  ✅ MLIR verification passed!')
    return True
  else:
    print('  ✗ MLIR verification failed:')
    print(result.stderr)
    return False


def main():
  """Main workflow."""
  # Parse command-line arguments.
  parser = argparse.ArgumentParser(
      description='Build modular MLIR from NeRF components')
  parser.add_argument(
      '--output',
      '-o',
      default='nerf_modular_3funcs.mlir',
      help='Output file path (default: nerf_modular_3funcs.mlir)')
  args = parser.parse_args()

  print('=' * 70)
  print('Build Modular MLIR (Auto Signature Extraction)')
  print('=' * 70)
  print(f'Output file: {args.output}')
  print('=' * 70)

  device = torch.device('cpu')

  # Compile 3 modules.
  sampler = RaySampler(num_samples=16)
  sampler.eval()
  mlir1 = compile_single_module(sampler,
                                 (torch.randn(2, 3), torch.randn(2, 3)),
                                 'ray_sampler')

  encoder = HashGridEncoder(
      num_levels=2, features_per_level=2, log2_hashmap_size=8)
  encoder.eval()
  mlir2 = compile_single_module(encoder, (torch.randn(2, 16, 3),),
                                 'hash_encoder')

  mlp = NeRFMLP(input_dim=4, hidden_dim=32, num_layers=2)
  mlp.eval()
  mlir3 = compile_single_module(
      mlp, (torch.randn(2, 16, 4), torch.randn(2, 3)), 'nerf_mlp')

  if not all([mlir1, mlir2, mlir3]):
    print('\n✗ Some modules failed to compile')
    return 1

  # Merge modules.
  merged = merge_mlir_modules(mlir1, mlir2, mlir3)

  if not merged:
    print('\n✗ Module merging failed')
    return 1

  # Fix tensor.expand_shape syntax for LLVM 20+ compatibility.
  print('\nApplying syntax fixes for LLVM 20+ compatibility...')
  merged = fix_tensor_expand_shape_syntax(merged)
  
  if 'output_shape [' in merged:
    print('  ✓ Fixed tensor.expand_shape syntax')
  else:
    print('  ℹ No tensor.expand_shape operations found')

  # Save output.
  output_file = args.output

  # Ensure output directory exists.
  output_dir = os.path.dirname(output_file)
  if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

  with open(output_file, 'w') as f:
    f.write(merged)

  print('\n' + '=' * 70)
  print('✓ Modular MLIR generated successfully!')
  print('=' * 70)
  print(f'  File: {output_file}')
  print(f'  Size: {len(merged):,} characters')

  # Statistics.
  num_funcs = merged.count('func.func')
  num_calls = merged.count('func.call')

  print('\nStructure:')
  print(f'  Function definitions: {num_funcs} (3 modules + 1 top-level)')
  print(f'  Function calls: {num_calls} (top-level calls 3 modules)')

  # Verification.
  if verify_mlir(output_file):
    print('\n' + '=' * 70)
    print('Next Step: Compile to Taskflow')
    print('=' * 70)
    print(f'\nmlir-neura-opt {output_file} \\')
    print('  --one-shot-bufferize \\')
    print('  --pass-pipeline=\'func.func(convert-linalg-to-affine-loops)\' \\')
    print('  --convert-affine-to-taskflow \\')
    print('  -o nerf_taskflow_3tasks.mlir')
    print('\nExpected: Generate 3 taskflow.task operations')

    return 0
  else:
    print('\n⚠ MLIR verification failed, but file was generated')
    print(f'  You can try manual inspection: {output_file}')
    return 1


if __name__ == '__main__':
  sys.exit(main())
