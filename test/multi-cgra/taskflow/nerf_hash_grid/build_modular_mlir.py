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
      # Rename and mark as private.
      line = line.replace('func.func @forward',
                          f'func.func private @{new_name}')

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
    List of unique affine_map definition strings.
  """
  maps = []
  for line in mlir_str.split('\n'):
    if line.startswith('#map'):
      if line not in maps:  # Deduplicate.
        maps.append(line)
  return maps


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
               : {out1[0]} -> {out2[0]}
    
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
                      : {out2[0]} -> ({', '.join(out3)})
    
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

  # Collect all map definitions.
  print('\nCollecting affine_map definitions...')
  all_maps = collect_map_definitions(mlir1)
  all_maps.extend(collect_map_definitions(mlir2))
  all_maps.extend(collect_map_definitions(mlir3))
  all_maps = list(dict.fromkeys(all_maps))  # Deduplicate, preserve order.

  print(f'  ✓ Collected {len(all_maps)} map definitions')

  # Generate top-level function.
  print('\nGenerating top-level function...')
  wrapper = build_wrapper_function(sig1, sig2, sig3)
  print('  ✓ Top-level function generation successful')

  # Assemble final MLIR.
  merged = '\n'.join(all_maps) + '\n' if all_maps else ''
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
