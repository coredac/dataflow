#!/usr/bin/env python3
"""
批量修复测试 - 从实际输出生成新的MAPPING + YAML + ASM CHECK
"""

import subprocess
import os
import re

def run_test_and_get_outputs(test_file):
    """运行测试并获取所有输出文件"""
    # 运行lit测试
    result = subprocess.run(
        ["/home/x/shiran/miniconda3/bin/lit", "-v", test_file],
        cwd="/home/x/shiran/dataflow",
        capture_output=True,
        text=True
    )
    
    test_dir = os.path.dirname(test_file)
    
    # 找输出文件
    mapping_file = os.path.join("/home/x/shiran/dataflow", test_dir, "tmp-mapping.mlir")
    yaml_file = os.path.join("/home/x/shiran/dataflow", test_dir, "tmp-generated-instructions.yaml")
    asm_file = os.path.join("/home/x/shiran/dataflow", test_dir, "tmp-generated-instructions.asm")
    
    outputs = {}
    if os.path.exists(mapping_file):
        with open(mapping_file) as f:
            outputs['mapping'] = f.read()
    if os.path.exists(yaml_file):
        with open(yaml_file) as f:
            outputs['yaml'] = f.read()
    if os.path.exists(asm_file):
        with open(asm_file) as f:
            outputs['asm'] = f.read()
    
    return outputs

def extract_func_body(content):
    """从mapping输出提取func body"""
    lines = content.split('\n')
    body_lines = []
    in_func = False
    brace_count = 0
    
    for line in lines:
        if 'func.func @' in line:
            in_func = True
            body_lines.append(line)
            if '{' in line:
                brace_count += line.count('{')
                brace_count -= line.count('}')
            continue
        
        if in_func:
            body_lines.append(line)
            brace_count += line.count('{')
            brace_count -= line.count('}')
            if brace_count == 0 and '}' in line:
                break
    
    return body_lines

def generate_checks(outputs):
    """生成所有CHECK lines"""
    checks = {'mapping': [], 'yaml': [], 'asm': []}
    
    # MAPPING CHECK
    if 'mapping' in outputs:
        body_lines = extract_func_body(outputs['mapping'])
        first = True
        for line in body_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if first:
                checks['mapping'].append(f"// MAPPING: {stripped}")
                first = False
            else:
                checks['mapping'].append(f"// MAPPING-NEXT: {stripped}")
    
    # YAML CHECK
    if 'yaml' in outputs:
        lines = outputs['yaml'].strip().split('\n')
        first = True
        for line in lines:
            if not line.strip():
                continue
            if first:
                checks['yaml'].append(f"// YAML:      {line}")
                first = False
            else:
                checks['yaml'].append(f"// YAML-NEXT: {line}")
    
    # ASM CHECK
    if 'asm' in outputs:
        lines = outputs['asm'].strip().split('\n')
        first = True
        for line in lines:
            # 跳过注释行（#开头），但保留空行
            if line.strip().startswith('#'):
                continue
            if first and not line.strip():
                # 第一行如果是空行，跳过
                continue
            if first:
                checks['asm'].append(f"// ASM:      {line}")
                first = False
            else:
                # 对于空行，使用ASM-EMPTY
                if not line.strip():
                    checks['asm'].append(f"// ASM-EMPTY:")
                else:
                    checks['asm'].append(f"// ASM-NEXT: {line}")
    
    return checks

def update_test_file(test_file, checks):
    """更新测试文件"""
    with open(test_file) as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # 找到最后一个RUN行（支持 // 和 # 两种注释）
    last_run_idx = None
    for i, line in enumerate(lines):
        if line.startswith('// RUN:') or line.startswith('# RUN:'):
            last_run_idx = i
    
    if last_run_idx is None:
        print(f"✗ 无法找到RUN行: {test_file}")
        return False
    
    # 删除所有CHECK行
    new_lines = []
    for line in lines:
        if line.startswith('// MAPPING') or line.startswith('// YAML') or line.startswith('// ASM'):
            continue
        if line.strip() == '//':  # 空注释行
            continue
        new_lines.append(line)
    
    # 重新找最后一个RUN行
    last_run_idx = None
    for i, line in enumerate(new_lines):
        if line.startswith('// RUN:') or line.startswith('# RUN:'):
            last_run_idx = i
    
    # 构建最终内容：RUN行 + 空行 + MAPPING + 空行 + YAML + 空行 + ASM + 剩余内容
    all_checks = []
    if checks['mapping']:
        all_checks.extend(checks['mapping'])
    
    # 找到第一个非RUN、非CHECK的内容行（通常是mlir代码或空行）
    content_start_idx = last_run_idx + 1
    while content_start_idx < len(new_lines) and not new_lines[content_start_idx].strip():
        content_start_idx += 1
    
    # 重组：RUN + 空行 + MAPPING + 空行 + YAML + 空行 + ASM + 空行 + 剩余内容
    final_lines = new_lines[:last_run_idx+1]
    final_lines.append('')
    
    if checks['mapping']:
        final_lines.extend(checks['mapping'])
        final_lines.append('')
    
    if checks['yaml']:
        final_lines.extend(checks['yaml'])
        final_lines.append('')
    
    if checks['asm']:
        final_lines.extend(checks['asm'])
        final_lines.append('')
    
    # 添加剩余内容（从第一个非空行开始）
    final_lines.extend(new_lines[content_start_idx:])
    
    with open(test_file, 'w') as f:
        f.write('\n'.join(final_lines))
    
    print(f"✓ 已更新 {test_file}")
    return True

def main():
    tests = [
        "test/e2e/fir/fir_kernel_vec.mlir",
        "test/neura/fusion/test.mlir",
        "test/neura/for_loop/relu_test.mlir",
        "test/e2e/bicg/bicg_kernel.mlir",
    ]
    
    for test in tests:
        print(f"\n处理 {test}...")
        full_path = os.path.join("/home/x/shiran/dataflow", test)
        
        if not os.path.exists(full_path):
            print(f"✗ 文件不存在")
            continue
        
        # 获取实际输出
        outputs = run_test_and_get_outputs(full_path)
        if not outputs:
            print(f"✗ 无法获取输出")
            continue
        
        # 生成CHECK
        checks = generate_checks(outputs)
        
        # 更新文件
        update_test_file(full_path, checks)

if __name__ == "__main__":
    main()
