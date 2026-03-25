#!/usr/bin/env python3
"""
Regenerates CHECK lines for failing tests from actual output files.

Usage:
  python3 update_test_checks.py <test_file> <prefix> <output_file> [--max-lines N]
  python3 update_test_checks.py <test_file> <prefix> <output_file> --asm

For YAML/MAPPING: generates // PREFIX: first_line then // PREFIX-NEXT: rest
For ASM (--asm): groups by PE blocks, each PE starts with // PREFIX: then -NEXT
"""

import sys
import os
import re


def replace_check_lines(test_file, prefix, new_lines):
    """Replace all CHECK lines with the given prefix in the test file."""
    with open(test_file, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    result = []
    inserted = False

    # Match lines like: // PREFIX:, // PREFIX-NEXT:, // PREFIX-NOT:, etc.
    prefix_pattern = re.compile(
        rf'^// {re.escape(prefix)}(-NEXT|-NOT|-DAG|-LABEL)?:\s*')

    i = 0
    while i < len(lines):
        if prefix_pattern.match(lines[i]):
            if not inserted:
                # Insert new check lines here
                for new_line in new_lines:
                    result.append(new_line)
                inserted = True
            # Skip old check line
            i += 1
            continue
        result.append(lines[i])
        i += 1

    with open(test_file, 'w') as f:
        f.write('\n'.join(result))


def generate_checks(output_text, prefix, max_lines=None, skip_lines=0):
    """Generate CHECK lines from output. First non-empty line uses PREFIX:,
    rest use PREFIX-NEXT:. Empty lines are skipped.
    skip_lines: number of non-empty lines to skip from the start."""
    lines = output_text.strip().split('\n')
    check_lines = []
    first = True
    skipped = 0
    for line in lines:
        if not line.strip():
            continue
        if skipped < skip_lines:
            skipped += 1
            continue
        if max_lines and len(check_lines) >= max_lines:
            break
        if first:
            check_lines.append(f'// {prefix}: {line.rstrip()}')
            first = False
        else:
            check_lines.append(f'// {prefix}-NEXT: {line.rstrip()}')
    return check_lines


def generate_asm_checks(output_text, prefix):
    """Generate ASM CHECK lines. Each PE(...) block starts with // PREFIX:,
    followed by // PREFIX-NEXT: lines. Empty lines are skipped.
    The first line (e.g. '# Compiled II: N') also starts a block."""
    lines = output_text.strip().split('\n')
    check_lines = []
    # First non-empty line starts with PREFIX:
    # PE(...): lines start new blocks with PREFIX:
    pe_pattern = re.compile(r'^PE\(')
    first_line = True

    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            continue
        if first_line:
            check_lines.append(f'// {prefix}: {stripped}')
            first_line = False
        elif pe_pattern.match(stripped):
            check_lines.append(f'// {prefix}: {stripped}')
        else:
            check_lines.append(f'// {prefix}-NEXT: {stripped}')
    return check_lines


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('test_file')
    parser.add_argument('prefix')
    parser.add_argument('output_file')
    parser.add_argument('--max-lines', type=int, default=None)
    parser.add_argument('--skip-lines', type=int, default=0)
    parser.add_argument('--asm', action='store_true')
    args = parser.parse_args()

    with open(args.output_file, 'r') as f:
        actual_output = f.read()

    if args.asm:
        new_checks = generate_asm_checks(actual_output, args.prefix)
    else:
        new_checks = generate_checks(actual_output, args.prefix,
                                     args.max_lines, args.skip_lines)

    replace_check_lines(args.test_file, args.prefix, new_checks)
    print(f"Updated {len(new_checks)} CHECK lines for prefix "
          f"{args.prefix} in {args.test_file}")
