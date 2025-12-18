#!/bin/bash
# 自动更新测试文件的期望输出
# 使用方法: ./update_test_expects.sh <test_mlir_file>
# 例如: ./update_test_expects.sh test/e2e/fir/fir_kernel.mlir

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <test_mlir_file>"
    echo "Example: $0 test/e2e/fir/fir_kernel.mlir"
    exit 1
fi

TEST_FILE="$1"
TEST_DIR=$(dirname "$TEST_FILE")
TEST_NAME=$(basename "$TEST_FILE")

cd /home/x/shiran/dataflow

echo "=== Updating test: $TEST_FILE ==="

echo "Running test to generate output files..."
/home/x/shiran/miniconda3/bin/lit -v "$TEST_FILE" 2>&1 || true

MAPPING_OUTPUT="$TEST_DIR/Output/${TEST_NAME}.tmp-mapping.mlir"
YAML_OUTPUT="$TEST_DIR/tmp-generated-instructions.yaml"
ASM_OUTPUT="$TEST_DIR/tmp-generated-instructions.asm"

echo ""
echo "=== Generated Output Files ==="

if [ -f "$MAPPING_OUTPUT" ]; then
    echo "Mapping output: $MAPPING_OUTPUT"
    echo "--- Content ---"
    cat "$MAPPING_OUTPUT"
    echo ""
else
    echo "WARNING: No mapping output found at $MAPPING_OUTPUT"
fi

if [ -f "$YAML_OUTPUT" ]; then
    echo ""
    echo "YAML output: $YAML_OUTPUT"
    echo "--- Content (first 50 lines) ---"
    head -50 "$YAML_OUTPUT"
    echo ""
else
    echo "WARNING: No YAML output found at $YAML_OUTPUT"
fi

if [ -f "$ASM_OUTPUT" ]; then
    echo ""
    echo "ASM output: $ASM_OUTPUT"
    echo "--- Content ---"
    cat "$ASM_OUTPUT"
    echo ""
else
    echo "WARNING: No ASM output found at $ASM_OUTPUT"
fi

echo ""
echo "=== Instructions ==="
echo "1. Review the output above"
echo "2. Update the CHECK lines in $TEST_FILE to match the new output"
echo "3. Run: /home/x/shiran/miniconda3/bin/lit -v $TEST_FILE"
echo ""
echo "=== Quick Reference for CHECK patterns ==="
echo "MAPPING-NEXT: lines should match the mapping output"
echo "YAML-NEXT: lines should match the YAML output"
echo "ASM-NEXT: lines should match the ASM output"
