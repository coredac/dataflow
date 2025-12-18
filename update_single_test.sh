#!/bin/bash
# 自动更新测试文件期望输出 (完整版本)
# 用法: ./update_single_test.sh <test_file>

set -e
cd /home/x/shiran/dataflow

TEST_FILE="$1"
TEST_DIR=$(dirname "$TEST_FILE")
TEST_NAME=$(basename "$TEST_FILE")

echo "=== Processing: $TEST_FILE ==="

# 1. 运行测试生成输出
echo "Running test to generate output..."
/home/x/shiran/miniconda3/bin/lit -v "$TEST_FILE" 2>&1 || true

# 输出文件位置
MAPPING_OUT="$TEST_DIR/Output/${TEST_NAME}.tmp-mapping.mlir"
YAML_OUT="$TEST_DIR/tmp-generated-instructions.yaml"
ASM_OUT="$TEST_DIR/tmp-generated-instructions.asm"

echo ""
echo "=== MAPPING OUTPUT ($MAPPING_OUT) ==="
if [ -f "$MAPPING_OUT" ]; then
    cat "$MAPPING_OUT"
else
    echo "[NOT FOUND]"
fi

echo ""
echo "=== YAML OUTPUT ($YAML_OUT) ==="
if [ -f "$YAML_OUT" ]; then
    cat "$YAML_OUT"
else
    echo "[NOT FOUND]"
fi

echo ""
echo "=== ASM OUTPUT ($ASM_OUT) ==="
if [ -f "$ASM_OUT" ]; then
    cat "$ASM_OUT"
else
    echo "[NOT FOUND]"
fi

echo ""
echo "=== 完成 ==="
echo "请根据上面的输出手动更新测试文件中的 MAPPING-NEXT, YAML, ASM 部分"
