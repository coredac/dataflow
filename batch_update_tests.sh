#!/bin/bash
# 批量更新所有失败的e2e测试
# 使用: ./batch_update_tests.sh

set -e
cd /home/x/shiran/dataflow

TESTS=(
    "test/e2e/fir/fir_kernel.mlir"
    "test/e2e/histogram/histogram_kernel.mlir"
    "test/e2e/bicg/bicg_kernel.mlir"
    "test/e2e/relu/relu_kernel.mlir"
)

for TEST in "${TESTS[@]}"; do
    echo "=============================================="
    echo "Processing: $TEST"
    echo "=============================================="
    
    # 运行测试生成输出
    /home/x/shiran/miniconda3/bin/lit -v "$TEST" 2>&1 || true
    
    TEST_DIR=$(dirname "$TEST")
    TEST_NAME=$(basename "$TEST")
    
    # 显示实际输出位置
    MAPPING="$TEST_DIR/Output/${TEST_NAME}.tmp-mapping.mlir"
    YAML="$TEST_DIR/tmp-generated-instructions.yaml"
    ASM="$TEST_DIR/tmp-generated-instructions.asm"
    
    echo ""
    echo "=== OUTPUT FILES ==="
    echo "Mapping: $MAPPING"
    [ -f "$MAPPING" ] && echo "  [EXISTS]" || echo "  [NOT FOUND]"
    echo "YAML: $YAML"
    [ -f "$YAML" ] && echo "  [EXISTS]" || echo "  [NOT FOUND]"
    echo "ASM: $ASM"
    [ -f "$ASM" ] && echo "  [EXISTS]" || echo "  [NOT FOUND]"
    
    echo ""
    echo "=== MAPPING OUTPUT ==="
    if [ -f "$MAPPING" ]; then
        cat "$MAPPING"
    fi
    
    echo ""
    echo "=== ASM OUTPUT ==="
    if [ -f "$ASM" ]; then
        cat "$ASM"
    fi
    
    echo ""
    echo ""
done

echo "=============================================="
echo "All output files have been generated."
echo "You can now manually update each test file"
echo "based on the actual outputs shown above."
echo "=============================================="
