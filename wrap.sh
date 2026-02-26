#!/bin/bash
gdb -q -ex "b TaskflowToNeuraPass.cpp:274" -ex "r" -ex "bt" -ex "q" --args build/tools/mlir-neura-opt/mlir-neura-opt /tmp/resnet.stream.mlir --resource-aware-task-optimization --architecture-spec=test/arch_spec/architecture.yaml
