mlir-neura-opt --assign-accelerator --lower-arith-to-neura --fuse-patterns test.mlir > test_fused.mlir
mlir-opt --allow-unregistered-dialect --view-op-graph test_fused.mlir 2> test.dot 
dot -Tpng test.dot -o test.png