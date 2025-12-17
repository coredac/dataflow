mlir-neura-opt --assign-accelerator --lower-arith-to-neura --fuse-pattern --view-op-graph test.mlir 2> test.dot 
dot -Tpng test.dot -o test.png