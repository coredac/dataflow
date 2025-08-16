# For simple example
mlir-neura-opt --assign-accelerator --lower-arith-to-neura  test.mlir > test_fused.mlir
mlir-opt --allow-unregistered-dialect --view-op-graph test2.mlir 2> test.dot 
dot -Tpng test.dot -o test.png

# For more complicated example
mlir-opt --lower-affine --convert-scf-to-cf --convert-cf-to-llvm test2.mlir -o test2_llvm.mlir
mlir-neura-opt test2_llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-cast --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow > test2_neura.mlir 
mlir-opt --allow-unregistered-dialect --view-op-graph test2_neura.mlir 2> test2.dot 
dot -Tpng test2.dot -o test2.png