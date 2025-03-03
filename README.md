# Neura -- a dataflow dialect.

Build LLVM & Neura
--------------------------------------------------------
 - Clone llvm-project.

 - Clone this repo.

 - Build LLVM:
   - Check out to commit `cd70802`.
   - Build:
```
 $ mkdir build && cd build
 $ cmake -G Ninja ../llvm    -DLLVM_ENABLE_PROJECTS="mlir"    -DLLVM_BUILD_EXAMPLES=OFF    -DLLVM_TARGETS_TO_BUILD="Native"    -DCMAKE_BUILD_TYPE=Release    -DLLVM_ENABLE_ASSERTIONS=ON    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_FLAGS="-std=c++17 -frtti" -DLLVM_ENABLE_LLD=ON -DMLIR_INSTALL_AGGREGATE_OBJECTS=ON -DLLVM_ENABLE_RTTI=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
 $ cmake --build . --target check-mlir
```

 - Build Neura:
   - Step into this repo, note that this repo is outside of llvm-project.
   - Build:
```
 $ mkdir build && cd build
 $ cmake -G Ninja ..   -DLLVM_DIR=/WORK_REPO/llvm-project/build/lib/cmake/llvm   -DMLIR_DIR=/WORK_REPO/llvm-project/build/lib/cmake/mlir -DCMAKE_CXX_FLAGS="-std=c++17"
 $ ninja
```

 - Test:
```
 $ cd ../test
 $ ../build/tools/mlir-neura-opt/mlir-neura-opt --debug test.mlir
```

