# Neura -- a dataflow dialect.

Build LLVM & Neura
--------------------------------------------------------
 - Clone llvm-project.

 - Clone this repo.

 - Build LLVM:
   - Check out to commit `6146a88` (a stable version randomly picked, will sync to the latest version).
   - Build:
```sh
 $ mkdir build && cd build
 # May need install ccache and lld.
 $ cmake -G Ninja ../llvm \
     -DLLVM_ENABLE_PROJECTS="mlir;clang" \
     -DLLVM_BUILD_EXAMPLES=OFF \
     -DLLVM_TARGETS_TO_BUILD="Native" \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_ASSERTIONS=ON \
     -DCMAKE_C_COMPILER=clang \
     -DCMAKE_CXX_COMPILER=clang++ \
     -DCMAKE_CXX_STANDARD=17 \
     -DCMAKE_CXX_FLAGS="-std=c++17 -frtti" \
     -DLLVM_ENABLE_LLD=ON \
     -DMLIR_INSTALL_AGGREGATE_OBJECTS=ON \
     -DLLVM_ENABLE_RTTI=ON \
     -DCMAKE_C_COMPILER_LAUNCHER=ccache \
     -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
 $ cmake --build . --target check-mlir
 $ cmake --build . --target check-clang
```

 - Build Neura:
   - Step into this repo, note that this repo is outside of llvm-project.
   - Build:
```sh
 $ mkdir build && cd build
 # Replace the path "/workspace" accordingly.
 $ cmake -G Ninja .. \
     -DLLVM_DIR=/workspace/llvm-project/build/lib/cmake/llvm \
     -DMLIR_DIR=/workspace/llvm-project/build/lib/cmake/mlir \
     -DMLIR_SOURCE_DIR=/workspace/llvm-project/mlir \
     -DMLIR_BINARY_DIR=/workspace/llvm-project/build \
     -DCMAKE_CXX_FLAGS="-std=c++17"
 $ ninja
```

 - Test:
```sh
 $ cd ../test
 $ ../build/tools/mlir-neura-opt/mlir-neura-opt --debug test.mlir

 # Or test with lit:
 $ /workspace/llvm-project/build/bin/llvm-lit *
```

