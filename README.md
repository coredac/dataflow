# Neura -- a dataflow dialect.

Build LLVM & Neura
--------------------------------------------------------
 - Clone llvm-project (we prefer llvm-19).
```sh
$ git clone --depth 1 --branch release/19.x https://github.com/llvm/llvm-project.git
```
 - Build LLVM:
```sh
 $ cd llvm-project
 $ mkdir build && cd build
 # May need install ccache and lld.
 $ cmake -G Ninja ../llvm \
     -DLLVM_ENABLE_PROJECTS="clang;mlir" \
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
     -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
     -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
     -DCMAKE_C_COMPILER_LAUNCHER=ccache \
     -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
 $ cmake --build . --target check-mlir
 $ ninja
 $ ninja check-clang check-mlir
 $ export PATH=/workspace/llvm-project/build/bin:$PATH
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

Docker-based Environment Setup
--------------------------------------------------------
**Option 1: Pull Pre-built Image**

You can directly pull and use the pre-built Docker image:
```sh
$ docker pull cgra/neura:v1
$ docker run --name myneura -it cgra/neura:v1
```

**Option 2: Build Image from Dockerfile**

Alternatively, you can build the Docker image yourself using the provided Dockerfile:
```sh
# In the root directory of this repo:
$ cd docker
$ docker build -t neura:v1 .
$ docker run --name myneura -it neura:v1
```

Both methods will provide a consistent environment with LLVM/MLIR(version 19) and Neura built and ready to use.
