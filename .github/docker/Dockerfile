FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
	&& apt install software-properties-common -y \
	&& add-apt-repository ppa:deadsnakes/ppa -y \
        && apt update \
        && apt install python3.12 -y \
	&& apt-get install ninja-build -y \
	&& apt-get install ccache -y \
	&& apt-get install lld -y \
	&& apt-get -y install cmake -y \
	&& apt-get install -y clang \
	&& apt-get -y install git \
	&& mkdir /cgra \
    	&& cd /cgra \
	&& git clone https://github.com/yuqisun/dataflow.git \
	&& cd dataflow \
	&& until git clone --depth 1 --branch release/19.x https://github.com/llvm/llvm-project.git; do sleep 5; done \
	&& cd llvm-project \
	&& mkdir build && cd build \
	&& cmake -G Ninja ../llvm \
          -DLLVM_ENABLE_PROJECTS="mlir" \
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
          -DCMAKE_C_COMPILER_LAUNCHER=ccache \
          -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
	&& cmake --build . --target check-mlir \
	&& dpkg -l | grep llvm \
	&& cd /cgra/dataflow \
	&& mkdir build && cd build \
	&& cmake -G Ninja .. \
          -DLLVM_DIR=/cgra/dataflow/llvm-project/build/lib/cmake/llvm \
          -DMLIR_DIR=/cgra/dataflow/llvm-project/build/lib/cmake/mlir \
          -DMLIR_SOURCE_DIR=/cgra/dataflow/llvm-project/mlir \
          -DMLIR_BINARY_DIR=/cgra/dataflow/llvm-project/build \
          -DCMAKE_CXX_FLAGS="-std=c++17" \
	&& ninja \
	&& cd /cgra/dataflow/test \
	&& /cgra/dataflow/llvm-project/build/bin/llvm-lit * -v

CMD ["tail", "-f", "/dev/null"]


