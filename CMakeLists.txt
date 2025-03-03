cmake_minimum_required(VERSION 3.13.4)

project(NeuraMLIRProject)

# Enable MLIR
set(LLVM_DIR /WORK_REPO/llvm-project/build/lib/cmake/llvm)
set(MLIR_DIR /WORK_REPO/llvm-project/build/lib/cmake/mlir)
set(MLIR_SOURCE_DIR /WORK_REPO/llvm-project/mlir)
set(MLIR_BINARY_DIR /WORK_REPO/llvm-project/build)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

# Add subdirectories for dialect
add_subdirectory(include)
add_subdirectory(lib)

# Add subdirectory for mlir-neura-opt
add_subdirectory(tools/mlir-neura-opt)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
link_directories(${MLIR_LIBRARY_DIRS})

add_definitions(${LLVM_DEFINITIONS})
add_definitions(${MLIR_DEFINITIONS})
