cmake_minimum_required(VERSION 3.13.4)

project(NeuraMLIRProject)

# Enable MLIR
# set(LLVM_DIR /workspace/llvm-project/build/lib/cmake/llvm)
# set(MLIR_DIR /workspace/llvm-project/build/lib/cmake/mlir)
# set(MLIR_SOURCE_DIR /workspace/llvm-project/mlir)
# set(MLIR_BINARY_DIR /workspace/llvm-project/build)

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

# ===--------------------------------------------------------------------===#
# LIT test setup
# ===--------------------------------------------------------------------===#

# Tools built by this project or LLVM
set(MLIR_NEURA_OPT ${CMAKE_BINARY_DIR}/tools/mlir-neura-opt/mlir-neura-opt)
set(FILECHECK ${LLVM_TOOLS_BINARY_DIR}/FileCheck)
set(MLIR_OPT ${LLVM_TOOLS_BINARY_DIR}/mlir-opt)
set(MLIR_TRANSLATE ${LLVM_TOOLS_BINARY_DIR}/mlir-translate)
set(LLC ${LLVM_TOOLS_BINARY_DIR}/llc)

# Configure lit.cfg from lit.cfg.in
configure_file(
  ${CMAKE_SOURCE_DIR}/test/lit.cfg.in
  #  ${CMAKE_BINARY_DIR}/test/lit.cfg
  ${CMAKE_SOURCE_DIR}/test/lit.cfg
  @ONLY
)

# Add a custom target for running lit tests
add_custom_target(check-neura
  COMMAND ${LLVM_EXTERNAL_LIT} -sv ${CMAKE_BINARY_DIR}/test
  DEPENDS mlir-neura-opt
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running Sora Dialect Tests with lit"
)
