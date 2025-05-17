# set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/FusePatterns.td)
#
# mlir_tablegen(FusePatterns.inc -gen-rewriters
#   -I ${MLIR_SOURCE_DIR}/include
#   -I ${MLIR_BINARY_DIR}/include
#   -I ${CMAKE_SOURCE_DIR}/include
#   -I ${CMAKE_CURRENT_SOURCE_DIR}
# )
#
# add_public_tablegen_target(NeuraFusePatternsGen)

add_mlir_library(NeuraTransforms
  InsertMovPass.cpp
  FusePatternsPass.cpp

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRPass
    MLIRSupport
    MLIRTransformUtils
    NeuraDialect
)
