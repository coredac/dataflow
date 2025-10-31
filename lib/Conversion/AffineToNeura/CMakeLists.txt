add_mlir_dialect_library(MLIRNeuraAffineToNeuraPass
  AffineToNeuraPass.cpp
  LoopNestAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/Conversion

  DEPENDS
  MLIRConversionIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRNeura
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
)
