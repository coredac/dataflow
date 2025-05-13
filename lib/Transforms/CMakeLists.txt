add_mlir_library(NeuraTransforms
  InsertMovPass.cpp

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRPass
    MLIRSupport
    MLIRTransformUtils
    NeuraDialect
)
