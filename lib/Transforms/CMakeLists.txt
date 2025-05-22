add_mlir_library(NeuraTransforms
  AssignAcceleratorPass.cpp
  InsertMovPass.cpp
  FusePatternsPass.cpp

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRPass
    MLIRSupport
    MLIRTransformUtils
    NeuraDialect
)
