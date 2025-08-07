get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(
    MLIRNeuraTransforms
    InsertDataMovPass.cpp
    InsertCtrlMovPass.cpp
    FusePatternPass.cpp
    AssignAcceleratorPass.cpp
    TransformCtrlToDataFlowPass.cpp
    LeveragePredicatedValuePass.cpp
    MapToAcceleratorPass.cpp
    GenerateCodePass.cpp
    FuseControlFlowPass.cpp
    CanonicalizeLiveInPass.cpp
    CanonicalizeCastPass.cpp
    FoldConstantPass.cpp

    DEPENDS
    MLIRNeuraTransformsIncGen

    LINK_LIBS PUBLIC
      MLIRIR
      MLIRPass
      MLIRSupport
      MLIRTransforms
      MLIRNeura
      ${dialect_libs}
      LLVMSupport
)