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
    CanonicalizeLiveInPass.cpp
    CanonicalizeCastPass.cpp
    PromoteFuncArgToConstPass.cpp
    IterMergePatternPass.cpp
    TransformToSteerControlPass.cpp
    RemovePredicatedTypePass.cpp
    WrapLoopInKernelPass.cpp

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

add_subdirectory(Optimizations)