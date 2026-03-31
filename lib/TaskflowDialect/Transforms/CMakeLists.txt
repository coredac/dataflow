get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRTaskflowTransforms
    ConstructHyperblockFromTaskPass.cpp
    ClassifyCountersPass.cpp
    MapTaskOnCgraPass.cpp
    FuseTaskPass.cpp
    TaskDivisibilityAnalysisPass.cpp

    DEPENDS
    MLIRTaskflowTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRTaskflow
    MLIRNeura
    MLIRNeuraTransforms
    MLIRConversion
    ${dialect_libs}
    LLVMSupport
)