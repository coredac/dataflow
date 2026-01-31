get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRTaskflowTransforms
    ConstructHyperblockFromTaskPass.cpp
    CanonicalizeTaskPass.cpp
    ClassifyCountersPass.cpp
    MapCTOnCGRAArrayPass.cpp

    DEPENDS
    MLIRTaskflowTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRTaskflow
    ${dialect_libs}
    LLVMSupport
)