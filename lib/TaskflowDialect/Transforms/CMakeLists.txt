get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRTaskflowTransforms
    ConstructHyperblockFromTaskPass.cpp
    CanonicalizeTaskPass.cpp
<<<<<<< HEAD
    ClassifyCountersPass.cpp
=======
    PlaceACTOnCGRAPass.cpp
>>>>>>> a0f7fc7 (feat(taskflow): implement graph-based ACT placement and memory management framework)

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