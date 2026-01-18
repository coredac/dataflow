add_mlir_library(MLIRTaskflowAnalysis
    HyperblockDependencyAnalysis.cpp
    # TaskDependencyAnalysis.cpp

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRMemRefDialect
    MLIRTaskflow
)
