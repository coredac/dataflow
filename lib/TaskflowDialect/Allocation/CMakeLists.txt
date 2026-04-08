add_mlir_library(MLIRTaskflowAllocation
    allocation_utils.cpp
    RoutingCriticalPathAllocation.cpp

    DEPENDS
    MLIRTaskflowTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRFuncDialect
    MLIRTaskflow
    LLVMSupport
)
