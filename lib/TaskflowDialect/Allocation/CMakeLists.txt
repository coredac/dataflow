add_mlir_library(MLIRTaskflowAllocation
    allocation_utils.cpp
    allocation_utils_mapper.cpp

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
