add_mlir_library(MLIRTaskflowUtil
    CgraPlacementUtils.cpp
    AllocateCgraTaskMapper.cpp

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
