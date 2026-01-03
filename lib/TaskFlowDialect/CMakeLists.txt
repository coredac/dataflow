add_mlir_dialect_library(MLIRTaskFlow
        TaskFlow.cpp

        DEPENDS
        MLIRConversionIncGen
        
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRInferTypeOpInterface
        )