add_mlir_dialect_library(MLIRTaskflow
        Taskflow.cpp

        DEPENDS
        MLIRConversionIncGen
        
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRInferTypeOpInterface
        )