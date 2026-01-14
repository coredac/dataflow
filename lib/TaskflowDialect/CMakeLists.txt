add_mlir_dialect_library(MLIRTaskflow
        Taskflow.cpp
        TaskflowPasses.cpp
        TaskflowOps.cpp

        DEPENDS
        MLIRConversionIncGen
        MLIRTaskflowTransformsIncGen
        
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRInferTypeOpInterface
)

add_subdirectory(Transforms)