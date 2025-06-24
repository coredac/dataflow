# Add the dialect library
add_mlir_dialect_library(MLIRNeura
        Neura.cpp
        NeuraTypes.cpp
        NeuraPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/NeuraDialect

        DEPENDS
        MLIRNeuraTransformsIncGen
        MLIRConversionIncGen
        
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRInferTypeOpInterface
        )

add_subdirectory(Transforms)