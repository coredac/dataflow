# Add the dialect library
add_mlir_dialect_library(MLIRNeura
        Neura.cpp
        NeuraTypes.cpp
        NeuraPasses.cpp
        Mapping/mapping_util.cpp
        Mapping/MappingState.cpp
        Mapping/MappingStrategy.cpp
        Mapping/SpatialTemporalMapping/SpatialTemporalMapping.cpp
        Architecture/Architecture.cpp

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