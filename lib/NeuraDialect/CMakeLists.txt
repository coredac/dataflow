# Set include paths for TableGen
set(MLIR_TABLEGEN_INCLUDES
    "-I${PROJECT_SOURCE_DIR}/include"
    "-I${PROJECT_SOURCE_DIR}/include/NeuraDialect"
    "-I${CMAKE_CURRENT_BINARY_DIR}/include/NeuraDialect")

# Generate TableGen files
set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/include/NeuraDialect/Neura.td)
mlir_tablegen(Neura.h.inc -gen-op-decls ${MLIR_TABLEGEN_INCLUDES})
mlir_tablegen(Neura.cpp.inc -gen-op-defs ${MLIR_TABLEGEN_INCLUDES})
mlir_tablegen(NeuraDialect.h.inc -gen-dialect-decls ${MLIR_TABLEGEN_INCLUDES})
mlir_tablegen(NeuraDialect.cpp.inc -gen-dialect-defs ${MLIR_TABLEGEN_INCLUDES})
mlir_tablegen(NeuraTypes.h.inc -gen-typedef-decls ${MLIR_TABLEGEN_INCLUDES})
mlir_tablegen(NeuraTypes.cpp.inc -gen-typedef-defs ${MLIR_TABLEGEN_INCLUDES})
add_public_tablegen_target(MLIRNeuraDialectIncGen)

# Add the dialect library
add_mlir_dialect_library(MLIRNeura
        Neura.cpp
        NeuraTypes.cpp
        NeuraPasses.cpp
        Mapping/mapping_util.cpp
        Mapping/MappingState.cpp
        Architecture/Architecture.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/NeuraDialect

        DEPENDS
        MLIRNeuraDialectIncGen
        MLIRNeuraTransformsIncGen
        MLIRConversionIncGen
        
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRInferTypeOpInterface
        )

add_subdirectory(Transforms)