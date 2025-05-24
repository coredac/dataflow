add_mlir_dialect_library(MLIRNeura
        Neura.cpp

        DEPENDS
        MLIRNeuraIncGen
        
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        )

add_subdirectory(Transforms)
add_subdirectory(Conversion)