add_executable(mlir-neura-opt mlir-neura-opt.cpp)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRNeuraTransforms
        MLIRConversion
        MLIRNeura
        MLIRTransforms
        MLIROptLib
        MLIRPass
        MLIRIR MLIRParser MLIRSupport
        )

target_link_libraries(mlir-neura-opt PRIVATE ${LIBS})