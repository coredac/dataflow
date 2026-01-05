add_executable(mlir-taskflow-opt mlir-taskflow-opt.cpp)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRConversion
        MLIRTaskflow
        MLIRTransforms
        MLIROptLib
        MLIRPass
        MLIRIR
        MLIRParser
        MLIRSupport
        )

target_link_libraries(mlir-taskflow-opt PRIVATE ${LIBS})