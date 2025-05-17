add_executable(neura-interpreter neura-interpreter.cpp)

target_include_directories(neura-interpreter
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(neura-interpreter
  PRIVATE
    MLIRParser
    MLIRIR
    MLIRArithDialect
    MLIRSupport
    MLIRFuncDialect
    MLIRDialect       # Optional: for builtin
    NeuraDialect      # Your dialect
)

mlir_check_all_link_libraries(neura-interpreter)

