get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_subdirectory(HwAgnosticOpt)
add_subdirectory(HwSpecificOpt)

add_library(MLIRNUERAOptimization INTERFACE)

add_dependencies(MLIRNUERAOptimization MLIRNeuraTransformsIncGen)

target_link_libraries(MLIRNUERAOptimization INTERFACE
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRNeura
  MLIRNeuraHwSpecificOpt
  MLIRNeuraHwAgnosticOpt
  ${dialect_libs}
)