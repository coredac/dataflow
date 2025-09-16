get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_subdirectory(HWAgnosticOpt)
add_subdirectory(HWSpecificOpt)

add_library(MLIRNUERAOptimization INTERFACE)

add_dependencies(MLIRNUERAOptimization MLIRNeuraTransformsIncGen)

target_link_libraries(MLIRNUERAOptimization INTERFACE
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRNeura
  MLIRNeuraHWSpecificOpt
  MLIRNeuraHWAgnosticOpt
  ${dialect_libs}
)