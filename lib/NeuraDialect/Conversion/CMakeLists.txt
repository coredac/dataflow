get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_subdirectory(ArithToNeura)
add_subdirectory(LlvmToNeura)

# add_mlir_library(
#     MLIRNeuraConversion

#     DEPENDS
#     MLIRNeuraTransformsIncGen

#     LINK_LIBS PUBLIC
#       MLIRIR
#       MLIRPass
#       MLIRSupport
#       MLIRTransforms
#       MLIRNeura
#       MLIRNeuraArithToNeuraPass
#       MLIRNeuraLlvmToNeuraPass
#       ${dialect_libs}
# )
add_library(MLIRNeuraConversion INTERFACE)

add_dependencies(MLIRNeuraConversion MLIRNeuraTransformsIncGen)

target_link_libraries(MLIRNeuraConversion INTERFACE
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRNeura
  MLIRNeuraArithToNeuraPass
  MLIRNeuraLlvmToNeuraPass
  ${dialect_libs}
)