get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_subdirectory(ArithToNeura)
add_subdirectory(LlvmToNeura)
add_subdirectory(AffineToNeura)

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
add_library(MLIRConversion INTERFACE)

add_dependencies(MLIRConversion MLIRConversionIncGen)

target_link_libraries(MLIRConversion INTERFACE
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRNeura
  MLIRNeuraArithToNeuraPass
  MLIRNeuraLlvmToNeuraPass
  ${dialect_libs}
)