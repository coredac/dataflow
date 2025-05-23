get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(
    NeuraTransforms
    InsertMovPass.cpp
    FusePatternsPass.cpp
    AssignAcceleratorPass.cpp

    DEPENDS
    MLIRNeuraTransformsIncGen

    LINK_LIBS PUBLIC
      MLIRIR
      MLIRPass
      MLIRSupport
      MLIRTransforms
      MLIRNeura
      MLIRNeuraArithToNeuraPass
      MLIRNeuraLlvmToNeuraPass
      ${dialect_libs}
)
add_subdirectory(ArithToNeura)
add_subdirectory(LlvmToNeura)