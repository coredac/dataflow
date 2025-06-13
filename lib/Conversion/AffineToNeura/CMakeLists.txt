include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_conversion_library(MLIRNeuraAffineToNeuraPass
  AffineToNeuraPass.cpp

  DEPENDS
  MLIRConversionIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    # MLIRNeura
)