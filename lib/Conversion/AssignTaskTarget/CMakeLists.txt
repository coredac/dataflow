include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_conversion_library(MLIRAssignTaskTargetPass
  AssignTaskTargetPass.cpp

  DEPENDS
  MLIRConversionIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)
