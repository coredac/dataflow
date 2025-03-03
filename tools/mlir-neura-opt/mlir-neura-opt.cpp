// tools/mlir-neura-opt/mlir-neura-opt.cpp

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "NeuraDialect/NeuraDialect.h"

int main(int argc, char **argv) {
  // Register MLIR dialects
  mlir::DialectRegistry registry;
  registry.insert<mlir::neura::NeuraDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  // mlir::registerAllDialects(registry);

  // Register MLIR passes
  // mlir::registerAllPasses();

  // Run the MLIR optimizer
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Neura Dialect Optimizer", registry));
}
