// tools/mlir-neura-opt/mlir-neura-opt.cpp

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "Conversion/ArithToNeura/ArithToNeura.h"
#include "NeuraDialect/NeuraDialect.h"
#include "Transforms/InsertMovPass.h"

int main(int argc, char **argv) {
  // Registers MLIR dialects.
  mlir::DialectRegistry registry;
  registry.insert<mlir::neura::NeuraDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::neura::createLowerArithToNeuraPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::neura::createInsertMovPass();
  });

  // Runs the MLIR optimizer.
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Neura Dialect Optimizer", registry));
}
