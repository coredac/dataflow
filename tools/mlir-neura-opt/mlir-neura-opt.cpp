// tools/mlir-neura-opt/mlir-neura-opt.cpp

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraPasses.h"
#include "Conversion/ConversionPasses.h"

int main(int argc, char **argv) {
  // Registers MLIR dialects.
  mlir::DialectRegistry registry;
  registry.insert<mlir::neura::NeuraDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::DLTIDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();

  mlir::neura::registerPasses();
  mlir::registerPasses();
  
  // Runs the MLIR optimizer.
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Neura Dialect Optimizer", registry));
}
