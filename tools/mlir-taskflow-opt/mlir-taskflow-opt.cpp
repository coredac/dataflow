// tools/mlir-neura-opt/mlir-neura-opt.cpp

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"

#include "Conversion/ConversionPasses.h"
#include "TaskflowDialect/TaskflowDialect.h"

int main(int argc, char **argv) {
  // Registers MLIR dialects.
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::DLTIDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::ml_program::MLProgramDialect>();
  registry.insert<mlir::taskflow::TaskflowDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();

  // Registers our custom conversion passes.
  mlir::registerPasses();

  // Register all standard conversion passes
  mlir::registerConversionPasses();

  // Runs the MLIR optimizer.
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Taskflow Dialect Optimizer", registry));
}
