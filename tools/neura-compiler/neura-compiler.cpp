// neura-compiler.cpp

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Conversion/ConversionPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/Architecture/ArchitectureSpec.h"
#include "TaskflowDialect/TaskflowDialect.h"

// Global variable to store architecture spec file path
static std::string architecture_spec_file;
static mlir::neura::TileDefaults tile_defaults;

// Function to get the architecture spec file path
std::string mlir::neura::getArchitectureSpecFile() {
  return architecture_spec_file;
}

// Function to get tile defaults configuration
mlir::neura::TileDefaults mlir::neura::getTileDefaults() {
  return tile_defaults;
}

int main(int argc, char **argv) {
  // Manually scan and strip --architecture-spec from argv, keep others for
  // MlirOptMain.
  std::vector<char *> forwarded_args;
  forwarded_args.reserve(argc);
  forwarded_args.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg_ref(argv[i]);
    if (arg_ref == "--architecture-spec") {
      if (i + 1 < argc) {
        architecture_spec_file = argv[i + 1];
        ++i; // skip value
        continue;
      }
    } else if (arg_ref.starts_with("--architecture-spec=")) {
      architecture_spec_file =
          arg_ref.substr(strlen("--architecture-spec=")).str();
      continue;
    }
    forwarded_args.push_back(argv[i]);
  }

  int new_argc = static_cast<int>(forwarded_args.size());
  char **new_argv = forwarded_args.data();
  // Registers MLIR dialects.
  mlir::DialectRegistry registry;
  registry.insert<mlir::neura::NeuraDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::DLTIDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::memref::MemRefDialect>();

  mlir::neura::registerNeuraConversionPassPipeline();

  // Print architecture spec file info
  if (!architecture_spec_file.empty()) {
    llvm::errs() << "[neura-compiler] Architecture specification file: "
                 << architecture_spec_file << "\n";
  } else {
    llvm::errs() << "[neura-compiler] No architecture specification file "
                    "provided, using default configuration\n";
  }
  // Runs the MLIR optimizer.
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(new_argc, new_argv, "Neura Dialect Compiler", registry));
}