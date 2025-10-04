// tools/mlir-neura-opt/mlir-neura-opt.cpp

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"

#include "Conversion/ConversionPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/Architecture/ArchitectureSpec.h"

// Global variable to store architecture spec file path
static std::string architectureSpecFile;
static mlir::neura::TileDefaults tileDefaults;

// Function to get the architecture spec file path
std::string mlir::neura::getArchitectureSpecFile() {
  return architectureSpecFile;
}

// Function to get tile defaults configuration
mlir::neura::TileDefaults mlir::neura::getTileDefaults() {
  return tileDefaults;
}

int main(int argc, char **argv) {
  // Manually scan and strip --architecture-spec from argv, keep others for MlirOptMain.
  std::vector<char *> forwardedArgs;
  forwardedArgs.reserve(argc);
  forwardedArgs.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef argRef(argv[i]);
    if (argRef == "--architecture-spec") {
      if (i + 1 < argc) {
        architectureSpecFile = argv[i + 1];
        ++i; // skip value
        continue;
      }
    } else if (argRef.starts_with("--architecture-spec=")) {
      architectureSpecFile = argRef.substr(strlen("--architecture-spec=")).str();
      continue;
    }
    forwardedArgs.push_back(argv[i]);
  }


  int newArgc = static_cast<int>(forwardedArgs.size());
  char **newArgv = forwardedArgs.data();

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
  mlir::registerViewOpGraphPass();

  // Print architecture spec file info
  if (!architectureSpecFile.empty()) {
    llvm::errs() << "[mlir-neura-opt] Architecture specification file: " 
                 << architectureSpecFile << "\n";
  } else {
    llvm::errs() << "[mlir-neura-opt] No architecture specification file provided, using default configuration\n";
  }

  // Runs the MLIR optimizer.
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(newArgc, newArgv, "Neura Dialect Optimizer", registry));
}
