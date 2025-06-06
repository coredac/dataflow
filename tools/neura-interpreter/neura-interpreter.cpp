#include "llvm/Support/Format.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/AsmState.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"

#include <unordered_map>
#include <iostream>

using namespace mlir;

// Add PredicatedData struct at the top
struct PredicatedData {
  float value;
  bool predicate;
};

int main(int argc, char **argv) {
  if (argc < 2) {
    llvm::errs() << "Usage: neura-interpreter <input.mlir>\n";
    return 1;
  }

  DialectRegistry registry;
  registry.insert<neura::NeuraDialect, func::FuncDialect, arith::ArithDialect>();

  MLIRContext context;
  context.appendDialectRegistry(registry);

  llvm::SourceMgr sourceMgr;
  auto fileOrErr = mlir::openInputFile(argv[1]);
  if (!fileOrErr) {
    llvm::errs() << "Error opening file\n";
    return 1;
  }

  sourceMgr.AddNewSourceBuffer(std::move(fileOrErr), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse MLIR input file\n";
    return 1;
  }

  // Change map to store PredicatedData instead of just float
  llvm::DenseMap<Value, PredicatedData> valueMap;

  for (auto func : module->getOps<func::FuncOp>()) {
    Block &block = func.getBody().front();

    for (Operation &op : block.getOperations()) {
      if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(op)) {
        auto attr = constOp.getValue();
        PredicatedData val{0.0f, true};  // arith constants always have true predicate
      
        if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
          val.value = floatAttr.getValueAsDouble();
        } else if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
          val.value = static_cast<float>(intAttr.getInt());
        } else {
          llvm::errs() << "Unsupported constant type in arith.constant\n";
          return 1;
        }
      
        valueMap[constOp.getResult()] = val;
      } else if (auto constOp = dyn_cast<neura::ConstantOp>(op)) {
        auto attr = constOp.getValue();
        PredicatedData val{0.0f, true};  // default to true
      
        // Handle value attribute
        if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
            val.value = floatAttr.getValueAsDouble();
        } else if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
            val.value = static_cast<float>(intAttr.getInt());
        } else {
            llvm::errs() << "Unsupported constant type in neura.constant\n";
            return 1;
        }

        // Try getting predicate attribute
        if (auto predAttr = constOp->getAttrOfType<BoolAttr>("predicate")) {
            val.predicate = predAttr.getValue();
        }
        
        valueMap[constOp.getResult()] = val;

      } else if (auto movOp = dyn_cast<neura::DataMovOp>(op)) {
        valueMap[movOp.getResult()] = valueMap[movOp.getOperand()];

      } else if (auto faddOp = dyn_cast<neura::FAddOp>(op)) {
        auto lhs = valueMap[faddOp.getLhs()];
        auto rhs = valueMap[faddOp.getRhs()];
        
        // Always perform addition, but combine predicates
        PredicatedData result;
        result.value = lhs.value + rhs.value;
        result.predicate = lhs.predicate && rhs.predicate;
        
        valueMap[faddOp.getResult()] = result;

      } else if (auto retOp = dyn_cast<func::ReturnOp>(op)) {
        auto result = valueMap[retOp.getOperand(0)];
        llvm::outs() << "[neura-interpreter] Output: " << llvm::format("%.6f", result.value);
        if (!result.predicate) {
          llvm::outs() << " (predicate=false)";
        }
        llvm::outs() << "\n";
      } else {
        llvm::errs() << "Unhandled op: ";
        op.print(llvm::errs());
        llvm::errs() << "\n";
        return 1;
      }
    }
  }

  return 0;
}
