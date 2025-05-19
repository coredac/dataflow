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

  llvm::DenseMap<Value, float> valueMap;

  for (auto func : module->getOps<func::FuncOp>()) {
    Block &block = func.getBody().front();

    for (Operation &op : block.getOperations()) {
      if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(op)) {
        auto attr = constOp.getValue();
      
        float val = 0.0f;
      
        if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
          val = floatAttr.getValueAsDouble();  // or .convertToFloat()
        } else if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
          val = static_cast<float>(intAttr.getInt()); // interpret integer as float
        } else {
          llvm::errs() << "Unsupported constant type in arith.constant\n";
          return 1;
        }
      
        valueMap[constOp.getResult()] = val;
      } else if (auto movOp = dyn_cast<neura::MovOp>(op)) {
        valueMap[movOp.getResult()] = valueMap[movOp.getOperand()];
      } else if (auto faddOp = dyn_cast<neura::FAddOp>(op)) {
        float lhs = valueMap[faddOp.getLhs()];
        float rhs = valueMap[faddOp.getRhs()];
        valueMap[faddOp.getResult()] = lhs + rhs;
      } else if (auto retOp = dyn_cast<func::ReturnOp>(op)) {
        float result = valueMap[retOp.getOperand(0)];
        llvm::outs() << "[neura-interpreter] Output: " << llvm::format("%.6f", result) << "\n";
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
