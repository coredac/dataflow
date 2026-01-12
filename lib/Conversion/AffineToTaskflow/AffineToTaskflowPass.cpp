#include "Conversion/ConversionPasses.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {
//------------------------------------------------------------------------------
// Helper Functions.
//------------------------------------------------------------------------------

// Collects all top-level affine.for operations in a function.
static SmallVector<affine::AffineForOp>
collectTopLevelLooops(func::FuncOp func_op) {
  SmallVector<affine::AffineForOp> top_level_loops;
  for (Block &block : func_op.getBlocks()) {
    for (Operation &op : block) {
      if (auto for_op = dyn_cast<affine::AffineForOp>(op)) {
        top_level_loops.push_back(for_op);
      }
    }
  }

  return top_level_loops;
}

//------------------------------------------------------------------------------
// Main Conversion Process.
//------------------------------------------------------------------------------
// Converts a single function to TaskFlow operations.
static LogicalResult convertFuncToTaskflow(func::FuncOp func_op) {
  OpBuilder builder(func_op.getContext());

  // Step 1: Collects top-level loops for the taskflow graph.
  SmallVector<affine::AffineForOp> top_level_loops =
      collectTopLevelLooops(func_op);

  if (top_level_loops.empty()) {
    // No loops to convert.
    llvm::errs() << "No top-level affine.for loops found in function '"
                 << func_op.getName() << "'.\n";
    return success();
  }

  llvm::errs() << "\n===Converting function: " << func_op.getName() << "===\n";
  llvm::errs() << "Found " << top_level_loops.size()
               << " top-level affine.for loops to convert:\n";
  for (affine::AffineForOp for_op : top_level_loops) {
    llvm::errs() << for_op.getLoc() << "\n";
  }

  // Step 2: Collects graph inputs (function arguments and values defined
  // outside collected loops).
  // TODO: We need to further supporting collecting operations between loops
  // that define inputs to the loops.
  // Example:
  // %1 = affine.for %i ... {
  //   %a = ...
  // }
  // %b = arith.add %1, %c  <-- defined between loops
  // %2 = affine.for %j ... {
  //   ...
  //   uses %b
  // }
}

class ConvertAffineToTaskflowPass
    : public PassWrapper<ConvertAffineToTaskflowPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAffineToTaskflowPass)

  StringRef getArgument() const final { return "convert-affine-to-taskflow"; }

  StringRef getDescription() const final {
    return "Convert Affine operations to Taskflow operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TaskflowDialect, affine::AffineDialect, func::FuncDialect,
                    arith::ArithDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    WalkResult result = module.walk([](func::FuncOp func_op) {
      if (failed(convertFuncToTaskflow(func_op))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createConvertAffineToTaskflowPass() {
  return std::make_unique<ConvertAffineToTaskflowPass>();
}