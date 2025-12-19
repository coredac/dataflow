#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

using namespace mlir;

namespace {

static bool isInnermostLoop(affine::AffineForOp for_op) {
  bool has_nested_loops = false;
  for_op.getBody()->walk([&](affine::AffineForOp) { has_nested_loops = true; });
  return !has_nested_loops;
}

// Wraps an innermost affine for loop in a neura.kernel operation.
static LogicalResult wrapInnermostLoopAsKernel(affine::AffineForOp for_op,
                                               OpBuilder &builder,
                                               unsigned &kernel_id) {
  Location loc = for_op.getLoc();

  // Collects values that need to be captured by the kernel.
  llvm::SetVector<Value> captured_values;
  getUsedValuesDefinedAbove(for_op.getRegion(), captured_values);

  // Checks if the loop has output values.
  bool has_outputs = !for_op.getResults().empty();

  // Creates the neura.kernel operation.
  builder.setInsertionPoint(for_op);
  SmallVector<Value> inputs(captured_values.begin(), captured_values.end());
  SmallVector<Type> input_types;
  for (Value val : inputs) {
    input_types.push_back(val.getType());
  }

  neura::KernelOp kernel_op = builder.create<neura::KernelOp>(
      loc, /*output_types=*/for_op->getResultTypes(),
      /*inputs=*/inputs);

  // Sets kernel name.
  std::string kernel_name = "kernel_" + std::to_string(kernel_id++);
  kernel_op.setKernelNameAttr(builder.getStringAttr(kernel_name));

  // Creats the kernel body block with arguments for captured values.
  Block *kernel_body = new Block();
  kernel_op.getBody().push_back(kernel_body);

  // Replaces uses of the original loop's results with kernel results.
  if (has_outputs) {
    for (auto [orig_result, kernel_result] :
         llvm::zip(for_op->getResults(), kernel_op.getResults())) {
      orig_result.replaceAllUsesWith(kernel_result);
    }
  }

  // Moves the loop directly in to the kernel body.
  builder.setInsertionPointToStart(kernel_body);
  for_op->moveBefore(kernel_body, kernel_body->end());

  builder.setInsertionPointToEnd(kernel_body);
  // Adds yield operation with proper operands.
  if (has_outputs) {
    // If the loop has outputs, yield the loop results.
    SmallVector<Value> yield_operands(for_op.getResults());
    builder.create<neura::YieldOp>(loc, yield_operands);
  } else {
    // If the loop has no outputs, create an empty yield.
    builder.create<neura::YieldOp>(loc, ValueRange{});
  }

  return success();
}

struct WrapLoopInKernelPass
    : public PassWrapper<WrapLoopInKernelPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WrapLoopInKernelPass)

  StringRef getArgument() const override { return "wrap-loop-in-kernel"; }
  StringRef getDescription() const override {
    return "Wraps loops in Neura kernel operations.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<neura::NeuraDialect, affine::AffineDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func_op = getOperation();

    // Skips if function already has kerenls.
    bool has_kernels = false;
    func_op.walk([&](neura::KernelOp) { has_kernels = true; });
    if (has_kernels) {
      return;
    }

    // Skips main function.
    if (func_op.getName() == "main") {
      return;
    }

    // Collects all innermost affine for loops in the function.
    // TODO: Support more kernel wrapping strategies.
    SmallVector<affine::AffineForOp> innermost_loops;
    func_op.walk([&](affine::AffineForOp for_op) {
      if (isInnermostLoop(for_op)) {
        innermost_loops.push_back(for_op);
      }
    });

    if (innermost_loops.empty()) {
      return;
    }

    // Wraps each innermost affine for loop in a neura.kernel operation.
    // TODO: Support more kernel wrapping strategies.
    OpBuilder builder(func_op->getContext());
    unsigned kernel_id = 0;
    for (affine::AffineForOp loop : innermost_loops) {
      if (failed(wrapInnermostLoopAsKernel(loop, builder, kernel_id))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::neura::createWrapLoopInKernelPass() {
  return std::make_unique<WrapLoopInKernelPass>();
}