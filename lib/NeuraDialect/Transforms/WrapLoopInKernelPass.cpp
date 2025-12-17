#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include <memory>

using namespace mlir;

namespace {

struct WrapLoopInKernelPass
    : public PassWrapper<WrapLoopInKernelPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WrapLoopInKernelPass)

  StringRef getArgument() const override { return "wrap-loop-in-kernel"; }
  StringRef getDescription() const override {
    return "Wraps loops in Neura kernel operations.";
  }

  void runOnOperation() override {}
};
} // namespace

std::unique_ptr<Pass> mlir::neura::createWrapLoopInKernelPass() {
  return std::make_unique<WrapLoopInKernelPass>();
}