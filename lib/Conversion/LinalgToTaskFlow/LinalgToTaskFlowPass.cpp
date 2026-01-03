#include "Conversion/ConversionPasses.h"
#include "TaskFlowDialect/TaskFlowDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::taskflow;
namespace {

// Converts a single function to TaskFlow operations.
static LogicalResult convertFuncToTaskflow(func::FuncOp func_op) {}

class ConvertLinalgToTaskFlowPass
    : public PassWrapper<ConvertLinalgToTaskFlowPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToTaskFlowPass)

  StringRef getArgument() const final { return "convert-linalg-to-taskflow"; }

  StringRef getDescription() const final {
    return "Convert Linalg operations to TaskFlow operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TaskFlowDialect, linalg::LinalgDialect, func::FuncDialect,
                    arith::ArithDialect, tensor::TensorDialect>();
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

std::unique_ptr<Pass> mlir::createConvertLinalgToTaskFlowPass() {
  return std::make_unique<ConvertLinalgToTaskFlowPass>();
}