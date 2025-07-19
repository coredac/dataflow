#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;

#define GEN_PASS_DEF_FUSECONTROLFLOW
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
// A class to hold loop information for the control flow fusion pass.
class LoopInfo {
public:
  // Key operations in a loop.
  Value reserve_val;
  Value phi_val;
  Value index_val;
  Value condition_val;
  Value not_condition_val;

  // Loop iteration parameters.
  Value start_val;
  Value end_val;
  Value step_val;

  // Backward edge information.
  Operation *ctrl_mov = nullptr; // Initialized to nullptr.

  // Used for replace and update operations.
  llvm::SetVector<Operation *> ops_to_remove;
  llvm::MapVector<Value, SmallVector<std::pair<Operation *, unsigned>>>
      users_to_update;

  // Adds operations to remove.
  void addOpToRemove(Operation *op) {
    if (op) {
      ops_to_remove.insert(op);
    }
  }

  // Checks if the loop info is complete.
  // There is no not_condition_val because it is derived from condition_val.
  bool isComplete() const {
    return reserve_val && phi_val && index_val && condition_val && start_val &&
           end_val && step_val && ctrl_mov;
  }

  // Records the users that use the loop index and (not-)condition values.
  void recordUsersToUpdate() {
    // TODO: Implements the logic to record users of loop index and condition
    // values.
  }
};

struct FuseControlFlowPass
    : public PassWrapper<FuseControlFlowPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseControlFlowPass)

  StringRef getArgument() const override { return "fuse-control-flow"; }
  StringRef getDescription() const override {
    return "Fuses control flow operations into optimized neura dialect "
           "operations";
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    // TODO: Adds the logic to fuse determined control flow operations.
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createFuseControlFlowPass() {
  return std::make_unique<FuseControlFlowPass>();
}
} // namespace mlir::neura