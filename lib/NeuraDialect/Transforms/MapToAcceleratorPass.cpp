#include <deque>

#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_DEF_MapToAccelerator
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

/// Represents a recurrence cycle rooted at a reserve operation and ending at a ctrl_mov.
/// The cycle consists of a sequence of operations and its corresponding length.
struct RecurrenceCycle {
  SmallVector<Operation *> operations;  // Ordered list of operations in the cycle.
  int length = 0;                       // Number of operations excluding ctrl_mov and reserve_op.
};

// Traverses (backward) the operation graph starting from the given operation
// towards reserve_value.
void traverseAlongPath(Operation *op, Value reserve_value,
                       std::deque<Operation *> &current_path,
                       DenseSet<Operation *> &visited_in_path,
                       SmallVector<RecurrenceCycle, 4> &collected_paths) {
  if (!op || visited_in_path.contains(op))
    return;

  visited_in_path.insert(op);
  current_path.push_front(op);

  for (Value operand : op->getOperands()) {
    if (operand == reserve_value) {
      Operation *res_op = reserve_value.getDefiningOp();
      if (res_op) current_path.push_front(res_op);

      constexpr int kNumExcludedOps = 2;
      collected_paths.push_back(RecurrenceCycle{
        operations: SmallVector<Operation *>(current_path.begin(), current_path.end()),
        length: static_cast<int>(current_path.size()) - kNumExcludedOps
      });

      if (res_op) current_path.pop_front(); // Remove reserve before backtracking
      continue;
    }

    if (Operation *def_op = operand.getDefiningOp()) {
      traverseAlongPath(def_op, reserve_value, current_path, visited_in_path, collected_paths);
    }
  }

  current_path.pop_front();         // Backtrack
  visited_in_path.erase(op);        // Unmark from path
}

/// Collects all recurrence cycles rooted at reserve operations and closed by ctrl_mov.
/// Each cycle contains the operation sequence and its corresponding length.
SmallVector<RecurrenceCycle, 4> collectRecurrenceCycles(Operation *root_op) {
  SmallVector<RecurrenceCycle, 4> recurrence_cycles;

  root_op->walk([&](neura::CtrlMovOp ctrl_mov_op) {
    Value target = ctrl_mov_op.getTarget();
    auto reserve_op = target.getDefiningOp<neura::ReserveOp>();
    if (!reserve_op)
      return;

    Value reserve_value = reserve_op.getResult();
    Value ctrl_mov_from = ctrl_mov_op.getValue();

    Operation *parent_op = ctrl_mov_from.getDefiningOp();
    if (!parent_op)
      return;

    std::deque<Operation *> current_path;
    SmallVector<RecurrenceCycle, 4> collected_paths;
    DenseSet<Operation *> visited_in_path;
    traverseAlongPath(parent_op, reserve_value, current_path, visited_in_path, collected_paths);

    for (auto &cycle : collected_paths) {
      cycle.operations.push_back(ctrl_mov_op);
      ++cycle.length;
      recurrence_cycles.push_back(std::move(cycle));
    }
  });

  return recurrence_cycles;
}

struct MapToAcceleratorPass
    : public PassWrapper<MapToAcceleratorPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapToAcceleratorPass)

  StringRef getArgument() const override { return "map-to-accelerator"; }
  StringRef getDescription() const override {
    return "Maps IR to the target accelerator.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp func) {
      // Skips functions not targeting the neura accelerator.
      auto accel_attr = func->getAttrOfType<StringAttr>("accelerator");
      if (!accel_attr || accel_attr.getValue() != "neura")
        return;

      // Collects and reports recurrence cycles found in the function.
      auto recurrence_cycles = collectRecurrenceCycles(func);
      RecurrenceCycle *longest = nullptr;
      for (auto &cycle : recurrence_cycles) {
        if (!longest || cycle.length > longest->length)
          longest = &cycle;
      }

      if (longest) {
        llvm::errs() << "[MapToAcceleratorPass] Longest recurrence cycle (length "
                    << longest->length << "):\n";
        for (Operation *op : longest->operations)
          op->print(llvm::errs()), llvm::errs() << "\n";
        IntegerAttr mii_attr = IntegerAttr::get(
            IntegerType::get(func.getContext(), 32), longest->length);
        func->setAttr("RecMII", mii_attr);
      }
    });
  }
};

} // namespace

namespace mlir::neura {

std::unique_ptr<Pass> createMapToAcceleratorPass() {
  return std::make_unique<MapToAcceleratorPass>();
}

} // namespace mlir::neura
