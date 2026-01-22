#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

using namespace mlir;
using namespace mlir::taskflow;

namespace {
void classifyCountersInTask(TaskflowTaskOp task_op) {
  // Collects all counters in the task.
  SmallVector<TaskflowCounterOp> counters;
  task_op.walk(
      [&](TaskflowCounterOp counter_op) { counters.push_back(counter_op); });

  if (counters.empty()) {
    return;
  }

  // Builds parent-child relationships.
  // Maps from counter results to counter ops.
  DenseMap<Value, TaskflowCounterOp> value_to_counter;
  for (TaskflowCounterOp counter_op : counters) {
    value_to_counter[counter_op.getCounterIndex()] = counter_op;
  }

  // Finds which counters have children.
  DenseSet<TaskflowCounterOp> counters_with_children;
  for (TaskflowCounterOp counter_op : counters) {
    if (auto parent_idx = counter_op.getParentIndex()) {
      if (auto parent_counter = value_to_counter.lookup(parent_idx)) {
        counters_with_children.insert(parent_counter);
      }
    }
  }

  // Classifies each counter.
  OpBuilder builder(task_op.getContext());
  for (TaskflowCounterOp counter_op : counters) {
    bool has_parent = (counter_op.getParentIndex() != nullptr);
    bool has_child = counters_with_children.contains(counter_op);
    StringRef counter_type;
    if (!has_parent && !has_child) {
      // Single loop: treat as leaf counter (can be mapped to the CGRA tile
      // array).
      counter_type = "leaf";
    } else if (!has_parent && has_child) {
      // Root counter: top-level loop with nested loops.
      counter_type = "root";
    } else if (has_parent && has_child) {
      // Relay counter: nested loop with further nested loops.
      counter_type = "relay";
    } else {
      // Leaf counter: innermost loop.
      counter_type = "leaf";
    }

    // Sets the counter type attribute.
    counter_op.setCounterTypeAttr(builder.getStringAttr(counter_type));
  }
}

struct ClassifyCountersPass
    : public PassWrapper<ClassifyCountersPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ClassifyCountersPass)

  StringRef getArgument() const override { return "classify-counters"; }
  StringRef getDescription() const override {
    return "Classify taskflow counters as root/relay/leaf.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk(
        [&](TaskflowTaskOp task_op) { classifyCountersInTask(task_op); });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::taskflow::createClassifyCountersPass() {
  return std::make_unique<ClassifyCountersPass>();
}