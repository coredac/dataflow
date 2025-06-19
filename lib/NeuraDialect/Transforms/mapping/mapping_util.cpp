#include <deque>

#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::neura;

namespace {

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

      if (res_op) current_path.pop_front();
      continue;
    }

    if (Operation *def_op = operand.getDefiningOp()) {
      traverseAlongPath(def_op, reserve_value, current_path, visited_in_path, collected_paths);
    }
  }

  current_path.pop_front();
  visited_in_path.erase(op);
}

} // namespace

SmallVector<RecurrenceCycle, 4> mlir::neura::collectRecurrenceCycles(Operation *root_op) {
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
