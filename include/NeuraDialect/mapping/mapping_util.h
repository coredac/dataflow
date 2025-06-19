#pragma once

#include "mlir/IR/Operation.h"

namespace mlir {
namespace neura {

// Represents a recurrence cycle rooted at a reserve operation and closed by ctrl_mov.
struct RecurrenceCycle {
  SmallVector<Operation *> operations;  // Ordered list of operations in the cycle.
  int length = 0;                       // Number of operations excluding reserve/ctrl_mov.
};

// Collects recurrence cycles rooted at reserve and closed by ctrl_mov.
SmallVector<RecurrenceCycle, 4> collectRecurrenceCycles(Operation *root_op);

} // namespace neura
} // namespace mlir
