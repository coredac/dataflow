#pragma once

#include "mlir/IR/Operation.h"
#include "NeuraDialect/Architecture/Architecture.h"

namespace mlir {
namespace neura {

// Represents a recurrence cycle rooted at a reserve operation and closed by ctrl_mov.
struct RecurrenceCycle {
  SmallVector<Operation *> operations;  // Ordered list of operations in the cycle.
  int length = 0;                       // Number of operations excluding reserve/ctrl_mov.
};

// Collects recurrence cycles rooted at reserve and closed by ctrl_mov.
SmallVector<RecurrenceCycle, 4> collectRecurrenceCycles(Operation *func_op);

// Calculates ResMII: ceil(#ops / #tiles).
int calculateResMii(Operation *func_op, const Architecture &architecture);

} // namespace neura
} // namespace mlir
