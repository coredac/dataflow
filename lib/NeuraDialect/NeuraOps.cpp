#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::neura;

LogicalResult PhiStartOp::verify() {
  // Checks if this phi_start is inside a fused_op.
  Operation *parent_op = getOperation()->getParentOp();
  bool inside_fused_op = false;
  while (parent_op) {
    if (isa<FusedOp>(parent_op)) {
      inside_fused_op = true;
      break;
    }
    parent_op = parent_op->getParentOp();
  }

  if (!inside_fused_op) {
    // Verifies that the reserved operand is produced by a neura.reserve
    // operation.
    Value reserved = getReserved();
    Operation *def_op = reserved.getDefiningOp();

    if (!def_op) {
      return emitOpError("reserve operand must be defined by an operation.");
    }

    if (!isa<ReserveOp>(def_op)) {
      return emitOpError("reserve operand must be produced by a neura.reserve "
                         "operation.");
    }
  }

  // Verifies that there is at least one initialization value.
  if (getInitValues().empty()) {
    return emitOpError("At least one initialization value is required.");
  }

  // Verifies type consistency.
  Type result_type = getResult().getType();
  Type reserved_type = getReserved().getType();

  if (result_type != reserved_type) {
    return emitOpError("Result type must match the reserved value type.");
  }

  for (Value init_val : getInitValues()) {
    if (init_val.getType() != result_type) {
      return emitOpError(
          "All initialization values must match the result type.");
    }
  }

  return success();
}