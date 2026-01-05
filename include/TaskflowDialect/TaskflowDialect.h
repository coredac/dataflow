#ifndef TASKFLOW_DIALECT_H
#define TASKFLOW_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace taskflow {
class TaskflowDialect;
} // End namespace taskflow.
} // End namespace mlir.

#include "TaskflowDialect/TaskflowDialect.h.inc"
#endif // TASKFLOW_DIALECT_H