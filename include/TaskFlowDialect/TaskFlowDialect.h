#ifndef TASKFLOW_DIALECT_H
#define TASKFLOW_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace taskflow {
class TaskFlowDialect;
} // End namespace taskflow.
} // End namespace mlir.

#include "TaskFlowDialect/TaskFlowDialect.h.inc"
#endif // TASKFLOW_DIALECT_H