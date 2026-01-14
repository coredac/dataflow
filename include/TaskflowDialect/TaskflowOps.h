// TaskFlowOps.h - TaskFlow dialect operations.
#ifndef TASKFLOW_OPS_H
#define TASKFLOW_OPS_H

#include "TaskflowDialect/TaskflowTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// First includes the interface declarations.
#define GET_OP_INTERFACE_CLASSES
#include "TaskflowDialect/Taskflow.h.inc"
#undef GET_OP_INTERFACE_CLASSES

// Then includes the op declarations.
#define GET_OP_DECLARATIONS
#include "TaskflowDialect/Taskflow.h.inc"
#undef GET_OP_DECLARATIONS

// Finally includes the op definitions.
#define GET_OP_CLASSES
#include "TaskflowDialect/Taskflow.h.inc"
#undef GET_OP_CLASSES

#endif // TASKFLOW_OPS_H