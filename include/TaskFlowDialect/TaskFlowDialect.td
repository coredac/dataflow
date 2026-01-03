// TaskFlowDialect.td - TableGen description of the dialect.
#ifndef TASKFLOW_DIALECT_TD
#define TASKFLOW_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

// Defines the TaskFlow dialect.
def TaskFlowDialect : Dialect {
  let name = "taskflow";
  let cppNamespace = "::mlir::taskflow";

  let summary = "A computation abstraction dialect for both scale-out and scale-up spatial architectures.";
  let description = [{
     This dialect provides abstractions for expressing task-based computations
     suitable for execution on various spatial architectures.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // TASKFLOW_DIALECT_TD