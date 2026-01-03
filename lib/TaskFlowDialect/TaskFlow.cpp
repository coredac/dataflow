#include "TaskFlowDialect/TaskFlowDialect.h"
#include "TaskFlowDialect/TaskFlowOps.h"
#include "TaskFlowDialect/TaskFlowTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::taskflow;

// Includes the generated dialect definitions.
#include "TaskFlowDialect/TaskFlowDialect.cpp.inc"

// Includes the generated type classes and storage definitions.
#define GET_TYPEDEF_CLASSES
#include "TaskFlowDialect/TaskFlowTypes.cpp.inc"

// Includes the generated operation classes.
#define GET_OP_CLASSES
#include "TaskFlowDialect/TaskFlow.cpp.inc"

void TaskFlowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TaskFlowDialect/TaskFlow.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "TaskFlowDialect/TaskFlowTypes.cpp.inc"
      >();
}

mlir::Attribute TaskFlowDialect::parseAttribute(mlir::DialectAsmParser &parser,
                                                mlir::Type type) const {
  // Currently no custom attributes to parse.
  parser.emitError(parser.getNameLoc()) << "unknown TaskFlow attribute";
  return mlir::Attribute();
}

void TaskFlowDialect::printAttribute(mlir::Attribute attr,
                                     mlir::DialectAsmPrinter &printer) const {
  // Currently no custom attributes to print.
  llvm_unreachable("Unknown TaskFlow attribute");
}