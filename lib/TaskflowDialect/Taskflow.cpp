#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::taskflow;

// Includes the generated dialect definitions.
#include "TaskflowDialect/TaskflowDialect.cpp.inc"
// Includes the generated type classes and storage definitions.
#define GET_TYPEDEF_CLASSES
#include "TaskflowDialect/TaskflowTypes.cpp.inc"

// Includes the generated operation classes.
#define GET_OP_CLASSES
#include "TaskflowDialect/Taskflow.cpp.inc"

void TaskflowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TaskflowDialect/Taskflow.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "TaskflowDialect/TaskflowTypes.cpp.inc"
      >();
}

mlir::Attribute TaskflowDialect::parseAttribute(mlir::DialectAsmParser &parser,
                                                mlir::Type type) const {
  // Currently no custom attributes to parse.
  parser.emitError(parser.getNameLoc()) << "unknown Taskflow attribute";
  return mlir::Attribute();
}

void TaskflowDialect::printAttribute(mlir::Attribute attr,
                                     mlir::DialectAsmPrinter &printer) const {
  // Currently no custom attributes to print.
  llvm_unreachable("Unknown Taskflow attribute");
}

mlir::Type TaskflowDialect::parseType(mlir::DialectAsmParser &parser) const {
  // Currently no custom types to parse.
  parser.emitError(parser.getNameLoc()) << "unknown Taskflow type";
  return mlir::Type();
}

void TaskflowDialect::printType(mlir::Type type,
                                mlir::DialectAsmPrinter &printer) const {
  // Currently no custom types to print.
  llvm_unreachable("Unknown Taskflow type");
}