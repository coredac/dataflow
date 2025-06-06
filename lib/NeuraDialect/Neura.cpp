// #include "NeuraDialect/NeuraDialect.h"
// #include "NeuraDialect/NeuraOps.h"
// #include "NeuraDialect/NeuraTypes.h"

// using namespace mlir;
// using namespace mlir::neura;

// // Include the generated operation classes first
// #define GET_OP_CLASSES
// #include "NeuraDialect/Neura.cpp.inc"

// // Include the generated definitions
// #include "NeuraDialect/NeuraDialect.cpp.inc"

// // void mlir::neura::registerNeuraDialect() {
// //   registerDialect<NeuraDialect>();
// // }

// // Add any additional method implementations needed
// void NeuraDialect::initialize() {
//   addOperations<
// #define GET_OP_LIST
// #include "NeuraDialect/Neura.cpp.inc"
//   >();

//   addTypes<PredicatedValue>();
// }

#include "mlir/IR/DialectImplementation.h" // Required for AsmPrinter/Parser
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"

using namespace mlir;
using namespace mlir::neura;

// Include the generated dialect definitions (type ID + constructor/destructor)
#include "NeuraDialect/NeuraDialect.cpp.inc"

// Include the generated operation classes first
#define GET_OP_CLASSES
#include "NeuraDialect/Neura.cpp.inc"

// NeuraDialect::NeuraDialect(MLIRContext *context) 
//     : Dialect(getDialectNamespace(), context, TypeID::get<NeuraDialect>()) {
//   initialize();
// }

void NeuraDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NeuraDialect/Neura.cpp.inc"
  >();

  addTypes<PredicatedValue>();
}

// Type parsing/printing
Type NeuraDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == PredicatedValue::name) {
    return PredicatedValue::parse(parser);
  }

  parser.emitError(parser.getNameLoc()) << "unknown Neura type: " << keyword;
  return Type();
}

void NeuraDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (auto predType = dyn_cast<PredicatedValue>(type)) {
    printer << PredicatedValue::name;
    predType.print(printer);
    return;
  }
  llvm_unreachable("Unknown Neura type");
}

// Attribute parsing/printing
Attribute NeuraDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  // Currently no custom attributes to parse
  parser.emitError(parser.getNameLoc()) << "unknown Neura attribute";
  return Attribute();
}

void NeuraDialect::printAttribute(Attribute attr, DialectAsmPrinter &printer) const {
  // Currently no custom attributes to print
  llvm_unreachable("Unknown Neura attribute");
}

