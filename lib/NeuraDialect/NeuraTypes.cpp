#include "NeuraDialect/NeuraTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::neura;

Type PredicatedValue::parse(AsmParser &parser) {
  // Parse: !neura.predicated<type, i1>
  Type valueType, predicateType;

  if (parser.parseLess() || parser.parseType(valueType) ||
      parser.parseComma() || parser.parseType(predicateType) ||
      parser.parseGreater()) {
    return Type();
  }

  // Verify predicate is i1
  auto intType = mlir::dyn_cast<IntegerType>(predicateType);
  if (!intType || !intType.isInteger(1)) {
    parser.emitError(parser.getNameLoc())
        << "predicate type must be i1, got " << predicateType;
    return Type();
  }

  return get(parser.getContext(), valueType, intType);
}

void PredicatedValue::print(AsmPrinter &printer) const {
  printer << "<" << getValueType() << ", " << getPredicateType() << ">";
}