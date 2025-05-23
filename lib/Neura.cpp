#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::neura;

#include "NeuraDialect/NeuraDialect.cpp.inc"

void NeuraDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "NeuraDialect/Neura.cpp.inc"
  >();
}

#define GET_OP_CLASSES
#include "NeuraDialect/Neura.cpp.inc"


