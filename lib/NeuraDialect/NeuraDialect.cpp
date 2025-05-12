#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace neura;

#include "NeuraDialect.cpp.inc"

void NeuraDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "NeuraOps.cpp.inc"
  >();
}

