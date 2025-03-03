#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace neura;

// Registers the dialect inside MLIR.
NeuraDialect::NeuraDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<NeuraDialect>()) {
  initialize();
}

void NeuraDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "NeuraDialect/NeuraOps.cpp.inc"
  >();
}

// Defines the virtual destructor.
NeuraDialect::~NeuraDialect() = default;

// Registers the dialect with MLIR.
// static mlir::DialectRegistration<NeuraDialect> NeuraDialectRegistration;
void registerNeuraDialect(mlir::DialectRegistry &registry) {
  registry.insert<NeuraDialect>();
}

// Defines the TypeID for NeuraDialect.
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::neura::NeuraDialect)

