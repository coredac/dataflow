// NeuraDialect.td - TableGen description of the dialect.
#ifndef NEURADIALECT_TD
#define NEURADIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"

def NeuraDialect : Dialect {
  let name = "neura";
  let cppNamespace = "::mlir::neura";
}

#endif // NEURADIALECT_TD