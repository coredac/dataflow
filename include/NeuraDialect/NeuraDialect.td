// NeuraDialect.td - TableGen description of the dialect.
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/OpBase.td"

def NeuraDialect : Dialect {
  let name = "neura";
  let cppNamespace = "::mlir::neura";
}
