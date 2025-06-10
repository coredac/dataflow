#ifndef NEURA_DIALECT_H
#define NEURA_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#ifdef _WIN32
#define NEURA_DIALECT_EXPORT __declspec(dllexport)
#else
#define NEURA_DIALECT_EXPORT __attribute__((visibility("default")))
#endif

namespace mlir {
namespace neura {

// Forward declare before including generated code
class NeuraDialect;

} // end namespace neura
} // end namespace mlir

// Include the generated dialect declarations
#include "NeuraDialect/NeuraDialect.h.inc"

namespace mlir {
namespace neura {

// Declare additional methods for the generated dialect class
NEURA_DIALECT_EXPORT void registerNeuraDialect();

} // end namespace neura
} // end namespace mlir

#endif // NEURA_DIALECT_H