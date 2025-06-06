// NeuraOps.h
#ifndef NEURA_OPS_H
#define NEURA_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"

// First include the interface declarations
#define GET_OP_INTERFACE_CLASSES
#include "NeuraDialect/Neura.h.inc"
#undef GET_OP_INTERFACE_CLASSES

// Then include the op declarations
#define GET_OP_DECLARATIONS
#include "NeuraDialect/Neura.h.inc"
#undef GET_OP_DECLARATIONS

// Finally include the op definitions
#define GET_OP_CLASSES
#include "NeuraDialect/Neura.h.inc"
#undef GET_OP_CLASSES

#endif // NEURA_OPS_H
