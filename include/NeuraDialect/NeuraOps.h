// NeuraOps.h
#ifndef NEURA_OPS_H
#define NEURA_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "NeuraOps.h.inc"

// Additional definitions or includes can go here.

#endif // NEURA_OPS_H
