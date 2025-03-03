#ifndef NEURADIALECT_NEURADIALECT_H
#define NEURADIALECT_NEURADIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

// Defines the export macro.
#ifdef _WIN32
  #define NEURA_DIALECT_EXPORT __declspec(dllexport)
#else
  #define NEURA_DIALECT_EXPORT __attribute__((visibility("default")))
#endif

// Includes generated TableGen headers.
#include "NeuraDialect/NeuraDialect.h.inc"

#endif // NEURADIALECT_NEURADIALECT_H
