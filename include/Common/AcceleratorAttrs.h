#ifndef COMMON_ACCELERATOR_ATTRS_H
#define COMMON_ACCELERATOR_ATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace accel {

// Common attribute key.
constexpr llvm::StringRef kAcceleratorAttr = "accelerator";

// Common accelerator targets.
constexpr llvm::StringRef kNeuraTarget = "neura";
constexpr llvm::StringRef kGpuTarget = "gpu";
constexpr llvm::StringRef kTpuTarget = "tpu";

} // namespace accel
} // namespace mlir

#endif // COMMON_ACCELERATOR_ATTRS_H
