#ifndef NEURA_CONVERSION_LLVMTONEURA_LLVMTONEURAPASS_H
#define NEURA_CONVERSION_LLVMTONEURA_LLVMTONEURAPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace neura {
  std::unique_ptr<Pass> createLowerLlvmToNeuraPass();
} // namespace neura
} // namespace mlir

#endif // NEURA_CONVERSION_LLVMTONEURA_LLVMTONEURAPASS_H
