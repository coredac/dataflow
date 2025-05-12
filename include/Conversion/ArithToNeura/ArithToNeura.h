#ifndef NEURA_CONVERSION_ARITHTONEURA_ARITHTONEURAPASS_H
#define NEURA_CONVERSION_ARITHTONEURA_ARITHTONEURAPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace neura {
  std::unique_ptr<Pass> createLowerArithToNeuraPass();
} // namespace neura
} // namespace mlir

#endif // NEURA_CONVERSION_ARITHTONEURA_ARITHTONEURAPASS_H
