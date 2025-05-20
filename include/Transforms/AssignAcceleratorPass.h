#ifndef NEURA_TRANSFORMS_ASSIGN_ACCELERATORPASS_H
#define NEURA_TRANSFORMS_ASSIGN_ACCELERATORPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace neura {
  std::unique_ptr<mlir::Pass> createAssignAcceleratorPass();
} // namespace neura
} // namespace mlir

#endif // NEURA_TRANSFORMS_ASSIGN_ACCELERATORPASS_H

