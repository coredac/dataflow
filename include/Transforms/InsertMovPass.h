#ifndef NEURA_TRANSFORMS_INSERTMOVPASS_H
#define NEURA_TRANSFORMS_INSERTMOVPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace neura {
  std::unique_ptr<mlir::Pass> createInsertMovPass();
} // namespace neura
} // namespace mlir

#endif // NEURA_TRANSFORMS_INSERTMOVPASS_H

