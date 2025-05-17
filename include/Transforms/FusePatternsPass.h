#ifndef NEURA_TRANSFORMS_FUSEPATTERNSPASS_H
#define NEURA_TRANSFORMS_FUSEPATTERNSPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir::neura {
std::unique_ptr<mlir::Pass> createFusePatternsPass();
}

#endif // NEURA_TRANSFORMS_FUSEPATTERNSPASS_H

