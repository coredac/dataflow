#ifndef NEURA_MAPPING_STRATEGY_H
#define NEURA_MAPPING_STRATEGY_H

#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/MappingState.h"
#include <vector>

namespace mlir {
namespace neura {

// Abstract base class for different mapping strategies.
class MappingStrategy {
public:
  virtual ~MappingStrategy() = default;

  // Applies the mapping strategy to map operations onto hardware
  virtual bool map(std::vector<Operation *> &sorted_ops,
                   const Architecture &architecture,
                   MappingState &mapping_state) = 0;

  // Gets the name of this strategy
  virtual std::string getName() const = 0;
};

} // namespace neura
} // namespace mlir

#endif // NEURA_MAPPING_STRATEGY_H