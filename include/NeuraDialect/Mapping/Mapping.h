#ifndef NEURA_MAPPING_H
#define NEURA_MAPPING_H

#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/MappingState.h"
#include <vector>

namespace mlir {
namespace neura {

// Abstract base class for different mapping strategies.
class Mapping {
public:
  virtual ~Mapping() = default;

  // Applies the mapping strategy to map operations onto hardware
  virtual bool
  map(std::vector<std::pair<Operation *, int>> &sorted_ops_with_alap_levels,
      std::set<Operation *> &critical_ops, const Architecture &architecture,
      MappingState &mapping_state) = 0;

  // Gets the name of this strategy
  virtual std::string getName() const = 0;

private:
};

} // namespace neura
} // namespace mlir

#endif // NEURA_MAPPING_H