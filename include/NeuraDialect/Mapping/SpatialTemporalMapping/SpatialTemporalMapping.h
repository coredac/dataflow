#ifndef NEURA_SPATIALTEMPORAL_MAPPING_H
#define NEURA_SPATIALTEMPORAL_MAPPING_H

#include "NeuraDialect/Mapping/MappingState.h"
#include "NeuraDialect/Mapping/MappingStrategy.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include <climits>
#include <map>
#include <set>

namespace mlir {
namespace neura {
class SpatialTemporalMapping : public MappingStrategy {
public:
  SpatialTemporalMapping(int max_location_to_try = 5,
                         int max_backtrack_depth = 3)
      : max_location_to_try(max_location_to_try),
        max_backtrack_depth(max_backtrack_depth) {}

  bool map(std::vector<std::pair<Operation *, int>> &sorted_ops_with_levels,
           std::set<Operation *> &critical_ops,
           const Architecture &architecture,
           MappingState &mapping_state) override;

  std::string getName() const override {
    if (max_location_to_try == 1 && max_backtrack_depth == 1) {
      return "simple";
    } else if (max_location_to_try == INT_MAX && max_backtrack_depth == 1) {
      return "greedy";
    } else if (max_location_to_try == INT_MAX &&
               max_backtrack_depth == INT_MAX) {
      return "exhaustive";
    } else {
      return "heuristic";
    }
  }

private:
  bool mapWithBacktrack(
      std::vector<std::pair<Operation *, int>> &sorted_ops_with_levels,
      std::set<Operation *> &critical_ops, const Architecture &architecture,
      MappingState &mapping_state);

  // Configuration parameters.
  int max_location_to_try; // Maximum number of locations to try for
                           // each op
  int max_backtrack_depth; // Maximum depth for backtracking
};
} // namespace neura
} // namespace mlir

#endif // NEURA_SPATIALTEMPORAL_MAPPING_H