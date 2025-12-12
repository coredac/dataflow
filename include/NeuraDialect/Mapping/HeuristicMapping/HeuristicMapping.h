#ifndef NEURA_HEURISTIC_MAPPING_H
#define NEURA_HEURISTIC_MAPPING_H

#include "NeuraDialect/Mapping/Mapping.h"
#include "NeuraDialect/Mapping/MappingState.h"
#include <climits>
#include <set>

namespace mlir {
namespace neura {
class HeuristicMapping : public Mapping {
public:
  HeuristicMapping(int max_location_to_try = 500, int max_backtrack_depth = 20)
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
      return "customized";
    }
  }

private:
  bool mapWithBacktrack(
      std::vector<std::pair<Operation *, int>> &sorted_ops_with_levels,
      std::set<Operation *> &critical_ops, const Architecture &architecture,
      MappingState &mapping_state);

  // Gets the sorted candidate locations for a given operation based on spatial
  // execution model.
  std::vector<MappingLoc>
  calculateSpatialAward(Operation *op, std::set<Operation *> &critical_ops,
                        int target_level, const Architecture &architecture,
                        const MappingState &mapping_state);

  // Configuration parameters.
  // Maximum number of locations to try for each op.
  int max_location_to_try;
  // Maximum depth for backtracking.
  int max_backtrack_depth;
};
} // namespace neura
} // namespace mlir

#endif // NEURA_HEURISTIC_MAPPING_H