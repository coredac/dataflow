#ifndef NEURA_SPATIAL_MAPPING_H
#define NEURA_SPATIAL_MAPPING_H

#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/MappingState.h"
#include "NeuraDialect/Mapping/MappingStrategy.h"
#include <climits>
#include <map>
#include <set>
#include <vector>

namespace mlir {
namespace neura {
// Implements a spatial-only mapping strategy where each tile can only be
// assigned to one operation regardless of time.
class SpatialMapping : public MappingStrategy {
public:
  SpatialMapping(int max_location_to_try = 5, int max_backtrack_depth = 3)
      : max_location_to_try(max_location_to_try),
        max_backtrack_depth(max_backtrack_depth) {}

  bool map(std::vector<std::pair<Operation *, int>> &sorted_ops_with_levels,
           std::set<Operation *> &critical_ops,
           const Architecture &architecture,
           MappingState &mapping_state) override;

  std::string getName() const override {
    if (max_location_to_try == 1 && max_backtrack_depth == 1) {
      return "spatial-simple";
    } else if (max_location_to_try == INT_MAX && max_backtrack_depth == 1) {
      return "spatial-greedy";
    } else if (max_location_to_try == INT_MAX &&
               max_backtrack_depth == INT_MAX) {
      return "spatial-exhaustive";
    } else {
      return "spatial-heuristic";
    }
  }

private:
  bool mapWithBacktrack(
      std::vector<std::pair<Operation *, int>> &sorted_ops_with_levels,
      std::set<Operation *> &critical_ops, const Architecture &architecture,
      MappingState &mapping_state, size_t current_index, int backtrack_depth);

  std::vector<MappingLoc>
  caculateSpatialAward(Operation *op, std::set<Operation *> &critical_ops,
                       int target_level, const Architecture &architecture,
                       const MappingState &mapping_state);

  bool canReachLocSpatial(const std::vector<Operation *> &producers,
                          const MappingLoc &target_loc, int deadline_step,
                          const MappingState &mapping_state);

  bool canReachLocSpatial(const MappingLoc &src_loc, const MappingLoc &dst_loc,
                          int deadline_step, const MappingState &mapping_state);

  // Maximum number of locations to try for each operation.
  int max_location_to_try;
  // Maximum depth for backtracking.
  int max_backtrack_depth;
};
} // namespace neura
} // namespace mlir
#endif