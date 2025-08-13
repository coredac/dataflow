#ifndef NEURA_HEURISTIC_MAPPING_H
#define NEURA_HEURISTIC_MAPPING_H

#include "NeuraDialect/Mapping/Mapping.h"
#include "NeuraDialect/Mapping/MappingState.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include <climits>
#include <map>
#include <set>

namespace mlir {
namespace neura {
class HeuristicMapping : public Mapping {
public:
  HeuristicMapping(int max_location_to_try = 5, int max_backtrack_depth = 3)
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

  // Checks if the current operation is after a no-producer operation.
  bool
  isAfterNoProducerOp(const std::unordered_set<Operation *> &no_producer_ops,
                      const std::vector<std::pair<Operation *, int>>
                          &materialized_ops_with_levels,
                      int current_op_index);

  // Performs backtracking to restore the previous mapping state.
  bool performBacktrack(const std::unordered_set<Operation *> &no_producer_ops,
                        const std::vector<std::pair<Operation *, int>>
                            &materialized_ops_with_levels,
                        std::vector<MappingStateSnapshot> &snapshots,
                        std::vector<int> &candidate_history,
                        std::vector<int> &operation_index_history,
                        int current_op_index, MappingState &mapping_state);

  // Gets the sorted candidate locations for a given operation based on
  // spatial execution model.
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