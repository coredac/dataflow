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

  bool map(const Architecture &architecture,
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

  // Temporary structure to hold the result of no-producer operation mapping.
  struct NoProducerOpCandidate {
    MappingStateSnapshot state;
    int mapped_ops_count;
    bool fully_mapped;
    int current_ii;

    // Calculates the quality score of the solution.
    double getQualityScore() const {
      // If it is not fully mapped, return the number of mapped operations.
      if (!fully_mapped) {
        return mapped_ops_count;
      }

      // If it is fully mapped, return a score inversely proportional to II.
      return 1000.0 - this->current_ii * 100.0;
    }
  };

private:
  bool mapWithBacktrack(const Architecture &architecture,
                        MappingState &mapping_state);

  // Checks if the current operation is after a no-producer operation.
  bool
  isAfterNoProducerOp(const std::unordered_set<Operation *> &no_producer_ops,
                      int current_op_index);

  // Performs backtracking to restore the previous mapping state.
  bool performBacktrack(const std::unordered_set<Operation *> &no_producer_ops,
                        std::vector<MappingStateSnapshot> &snapshots,
                        std::vector<int> &candidate_history,
                        std::vector<int> &operation_index_history,
                        int current_op_index, MappingState &mapping_state);

  // Attempts to map a no-producer operation with global exploration.
  bool tryToMapNoProducerOp(Operation *current_op, int current_op_index,
                            const std::vector<MappingLoc> &candidate_locs,
                            std::vector<MappingStateSnapshot> &snapshots,
                            std::vector<int> &candidate_history,
                            std::vector<int> &operation_index_history,
                            const Architecture &architecture,
                            MappingState &mapping_state);

  // Evaluates a candidate location for a no-producer operation.
  NoProducerOpCandidate evaluateNoProducerOpCandidate(
      Operation *current_op, int current_op_index,
      const MappingLoc &candidate_loc, int candidate_index,
      int total_candidates, const Architecture &architecture,
      MappingState &mapping_state, MappingStateSnapshot &initial_state);

  // Configuration parameters.
  // Maximum number of locations to try for each op.
  int max_location_to_try;
  // Maximum depth for backtracking.
  int max_backtrack_depth;
};
} // namespace neura
} // namespace mlir

#endif // NEURA_HEURISTIC_MAPPING_H