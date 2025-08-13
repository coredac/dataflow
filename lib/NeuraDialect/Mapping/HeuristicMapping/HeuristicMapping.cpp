#include "NeuraDialect/Mapping/HeuristicMapping/HeuristicMapping.h"
#include "NeuraDialect/Mapping/MappingState.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraOps.h"
#include "llvm/Support/raw_ostream.h"
#include <ctime>
#include <random>

namespace mlir {
namespace neura {
bool HeuristicMapping::map(
    std::vector<std::pair<Operation *, int>> &sorted_ops_with_levels,
    std::set<Operation *> &critical_ops, const Architecture &architecture,
    MappingState &mapping_state) {
  // Starts the backtracking mapping process.
  return mapWithBacktrack(sorted_ops_with_levels, critical_ops, architecture,
                          mapping_state);
}

// Temporary structure to hold the result of no-producer operation mapping.
struct NoProducerOpMappingResult {
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

bool HeuristicMapping::isAfterNoProducerOp(
    const std::unordered_set<Operation *> &no_producer_ops,
    const std::vector<std::pair<Operation *, int>>
        &materialized_ops_with_levels,
    int current_op_index) {
  if (current_op_index == 0) {
    return false;
  }
  Operation *prev_op = materialized_ops_with_levels[current_op_index - 1].first;
  return no_producer_ops.count(prev_op) > 0;
}

bool HeuristicMapping::performBacktrack(
    const std::unordered_set<Operation *> &no_producer_ops,
    const std::vector<std::pair<Operation *, int>>
        &materialized_ops_with_levels,
    std::vector<MappingStateSnapshot> &snapshots,
    std::vector<int> &candidate_history,
    std::vector<int> &operation_index_history, int current_op_index,
    MappingState &mapping_state) {
  // Removes the current mapping state snapshot.
  snapshots.pop_back();
  candidate_history.pop_back();
  operation_index_history.pop_back();

  if (snapshots.empty()) {
    llvm::outs() << "[HeuristicMapping] No more snapshots to restore, "
                 << "mapping failed.\n";
    return false; // No more snapshots to restore, mapping failed.
  }

  // Because we have already thoroughly iterated candidates of no-producer
  // operations. If the current operation is after a no-producer operation, we
  // need backtrack 2 depths.
  if (this->isAfterNoProducerOp(no_producer_ops, materialized_ops_with_levels,
                                current_op_index)) {
    llvm::outs() << "[HeuristicMapping] Failed after no-producer op, "
                 << "performing deep backtrack\n";

    // Removes the mapping state snapshot of the no-producer operation.
    snapshots.pop_back();
    candidate_history.pop_back();
    operation_index_history.pop_back();

    // Checks if there are any operations left to backtrack.
    if (operation_index_history.empty()) {
      llvm::outs() << "[HeuristicMapping] No more operations available "
                   << "after deep backtrack.\n";
      return false;
    }

    // Restores the state before the previous operation of the no-producer
    // operation.
    snapshots.back().restore(mapping_state);
    candidate_history.back()++;
  } else {
    // Checks if there are any operations left to backtrack.
    if (operation_index_history.empty()) {
      llvm::outs() << "[HeuristicMapping] No more operations available "
                   << "after deep backtrack.\n";
      return false;
    }
    // Standard backtrack to the previous operation.
    snapshots.back().restore(mapping_state);
    candidate_history.back()++;
  }
  return true; // Successfully backtracked to a previous state.
}

bool HeuristicMapping::mapWithBacktrack(
    std::vector<std::pair<Operation *, int>> &sorted_ops_with_levels,
    std::set<Operation *> &critical_ops, const Architecture &architecture,
    MappingState &mapping_state) {
  llvm::outs() << "---------------------------------------------------------\n";
  llvm::outs() << "[HeuristicMapping] Starting mapping with "
               << sorted_ops_with_levels.size() << " operations.\n";
  llvm::outs() << "Configuration: MAX Backtrack Depth = "
               << this->max_backtrack_depth
               << ", MAX Candidate Locations = " << this->max_location_to_try
               << ", II = " << mapping_state.getII() << "\n";

  std::vector<std::pair<Operation *, int>> materialized_ops_with_levels;
  for (auto [op, level] : sorted_ops_with_levels) {
    if (!is_non_materialized(op)) {
      materialized_ops_with_levels.emplace_back(op, level);
    }
  }

  llvm::outs() << "[HeuristicMapping] Filtered "
               << sorted_ops_with_levels.size() -
                      materialized_ops_with_levels.size()
               << " non-materialized operations, "
               << materialized_ops_with_levels.size()
               << " operations require physical mapping." << "\n";

  std::unordered_set<Operation *> no_producer_ops;
  for (auto [op, level] : materialized_ops_with_levels) {
    bool has_producer = false;
    for (Value operand : op->getOperands()) {
      if (!isa<neura::ReserveOp>(operand.getDefiningOp())) {
        has_producer = true;
        break;
      }
    }
    if (!has_producer) {
      no_producer_ops.insert(op);
    }
  }

  // Configures the max number of candidate locations for each no-producer
  // operation.
  const int no_producer_candidates_to_try = 16;

  // Stores the mapping state snapshots for backtracking.
  std::vector<MappingStateSnapshot> snapshots;

  // For each operation, stores the candidate locations.
  std::vector<int> candidate_history;

  // Stores the mapping history for each operation.
  std::vector<int> operation_index_history;

  // Starts from the initial mapping state.
  snapshots.push_back(MappingStateSnapshot(mapping_state));
  candidate_history.push_back(0);
  operation_index_history.push_back(0);

  // Tracks the maximum depth of operations reached during mapping.
  int max_op_reached = 0;

  while (!operation_index_history.empty()) {
    // Gets the current operation index to map.
    int current_op_index = operation_index_history.back();

    if (current_op_index >=
        static_cast<int>(materialized_ops_with_levels.size())) {
      // All operations have been mapped successfully.
      llvm::outs() << "[HeuristicMapping] Successfully mapped all "
                   << materialized_ops_with_levels.size() << " operations.\n";
      return true;
    }

    max_op_reached =
        std::max(max_op_reached,
                 current_op_index); // Updates the max operation reached.

    if (max_op_reached - current_op_index > this->max_backtrack_depth) {
      llvm::outs() << "[HeuristicMapping] Max backtrack depth exceeded: "
                   << (max_op_reached - current_op_index) << " > "
                   << this->max_backtrack_depth << ".\n";
      return false; // Backtrack failed, max depth exceeded.
    }

    Operation *current_op =
        materialized_ops_with_levels[current_op_index].first;
    std::vector<MappingLoc> candidate_locs;
    candidate_locs =
        calculateAward(current_op, critical_ops,
                       materialized_ops_with_levels[current_op_index].second,
                       architecture, mapping_state);

    if (candidate_locs.empty()) {
      llvm::outs() << "[HeuristicMapping] No candidate locations found "
                   << "for operation: " << *current_op << "\n";
      if (!this->performBacktrack(no_producer_ops, materialized_ops_with_levels,
                                  snapshots, candidate_history,
                                  operation_index_history, current_op_index,
                                  mapping_state)) {
        return false;
      }
      continue; // Backtrack to the previous operation.
    }

    llvm::outs() << "[HeuristicMapping] Found " << candidate_locs.size()
                 << " candidate locations for operation: " << *current_op
                 << "\n";

    // Handles no-producer operations with global exploration.
    if (no_producer_ops.count(current_op) > 0 && candidate_locs.size() > 1) {
      llvm::outs()
          << "[HeuristicMapping] Using global exploration for no-producer op: "
          << *current_op << "\n";

      // Stores the initial state before global exploration.
      MappingStateSnapshot initial_state(mapping_state);
      std::vector<NoProducerOpMappingResult> solutions;

      // Determines the number of candidate locations to try.
      int candidates_to_try = std::min(static_cast<int>(candidate_locs.size()),
                                       no_producer_candidates_to_try);

      // Tries full mapping with greedy strategy (All candidate locations and
      // zero backtrack) for each candidate location. And use the number of
      // mapped operations as the score.
      for (int i = 0; i < candidates_to_try; i++) {
        MappingLoc candidate_loc = candidate_locs[i];

        llvm::outs() << "[GlobalExploration] Trying candidate position "
                     << (i + 1) << "/" << candidates_to_try << " at "
                     << candidate_loc.resource->getType() << "#"
                     << candidate_loc.resource->getId()
                     << " @t=" << candidate_loc.time_step << " for "
                     << *current_op << "\n";

        // Restores the initial mapping state for each candidate.
        initial_state.restore(mapping_state);

        if (!placeAndRoute(current_op, candidate_loc, mapping_state)) {
          llvm::outs() << "[GlobalExploration] Failed to map at position "
                       << (i + 1) << "\n";
          continue;
        }

        // Creates a backup to save the mapping state for this position.
        MappingStateSnapshot position_state(mapping_state);
        // Indicates if we successfully map all the remaining operations.
        bool success = true;
        // Stores the number of mapped operations if we cannot map all the
        // remaining operations.
        int mapped_count = 1; // Current operation is mapped.

        // Tries to map the remaining operations. Starts from the next operation
        // index.
        for (int next_idx = current_op_index + 1;
             next_idx < static_cast<int>(materialized_ops_with_levels.size());
             next_idx++) {

          Operation *next_op = materialized_ops_with_levels[next_idx].first;
          int next_level = materialized_ops_with_levels[next_idx].second;

          // Calculates the candidate locations for the next operation.
          auto next_candidates = calculateAward(
              next_op, critical_ops, next_level, architecture, mapping_state);

          if (next_candidates.empty()) {
            success = false;
            break;
          }

          // Tries to place and route the next operation at each candidate
          // location.
          bool mapped = false;
          for (const auto &next_loc : next_candidates) {
            if (placeAndRoute(next_op, next_loc, mapping_state)) {
              mapped = true;
              mapped_count++;
              break;
            }
          }

          if (!mapped) {
            success = false;
            break;
          }
        }

        // If we successfully mapped all the remaining operations, we create a
        // new snapshot of the mapping state. We directly use it as a solution.
        if (success) {
          // Creates a final state snapshot after mapping all operations.
          MappingStateSnapshot final_state(mapping_state);

          NoProducerOpMappingResult solution = {
              final_state, // Use the final state after mapping all ops.
              mapped_count, success, mapping_state.getII()};

          solutions.push_back(solution);

          llvm::outs()
              << "[GlobalExploration] Found complete mapping solution with "
              << "mapped=" << mapped_count << "/"
              << materialized_ops_with_levels.size()
              << ", estimated_II=" << mapping_state.getII()
              << ", score=" << solution.getQualityScore() << "\n";
        } else {
          // If we cannot map all operations, we create a solution with the
          // current position state.
          NoProducerOpMappingResult solution = {position_state, mapped_count,
                                                success, mapping_state.getII()};

          solutions.push_back(solution);

          llvm::outs() << "[GlobalExploration] Solution quality: "
                       << "mapped=" << mapped_count << "/"
                       << materialized_ops_with_levels.size()
                       << ", fully_mapped=" << (success ? "yes" : "no")
                       << ", estimated_II=" << mapping_state.getII()
                       << ", score=" << solution.getQualityScore() << "\n";
        }
      }

      // If we have no solutions, we need to backtrack.
      if (solutions.empty()) {
        llvm::outs() << "[GlobalExploration] No feasible solutions found, "
                        "backtracking\n";

        snapshots.pop_back();
        candidate_history.pop_back();
        operation_index_history.pop_back();

        if (snapshots.empty()) {
          llvm::outs() << "[HeuristicMapping] No more snapshots to restore, "
                          "mapping failed.\n";
          return false;
        }

        snapshots.back().restore(mapping_state);
        candidate_history.back()++;
        continue;
      }

      // Selects the best solution for mapping this no-producer op based on the
      // quality score.
      NoProducerOpMappingResult best_solution =
          *std::max_element(solutions.begin(), solutions.end(),
                            [](const NoProducerOpMappingResult &a,
                               const NoProducerOpMappingResult &b) {
                              return a.getQualityScore() < b.getQualityScore();
                            });

      best_solution.state.restore(mapping_state);

      llvm::outs() << "[GlobalExploration] Selected best solution with "
                   << "score=" << best_solution.getQualityScore()
                   << ", mapped=" << best_solution.mapped_ops_count
                   << ", fully_mapped="
                   << (best_solution.fully_mapped ? "yes" : "no")
                   << ", estimated_II=" << best_solution.current_ii << "\n";

      // If we have a fully mapped solution, we can finalize the mapping.
      if (best_solution.fully_mapped) {
        llvm::outs()
            << "[HeuristicMapping] Found complete mapping solution through "
            << "global exploration, finalizing...\n";
        return true;
      }

      // Otherwise continues the normal mapping process from the current best
      // state
      snapshots.push_back(MappingStateSnapshot(mapping_state));
      candidate_history.push_back(0);
      operation_index_history.push_back(current_op_index + 1);
      continue;
    }

    // Limits the number of locations to try.
    if (candidate_locs.size() >
        static_cast<size_t>(this->max_location_to_try)) {
      candidate_locs.resize(static_cast<size_t>(this->max_location_to_try));
    }

    int &current_candidate_index = candidate_history.back();

    if (current_candidate_index >= static_cast<int>(candidate_locs.size())) {
      // Needs to backtrack since all candidate locations have been tried.
      llvm::outs() << "[HeuristicMapping] All " << candidate_locs.size()
                   << " locations for " << current_op_index
                   << " tried, backtracking...\n";
      if (!this->performBacktrack(no_producer_ops, materialized_ops_with_levels,
                                  snapshots, candidate_history,
                                  operation_index_history, current_op_index,
                                  mapping_state)) {
        return false; // Backtrack failed, no more snapshots to restore.
      }
      continue;
    }

    MappingLoc candidate_loc = candidate_locs[current_candidate_index];

    llvm::outs() << "[HeuristicMapping] Trying candidate "
                 << (current_candidate_index + 1) << "/"
                 << candidate_locs.size() << " at "
                 << candidate_loc.resource->getType() << "#"
                 << candidate_loc.resource->getId()
                 << " @t=" << candidate_loc.time_step << "\n";

    // Saves the current mapping state snapshot before attempting to map.
    MappingStateSnapshot snapshot(mapping_state);

    // Attempts to place and route the operation at the candidate location.
    if (placeAndRoute(current_op, candidate_loc, mapping_state)) {
      llvm::outs() << "[HeuristicMapping] Successfully mapped operation "
                   << *current_op << "\n";

      // Adds a new decision point for the next operation.
      snapshots.push_back(MappingStateSnapshot(mapping_state));
      candidate_history.push_back(0);
      operation_index_history.push_back(current_op_index + 1);
    } else {
      // Mapping failed, restores state and tries the next candidate.
      llvm::outs() << "[HeuristicMapping] Failed to map operation "
                   << *current_op << " to candidate location "
                   << (current_candidate_index + 1) << "/"
                   << candidate_locs.size() << "\n";

      snapshot.restore(mapping_state);
      current_candidate_index++;
    }
  }

  // If we reach here, it means we have exhausted all possibilities.
  llvm::errs() << "[HeuristicMapping] FAILURE: Exhausted all possibilities\n";
  return false;
}

} // namespace neura
} // namespace mlir
