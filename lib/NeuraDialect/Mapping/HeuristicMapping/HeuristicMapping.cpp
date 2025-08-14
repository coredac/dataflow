#include "NeuraDialect/Mapping/HeuristicMapping/HeuristicMapping.h"
#include "NeuraDialect/Mapping/MappingState.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraOps.h"
#include "llvm/Support/raw_ostream.h"
#include <ctime>

namespace mlir {
namespace neura {
bool HeuristicMapping::map(const Architecture &architecture,
                           MappingState &mapping_state) {
  // Starts the backtracking mapping process.
  return mapWithBacktrack(architecture, mapping_state);
}

bool HeuristicMapping::isAfterNoProducerOp(
    const std::unordered_set<Operation *> &no_producer_ops,
    int current_op_index) {
  if (current_op_index == 0) {
    return false;
  }
  Operation *prev_op =
      getMaterializedOpsWithLevels()[current_op_index - 1].first;
  return no_producer_ops.count(prev_op) > 0;
}

HeuristicMapping::NoProducerOpCandidate
HeuristicMapping::evaluateNoProducerOpCandidate(
    Operation *current_op, int current_op_index,
    const MappingLoc &candidate_loc, int candidate_index, int total_candidates,
    const Architecture &architecture, MappingState &mapping_state,
    MappingStateSnapshot &initial_state) {
  llvm::outs() << "[GlobalExploration] Trying candidate position "
               << candidate_index << "/" << total_candidates << " at "
               << candidate_loc.resource->getType() << "#"
               << candidate_loc.resource->getId()
               << " @t=" << candidate_loc.time_step << " for " << *current_op
               << "\n";
  // Restores the initial mapping state for each candidate.
  initial_state.restore(mapping_state);

  if (!placeAndRoute(current_op, candidate_loc, mapping_state)) {
    llvm::outs() << "[GlobalExploration] Failed to map at position "
                 << candidate_index << "\n";
    return {MappingStateSnapshot(mapping_state), 0, false,
            mapping_state.getII()};
  }

  // Creates a backup to save the mapping state for this position.
  MappingStateSnapshot position_state(mapping_state);
  // Indicates if we successfully map all the remaining operations.
  bool success = true;
  // Stores the number of mapped operations if we cannot map all the remaining
  // operations.
  int mapped_count = 1; // Current operation is mapped.

  // Tries to map the remaining operations. Starts from the next operation
  // index.
  for (int next_idx = current_op_index + 1;
       next_idx < static_cast<int>(getMaterializedOpsWithLevels().size());
       next_idx++) {

    Operation *next_op = getMaterializedOpsWithLevels()[next_idx].first;
    int next_level = getMaterializedOpsWithLevels()[next_idx].second;

    // Calculates the candidate locations for the next operation.
    auto next_candidates = calculateAward(next_op, getCriticalOps(), next_level,
                                          architecture, mapping_state);

    if (next_candidates.empty()) {
      success = false;
      break;
    }

    // Tries to place and route the next operation at each candidate location.
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

  // Returns the candidate result.
  if (success) {
    // If fully mapped, creates the final state snapshot.
    MappingStateSnapshot final_state(mapping_state);
    return {final_state, mapped_count, true, mapping_state.getII()};
  } else {
    // If not fully mapped, returns the position state.
    return {position_state, mapped_count, false, mapping_state.getII()};
  }
}

bool HeuristicMapping::tryToMapNoProducerOp(
    Operation *current_op, int current_op_index,
    const std::vector<MappingLoc> &candidate_locs,
    std::vector<MappingStateSnapshot> &snapshots,
    std::vector<int> &candidate_history,
    std::vector<int> &operation_index_history, const Architecture &architecture,
    MappingState &mapping_state) {
  llvm::outs()
      << "[HeuristicMapping] Using global exploration for no-producer op: "
      << *current_op << "\n";

  // Stores the initial state before global exploration.
  MappingStateSnapshot initial_state(mapping_state);
  std::vector<NoProducerOpCandidate> solutions;

  // Configures the max number of candidate locations for each no-producer
  // operation. We set it to a reasonable number based on the architecture size.
  const int no_producer_candidates_to_try =
      architecture.getHeight() * architecture.getWidth();

  // Determines the number of candidate locations to try.
  int candidates_to_try = std::min(static_cast<int>(candidate_locs.size()),
                                   no_producer_candidates_to_try);

  // Tries full mapping with greedy strategy (all candidate locations and zero
  // backtrack) for each candidate location. And use the number of
  // mapped operations as the score.
  for (int i = 0; i < candidates_to_try; i++) {
    // Gets the mapping results for each candidate location.
    NoProducerOpCandidate result = evaluateNoProducerOpCandidate(
        current_op, current_op_index, candidate_locs[i], i + 1,
        candidates_to_try, architecture, mapping_state, initial_state);

    if (result.mapped_ops_count > 0) {
      solutions.push_back(result);

      // Outputs the solution quality information.
      if (result.fully_mapped) {
        llvm::outs()
            << "[GlobalExploration] Found complete mapping solution with "
            << "mapped=" << result.mapped_ops_count << "/"
            << getMaterializedOpsWithLevels().size()
            << ", estimated_II=" << result.current_ii
            << ", score=" << result.getQualityScore() << "\n";
      } else {
        llvm::outs() << "[GlobalExploration] Solution quality: "
                     << "mapped=" << result.mapped_ops_count << "/"
                     << getMaterializedOpsWithLevels().size()
                     << ", fully_mapped="
                     << (result.fully_mapped ? "yes" : "no")
                     << ", estimated_II=" << result.current_ii
                     << ", score=" << result.getQualityScore() << "\n";
      }
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
    return false;
  }

  // Selects the best solution for mapping this no-producer op based on the
  // quality score.
  NoProducerOpCandidate best_solution = *std::max_element(
      solutions.begin(), solutions.end(),
      [](const NoProducerOpCandidate &a, const NoProducerOpCandidate &b) {
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

  // Otherwise continues the normal mapping process from the current best state.
  snapshots.push_back(MappingStateSnapshot(mapping_state));
  candidate_history.push_back(0);
  operation_index_history.push_back(current_op_index + 1);
  return false;
}

bool HeuristicMapping::performBacktrack(
    const std::unordered_set<Operation *> &no_producer_ops,
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
  if (this->isAfterNoProducerOp(no_producer_ops, current_op_index)) {
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

bool HeuristicMapping::mapWithBacktrack(const Architecture &architecture,
                                        MappingState &mapping_state) {
  llvm::outs() << "---------------------------------------------------------\n";
  llvm::outs() << "[HeuristicMapping] Starting mapping with "
               << getSortedOpsWithLevels().size() << " operations.\n";
  llvm::outs() << "Configuration: MAX Backtrack Depth = "
               << this->max_backtrack_depth
               << ", MAX Candidate Locations = " << this->max_location_to_try
               << ", II = " << mapping_state.getII() << "\n";

  llvm::outs() << "[HeuristicMapping] Filtered "
               << getSortedOpsWithLevels().size() -
                      getMaterializedOpsWithLevels().size()
               << " non-materialized operations, "
               << getMaterializedOpsWithLevels().size()
               << " operations require physical mapping." << "\n";

  std::unordered_set<Operation *> no_producer_ops;
  for (auto [op, level] : getMaterializedOpsWithLevels()) {
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
        static_cast<int>(getMaterializedOpsWithLevels().size())) {
      // All operations have been mapped successfully.
      llvm::outs() << "[HeuristicMapping] Successfully mapped all "
                   << getMaterializedOpsWithLevels().size() << " operations.\n";
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
        getMaterializedOpsWithLevels()[current_op_index].first;
    std::vector<MappingLoc> candidate_locs;
    candidate_locs =
        calculateAward(current_op, getCriticalOps(),
                       getMaterializedOpsWithLevels()[current_op_index].second,
                       architecture, mapping_state);

    if (candidate_locs.empty()) {
      llvm::outs() << "[HeuristicMapping] No candidate locations found "
                   << "for operation: " << *current_op << "\n";
      if (!this->performBacktrack(no_producer_ops, snapshots, candidate_history,
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
      bool fully_mapped = tryToMapNoProducerOp(
          current_op, current_op_index, candidate_locs, snapshots,
          candidate_history, operation_index_history, architecture,
          mapping_state);
      if (fully_mapped) {
        llvm::outs() << "[HeuristicMapping] Found complete mapping solution "
                        "for no-producer op, finalizing...\n";
        return true; // Mapping completed successfully.
      }
      continue; // Continue to the next operation.
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
      if (!this->performBacktrack(no_producer_ops, snapshots, candidate_history,
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
