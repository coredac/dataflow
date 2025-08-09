#include "NeuraDialect/Mapping/HeuristicMapping/HeuristicMapping.h"
#include "NeuraDialect/Mapping/MappingState.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraOps.h"
#include "llvm/Support/raw_ostream.h"

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
               << "\n";

  std::vector<std::pair<Operation *, int>> materialized_ops;
  for (auto [op, level] : sorted_ops_with_levels) {
    if (!is_non_materialized(op)) {
      materialized_ops.emplace_back(op, level);
    }
  }

  llvm::outs() << "[HeuristicMapping] Filtered "
               << sorted_ops_with_levels.size() - materialized_ops.size()
               << " non-materialized operations, " << materialized_ops.size()
               << " operations require physical mapping." << "\n";

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

    if (current_op_index >= static_cast<int>(materialized_ops.size())) {
      // All operations have been mapped successfully.
      llvm::outs() << "[HeuristicMapping] Successfully mapped all "
                   << materialized_ops.size() << " operations.\n";
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

    Operation *current_op = materialized_ops[current_op_index].first;
    std::vector<MappingLoc> candidate_locs;
    if (this->is_spatial) {
      candidate_locs = calculateSpatialAward(
          current_op, critical_ops, materialized_ops[current_op_index].second,
          architecture, mapping_state);
    } else {
      // For non-spatial mapping, we can use the existing calculateAward
      // function.
      candidate_locs = calculateAward(current_op, critical_ops,
                                      materialized_ops[current_op_index].second,
                                      architecture, mapping_state);
    }

    if (candidate_locs.empty()) {
      llvm::outs() << "[HeuristicMapping] No candidate locations found "
                   << "for operation: " << *current_op << "\n";
      // No candidate locations available, backtrack to the previous operation.
      snapshots.pop_back(); // Restore the previous mapping state.
      candidate_history.pop_back();
      operation_index_history.pop_back();

      if (snapshots.empty()) {
        llvm::outs() << "[HeuristicMapping] No more snapshots to restore, "
                     << "mapping failed.\n";
        return false; // No more snapshots to restore, mapping failed.
      }

      snapshots.back().restore(mapping_state);
      candidate_history.back()++;
      llvm::outs() << "[HeuristicMapping] Backtracking to operation "
                   << operation_index_history.back() << "(depth = "
                   << (max_op_reached - operation_index_history.back())
                   << ")\n";
      continue; // Backtrack to the previous operation.
    }

    llvm::outs() << "[HeuristicMapping] Found " << candidate_locs.size()
                 << " candidate locations for operation: " << *current_op
                 << "\n";
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

      // Removes the last mapping state snapshot and candidate history.
      snapshots.pop_back();
      candidate_history.pop_back();
      operation_index_history.pop_back();

      // If no more operation indices to backtrack, mapping failed.
      if (operation_index_history.empty()) {
        llvm::outs() << "[HeuristicMapping] FAILURE: No more operations "
                        "available for backtracking.\n";
        return false;
      }

      // Restores the previous state
      snapshots.back().restore(mapping_state);

      // Increments the candidate location index for the previous decision point
      candidate_history.back()++;

      llvm::outs() << "[HeuristicMapping] Backtracking to operation "
                   << operation_index_history.back() << " (depth = "
                   << (max_op_reached - operation_index_history.back())
                   << ").\n";

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

std::vector<MappingLoc> HeuristicMapping::calculateSpatialAward(
    Operation *op, std::set<Operation *> &critical_ops, int target_level,
    const Architecture &architecture, const MappingState &mapping_state) {
  llvm::outs() << "[caculateSpatialAward] Calculating spatial award for " << *op
               << "\n";
  // Early exit if the operation is not supported by all the tiles.
  bool op_can_be_supported = false;
  for (Tile *tile : architecture.getAllTiles()) {
    if (tile->canSupportOperation(
            mlir::neura::getOperationKindFromMlirOp(op))) {
      op_can_be_supported = true;
    }
  }
  if (!op_can_be_supported) {
    llvm::errs() << "[caculateSpatialAward] Operation: " << *op
                 << " is not supported by any tile.\n";
    return {};
  }

  // A map of locations with their associated award.
  std::map<MappingLoc, int> locs_with_award;

  // Assembles all the producers.
  std::vector<Operation *> producers;
  for (Value operand : op->getOperands()) {
    if (isa<neura::ReserveOp>(operand.getDefiningOp())) {
      // Skips Reserve ops (backward ctrl move) when calculating award.
      continue;
    }
    Operation *producer = getMaterializedProducer(operand);
    assert(producer && "Expected a materialized producer");
    producers.push_back(producer);
  }

  // Assembles all the backward users if exist.
  std::vector<Operation *> backward_users;
  for (Operation *user : getCtrlMovUsers(op)) {
    auto ctrl_mov = dyn_cast<neura::CtrlMovOp>(user);
    assert(ctrl_mov && "Expected user to be a CtrlMovOp");
    mlir::Operation *materialized_backward_op =
        getMaterializedBackwardUser(ctrl_mov);
    assert(isa<neura::PhiOp>(materialized_backward_op) &&
           "Expected materialized operation of ctrl_mov to be a PhiOp");
    backward_users.push_back(materialized_backward_op);
  }

  llvm::errs() << "[caculateSpatialAward] Operation: " << *op
               << "; Producers: " << producers.size() << "\n";

  // Tracks which tiles are already allocated to operations
  std::set<Tile *> occupied_tiles;
  for (const auto &entry : mapping_state.getLocToOp()) {
    if (entry.first.resource->getKind() == ResourceKind::Tile) {
      llvm::outs() << "[caculateSpatialAward] Tile: "
                   << entry.first.resource->getId()
                   << " is already occupied by operation: " << *(entry.second)
                   << "\n";
      occupied_tiles.insert(dyn_cast<Tile>(entry.first.resource));
    }
  }

  for (Tile *tile : architecture.getAllTiles()) {
    // Skip tiles that are already allocated to other operations
    if (occupied_tiles.count(tile)) {
      llvm::outs() << "[caculateSpatialAward] Tile: " << tile->getId()
                   << " is already occupied, skip it.\n";
      continue;
    }

    if (!tile->canSupportOperation(getOperationKindFromMlirOp(op))) {
      llvm::errs() << "[caculateSpatialAward] Tile: " << tile->getId()
                   << " does not support operation: " << *op << "\n";
      continue; // Skip tiles that cannot support the operation.
    }

    int earliest_start_time_step = target_level;

    for (Operation *producer : producers) {
      std::vector<MappingLoc> producer_locs =
          mapping_state.getAllLocsOfOp(producer);
      assert(!producer_locs.empty() && "No locations found for producer");

      MappingLoc producer_loc = producer_locs.back();
      earliest_start_time_step =
          std::max(earliest_start_time_step, producer_loc.time_step + 1);
    }

    int latest_end_time_step = earliest_start_time_step + mapping_state.getII();
    std::vector<MappingLoc> backward_users_locs;
    for (Operation *user : backward_users) {
      std::vector<MappingLoc> user_locs = mapping_state.getAllLocsOfOp(user);
      assert(!user_locs.empty() && "No locations found for backward user");

      MappingLoc backward_user_loc = user_locs.back();
      latest_end_time_step =
          std::min(latest_end_time_step,
                   backward_user_loc.time_step + mapping_state.getII());
      llvm::outs() << "[caculateSpatialAward] Latest end time step for " << *op
                   << " on tile: " << tile->getId()
                   << " is now: " << latest_end_time_step << "\n";
      backward_users_locs.push_back(backward_user_loc);
    }

    int award = 2 * mapping_state.getII();
    if (critical_ops.count(op)) {
      award += tile->getDstTiles().size();
      award += op->getOperands().size() -
               getPhysicalHops(producers, tile, mapping_state);
    }

    if (!backward_users.empty()) {
      llvm::outs() << "[caculateSpatialAward] earliest_start_time_step: "
                   << earliest_start_time_step
                   << ", latest_end_time_step: " << latest_end_time_step
                   << " for tile: " << tile->getType() << "#" << tile->getId()
                   << "\n";
    }

    for (int t = earliest_start_time_step; t < latest_end_time_step; t += 1) {
      MappingLoc tile_loc_candidate = {tile, t};

      // If the tile at time `t` is available, we can consider it for mapping.
      if (mapping_state.isAvailableAcrossTime(tile_loc_candidate)) {
        if (!backward_users.empty()) {
          llvm::outs() << "[caculateSpatialAward] Tile: " << tile->getType()
                       << "#" << tile->getId() << " at time step: " << t
                       << " is available for mapping operation: " << *op
                       << "\n";
        }

        bool meet_producer_constraint =
            producers.empty() ||
            canReachLocInTime(producers, tile_loc_candidate, t, mapping_state);
        if (!backward_users.empty()) {
          llvm::outs() << "[caculateSpatialAward] "
                       << "Meet producer constraint: "
                       << meet_producer_constraint << "\n";
        }
        bool meet_backward_user_constraint = true;
        for (auto &backward_user_loc : backward_users_locs) {
          // Check if the location can reach all backward users.
          if (!canReachLocInTime(tile_loc_candidate, backward_user_loc,
                                 backward_user_loc.time_step +
                                     mapping_state.getII(),
                                 mapping_state)) {
            meet_backward_user_constraint = false;
            break; // No need to check further.
          }
        }
        if (!backward_users.empty()) {
          llvm::outs() << "[caculateSpatialAward] "
                       << "Meet backward user constraint: "
                       << meet_backward_user_constraint << "\n";
        }

        if (meet_producer_constraint && meet_backward_user_constraint) {
          updateAward(locs_with_award, tile_loc_candidate, award);
        }
      }
      // The mapping location with earlier time step is granted with a higher
      // award.
      award -= 1;
    }
  }

  // Copies map entries into a vector of pairs for sorting.
  std::vector<std::pair<MappingLoc, int>> locs_award_vec(
      locs_with_award.begin(), locs_with_award.end());

  // Sorts by award (descending).
  std::sort(
      locs_award_vec.begin(), locs_award_vec.end(),
      [](const std::pair<MappingLoc, int> &a,
         const std::pair<MappingLoc, int> &b) { return a.second > b.second; });

  // Extracts just the MappingLocs, already sorted by award.
  std::vector<MappingLoc> sorted_locs;
  sorted_locs.reserve(locs_award_vec.size());
  for (const auto &pair : locs_award_vec)
    sorted_locs.push_back(pair.first);

  return sorted_locs;
}

} // namespace neura
} // namespace mlir
