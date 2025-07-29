#include "NeuraDialect/Mapping/HeuristicMapping/HeuristicMapping.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraOps.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace neura {

bool HeuristicMapping::map(std::vector<std::pair<Operation *, int>> &sorted_ops_with_alap_levels,
                           std::set<Operation *> &critical_ops,
                           const Architecture &architecture,
                           MappingState &mapping_state) {
  // Start the backtracking mapping process from the first operation.
  return mapWithBacktrack(sorted_ops_with_alap_levels,
                          critical_ops,
                          architecture,
                          mapping_state,
                          0,
                          0);
}

bool HeuristicMapping::mapWithBacktrack(
    std::vector<std::pair<Operation *, int>> &sorted_ops_with_alap_levels,
    std::set<Operation *> &critical_ops,
    const Architecture &architecture,
    MappingState &mapping_state,
    size_t op_index, int backtrack_depth) {
  // Checks if the backtrack depth exceeds the maximum allowed.
  if (backtrack_depth > this->max_backtrack_depth) {
    llvm::errs() << "[BacktrackMapping] Max backtrack depth reached\n";
    return false; // Backtrack failed, max depth reached.
  }

  // Success condition: all operations are mapped (The op_index is larger than
  // or equal to the number of operations).
  if (op_index >= sorted_ops_with_alap_levels.size()) {
    llvm::errs() << "[BacktrackMapping] Successfully mapped all operations.\n";
    return true;
  }

  // Gets the current operation with expected ALAP level (time step) to map.
  Operation *target_op = sorted_ops_with_alap_levels[op_index].first;
  int target_level = sorted_ops_with_alap_levels[op_index].second;

  // Skips non-materialized operations.
  if (isa<neura::DataMovOp, neura::CtrlMovOp, neura::ReserveOp>(target_op)) {
    // Increments the op_index to skip non-materialized operations.
    return mapWithBacktrack(sorted_ops_with_alap_levels,
                            critical_ops,
                            architecture,
                            mapping_state,
                            op_index + 1,
                            backtrack_depth);
  }

  // Gets candidate locations sorted by award.
  std::vector<MappingLoc> loc_candidates =
      calculateAward(target_op, critical_ops, target_level, architecture, mapping_state);

  if (loc_candidates.empty()) {
    llvm::errs() << "No locations found for op: " << *target_op << "\n";
    return false; // No locations available for this operation.
  }
  assert(!loc_candidates.empty() && "No locations found for the operation to map");

  // Limits the number of locations to try.
  int locations_to_try =
      std::min(static_cast<int>(loc_candidates.size()), this->max_location_to_try);

  // Tries each candicate location in order of decreasing award.
  for (int i = 0; i < locations_to_try; ++i) {
    MappingLoc target_loc = loc_candidates[i];
    // Creates a mapping snapshot of current state before attempting to map.
    MappingStateSnapshot mappingstate_snapshot(mapping_state);

    // Attempts to place and route the operation at the target location.
    if (placeAndRoute(target_op, target_loc, mapping_state)) {
      // Successfully placed and routed current operation, tries to map the next
      // operation.
      if (mapWithBacktrack(sorted_ops_with_alap_levels,
                           critical_ops,
                           architecture,
                           mapping_state,
                           op_index + 1,
                           backtrack_depth)) {
        return true; // Successfully mapped all operations.
      }

      // Failed to place next operation, restores the mapping state and try next
      // location.
      llvm::errs() << "[BACKTRACK] Failed to map in current location, "
                   << "restoring mapping state and trying next location.\n";
      llvm::errs() << "[BACKTRACK] Backtracking from op: " << *target_op << "\n";
      mappingstate_snapshot.restore(mapping_state);
      // Increments backtrack depth.
      backtrack_depth++;
    }
  }

  // All candidate locations failed.
  return false;
}

} // namespace neura
} // namespace mlir

namespace mlir {
namespace neura {
MappingStateSnapshot::MappingStateSnapshot(const MappingState &mapping_state) {
  this->occupied_locs = mapping_state.getOccupiedLocs();
  this->loc_to_op = mapping_state.getLocToOp();
  this->op_to_locs = mapping_state.getOpToLocs();
}

void MappingStateSnapshot::restore(MappingState &mapping_state) {
  mapping_state.setOccupiedLocs(this->occupied_locs);
  mapping_state.setLocToOp(this->loc_to_op);
  mapping_state.setOpToLocs(this->op_to_locs);
}
} // namespace neura
} // namespace mlir