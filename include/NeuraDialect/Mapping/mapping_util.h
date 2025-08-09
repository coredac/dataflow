#pragma once

#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/MappingState.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace neura {
// Returns the kind of operation from the MLIR operation.
OperationKind getOperationKindFromMlirOp(Operation *op);

// Returns true if the operation does not need CGRA tile placement.
bool is_non_materialized(Operation *op);

// Represents a recurrence cycle rooted at a reserve operation and closed by
// ctrl_mov.
struct RecurrenceCycle {
  SmallVector<Operation *>
      operations; // Ordered list of operations in the cycle.
  int length = 0; // Number of operations excluding reserve/ctrl_mov.
};

// Collects recurrence cycles rooted at reserve and closed by ctrl_mov.
SmallVector<RecurrenceCycle, 4> collectRecurrenceCycles(Operation *func_op);

// Calculates ResMII: ceil(#ops / #tiles).
int calculateResMii(Operation *func_op, const Architecture &architecture);

// Returns topologically sorted operations in func_op.
std::vector<Operation *> getTopologicallySortedOps(Operation *func_op);

// Given the sorted operations, returns a vector of pairs where each pair
// contains a vector of operations at the same ALAP (as late as possible)
// level and the level number.
std::vector<std::vector<Operation *>>
getOpsInAlapLevels(const std::vector<Operation *> &sorted_ops,
                   const std::set<Operation *> &critical_ops);

// Flattens the level buckets into a vector of pairs (operation, level).
std::vector<std::pair<Operation *, int>> flatten_level_buckets(
    const std::vector<std::vector<Operation *>> &level_buckets);

// Gets the physical hops from the producers to the tile, which is used for
// estimating the award of a location for placement.
int getPhysicalHops(const std::vector<Operation *> &producers, Tile *tile,
                    const MappingState &mapping_state);

Operation *getMaterializedProducer(Value operand);

// Collects the real users of an operation, excluding ctrl_mov and data_mov.
llvm::SmallVector<mlir::Operation *> getMaterializedUserOps(Operation *op);

// Gets the last materialized backward user of an operation, which is expected
// to be a phi operation.
Operation *getMaterializedBackwardUser(Operation *op);

// Attempts to route a data move operation from src_loc to dst_loc.
bool tryRouteDataMove(Operation *mov, MappingLoc src_loc, MappingLoc dst_loc,
                      bool is_backward_move, const MappingState &mapping_state,
                      std::vector<MappingLoc> &path_out);

bool tryRouteForwardMove(Operation *mov_op, MappingLoc src_loc,
                         MappingLoc dst_loc, const MappingState &state,
                         std::vector<MappingLoc> &path_out);

bool tryRouteBackwardMove(Operation *mov_op, MappingLoc src_loc,
                          MappingLoc dst_loc, const MappingState &state,
                          std::vector<MappingLoc> &path_out);

// Gets the ctrl_mov users of an operation, empty vector is returned if no
// ctrl_mov users found.
llvm::SmallVector<Operation *> getCtrlMovUsers(Operation *op);

// Maps a materialized operation to the accelerator, and routes the dataflow
// from the producers to the given op.
bool placeAndRoute(Operation *op, const MappingLoc &target_loc,
                   MappingState &mapping_state);

// Calculates the award of mapping locations for a given op, the returned
// locations are sorted based on the award.
std::vector<MappingLoc> calculateAward(Operation *op,
                                       std::set<Operation *> &critical_ops,
                                       int target_level,
                                       const Architecture &architecture,
                                       const MappingState &mapping_state);

void updateAward(std::map<MappingLoc, int> &locs_with_award, MappingLoc loc,
                 int award);

bool canReachLocInTime(const MappingLoc &src_loc, const MappingLoc &dst_loc,
                       int deadline_step, const MappingState &mapping_state);

bool canReachLocInTime(const std::vector<Operation *> &producers,
                       const MappingLoc &target_loc, int deadline_step,
                       const MappingState &mapping_state);

// Gets an available register (for the given time range) in the given tile.
// The end_time is exclusive, meaning the register should be available
// until end_time - 1. Returns nullptr if no available register found.
Register *getAvailableRegister(const MappingState &mapping_state, Tile *tile,
                               int start_time, int exclusive_end_time);

} // namespace neura
} // namespace mlir
