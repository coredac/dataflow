#pragma once

#include "mlir/IR/Operation.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/MappingState.h"

namespace mlir {
namespace neura {

// Represents a recurrence cycle rooted at a reserve operation and closed by ctrl_mov.
struct RecurrenceCycle {
  SmallVector<Operation *> operations;  // Ordered list of operations in the cycle.
  int length = 0;                       // Number of operations excluding reserve/ctrl_mov.
};

// Collects recurrence cycles rooted at reserve and closed by ctrl_mov.
SmallVector<RecurrenceCycle, 4> collectRecurrenceCycles(Operation *func_op);

// Calculates ResMII: ceil(#ops / #tiles).
int calculateResMii(Operation *func_op, const Architecture &architecture);

// Returns topologically sorted operations in func_op.
std::vector<Operation *> getTopologicallySortedOps(Operation *func_op);

Operation* getMaterializedProducer(Value operand);

// Collects the real users of an operation, excluding ctrl_mov and data_mov.
llvm::SmallVector<mlir::Operation *> getMaterializedUserOps(Operation *op);

// Gets the last materialized backward user of an operation, which is expected
// to be a phi operation.
Operation *getMaterializedBackwardUser(Operation *op);

// Attempts to map a function operation to the accelerator using heuristics.
bool tryHeuristicMapping(std::vector<Operation *> &sorted_ops,
                         const Architecture &architecture,
                         MappingState &mapping_state);

// Attempts to route a data move operation from src_loc to dst_loc.
bool tryRouteDataMove(Operation *mov,
                      MappingLoc src_loc,
                      MappingLoc dst_loc,
                      bool is_backward_move,
                      const MappingState &mapping_state,
                      std::vector<MappingLoc> &path_out);

bool tryRouteForwardMove(Operation *mov_op,
                         MappingLoc src_loc,
                         MappingLoc dst_loc,
                         const MappingState &state,
                         std::vector<MappingLoc> &path_out);

bool tryRouteBackwardMove(Operation *mov_op,
                           MappingLoc src_loc,
                           MappingLoc dst_loc,
                           const MappingState &state,
                           std::vector<MappingLoc> &path_out);

// Calculates the cost of mapping locations for a given op, the returned locations
// are sorted based on the cost.
std::vector<MappingLoc> calculateCost(Operation *op, const MappingState &mapping_state);

// Gets the ctrl_mov users of an operation, empty vector is returned if no ctrl_mov users found.
llvm::SmallVector<Operation *> getCtrlMovUsers(Operation *op);

// Maps a materialized operation to the accelerator, and routes the dataflow from
// the producers to the given op.
bool placeAndRoute(Operation *op, const MappingLoc &target_loc, MappingState &mapping_state);

std::vector<MappingLoc> calculateAward(Operation *op,
                                       const Architecture &architecture,
                                       const MappingState &mapping_state);

void updateAward(std::map<MappingLoc, int> &locs_with_award,
                 MappingLoc loc, int award);

bool canReachLocInTime(const MappingLoc &src_loc,
                       const MappingLoc &dst_loc,
                       int deadline_step,
                       const MappingState &mapping_state);

bool canReachLocInTime(const std::vector<Operation *> &producers,
                       const MappingLoc &target_loc,
                       int deadline_step,
                       const MappingState &mapping_state);

} // namespace neura
} // namespace mlir
