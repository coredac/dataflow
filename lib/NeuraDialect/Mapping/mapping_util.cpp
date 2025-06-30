#include <deque>
#include <queue>

#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cassert>
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::neura;

namespace {

// Traverses (backward) the operation graph starting from the given operation
// towards reserve_value.
void traverseAlongPath(Operation *op, Value reserve_value,
                       std::deque<Operation *> &current_path,
                       DenseSet<Operation *> &visited_in_path,
                       SmallVector<RecurrenceCycle, 4> &collected_paths) {
  if (!op || visited_in_path.contains(op))
    return;

  visited_in_path.insert(op);
  current_path.push_front(op);

  for (Value operand : op->getOperands()) {
    if (operand == reserve_value) {
      Operation *res_op = reserve_value.getDefiningOp();
      if (res_op) current_path.push_front(res_op);

      int effective_length = 0;
      for (Operation *op : current_path) {
        // Skips the non-materialized ops when counting the cycle length.
        if (!isa<neura::ReserveOp,
                 neura::CtrlMovOp,
                 neura::DataMovOp>(op)) {
          ++effective_length;
        }
      }
      collected_paths.push_back(RecurrenceCycle{
        operations: SmallVector<Operation *>(current_path.begin(), current_path.end()),
        length: static_cast<int>(effective_length)
      });

      if (res_op) current_path.pop_front();
      continue;
    }

    if (Operation *def_op = operand.getDefiningOp()) {
      traverseAlongPath(def_op, reserve_value, current_path, visited_in_path, collected_paths);
    }
  }

  current_path.pop_front();
  visited_in_path.erase(op);
}

} // namespace

SmallVector<RecurrenceCycle, 4> mlir::neura::collectRecurrenceCycles(Operation *func_op) {
  SmallVector<RecurrenceCycle, 4> recurrence_cycles;

  func_op->walk([&](neura::CtrlMovOp ctrl_mov_op) {
    Value target = ctrl_mov_op.getTarget();
    auto reserve_op = target.getDefiningOp<neura::ReserveOp>();
    if (!reserve_op)
      return;

    Value reserve_value = reserve_op.getResult();
    Value ctrl_mov_from = ctrl_mov_op.getValue();

    Operation *parent_op = ctrl_mov_from.getDefiningOp();
    if (!parent_op)
      return;

    std::deque<Operation *> current_path;
    SmallVector<RecurrenceCycle, 4> collected_paths;
    DenseSet<Operation *> visited_in_path;
    traverseAlongPath(parent_op, reserve_value, current_path, visited_in_path, collected_paths);

    for (auto &cycle : collected_paths) {
      cycle.operations.push_back(ctrl_mov_op);
      recurrence_cycles.push_back(std::move(cycle));
    }
  });

  return recurrence_cycles;
}

int mlir::neura::calculateResMii(Operation *func_op,
                                 const Architecture &architecture) {
  int num_ops = 0;

  // Count all "compute" operations (non-terminators, non-block ops).
  func_op->walk([&](Operation *op) {
    // Skips non-materialized ops.
    if (isa<func::FuncOp>(op) ||
        isa<neura::ConstantOp,
            neura::CtrlMovOp,
            neura::DataMovOp,
            neura::ReserveOp>(op)) {
      return;
    }
    ++num_ops;
  });

  llvm::errs() << "[calculateResMii] Total operations: " << num_ops << "\n";

  // Avoid divide-by-zero
  int num_tiles = std::max(1, architecture.getNumTiles());

  return llvm::divideCeil(num_ops, num_tiles);
}

std::vector<Operation *> mlir::neura::getTopologicallySortedOps(Operation *func_op) {
  std::vector<Operation *> sorted_ops;
  llvm::DenseMap<Operation *, int> pending_deps;
  std::deque<Operation *> ready_queue;

  // Collects recurrence cycle ops.
  auto recurrence_cycles = collectRecurrenceCycles(func_op);
  llvm::DenseSet<Operation *> recurrence_ops;
  for (const auto &cycle : recurrence_cycles)
    for (Operation *op : cycle.operations)
      recurrence_ops.insert(op);

  // Counts unresolved dependencies for each op.
  func_op->walk([&](Operation *op) {
    if (op == func_op) return;
    int dep_count = 0;
    for (Value operand : op->getOperands())
      if (operand.getDefiningOp())
        ++dep_count;
    pending_deps[op] = dep_count;
    if (dep_count == 0) {
      // TODO: Prioritize recurrence ops. But cause compiled II regression.
      // https://github.com/coredac/dataflow/issues/59.
      if (recurrence_ops.contains(op)) {
        // ready_queue.push_front(op);
        ready_queue.push_back(op);
      } else {
        ready_queue.push_back(op);
      }
    }
  });

  // BFS-style topological sort with recurrence priority.
  while (!ready_queue.empty()) {
    Operation *op = ready_queue.front();
    ready_queue.pop_front();
    sorted_ops.push_back(op);

    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (--pending_deps[user] == 0) {
          // TODO: Prioritize recurrence ops. But cause compiled II regression.
          // https://github.com/coredac/dataflow/issues/59.
          if (recurrence_ops.contains(user)) {
            // ready_queue.push_front(user);
            ready_queue.push_back(user);
          } else {
            ready_queue.push_back(user);
          }
        }
      }
    }
  }

  return sorted_ops;
}

mlir::Operation *mlir::neura::getMaterializedBackwardUser(Operation *op) {
  assert(isa<neura::CtrlMovOp>(op) && "Expected a ctrl_mov operation");
  auto ctrl_mov = dyn_cast<neura::CtrlMovOp>(op);
  Value target = ctrl_mov.getTarget();

  assert(isa<neura::ReserveOp>(target.getDefiningOp()) &&
         "Expected the user of ctrl_mov target to be a reserve operation");
  auto reserve_op = dyn_cast<neura::ReserveOp>(target.getDefiningOp());

  // Skip ctrl_mov users of reserve; return the first phi user.
  for (Operation *user : reserve_op.getResult().getUsers()) {
    if (isa<neura::CtrlMovOp>(user)) continue; // skip ctrl_mov user
    if (isa<neura::PhiOp>(user)) return user;
  }
  assert(false && "No materialized backward user (i.e., phi) found for ctrl_mov");
}

llvm::SmallVector<mlir::Operation *> mlir::neura::getMaterializedUserOps(Operation *op) {
  llvm::SmallVector<Operation *> result;
  llvm::DenseSet<Operation *> visited;
  visited.insert(op);
  llvm::errs() << "Starting to collect materialized users for: " << *op << "\n";
  llvm::SmallVector<Operation *> worklist(op->getUsers().begin(), op->getUsers().end());

  while (!worklist.empty()) {
    Operation *curr = worklist.pop_back_val();
    llvm::errs() << "Visiting operation: " << *curr << "\n";
    if (!visited.insert(curr).second) {
      llvm::errs() << "Already visited, so skip: " << *curr << "\n";
      continue;
    }

    if (isa<neura::DataMovOp>(curr)) {
      for (Operation *next : curr->getUsers()) {
        if (visited.insert(next).second) {
          // Only adds the next operation if it hasn't been visited yet.
          worklist.push_back(next);
        }
      }
      continue;
    }

    // Specially handles the ctrl_mov, i.e., the second operand of ctrl_mov is
    // treated as a target/destination/user in terms of dataflow.
    if (auto ctrl_mov = dyn_cast<neura::CtrlMovOp>(curr)) {
      Value target = ctrl_mov.getTarget();
      for (Operation *user : target.getUsers()) {
        if (visited.insert(user).second) {
          worklist.push_back(user);
        }
      }
      continue;
    }

    // Materialized op
    result.push_back(curr);
  }

  for (Operation *res : result) {
    llvm::errs() << "Materialized user: " << *res << "\n";
  }
  return result;
}

bool mlir::neura::tryRouteForwardMove(Operation *mov_op,
                                      MappingLoc src_loc,
                                      MappingLoc dst_loc,
                                      const MappingState &state,
                                      std::vector<MappingLoc> &path_out) {
  return tryRouteDataMove(mov_op, src_loc, dst_loc, false, state, path_out);
}

bool mlir::neura::tryRouteBackwardMove(Operation *mov_op,
                                       MappingLoc src_loc,
                                       MappingLoc dst_loc,
                                       const MappingState &state,
                                       std::vector<MappingLoc> &path_out) {
  llvm::errs() << "[tryRouteBackwardMove] src_loc: " << src_loc.resource->getType()
            << "#" << src_loc.resource->getId()
            << " @t=" << src_loc.time_step
            << ", dst_loc: " << dst_loc.resource->getType()
            << "#" << dst_loc.resource->getId()
            << " @t=" << dst_loc.time_step << "\n";
  return tryRouteDataMove(mov_op, src_loc, dst_loc, true, state, path_out);
}

bool mlir::neura::tryRouteDataMove(Operation *mov_op,
                                   MappingLoc src_loc,
                                   MappingLoc dst_loc,
                                   bool is_backward_move,
                                   const MappingState &state,
                                   std::vector<MappingLoc> &path_out) {
  // Specially handles the case where src and dst are the same tile.
  if (src_loc.resource == dst_loc.resource) {
    return true;
  }
  struct QueueEntry {
    Tile *tile;
    int time;
    std::vector<MappingLoc> path;
  };

  Tile *src_tile = dyn_cast<Tile>(src_loc.resource);
  Tile *dst_tile = dyn_cast<Tile>(dst_loc.resource);

  std::queue<QueueEntry> queue;
  std::set<Tile*> visited;

  queue.push({src_tile, src_loc.time_step, {}});
  visited.insert(src_tile);

  // Tolerates the deadline step by II for backward moves (as the data should
  // arrive at the next iteration).
  const int deadline_step = dst_loc.time_step + (is_backward_move ? state.getII() : 0);

  // BFS-style search for a path from src_tile to dst_tile.
  while (!queue.empty()) {
    auto [current_tile, current_time, current_path] = queue.front();
    queue.pop();

    if (current_tile == dst_tile) {
      // Confirms path reaches the target tile no later than deadline step.
      if (current_time <= deadline_step) {
        // Either arrives exactly right before the dst starts computation.
        // So the current_time on the target tile is the same as deadline step.
        if (current_time == deadline_step) {
          path_out = current_path;
          return true;
        }

        // The last link can be held from arrival_time to dst_time - 1.
        // TODO: We actually don't need to occupy the last link if the registers
        // within the tile can be explicitly represented.
        // https://github.com/coredac/dataflow/issues/52.
        bool all_free = true;
        assert(!current_path.empty() && "Path should not be empty when checking last link");
        MappingLoc last_link = current_path.back();
        std::vector<MappingLoc> last_link_occupying;
        for (int t = current_time; t < deadline_step; ++t) {
          MappingLoc repeated{last_link.resource, t};
          last_link_occupying.push_back(repeated);
          if (!state.isAvailableAcrossTime(repeated)) {
            all_free = false;
            break;
          }
        }
        if (all_free) {
          path_out = current_path;
          path_out.insert(path_out.end(), last_link_occupying.begin(), last_link_occupying.end());
          return true;
        }

      } else {
        // Arrives too late, not schedulable.
        continue;
      }
    }

    for (MappingLoc current_step_next_link : state.getCurrentStepLinks({current_tile, current_time})) {
      if (!state.isAvailableAcrossTime(current_step_next_link)) continue;

      Link *next_link = dyn_cast<Link>(current_step_next_link.resource);
      Tile *next_tile = next_link->getDstTile();
      int next_time = current_time + 1;

      if (!visited.insert(next_tile).second) continue;

      std::vector<MappingLoc> extended_path = current_path;
      extended_path.push_back(current_step_next_link);
      queue.push({next_tile, next_time, std::move(extended_path)});
    }
  }

  return false;
}

Operation* mlir::neura::getMaterializedProducer(Value operand) {
  Operation *producer = operand.getDefiningOp();
  assert(isa<neura::DataMovOp>(producer) && "Expected operand to be defined by a DataMovOp");
  // Finds the actual producer.
  auto mov_op = dyn_cast<neura::DataMovOp>(producer);
  auto materialized_producer = mov_op.getOperand().getDefiningOp();
  return materialized_producer;
}

bool mlir::neura::tryHeuristicMapping(std::vector<Operation *> &sorted_ops,
                                      const Architecture &architecture,
                                      MappingState &mapping_state) {
  DenseSet<Operation *> visited;

  for (Operation *op : sorted_ops) {
    // TODO: Build up util func to distinguish materialized and non-materialized ops.
    if (isa<neura::DataMovOp, neura::CtrlMovOp, neura::ReserveOp>(op))
      continue;

    std::vector<MappingLoc> sorted_locs = calculateAward(op, architecture, mapping_state);
    // auto target_loc = getLocWithMinCost(loc_with_cost);
    if (sorted_locs.empty()) {
      llvm::errs() << "[DEBUG] No locations found for op: " << *op << "\n";
      return false; // No locations available for this operation.
    }
    assert(!sorted_locs.empty() &&
           "No locations found for the operation to map");
    MappingLoc target_loc = sorted_locs.front();
    if (placeAndRoute(op, target_loc, mapping_state)) {
      llvm::errs() << "[DEBUG] Successfully scheduled op: " << *op
                   << " at loc: " << target_loc.resource->getType()
                   << "#" << target_loc.resource->getId()
                   << " @t=" << target_loc.time_step << "\n";
      continue;
    } else {
      llvm::errs() << "[DEBUG] Failed to schedule op: " << *op << "; target loc: " << target_loc.resource->getType() << "#" << target_loc.resource->getId() << " @t=" << target_loc.time_step << "\n";
    }
    // TODO: Optimization -- backtrack a few times if failed to schedule the op.
    // https://github.com/coredac/dataflow/issues/59
    return false;
  }

  return true;
}

bool mlir::neura::canReachLocInTime(const std::vector<Operation *> &producers,
                                    const MappingLoc &target_loc,
                                    int deadline_step,
                                    const MappingState &mapping_state) {

  for (Operation *producer : producers) {
    // Get the last location of the producer.
    auto producer_locs = mapping_state.getAllLocsOfOp(producer);
    assert(!producer_locs.empty() && "No locations found for producer");

    MappingLoc producer_loc = producer_locs.back();
    if (!canReachLocInTime(producer_loc, target_loc, deadline_step, mapping_state)) {
      return false;
    }
  }
  return true;
}

bool mlir::neura::canReachLocInTime(const MappingLoc &src_loc,
                                    const MappingLoc &dst_loc,
                                    int deadline_step,
                                    const MappingState &mapping_state) {
  // Checks if the destination is reachable from the source within the given time window.
  if (src_loc.resource == dst_loc.resource &&
      dst_loc.time_step <= deadline_step) {
    return true;
  }

  // Checks if the destination is reachable from the source tile within given steps.
  assert(isa<Tile>(src_loc.resource));
  assert(isa<Tile>(dst_loc.resource));

  struct QueueEntry {
    MappingLoc loc;
    int current_time;
  };

  std::queue<QueueEntry> queue;
  llvm::DenseSet<Tile *> visited;

  queue.push({src_loc, src_loc.time_step});
  visited.insert(dyn_cast<Tile>(src_loc.resource));

  while (!queue.empty()) {
    auto [current_loc, current_time] = queue.front();
    queue.pop();

    // If we reach the destination tile and time step is not after dst_loc
    if (current_loc.resource == dst_loc.resource &&
        current_time <= dst_loc.time_step &&
        dst_loc.time_step <= deadline_step) {
      return true;
    }

    if (current_time >= deadline_step)
      continue;

    // Explores all next step tiles from the current location.
    for (const MappingLoc &next_loc : mapping_state.getNextStepTiles(current_loc)) {
      if (!mapping_state.isAvailableAcrossTime(next_loc))
        continue;

      int next_time = current_time + 1;
      if (next_time > deadline_step)
        continue;

      Tile *next_tile = llvm::dyn_cast<Tile>(next_loc.resource);
      assert(next_tile && "Next location must be a Tile");
      if (visited.contains(next_tile)) {
        continue;
      }

      visited.insert(next_tile);

      MappingLoc next_step_loc = next_loc;
      next_step_loc.time_step = next_time;

      queue.push({next_step_loc, next_time});
    }
  }

  return false;
}

void mlir::neura::updateAward(std::map<MappingLoc, int> &locs_with_award,
                              MappingLoc loc, int award) {
  // Updates the award of the top element in the priority queue.
  if (locs_with_award.find(loc) != locs_with_award.end()) {
    locs_with_award[loc] += award;
  } else {
    locs_with_award[loc] = award;
  }
}

std::vector<MappingLoc> mlir::neura::calculateAward(Operation *op,
                                                    const Architecture &architecture,
                                                    const MappingState &mapping_state) {
  // A heap of locations with their associated award. Note that we use a max-heap
  // to prioritize locations with higher awards.
  std::map<MappingLoc, int> locs_with_award;

  // Assembles all the producers.
  std::vector<Operation *> producers;
  for (Value operand : op->getOperands()) {
    if (isa<neura::ReserveOp>(operand.getDefiningOp())) {
      // Skips Reserve ops (backward ctrl move) when estimate cost.
      continue;
    }
    Operation *producer = getMaterializedProducer(operand);
    assert(producer && "Expected a materialized producer");
    producers.push_back(producer);
  }

  llvm::errs() << "[calculateAward] Operation: " << *op
             << "; Producers: " << producers.size() << "\n";
  for (Tile *tile : architecture.getAllTiles()) {
    int earliest_start_time_step = 0;
    for (Operation *producer : producers) {
      std::vector<MappingLoc> producer_locs = mapping_state.getAllLocsOfOp(producer);
      assert(!producer_locs.empty() && "No locations found for producer");

      MappingLoc producer_loc = producer_locs.back();
      earliest_start_time_step = std::max(earliest_start_time_step,
                                          producer_loc.time_step + 1);
    }
    int award = mapping_state.getII() + tile->getDstTiles().size();
    for (int t = earliest_start_time_step;
         t < earliest_start_time_step + mapping_state.getII(); t += 1) {
      MappingLoc tile_loc_candidate = {tile, t};
      // If the tile at time `t` is available, we can consider it for mapping.
      if (mapping_state.isAvailableAcrossTime(tile_loc_candidate)) {
        // If no producer or the location is reachable by all producers,
        // we can consider it for mapping and grant reward.
        if (producers.empty() ||
            canReachLocInTime(producers,
                                  tile_loc_candidate,
                                  t,
                                  mapping_state)) {
          updateAward(locs_with_award, tile_loc_candidate, award);
        }
      }
      // The mapping location with earlier time step is granted with a higher award.
      award -= 1;
    }
    assert(award >= 0 && "Award should not be negative");
  }

  // Copies map entries into a vector of pairs for sorting.
  std::vector<std::pair<MappingLoc, int>> locs_award_vec(locs_with_award.begin(), locs_with_award.end());

  // Sorts by award (descending).
  std::sort(locs_award_vec.begin(), locs_award_vec.end(),
            [](const std::pair<MappingLoc, int> &a, const std::pair<MappingLoc, int> &b) {
              return a.second > b.second;
            });
  // TODO: Needs to handle tie case and prioritize lower resource utilization, however,
  // compiled II becomes worse after adding this tie-breaker: https://github.com/coredac/dataflow/issues/59.
  // std::sort(locs_award_vec.begin(), locs_award_vec.end(),
  //           [&](const std::pair<MappingLoc, int> &a, const std::pair<MappingLoc, int> &b) {
  //               if (a.second != b.second) {
  //                 return a.second > b.second;
  //               }
  //               // Tie-breaker: prioritizes lower resource utilization and earlier time step.
  //               if (a.first.time_step != b.first.time_step) {
  //                 return a.first.time_step > b.first.time_step;
  //               }
  //               const bool is_resource_a_lower_utilized =
  //                   mapping_state.countOpsAtResource(a.first.resource) >
  //                   mapping_state.countOpsAtResource(b.first.resource);
  //               return is_resource_a_lower_utilized;
  //             });

  // Extracts just the MappingLocs, already sorted by award.
  std::vector<MappingLoc> sorted_locs;
  sorted_locs.reserve(locs_award_vec.size());
  for (const auto &pair : locs_award_vec)
    sorted_locs.push_back(pair.first);

  return sorted_locs;
}

llvm::SmallVector<Operation *> mlir::neura::getCtrlMovUsers(Operation *op) {
  llvm::SmallVector<Operation *> result;
  for (Operation *user : op->getUsers()) {
    if (isa<neura::CtrlMovOp>(user)) {
      result.push_back(user);
    }
  }
  return result;
}

bool mlir::neura::placeAndRoute(Operation *op, const MappingLoc &target_loc, MappingState &mapping_state) {
  if (mapping_state.bindOp(target_loc, op)) {
    // Tries to route the data move operations.
    for (Value operand : op->getOperands()) {
      if (isa<neura::ReserveOp>(operand.getDefiningOp())) {
        // Skips Reserve ops (backward ctrl move) when estimate cost.
        continue;
      }
      Operation *data_move = operand.getDefiningOp();
      assert(isa<neura::DataMovOp>(data_move) && "Expected a DataMovOp as operand producer");
      Operation *producer = getMaterializedProducer(operand);
      MappingLoc src_loc = mapping_state.getAllLocsOfOp(producer).back();

      std::vector<MappingLoc> route_path;
      if (tryRouteForwardMove(data_move, src_loc, target_loc, mapping_state, route_path)) {
        mapping_state.reserveRoute(data_move, route_path);
        llvm::errs() << "[DEBUG] Successfully routed data move: " << *data_move
                     << " from " << src_loc.resource->getType() << "#" << src_loc.resource->getId()
                     << " @t=" << src_loc.time_step
                     << " to " << target_loc.resource->getType() << "#" << target_loc.resource->getId()
                     << " @t=" << target_loc.time_step << "\n";
        continue;
      }
      llvm::errs() << "[DEBUG] Failed to route data move: " << *data_move
                   << " from " << src_loc.resource->getType() << "#" << src_loc.resource->getId()
                   << " @t=" << src_loc.time_step
                   << " to " << target_loc.resource->getType() << "#" << target_loc.resource->getId()
                   << " @t=" << target_loc.time_step << "\n";
      mapping_state.unbindOp(op);
      mapping_state.releaseRoute(data_move);
      return false;
    }
    // Checks whether the operation's user is a ctrl_mov.
    for (Operation *user : getCtrlMovUsers(op)) {
      auto ctrl_mov = dyn_cast<neura::CtrlMovOp>(user);
      llvm::errs() << "[DEBUG] Found ctrl_mov user: " << *ctrl_mov << "\n";
      assert(ctrl_mov && "Expected user to be a CtrlMovOp");
      mlir::Operation *materialized_backward_op = getMaterializedBackwardUser(ctrl_mov);
      assert(isa<neura::PhiOp>(materialized_backward_op) &&
             "Expected materialized operation of ctrl_mov to be a PhiOp");
      // Gets the last location of the materialized operation.
      MappingLoc backward_loc = mapping_state.getAllLocsOfOp(materialized_backward_op).back();
      // Routes the ctrl_mov to the phi location.
      std::vector<MappingLoc> route_path;
      if (tryRouteBackwardMove(ctrl_mov, target_loc, backward_loc, mapping_state, route_path)) {
        mapping_state.reserveRoute(ctrl_mov, route_path);
        llvm::errs() << "[DEBUG] Successfully routed ctrl_mov: " << *ctrl_mov
                     << " to " << backward_loc.resource->getType() << "#" << backward_loc.resource->getId()
                     << " @t=" << backward_loc.time_step << "\n";
        continue;
      }
      llvm::errs() << "[DEBUG] Failed to route ctrl_mov: " << *ctrl_mov
                   << " to " << backward_loc.resource->getType() << "#" << backward_loc.resource->getId()
                   << " @t=" << backward_loc.time_step << "\n";
      mapping_state.unbindOp(op);
      mapping_state.releaseRoute(ctrl_mov);
      return false;
    }
    return true;
  }
  return false;
}