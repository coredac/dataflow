#include <deque>
#include <queue>

#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace mlir;
using namespace mlir::neura;

namespace mlir {
namespace neura {
OperationKind getOperationKindFromMlirOp(Operation *op) {
  // Integer arithmetic operations
  if (isa<neura::AddOp>(op)) return IAdd;
  if (isa<neura::SubOp>(op)) return ISub;
  if (isa<neura::MulOp>(op)) return IMul;
  if (isa<neura::DivOp>(op)) return IDiv;
  if (isa<neura::RemOp>(op)) return IRem;
  
  // Floating-point arithmetic operations
  if (isa<neura::FAddOp>(op)) return FAdd;
  if (isa<neura::FSubOp>(op)) return FSub;
  if (isa<neura::FMulOp>(op)) return FMul;
  if (isa<neura::FDivOp>(op)) return FDiv;
  
  // Memory operations
  if (isa<neura::LoadOp>(op)) return ILoad;
  if (isa<neura::StoreOp>(op)) return IStore;
  if (isa<neura::LoadIndexedOp>(op)) return ILoadIndexed;
  if (isa<neura::StoreIndexedOp>(op)) return IStoreIndexed;
  if (isa<neura::AllocaOp>(op)) return IAlloca;
  
  // Logical operations
  if (isa<neura::OrOp>(op)) return IOr;
  if (isa<neura::NotOp>(op)) return INot;
  if (isa<neura::ICmpOp>(op)) return ICmp;
  if (isa<neura::FCmpOp>(op)) return FCmp;
  if (isa<neura::SelOp>(op)) return ISel;
  
  // Type conversion operations
  if (isa<neura::CastOp>(op)) return ICast;
  if (isa<neura::SExtOp>(op)) return ISExt;
  if (isa<neura::ZExtOp>(op)) return IZExt;
  if (isa<neura::ShlOp>(op)) return IShl;
  
  // Vector operations
  if (isa<neura::VFMulOp>(op)) return VFMul;
  
  // Fused operations
  if (isa<neura::FAddFAddOp>(op)) return FAddFAdd;
  if (isa<neura::FMulFAddOp>(op)) return FMulFAdd;
  
  // Steering control fused operations
  if (isa<neura::CarryInvariantOp>(op)) return ICarryInvariant;
  if (isa<neura::ConditionalSelectOp>(op)) return IConditionalSelect;
  if (isa<neura::InvariantGroupOp>(op)) return IInvariantGroup;
  
  // Control flow operations
  if (isa<neura::ReturnOp>(op)) return IReturn;
  if (isa<neura::PhiOp>(op)) return IPhi;
  
  // Data movement operations
  if (isa<neura::DataMovOp>(op)) return IDataMov;
  if (isa<neura::CtrlMovOp>(op)) return ICtrlMov;
  
  // Predicate operations
  if (isa<neura::ReserveOp>(op)) return IReserve;
  if (isa<neura::GrantPredicateOp>(op)) return IGrantPredicate;
  if (isa<neura::GrantOnceOp>(op)) return IGrantOnce;
  if (isa<neura::GrantAlwaysOp>(op)) return IGrantAlways;
  
  // Loop control operations
  if (isa<neura::LoopControlOp>(op)) return ILoopControl;
  
  // Constant operations
  if (isa<neura::ConstantOp>(op)) return IConstant;
  
  // Default fallback
  return IAdd;
}

// Returns true if the operation does not need CGRA tile placement.
bool is_non_materialized(Operation *op) {
  // Returns true if the operation does not need CGRA tile placement.
  return mlir::isa<neura::ReserveOp, neura::CtrlMovOp, neura::DataMovOp>(op);
}

} // namespace neura
} // namespace mlir

namespace {
// Traverses (backward) the operation graph starting from the given operation
// towards reserve_value.
void traverseAlongPath(Operation *op, Value reserve_value,
                       std::deque<Operation *> &current_path,
                       DenseSet<Operation *> &visited_in_path,
                       SmallVector<RecurrenceCycle, 4> &collected_paths) {
  if (!op || visited_in_path.contains(op)) {
    if (visited_in_path.contains(op)) {
      llvm::errs() << "Skipping already visited operation: " << *op << "\n";
    }
    return;
  }
  visited_in_path.insert(op);
  current_path.push_front(op);

  for (Value operand : op->getOperands()) {
    if (operand == reserve_value) {
      Operation *res_op = reserve_value.getDefiningOp();
      if (res_op) {
        current_path.push_front(res_op);
      }

      int effective_length = 0;
      for (Operation *op : current_path) {
        // Skips the non-materialized ops when counting the cycle length.
        if (!is_non_materialized(op)) {
          ++effective_length;
        }
      }
      collected_paths.push_back(
          RecurrenceCycle{/* operations = */ SmallVector<Operation *>(
                              current_path.begin(), current_path.end()),
                          /* length = */ static_cast<int>(effective_length)});

      if (res_op) {
        current_path.pop_front();
      }
      continue;
    }

    if (Operation *def_op = operand.getDefiningOp()) {
      traverseAlongPath(def_op, reserve_value, current_path, visited_in_path,
                        collected_paths);
    }
  }

  current_path.pop_front();
  visited_in_path.erase(op);
}

} // namespace

SmallVector<RecurrenceCycle, 4>
mlir::neura::collectRecurrenceCycles(Operation *func_op) {
  SmallVector<RecurrenceCycle, 4> recurrence_cycles;

  func_op->walk([&](neura::CtrlMovOp ctrl_mov_op) {
    Value target = ctrl_mov_op.getTarget();
    auto reserve_op = target.getDefiningOp<neura::ReserveOp>();
    if (!reserve_op) {
      return;
    }

    Value reserve_value = reserve_op.getResult();
    Value ctrl_mov_from = ctrl_mov_op.getValue();

    Operation *parent_op = ctrl_mov_from.getDefiningOp();
    if (!parent_op) {
      return;
    }

    std::deque<Operation *> current_path;
    SmallVector<RecurrenceCycle, 4> collected_paths;
    DenseSet<Operation *> visited_in_path;
    llvm::errs() << "Collecting recurrence cycles from back edge: parent_op "
                 << *parent_op << "->" << reserve_op << "\n";
    traverseAlongPath(parent_op, reserve_value, current_path, visited_in_path,
                      collected_paths);

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
        isa<neura::CtrlMovOp, neura::DataMovOp, neura::ReserveOp>(op)) {
      return;
    }
    ++num_ops;
  });

  llvm::errs() << "[calculateResMii] Total operations: " << num_ops << "\n";

  // Avoid divide-by-zero
  int num_tiles = std::max(1, architecture.getNumTiles());

  return llvm::divideCeil(num_ops, num_tiles);
}

std::vector<Operation *>
mlir::neura::getTopologicallySortedOps(Operation *func_op) {
  std::vector<Operation *> sorted_ops;
  llvm::DenseMap<Operation *, int> pending_deps;
  std::deque<Operation *> ready_queue;

  // Collects recurrence cycle ops.
  auto recurrence_cycles = collectRecurrenceCycles(func_op);
  llvm::DenseSet<Operation *> recurrence_ops;
  for (const auto &cycle : recurrence_cycles) {
    for (Operation *op : cycle.operations) {
      recurrence_ops.insert(op);
    }
  }
  // Counts unresolved dependencies for each op.
  func_op->walk([&](Operation *op) {
    if (op == func_op) {
      return;
    }
    int dep_count = 0;
    for (Value operand : op->getOperands()) {
      if (operand.getDefiningOp()) {
        ++dep_count;
      }
    }
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

std::vector<std::vector<Operation *>>
mlir::neura::getOpsInAlapLevels(const std::vector<Operation *> &sorted_ops,
                                const std::set<Operation *> &critical_ops) {
  llvm::DenseMap<Operation *, int> op_level;
  int max_level = 0;

  // Step 1: Computes raw ALAP level: longest path to any sink.
  for (auto it = sorted_ops.rbegin(); it != sorted_ops.rend(); ++it) {
    Operation *op = *it;

    int level = 0;
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (!op_level.count(user)) {
          continue;
        }

        int user_level = op_level[user];

        // Increments level only for materialized ops.
        if (!is_non_materialized(user)) {
          level = std::max(level, user_level + 1);
        } else {
          level = std::max(level, user_level);
        }
      }
    }

    op_level[op] = level;
    max_level = std::max(max_level, level);
  }

  // Step 2: Reverses the level so the earliest op gets level 0.
  for (Operation *op : sorted_ops) {
    int raw_level = op_level[op];
    int normalized_level = max_level - raw_level;
    op_level[op] = normalized_level;
  }

  // Step 3: Overwrites critical ops with ASAP schedule: shortest path from
  // source.
  for (Operation *op : sorted_ops) {
    if (!critical_ops.count(op)) {
      continue;
    }

    int level = -1;
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || !op_level.count(def)) {
        continue;
      }

      int def_level = op_level[def];

      assert(def_level <= op_level[op] &&
             "Critical op should not have a lower level than its operands");
      // Increments level only for materialized ops.
      if (!is_non_materialized(op)) {
        level = std::max(level, def_level + 1);
      } else {
        level = std::max(level, def_level);
      }
    }

    if (level != -1) {
      // If there exists operand, moves the critical op earlier.
      op_level[op] = level;
    }
  }

  // Step 4: Assembles the ops into level buckets.
  std::vector<std::vector<Operation *>> level_buckets(max_level + 1);

  for (Operation *op : sorted_ops) {
    level_buckets[op_level[op]].push_back(op);
  }

  return level_buckets;
}

std::vector<std::pair<Operation *, int>> mlir::neura::flatten_level_buckets(
    const std::vector<std::vector<Operation *>> &level_buckets) {
  std::vector<std::pair<Operation *, int>> result;

  for (int level = 0; level < static_cast<int>(level_buckets.size()); ++level) {
    for (Operation *op : level_buckets[level]) {
      result.emplace_back(op, level);
    }
  }

  return result;
}

mlir::Operation *mlir::neura::getMaterializedBackwardUser(Operation *op) {
  assert(isa<neura::CtrlMovOp>(op) && "Expected a ctrl_mov operation");
  auto ctrl_mov = dyn_cast<neura::CtrlMovOp>(op);
  Value target = ctrl_mov.getTarget();

  assert(isa<neura::ReserveOp>(target.getDefiningOp()) &&
         "Expected the user of ctrl_mov target to be a reserve operation");
  auto reserve_op = dyn_cast<neura::ReserveOp>(target.getDefiningOp());

  // Skip ctrl_mov users of reserve; return the first materialized user.
  for (Operation *user : reserve_op.getResult().getUsers()) {
    if (isMaterializedReserveUser(user)) {
      return user;
    }
  }
  assert(false &&
         "No materialized backward user (i.e., phi) found for ctrl_mov");
}

llvm::SmallVector<mlir::Operation *>
mlir::neura::getMaterializedUserOps(Operation *op) {
  llvm::SmallVector<Operation *> result;
  llvm::DenseSet<Operation *> visited;
  visited.insert(op);
  llvm::errs() << "Starting to collect materialized users for: " << *op << "\n";
  llvm::SmallVector<Operation *> worklist(op->getUsers().begin(),
                                          op->getUsers().end());

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

bool mlir::neura::tryRouteForwardMove(Operation *mov_op, MappingLoc src_loc,
                                      MappingLoc dst_loc,
                                      const MappingState &state,
                                      std::vector<MappingLoc> &path_out) {
  return tryRouteDataMove(mov_op, src_loc, dst_loc, false, state, path_out);
}

bool mlir::neura::tryRouteBackwardMove(Operation *mov_op, MappingLoc src_loc,
                                       MappingLoc dst_loc,
                                       const MappingState &state,
                                       std::vector<MappingLoc> &path_out) {
  llvm::errs() << "[tryRouteBackwardMove] src_loc: "
               << src_loc.resource->getType() << "#"
               << src_loc.resource->getId() << " @t=" << src_loc.time_step
               << ", dst_loc: " << dst_loc.resource->getType() << "#"
               << dst_loc.resource->getId() << " @t=" << dst_loc.time_step
               << "\n";
  return tryRouteDataMove(mov_op, src_loc, dst_loc, true, state, path_out);
}

Register *mlir::neura::getAvailableRegister(const MappingState &state,
                                            Tile *tile, int start_time,
                                            int exclusive_end_time) {
  for (Register *reg : tile->getRegisters()) {
    // FIXME: We may need constrain the register availability to the conflicting
    // input channel (either the input channel or a register file on the same
    // input direction could be active at one time).
    if (state.isAvailableAcrossTimeInRange(reg, start_time,
                                           exclusive_end_time)) {
      return reg;
    }
  }
  return nullptr;
}

bool mlir::neura::tryRouteDataMove(Operation *mov_op, MappingLoc src_loc,
                                   MappingLoc dst_loc, bool is_backward_move,
                                   const MappingState &state,
                                   std::vector<MappingLoc> &path_out) {
  assert(path_out.empty() && "Output path should be empty");

  // Gets the source tile and destination tile.
  Tile *src_tile = dyn_cast<Tile>(src_loc.resource);
  Tile *dst_tile = dyn_cast<Tile>(dst_loc.resource);

  assert(src_tile && dst_tile &&
         "Source and destination locations must be tiles");

  // Calculates the deadline time step (adds II for backward moves).
  int exclusive_deadline_step = dst_loc.time_step;
  if (is_backward_move) {
    exclusive_deadline_step += state.getII();
  }

  llvm::outs() << "[tryRouteDataMove] Routing from Tile#" << src_tile->getId()
               << " @t=" << src_loc.time_step << " to Tile#"
               << dst_tile->getId() << " @t=" << exclusive_deadline_step
               << "\n";

  // Special case: source tile and destination tile are the same.
  if (src_tile == dst_tile) {
    // Uses register as routing resource within the same tile.
    // Finds an available register to store the data.
    Register *available_reg = getAvailableRegister(
        state, src_tile, src_loc.time_step, exclusive_deadline_step);
    if (!available_reg) {
      llvm::outs()
          << "[tryRouteDataMove] Cannot find available register on Tile#"
          << src_tile->getId() << " for time range: t=" << src_loc.time_step
          << " to t=" << exclusive_deadline_step << "\n";
      return false;
    }

    // Builds path: uses register to store data for the specified time period.
    for (int t = src_loc.time_step; t < exclusive_deadline_step; ++t) {
      path_out.push_back({available_reg, t});
    }

    llvm::outs() << "[tryRouteDataMove] Successfully routed on same tile using "
                    "Register #"
                 << available_reg->getId() << "\n";
    return true;
  }

  // Search state: records current tile, time step, and path to reach this
  // state.
  struct SearchState {
    Tile *current_tile; // Current tile location.
    int current_time;   // Current time step.
    std::vector<MappingLoc>
        path; // Routing resource path to reach current state.
  };

  // BFS search.
  std::queue<SearchState> search_queue;
  std::set<std::pair<Tile *, int>>
      visited; // Records visited (tile, time) combinations.

  // Initial state: starts from source tile.
  search_queue.push({src_tile, src_loc.time_step, {}});
  visited.insert({src_tile, src_loc.time_step});

  while (!search_queue.empty()) {
    SearchState current_state = search_queue.front();
    search_queue.pop();

    // Checks if destination tile is reached with appropriate timing.
    if (current_state.current_tile == dst_tile) {
      // The link/register between producer tile and consumer tile is belonging
      // to the producer tile with same time step.
      if (current_state.current_time <= exclusive_deadline_step) {
        if (current_state.current_time == exclusive_deadline_step) {
          // Arrives exactly at deadline, no additional register needed.
          path_out = current_state.path;
          return true;
        } else {
          // Arrives early, needs register on destination tile to wait.
          Register *wait_reg =
              getAvailableRegister(state, dst_tile, current_state.current_time,
                                   exclusive_deadline_step);
          if (!wait_reg) {
            llvm::outs() << "[tryRouteDataMove] Cannot find available waiting"
                            "register on destination Tile#"
                         << dst_tile->getId() << "\n";
            continue; // Tries other paths.
          }

          // Builds complete path.
          path_out = current_state.path;
          for (int t = current_state.current_time; t < exclusive_deadline_step;
               ++t) {
            path_out.push_back({wait_reg, t});
          }
          return true;
        }
      } else {
        // Arrives too late, skips this path.
        continue;
      }
    }

    // Skips if current time already exceeds deadline.
    if (current_state.current_time >= exclusive_deadline_step) {
      continue;
    }

    // Explores two routing options from current tile:

    // Option 1: Moves to adjacent tile through link.
    for (Link *out_link : current_state.current_tile->getOutLinks()) {
      MappingLoc link_loc = {out_link, current_state.current_time};

      // Checks if link is available at current time step.
      if (!state.isAvailableAcrossTime(link_loc)) {
        continue;
      }

      Tile *next_tile = out_link->getDstTile();
      int next_time = current_state.current_time + 1;

      // Checks if this (tile, time) combination has been visited.
      if (visited.insert({next_tile, next_time}).second) {
        std::vector<MappingLoc> new_path = current_state.path;
        new_path.push_back(link_loc);

        search_queue.push({next_tile, next_time, new_path});
      }
    }

    // Option 2: Uses register on current tile to wait one time step.
    Register *wait_register = getAvailableRegister(
        state, current_state.current_tile, current_state.current_time,
        current_state.current_time + 1);
    if (wait_register) {
      int next_time = current_state.current_time + 1;
      // Checks if this(tile, time) combination has been visited.
      // Though theoretically we can revisit a tile at different time steps
      // to explore alternative routing paths, we disallow this during the
      // routing search to prevent exponential search complexity and ensure
      // algorithm termination within reasonable time bounds.
      if (visited.insert({current_state.current_tile, next_time}).second) {
        std::vector<MappingLoc> new_path = current_state.path;
        new_path.push_back({wait_register, current_state.current_time});

        search_queue.push({current_state.current_tile, next_time, new_path});
      }
    }
  }

  // Search failed.
  llvm::outs() << "[tryRouteDataMove] Cannot find routing path from Tile#"
               << src_tile->getId() << " @t=" << src_loc.time_step
               << " to Tile#" << dst_tile->getId()
               << " @t=" << exclusive_deadline_step << "\n";
  return false;
}

Operation *mlir::neura::getMaterializedProducer(Value operand) {
  Operation *producer = operand.getDefiningOp();
  
  // In steering mode, some operations (like constants, carry, invariant, etc.)
  // may not be wrapped by DataMovOp. Return them directly.
  if (!isa<neura::DataMovOp>(producer)) {
    // This is likely a steering mode operation that doesn't need DataMovOp wrapping
    return producer;
  }
  
  // For operations wrapped by DataMovOp, find the actual producer.
  auto mov_op = dyn_cast<neura::DataMovOp>(producer);
  auto materialized_producer = mov_op.getOperand().getDefiningOp();
  return materialized_producer;
}

int mlir::neura::getPhysicalHops(const std::vector<Operation *> &producers,
                                 Tile *tile,
                                 const MappingState &mapping_state) {

  // Counts the number of physical hops from the producers to the tile.
  int hops = 0;

  for (Operation *producer : producers) {
    // Get the last location of the producer.
    auto producer_locs = mapping_state.getAllLocsOfOp(producer);
    assert(!producer_locs.empty() && "No locations found for producer");

    MappingLoc producer_loc = producer_locs.back();
    Tile *producer_tile = dyn_cast<Tile>(producer_loc.resource);
    assert(producer_tile && "Producer location must be a Tile");
    hops += std::abs(producer_tile->getX() - tile->getX()) +
            std::abs(producer_tile->getY() - tile->getY());
  }
  return hops;
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
    if (!canReachLocInTime(producer_loc, target_loc, deadline_step,
                           mapping_state)) {
      return false;
    }
  }
  return true;
}

bool mlir::neura::canReachLocInTime(const MappingLoc &src_loc,
                                    const MappingLoc &dst_loc,
                                    int deadline_step,
                                    const MappingState &mapping_state) {
  // Checks if the destination is reachable from the source within the given
  // time window.
  if (src_loc.resource == dst_loc.resource &&
      src_loc.time_step < deadline_step && dst_loc.time_step <= deadline_step) {
    return true;
  }

  // Checks if the destination is reachable from the source tile within given
  // steps.
  assert(isa<Tile>(src_loc.resource));
  assert(isa<Tile>(dst_loc.resource));

  struct QueueEntry {
    MappingLoc loc;
    int current_step;
  };

  std::queue<QueueEntry> queue;
  llvm::DenseSet<Tile *> visited;

  queue.push({src_loc, src_loc.time_step});
  visited.insert(dyn_cast<Tile>(src_loc.resource));

  while (!queue.empty()) {
    auto [current_loc, current_step] = queue.front();
    queue.pop();

    // If we reach the destination tile and time step is not after dst_loc
    if (current_loc.resource == dst_loc.resource &&
        current_step <= deadline_step) {
      return true;
    }

    if (current_step >= deadline_step) {
      continue;
    }

    // // Explores all next step tiles from the current location.
    // for (const MappingLoc &next_loc_tile :
    //      mapping_state.getNextStepTiles(current_loc)) {

    // Explores all next step tiles from the current location.
    for (const MappingLoc &current_loc_out_link :
         mapping_state.getCurrentStepLinks(current_loc)) {

      // Makes sure the link is not occupied.
      if (!mapping_state.isAvailableAcrossTime(current_loc_out_link)) {
        continue;
      }

      // Skips if already miss the deadline.
      int next_step = current_step + 1;
      if (next_step > deadline_step) {
        continue;
      }

      // Records the tile for further exploration.
      Tile *next_tile =
          llvm::dyn_cast<Link>(current_loc_out_link.resource)->getDstTile();
      assert(next_tile && "Next location must be a Tile");
      if (visited.contains(next_tile)) {
        continue;
      }

      visited.insert(next_tile);
      MappingLoc next_loc_tile_with_step = {next_tile, next_step};
      queue.push({next_loc_tile_with_step, next_step});
    }
  }

  return false;
}

bool mlir::neura::isMaterializedReserveUser(Operation *user) {
  if (isa<neura::PhiOp>(user)) {
    return true;
  }
  if (isa<neura::InvariantOp>(user)) {
    return true;
  }
  if (isa<neura::CarryOp>(user)) {
    return true;
  }
  // Fused steering control operations
  if (isa<neura::CarryInvariantOp>(user)) {
    return true;
  }
  if (isa<neura::ConditionalSelectOp>(user)) {
    return true;
  }
  if (isa<neura::InvariantGroupOp>(user)) {
    return true;
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

std::vector<MappingLoc>
mlir::neura::calculateAward(Operation *op, std::set<Operation *> &critical_ops,
                            int target_level, const Architecture &architecture,
                            const MappingState &mapping_state) {
  // Early exit if the operation is not supported by all the tiles.
  bool op_can_be_supported = false;
  for (Tile *tile : architecture.getAllTiles()) {
    if (tile->canSupportOperation(getOperationKindFromMlirOp(op))) {
      op_can_be_supported = true;
    }
  }
  if (!op_can_be_supported) {
    llvm::errs() << "[calculateAward] Operation: " << *op
                 << " is not supported by any tile.\n";
    return {};
  }

  // A heap of locations with their associated award. Note that we use a
  // max-heap to prioritize locations with higher awards.
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
    assert(isMaterializedReserveUser(materialized_backward_op) &&
           "Expected materialized operation of ctrl_mov to be a "
           "PhiOp/InvariantOp/CarryOp.");
    backward_users.push_back(materialized_backward_op);
  }

  llvm::errs() << "[calculateAward] Operation: " << *op
               << "; Producers: " << producers.size() << "\n";

  for (Tile *tile : architecture.getAllTiles()) {
    if (!tile->canSupportOperation(getOperationKindFromMlirOp(op))) {
      llvm::errs() << "[calculateAward] Tile: " << tile->getType()
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
      backward_users_locs.push_back(backward_user_loc);
    }
    int award = 2 * mapping_state.getII();
    if (critical_ops.count(op)) {
      award += tile->getDstTiles().size();
      award += op->getOperands().size() -
               getPhysicalHops(producers, tile, mapping_state);
    }

    for (int t = earliest_start_time_step; t < latest_end_time_step; t += 1) {
      MappingLoc tile_loc_candidate = {tile, t};
      // If the tile at time `t` is available, we can consider it for mapping.
      if (mapping_state.isAvailableAcrossTime(tile_loc_candidate)) {
        bool meet_producer_constraint =
            producers.empty() ||
            canReachLocInTime(producers, tile_loc_candidate, t, mapping_state);
        bool meet_backward_user_constraint = true;
        for (auto &backward_user_loc : backward_users_locs) {
          // If there is no backward user, we can consider it for mapping.
          // Otherwise, check if the location can reach all backward users.
          if (!canReachLocInTime(tile_loc_candidate, backward_user_loc,
                                 backward_user_loc.time_step +
                                     mapping_state.getII(),
                                 mapping_state)) {
            meet_backward_user_constraint = false;
            break; // No need to check further.
          }
        }
        // If no producer or the location is reachable by all producers, and
        // no backward user or the location can reach all backward users,
        // we can consider it for mapping and grant reward.
        if (meet_producer_constraint && meet_backward_user_constraint) {
          // Grants higher award if the location is physically closed to
          // producers. award += producers.size() - getPhysicalHops(producers,
          // tile, mapping_state); if (op->getOperands().size() > 1 &&
          // getPhysicalHops(producers, tile, mapping_state) < 2) {
          //   award += 1;
          // }
          updateAward(locs_with_award, tile_loc_candidate, award);
        }
      }
      // The mapping location with earlier time step is granted with a higher
      // award.
      award -= 1;
    }
    // assert(award >= 0 && "Award should not be negative");
  }

  // Copies map entries into a vector of pairs for sorting.
  std::vector<std::pair<MappingLoc, int>> locs_award_vec(
      locs_with_award.begin(), locs_with_award.end());

  // Sorts by award (descending).
  std::sort(
      locs_award_vec.begin(), locs_award_vec.end(),
      [](const std::pair<MappingLoc, int> &a,
         const std::pair<MappingLoc, int> &b) { return a.second > b.second; });
  // TODO: Needs to handle tie case and prioritize lower resource utilization,
  // however, compiled II becomes worse after adding this tie-breaker:
  // https://github.com/coredac/dataflow/issues/59.
  // std::sort(locs_award_vec.begin(), locs_award_vec.end(),
  //           [&](const std::pair<MappingLoc, int> &a, const
  //           std::pair<MappingLoc, int> &b) {
  //               if (a.second != b.second) {
  //                 return a.second > b.second;
  //               }
  //               // Tie-breaker: prioritizes lower resource utilization and
  //               // earlier time step.
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
  llvm::SmallVector<Operation *> results;
  for (Operation *user : op->getUsers()) {
    if (isa<neura::CtrlMovOp>(user)) {
      results.push_back(user);
    }
  }
  return results;
}

bool mlir::neura::placeAndRoute(Operation *op, const MappingLoc &target_loc,
                                MappingState &mapping_state) {
  if (mapping_state.bindOp(target_loc, op)) {
    std::vector<Operation *> routed_operands;
    std::vector<Operation *> routed_ctrl_movs;
    llvm::errs() << "[DEBUG] Schedule op " << *op
                 << " onto loc: " << target_loc.resource->getType() << "#"
                 << target_loc.resource->getId()
                 << " @t=" << target_loc.time_step << "\n";
    // Tries to route the data move operations.
    for (Value operand : op->getOperands()) {
      llvm::errs() << "Processing operand: " << operand << "\n";
      if (isa<neura::ReserveOp>(operand.getDefiningOp())) {
        // Skips Reserve ops (backward ctrl move) when estimate cost.
        continue;
      }
      Operation *data_move = operand.getDefiningOp();
      
      // In steering mode, some operands may not be DataMovOp (e.g., constants, carry, etc.)
      if (!isa<neura::DataMovOp>(data_move)) {
        // Skip non-DataMovOp operands in steering mode
        llvm::errs() << "Skipping non-DataMovOp operand in steering mode\n";
        continue;
      }
      
      Operation *producer = getMaterializedProducer(operand);
      MappingLoc src_loc = mapping_state.getAllLocsOfOp(producer).back();

      std::vector<MappingLoc> route_path;
      if (tryRouteForwardMove(data_move, src_loc, target_loc, mapping_state,
                              route_path)) {
        // Reserves the route for the data move operation.
        mapping_state.reserveRoute(data_move, route_path);
        routed_operands.push_back(data_move);
        llvm::errs() << "[DEBUG] Successfully routed data move: " << *data_move
                     << " from " << src_loc.resource->getType() << "#"
                     << src_loc.resource->getId() << " @t=" << src_loc.time_step
                     << " to " << target_loc.resource->getType() << "#"
                     << target_loc.resource->getId()
                     << " @t=" << target_loc.time_step << "\n";
        continue;
      }
      llvm::errs() << "[DEBUG] Failed to route data move: " << *data_move
                   << " from " << src_loc.resource->getType() << "#"
                   << src_loc.resource->getId() << " @t=" << src_loc.time_step
                   << " to " << target_loc.resource->getType() << "#"
                   << target_loc.resource->getId()
                   << " @t=" << target_loc.time_step << "; so unschedule op\n";
      mapping_state.unbindOp(op);
      for (Operation *routed_op : routed_operands) {
        llvm::errs() << "[DEBUG] Releasing route for routed operand: "
                     << *routed_op << "\n";
        mapping_state.releaseRoute(routed_op);
      }
      return false;
    }
    // Checks whether the operation's user is a ctrl_mov.
    for (Operation *user : getCtrlMovUsers(op)) {
      auto ctrl_mov = dyn_cast<neura::CtrlMovOp>(user);
      llvm::errs() << "[DEBUG] Found ctrl_mov user: " << *ctrl_mov << "\n";
      assert(ctrl_mov && "Expected user to be a CtrlMovOp");
      mlir::Operation *materialized_backward_op =
          getMaterializedBackwardUser(ctrl_mov);
      assert(isMaterializedReserveUser(materialized_backward_op) &&
             "Expected materialized operation of ctrl_mov to be a "
             "PhiOp/InvariantOp/CarryOp");
      // Gets the last location of the materialized operation.
      MappingLoc backward_loc =
          mapping_state.getAllLocsOfOp(materialized_backward_op).back();
      // Routes the ctrl_mov to the phi location.
      std::vector<MappingLoc> route_path;
      if (tryRouteBackwardMove(ctrl_mov, target_loc, backward_loc,
                               mapping_state, route_path)) {
        mapping_state.reserveRoute(ctrl_mov, route_path);
        routed_ctrl_movs.push_back(ctrl_mov);
        llvm::errs() << "[DEBUG] Successfully routed ctrl_mov: " << *ctrl_mov
                     << " to " << backward_loc.resource->getType() << "#"
                     << backward_loc.resource->getId()
                     << " @t=" << backward_loc.time_step << "\n";
        continue;
      }
      llvm::errs() << "[DEBUG] Failed to route ctrl_mov: " << *ctrl_mov
                   << " from " << target_loc.resource->getType() << "#"
                   << target_loc.resource->getId()
                   << " @t=" << target_loc.time_step << " to "
                   << backward_loc.resource->getType() << "#"
                   << backward_loc.resource->getId()
                   << " @t=" << backward_loc.time_step
                   << "; so unschedule op\n";
      mapping_state.unbindOp(op);
      for (Operation *routed_ctrl_mov : routed_ctrl_movs) {
        llvm::errs() << "[DEBUG] Releasing route for routed ctrl_mov: "
                     << *routed_ctrl_mov << "\n";
        mapping_state.releaseRoute(routed_ctrl_mov);
      }

      for (Operation *routed_op : routed_operands) {
        llvm::errs() << "[DEBUG] Releasing route for routed operand: "
                     << *routed_op << "\n";
        mapping_state.releaseRoute(routed_op);
      }
      return false;
    }
    return true;
  }
  return false;
}