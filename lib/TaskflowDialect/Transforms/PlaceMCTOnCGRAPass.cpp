//===- PlaceMCTOnCGRAPass.cpp - MCT to CGRA Placement Pass ----------------===//
//
// This pass places Minimized Canonicalized Tasks (MCTs) onto a 2D CGRA grid：
// 1. SSA use-def placement: Tasks with SSA dependencies placed on adjacent CGRAs.
// 2. Memory mapping: Assigns memrefs to SRAMs (single-SRAM constraint per data).
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===----------------------------------------------------------------------===//
// CGRA Grid Position
//===----------------------------------------------------------------------===//
/// Represents a position on the 2D CGRA grid.
struct CGRAPosition {
  int row;
  int col;

  bool operator==(const CGRAPosition &other) const {
    return row == other.row && col == other.col;
  }

  /// Computes Manhattan distance to another position.
  int manhattanDistance(const CGRAPosition &other) const {
    return std::abs(row - other.row) + std::abs(col - other.col);
  }

  /// Checks if adjacent (Manhattan distance = 1).
  bool isAdjacent(const CGRAPosition &other) const {
    return manhattanDistance(other) == 1;
  }
};

//===----------------------------------------------------------------------===//
// Task Placement Info
//===----------------------------------------------------------------------===//
/// Stores placement info for a task: can span multiple combined CGRAs.
struct TaskPlacement {
  SmallVector<CGRAPosition> cgra_positions; // CGRAs assigned to this task.

  /// Returns the primary (first) position.
  CGRAPosition primary() const {
    return cgra_positions.empty() ? CGRAPosition{-1, -1} : cgra_positions[0];
  }

  /// Returns the number of CGRAs assigned.
  size_t cgraCount() const { return cgra_positions.size(); }

  /// Checks if any CGRA in this task is adjacent to any in other task.
  bool hasAdjacentCGRA(const TaskPlacement &other) const {
    for (const auto &pos : cgra_positions) {
      for (const auto &other_pos : other.cgra_positions) {
        if (pos.isAdjacent(other_pos)) {
          return true;
        }
      }
    }
    return false;
  }
};

//===----------------------------------------------------------------------===//
// Counter Chain Info
//===----------------------------------------------------------------------===//
/// Stores counter chain bounds for same-header comparison.
struct CounterChainInfo {
  SmallVector<int64_t> bounds;

  bool operator==(const CounterChainInfo &other) const {
    return bounds == other.bounds;
  }

  static CounterChainInfo extract(TaskflowTaskOp task) {
    CounterChainInfo info;
    task.walk([&](TaskflowCounterOp counter) {
      if (!counter.getParentIndex()) {
        info.collectChain(counter);
      }
    });
    return info;
  }

private:
  void collectChain(TaskflowCounterOp counter) {
    bounds.push_back(counter.getUpperBound().getSExtValue());
    for (Operation *user : counter.getResult().getUsers()) {
      if (auto child = dyn_cast<TaskflowCounterOp>(user)) {
        collectChain(child);
        break;
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// SSA Dependency
//===----------------------------------------------------------------------===//
struct SSADependency {
  size_t producer_idx;
  size_t consumer_idx;
  bool same_header;
};

//===----------------------------------------------------------------------===//
// Memory Mapping
//===----------------------------------------------------------------------===//
/// Assigns memrefs to SRAMs. Single-SRAM constraint: each memref can only
/// reside in one SRAM (but can be accessed from DRAM dynamically).
struct MemoryMapper {
  DenseMap<Value, int> memref_to_sram; // Maps memref to SRAM ID.

  /// Direct wire connections: For fusion candidates on adjacent CGRAs,
  /// data can bypass SRAM and flow directly through interconnect.
  /// Stores: (producer_task_idx, consumer_task_idx, via_value).
  struct DirectWire {
    size_t producer_idx;
    size_t consumer_idx;
    Value via_value;
  };
  SmallVector<DirectWire> direct_wires;

  /// Assigns a memref to the closest SRAM near the given task position.
  int assignSRAM(Value memref, const TaskPlacement &placement) {
    auto it = memref_to_sram.find(memref);
    if (it != memref_to_sram.end()) {
      return it->second; // Already assigned.
    }

    // Assigns to a new SRAM near the task's primary CGRA.
    // In baseline, SRAM ID corresponds to CGRA position for locality.
    auto pos = placement.primary();
    int sram_id = pos.row * 100 + pos.col; // Simple encoding: row*100 + col.
    memref_to_sram[memref] = sram_id;
    return sram_id;
  }

  /// Configures direct wire for adjacent fusion candidates.
  /// Producer output goes directly to consumer without SRAM roundtrip.
  void configureDirectWire(size_t producer_idx, size_t consumer_idx,
                           Value via_value) {
    direct_wires.push_back({producer_idx, consumer_idx, via_value});
  }

  /// Prints memory mapping summary.
  void printMapping() const {
    llvm::outs() << "\n=== Memory Mapping ===\n";
    for (const auto &entry : memref_to_sram) {
      llvm::outs() << "  ";
      if (auto arg = dyn_cast<BlockArgument>(entry.first)) {
        llvm::outs() << "func_arg" << arg.getArgNumber();
      } else {
        entry.first.print(llvm::outs());
      }
      llvm::outs() << " -> SRAM_" << entry.second << "\n";
    }

    if (!direct_wires.empty()) {
      llvm::outs() << "\n=== Direct Wires (bypass SRAM) ===\n";
      for (const auto &dw : direct_wires) {
        llvm::outs() << "  Task_" << dw.producer_idx << " -> Task_"
                     << dw.consumer_idx << " (direct)\n";
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// CGRA Placer
//===----------------------------------------------------------------------===//
/// Places MCTs onto a 2D CGRA grid with memory mapping.
class CGRAPlacer {
public:
  CGRAPlacer(int grid_rows, int grid_cols)
      : grid_rows_(grid_rows), grid_cols_(grid_cols) {
    occupied_.resize(grid_rows_);
    for (auto &row : occupied_) {
      row.resize(grid_cols_, false);
    }
  }

  /// Places all tasks and performs memory mapping.
  void place(func::FuncOp func) {
    SmallVector<TaskflowTaskOp> tasks;
    func.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });

    if (tasks.empty()) {
      llvm::errs() << "No tasks to place.\n";
      return;
    }


    // Extracts counter chains and builds dependency graph.
    SmallVector<CounterChainInfo> counter_chains;
    DenseMap<Value, size_t> output_to_producer;
    SmallVector<SSADependency> deps;

    for (size_t idx = 0; idx < tasks.size(); ++idx) {
      counter_chains.push_back(CounterChainInfo::extract(tasks[idx]));
      for (Value output : tasks[idx].getMemoryOutputs()) {
        output_to_producer[output] = idx;
      }
    }

    for (size_t idx = 0; idx < tasks.size(); ++idx) {
      for (Value input : tasks[idx].getMemoryInputs()) {
        auto it = output_to_producer.find(input);
        if (it != output_to_producer.end()) {
          size_t producer_idx = it->second;
          bool same_header = counter_chains[producer_idx] == counter_chains[idx];
          deps.push_back({producer_idx, idx, same_header});
        }
      }
    }

    // Critical path priority placement:
    // 1. Computes ALAP level for each task (longest path to sink).
    // 2. Sorts tasks by: (a) ALAP level, (b) criticality, (c) degree.
    // 3. Places tasks in sorted order with heuristic scoring.
    SmallVector<TaskPlacement> placements(tasks.size());
    SmallVector<size_t> placement_order = computePlacementOrder(tasks, deps);

    for (size_t idx : placement_order) {
      // Reads cgra_count from task attribute (default: 1).
      // Assumes the fusion pass will set this attribute for tasks needing multiple CGRAs.
      // TODO: Rewrite this after the fusion pass is updated.
      int cgra_count = 1;
      if (auto attr = tasks[idx]->getAttrOfType<IntegerAttr>("cgra_count")) {
        cgra_count = attr.getInt();
      }

      TaskPlacement placement = findBestPlacement(idx, cgra_count, placements, deps);
      placements[idx] = placement;

      // Marks occupied.
      for (const auto &pos : placement.cgra_positions) {
        occupied_[pos.row][pos.col] = true;
      }


      // Checks adjacency to dependent tasks and configures direct wires.
      for (const auto &dep : deps) {
        if (dep.consumer_idx == idx && placements[dep.producer_idx].cgraCount() > 0) {
          if (placement.hasAdjacentCGRA(placements[dep.producer_idx])) {
            // llvm::outs() << " [ADJACENT TO " << tasks[dep.producer_idx].getTaskName() << "]";

            // Direct wire for same-header fusion candidates on adjacent CGRAs.
            // Data bypasses SRAM and flows directly through interconnect.
            if (dep.same_header) {
              // Gets the SSA value connecting producer output to consumer input.
              for (Value input : tasks[idx].getMemoryInputs()) {
                for (Value output : tasks[dep.producer_idx].getMemoryOutputs()) {
                  if (input == output) {
                    memory_mapper_.configureDirectWire(dep.producer_idx, idx, input);
                    // llvm::outs() << " [DIRECT WIRE]";
                  }
                }
              }
            }
          }
        }
      }
      llvm::outs() << "\n";

      // Memory mapping: Assigns input/output memrefs to SRAMs.
      for (Value input : tasks[idx].getMemoryInputs()) {
        memory_mapper_.assignSRAM(input, placement);
      }
      for (Value output : tasks[idx].getMemoryOutputs()) {
        memory_mapper_.assignSRAM(output, placement);
      }
    }

    // Annotates tasks with placement info.
    OpBuilder builder(func.getContext());
    for (size_t idx = 0; idx < tasks.size(); ++idx) {
      const auto &placement = placements[idx];
      auto pos = placement.primary();
      tasks[idx]->setAttr("cgra_row", builder.getI32IntegerAttr(pos.row));
      tasks[idx]->setAttr("cgra_col", builder.getI32IntegerAttr(pos.col));
      tasks[idx]->setAttr("cgra_count",
                          builder.getI32IntegerAttr(placement.cgraCount()));
    }

    // llvm::outs() << "\n=== Placement Summary ===\n";
    // printGrid(tasks, placements);
    // memory_mapper_.printMapping();
  }

private:
  /// Finds best placement for a task requiring cgra_count CGRAs.
  TaskPlacement findBestPlacement(size_t task_idx, int cgra_count,
                                  const SmallVector<TaskPlacement> &placements,
                                  const SmallVector<SSADependency> &deps) {
    int best_score = INT_MIN;
    TaskPlacement best_placement;

    // Baseline: For cgra_count=1, finds single best position.
    for (int r = 0; r < grid_rows_; ++r) {
      for (int c = 0; c < grid_cols_; ++c) {
        if (occupied_[r][c])
          continue;

        TaskPlacement candidate;
        candidate.cgra_positions.push_back({r, c});

        int score = computeScore(task_idx, candidate, placements, deps);
        if (score > best_score) {
          best_score = score;
          best_placement = candidate;
        }
      }
    }

    // Error handling: No available position found (grid over-subscribed).
    if (best_placement.cgra_positions.empty()) {
      llvm::errs() << "Warning: No available CGRA position for task "
                   << task_idx << ". Grid is over-subscribed (" << grid_rows_
                   << "x" << grid_cols_ << " grid with all cells occupied).\n";
      // Fallback: Assign to position (0,0) with a warning.
      best_placement.cgra_positions.push_back({0, 0});
    }

    return best_placement;
  }

  /// Computes placement score.
  ///
  /// Formula: Award(Task, CGRA) = α·Pneigh + β·Psib - γ·Comm + δ·Bal + Adj
  ///
  ///   Pneigh: Proximity to placed neighbors (count of adjacent dependencies).
  ///   Psib:   Proximity to same-header siblings (fusion candidates).
  ///   Comm:   Communication cost = Σ wj · Dist(C, posj).
  ///   Bal:    Bonus for under-utilized CGRAs (load balancing).
  ///
  /// Adjacency bonus (Adj):
  ///   if adjacent:  +100 (same_header) or +50 (different_header)
  ///   else:         -distance * 10 (same_header) or -distance (different)
  ///
  /// Weights: α=50, β=100, γ=10, δ=20 (tunable).
  int computeScore(size_t task_idx, const TaskPlacement &placement,
                   const SmallVector<TaskPlacement> &placements,
                   const SmallVector<SSADependency> &deps) {
    // Weight constants (tunable).
    constexpr int kAlpha = 50;   // Pneigh weight.
    constexpr int kBeta = 100;   // Psib weight (same-header bonus).
    constexpr int kGamma = 10;   // Comm weight (distance penalty).
    constexpr int kDelta = 20;   // Bal weight (load balance bonus).

    int pneigh = 0;  // Proximity to placed neighbors.
    int psib = 0;    // Proximity to same-header siblings.
    int comm = 0;    // Communication cost.
    int adj = 0;     // Original adjacency bonus.

    for (const auto &dep : deps) {
      // Checks if this task is consumer.
      if (dep.consumer_idx == task_idx) {
        const auto &producer = placements[dep.producer_idx];
        if (producer.cgraCount() == 0)
          continue;

        int dist = placement.primary().manhattanDistance(producer.primary());

        if (placement.hasAdjacentCGRA(producer)) {
          // Adjacent neighbor bonus (SARA factor).
          pneigh += 1;
          if (dep.same_header) {
            psib += 1;  // Extra bonus for fusion candidates.
          }
          // Original adjacency bonus.
          adj += dep.same_header ? 100 : 50;
        } else {
          // Distance penalty (original formula).
          adj -= dep.same_header ? dist * 10 : dist;
        }

        // Communication cost (weighted by same_header priority).
        int weight = dep.same_header ? 2 : 1;
        comm += weight * dist;
      }

      // Checks if this task is producer.
      if (dep.producer_idx == task_idx) {
        const auto &consumer = placements[dep.consumer_idx];
        if (consumer.cgraCount() == 0)
          continue;

        int dist = placement.primary().manhattanDistance(consumer.primary());

        if (placement.hasAdjacentCGRA(consumer)) {
          pneigh += 1;
          if (dep.same_header) {
            psib += 1;
          }
          adj += dep.same_header ? 100 : 50;
        } else {
          adj -= dep.same_header ? dist * 10 : dist;
        }

        int weight = dep.same_header ? 2 : 1;
        comm += weight * dist;
      }
    }

    // Load balance bonus: Prefer under-utilized CGRAs.
    // Counts tasks already placed in same row/column.
    int bal = 0;
    auto pos = placement.primary();
    int row_count = 0, col_count = 0;
    for (int c = 0; c < grid_cols_; ++c) {
      if (occupied_[pos.row][c]) row_count++;
    }
    for (int r = 0; r < grid_rows_; ++r) {
      if (occupied_[r][pos.col]) col_count++;
    }
    // Bonus for less crowded positions.
    bal = (grid_cols_ - row_count) + (grid_rows_ - col_count);

    // Final score: Award = α·Pneigh + β·Psib - γ·Comm + δ·Bal + Adj
    int score = kAlpha * pneigh + kBeta * psib - kGamma * comm + kDelta * bal + adj;
    return score;
  }

  /// Computes placement order using critical path priority.
  /// Priority: (1) ALAP level, (2) degree (connectivity), (3) original order.
  SmallVector<size_t> computePlacementOrder(
      const SmallVector<TaskflowTaskOp> &tasks,
      const SmallVector<SSADependency> &deps) {
    size_t n = tasks.size();
    SmallVector<int> alap_level(n, 0);
    SmallVector<int> degree(n, 0);

    // Builds adjacency for ALAP computation.
    SmallVector<SmallVector<size_t>> successors(n);
    for (const auto &dep : deps) {
      successors[dep.producer_idx].push_back(dep.consumer_idx);
    }

    // Computes ALAP level (longest path to any sink).
    // Process in reverse topological order.
    for (int i = n - 1; i >= 0; --i) {
      int level = 0;
      for (size_t succ : successors[i]) {
        level = std::max(level, alap_level[succ] + 1);
      }
      alap_level[i] = level;
    }

    // Computes degree (number of dependencies).
    for (const auto &dep : deps) {
      degree[dep.producer_idx]++;
      degree[dep.consumer_idx]++;
    }

    // Creates sorted order.
    SmallVector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;

    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      // Priority 1: Higher ALAP level first (critical path).
      if (alap_level[a] != alap_level[b])
        return alap_level[a] > alap_level[b];
      // Priority 2: Higher degree first.
      if (degree[a] != degree[b])
        return degree[a] > degree[b];
      // Priority 3: Original order (stability).
      return a < b;
    });

    return order;
  }

  /// Prints the placement grid.
  void printGrid(const SmallVector<TaskflowTaskOp> &tasks,
                 const SmallVector<TaskPlacement> &placements) {
    std::vector<std::vector<std::string>> grid(
        grid_rows_, std::vector<std::string>(grid_cols_, "  .  "));

    for (size_t idx = 0; idx < tasks.size(); ++idx) {
      for (const auto &pos : placements[idx].cgra_positions) {
        grid[pos.row][pos.col] = "  T" + std::to_string(idx) + " ";
      }
    }

    for (int r = 0; r < grid_rows_; ++r) {
      for (int c = 0; c < grid_cols_; ++c) {
        llvm::outs() << grid[r][c];
      }
      llvm::outs() << "\n";
    }
  }

  int grid_rows_;
  int grid_cols_;
  std::vector<std::vector<bool>> occupied_;
  MemoryMapper memory_mapper_;
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//
struct PlaceMCTOnCGRAPass
    : public PassWrapper<PlaceMCTOnCGRAPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlaceMCTOnCGRAPass)

  PlaceMCTOnCGRAPass() = default;

  StringRef getArgument() const override { return "place-mct-on-cgra"; }

  StringRef getDescription() const override {
    return "Places MCTs onto a 2D CGRA grid with adjacency optimization and "
           "memory mapping.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    constexpr int kDefaultGridRows = 4;
    constexpr int kDefaultGridCols = 4;
    CGRAPlacer placer(kDefaultGridRows, kDefaultGridCols);
    placer.place(func);
  }
};

} // namespace

namespace mlir {
namespace taskflow {

std::unique_ptr<Pass> createPlaceMCTOnCGRAPass() {
  return std::make_unique<PlaceMCTOnCGRAPass>();
}

} // namespace taskflow
} // namespace mlir
