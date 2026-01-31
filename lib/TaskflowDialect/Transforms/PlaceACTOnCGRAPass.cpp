//===- PlaceACTOnCGRAPass.cpp - ACT to CGRA Placement Pass ----------------===//
//
// This pass places Atomic Canonical Tasks (ACTs) onto a 2D CGRA grid：
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
// Task-Memory Graph
//===----------------------------------------------------------------------===//

struct MemoryNode;

/// Represents a Task node in the graph.
struct TaskNode {
  size_t id;
  TaskflowTaskOp op;
  int alap_level = 0;
  
  // Edges
  SmallVector<MemoryNode *> read_memrefs;
  SmallVector<MemoryNode *> write_memrefs;
  SmallVector<TaskNode *> ssa_users;
  SmallVector<TaskNode *> ssa_operands;

  // Placement result
  SmallVector<CGRAPosition> placement;

  TaskNode(size_t id, TaskflowTaskOp op) : id(id), op(op) {}
};

/// Represents a Memory node (MemRef) in the graph.
struct MemoryNode {
  Value memref;
  
  // Edges
  SmallVector<TaskNode *> readers;
  SmallVector<TaskNode *> writers;

  // Mapping result
  int assigned_sram_id = -1;

  MemoryNode(Value memref) : memref(memref) {}
};

/// The Task-Memory Dependency Graph.
class TaskMemoryGraph {
public:
  SmallVector<std::unique_ptr<TaskNode>> task_nodes;
  SmallVector<std::unique_ptr<MemoryNode>> memory_nodes;
  DenseMap<Value, MemoryNode *> memref_to_node;
  DenseMap<Operation *, TaskNode *> op_to_node;

  void build(func::FuncOp func) {
    // 1. Creates TaskNodes.
    size_t task_id = 0;
    func.walk([&](TaskflowTaskOp task) {
      auto node = std::make_unique<TaskNode>(task_id++, task);
      op_to_node[task] = node.get();
      task_nodes.push_back(std::move(node));
    });

    // 2. Creates MemoryNodes and defines Edges.
    for (auto &t_node : task_nodes) {
      // Memory Inputs (Reads)
      for (Value input : t_node->op.getMemoryInputs()) {
        MemoryNode *m_node = getOrCreateMemoryNode(input);
        t_node->read_memrefs.push_back(m_node);
        m_node->readers.push_back(t_node.get());
      }

      // Memory Outputs (Writes)
      for (Value output : t_node->op.getMemoryOutputs()) {
        MemoryNode *m_node = getOrCreateMemoryNode(output);
        t_node->write_memrefs.push_back(m_node);
        m_node->writers.push_back(t_node.get());
      }
    }

    // 3. Builds SSA Edges (Inter-Task Value Dependencies).
    // Identifies if a task uses a value produced by another task.
    for (auto &consumer_node : task_nodes) {
        // Iterate all operands for now to be safe.
        for (Value operand : consumer_node->op->getOperands()) {
            if (auto producer_op = operand.getDefiningOp<TaskflowTaskOp>()) {
                if (auto *producer_node = op_to_node[producer_op]) {
                    producer_node->ssa_users.push_back(consumer_node.get());
                    consumer_node->ssa_operands.push_back(producer_node);
                }
            }
        }
    }
  }

private:
  MemoryNode *getOrCreateMemoryNode(Value memref) {
    if (memref_to_node.count(memref))
      return memref_to_node[memref];
    
    auto node = std::make_unique<MemoryNode>(memref);
    MemoryNode *ptr = node.get();
    memref_to_node[memref] = ptr;
    memory_nodes.push_back(std::move(node));
    return ptr;
  }
};


//===----------------------------------------------------------------------===//
// CGRA Placer
//===----------------------------------------------------------------------===//
/// Places ACTs onto a 2D CGRA grid with memory mapping.
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
    // Builds Task-Memory Graph.
    TaskMemoryGraph graph;
    graph.build(func);

    if (graph.task_nodes.empty()) {
      llvm::errs() << "No tasks to place.\n";
      return;
    }

    // Computes ALAP Levels on the Task Graph.
    computeALAP(graph);

    // Sorts tasks by ALAP level (Critical Path First).
    SmallVector<TaskNode *> sorted_tasks;
    for (auto &node : graph.task_nodes) sorted_tasks.push_back(node.get());
    
    std::stable_sort(sorted_tasks.begin(), sorted_tasks.end(), 
        [](TaskNode *a, TaskNode *b) {
            return a->alap_level > b->alap_level; 
        });

    // Critical path priority placement:
    // 1. Computes ALAP level for each task (longest path to sink).
    // 2. Sorts tasks by: (a) ALAP level, (b) criticality, (c) degree.
    // 3. Places tasks in sorted order with heuristic scoring.
    // Placement Loop.
    for (TaskNode *task_node : sorted_tasks) {
      int cgra_count = 1;
      if (auto attr = task_node->op->getAttrOfType<IntegerAttr>("cgra_count")) {
        cgra_count = attr.getInt();
      }

      // Finds Best Placement.
      // Heuristic: Minimizes distance to:
      // 1. SSA Producers (that are already placed).
      // 2. SRAMs of Input MemRefs (if already assigned).
      TaskPlacement placement = findBestPlacement(task_node, cgra_count, graph);
      
      // Commits Placement.
      task_node->placement.push_back(placement.primary());
      // Handles multi-cgra if needed.
      for (size_t i = 1; i < placement.cgra_positions.size(); ++i) {
         task_node->placement.push_back(placement.cgra_positions[i]);
      }

      // Marks Occupied.
      for (const auto &pos : placement.cgra_positions) {
        if (pos.row >= 0 && pos.row < grid_rows_ && pos.col >= 0 && pos.col < grid_cols_)
            occupied_[pos.row][pos.col] = true;
      }

      // Maps Associated Memory Nodes.
      // For each MemRef this task touches, if not yet assigned to SRAM, assign to nearest.
      for (MemoryNode *mem_node : task_node->read_memrefs) {
        if (mem_node->assigned_sram_id == -1) {
            mem_node->assigned_sram_id = assignSRAM(mem_node, placement);
        }
      }
      for (MemoryNode *mem_node : task_node->write_memrefs) {
        if (mem_node->assigned_sram_id == -1) {
            mem_node->assigned_sram_id = assignSRAM(mem_node, placement);
        }
      }
    }

    // Annotates Result.
    OpBuilder builder(func.getContext());
    for (auto &task_node : graph.task_nodes) {
        if (task_node->placement.empty()) continue;
        CGRAPosition pos = task_node->placement[0];
        task_node->op->setAttr("cgra_row", builder.getI32IntegerAttr(pos.row));
        task_node->op->setAttr("cgra_col", builder.getI32IntegerAttr(pos.col));
        task_node->op->setAttr("cgra_count", builder.getI32IntegerAttr(task_node->placement.size()));
    }
  }

private:
  /// Assigns a memref to the closest SRAM near the given task position.
  /// TODO: Integrate with Arch Spec to map logical SRAM IDs (row*100 + col) to
  /// physical hardware Block IDs, especially for shared or asymmetric SRAMs.
  int assignSRAM(MemoryNode *mem_node, const TaskPlacement &placement) {
    if (mem_node->assigned_sram_id != -1)
      return mem_node->assigned_sram_id;

    // Assigns to a new SRAM near the task's primary CGRA.
    CGRAPosition pos = placement.primary();
    int sram_id = pos.row * 100 + pos.col; // Simple encoding: row*100 + col.
    mem_node->assigned_sram_id = sram_id;
    return sram_id;
  }

private:
  /// Finds best placement for a task requiring cgra_count CGRAs.
  /// TODO: Implement a block-search algorithm for tasks with cgra_count > 1 to
  /// find contiguous rectangular regions instead of single tiles.
  TaskPlacement findBestPlacement(TaskNode *task_node, int cgra_count,
                                  TaskMemoryGraph &graph) {
    int best_score = INT_MIN;
    TaskPlacement best_placement;

    // Baseline: For cgra_count=1, finds single best position.
    for (int r = 0; r < grid_rows_; ++r) {
      for (int c = 0; c < grid_cols_; ++c) {
        if (occupied_[r][c])
          continue;

        TaskPlacement candidate;
        candidate.cgra_positions.push_back({r, c});

        int score = computeScore(task_node, candidate, graph);
        if (score > best_score) {
          best_score = score;
          best_placement = candidate;
        }
      }
    }

    // Error handling: No available position found (grid over-subscribed).
    if (best_placement.cgra_positions.empty()) {
      llvm::errs() << "Warning: No available CGRA position for task "
                   << task_node->id << ". Grid is over-subscribed (" << grid_rows_
                   << "x" << grid_cols_ << " grid with all cells occupied).\n";
      // Fallback: Assign to position (0,0) with a warning.
      best_placement.cgra_positions.push_back({0, 0});
    }

    return best_placement;
  }

  /// Computes placement score based on Task-Memory Graph.
  /// TODO: Introduce explicit 'direct_wires' attributes in the IR for
  /// downstream hardware generators to configure fast bypass paths between
  /// adjacent PEs with dependencies.
  ///
  /// Score = α·SSA_Dist + β·Mem_Dist + γ·Balance
  ///
  /// SSA_Dist: Minimize distance to placed SSA predecessors (ssa_operands).
  /// Mem_Dist: Minimize distance to assigned SRAMs for read/write memrefs.
  int computeScore(TaskNode *task_node, const TaskPlacement &placement,
                   TaskMemoryGraph &graph) {
    // Weight constants (tunable).
    constexpr int kAlpha = 10;   // SSA proximity weight
    constexpr int kBeta = 50;    // Memory proximity weight (High priority)
    constexpr int kGamma = 20;   // Load balance weight

    int ssa_score = 0;
    int mem_score = 0;
    int bal_score = 0;
    
    CGRAPosition current_pos = placement.primary();

    // 1. SSA Proximity (Predecessors)
    for (TaskNode *producer : task_node->ssa_operands) {
        if (!producer->placement.empty()) {
            int dist = current_pos.manhattanDistance(producer->placement[0]);
            // Uses negative distance to penalize far-away placements.
            ssa_score -= dist;
        }
    }

    // 2. Memory Proximity
    // For Read MemRefs
    for (MemoryNode *mem : task_node->read_memrefs) {
        if (mem->assigned_sram_id != -1) {
            // SRAM ID encoding: row*100 + col
            int sram_r = mem->assigned_sram_id / 100;
            int sram_c = mem->assigned_sram_id % 100;
            CGRAPosition sram_pos{sram_r, sram_c};
            int dist = current_pos.manhattanDistance(sram_pos);
            mem_score -= dist;
        }
    }
    // For Write MemRefs
    // If we write to a memory that is already assigned (e.g. read by previous task),
    // we want to be close to it too.
    for (MemoryNode *mem : task_node->write_memrefs) {
         if (mem->assigned_sram_id != -1) {
            int sram_r = mem->assigned_sram_id / 100;
            int sram_c = mem->assigned_sram_id % 100;
            CGRAPosition sram_pos{sram_r, sram_c};
            int dist = current_pos.manhattanDistance(sram_pos);
            mem_score -= dist;
        }
    }

    // 3. Load Balance
    // Prefers less crowded rows/cols.
    int row_count = 0, col_count = 0;
    for (int c = 0; c < grid_cols_; ++c) { if (occupied_[current_pos.row][c]) row_count++; }
    for (int r = 0; r < grid_rows_; ++r) { if (occupied_[r][current_pos.col]) col_count++; }
    bal_score = (grid_cols_ - row_count) + (grid_rows_ - col_count);

    return kAlpha * ssa_score + kBeta * mem_score + kGamma * bal_score;
  }

  /// Computes ALAP levels for efficient scheduling order.
  void computeALAP(TaskMemoryGraph &graph) {
    // 1. Calculates in-degrees for topological sort simulation.
    DenseMap<TaskNode*, int> in_degree;
    for (auto &node : graph.task_nodes) {
        for (TaskNode *user : node->ssa_users) in_degree[user]++;
    }
    
    // 2. DFS for longest path from node to any sink (ALAP Level).
    DenseMap<TaskNode*, int> memo;
    for (auto &node : graph.task_nodes) {
        node->alap_level = calculateLevel(node.get(), memo);
    }
  }

  int calculateLevel(TaskNode *node, DenseMap<TaskNode*, int> &memo) {
    if (memo.count(node)) return memo[node];

    int max_child_level = 0;
    for (TaskNode *child : node->ssa_users) {
        max_child_level = std::max(max_child_level, calculateLevel(child, memo) + 1);
    }

    // Check memory dependencies too (Producer -> Mem -> Consumer)
    for (MemoryNode *mem : node->write_memrefs) {
        for (TaskNode *reader : mem->readers) {
            if (reader != node)
                max_child_level = std::max(max_child_level, calculateLevel(reader, memo) + 1);
        }
    }

    return memo[node] = max_child_level;
  }



  int grid_rows_;
  int grid_cols_;
  std::vector<std::vector<bool>> occupied_;
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//
struct PlaceACTOnCGRAPass
    : public PassWrapper<PlaceACTOnCGRAPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlaceACTOnCGRAPass)

  PlaceACTOnCGRAPass() = default;

  StringRef getArgument() const override { return "place-act-on-cgra"; }

  StringRef getDescription() const override {
    return "Places ACTs onto a 2D CGRA grid with adjacency optimization and "
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

std::unique_ptr<Pass> createPlaceACTOnCGRAPass() {
  return std::make_unique<PlaceACTOnCGRAPass>();
}

} // namespace taskflow
} // namespace mlir
