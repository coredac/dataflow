//===- MapCTOnCGRAArrayPass.cpp - CT to CGRA Mapping Pass ----------------===//
//
// This pass maps Canonical Tasks (CTs) onto a 2D CGRA grid array:
// 1. Places tasks with SSA dependencies on adjacent CGRAs.
// 2. Assigns memrefs to SRAMs (each MemRef is assigned to exactly one SRAM,
//    determined by proximity to the task that first accesses it).
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
  
  // Edges (Note: read/write naming refers to taskflow memory_inputs/outputs)
  SmallVector<MemoryNode *> read_memrefs;  // taskflow.task memory_inputs (readiness triggers)
  SmallVector<MemoryNode *> write_memrefs; // taskflow.task memory_outputs (produce readiness)
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

    // 2. Creates MemoryNodes using ORIGINAL memrefs (canonical identity).
    // Uses original_read_memrefs/original_write_memrefs to ensure aliased
    // memories share the same MemoryNode.
    for (auto &t_node : task_nodes) {
      // Uses original_read_memrefs for canonical memory identity.
      for (Value orig_memref : t_node->op.getOriginalReadMemrefs()) {
        MemoryNode *m_node = getOrCreateMemoryNode(orig_memref);
        t_node->read_memrefs.push_back(m_node);
        m_node->readers.push_back(t_node.get());
      }

      // Uses original_write_memrefs for canonical memory identity.
      for (Value orig_memref : t_node->op.getOriginalWriteMemrefs()) {
        MemoryNode *m_node = getOrCreateMemoryNode(orig_memref);
        t_node->write_memrefs.push_back(m_node);
        m_node->writers.push_back(t_node.get());
      }
    }

    // 3. Builds SSA Edges (Inter-Task Value Dependencies).
    // Identifies if a task uses a value produced by another task.
    for (auto &consumer_node : task_nodes) {
        // Interates all operands for now to be safe.
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

/// Prints the Task-Memory graph in DOT format for visualization.
void printGraphDOT(TaskMemoryGraph &graph, llvm::raw_ostream &os) {
  os << "digraph TaskMemGraph {\n";
  os << "  rankdir=TB;\n";
  // Task nodes (circles).
  for (auto &t : graph.task_nodes) {
    os << "  T" << t->id << " [shape=circle, label=\"" << t->id << "\"];\n";
  }
  // Memory nodes (rectangles).
  for (size_t i = 0; i < graph.memory_nodes.size(); ++i) {
    os << "  M" << i << " [shape=box, label=\"mem" << i << "\"];\n";
  }
  // Edges: Task -> Memory (write) and Memory -> Task (read).
  for (auto &t : graph.task_nodes) {
    for (size_t i = 0; i < graph.memory_nodes.size(); ++i) {
      MemoryNode *m = graph.memory_nodes[i].get();
      for (auto *writer : m->writers) {
        if (writer == t.get()) {
          os << "  T" << t->id << " -> M" << i << ";\n";
        }
      }
      for (auto *reader : m->readers) {
        if (reader == t.get()) {
          os << "  M" << i << " -> T" << t->id << ";\n";
        }
      }
    }
  }
  os << "}\n";
}


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

    // Prints graph visualization to stderr for debugging.
    // llvm::errs() << "\n=== Task-Memory Graph (DOT format) ===\n";
    // printGraphDOT(graph, llvm::errs());
    // llvm::errs() << "=== Graph Stats: " << graph.task_nodes.size() << " tasks, "
    //              << graph.memory_nodes.size() << " memories ===\n\n";

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
    // Iterative Refinement Loop (Coordinate Descent).
    // Alternates between Task Placement (Phase 1) and SRAM Assignment (Phase 2).
    constexpr int kMaxIterations = 10;
    
    llvm::errs() << "\n=== Starting Iterative Placement (Max " << kMaxIterations << ") ===\n";

    for (int iter = 0; iter < kMaxIterations; ++iter) {
        // Phase 1: Place Tasks (assuming fixed SRAMs).
        if (iter > 0) resetTaskPlacements(graph);

        for (TaskNode *task_node : sorted_tasks) {
          int cgra_count = 1;
          if (auto attr = task_node->op->getAttrOfType<IntegerAttr>("cgra_count")) {
            cgra_count = attr.getInt();
          }

          // Finds Best Placement using SRAM positions from previous iter (or -1/default).
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
        }

        // Phase 2: Assign SRAMs (assuming fixed Tasks).
        bool sram_moved = assignAllSRAMs(graph);
        
        llvm::errs() << "Iter " << iter << ": SRAMs moved = " << (sram_moved ? "Yes" : "No") << "\n";

        // Convergence Check.
        // If SRAMs didn't move, it means task placement based on them likely won't change either.
        if (iter > 0 && !sram_moved) {
            llvm::errs() << "Converged at iteration " << iter << ".\n";
            break; 
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
  /// Clears task placement and occupied grid.
  void resetTaskPlacements(TaskMemoryGraph &graph) {
    for (auto &task : graph.task_nodes) {
        task->placement.clear();
    }
    // Clears grid.
    for (int r = 0; r < grid_rows_; ++r) {
        std::fill(occupied_[r].begin(), occupied_[r].end(), false);
    }
  }

  /// Assigns all memory nodes to SRAMs based on centroid of accessing tasks.
  /// Returns true if any SRAM assignment changed.
  bool assignAllSRAMs(TaskMemoryGraph &graph) {
    bool changed = false;
    for (auto &mem_node : graph.memory_nodes) {
      // Computes centroid of all tasks that access this memory.
      int total_row = 0, total_col = 0, count = 0;
      for (TaskNode *reader : mem_node->readers) {
        if (!reader->placement.empty()) {
          total_row += reader->placement[0].row;
          total_col += reader->placement[0].col;
          count++;
        }
      }
      for (TaskNode *writer : mem_node->writers) {
        if (!writer->placement.empty()) {
          total_row += writer->placement[0].row;
          total_col += writer->placement[0].col;
          count++;
        }
      }
      
      int new_sram_id = 0;
      if (count > 0) {
        // Rounds to the nearest integer.
        int avg_row = (total_row + count / 2) / count;
        int avg_col = (total_col + count / 2) / count;
        new_sram_id = avg_row * 100 + avg_col;
      } else {
        new_sram_id = 0; // Default fallback
      }

      if (mem_node->assigned_sram_id != new_sram_id) {
        mem_node->assigned_sram_id = new_sram_id;
        changed = true;
      }
    }
    return changed;
  }


  /// Finds best placement for a task.
  /// TODO: Currently defaults to single-CGRA placement. Multi-CGRA binding logic
  /// (cgra_count > 1) is experimental/placeholder and should ideally be handled 
  /// by an upstream resource binding pass.
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

  /// Computes ALAP levels considering both SSA and memory dependencies.
  void computeALAP(TaskMemoryGraph &graph) {
    // DFS for longest path from node to any sink (ALAP Level).
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

    // Checks memory dependencies too (Producer -> Mem -> Consumer).
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
struct MapCTOnCGRAArrayPass
    : public PassWrapper<MapCTOnCGRAArrayPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapCTOnCGRAArrayPass)

  MapCTOnCGRAArrayPass() = default;

  StringRef getArgument() const override { return "map-ct-on-cgra-array"; }

  StringRef getDescription() const override {
    return "Maps Canonical Tasks (CTs) onto a 2D CGRA grid with adjacency "
           "optimization and memory mapping.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    constexpr int kDefaultGridRows = 3;
    constexpr int kDefaultGridCols = 3;
    CGRAPlacer placer(kDefaultGridRows, kDefaultGridCols);
    placer.place(func);
  }
};

} // namespace

namespace mlir {
namespace taskflow {

std::unique_ptr<Pass> createMapCTOnCGRAArrayPass() {
  return std::make_unique<MapCTOnCGRAArrayPass>();
}

} // namespace taskflow
} // namespace mlir
