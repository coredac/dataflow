//===- AllocateCgraToTaskPass.cpp - Task to CGRA Mapping Pass ----------------===//
//
// This pass maps Taskflow tasks onto a 2D CGRA grid array:
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
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===----------------------------------------------------------------------===//
// CGRA Grid Position
//===----------------------------------------------------------------------===//
// Represents a position on the 2D CGRA grid.
struct CgraPosition {
  int row;
  int col;

  bool operator==(const CgraPosition &other) const {
    return row == other.row && col == other.col;
  }

  bool operator!=(const CgraPosition &other) const {
    return !(*this == other);
  }

  // Computes Manhattan distance to another position.
  int manhattanDistance(const CgraPosition &other) const {
    return std::abs(row - other.row) + std::abs(col - other.col);
  }

  // Checks if adjacent (Manhattan distance = 1).
  bool isAdjacent(const CgraPosition &other) const {
    return manhattanDistance(other) == 1;
  }
};

//===----------------------------------------------------------------------===//
// Task Placement Info
//===----------------------------------------------------------------------===//
// Stores placement info for a task: can span multiple combined CGRAs.
struct TaskPlacement {
  SmallVector<CgraPosition> cgra_positions; // CGRAs assigned to this task.

  // Returns the primary (first) position.
  CgraPosition primary() const {
    return cgra_positions.empty() ? CgraPosition{-1, -1} : cgra_positions[0];
  }

  // Returns the number of CGRAs assigned.
  size_t cgraCount() const { return cgra_positions.size(); }

  // Checks if any CGRA in this task is adjacent to any in other task.
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

// Represents a Task node in the graph.
struct TaskNode {
  size_t id;
  TaskflowTaskOp op;
  int dependency_depth = 0;  // Longest path to any sink in the dependency graph.
  
  // Edges based on original memory access.
  SmallVector<MemoryNode *> read_memrefs;  // Original read memrefs.
  SmallVector<MemoryNode *> write_memrefs; // Original write memrefs.
  SmallVector<TaskNode *> ssa_users;
  SmallVector<TaskNode *> ssa_operands;

  // Placement result
  SmallVector<CgraPosition> placement;

  TaskNode(size_t id, TaskflowTaskOp op) : id(id), op(op) {}
};

// Represents a Memory node (MemRef) in the graph.
struct MemoryNode {
  Value memref;
  
  // Edges.
  SmallVector<TaskNode *> readers;
  SmallVector<TaskNode *> writers;
  
  // Mapping result.
  std::optional<CgraPosition> assigned_sram_pos;

  MemoryNode(Value memref) : memref(memref) {}
};

// The Task-Memory Dependency Graph.
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
        // Iterates all operands for now to be safe.
        for (Value operand : consumer_node->op.getValueInputs()) {
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
    if (memref_to_node.count(memref)) {
      return memref_to_node[memref];
    }
    
    auto node = std::make_unique<MemoryNode>(memref);
    MemoryNode *ptr = node.get();
    memref_to_node[memref] = ptr;
    memory_nodes.push_back(std::move(node));
    return ptr;
  }
};



//===----------------------------------------------------------------------===//
// Task Mapper
//===----------------------------------------------------------------------===//
// Maps a task-memory graph onto a 2D CGRA grid.

class TaskMapper {
public:
  TaskMapper(int grid_rows, int grid_cols)
      : grid_rows_(grid_rows), grid_cols_(grid_cols) {
    occupied_.resize(grid_rows_);
    for (auto &row : occupied_) {
      row.resize(grid_cols_, false);
    }
  }

  // Places all tasks and performs memory mapping.
  void place(func::FuncOp func) {
    SmallVector<TaskflowTaskOp> tasks;
    func.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });

    if (tasks.empty()) {
      llvm::errs() << "No tasks to place.\n";
      return;
    }


    // Builds Task-Memory Graph.
    TaskMemoryGraph graph;
    graph.build(func);



    if (graph.task_nodes.empty()) {
      llvm::errs() << "No tasks to place.\n";
      return;
    }

    // Computes Dependency Depth for each task.
    // Dependency depth = longest path from this node to any sink in the
    // dependency graph (considering both SSA and memory edges). Tasks with
    // higher depth are more "critical" and are placed first to ensure their
    // dependent chains have good locality.
    computeDependencyDepth(graph);

    // Sorts tasks by dependency depth (Critical Path First).
    SmallVector<TaskNode *> sorted_tasks;
    for (auto &node : graph.task_nodes) sorted_tasks.push_back(node.get());
    
    std::stable_sort(sorted_tasks.begin(), sorted_tasks.end(), 
        [](TaskNode *a, TaskNode *b) {
            return a->dependency_depth > b->dependency_depth; 
        });

    // Critical-path-first placement:
    // 1. Computes dependency depth for each task (longest path to sink).
    // 2. Sorts tasks by dependency depth (higher = more critical).
    // 3. Places tasks in sorted order with heuristic scoring.
    // Iterative Refinement Loop (Coordinate Descent).
    // Alternates between Task Placement (Phase 1) and SRAM Assignment (Phase 2).
    constexpr int kMaxIterations = 10;
  
    for (int iter = 0; iter < kMaxIterations; ++iter) {
        // Phase 1: Place Tasks (assuming fixed SRAMs).
        if (iter > 0) {
            resetTaskPlacements(graph);
        }

        for (TaskNode *task_node : sorted_tasks) {
          int cgra_count = 1;
          if (auto attr = task_node->op->getAttrOfType<IntegerAttr>("cgra_count")) {
            cgra_count = attr.getInt();
          }

          // Finds best placement using SRAM positions from previous iter (or -1/default).
          TaskPlacement placement = findBestPlacement(task_node, cgra_count, graph);

          // If the requested cgra_count doesn't fit, fall back to cgra_count-1
          // (i.e. reject the extra CGRA and keep previous allocation).
          if (placement.cgra_positions.empty() && cgra_count > 1) {
            int fallback = cgra_count - 1;
            llvm::errs() << "[AllocateCgraToTask] Cannot place "
                         << task_node->op.getTaskName()
                         << " with cgra_count=" << cgra_count
                         << ", falling back to " << fallback << "\n";
            placement = findBestPlacement(task_node, fallback, graph);
          }

          // Commits Placement.
          task_node->placement.push_back(placement.primary());
          for (size_t i = 1; i < placement.cgra_positions.size(); ++i) {
             task_node->placement.push_back(placement.cgra_positions[i]);
          }

          // Marks occupied.
          for (const auto &pos : placement.cgra_positions) {
            if (pos.row >= 0 && pos.row < grid_rows_ && pos.col >= 0 && pos.col < grid_cols_) {
                occupied_[pos.row][pos.col] = true;
            }
          }
        }

        // Phase 2: Assign SRAMs (assuming fixed tasks).
        bool sram_moved = assignAllSRAMs(graph);
        


        // Convergence Check.
        // If SRAMs didn't move, it means task placement based on them likely won't change either.
        if (iter > 0 && !sram_moved) {
            break; 
        }
    }



    // Annotates result.
    OpBuilder builder(func.getContext());
    for (auto &task_node : graph.task_nodes) {
        if (task_node->placement.empty()) {
            continue;
        }
        
        SmallVector<NamedAttribute, 4> mapping_attrs;

        // 1. CGRA positions.
        SmallVector<Attribute> pos_attrs;
        for (const auto &pos : task_node->placement) {
            SmallVector<NamedAttribute, 2> coord_attrs;
            coord_attrs.push_back(NamedAttribute(
                StringAttr::get(func.getContext(), "row"),
                builder.getI32IntegerAttr(pos.row)));
            coord_attrs.push_back(NamedAttribute(
                StringAttr::get(func.getContext(), "col"),
                builder.getI32IntegerAttr(pos.col)));
            pos_attrs.push_back(DictionaryAttr::get(func.getContext(), coord_attrs));
        }
        mapping_attrs.push_back(NamedAttribute(
            StringAttr::get(func.getContext(), "cgra_positions"),
            builder.getArrayAttr(pos_attrs)));

        // 2. Reads SRAM Locations.
        SmallVector<Attribute> read_sram_attrs;
        for (MemoryNode *mem : task_node->read_memrefs) {
            if (mem->assigned_sram_pos) {
                SmallVector<NamedAttribute, 2> sram_coord;
                sram_coord.push_back(NamedAttribute(StringAttr::get(func.getContext(), "row"), builder.getI32IntegerAttr(mem->assigned_sram_pos->row)));
                sram_coord.push_back(NamedAttribute(StringAttr::get(func.getContext(), "col"), builder.getI32IntegerAttr(mem->assigned_sram_pos->col)));
                read_sram_attrs.push_back(DictionaryAttr::get(func.getContext(), sram_coord));
            }
        }
        mapping_attrs.push_back(NamedAttribute(
            StringAttr::get(func.getContext(), "read_sram_locations"),
            builder.getArrayAttr(read_sram_attrs)));

        // 3. Writes SRAM Locations.
        SmallVector<Attribute> write_sram_attrs;
        for (MemoryNode *mem : task_node->write_memrefs) {
            if (mem->assigned_sram_pos) {
              SmallVector<NamedAttribute, 2> sram_coord;
              sram_coord.push_back(NamedAttribute(StringAttr::get(func.getContext(), "row"), builder.getI32IntegerAttr(mem->assigned_sram_pos->row)));
              sram_coord.push_back(NamedAttribute(StringAttr::get(func.getContext(), "col"), builder.getI32IntegerAttr(mem->assigned_sram_pos->col)));

              write_sram_attrs.push_back(DictionaryAttr::get(func.getContext(), sram_coord));
            }
        }
        mapping_attrs.push_back(NamedAttribute(
            StringAttr::get(func.getContext(), "write_sram_locations"),
            builder.getArrayAttr(write_sram_attrs)));

        // Sets Attribute.
        task_node->op->setAttr("task_mapping_info", DictionaryAttr::get(func.getContext(), mapping_attrs));
    }
  }

private:
  // Clears task placement and occupied grid.
  void resetTaskPlacements(TaskMemoryGraph &graph) {
    for (auto &task : graph.task_nodes) {
        task->placement.clear();
    }
    // Clears grid.
    for (int r = 0; r < grid_rows_; ++r) {
        std::fill(occupied_[r].begin(), occupied_[r].end(), false);
    }
  }

  // Assigns all memory nodes to SRAMs based on centroid of accessing tasks.
  // Returns true if any SRAM assignment changed.
  bool assignAllSRAMs(TaskMemoryGraph &graph) {
    bool changed = false;
    for (auto &mem_node : graph.memory_nodes) {
      // Computes centroid of all tasks that access this memory.
      int total_row = 0, total_col = 0, count = 0;
      for (TaskNode *reader : mem_node->readers) {
        for (const auto &pos : reader->placement) {
          total_row += pos.row;
          total_col += pos.col;
          count++;
        }
      }
      for (TaskNode *writer : mem_node->writers) {
        for (const auto &pos : writer->placement) {
          total_row += pos.row;
          total_col += pos.col;
          count++;
        }
      }
      
      std::optional<CgraPosition> new_sram_pos;
      if (count > 0) {
        // Rounds to the nearest integer.
        int avg_row = (total_row + count / 2) / count;
        int avg_col = (total_col + count / 2) / count;
        new_sram_pos = CgraPosition{avg_row, avg_col};
      }

      if (mem_node->assigned_sram_pos != new_sram_pos) {
        mem_node->assigned_sram_pos = new_sram_pos;
        changed = true;
      }
    }
    return changed;
  }


  // Parses a tile_shape string like "2x2" or "2x2[(0,0)(1,0)(0,1)]".
  // Returns a list of (col, row) offsets relative to the placement origin.
  // For rectangular shapes "NxM", generates all NxM positions.
  // For non-rectangular shapes with explicit positions, uses the listed coords.
  SmallVector<std::pair<int, int>> parseTileShapeOffsets(
      StringRef tile_shape, int cgra_count) {
    SmallVector<std::pair<int, int>> offsets;

    if (tile_shape.empty() || cgra_count <= 1) {
      offsets.push_back({0, 0});
      return offsets;
    }

    // Checks for explicit position list: "NxM[(c0,r0)(c1,r1)...]"
    size_t bracket_pos = tile_shape.find('[');
    if (bracket_pos != StringRef::npos) {
      StringRef positions_str = tile_shape.substr(bracket_pos);
      // Parses each (c,r) pair.
      size_t pos = 0;
      while (pos < positions_str.size()) {
        size_t open = positions_str.find('(', pos);
        if (open == StringRef::npos) break;
        size_t close = positions_str.find(')', open);
        if (close == StringRef::npos) break;
        StringRef pair_str = positions_str.slice(open + 1, close);
        auto [col_str, row_str] = pair_str.split(',');
        int col_off = 0, row_off = 0;
        col_str.getAsInteger(10, col_off);
        row_str.getAsInteger(10, row_off);
        offsets.push_back({col_off, row_off});
        pos = close + 1;
      }
    } else {
      // Rectangular shape: "NxM" — parse rows × cols.
      auto [rows_str, cols_str] = tile_shape.split('x');
      int rows = 1, cols = 1;
      rows_str.getAsInteger(10, rows);
      cols_str.getAsInteger(10, cols);
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          offsets.push_back({c, r});
        }
      }
    }

    // Sanity: if parsing failed, assert.
    assert(!offsets.empty() && "tile_shape parsing yielded empty offsets");
    return offsets;
  }

  // Tries placing a shape (given as col/row offsets) at every grid origin.
  // Returns the best-scoring valid placement, or empty if none fits.
  TaskPlacement tryPlaceShape(
      TaskNode *task_node,
      const SmallVector<std::pair<int, int>> &shape_offsets,
      TaskMemoryGraph &graph) {
    int best_score = INT_MIN;
    TaskPlacement best_placement;
    for (int r = 0; r < grid_rows_; ++r) {
      for (int c = 0; c < grid_cols_; ++c) {
        bool valid = true;
        TaskPlacement candidate;
        for (auto &[col_off, row_off] : shape_offsets) {
          int pr = r + row_off;
          int pc = c + col_off;
          if (pr < 0 || pr >= grid_rows_ || pc < 0 || pc >= grid_cols_ ||
              occupied_[pr][pc]) {
            valid = false;
            break;
          }
          candidate.cgra_positions.push_back({pr, pc});
        }
        if (!valid) continue;
        int score = computeScore(task_node, candidate, graph);
        if (score > best_score) {
          best_score = score;
          best_placement = candidate;
        }
      }
    }
    return best_placement;
  }

  // Generates all rectangular shapes (as col/row offset lists) of size k.
  // E.g. k=4 → 1×4, 2×2, 4×1.
  SmallVector<SmallVector<std::pair<int, int>>> getRectShapes(int k) {
    SmallVector<SmallVector<std::pair<int, int>>> shapes;
    for (int rows = 1; rows <= k; ++rows) {
      if (k % rows != 0) continue;
      int cols = k / rows;
      SmallVector<std::pair<int, int>> offsets;
      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
          offsets.push_back({c, r}); // {col_off, row_off}
      shapes.push_back(offsets);
    }
    return shapes;
  }

  // Searches all connected non-rectangular shapes of size k on the grid
  // and returns the best-scoring valid placement, or empty if none found.
  TaskPlacement tryNonRectShapes(TaskNode *task_node, int k,
                                 TaskMemoryGraph &graph) {
    std::set<uint64_t> visited_masks;
    int best_score = INT_MIN;
    TaskPlacement best_placement;

    std::function<void(SmallVector<CgraPosition> &, uint64_t)> search =
        [&](SmallVector<CgraPosition> &current, uint64_t mask) {
          if ((int)current.size() == k) {
            if (visited_masks.insert(mask).second) {
              TaskPlacement candidate;
              candidate.cgra_positions = current;
              int score = computeScore(task_node, candidate, graph);
              if (score > best_score) {
                best_score = score;
                best_placement = candidate;
              }
            }
            return;
          }
          constexpr int dr[] = {-1, 1, 0, 0};
          constexpr int dc[] = {0, 0, -1, 1};
          for (size_t i = 0; i < current.size(); ++i) {
            auto pos = current[i];
            for (int d = 0; d < 4; ++d) {
              int nr = pos.row + dr[d];
              int nc = pos.col + dc[d];
              if (nr >= 0 && nr < grid_rows_ && nc >= 0 && nc < grid_cols_ &&
                  !occupied_[nr][nc]) {
                uint64_t bit = 1ULL << (nr * grid_cols_ + nc);
                if ((mask & bit) == 0) {
                  current.push_back({nr, nc});
                  search(current, mask | bit);
                  current.pop_back();
                }
              }
            }
          }
        };

    for (int r = 0; r < grid_rows_; ++r) {
      for (int c = 0; c < grid_cols_; ++c) {
        if (!occupied_[r][c]) {
          SmallVector<CgraPosition> start = {{r, c}};
          search(start, 1ULL << (r * grid_cols_ + c));
        }
      }
    }
    return best_placement;
  }

  // Finds best placement for a task on the CGRA grid.
  //
  // Search order:
  //   1. Try all rectangular shapes of size cgra_count.
  //   2. If none fits, try all connected non-rectangular shapes of size cgra_count.
  //   3. If still nothing, return empty (caller handles fallback to cgra_count-1).
  TaskPlacement findBestPlacement(TaskNode *task_node, int cgra_count,
                                  TaskMemoryGraph &graph) {
    // 1. Rectangular shapes.
    for (auto &shape : getRectShapes(cgra_count)) {
      TaskPlacement p = tryPlaceShape(task_node, shape, graph);
      if (!p.cgra_positions.empty()) return p;
    }

    // 2. Non-rectangular connected shapes.
    if (cgra_count > 1) {
      TaskPlacement p = tryNonRectShapes(task_node, cgra_count, graph);
      if (!p.cgra_positions.empty()) return p;
    }

    // Nothing fits — return empty so caller can decide.
    return {};
  }

  // Computes placement score based on Task-Memory Graph.
  // For multi-CGRA placements, uses the minimum distance from any position
  // in the placement to the target, since adjacent CGRAs can communicate
  // via fast bypass paths.
  //
  // Score = α·SSA_Dist + β·Mem_Dist.
  //
  // SSA_Dist: Minimize distance to placed SSA predecessors (ssa_operands).
  // Mem_Dist: Minimize distance to assigned SRAMs for read/write memrefs.
  int computeScore(TaskNode *task_node, const TaskPlacement &placement,
                   TaskMemoryGraph &graph) {
    // Weight constants (tunable).
    constexpr int kAlpha = 10;   // SSA proximity weight.
    constexpr int kBeta = 50;    // Memory proximity weight (high priority).

    int ssa_score = 0;
    int mem_score = 0;

    // Helper: minimum Manhattan distance between any position in this
    // placement and any position in another task's placement.
    auto minDistToPlacement = [&](const SmallVector<CgraPosition> &other) -> int {
      int min_dist = INT_MAX;
      for (const auto &pos : placement.cgra_positions) {
        for (const auto &opos : other) {
          min_dist = std::min(min_dist, pos.manhattanDistance(opos));
        }
      }
      return min_dist;
    };

    // Helper: minimum Manhattan distance from any position in this placement
    // to a single target position.
    auto minDistToTarget = [&](const CgraPosition &target) -> int {
      int min_dist = INT_MAX;
      for (const auto &pos : placement.cgra_positions) {
        min_dist = std::min(min_dist, pos.manhattanDistance(target));
      }
      return min_dist;
    };

    // 1. SSA proximity (predecessors & successors).
    for (TaskNode *producer : task_node->ssa_operands) {
      if (!producer->placement.empty()) {
        int dist = minDistToPlacement(producer->placement);
        ssa_score -= dist;
      }
    }
    for (TaskNode *consumer : task_node->ssa_users) {
      if (!consumer->placement.empty()) {
        int dist = minDistToPlacement(consumer->placement);
        ssa_score -= dist;
      }
    }

    // 2. Memory proximity.
    for (MemoryNode *mem : task_node->read_memrefs) {
      if (mem->assigned_sram_pos) {
        int dist = minDistToTarget(*mem->assigned_sram_pos);
        mem_score -= dist;
      }
    }
    for (MemoryNode *mem : task_node->write_memrefs) {
      if (mem->assigned_sram_pos) {
        int dist = minDistToTarget(*mem->assigned_sram_pos);
        mem_score -= dist;
      }
    }

    return kAlpha * ssa_score + kBeta * mem_score;
  }

  // Computes dependency depth for all tasks in the graph.
  //
  // Dependency depth = longest path from this node to any sink node in the
  // dependency graph (via SSA or memory edges).
  //
  // Tasks with higher dependency depth have longer chains of dependent tasks
  // after them. By placing these tasks first:
  // 1. They get priority access to good grid positions.
  // 2. Their dependent tasks can then be positioned adjacent to them,
  //    minimizing inter-task communication distance.
  void computeDependencyDepth(TaskMemoryGraph &graph) {
    DenseMap<TaskNode*, int> depth_cache;
    for (auto &node : graph.task_nodes) {
        node->dependency_depth = calculateDepth(node.get(), depth_cache);
    }
  }

  // Recursively calculates dependency depth for a single task.
  int calculateDepth(TaskNode *node, DenseMap<TaskNode*, int> &depth_cache) {
    if (depth_cache.count(node)) {
        return depth_cache[node];
    }

    int max_child_depth = 0;
    // SSA dependencies.
    for (TaskNode *child : node->ssa_users) {
        max_child_depth = std::max(max_child_depth, calculateDepth(child, depth_cache) + 1);
    }

    // Memory dependencies (Producer -> Mem -> Consumer).
    for (MemoryNode *mem : node->write_memrefs) {
        for (TaskNode *reader : mem->readers) {
            if (reader != node) {
                max_child_depth = std::max(max_child_depth, calculateDepth(reader, depth_cache) + 1);
            }
        }
    }

    return depth_cache[node] = max_child_depth;
  }



  int grid_rows_;
  int grid_cols_;
  std::vector<std::vector<bool>> occupied_;
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//
struct AllocateCgraToTaskPass
    : public PassWrapper<AllocateCgraToTaskPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AllocateCgraToTaskPass)

  AllocateCgraToTaskPass() = default;

  StringRef getArgument() const override { return "allocate-cgra-to-task"; }

  StringRef getDescription() const override {
    return "Maps Taskflow tasks onto a 2D CGRA grid with adjacency "
           "optimization and memory mapping.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    constexpr int kDefaultGridRows = 4;
    constexpr int kDefaultGridCols = 4;
    TaskMapper mapper(kDefaultGridRows, kDefaultGridCols);
    mapper.place(func);
  }
};

} // namespace

namespace mlir {
namespace taskflow {

std::unique_ptr<Pass> createAllocateCgraToTaskPass() {
  return std::make_unique<AllocateCgraToTaskPass>();
}

void runAllocateCgraToTask(func::FuncOp func, int grid_rows, int grid_cols) {
  TaskMapper mapper(grid_rows, grid_cols);
  mapper.place(func);
}

} // namespace taskflow
} // namespace mlir
