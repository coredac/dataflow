//===- AllocateCgraTaskMapper.cpp - Task-to-CGRA mapping implementation ---===//
//
// Implements runAllocateCgraToTask and the internal TaskMapper used by
// AllocateCgraToTaskPass.  Kept under TaskflowDialect/Util per code review.
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/Util/CgraPlacementUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
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
/// Represents a position on the 2D CGRA grid.
struct CgraPosition {
  int row;
  int col;

  bool operator==(const CgraPosition &other) const {
    return row == other.row && col == other.col;
  }

  bool operator!=(const CgraPosition &other) const { return !(*this == other); }

  int manhattanDistance(const CgraPosition &other) const {
    return std::abs(row - other.row) + std::abs(col - other.col);
  }

  /// Returns true if the two positions are directly adjacent (Manhattan
  /// distance == 1), i.e. share an edge on the grid.
  bool isAdjacent(const CgraPosition &other) const {
    return manhattanDistance(other) == 1;
  }
};

//===----------------------------------------------------------------------===//
// Task Placement Info
//===----------------------------------------------------------------------===//
/// Stores the placement result for a task: the set of CGRAs assigned to it.
/// A task can span one or more contiguous CGRAs (rectangular or non-rect).
struct TaskPlacement {
  SmallVector<CgraPosition> cgra_positions; // CGRAs assigned to this task.

  /// Returns the primary (first) CGRA position.
  CgraPosition primary() const {
    return cgra_positions.empty() ? CgraPosition{-1, -1} : cgra_positions[0];
  }

  /// Returns the number of CGRAs assigned to this task.
  size_t cgraCount() const { return cgra_positions.size(); }

  /// Returns true if any CGRA in this task is grid-adjacent to any CGRA
  /// in `other`, indicating that direct data forwarding between tasks is
  /// possible without going through the network.
  bool hasTaskAdjacentCgra(const TaskPlacement &other) const {
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

/// Represents a Task node in the dependency graph.
struct TaskNode {
  size_t id;
  TaskflowTaskOp op;
  int dependency_depth = 0; // Longest path to any sink in the dependency graph.

  // Edges based on original (pre-streaming-fusion) memory accesses.
  SmallVector<MemoryNode *> read_memrefs;  // MemoryNodes this task reads.
  SmallVector<MemoryNode *> write_memrefs; // MemoryNodes this task writes.
  // SSA value edges between tasks.
  SmallVector<TaskNode *> ssa_users; // Tasks that consume this task's output.
  SmallVector<TaskNode *>
      ssa_operands; // Tasks whose output this task consumes.

  // Placement result — populated by TaskMapper::place().
  SmallVector<CgraPosition> placement;

  TaskNode(size_t id, TaskflowTaskOp op) : id(id), op(op) {}
};

/// Represents a MemRef node in the dependency graph.
struct MemoryNode {
  Value memref;

  // Access edges.
  SmallVector<TaskNode *> readers; // Tasks that read this memref.
  SmallVector<TaskNode *> writers; // Tasks that write this memref.

  // SRAM assignment result — populated by TaskMapper::assignAllSrams().
  std::optional<CgraPosition> assigned_sram_pos;

  MemoryNode(Value memref) : memref(memref) {}
};

class TaskMemoryGraph {
public:
  SmallVector<std::unique_ptr<TaskNode>> task_nodes;
  SmallVector<std::unique_ptr<MemoryNode>> memory_nodes;
  DenseMap<Value, MemoryNode *> memref_to_node;
  DenseMap<Operation *, TaskNode *> op_to_node;

  void build(func::FuncOp func) {
    // Phase 1: Create a TaskNode for every TaskflowTaskOp in the function.
    size_t task_id = 0;
    func.walk([&](TaskflowTaskOp task) {
      auto node = std::make_unique<TaskNode>(task_id++, task);
      op_to_node[task] = node.get();
      task_nodes.push_back(std::move(node));
    });

    // Phase 2: Create MemoryNodes using ORIGINAL memrefs (canonical identity).
    // Uses original_read_memrefs / original_write_memrefs so that aliased
    // memories (created by streaming-fusion) share the same MemoryNode.
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

    // Phase 3: Build SSA edges (inter-task value dependencies).
    // A consumer task directly uses a value produced by a producer task.
    for (auto &consumer_node : task_nodes) {
      // Iterates all operands to be safe (not only getValueInputs()).
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
/// Maps a task-memory graph onto a 2D CGRA grid.
///
/// Uses a two-phase fixed-point iteration:
///   Phase 1: Place tasks on the grid (scoring by SSA + memory proximity).
///   Phase 2: Assign each MemRef to the nearest SRAM given task positions.
/// Iterates until SRAM assignments converge (critical-path-first ordering).
class TaskMapper {
public:
  TaskMapper(int grid_rows, int grid_cols)
      : grid_rows_(grid_rows), grid_cols_(grid_cols) {
    occupied_.resize(grid_rows_);
    for (auto &row : occupied_) {
      row.resize(grid_cols_, false);
    }
  }

  /// Places all tasks and performs iterative SRAM assignment for `func`.
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

    // Computes dependency depth for each task.
    // Dependency depth = longest path from this node to any sink node in the
    // dependency graph (via SSA or memory edges).  Tasks with higher depth
    // have longer dependent chains after them; placing them first gives their
    // successors the best chance of landing on adjacent grid cells.
    computeDependencyDepth(graph);

    // Sorts tasks by dependency depth (Critical Path First).
    SmallVector<TaskNode *> sorted_tasks;
    for (auto &node : graph.task_nodes)
      sorted_tasks.push_back(node.get());

    std::stable_sort(sorted_tasks.begin(), sorted_tasks.end(),
                     [](TaskNode *a, TaskNode *b) {
                       return a->dependency_depth > b->dependency_depth;
                     });

    // Fixed-point iteration: task placement scoring depends on SRAM
    // positions (memory proximity), and SRAM assignment depends on task
    // positions (centroid of accessing tasks).  Each iteration re-places
    // all tasks using the latest SRAM assignments, then re-assigns SRAMs.
    // Converges when SRAM assignments stabilise (no change between iters).
    constexpr int kMaxIterations = 10;

    for (int iter = 0; iter < kMaxIterations; ++iter) {
      if (iter > 0) {
        resetTaskPlacements(graph);
      }

      // Phase 1: Place tasks (scoring uses current SRAM assignments).
      for (TaskNode *task_node : sorted_tasks) {
        int cgra_count = 1;
        if (auto attr =
                task_node->op->getAttrOfType<IntegerAttr>("cgra_count")) {
          cgra_count = attr.getInt();
        }

        TaskPlacement placement =
            findBestPlacement(task_node, cgra_count, graph);

        assert(!placement.cgra_positions.empty() &&
               "findBestPlacement must succeed: cgra_count should be "
               "validated by the upstream resource-aware optimization pass "
               "or manually assigned resource binding attributes");

        // Commits placement and marks occupied grid cells.
        for (const auto &pos : placement.cgra_positions) {
          task_node->placement.push_back(pos);
        }

        for (const auto &pos : placement.cgra_positions) {
          if (pos.row >= 0 && pos.row < grid_rows_ && pos.col >= 0 &&
              pos.col < grid_cols_) {
            occupied_[pos.row][pos.col] = true;
          }
        }
      }

      // Phase 2: Assign SRAMs (assuming fixed task positions).
      // If nothing moved, task scores won't change → convergence reached.
      bool sram_moved = assignAllSrams(graph);

      if (iter > 0 && !sram_moved) {
        break;
      }
    }

    // Annotates result: writes task_mapping_info attribute to each task op.
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
        coord_attrs.push_back(
            NamedAttribute(StringAttr::get(func.getContext(), "row"),
                           builder.getI32IntegerAttr(pos.row)));
        coord_attrs.push_back(
            NamedAttribute(StringAttr::get(func.getContext(), "col"),
                           builder.getI32IntegerAttr(pos.col)));
        pos_attrs.push_back(
            DictionaryAttr::get(func.getContext(), coord_attrs));
      }
      mapping_attrs.push_back(
          NamedAttribute(StringAttr::get(func.getContext(), "cgra_positions"),
                         builder.getArrayAttr(pos_attrs)));

      // 2. Read SRAM locations.
      SmallVector<Attribute> read_sram_attrs;
      for (MemoryNode *mem : task_node->read_memrefs) {
        if (mem->assigned_sram_pos) {
          SmallVector<NamedAttribute, 2> sram_coord;
          sram_coord.push_back(NamedAttribute(
              StringAttr::get(func.getContext(), "row"),
              builder.getI32IntegerAttr(mem->assigned_sram_pos->row)));
          sram_coord.push_back(NamedAttribute(
              StringAttr::get(func.getContext(), "col"),
              builder.getI32IntegerAttr(mem->assigned_sram_pos->col)));
          read_sram_attrs.push_back(
              DictionaryAttr::get(func.getContext(), sram_coord));
        }
      }
      mapping_attrs.push_back(NamedAttribute(
          StringAttr::get(func.getContext(), "read_sram_locations"),
          builder.getArrayAttr(read_sram_attrs)));

      // 3. Write SRAM locations.
      SmallVector<Attribute> write_sram_attrs;
      for (MemoryNode *mem : task_node->write_memrefs) {
        if (mem->assigned_sram_pos) {
          SmallVector<NamedAttribute, 2> sram_coord;
          sram_coord.push_back(NamedAttribute(
              StringAttr::get(func.getContext(), "row"),
              builder.getI32IntegerAttr(mem->assigned_sram_pos->row)));
          sram_coord.push_back(NamedAttribute(
              StringAttr::get(func.getContext(), "col"),
              builder.getI32IntegerAttr(mem->assigned_sram_pos->col)));

          write_sram_attrs.push_back(
              DictionaryAttr::get(func.getContext(), sram_coord));
        }
      }
      mapping_attrs.push_back(NamedAttribute(
          StringAttr::get(func.getContext(), "write_sram_locations"),
          builder.getArrayAttr(write_sram_attrs)));

      // Sets task_mapping_info attribute on the task op.
      task_node->op->setAttr(
          "task_mapping_info",
          DictionaryAttr::get(func.getContext(), mapping_attrs));
    }
  }

private:
  /// Clears all task placements and resets the occupied-cell grid.
  void resetTaskPlacements(TaskMemoryGraph &graph) {
    for (auto &task : graph.task_nodes) {
      task->placement.clear();
    }
    // Clears grid.
    for (int r = 0; r < grid_rows_; ++r) {
      std::fill(occupied_[r].begin(), occupied_[r].end(), false);
    }
  }

  /// Assigns each MemoryNode to the SRAM at the centroid of all CGRAs that
  /// access it (readers + writers).  Returns true if any assignment changed,
  /// which is used as the convergence criterion for the outer iteration loop.
  bool assignAllSrams(TaskMemoryGraph &graph) {
    bool changed = false;
    for (auto &mem_node : graph.memory_nodes) {
      int total_row = 0, total_col = 0, count = 0;
      // Computes centroid of all tasks that read this memory.
      for (TaskNode *reader : mem_node->readers) {
        for (const CgraPosition &pos : reader->placement) {
          total_row += pos.row;
          total_col += pos.col;
          count++;
        }
      }
      // Computes centroid of all tasks that write this memory.
      for (TaskNode *writer : mem_node->writers) {
        for (const CgraPosition &pos : writer->placement) {
          total_row += pos.row;
          total_col += pos.col;
          count++;
        }
      }

      std::optional<CgraPosition> new_sram_pos;
      if (count > 0) {
        // Rounds to the nearest integer (round-half-up).
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
  // Returns (col, row) offsets relative to the placement origin.
  // Reserved for IR-driven tile shapes; placement currently uses implicit
  // rectangular enumeration in findBestPlacement.
  SmallVector<std::pair<int, int>> parseTileShapeOffsets(StringRef tile_shape,
                                                         int cgra_count) {
    SmallVector<std::pair<int, int>> offsets;

    if (tile_shape.empty() || cgra_count <= 1) {
      offsets.push_back({0, 0});
      return offsets;
    }

    size_t bracket_pos = tile_shape.find('[');
    if (bracket_pos != StringRef::npos) {
      StringRef positions_str = tile_shape.substr(bracket_pos);
      size_t pos = 0;
      while (pos < positions_str.size()) {
        size_t open = positions_str.find('(', pos);
        if (open == StringRef::npos)
          break;
        size_t close = positions_str.find(')', open);
        if (close == StringRef::npos)
          break;
        StringRef pair_str = positions_str.slice(open + 1, close);
        auto [col_str, row_str] = pair_str.split(',');
        int col_off = 0, row_off = 0;
        col_str.getAsInteger(10, col_off);
        row_str.getAsInteger(10, row_off);
        offsets.push_back({col_off, row_off});
        pos = close + 1;
      }
    } else {
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

    assert(!offsets.empty() && "tile_shape parsing yielded empty offsets");
    return offsets;
  }

  // Finds the best placement for `task_node` requiring exactly `cgra_count`
  // CGRAs.  Strategy:
  //   1. Rectangular: tries all (rows × cols) factorizations of cgra_count,
  //      preferring square-like shapes (lower |rows-cols|).  For each shape,
  //      sweeps every origin on the grid and picks the highest-scoring free
  //      position.
  //   2. Non-rectangular fallback: if no rectangle fits (fragmented grid),
  //      runs a polyomino DFS (tryNonRectShapes) to find any connected
  //      k-CGRA cluster.
  // Returns an empty TaskPlacement only if the grid is completely full
  // (should not happen if cgra_count was validated upstream).
  TaskPlacement findBestPlacement(TaskNode *task_node, int cgra_count,
                                  TaskMemoryGraph &graph) {
    // Phase 1: try all rectangular factorizations of cgra_count.
    for (int rows = 1; rows <= cgra_count; ++rows) {
      if (cgra_count % rows != 0) {
        continue;
      }
      int cols = cgra_count / rows;

      // Builds the list of (col_offset, row_offset) cells relative to the
      // top-left origin for this rows×cols rectangle.
      SmallVector<std::pair<int, int>> shape_offsets;
      for (int shape_row = 0; shape_row < rows; ++shape_row) {
        for (int shape_col = 0; shape_col < cols; ++shape_col) {
          shape_offsets.push_back({shape_col, shape_row});
        }
      }

      // Sweeps every valid origin on the grid and keeps the highest-scoring
      // non-overlapping placement for this shape.
      int best_score = INT_MIN;
      TaskPlacement best_placement;
      for (int origin_row = 0; origin_row < grid_rows_; ++origin_row) {
        for (int origin_col = 0; origin_col < grid_cols_; ++origin_col) {
          // Checks that every cell of the rectangle is within bounds and free.
          bool valid = true;
          TaskPlacement candidate;
          for (auto &[col_off, row_off] : shape_offsets) {
            int abs_row = origin_row + row_off;
            int abs_col = origin_col + col_off;
            if (abs_row < 0 || abs_row >= grid_rows_ || abs_col < 0 ||
                abs_col >= grid_cols_ || occupied_[abs_row][abs_col]) {
              valid = false;
              break;
            }
            candidate.cgra_positions.push_back({abs_row, abs_col});
          }
          if (!valid) {
            continue;
          }
          // Scores the candidate by proximity to dependent tasks and SRAMs.
          int score = computeScore(task_node, candidate, graph);
          if (score > best_score) {
            best_score = score;
            best_placement = candidate;
          }
        }
      }
      // Returns the best placement found for this shape, if any.
      if (!best_placement.cgra_positions.empty()) {
        return best_placement;
      }
    }

    // Phase 2: no rectangle fit — try non-rectangular connected shapes via DFS.
    if (cgra_count > 1) {
      TaskPlacement p = tryNonRectShapes(task_node, cgra_count, graph);
      if (!p.cgra_positions.empty()) {
        return p;
      }
    }

    return {};
  }

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
          // Explores all 4-connected neighbours of every cell already in the
          // current polyomino.  delta_row/delta_col encode the four cardinal
          // directions: up, down, left, right.
          constexpr int delta_row[] = {-1, 1, 0, 0};
          constexpr int delta_col[] = {0, 0, -1, 1};
          for (size_t i = 0; i < current.size(); ++i) {
            const CgraPosition &cell = current[i];
            for (int dir = 0; dir < 4; ++dir) {
              int next_row = cell.row + delta_row[dir];
              int next_col = cell.col + delta_col[dir];
              if (next_row >= 0 && next_row < grid_rows_ && next_col >= 0 &&
                  next_col < grid_cols_ && !occupied_[next_row][next_col]) {
                uint64_t bit = 1ULL << (next_row * grid_cols_ + next_col);
                if ((mask & bit) == 0) {
                  current.push_back({next_row, next_col});
                  search(current, mask | bit);
                  current.pop_back();
                }
              }
            }
          }
        };

    // Seeds the DFS from every free cell on the grid.
    for (int seed_row = 0; seed_row < grid_rows_; ++seed_row) {
      for (int seed_col = 0; seed_col < grid_cols_; ++seed_col) {
        if (!occupied_[seed_row][seed_col]) {
          SmallVector<CgraPosition> start = {{seed_row, seed_col}};
          search(start, 1ULL << (seed_row * grid_cols_ + seed_col));
        }
      }
    }
    return best_placement;
  }

  /// Computes the placement score for `task_node` at `placement`.
  ///
  /// Score = α·SSA_Dist + β·Mem_Dist.
  ///   SSA_Dist : sum of distances to already-placed SSA predecessors and
  ///              successors (negative; penalises far-away neighbours).
  ///   Mem_Dist : sum of distances to assigned SRAMs for read/write memrefs
  ///              (negative; memory proximity is weighted more heavily).
  ///
  /// Higher score is better; 0 means all neighbours are co-located.
  int computeScore(TaskNode *task_node, const TaskPlacement &placement,
                   TaskMemoryGraph &graph) {
    // Weight constants (tunable).
    constexpr int kAlpha = 10; // SSA proximity weight.
    constexpr int kBeta = 50;  // Memory proximity weight (high priority).

    int ssa_score = 0;
    int mem_score = 0;

    auto minDistToPlacement =
        [&](const SmallVector<CgraPosition> &other) -> int {
      int min_dist = INT_MAX;
      for (const auto &pos : placement.cgra_positions) {
        for (const auto &opos : other) {
          min_dist = std::min(min_dist, pos.manhattanDistance(opos));
        }
      }
      return min_dist;
    };

    auto minDistToTarget = [&](const CgraPosition &target) -> int {
      int min_dist = INT_MAX;
      for (const auto &pos : placement.cgra_positions) {
        min_dist = std::min(min_dist, pos.manhattanDistance(target));
      }
      return min_dist;
    };

    // 1. SSA proximity — penalise distance to producers and consumers.
    for (TaskNode *producer : task_node->ssa_operands) {
      if (!producer->placement.empty()) {
        // Uses negative distance: closer = higher score.
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

    // 2. Memory proximity — penalise distance to assigned SRAMs.
    // For read memrefs (data sources).
    for (MemoryNode *mem : task_node->read_memrefs) {
      if (mem->assigned_sram_pos) {
        int dist = minDistToTarget(*mem->assigned_sram_pos);
        mem_score -= dist;
      }
    }
    // For write memrefs: if the SRAM is already assigned (e.g. read by a
    // previous task), we want to be close to it too.
    for (MemoryNode *mem : task_node->write_memrefs) {
      if (mem->assigned_sram_pos) {
        int dist = minDistToTarget(*mem->assigned_sram_pos);
        mem_score -= dist;
      }
    }

    return kAlpha * ssa_score + kBeta * mem_score;
  }

  /// Computes dependency depth for every task in the graph.
  ///
  /// Dependency depth = longest path from a node to any sink in the dependency
  /// graph (traversing both SSA and write→read memory edges).
  ///
  /// Tasks with higher dependency depth have longer chains of dependent tasks
  /// downstream.  Placing them first (critical-path-first) ensures that:
  ///   1. They receive priority access to good grid positions.
  ///   2. Their dependent tasks can later be placed adjacent, minimising
  ///      inter-task communication distance.
  void computeDependencyDepth(TaskMemoryGraph &graph) {
    DenseMap<TaskNode *, int> depth_cache;
    for (auto &node : graph.task_nodes) {
      node->dependency_depth = calculateDepth(node.get(), depth_cache);
    }
  }

  /// Recursively calculates dependency depth for a single task (memoised).
  int calculateDepth(TaskNode *node, DenseMap<TaskNode *, int> &depth_cache) {
    if (depth_cache.count(node)) {
      return depth_cache[node];
    }

    int max_child_depth = 0;
    // SSA dependencies: tasks that consume this task's output values.
    for (TaskNode *child : node->ssa_users) {
      max_child_depth =
          std::max(max_child_depth, calculateDepth(child, depth_cache) + 1);
    }

    // Memory dependencies: Producer → Mem → Consumer write-after-read chains.
    for (MemoryNode *mem : node->write_memrefs) {
      for (TaskNode *reader : mem->readers) {
        if (reader != node) {
          max_child_depth = std::max(max_child_depth,
                                     calculateDepth(reader, depth_cache) + 1);
        }
      }
    }

    return depth_cache[node] = max_child_depth;
  }

  int grid_rows_;
  int grid_cols_;
  std::vector<std::vector<bool>> occupied_;
};

} // namespace

namespace mlir {
namespace taskflow {

void runAllocateCgraToTask(func::FuncOp func, int grid_rows, int grid_cols) {
  TaskMapper mapper(grid_rows, grid_cols);
  mapper.place(func);
}

} // namespace taskflow
} // namespace mlir
