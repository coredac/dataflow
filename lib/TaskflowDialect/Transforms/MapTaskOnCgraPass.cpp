//===- MapTaskOnCgraPass.cpp - Task to CGRA Mapping Pass ----------------===//
//
// This pass maps Taskflow tasks onto a 2D CGRA grid array using a
// spatial-temporal approach:
// 1. Assigns each task a (row, col, start_time, duration) allocation tuple.
// 2. Tasks with no data dependencies may share the same time window on
//    different CGRAs (spatial parallelism).
// 3. Tasks that exceed the number of CGRAs are time-multiplexed onto the
//    same CGRAs at non-overlapping intervals (temporal reuse).
// 4. Assigns memrefs to SRAMs determined by proximity to accessing tasks.
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
#include <optional>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===----------------------------------------------------------------------===//
// CGRA Allocation Position (spatial + temporal)
//===----------------------------------------------------------------------===//
/// Represents a spatial-temporal allocation for a task on the 2D CGRA grid.
///
/// A task assigned to (row, col) occupies that CGRA for the half-open interval
/// [start_time, start_time + duration).  Two tasks may share the same CGRA as
/// long as their intervals do not overlap, enabling time-multiplexed reuse.
///
/// duration is read from the 'steps' IR attribute when present (written by
/// ResourceAwareTaskOptimizationPass).  It defaults to 1 when the task has not
/// yet been profiled, which is a conservative approximation (one ordinal slot).
struct CGRAPosition {
  int row;
  int col;
  int start_time; // First time slot this CGRA is occupied by the task.
  int duration;   // Number of consecutive time slots occupied (>= 1).

  bool operator==(const CGRAPosition &other) const {
    return row == other.row && col == other.col &&
           start_time == other.start_time && duration == other.duration;
  }

  bool operator!=(const CGRAPosition &other) const { return !(*this == other); }

  /// Computes spatial Manhattan distance.
  int manhattanDistance(const CGRAPosition &other) const {
    return std::abs(row - other.row) + std::abs(col - other.col);
  }

  /// Checks if spatially adjacent.
  bool isAdjacent(const CGRAPosition &other) const {
    return manhattanDistance(other) == 1;
  }
};

//===----------------------------------------------------------------------===//
// Task Placement Info
//===----------------------------------------------------------------------===//
/// Stores spatial-temporal placement info for a task.
struct TaskPlacement {
  SmallVector<CGRAPosition> cgra_positions; // CGRAs assigned to this task.

  /// Returns the primary (first) position.
  CGRAPosition primary() const {
    return cgra_positions.empty() ? CGRAPosition{-1, -1, -1, 0}
                                  : cgra_positions[0];
  }

  /// Returns the number of CGRAs assigned.
  size_t cgraCount() const { return cgra_positions.size(); }

  /// Checks if any CGRA in this task is spatially adjacent to any in other.
  bool hasAdjacentCGRA(const TaskPlacement &other) const {
    for (const CGRAPosition &pos : cgra_positions) {
      for (const CGRAPosition &other_pos : other.cgra_positions) {
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

  // Edges based on original memory access.
  SmallVector<MemoryNode *> read_memrefs;  // Original read memrefs.
  SmallVector<MemoryNode *> write_memrefs; // Original write memrefs.
  SmallVector<TaskNode *> ssa_users;
  SmallVector<TaskNode *> ssa_operands;

  // Placement result (spatial + temporal).
  SmallVector<CGRAPosition> placement;

  TaskNode(size_t id, TaskflowTaskOp op) : id(id), op(op) {}

  /// Returns the task's execution duration in time slots.
  /// Reads the 'steps' attribute when present (set by
  /// ResourceAwareTaskOptimizationPass); defaults to 1 when not yet annotated.
  int getDuration() const {
    if (auto attr = op->getAttrOfType<IntegerAttr>("steps")) {
      return std::max(1, static_cast<int>(attr.getInt()));
    }
    return 1;
  }
};

/// Represents a Memory node (MemRef) in the dependency graph.
struct MemoryNode {
  Value memref;

  // Edges.
  SmallVector<TaskNode *> readers;
  SmallVector<TaskNode *> writers;

  // Mapping result (spatial position only; SRAMs are not time-multiplexed).
  std::optional<CGRAPosition> assigned_sram_pos;

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
      // Iterates all operands for now to be safe.
      for (Value operand : consumer_node->op.getValueInputs()) {
        if (auto producer_op = operand.getDefiningOp<TaskflowTaskOp>()) {
          if (TaskNode *producer_node = op_to_node[producer_op]) {
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
// Allocation Mode
//===----------------------------------------------------------------------===//
/// Selects the task-to-CGRA allocation strategy.
enum class AllocationMode {
  Spatial,
  SpatialTemporal,
};

//===----------------------------------------------------------------------===//
// Task Mapper
//===----------------------------------------------------------------------===//
/// Maps a task-memory graph onto a 2D CGRA grid.
///
/// In Spatial mode each CGRA hosts at most one task (original behaviour).
/// In SpatialTemporal mode each CGRA may host multiple tasks at non-overlapping
/// intervals [start_time, start_time + duration).  Durations are read from the
/// 'steps' attribute when present, defaulting to 1.  ASAP scheduling ensures
/// every task starts as early as its predecessors allow.
class TaskMapper {
public:
  TaskMapper(int grid_rows, int grid_cols, AllocationMode mode)
      : grid_rows_(grid_rows), grid_cols_(grid_cols), mode_(mode) {
    // Initializes per-CGRA occupancy interval lists.
    intervals_.resize(grid_rows_);
    for (auto &row : intervals_) {
      row.resize(grid_cols_);
    }
  }

  /// Places all tasks using the active allocation mode and annotates the IR.
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
    // Dependency depth = longest path from this node to any sink in the
    // dependency graph (via SSA or memory edges). Tasks with higher depth are
    // placed first; this order is a valid topological ordering from sources to
    // sinks, required for correct earliest-start computation.
    computeDependencyDepth(graph);

    // Sorts tasks by dependency depth (Critical Path First).
    SmallVector<TaskNode *> sorted_tasks;
    for (auto &node : graph.task_nodes)
      sorted_tasks.push_back(node.get());

    std::stable_sort(sorted_tasks.begin(), sorted_tasks.end(),
                     [](TaskNode *a, TaskNode *b) {
                       return a->dependency_depth > b->dependency_depth;
                     });

    // Iterative Refinement Loop (Coordinate Descent).
    // Alternates between Task Placement (Phase 1) and SRAM Assignment (Phase
    // 2) until SRAM positions stabilize.
    constexpr int kMaxIterations = 10;

    for (int iter = 0; iter < kMaxIterations; ++iter) {
      // Phase 1: Place tasks (assuming fixed SRAM assignments from prev iter).
      if (iter > 0) {
        resetTaskPlacements(graph);
      }

      for (TaskNode *task_node : sorted_tasks) {
        int cgra_count = 1;
        if (auto attr =
                task_node->op->getAttrOfType<IntegerAttr>("cgra_count")) {
          cgra_count = attr.getInt();
        }

        TaskPlacement placement =
            findBestPlacement(task_node, cgra_count, graph);

        // Commits placement.
        for (const CGRAPosition &pos : placement.cgra_positions) {
          task_node->placement.push_back(pos);
        }

        // Marks occupied intervals.
        for (const CGRAPosition &pos : placement.cgra_positions) {
          if (pos.row >= 0 && pos.row < grid_rows_ && pos.col >= 0 &&
              pos.col < grid_cols_) {
            markOccupied(pos.row, pos.col, pos.start_time, pos.duration);
          }
        }
      }

      // Phase 2: Assign SRAMs (assuming fixed task placements).
      bool sram_moved = assignAllSRAMs(graph);

      // Convergence check: stable SRAM positions imply stable task placements.
      if (iter > 0 && !sram_moved) {
        break;
      }
    }

    // Annotates result onto the IR.
    OpBuilder builder(func.getContext());
    for (auto &task_node : graph.task_nodes) {
      if (task_node->placement.empty()) {
        continue;
      }

      SmallVector<NamedAttribute, 4> mapping_attrs;

      // 1. CGRA positions (col, duration, row, start_time — alphabetical).
      SmallVector<Attribute> pos_attrs;
      for (const CGRAPosition &pos : task_node->placement) {
        SmallVector<NamedAttribute, 4> coord_attrs;
        coord_attrs.push_back(
            NamedAttribute(StringAttr::get(func.getContext(), "col"),
                           builder.getI32IntegerAttr(pos.col)));
        coord_attrs.push_back(
            NamedAttribute(StringAttr::get(func.getContext(), "duration"),
                           builder.getI32IntegerAttr(pos.duration)));
        coord_attrs.push_back(
            NamedAttribute(StringAttr::get(func.getContext(), "row"),
                           builder.getI32IntegerAttr(pos.row)));
        coord_attrs.push_back(
            NamedAttribute(StringAttr::get(func.getContext(), "start_time"),
                           builder.getI32IntegerAttr(pos.start_time)));
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
              StringAttr::get(func.getContext(), "col"),
              builder.getI32IntegerAttr(mem->assigned_sram_pos->col)));
          sram_coord.push_back(NamedAttribute(
              StringAttr::get(func.getContext(), "row"),
              builder.getI32IntegerAttr(mem->assigned_sram_pos->row)));
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
              StringAttr::get(func.getContext(), "col"),
              builder.getI32IntegerAttr(mem->assigned_sram_pos->col)));
          sram_coord.push_back(NamedAttribute(
              StringAttr::get(func.getContext(), "row"),
              builder.getI32IntegerAttr(mem->assigned_sram_pos->row)));
          write_sram_attrs.push_back(
              DictionaryAttr::get(func.getContext(), sram_coord));
        }
      }
      mapping_attrs.push_back(NamedAttribute(
          StringAttr::get(func.getContext(), "write_sram_locations"),
          builder.getArrayAttr(write_sram_attrs)));

      // Sets attribute.
      task_node->op->setAttr(
          "task_mapping_info",
          DictionaryAttr::get(func.getContext(), mapping_attrs));
    }
  }

private:
  /// Checks whether placing a task of duration d at (row, col, t) would
  /// overlap with any already-allocated interval at that CGRA.
  /// In Spatial mode any existing allocation blocks the CGRA entirely.
  bool isOccupied(int row, int col, int t, int d) const {
    if (mode_ == AllocationMode::Spatial) {
      return !intervals_[row][col].empty();
    }
    for (auto [s, e] : intervals_[row][col]) {
      if (t < e && t + d > s) {
        return true;
      }
    }
    return false;
  }

  /// Records that CGRA (row, col) is occupied over [t, t+d).
  void markOccupied(int row, int col, int t, int d) {
    intervals_[row][col].push_back({t, t + d});
  }

  /// Clears all task placements and CGRA occupancy intervals.
  void resetTaskPlacements(TaskMemoryGraph &graph) {
    for (auto &task : graph.task_nodes) {
      task->placement.clear();
    }
    for (auto &row : intervals_) {
      for (auto &col_intervals : row) {
        col_intervals.clear();
      }
    }
  }

  /// Assigns all memory nodes to SRAMs based on spatial centroid of accessing
  /// tasks.  Returns true if any SRAM assignment changed.
  bool assignAllSRAMs(TaskMemoryGraph &graph) {
    bool changed = false;
    for (auto &mem_node : graph.memory_nodes) {
      // Computes spatial centroid of all tasks that access this memory.
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

      std::optional<CGRAPosition> new_sram_pos;
      if (count > 0) {
        int avg_row = (total_row + count / 2) / count;
        int avg_col = (total_col + count / 2) / count;
        // SRAM positions are purely spatial; temporal fields are unused.
        new_sram_pos = CGRAPosition{avg_row, avg_col, 0, 0};
      }

      if (mem_node->assigned_sram_pos != new_sram_pos) {
        mem_node->assigned_sram_pos = new_sram_pos;
        changed = true;
      }
    }
    return changed;
  }

  /// Computes the earliest feasible start time for a task (ASAP scheduling).
  ///
  /// A task may not start until all its producers have fully completed.
  int computeEarliestStartTime(const TaskNode *task_node) const {
    int min_time = 0;
    // SSA producers must complete before this task starts.
    for (const TaskNode *pred : task_node->ssa_operands) {
      if (!pred->placement.empty()) {
        const CGRAPosition &pos = pred->placement[0];
        min_time = std::max(min_time, pos.start_time + pos.duration);
      }
    }
    // Memory producers must also finish.
    for (const MemoryNode *mem : task_node->read_memrefs) {
      for (const TaskNode *writer : mem->writers) {
        if (writer != task_node && !writer->placement.empty()) {
          const CGRAPosition &pos = writer->placement[0];
          min_time = std::max(min_time, pos.start_time + pos.duration);
        }
      }
    }
    return min_time;
  }

  /// Finds the best spatial-temporal placement for a task.
  ///
  /// Spatial mode: scans the grid at the fixed time 0; fails with an assertion
  ///   when no free cell exists (grid over-subscribed).
  ///
  /// SpatialTemporal mode: performs ASAP scheduling: searches start times
  ///   from the earliest feasible value upward, incrementing by the task
  ///   duration so that every candidate aligns to a natural slot boundary.
  ///   Within each candidate start time, picks the spatial position that
  ///   maximises the proximity score.
  ///
  /// TODO: Introduce explicit multi-CGRA binding logic for cgra_count > 1.
  TaskPlacement findBestPlacement(TaskNode *task_node, int cgra_count,
                                  TaskMemoryGraph &graph) {
    int task_duration = task_node->getDuration();

    int min_time = (mode_ == AllocationMode::SpatialTemporal)
                       ? computeEarliestStartTime(task_node)
                       : 0;

    // Upper bound: in the worst case all tasks form a linear chain and
    // time-multiplex onto a single CGRA.  Each occupies task_duration slots,
    // so at most N * task_duration steps are needed beyond min_time.
    int max_time = (mode_ == AllocationMode::SpatialTemporal)
                       ? min_time + grid_rows_ * grid_cols_ * task_duration
                       : 0;

    for (int t = min_time; t <= max_time; t += task_duration) {
      int best_score = INT_MIN;
      TaskPlacement best_at_t;

      for (int r = 0; r < grid_rows_; ++r) {
        for (int c = 0; c < grid_cols_; ++c) {
          if (isOccupied(r, c, t, task_duration)) {
            continue;
          }

          TaskPlacement candidate;
          candidate.cgra_positions.push_back({r, c, t, task_duration});

          int score = computeScore(task_node, candidate, graph);
          if (score > best_score) {
            best_score = score;
            best_at_t = candidate;
          }
        }
      }

      // Returns the earliest start time that yields a valid placement (ASAP).
      if (!best_at_t.cgra_positions.empty()) {
        return best_at_t;
      }

      // Spatial mode has no time dimension; over-subscription is an error.
      if (mode_ == AllocationMode::Spatial) {
        assert(false &&
               "No available CGRA position found (grid over-subscribed). "
               "Consider using --allocation-mode=spatial-temporal.");
      }
    }

    // Theoretically unreachable: the upper bound guarantees at least one slot is available.
    assert(false && "Spatial-temporal placement failed: max_time exceeded.");
    return TaskPlacement{};
  }

  /// Computes placement score based on the Task-Memory Graph.
  ///
  /// Uses spatial (row, col) distances only; the temporal dimension is
  /// enforced separately via computeEarliestStartTime.
  ///
  /// Score = α·SSA_Dist + β·Mem_Dist.
  ///
  /// SSA_Dist: Minimize distance to placed SSA predecessors (ssa_operands).
  /// Mem_Dist: Minimize distance to assigned SRAMs for read/write memrefs.
  ///
  /// TODO: Introduce explicit 'direct_wires' attributes in the IR for
  /// downstream hardware generators to configure fast bypass paths between
  /// adjacent PEs with dependencies.
  int computeScore(TaskNode *task_node, const TaskPlacement &placement,
                   TaskMemoryGraph &graph) {
    // Weight constants (tunable).
    constexpr int kAlpha = 10; // SSA proximity weight.
    constexpr int kBeta = 50;  // Memory proximity weight (high priority).

    int ssa_score = 0;
    int mem_score = 0;

    CGRAPosition current_pos = placement.primary();

    // 1. SSA proximity (predecessors & successors).
    for (TaskNode *producer : task_node->ssa_operands) {
      if (!producer->placement.empty()) {
        int dist = current_pos.manhattanDistance(producer->placement[0]);
        // Uses negative distance to penalize far-away placements.
        ssa_score -= dist;
      }
    }
    for (TaskNode *consumer : task_node->ssa_users) {
      if (!consumer->placement.empty()) {
        int dist = current_pos.manhattanDistance(consumer->placement[0]);
        ssa_score -= dist;
      }
    }

    // 2. Memory proximity.
    // For read memrefs.
    for (MemoryNode *mem : task_node->read_memrefs) {
      if (mem->assigned_sram_pos) {
        int dist = current_pos.manhattanDistance(*mem->assigned_sram_pos);
        mem_score -= dist;
      }
    }
    // For write memrefs.
    // If we write to a memory that is already assigned (e.g. read by previous
    // task), we want to be close to it too.
    for (MemoryNode *mem : task_node->write_memrefs) {
      if (mem->assigned_sram_pos) {
        int dist = current_pos.manhattanDistance(*mem->assigned_sram_pos);
        mem_score -= dist;
      }
    }

    return kAlpha * ssa_score + kBeta * mem_score;
  }

  /// Computes dependency depth for all tasks in the graph.
  ///
  /// Dependency depth = longest path from this node to any sink node in the
  /// dependency graph (via SSA or memory edges).
  ///
  /// Tasks with higher dependency depth have longer chains of dependent tasks
  /// after them. By placing these tasks first:
  /// 1. They get priority access to good grid positions.
  /// 2. Their dependent tasks can then be positioned adjacent to them,
  ///    minimizing inter-task communication distance.
  void computeDependencyDepth(TaskMemoryGraph &graph) {
    DenseMap<TaskNode *, int> depth_cache;
    for (auto &node : graph.task_nodes) {
      node->dependency_depth = calculateDepth(node.get(), depth_cache);
    }
  }

  /// Recursively calculates dependency depth for a single task.
  int calculateDepth(TaskNode *node, DenseMap<TaskNode *, int> &depth_cache) {
    if (depth_cache.count(node)) {
      return depth_cache[node];
    }

    int max_child_depth = 0;
    // SSA dependencies.
    for (TaskNode *child : node->ssa_users) {
      max_child_depth =
          std::max(max_child_depth, calculateDepth(child, depth_cache) + 1);
    }

    // Memory dependencies (Producer -> Mem -> Consumer).
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
  AllocationMode mode_;
  // Per-CGRA occupancy intervals: intervals_[row][col] holds a list of
  // (start, end_exclusive) pairs representing occupied time windows.
  // In Spatial mode, non-empty means the CGRA is fully occupied.
  std::vector<std::vector<SmallVector<std::pair<int, int>, 4>>> intervals_;
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//
struct MapTaskOnCgraPass
    : public PassWrapper<MapTaskOnCgraPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapTaskOnCgraPass)

  MapTaskOnCgraPass() = default;
  MapTaskOnCgraPass(const MapTaskOnCgraPass &other) : PassWrapper(other) {}

  StringRef getArgument() const override { return "map-task-on-cgra"; }

  StringRef getDescription() const override {
    return "Maps Taskflow tasks onto a 2D CGRA grid with adjacency "
           "optimization and memory mapping. Use --allocation-mode to choose "
           "between 'spatial' (original) and 'spatial-temporal' (default).";
  }

  // Selects the task-to-CGRA allocation strategy:
  //   "spatial"          – one task per CGRA, fails if tasks > grid size.
  //   "spatial-temporal" – time-multiplexes CGRAs; no hard task count limit.
  Option<std::string> allocationMode{
      *this, "allocation-mode",
      llvm::cl::desc(
          "Task allocation mode: 'spatial' (one task per CGRA, asserts if "
          "tasks exceed grid size) or 'spatial-temporal' (default, adds "
          "start_time/duration enabling task count > grid size)."),
      llvm::cl::init("spatial-temporal")};

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    AllocationMode mode = AllocationMode::SpatialTemporal;
    if (allocationMode.getValue() == "spatial") {
      mode = AllocationMode::Spatial;
    } else if (allocationMode.getValue() != "spatial-temporal") {
      func.emitError("map-task-on-cgra: unknown allocation-mode '")
          << allocationMode.getValue()
          << "'. Valid values: 'spatial', 'spatial-temporal'.";
      return signalPassFailure();
    }

    constexpr int kDefaultGridRows = 3;
    constexpr int kDefaultGridCols = 3;
    TaskMapper mapper(kDefaultGridRows, kDefaultGridCols, mode);
    mapper.place(func);
  }
};

} // namespace

namespace mlir {
namespace taskflow {

std::unique_ptr<Pass> createMapTaskOnCgraPass() {
  return std::make_unique<MapTaskOnCgraPass>();
}

} // namespace taskflow
} // namespace mlir
