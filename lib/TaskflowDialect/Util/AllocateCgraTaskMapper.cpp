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

  bool isAdjacent(const CgraPosition &other) const {
    return manhattanDistance(other) == 1;
  }
};

//===----------------------------------------------------------------------===//
struct TaskPlacement {
  SmallVector<CgraPosition> cgra_positions;

  CgraPosition primary() const {
    return cgra_positions.empty() ? CgraPosition{-1, -1} : cgra_positions[0];
  }

  size_t cgraCount() const { return cgra_positions.size(); }

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
struct MemoryNode;

struct TaskNode {
  size_t id;
  TaskflowTaskOp op;
  int dependency_depth = 0;

  SmallVector<MemoryNode *> read_memrefs;
  SmallVector<MemoryNode *> write_memrefs;
  SmallVector<TaskNode *> ssa_users;
  SmallVector<TaskNode *> ssa_operands;

  SmallVector<CgraPosition> placement;

  TaskNode(size_t id, TaskflowTaskOp op) : id(id), op(op) {}
};

struct MemoryNode {
  Value memref;

  SmallVector<TaskNode *> readers;
  SmallVector<TaskNode *> writers;

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
    size_t task_id = 0;
    func.walk([&](TaskflowTaskOp task) {
      auto node = std::make_unique<TaskNode>(task_id++, task);
      op_to_node[task] = node.get();
      task_nodes.push_back(std::move(node));
    });

    for (auto &t_node : task_nodes) {
      for (Value orig_memref : t_node->op.getOriginalReadMemrefs()) {
        MemoryNode *m_node = getOrCreateMemoryNode(orig_memref);
        t_node->read_memrefs.push_back(m_node);
        m_node->readers.push_back(t_node.get());
      }

      for (Value orig_memref : t_node->op.getOriginalWriteMemrefs()) {
        MemoryNode *m_node = getOrCreateMemoryNode(orig_memref);
        t_node->write_memrefs.push_back(m_node);
        m_node->writers.push_back(t_node.get());
      }
    }

    for (auto &consumer_node : task_nodes) {
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
class TaskMapper {
public:
  TaskMapper(int grid_rows, int grid_cols)
      : grid_rows_(grid_rows), grid_cols_(grid_cols) {
    occupied_.resize(grid_rows_);
    for (auto &row : occupied_) {
      row.resize(grid_cols_, false);
    }
  }

  void place(func::FuncOp func) {
    SmallVector<TaskflowTaskOp> tasks;
    func.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });

    if (tasks.empty()) {
      llvm::errs() << "No tasks to place.\n";
      return;
    }

    TaskMemoryGraph graph;
    graph.build(func);

    if (graph.task_nodes.empty()) {
      llvm::errs() << "No tasks to place.\n";
      return;
    }

    computeDependencyDepth(graph);

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

      bool sram_moved = assignAllSrams(graph);

      if (iter > 0 && !sram_moved) {
        break;
      }
    }

    OpBuilder builder(func.getContext());
    for (auto &task_node : graph.task_nodes) {
      if (task_node->placement.empty()) {
        continue;
      }

      SmallVector<NamedAttribute, 4> mapping_attrs;

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

      task_node->op->setAttr(
          "task_mapping_info",
          DictionaryAttr::get(func.getContext(), mapping_attrs));
    }
  }

private:
  void resetTaskPlacements(TaskMemoryGraph &graph) {
    for (auto &task : graph.task_nodes) {
      task->placement.clear();
    }
    for (int r = 0; r < grid_rows_; ++r) {
      std::fill(occupied_[r].begin(), occupied_[r].end(), false);
    }
  }

  bool assignAllSrams(TaskMemoryGraph &graph) {
    bool changed = false;
    for (auto &mem_node : graph.memory_nodes) {
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

  // Rectangular multi-CGRA placement: factorizations (rows×cols) and origin
  // search are combined here (review: merge getRectShapes + tryPlaceShape).
  TaskPlacement findBestPlacement(TaskNode *task_node, int cgra_count,
                                  TaskMemoryGraph &graph) {
    for (int rows = 1; rows <= cgra_count; ++rows) {
      if (cgra_count % rows != 0) {
        continue;
      }
      int cols = cgra_count / rows;
      SmallVector<std::pair<int, int>> shape_offsets;
      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
          shape_offsets.push_back({c, r});

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
          if (!valid)
            continue;
          int score = computeScore(task_node, candidate, graph);
          if (score > best_score) {
            best_score = score;
            best_placement = candidate;
          }
        }
      }
      if (!best_placement.cgra_positions.empty()) {
        return best_placement;
      }
    }

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

  int computeScore(TaskNode *task_node, const TaskPlacement &placement,
                   TaskMemoryGraph &graph) {
    constexpr int kAlpha = 10;
    constexpr int kBeta = 50;

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

  void computeDependencyDepth(TaskMemoryGraph &graph) {
    DenseMap<TaskNode *, int> depth_cache;
    for (auto &node : graph.task_nodes) {
      node->dependency_depth = calculateDepth(node.get(), depth_cache);
    }
  }

  int calculateDepth(TaskNode *node, DenseMap<TaskNode *, int> &depth_cache) {
    if (depth_cache.count(node)) {
      return depth_cache[node];
    }

    int max_child_depth = 0;
    for (TaskNode *child : node->ssa_users) {
      max_child_depth =
          std::max(max_child_depth, calculateDepth(child, depth_cache) + 1);
    }

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
