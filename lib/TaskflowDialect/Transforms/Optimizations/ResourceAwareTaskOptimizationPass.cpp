//===- ResourceAwareTaskOptimizationPass.cpp - Pipeline Balance & Fusion --===//
// This pass performs two-phase optimization on the task graph:
// 1. Utilization Fusion: merges independent (no-edge) tasks, selecting pairs
//    that minimize |trip_count_a - trip_count_b| for balanced utilization.
// 2. Pipeline Balance: allocates extra CGRAs to critical-path bottleneck tasks.
//    More CGRAs combine tile arrays into larger arrays for mapping, potentially
//    lowering compiled_ii.  Latency model: II * (trip_count - 1) + steps.
//
// Targets a 4x4 CGRA grid (16 CGRAs total).  Each task may use up to 4 CGRAs.
// Supported per-task shapes: rect (1×1..4×1/1×4/2×2), L (3 or 4 CGRAs), T (4 CGRAs).
// Compiled_ii must come from the downstream pipeline (asserts on failure).
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraAttributes.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <set>


using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

constexpr int kCgraGridRows = 4;
constexpr int kCgraGridCols = 4;
constexpr int kTotalCGRAs = kCgraGridRows * kCgraGridCols; // 16
constexpr int kMaxBalanceIterations = 100;
constexpr int kMaxCgrasPerTask = 4;  // Max CGRAs allocatable to a single task.

// Sentinel value: 0 means "not yet profiled". After profileTask() runs,
// both steps and ii MUST be > 0. An assert fires if profiling fails.
constexpr int64_t kUnprofiled = 0;

//===----------------------------------------------------------------------===//
// CGRA Shape Utilities
//===----------------------------------------------------------------------===//

// Represents a CGRA allocation shape on the grid.
//
// For rectangular shapes: rows × cols == cgra_count, and `cgra_positions`
// is empty (all cells in the bounding box are used).
//
// For non-rectangular shapes (L, T): `cgra_positions` stores the explicit
// (col, row) coordinates of the occupied CGRAs.  `rows`/`cols` give the
// bounding box so that tile-level x_tiles/y_tiles can be computed.
struct CgraShape {
  int rows;  // Bounding-box CGRA rows.
  int cols;  // Bounding-box CGRA columns.
  bool is_rectangular;  // True if all cells in the bbox are used.
  // Explicit CGRA positions for non-rectangular shapes.
  // Each pair is (col, row) in CGRA coordinates.  Empty for rectangles.
  SmallVector<std::pair<int, int>> cgra_positions;

  int area() const { return rows * cols; }

  // Returns a human-readable description for log messages only (not IR).
  std::string describe(int cgra_count) const {
    std::string s = std::to_string(rows) + "x" + std::to_string(cols);
    if (!is_rectangular) {
      s += "(non-rect, " + std::to_string(cgra_count) + " CGRAs:";
      for (auto &[c, r] : cgra_positions)
        s += " (" + std::to_string(c) + "," + std::to_string(r) + ")";
      s += ")";
    }
    return s;
  }

  // Returns the shape string written into the IR tile_shape attribute.
  // For rectangular shapes: "NxM" (e.g. "2x2").
  // For non-rectangular shapes: "NxM[(c0,r0)(c1,r1)...]" listing only the
  // occupied CGRA positions so that downstream passes can reconstruct the
  // exact valid tile set for multi-CGRA mapping.
  std::string irAttr() const {
    std::string s = std::to_string(rows) + "x" + std::to_string(cols);
    if (!is_rectangular && !cgra_positions.empty()) {
      s += "[";
      for (auto &[c, r] : cgra_positions)
        s += "(" + std::to_string(c) + "," + std::to_string(r) + ")";
      s += "]";
    }
    return s;
  }
};

// Returns all valid rectangular shapes for `cgra_count` CGRAs.
static SmallVector<CgraShape> getRectangularShapes(int cgra_count) {
  SmallVector<CgraShape> shapes;
  for (int r = 1; r <= kCgraGridRows; ++r) {
    for (int c = 1; c <= kCgraGridCols; ++c) {
      if (r * c == cgra_count) {
        shapes.push_back({r, c, /*is_rectangular=*/true, /*cgra_positions=*/{}});
      }
    }
  }
  return shapes;
}

// Returns true if `cgra_count` CGRAs can fit on the grid and does not
// exceed the per-task limit.
static bool canFitOnGrid(int cgra_count) {
  return cgra_count >= 1 && cgra_count <= kMaxCgrasPerTask;
}

// Returns the set of non-rectangular shapes for `cgra_count` CGRAs.
// Currently defined for cgra_count == 3 (L-shape) and cgra_count == 4
// (L-shape and T-shape variants).  Each shape's coordinates are chosen
// so the bounding box is as small as possible.
static SmallVector<CgraShape> getNonRectangularShapes(int cgra_count) {
  SmallVector<CgraShape> shapes;

  if (cgra_count == 3) {
    // L-shape 3 CGRAs: (0,0)(1,0)(0,1) — bbox 2×2
    shapes.push_back({2, 2, false, {{0,0},{1,0},{0,1}}});
  }

  if (cgra_count == 4) {
    // T-shape: three in a row + one below centre
    //   (0,0)(1,0)(2,0)(1,1)  — bbox 2×3
    shapes.push_back({2, 3, false, {{0,0},{1,0},{2,0},{1,1}}});

    // L-shape: three in a column + one offset
    //   (0,0)(0,1)(0,2)(1,2)  — bbox 3×2
    shapes.push_back({3, 2, false, {{0,0},{0,1},{0,2},{1,2}}});
  }

  return shapes;
}

// Picks the best shape for display/profiling.
// We prefer shapes with the most compact physical layout (smallest maximum distance
// between nodes) to minimize communication latency. In cases of identical bounding
// box area, we prefer more square-like bounds over long rectangles.
//
// Note: This function only picks a localized shape for an idealized single task mapping.
// Global placement and conflict resolution across multiple tasks is legitimately deferred 
// to downstream map-on-cgra pass, as speculative profiling assumes unconstrained placement.
static CgraShape pickBestShape(int cgra_count) {
  // For cgra_count == 3, the 2x2 L-shape has a smaller maximum physical routing distance 
  // (dist=2) compared to a 1x3 rectangle (dist=3), despite having a larger bounding box. 
  // We explicitly prefer the more compact L-shape here for better speculative latency.
  if (cgra_count == 3) {
    auto non_rect_shapes = getNonRectangularShapes(3);
    if (!non_rect_shapes.empty()) {
      return non_rect_shapes.front();
    }
  }

  SmallVector<CgraShape> candidates = getRectangularShapes(cgra_count);
  for (const auto &s : getNonRectangularShapes(cgra_count)) {
    candidates.push_back(s);
  }

  if (!candidates.empty()) {
    return *std::min_element(candidates.begin(), candidates.end(),
        [](const CgraShape &a, const CgraShape &b) {
          int area_a = a.area();
          int area_b = b.area();
          if (area_a != area_b) return area_a < area_b;
          return std::abs(a.rows - a.cols) < std::abs(b.rows - b.cols);
        });
  }

  // Fallback: smallest bounding box (should not be reached for 1..4 CGRAs).
  CgraShape best = {kCgraGridRows, kCgraGridCols, false, {}};
  for (int r = 1; r <= kCgraGridRows; ++r) {
    for (int c = 1; c <= kCgraGridCols; ++c) {
      if (r * c >= cgra_count && r * c < best.area()) {
        best = {r, c, false, {}};
      }
    }
  }
  return best;
}

//===----------------------------------------------------------------------===//
// Task Dependency Graph
//===----------------------------------------------------------------------===//

struct TaskGraphNode {
  size_t id;
  TaskflowTaskOp op;
  int64_t trip_count = 1;
  int64_t steps = kUnprofiled;
  int64_t ii = kUnprofiled;
  int cgra_count = 1;
  CgraShape shape = {1, 1, true};

  // Dependency edges (both SSA and memory).
  SmallVector<TaskGraphNode *> predecessors;
  SmallVector<TaskGraphNode *> successors;

  TaskGraphNode(size_t id, TaskflowTaskOp op) : id(id), op(op) {}

  // Returns estimated task latency using the pipelined execution model:
  //   latency = II * (trip_count - 1) + steps.
  int64_t estimatedLatency() const {
    return ii * (trip_count - 1) + steps;
  }
};

class TaskDependencyGraph {
public:
  SmallVector<std::unique_ptr<TaskGraphNode>> nodes;
  DenseMap<Operation *, TaskGraphNode *> op_to_node;

  void build(func::FuncOp func, bool skip_mapper = false) {
    // 1. Creates TaskGraphNodes.
    size_t task_id = 0;
    func.walk([&](TaskflowTaskOp task) {
      auto node = std::make_unique<TaskGraphNode>(task_id++, task);
      
      // If the task already has profiling attributes (e.g., from fusion),
      // skip expensive speculative lowering and use those directly.
      bool has_precomputed = task->hasAttr("compiled_ii") && task->hasAttr("steps");
      if (!has_precomputed) {
        // Speculative lowering to Neura to get real metrics.
        profileTask(node.get(), task, skip_mapper);
      }

      // Reads existing trip_count attribute if set by fusion.
      if (auto attr = task->getAttrOfType<IntegerAttr>("trip_count")) {
        node->trip_count = attr.getInt();
      } else {
        node->trip_count = computeTripCount(task);
      }
      
      // Overrides with explicit attributes if present (e.g. from manual tuning).
      if (auto attr = task->getAttrOfType<IntegerAttr>("steps")) {
        node->steps = attr.getInt();
      }
      if (auto attr = task->getAttrOfType<IntegerAttr>("compiled_ii")) {
        node->ii = attr.getInt();
      }
      if (auto attr = task->getAttrOfType<IntegerAttr>("cgra_count")) {
        node->cgra_count = attr.getInt();
      }
      
      op_to_node[task] = node.get();
      nodes.push_back(std::move(node));
    });

    // 2. Builds SSA edges (value dependencies between tasks).
    for (auto &consumer : nodes) {
      for (Value operand : consumer->op.getValueInputs()) {
        if (auto producer_op = operand.getDefiningOp<TaskflowTaskOp>()) {
          if (auto *producer = op_to_node[producer_op.getOperation()]) {
            addEdge(producer, consumer.get());
          }
        }
      }
    }

    // 3. Builds memory edges via SSA def-use on write_outputs.
    //
    //   RAW  (write → read):  producer's write_output consumed via
    //          consumer's read_memrefs.
    //   WAW  (write → write): producer's write_output consumed via
    //          consumer's write_memrefs (chain write).
    //   WAR  (read → write):  any task that consumed a memref via
    //          read_memrefs and whose write_output is then passed to
    //          another task's write_memrefs is already captured by the
    //          write chain above; the ordering is preserved by the SSA
    //          chain itself.
    for (auto &consumer : nodes) {
      // RAW: producer wrote a memref that this task reads.
      for (Value memref : consumer->op.getReadMemrefs()) {
        if (auto producer_op = memref.getDefiningOp<TaskflowTaskOp>()) {
          if (auto *producer = op_to_node[producer_op.getOperation()]) {
            addEdge(producer, consumer.get());
          }
        }
      }
      // WAW: producer wrote a memref that this task also writes.
      for (Value memref : consumer->op.getWriteMemrefs()) {
        if (auto producer_op = memref.getDefiningOp<TaskflowTaskOp>()) {
          if (auto *producer = op_to_node[producer_op.getOperation()]) {
            addEdge(producer, consumer.get());
          }
        }
      }
    }

    llvm::errs() << "TaskDependencyGraph: " << nodes.size()
                 << " tasks\n";
    for (auto &n : nodes) {
      llvm::errs() << "  Task " << n->id << " ("
                   << n->op.getTaskName().str() << "): trip_count="
                   << n->trip_count << ", ii=" << n->ii
                   << ", steps=" << n->steps
                   << ", preds=" << n->predecessors.size()
                   << ", succs=" << n->successors.size() << "\n";
    }
  }

  // Returns true if there is any (direct or transitive) dependency from
  // source_node to dest_node.
  bool hasDependency(TaskGraphNode *source_node,
                     TaskGraphNode *dest_node) const {
    if (source_node == dest_node) return true;
    DenseSet<TaskGraphNode *> visited;
    SmallVector<TaskGraphNode *> worklist;
    worklist.push_back(source_node);
    while (!worklist.empty()) {
      auto *current = worklist.pop_back_val();
      if (current == dest_node) return true;
      if (!visited.insert(current).second) continue;
      for (auto *succ : current->successors) {
        worklist.push_back(succ);
      }
    }
    return false;
  }

  // Returns true if a and b are completely independent (no path in either
  // direction).
  bool areIndependent(TaskGraphNode *a, TaskGraphNode *b) const {
    return !hasDependency(a, b) && !hasDependency(b, a);
  }

  // Returns total CGRAs allocated.
  int getTotalAllocatedCGRAs() const {
    int total = 0;
    for (auto &node : nodes) {
      total += node->cgra_count;
    }
    return total;
  }

  // Public wrapper for profileTask — used by UtilizationFuser to re-profile
  // fused tasks with the real downstream Neura pipeline.
  // When skip_mapper=true, only ResMII/RecMII analytical estimates are used
  // (no MapToAcceleratorPass). This is safe for speculative balance checks
  // where the mapper may backtrack indefinitely on larger tile arrays.
  void profileTaskPublic(TaskGraphNode *node, TaskflowTaskOp task,
                         bool skip_mapper = false) {
    profileTask(node, task, skip_mapper);
  }

private:
  llvm::DenseSet<std::pair<TaskGraphNode *, TaskGraphNode *>> edge_set;

  void addEdge(TaskGraphNode *from, TaskGraphNode *to) {
    auto key = std::make_pair(from, to);
    if (edge_set.insert(key).second) {
      from->successors.push_back(to);
      to->predecessors.push_back(from);
    }
  }

  // Profiles a single TaskflowTaskOp to extract compiled_ii and steps.
  //
  // Precondition: the pass runs AFTER full taskflow→neura lowering +
  // dataflow transformation, so each task body contains neura.kernel ops
  // in dataflow IR.  Typically one kernel per task, but fused tasks
  // (from streaming fusion) may contain multiple kernels.
  //
  // This method clones the task, extracts each kernel, wraps it in a
  // standalone func::FuncOp with accelerator="neura", and runs
  // InsertDataMov + MapToAcceleratorPass to obtain real compiled_ii.
  //
  // For multi-kernel fused tasks, kernels execute concurrently, so:
  //   ii    = max(ii across kernels)
  //   steps = max(steps across kernels)
  //
  // ASSERTS if no kernel is found — the pass must run post-lowering.
  //
  // skip_mapper: when true, skip MapToAcceleratorPass and use only
  //   ResMII/RecMII analytical estimates.  This is set by the
  //   --estimation-mode=analytical pass option, or internally for speculative
  //   balance probes where the mapper may loop indefinitely on large tile
  //   arrays.
  void profileTask(TaskGraphNode *node, TaskflowTaskOp task,
                   bool skip_mapper = false) {
    MLIRContext *ctx = task.getContext();
    OpBuilder builder(ctx);
    Location loc = task.getLoc();

    auto parent_func = task->getParentOfType<func::FuncOp>();
    assert(parent_func &&
           "[profileTask] FATAL: task has no parent func::FuncOp. "
           "compiled_ii must come from downstream pipeline.");

    // ================================================================
    // Kernel Extraction (post-lowering: tasks have neura.kernel ops)
    // ================================================================
    SmallVector<neura::KernelOp> preexisting_kernels;
    task.walk([&](neura::KernelOp k) { preexisting_kernels.push_back(k); });

    assert(!preexisting_kernels.empty() &&
           "[profileTask] FATAL: task has no neura.kernel ops. "
           "This pass must run after full taskflow-to-neura lowering.");

    // Clone the task into a temporary module so we don't mutate the real IR.
    SmallVector<neura::KernelOp> cloned_kernels;
    auto tmp_mod = ModuleOp::create(loc);
    {
      OpBuilder b(tmp_mod.getBodyRegion());
      IRMapping mapping;
      Operation* cloned_task = b.clone(*task.getOperation(), mapping);
      cast<TaskflowTaskOp>(cloned_task).walk([&](neura::KernelOp k) {
        cloned_kernels.push_back(k);
      });
    }

    // ================================================================
    // Run Neura pipeline on each kernel to get compiled_ii and steps
    // ================================================================
    int best_compiled_ii = 0;
    int best_cp_depth = 1;

    // Compute tile dimensions for the target CGRA shape.
    // Bounding box in tiles: x_tiles = cols * per_cgra_cols,
    //                        y_tiles = rows * per_cgra_rows.
    int per_cgra_cols = neura::getArchitecture().getPerCgraColumns();
    int per_cgra_rows = neura::getArchitecture().getPerCgraRows();
    int x_tiles = node->shape.cols * per_cgra_cols;
    int y_tiles = node->shape.rows * per_cgra_rows;
    std::string valid_tiles;
    if (!node->shape.is_rectangular) {
      // Build an explicit tile list from the shape's CGRA positions.
      // Each CGRA at position (cgra_c, cgra_r) contributes a per_cgra_cols ×
      // per_cgra_rows block of tiles.
      llvm::raw_string_ostream os(valid_tiles);
      for (auto &[cgra_c, cgra_r] : node->shape.cgra_positions) {
        for (int tr = 0; tr < per_cgra_rows; ++tr) {
          for (int tc = 0; tc < per_cgra_cols; ++tc) {
            if (!os.str().empty()) os << ",";
            os << (cgra_c * per_cgra_cols + tc)
               << "_"
               << (cgra_r * per_cgra_rows + tr);
          }
        }
      }
    }

    for (auto cloned_kernel : cloned_kernels) {
      auto phase2_module = ModuleOp::create(loc);
      int compiled_ii = 0;
      int cp_depth = 1;

      if (succeeded(
              runNeuraPipelineOnKernel(ctx, cloned_kernel, phase2_module,
                                      compiled_ii, cp_depth,
                                      x_tiles, y_tiles, valid_tiles,
                                      skip_mapper))) {
        llvm::errs() << "[profileTask] kernel in " << task.getTaskName()
                     << ": compiled_ii=" << compiled_ii
                     << ", cp_depth=" << cp_depth << "\n";
      } else {
        llvm::errs() << "[profileTask] Phase 2 failed for kernel in "
                     << task.getTaskName() << ", extracting partial\n";
        extractMetricsFromPartialIR(phase2_module, compiled_ii, cp_depth,
                                    x_tiles, y_tiles);
      }

      // Kernels in a fused task execute concurrently, so take max.
      best_compiled_ii = std::max(best_compiled_ii, compiled_ii);
      best_cp_depth = std::max(best_cp_depth, cp_depth);
      phase2_module.erase();
    }

    assert(best_compiled_ii > 0 &&
           "[profileTask] FATAL: compiled_ii is 0 after downstream pipeline. "
           "All profiling paths must produce a valid compiled_ii > 0.");
    node->ii = best_compiled_ii;
    node->steps = std::max(best_cp_depth, 1);

    llvm::errs() << "[profileTask] " << task.getTaskName()
                 << ": compiled_ii=" << node->ii
                 << ", steps=" << node->steps
                 << " (" << cloned_kernels.size() << " kernel(s))\n";

    // Erase the temporary module.
    tmp_mod.erase();
  }

  // Clones a neura.kernel body into a standalone func::FuncOp inside
  // dst_module, then runs InsertDataMov + mapper to get compiled_ii.
  //
  // Precondition: the kernel body is already in neura dataflow IR (all
  // lowering passes have been applied before this pass runs).  Only
  // InsertDataMov is needed before the mapper.
  //
  // Returns success if MapToAccelerator ran and produced compiled_ii.
  //
  // x_tiles / y_tiles: total tile dimensions of the target CGRA array.
  //   These are passed to MapToAcceleratorPass so it maps onto the correct
  //   multi-CGRA tile grid rather than the default 1-CGRA singleton.
  // valid_tiles: explicit comma-separated tile list for non-rectangular shapes.
  //   Empty string means "use the full x_tiles × y_tiles rectangle".
  // skip_mapper: when true, skip MapToAcceleratorPass entirely and rely only
  //   on ResMII/RecMII analytical estimates. Used for speculative balance
  //   probes to prevent infinite mapper backtracking on larger tile arrays.
  LogicalResult runNeuraPipelineOnKernel(MLIRContext *ctx,
                                         neura::KernelOp kernel,
                                         ModuleOp dst_module,
                                         int &compiled_ii,
                                         int &cp_depth,
                                         int x_tiles = 0,
                                         int y_tiles = 0,
                                         const std::string &valid_tiles = "",
                                         bool skip_mapper = false) {
    Location loc = kernel.getLoc();
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(dst_module.getBody());

    // Builds function signature: all kernel inputs + iter_args as arguments.
    Region &kernel_body = kernel.getBody();
    if (kernel_body.empty())
      return failure();

    Block &entry = kernel_body.front();
    SmallVector<Type> arg_types;
    for (BlockArgument arg : entry.getArguments())
      arg_types.push_back(arg.getType());

    // Result types from the kernel op.
    SmallVector<Type> result_types(kernel.getResultTypes());

    auto func_type = builder.getFunctionType(arg_types, result_types);
    auto wrapper_func = builder.create<func::FuncOp>(
        loc, "__speculative_kernel__", func_type);

    // Tags as neura accelerator — all downstream passes check this.
    wrapper_func->setAttr("accelerator",
                          builder.getStringAttr("neura"));

    // Clones the entire kernel region (all blocks) into the func body.
    Region &func_region = wrapper_func.getBody();
    IRMapping mapping;
    kernel_body.cloneInto(&func_region, mapping);

    // The cloned region now contains a copy of every block from the kernel.
    // Walk through and replace neura.yield terminators with func.return.
    for (Block &block : func_region) {
      if (auto yield = dyn_cast<neura::YieldOp>(block.getTerminator())) {
        builder.setInsertionPoint(yield);
        SmallVector<Value> return_vals;
        for (Value v : yield.getResults()) {
          return_vals.push_back(v);
        }
        builder.create<func::ReturnOp>(loc, return_vals);
        yield.erase();
      }
    }

    // Since this pass runs after the full neura lowering + dataflow pipeline
    // (lower-affine, convert-scf-to-cf, convert-cf-to-llvm, assign-accelerator,
    // lower-memref-to-neura, lower-arith-to-neura, lower-builtin-to-neura,
    // lower-llvm-to-neura, promote-input-arg-to-const, canonicalize-*,
    // transform-ctrl-to-data-flow), the kernel body is already in neura
    // dataflow IR.  Only InsertDataMov is needed before the mapper.
    PassManager pm(ctx);
    pm.enableVerifier(false);

    // InsertDataMov: wraps operands with neura.data_mov for the mapper.
    pm.addPass(neura::createInsertDataMovPass());

    if (failed(pm.run(dst_module))) {
      // Pre-mapper pipeline failed — extract best-effort metrics from partial
      // Neura IR using ResMII/RecMII analysis with the correct multi-CGRA arch.
      extractMetricsFromPartialIR(dst_module, compiled_ii, cp_depth,
                                  x_tiles, y_tiles);
      return failure();
    }

    // Extracts ResMII/RecMII from the post-InsertDataMov Neura IR. These are
    // the authoritative lower-bounds and the fallback metrics when the mapper
    // is skipped. We compute them now (before MapToAccelerator modifies the IR
    // with dfg_id attrs) so that the fallback always uses the same IR.
    //
    // Uses a custom architecture sized to the actual tile array
    // (x_tiles × y_tiles) so ResMII reflects the real resource pool.
    // Fall back to the global singleton if tile dims are not specified.
    {
      std::unique_ptr<neura::Architecture> custom_arch;
      const neura::Architecture *arch_ptr = &neura::getArchitecture();
      if (x_tiles > 0 && y_tiles > 0) {
        custom_arch = neura::getArchitecture().cloneWithNewDimensions(
            y_tiles, x_tiles);
        arch_ptr = custom_arch.get();
      }
      const neura::Architecture &architecture = *arch_ptr;

      dst_module.walk([&](func::FuncOp fn) {
        if (!fn->hasAttr("accelerator")) return;
        Region &region = fn.getBody();
        if (region.empty()) return;
        int res_mii = neura::calculateResMii(region, architecture);
        auto cycles = neura::collectRecurrenceCycles(region);
        int rec_mii = 1;
        for (auto &cycle : cycles)
          rec_mii = std::max(rec_mii, cycle.length);
        compiled_ii = std::max({compiled_ii, res_mii, rec_mii});
        // cp_depth from ALAP.
        std::set<Operation *> critical_ops;
        for (auto &cycle : cycles)
          for (Operation *op : cycle.operations) critical_ops.insert(op);
        auto sorted_ops = neura::getTopologicallySortedOps(region);
        if (!sorted_ops.empty()) {
          auto level_buckets = neura::getOpsInAlapLevels(sorted_ops, critical_ops);
          cp_depth = std::max(cp_depth, (int)level_buckets.size());
        }
        llvm::errs() << "[profileTask] analytical fallback: res_mii=" << res_mii
                     << " rec_mii=" << rec_mii
                     << " tiles=" << architecture.getNumTiles() << "\n";
      });
    }

    // Optionally run MapToAcceleratorPass to get the true compiled_ii.
    //
    // Guards:
    //   1. skip_mapper=true: caller explicitly requests analytical-only (e.g.
    //      speculative balance probes where the mapper may loop indefinitely).
    //   2. All non-Reserve operand producers must be DataMovOp (mapper asserts
    //      otherwise).
    //   3. Kernel must be small enough (<= kMapperOpLimit ops) to avoid
    //      exponential backtracking blowup during speculative profiling.
    //
    // If any guard fires, the ResMII/RecMII values computed above serve as
    // the analytical lower-bound estimate (under-estimates true II on smaller
    // arrays, but are safe and instant).
    if (skip_mapper) {
      llvm::errs() << "[profileTask] Skipping mapper (analytical-only mode). "
                   << "Using analytical compiled_ii=" << compiled_ii << "\n";
      return success();
    }

    constexpr int kMapperOpLimit = 150;
    bool all_data_movs_ok = true;
    int total_mapped_ops = 0;
    dst_module.walk([&](func::FuncOp fn) {
      if (!fn->hasAttr("accelerator")) return;
      fn.walk([&](Operation *op) {
        if (isa<func::ReturnOp>(op)) return;
        total_mapped_ops++;
        if (isa<neura::ReserveOp, neura::DataMovOp, neura::CtrlMovOp>(op))
          return;
        for (Value operand : op->getOperands()) {
          Operation *producer = operand.getDefiningOp();
          if (!producer) continue;
          if (!isa<neura::DataMovOp, neura::ReserveOp,
                   neura::PhiStartOp, neura::GrantOnceOp,
                   neura::GrantPredicateOp, neura::YieldOp,
                   neura::KernelOp>(producer))
            all_data_movs_ok = false;
        }
      });
    });

    llvm::errs() << "[profileTask] mapper guard: total_ops=" << total_mapped_ops
                 << " all_data_movs=" << all_data_movs_ok
                 << " limit=" << kMapperOpLimit << "\n";

    if (all_data_movs_ok && total_mapped_ops <= kMapperOpLimit) {
      // Runs MapToAcceleratorPass in a fresh pass manager on the already-lowered
      // dst_module (pre-mapper pipeline already ran above).
      // Pass the correct tile dimensions so the mapper uses the right array.
      PassManager pm2(ctx);
      pm2.enableVerifier(false);
      if (x_tiles > 0 && y_tiles > 0) {
        neura::MapToAcceleratorOptions map_options;
        map_options.x_tiles = x_tiles;
        map_options.y_tiles = y_tiles;
        map_options.valid_tiles = valid_tiles;
        pm2.addPass(neura::createMapToAcceleratorPass(map_options));
      } else {
        pm2.addPass(neura::createMapToAcceleratorPass());
      }

      if (succeeded(pm2.run(dst_module))) {
        // Reads the true compiled_ii from mapping_info (overrides ResMII/RecMII).
        // compiled_ii and cp_depth are already initialized from the pre-mapper
        // ResMII/RecMII analysis above; mapper result takes precedence.
        dst_module.walk([&](func::FuncOp fn) {
          if (!fn->hasAttr("accelerator")) return;
          if (auto mapping_info =
                  fn->getAttrOfType<DictionaryAttr>(neura::attr::kMappingInfo)) {
            if (auto ii_attr =
                    mapping_info.getAs<IntegerAttr>(neura::attr::kCompiledII)) {
              compiled_ii = (int)ii_attr.getInt(); // authoritative value
              llvm::errs() << "[profileTask] mapper returned real II="
                           << compiled_ii << "\n";
            }
          }
        });
        return success();
      }
      // Mapper failed for all II values — keep ResMII/RecMII from above.
      llvm::errs() << "[profileTask] WARNING: MapToAcceleratorPass failed, "
                   << "keeping analytical fallback compiled_ii=" << compiled_ii
                   << "\n";
    } else {
      llvm::errs() << "[profileTask] Skipping mapper (too large or DataMov "
                   << "check failed). Using analytical compiled_ii="
                   << compiled_ii << "\n";
    }

    // Fallback already computed via ResMII/RecMII above; nothing more to do.
    return success();
  }


  // Extracts metrics from partially-lowered Neura IR when the full pipeline
  // fails. Uses ResMII/RecMII analysis and critical path depth on whatever
  // Neura ops were successfully created.
  //
  // x_tiles / y_tiles: if > 0, use a custom architecture sized to this tile
  //   array so that ResMII reflects the real resource pool for multi-CGRA
  //   shapes. Falls back to the global singleton (1-CGRA) otherwise.
  void extractMetricsFromPartialIR(ModuleOp tmp_module,
                                   int &out_ii, int &out_cp_depth,
                                   int x_tiles = 0, int y_tiles = 0) {
    // Build architecture: use custom tile dimensions if provided.
    std::unique_ptr<neura::Architecture> custom_arch;
    const neura::Architecture *arch_ptr = &neura::getArchitecture();
    if (x_tiles > 0 && y_tiles > 0) {
      custom_arch = neura::getArchitecture().cloneWithNewDimensions(
          y_tiles, x_tiles);
      arch_ptr = custom_arch.get();
    }
    const neura::Architecture &architecture = *arch_ptr;

    int res_mii = 1;
    int rec_mii = 1;
    int cp_depth = 1;

    // Try func-level analysis on partially-lowered funcs.
    tmp_module.walk([&](func::FuncOp fn) {
      if (!fn->hasAttr("accelerator"))
        return;
      Region &region = fn.getBody();
      if (region.empty())
        return;

      int local_res = neura::calculateResMii(region, architecture);
      res_mii = std::max(res_mii, local_res);

      auto cycles = neura::collectRecurrenceCycles(region);
      std::set<Operation *> critical_ops;
      for (auto &cycle : cycles) {
        rec_mii = std::max(rec_mii, (int)cycle.length);
        for (Operation *op : cycle.operations)
          critical_ops.insert(op);
      }

      auto sorted_ops = neura::getTopologicallySortedOps(region);
      if (!sorted_ops.empty()) {
        auto level_buckets =
            neura::getOpsInAlapLevels(sorted_ops, critical_ops);
        cp_depth = std::max(cp_depth, (int)level_buckets.size());
      }
    });

    out_ii = std::max(res_mii, rec_mii);
    out_cp_depth = std::max(cp_depth, 1);

    llvm::errs() << "[profileTask] (partial) ii=" << out_ii
                 << " (res_mii=" << res_mii
                 << ", rec_mii=" << rec_mii
                 << "), steps=" << out_cp_depth << "\n";
  }

  // Computes total trip count for a task.
  //
  // Post-lowering, the task body contains taskflow.counter ops and
  // neura.kernel ops.  The trip count is computed from:
  //   1. taskflow.counter ops (product of counter chain ranges), or
  //   2. neura.counter ops inside kernels (if no taskflow.counter exists).
  //
  // For each root counter (no parent_index), walks its chain of child counters
  // and multiplies all (upper_bound - lower_bound) / step values.
  // Multiple independent counter chains execute concurrently, so the trip
  // count is max(chain_product) across chains.
  static int64_t computeTripCount(TaskflowTaskOp task) {
    // Collects all counter ops in the task body.
    SmallVector<TaskflowCounterOp> all_counters;
    task.walk([&](TaskflowCounterOp c) { all_counters.push_back(c); });

    if (all_counters.empty()) {
      // Post-neura-lowering path: task bodies contain neura.kernel ops with
      // neura.counter ops instead of taskflow.counter ops.
      // Compute trip count as the product of all "leaf" neura.counter ranges
      // across all kernels (each kernel has one leaf counter chain driving
      // the innermost loop).  Multiple independent kernels are summed.
      SmallVector<neura::CounterOp> leaf_counters;
      task.walk([&](neura::CounterOp nc) {
        if (auto ct = nc->getAttrOfType<StringAttr>("counter_type"))
          if (ct.getValue() == "leaf")
            leaf_counters.push_back(nc);
      });

      if (!leaf_counters.empty()) {
        // For each kernel, find all its leaf counters and multiply their
        // ranges, then sum across kernels.
        int64_t total = 0;
        task.walk([&](neura::KernelOp kernel) {
          int64_t kernel_trip = 1;
          bool has_leaf = false;
          kernel.walk([&](neura::CounterOp nc) {
            if (auto ct = nc->getAttrOfType<StringAttr>("counter_type")) {
              if (ct.getValue() == "leaf" || ct.getValue() == "relay" ||
                  ct.getValue() == "root") {
                auto lb = nc.getLowerBound().getSExtValue();
                auto ub = nc.getUpperBound().getSExtValue();
                auto step = nc.getStep().getSExtValue();
                int64_t count = (step > 0) ? (ub - lb + step - 1) / step : 1;
                kernel_trip *= (count > 0 ? count : 1);
                if (ct.getValue() == "leaf")
                  has_leaf = true;
              }
            }
          });
          if (has_leaf)
            total += kernel_trip;
        });
        return total > 0 ? total : 1;
      }

      // No counters found at all — default to 1.
      return 1;
    }

    // Builds a map from counter result -> counter op for parent chain traversal.
    // Also finds root counters (no parent_index).
    SmallVector<TaskflowCounterOp> root_counters;
    DenseMap<Value, TaskflowCounterOp> result_to_counter;
    for (auto c : all_counters) {
      // Counter op produces an induction variable result.
      if (c->getNumResults() > 0)
        result_to_counter[c->getResult(0)] = c;
      if (!c.getParentIndex())
        root_counters.push_back(c);
    }

    // For each root counter, computes the product of its chain.
    // Independent chains execute concurrently, so take max.
    int64_t total = 0;
    for (auto root : root_counters) {
      int64_t chain_product = 1;

      // Walks all counters and multiplies those in this root's chain.
      // A counter belongs to this chain if it IS the root, or its
      // parent_index traces back to the root.
      for (auto c : all_counters) {
        int64_t lb = c.getLowerBound().getSExtValue();
        int64_t ub = c.getUpperBound().getSExtValue();
        int64_t step = c.getStep().getSExtValue();
        int64_t count = (step > 0) ? (ub - lb + step - 1) / step : 1;
        if (count < 1) count = 1;

        if (c == root) {
          chain_product *= count;
        } else if (c.getParentIndex()) {
          // Checks if this counter's parent is in the same chain.
          // Walks up to see if we reach the root.
          Value parent = c.getParentIndex();
          bool in_chain = false;
          // Simple check: if parent is the root's result, it's in chain.
          // For deeper nesting, traces iteratively.
          while (parent) {
            auto it = result_to_counter.find(parent);
            if (it == result_to_counter.end())
              break;
            TaskflowCounterOp parent_counter = it->second;
            if (parent_counter == root) {
              in_chain = true;
              break;
            }
            parent = parent_counter.getParentIndex();
          }
          if (in_chain)
            chain_product *= count;
        }
      }
      total = std::max(total, chain_product);
    }

    return (total > 0) ? total : 1;
  }

};

//===----------------------------------------------------------------------===//
// Pipeline Balancer
//===----------------------------------------------------------------------===//
// Identifies critical-path bottlenecks and allocates extra CGRAs.

class PipelineBalancer {
public:
  using ProfileFn = std::function<void(TaskGraphNode *, TaskflowTaskOp)>;

  // Runs pipeline balance on the graph.
  //
  // For each iteration, speculatively increments the bottleneck task's
  // cgra_count by 1 and re-profiles it via profile_fn. If the new estimated
  // latency is lower, the change is accepted; otherwise it is reverted and
  // the node is marked saturated (no further CGRA additions help it).
  //
  // This avoids blindly assigning more CGRAs without checking whether the
  // larger array actually produces a better compiled_ii.
  //
  // Returns true if any changes were accepted.
  bool balance(TaskDependencyGraph &graph, ProfileFn profile_fn) {
    bool changed = false;
    // Tracks nodes for which adding one more CGRA did not reduce latency.
    // These are skipped in subsequent iterations.
    llvm::DenseSet<TaskGraphNode *> saturated_nodes;

    for (int iter = 0; iter < kMaxBalanceIterations; ++iter) {
      int total_cgras = graph.getTotalAllocatedCGRAs();
      if (total_cgras >= kTotalCGRAs) {
        break;
      }

      // Find the bottleneck: the node on the critical path with highest
      // estimated latency. We recompute the critical path every iteration
      // because adding CGRAs to the previous bottleneck may shift the
      // critical path to a different node.
      TaskGraphNode *bottleneck = findBottleneck(graph, saturated_nodes);
      if (!bottleneck) {
        break;
      }

      int old_cgra_count = bottleneck->cgra_count;
      int new_cgra_count = old_cgra_count + 1;

      // Check if incrementing cgra_count is feasible on the 4×4 grid.
      // TODO: This currently only checks the capacity (total CGRA count). Ideally, 
      // we should invoke a global placement pass (aka MapTaskOnCgraPass) here to 
      // verify if the speculatively increased CGRA count and its proposed shape 
      // actually fit on the 4x4 grid alongside other previously allocated tasks.
      //
      // Currently, MapTaskOnCgraPass does not support multi-CGRA task placement. 
      // Once it does, we should call it here; if global placement fails for the 
      // "best" shape, we should backtrack and try alternative shapes before 
      // saturating the node.
      if (!canFitOnGrid(new_cgra_count)) {
        saturated_nodes.insert(bottleneck);
        continue;
      }

      // Saves state for potential rollback.
      int64_t old_latency = bottleneck->estimatedLatency();
      int64_t old_ii     = bottleneck->ii;
      int64_t old_steps  = bottleneck->steps;
      CgraShape old_shape = bottleneck->shape;

      // Speculatively applies the new CGRA count and re-profiles.
      bottleneck->cgra_count = new_cgra_count;
      bottleneck->shape = pickBestShape(new_cgra_count);

      llvm::errs()
          << "  Balance: trying Task " << bottleneck->id << " ("
          << bottleneck->op.getTaskName().str()
          << ") cgra_count=" << old_cgra_count << "->" << new_cgra_count
          << ", shape=" << bottleneck->shape.describe(new_cgra_count)
          << ", tile_array="
          << (bottleneck->shape.rows * neura::getArchitecture().getPerCgraRows())
          << "x"
          << (bottleneck->shape.cols * neura::getArchitecture().getPerCgraColumns())
          << ", old_ii=" << old_ii << ", old_lat=" << old_latency << "\n";

      profile_fn(bottleneck, bottleneck->op);

      int64_t new_latency = bottleneck->estimatedLatency();

      if (new_latency < old_latency) {
        // Accepted: the larger array produces a measurably better latency.
        changed = true;
        llvm::errs()
            << "  Balance: ACCEPTED Task " << bottleneck->id << " ("
            << bottleneck->op.getTaskName().str()
            << ") cgra_count=" << new_cgra_count
            << ", ii=" << old_ii << "->" << bottleneck->ii
            << ", lat=" << old_latency << "->" << new_latency
            << ", total_cgras=" << graph.getTotalAllocatedCGRAs() << "\n";
      } else {
        // Rejected: no latency improvement — roll back and mark saturated.
        llvm::errs()
            << "  Balance: REJECTED Task " << bottleneck->id
            << " (ii=" << bottleneck->ii << ", lat=" << new_latency
            << " >= old_lat=" << old_latency << "). Reverting.\n";
        bottleneck->cgra_count = old_cgra_count;
        bottleneck->shape      = old_shape;
        bottleneck->ii         = old_ii;
        bottleneck->steps      = old_steps;
        saturated_nodes.insert(bottleneck);
      }
    }

    return changed;
  }

  private:
    // Computes the weighted critical path length from a given node to any sink.
    int64_t computeCriticalPathFrom(TaskGraphNode *node,
                                    DenseMap<TaskGraphNode *, int64_t> &cache) {
      auto it = cache.find(node);
      if (it != cache.end()) {
        return it->second;
      }

      int64_t max_successor_path = 0;
      for (auto *succ : node->successors) {
        max_successor_path =
            std::max(max_successor_path, computeCriticalPathFrom(succ, cache));
      }

      int64_t path = node->estimatedLatency() + max_successor_path;
      cache[node] = path;
      return path;
    }

    // Computes the longest path from any source to the given node
    // (depth_from_source). Uses dynamic programming with memoization.
    int64_t computeDepthFromSource(TaskGraphNode *node,
                                   DenseMap<TaskGraphNode *, int64_t> &cache) {
      auto it = cache.find(node);
      if (it != cache.end()) {
        return it->second;
      }

      int64_t max_predecessor_depth = 0;
      for (auto *pred : node->predecessors) {
        max_predecessor_depth =
            std::max(max_predecessor_depth,
                     computeDepthFromSource(pred, cache));
      }

      // depth_from_source(node) = max(depth_from_source(pred) for all preds)
      //                           + node's own latency.
      int64_t depth = max_predecessor_depth + node->estimatedLatency();
      cache[node] = depth;
      return depth;
    }

    // Finds the bottleneck node on the critical path using full slack analysis.
    //
    // For each node, slack is defined as:
    //   slack(node) = global_critical_path
    //                 - depth_from_source(node)
    //                 - depth_to_sink(node)
    //                 + node->estimatedLatency()
    //
    // where depth_from_source includes the node's own latency, and
    // depth_to_sink (computeCriticalPathFrom) also includes the node's own
    // latency, so we add it back once to avoid double-counting.
    //
    // A node is on the critical path iff slack == 0.
    // Among critical-path nodes, the one with highest individual latency
    // is the bottleneck (reducing its latency most benefits the pipeline).
    TaskGraphNode *findBottleneck(TaskDependencyGraph &graph,
                                  const llvm::DenseSet<TaskGraphNode *> &ignored) {
      llvm::DenseMap<TaskGraphNode *, int64_t> to_sink_cache;
      llvm::DenseMap<TaskGraphNode *, int64_t> from_source_cache;

      // Compute depth_to_sink for all nodes (via computeCriticalPathFrom).
      int64_t global_critical_path = 0;
      for (auto &node : graph.nodes) {
        int64_t cp = computeCriticalPathFrom(node.get(), to_sink_cache);
        global_critical_path = std::max(global_critical_path, cp);
      }

      // Compute depth_from_source for all nodes.
      for (auto &node : graph.nodes) {
        computeDepthFromSource(node.get(), from_source_cache);
      }

      // Finds the critical-path node with highest individual latency.
      TaskGraphNode *bottleneck = nullptr;
      int64_t max_latency = -1;

      for (auto &node : graph.nodes) {
        if (ignored.count(node.get())) continue;
        if (node->cgra_count >= node->trip_count) continue;
        // Per-task CGRA limit: no point trying to add more.
        if (node->cgra_count >= kMaxCgrasPerTask) continue;

        int64_t depth_from = from_source_cache[node.get()];
        int64_t depth_to = to_sink_cache[node.get()];

        // slack = global_cp - depth_from - depth_to + node_latency
        // (because both depth_from and depth_to include node's own latency).
        int64_t slack = global_critical_path - depth_from - depth_to
                        + node->estimatedLatency();

        if (slack != 0) continue; // Not on the critical path.

        if (node->estimatedLatency() > max_latency) {
          max_latency = node->estimatedLatency();
          bottleneck = node.get();
        }
      }
      return bottleneck;
    }

};

//===----------------------------------------------------------------------===//
// Utilization Fusion
//===----------------------------------------------------------------------===//
// Merges independent tasks (no edge in either direction) into a single task
// to reduce total CGRA count.  Fusion candidates are chosen to minimize
// |trip_count_a - trip_count_b| for balanced utilization.

class UtilizationFuser {
public:
  using ProfileFn = std::function<void(TaskGraphNode *, TaskflowTaskOp)>;

  // Runs utilization fusion. Returns true if any fusions occurred.
  // Only performs ONE fusion per call — the caller should rebuild the graph
  // and call again if more fusions are desired.
  bool fuse(func::FuncOp func, TaskDependencyGraph &graph,
            ProfileFn profile_fn) {
    auto pair = findBestFusionCandidate(graph);
    if (!pair) {
      return false;
    }

    auto [node_a, node_b] = *pair;

    llvm::errs()
        << "  Fuse: Task " << node_a->id << " ("
        << node_a->op.getTaskName().str() << ") + Task " << node_b->id
        << " (" << node_b->op.getTaskName().str() << ")\n";

    return performFusion(func, node_a, node_b, graph, profile_fn);
  }

private:
  // Finds the best pair of independent tasks to fuse.
  // Selects the pair with the most balanced trip_count (minimizes
  // |trip_count_a - trip_count_b|) to avoid wasting computation when
  // the fused task executes both loop nests concurrently on the shared array.
  std::optional<std::pair<TaskGraphNode *, TaskGraphNode *>>
  findBestFusionCandidate(TaskDependencyGraph &graph) {
    TaskGraphNode *best_a = nullptr;
    TaskGraphNode *best_b = nullptr;
    int64_t best_cost = INT64_MAX;

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
      for (size_t j = i + 1; j < graph.nodes.size(); ++j) {
        auto *a = graph.nodes[i].get();
        auto *b = graph.nodes[j].get();

        if (!graph.areIndependent(a, b)) {
          continue;
        }

        // Legality: check no intermediate task depends on a or b.
        if (!canSafelyFuse(a, b, graph)) {
          continue;
        }

        // Utilization metric: minimize |trip_count_a - trip_count_b|.
        // Balanced trip counts mean less wasted computation when fused
        // tasks execute concurrently on the shared tile array.
        int64_t cost = std::abs(a->trip_count - b->trip_count);
        if (cost < best_cost) {
          best_cost = cost;
          best_a = a;
          best_b = b;
        }
      }
    }

    if (!best_a || !best_b) {
      return std::nullopt;
    }
    return std::make_pair(best_a, best_b);
  }

  // Checks whether fusing tasks a and b is safe w.r.t. dominance.
  // Returns false if any other task positioned between a and b in the IR
  // has a dependency (edge) on either a or b — because moving the fused
  // task would break that intermediate dependency.
  bool canSafelyFuse(TaskGraphNode *a, TaskGraphNode *b,
                     TaskDependencyGraph &graph) {
    auto *task_a = a->op.getOperation();
    auto *task_b = b->op.getOperation();

    if (task_a->getBlock() != task_b->getBlock()) return false;

    // Ensure task_a is before task_b.
    if (!task_a->isBeforeInBlock(task_b)) {
      std::swap(task_a, task_b);
      std::swap(a, b);
    }

    // Check: no other task between a and b should have an edge from/to a or b.
    for (auto &node : graph.nodes) {
      if (node.get() == a || node.get() == b) continue;

      auto *other_op = node->op.getOperation();
      if (other_op->getBlock() != task_a->getBlock()) continue;

      // Is this node between task_a and task_b?
      if (task_a->isBeforeInBlock(other_op) &&
          other_op->isBeforeInBlock(task_b)) {
        // Check if this intermediate task has any dependency on a or b.
        if (!graph.areIndependent(a, node.get()) ||
            !graph.areIndependent(b, node.get())) {
          return false;
        }
      }
    }
    return true;
  }

  // Performs IR-level fusion of two independent tasks.
  //
  // DFG-Level Fusion:
  //   Since this pass runs post-lowering, each task body already contains
  //   one neura.kernel op in dataflow IR.  Fusion concatenates both DFGs
  //   into a single neura.kernel (they are independent, so just placed
  //   side-by-side).  The fused task is then profiled through
  //   InsertDataMov + mapper to get accurate compiled_ii.
  bool performFusion(func::FuncOp func, TaskGraphNode *node_a,
                     TaskGraphNode *node_b, TaskDependencyGraph &graph,
                     ProfileFn profile_fn) {
    auto task_a = node_a->op;
    auto task_b = node_b->op;

    // Safety: both tasks must be in the same block.
    if (task_a->getBlock() != task_b->getBlock()) {
      llvm::errs() << "  [Fuse] Skipping: tasks in different blocks\n";
      return false;
    }

    // Ensures task_a comes before task_b in the IR for correct dominance.
    if (!task_a->isBeforeInBlock(task_b)) {
      std::swap(task_a, task_b);
      std::swap(node_a, node_b);
    }

    llvm::errs() << "  [Fuse] Merging " << task_a.getTaskName() << " + "
                 << task_b.getTaskName() << "\n";
    llvm::errs() << "[Fuse] task_a body before fusion:\n";
    task_a.dump();
    llvm::errs() << "[Fuse] task_b body before fusion:\n";
    task_b.dump();

    // Computes the correct insertion point: must be after all operands of
    // both tasks are defined, but before any consumer of either task's
    // results. We find the latest-positioned operand definition and insert
    // right after it.
    Operation *latest_def = task_a.getOperation();
    auto updateLatest = [&](ValueRange operands) {
      for (Value v : operands) {
        if (auto *def_op = v.getDefiningOp()) {
          if (def_op->getBlock() == task_a->getBlock() &&
              latest_def->isBeforeInBlock(def_op)) {
            latest_def = def_op;
          }
        }
      }
    };
    updateLatest(task_a.getReadMemrefs());
    updateLatest(task_a.getWriteMemrefs());
    updateLatest(task_a.getValueInputs());
    updateLatest(task_b.getReadMemrefs());
    updateLatest(task_b.getWriteMemrefs());
    updateLatest(task_b.getValueInputs());

    // Inserts right after the latest operand definition.
    OpBuilder builder(latest_def->getBlock(),
                      std::next(Block::iterator(latest_def)));

    // Step 1: Builds merged operand lists.
    SmallVector<Value> merged_read_memrefs;
    SmallVector<Value> merged_write_memrefs;
    SmallVector<Value> merged_value_inputs;
    SmallVector<Value> merged_original_read_memrefs;
    SmallVector<Value> merged_original_write_memrefs;

    auto addUnique = [](SmallVector<Value> &target, ValueRange source) {
      for (Value v : source) {
        if (llvm::find(target, v) == target.end()) {
          target.push_back(v);
        }
      }
    };

    addUnique(merged_read_memrefs, task_a.getReadMemrefs());
    addUnique(merged_read_memrefs, task_b.getReadMemrefs());
    addUnique(merged_write_memrefs, task_a.getWriteMemrefs());
    addUnique(merged_write_memrefs, task_b.getWriteMemrefs());
    addUnique(merged_value_inputs, task_a.getValueInputs());
    addUnique(merged_value_inputs, task_b.getValueInputs());
    addUnique(merged_original_read_memrefs, task_a.getOriginalReadMemrefs());
    addUnique(merged_original_read_memrefs, task_b.getOriginalReadMemrefs());
    addUnique(merged_original_write_memrefs, task_a.getOriginalWriteMemrefs());
    addUnique(merged_original_write_memrefs, task_b.getOriginalWriteMemrefs());

    // Step 2: Builds result types.
    // Write outputs = merged write memrefs (each becomes a result).
    SmallVector<Type> write_output_types;
    for (Value v : merged_write_memrefs) {
      write_output_types.push_back(v.getType());
    }
    // Value outputs: union from both tasks.
    SmallVector<Type> value_output_types;
    // For independent tasks, we collect value outputs from both.
    // But for utilization fusion of independent tasks, value_outputs are rare.
    // We include them for correctness.
    for (Value v : task_a.getValueOutputs()) {
      value_output_types.push_back(v.getType());
    }
    for (Value v : task_b.getValueOutputs()) {
      value_output_types.push_back(v.getType());
    }

    // Step 3: Creates fused task name.
    std::string fused_name = task_a.getTaskName().str() + "_" +
                             task_b.getTaskName().str() + "_utilfused";

    // Step 4: Creates the fused task op using the correct API.
    auto fused_task = builder.create<TaskflowTaskOp>(
        task_a.getLoc(), write_output_types, value_output_types,
        merged_read_memrefs, merged_write_memrefs, merged_value_inputs,
        fused_name, merged_original_read_memrefs,
        merged_original_write_memrefs);

    // ================================================================
    // Region-Level Fusion (handles multi-block task bodies)
    // ================================================================
    // After CF lowering, task bodies may contain multiple blocks with
    // llvm.br/llvm.cond_br connecting them. The neura.kernel and
    // taskflow.yield ops may be in non-entry blocks. We use
    // Region::cloneInto() to clone the entire region, preserving all
    // control flow, then find and merge the cloned kernels.

    // Step 5: Clone both task regions into the fused task body.
    // First, build block-arg mappings for each source task.
    auto buildTaskArgMapping =
        [&](TaskflowTaskOp orig_task, Region &fused_region,
            IRMapping &mapping) {
      Block &src_entry = orig_task.getBody().front();
      // The entry block of the cloned region will be added by cloneInto.
      // We need to pre-map the entry block args to the fused task's entry
      // block args. But since cloneInto creates new blocks, we map the
      // source entry block args before cloning.
      unsigned src_idx = 0;
      unsigned read_count = orig_task.getReadMemrefs().size();
      unsigned write_count = orig_task.getWriteMemrefs().size();

      // We'll create a temporary entry block for the fused task if it
      // doesn't exist yet, and add block args.
      // For now, we just set up the mapping from source block args to
      // values that will exist in the fused region.

      // Map read_memrefs block args.
      for (unsigned i = 0; i < read_count; ++i) {
        Value orig_memref = orig_task.getReadMemrefs()[i];
        auto it = llvm::find(merged_read_memrefs, orig_memref);
        assert(it != merged_read_memrefs.end());
        unsigned fused_idx = std::distance(merged_read_memrefs.begin(), it);
        mapping.map(src_entry.getArgument(src_idx + i),
                    fused_region.front().getArgument(fused_idx));
      }
      src_idx += read_count;

      // Map write_memrefs block args.
      for (unsigned i = 0; i < write_count; ++i) {
        Value orig_memref = orig_task.getWriteMemrefs()[i];
        auto it = llvm::find(merged_write_memrefs, orig_memref);
        assert(it != merged_write_memrefs.end());
        unsigned fused_idx = merged_read_memrefs.size() +
                             std::distance(merged_write_memrefs.begin(), it);
        mapping.map(src_entry.getArgument(src_idx + i),
                    fused_region.front().getArgument(fused_idx));
      }
      src_idx += write_count;

      // Map value_inputs block args.
      for (unsigned i = 0; i < orig_task.getValueInputs().size(); ++i) {
        Value orig_val = orig_task.getValueInputs()[i];
        auto it = llvm::find(merged_value_inputs, orig_val);
        assert(it != merged_value_inputs.end());
        unsigned fused_idx = merged_read_memrefs.size() +
                             merged_write_memrefs.size() +
                             std::distance(merged_value_inputs.begin(), it);
        mapping.map(src_entry.getArgument(src_idx + i),
                    fused_region.front().getArgument(fused_idx));
      }
    };

    // Creates the fused task's entry block with merged block args.
    Block *entry_block = new Block();
    fused_task.getBody().push_back(entry_block);
    for (Value v : merged_read_memrefs)
      entry_block->addArgument(v.getType(), fused_task.getLoc());
    for (Value v : merged_write_memrefs)
      entry_block->addArgument(v.getType(), fused_task.getLoc());
    for (Value v : merged_value_inputs)
      entry_block->addArgument(v.getType(), fused_task.getLoc());

    // Clones task_a's entire region into the fused region.
    IRMapping mapping_a;
    buildTaskArgMapping(task_a, fused_task.getBody(), mapping_a);
    task_a.getBody().cloneInto(&fused_task.getBody(), mapping_a);

    // Clones task_b's entire region into the fused region.
    IRMapping mapping_b;
    buildTaskArgMapping(task_b, fused_task.getBody(), mapping_b);
    task_b.getBody().cloneInto(&fused_task.getBody(), mapping_b);

    // cloneInto creates new entry blocks for each cloned region.
    // We need to splice the cloned entry blocks' ops into our entry block
    // and redirect branches. The cloned entry blocks are the ones right
    // after our original entry_block.
    //
    // After cloneInto for task_a, the fused region has:
    //   [entry_block, cloned_a_entry, cloned_a_bb1, ..., cloned_a_bbN]
    // After cloneInto for task_b, it adds:
    //   [..., cloned_b_entry, cloned_b_bb1, ..., cloned_b_bbM]
    //
    // The cloned entry blocks have block args (copies of original entry
    // block args), but these are already mapped by mapping_a/mapping_b
    // to our entry_block args. We need to:
    // 1. Replace uses of cloned entry block args with mapped values
    // 2. Splice ops from cloned entry blocks into our entry block
    // 3. Redirect any branches to cloned entry blocks

    // Finds the cloned entry blocks. They are the blocks whose args were
    // mapped from the original tasks' entry block args.
    // After cloneInto, the mapping contains block mappings too.
    Block *cloned_a_entry = mapping_a.lookupOrNull(
        &task_a.getBody().front());
    Block *cloned_b_entry = mapping_b.lookupOrNull(
        &task_b.getBody().front());
    assert(cloned_a_entry && cloned_b_entry &&
           "cloneInto must map source entry blocks");

    // Helper: merges a cloned entry block into our entry block.
    // Replaces all uses of the cloned entry block's args with the mapped
    // values, then splices all ops into our entry block.
    auto mergeClonedEntry = [&](Block *cloned_entry, IRMapping &mapping,
                                TaskflowTaskOp orig_task) {
      Block &orig_entry = orig_task.getBody().front();
      // Replace all uses of cloned entry block args.
      for (unsigned i = 0; i < cloned_entry->getNumArguments(); ++i) {
        Value cloned_arg = cloned_entry->getArgument(i);
        Value mapped_arg = mapping.lookupOrDefault(orig_entry.getArgument(i));
        cloned_arg.replaceAllUsesWith(mapped_arg);
      }
      // Splice all ops from cloned entry into our entry block (before
      // any terminator of entry_block, if present).
      entry_block->getOperations().splice(
          entry_block->end(), cloned_entry->getOperations());
      // Redirect any branches that target cloned_entry to entry_block.
      cloned_entry->replaceAllUsesWith(entry_block);
      // Erase the now-empty cloned entry block.
      cloned_entry->erase();
    };

    mergeClonedEntry(cloned_a_entry, mapping_a, task_a);
    mergeClonedEntry(cloned_b_entry, mapping_b, task_b);

    // Now the fused region has entry_block (with merged ops from both
    // tasks' entry blocks) plus any non-entry blocks from both tasks.
    // All values are properly mapped through cloneInto's IRMapping.

    // Finds the cloned kernels in the fused region.
    neura::KernelOp cloned_kernel_a, cloned_kernel_b;
    {
      // We can identify them by looking up the original kernels through
      // the mapping. The cloned ops are tracked by the IRMapping.
      neura::KernelOp orig_kernel_a, orig_kernel_b;
      task_a.walk([&](neura::KernelOp k) { orig_kernel_a = k; });
      task_b.walk([&](neura::KernelOp k) { orig_kernel_b = k; });
      assert(orig_kernel_a && orig_kernel_b &&
             "[performFusion] tasks must have neura.kernel ops");

      // Walks the fused task to find all kernels. We match by checking
      // which mapping contains the block within the kernel.
      SmallVector<neura::KernelOp> fused_kernels;
      fused_task.walk([&](neura::KernelOp k) { fused_kernels.push_back(k); });
      assert(fused_kernels.size() == 2 &&
             "[performFusion] expected exactly 2 cloned kernels");

      // Determines which is which. The mapping maps orig blocks to cloned
      // blocks — we can check if a kernel's parent block was mapped from
      // task_a or task_b's blocks.
      for (auto k : fused_kernels) {
        // Check if this kernel's entry block was cloned from kernel_a
        bool is_from_a = false;
        for (Block &orig_blk : orig_kernel_a.getBody()) {
          if (mapping_a.lookupOrNull(&orig_blk) == &k.getBody().front()) {
            is_from_a = true;
            break;
          }
        }
        if (is_from_a) {
          cloned_kernel_a = k;
        } else {
          cloned_kernel_b = k;
        }
      }

      // Fallback: if block mapping didn't work (e.g., cloneInto created
      // fresh blocks), use ordering — first kernel is from task_a.
      if (!cloned_kernel_a || !cloned_kernel_b) {
        cloned_kernel_a = fused_kernels[0];
        cloned_kernel_b = fused_kernels[1];
      }
    }

    // Now merge the two cloned kernels into one fused kernel.
    // Both cloned kernels already have their inputs properly mapped
    // (through cloneInto), so their inputs reference fused task values.

    // Build merged kernel inputs from the cloned kernels' inputs.
    SmallVector<Value> merged_kernel_inputs;
    auto addKernelInputs = [&](neura::KernelOp kernel) {
      for (Value inp : kernel.getInputs()) {
        if (llvm::find(merged_kernel_inputs, inp) ==
            merged_kernel_inputs.end()) {
          merged_kernel_inputs.push_back(inp);
        }
      }
    };
    addKernelInputs(cloned_kernel_a);
    addKernelInputs(cloned_kernel_b);

    // Merged iter_args_init.
    SmallVector<Value> merged_iter_args;
    for (Value v : cloned_kernel_a.getIterArgsInit())
      merged_iter_args.push_back(v);
    for (Value v : cloned_kernel_b.getIterArgsInit())
      merged_iter_args.push_back(v);

    // Merged result types.
    SmallVector<Type> merged_kernel_results;
    for (Type t : cloned_kernel_a.getResultTypes())
      merged_kernel_results.push_back(t);
    for (Type t : cloned_kernel_b.getResultTypes())
      merged_kernel_results.push_back(t);

    // Creates the fused kernel op right before cloned_kernel_a.
    OpBuilder fused_kb(cloned_kernel_a);
    auto fused_kernel = fused_kb.create<neura::KernelOp>(
        task_a.getLoc(), merged_kernel_results, merged_kernel_inputs,
        merged_iter_args,
        /*cgra_id=*/nullptr, /*kernel_name=*/nullptr,
        /*accelerator=*/builder.getStringAttr("neura"));
    fused_kernel->setAttr("dataflow_mode",
                          builder.getStringAttr("predicate"));

    // Creates the kernel entry block.
    Region &fused_kernel_region = fused_kernel.getBody();
    Block *kernel_body = builder.createBlock(&fused_kernel_region);
    for (Value v : merged_kernel_inputs)
      kernel_body->addArgument(v.getType(), task_a.getLoc());
    for (Value v : merged_iter_args)
      kernel_body->addArgument(v.getType(), task_a.getLoc());

    // Build kernel block arg mapping for each source cloned kernel.
    auto buildKernelArgMapping =
        [&](neura::KernelOp kernel, unsigned iter_offset) -> IRMapping {
      IRMapping km;
      Block &src_entry = kernel.getBody().front();
      unsigned src_idx = 0;

      // Map kernel input args.
      for (Value inp : kernel.getInputs()) {
        auto it = llvm::find(merged_kernel_inputs, inp);
        assert(it != merged_kernel_inputs.end());
        unsigned fused_idx = std::distance(merged_kernel_inputs.begin(), it);
        km.map(src_entry.getArgument(src_idx),
               kernel_body->getArgument(fused_idx));
        src_idx++;
      }

      // Map iter_args.
      for (unsigned i = 0; i < kernel.getIterArgsInit().size(); ++i) {
        km.map(src_entry.getArgument(src_idx + i),
               kernel_body->getArgument(
                   merged_kernel_inputs.size() + iter_offset + i));
      }

      return km;
    };

    IRMapping kernel_mapping_a = buildKernelArgMapping(
        cloned_kernel_a, 0);
    IRMapping kernel_mapping_b = buildKernelArgMapping(
        cloned_kernel_b, cloned_kernel_a.getIterArgsInit().size());

    // Clone DFG ops from both cloned kernels into the fused kernel body.
    {
      OpBuilder kb = OpBuilder::atBlockEnd(kernel_body);
      for (auto &op : cloned_kernel_a.getBody().front().getOperations()) {
        if (isa<neura::YieldOp>(&op)) continue;
        kb.clone(op, kernel_mapping_a);
      }
      for (auto &op : cloned_kernel_b.getBody().front().getOperations()) {
        if (isa<neura::YieldOp>(&op)) continue;
        kb.clone(op, kernel_mapping_b);
      }

      // Create the combined neura.yield.
      SmallVector<Value> merged_iter_args_next;
      SmallVector<Value> merged_results;
      if (auto yield_a = dyn_cast<neura::YieldOp>(
              cloned_kernel_a.getBody().front().getTerminator())) {
        for (Value v : yield_a.getIterArgsNext())
          merged_iter_args_next.push_back(
              kernel_mapping_a.lookupOrDefault(v));
        for (Value v : yield_a.getResults())
          merged_results.push_back(kernel_mapping_a.lookupOrDefault(v));
      }
      if (auto yield_b = dyn_cast<neura::YieldOp>(
              cloned_kernel_b.getBody().front().getTerminator())) {
        for (Value v : yield_b.getIterArgsNext())
          merged_iter_args_next.push_back(
              kernel_mapping_b.lookupOrDefault(v));
        for (Value v : yield_b.getResults())
          merged_results.push_back(kernel_mapping_b.lookupOrDefault(v));
      }

      auto fused_yield = kb.create<neura::YieldOp>(
          task_a.getLoc(), merged_iter_args_next, merged_results);
      if (auto yield_a = dyn_cast<neura::YieldOp>(
              cloned_kernel_a.getBody().front().getTerminator())) {
        if (auto attr = yield_a->getAttr("yield_type"))
          fused_yield->setAttr("yield_type", attr);
      }
    }

    // Replaces uses of the cloned kernels' results with the fused kernel's
    // results, then erase the old cloned kernels.
    {
      unsigned result_idx = 0;
      for (unsigned i = 0; i < cloned_kernel_a.getNumResults(); ++i) {
        cloned_kernel_a.getResult(i).replaceAllUsesWith(
            fused_kernel.getResult(result_idx++));
      }
      for (unsigned i = 0; i < cloned_kernel_b.getNumResults(); ++i) {
        cloned_kernel_b.getResult(i).replaceAllUsesWith(
            fused_kernel.getResult(result_idx++));
      }
      cloned_kernel_a.erase();
      cloned_kernel_b.erase();
    }

    // Now handle taskflow.yield ops. The fused task has two cloned yields
    // (one from each original task). We need to merge them into one.
    // Find all taskflow.yield ops in the fused task.
    SmallVector<TaskflowYieldOp> cloned_yields;
    fused_task.walk([&](TaskflowYieldOp y) { cloned_yields.push_back(y); });

    // Builds the merged yield: write outputs + value outputs.
    // For write outputs, map each original write memref to the fused task's
    // block arg at the correct position.
    // For value outputs, we need to collect them from both original yields
    // and combine, using the fused kernel results.
    {
      // We'll replace all existing yields with a single merged yield.
      // For the multi-block case, there may be multiple yield ops
      // (one per control flow path). We need to handle this carefully.
      //
      // Strategy: if there's exactly one yield per original task (the
      // common case), merge them. For multi-yield cases, we replace each
      // yield block with a branch to a new exit block containing the
      // merged yield.

      // Collects write args for the yield.
      SmallVector<Value> yield_writes;
      for (size_t i = 0; i < merged_write_memrefs.size(); ++i) {
        yield_writes.push_back(
            entry_block->getArgument(merged_read_memrefs.size() + i));
      }

      // Value outputs from the fused kernel results.
      SmallVector<Value> yield_values;
      unsigned val_idx = 0;
      for (unsigned i = 0; i < task_a.getValueOutputs().size(); ++i)
        yield_values.push_back(fused_kernel.getResult(val_idx++));
      for (unsigned i = 0; i < task_b.getValueOutputs().size(); ++i)
        yield_values.push_back(fused_kernel.getResult(val_idx++));

      // Erases all cloned yields and create one new yield in an exit block.
      // If the fused task is single-block, just append the yield.
      // If multi-block, create an exit block and redirect.

      // First erase all existing yields.
      for (auto y : cloned_yields) {
        // Replace the yield with a branch to the exit block if we're
        // in a multi-block scenario. For now, just erase them.
        y.erase();
      }

      // Checks if there's a natural "last" block. After erasing yields,
      // some blocks may be empty (no terminator). We need to add the
      // merged yield as the terminator of each such block, or create
      // a common exit block.
      //
      // Simple approach: find all blocks without terminators and add
      // a yield to each one. Since both tasks are independent, both
      // yield blocks should eventually execute.
      // However, for correct semantics, we only need ONE yield.
      // The control flow from task_a and task_b are independent —
      // but they were merged into one region, so both control flows
      // exist. We need to ensure both complete before yielding.
      //
      // For single-block tasks (common case): both sets of ops are in
      // entry_block, we just add the yield at the end.
      //
      // For multi-block tasks: we need to chain the control flows.
      // After task_a's control flow completes (reaches its exit block),
      // branch to task_b's entry. After task_b completes, yield.

      // Finds blocks without terminators (yield was erased).
      SmallVector<Block *> unterminated_blocks;
      for (Block &blk : fused_task.getBody()) {
        if (blk.empty() || !blk.back().hasTrait<OpTrait::IsTerminator>()) {
          unterminated_blocks.push_back(&blk);
        }
      }

      if (unterminated_blocks.empty()) {
        // All blocks are properly terminated (shouldn't happen if yields
        // were erased). Add yield to entry_block as safety fallback.
        OpBuilder tb = OpBuilder::atBlockEnd(entry_block);
        tb.create<TaskflowYieldOp>(fused_task.getLoc(), yield_writes,
                                   yield_values);
      } else {
        // Adds a taskflow.yield to each unterminated block. Since all
        // control flow paths must terminate with a yield, and both
        // tasks' yields produce the same fused outputs (write_memrefs
        // and value outputs from the fused kernel), each unterminated
        // block gets an identical yield.
        for (Block *blk : unterminated_blocks) {
          OpBuilder tb = OpBuilder::atBlockEnd(blk);
          tb.create<TaskflowYieldOp>(fused_task.getLoc(), yield_writes,
                                     yield_values);
        }
      }
    }


    // Step 10: Sets fused attributes for the latency model.
    llvm::errs() << "[performFusion] Fused task IR after creation:\n";
    fused_task.dump();
    // trip_count: max of both, since independent tasks execute concurrently
    // on the shared tile array.
    int64_t fused_trip = std::max(node_a->trip_count, node_b->trip_count);
    fused_task->setAttr("trip_count",
                        OpBuilder(fused_task).getI64IntegerAttr(fused_trip));

    // Profile the fused task to get real ii and steps.
    // The fused kernel contains the combined DFG from both original kernels.
    {
      TaskGraphNode fused_node(/*id=*/0, fused_task);
      fused_node.trip_count = fused_trip;
      profile_fn(&fused_node, fused_task);
      fused_task->setAttr("steps",
                          OpBuilder(fused_task).getI64IntegerAttr(fused_node.steps));
      fused_task->setAttr("compiled_ii",
                          OpBuilder(fused_task).getI64IntegerAttr(fused_node.ii));
    }


    // Step 11: Replaces uses of original tasks' results.
    // Value outputs are ordered: task_a's value outputs first, then task_b's.
    unsigned val_offset_a = 0;
    unsigned val_offset_b = task_a.getValueOutputs().size();
    replaceTaskResults(task_a, fused_task, merged_write_memrefs, val_offset_a);
    replaceTaskResults(task_b, fused_task, merged_write_memrefs, val_offset_b);

    // Step 12: Erases original tasks.
    // Verify no remaining uses before erasing.
    auto verifyNoUses = [](TaskflowTaskOp task, StringRef label) {
      for (Value result : task->getResults()) {
        if (!result.use_empty()) {
          llvm::errs() << "[performFusion] ERROR: " << label
                       << " result #" << result.cast<OpResult>().getResultNumber()
                       << " still has uses:\n";
          for (auto &use : result.getUses()) {
            llvm::errs() << "  used by: ";
            use.getOwner()->print(llvm::errs());
            llvm::errs() << "\n";
          }
        }
      }
    };
    verifyNoUses(task_a, "task_a");
    verifyNoUses(task_b, "task_b");
    task_a.erase();
    task_b.erase();

    return true;
  }

  // Finds the index of a value in a list.
  unsigned findOperandIndex(const SmallVector<Value> &list, Value v) {
    for (unsigned i = 0; i < list.size(); ++i) {
      if (list[i] == v) return i;
    }
    llvm_unreachable("Value not found in operand list");
  }

  // Replaces results of an original task with corresponding results from the
  // fused task. Handles both write outputs (memrefs) and value outputs
  // (reductions, iter_args).
  void replaceTaskResults(TaskflowTaskOp orig_task, TaskflowTaskOp fused_task,
                          const SmallVector<Value> &merged_write_memrefs,
                          unsigned value_output_offset) {
    // Write outputs: maps by matching the original write memref to its
    // position in the merged write memrefs list.
    for (unsigned i = 0; i < orig_task.getWriteOutputs().size(); ++i) {
      Value orig_result = orig_task.getWriteOutputs()[i];
      Value orig_write = orig_task.getWriteMemrefs()[i];
      unsigned fused_idx = findOperandIndex(merged_write_memrefs, orig_write);
      orig_result.replaceAllUsesWith(fused_task.getWriteOutputs()[fused_idx]);
    }
    // Value outputs: each original task's value_output[i] maps to
    // fused_task.getValueOutputs()[value_output_offset + i].
    for (unsigned i = 0; i < orig_task.getValueOutputs().size(); ++i) {
      Value orig_val = orig_task.getValueOutputs()[i];
      orig_val.replaceAllUsesWith(
          fused_task.getValueOutputs()[value_output_offset + i]);
    }
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct ResourceAwareTaskOptimizationPass
    : public PassWrapper<ResourceAwareTaskOptimizationPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ResourceAwareTaskOptimizationPass)

  ResourceAwareTaskOptimizationPass() = default;
  ResourceAwareTaskOptimizationPass(const ResourceAwareTaskOptimizationPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "resource-aware-task-optimization";
  }

  StringRef getDescription() const override {
    return "Balances pipeline latency and fuses independent tasks for CGRA "
           "utilization";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<neura::NeuraDialect>();
    registry.insert<taskflow::TaskflowDialect>();
  }

  // Estimation mode for profiling task II / steps.
  //   "compiled" (default): runs the full Neura lowering + mapping pipeline
  //       to obtain accurate compiled_ii and steps from MapToAcceleratorPass.
  //   "analytical": uses only ResMII / RecMII analytical estimates without
  //       running the mapper.  Much faster but less accurate — useful for
  //       rapid design-space exploration or when the mapper is unavailable.
  Option<std::string> estimationMode{
      *this, "estimation-mode",
      llvm::cl::desc(
          "Profiling estimation mode: 'compiled' (default) runs the full "
          "Neura lowering + mapping pipeline for accurate II/steps; "
          "'analytical' uses only ResMII/RecMII analytical estimates "
          "(faster but less accurate)."),
      llvm::cl::init("compiled")};

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    bool use_analytical = (estimationMode.getValue() == "analytical");

    llvm::errs() << "=== ResourceAwareTaskOptimization on "
                 << func.getName()
                 << " (estimation-mode=" << estimationMode.getValue()
                 << ") ===\n";

    llvm::errs() << "[RESOPT] Input IR at start of runOnOperation:\n";
    func.dump();

    constexpr int kMaxOuterIterations = 10;

    for (int outer = 0; outer < kMaxOuterIterations; ++outer) {
      // Rebuilds graph from current IR state.
      TaskDependencyGraph graph;
      graph.build(func, use_analytical);

      if (graph.nodes.empty()) {
        return;
      }

      int num_tasks = graph.nodes.size();

      // Asserts that initial tasks fit in the grid.
      assert(num_tasks <= kTotalCGRAs &&
             "Number of tasks exceeds 4x4 CGRA grid capacity! "
             "Reduce task count via streaming fusion or increase grid size.");

      llvm::errs() << "[ResourceAware] Iteration " << outer << ": "
                   << num_tasks << " tasks\n";
      for (auto &node : graph.nodes) {
        llvm::errs() << "  Task " << node->id << " ("
                     << node->op.getTaskName() << "): trip_count="
                     << node->trip_count << ", cgra_count=" << node->cgra_count
                     << ", est_latency=" << node->estimatedLatency() << "\n";
      }

      // Phase 1: Utilization Fusion.
      // Fuse independent tasks to free up CGRA budget for balance.
      UtilizationFuser fuser;
      // Expose TaskDependencyGraph::profileTask to UtilizationFuser via a
      // lambda so fused tasks get real profiling.  In analytical mode, the
      // mapper is skipped entirely (only ResMII/RecMII estimates are used).
      auto profile_fn = [&graph, use_analytical](TaskGraphNode *node,
                                                  TaskflowTaskOp task) {
        graph.profileTaskPublic(node, task, /*skip_mapper=*/use_analytical);
      };
      bool fuse_changed = fuser.fuse(func, graph, profile_fn);

      llvm::errs() << "[ResourceAware] After fusion: total_cgras="
                   << graph.getTotalAllocatedCGRAs() << "\n";

      // Rebuild graph after fusion (tasks may have been erased/created).
      if (fuse_changed) {
        graph = TaskDependencyGraph();
        graph.build(func, use_analytical);
      }

      // Phase 2: Latency-Aware Pipeline Balance.
      // Balance probes always use analytical-only profiling (skip_mapper=true)
      // regardless of estimation-mode, because the mapper may backtrack
      // indefinitely on speculative larger tile arrays.  ResMII/RecMII
      // estimates are sufficient to decide if adding a CGRA reduces the
      // bottleneck's II.
      auto balance_profile_fn = [&graph](TaskGraphNode *node,
                                         TaskflowTaskOp task) {
        graph.profileTaskPublic(node, task, /*skip_mapper=*/true);
      };
      PipelineBalancer balancer;
      bool balance_changed = balancer.balance(graph, balance_profile_fn);

      // Writes cgra_count, ii, steps, and trip_count back to IR during
      // iterations so that the next iteration's graph.build() reads them
      // and skips expensive re-profiling for unchanged tasks.
      if (balance_changed || fuse_changed) {
        for (auto &node : graph.nodes) {
          OpBuilder b(node->op);
          node->op->setAttr(
              "cgra_count", b.getI32IntegerAttr(node->cgra_count));
          if (node->ii != kUnprofiled) {
            node->op->setAttr("compiled_ii", b.getI32IntegerAttr(node->ii));
          }
          if (node->steps != kUnprofiled) {
            node->op->setAttr("steps", b.getI32IntegerAttr(node->steps));
          }
          if (node->trip_count > 0) {
            node->op->setAttr("trip_count",
                              b.getI32IntegerAttr(node->trip_count));
          }
          if (balance_changed && node->cgra_count > 1) {
            llvm::errs() << "  [Balance] " << node->op.getTaskName()
                         << " -> cgra_count=" << node->cgra_count
                         << ", est_latency=" << node->estimatedLatency()
                         << "\n";
          }
        }
      }

      llvm::errs() << "[ResourceAware] After balance: total_cgras="
                   << graph.getTotalAllocatedCGRAs() << "\n";

      if (!balance_changed && !fuse_changed) {
        // Converged — writes ALL attributes (cgra_count, ii, steps) to IR
        // for every task. Non-fused tasks only got cgra_count written during
        // intermediate iterations; ii, steps, and trip_count live only in the
        // graph node and must be persisted here.
        for (auto &node : graph.nodes) {
          OpBuilder b(node->op);
          node->shape = pickBestShape(node->cgra_count);
          node->op->setAttr("cgra_count",
                            b.getI32IntegerAttr(node->cgra_count));
          node->op->setAttr("compiled_ii",
                            b.getI32IntegerAttr(node->ii));
          node->op->setAttr("steps",
                            b.getI32IntegerAttr(node->steps));
          node->op->setAttr("trip_count",
                            b.getI32IntegerAttr(node->trip_count));
          // Write tile_shape attribute: simple "NxM" bounding-box string.
          // The detailed occupancy diagram is printed in the summary below.
          std::string shape_str = node->shape.irAttr();
          node->op->setAttr("tile_shape", b.getStringAttr(shape_str));
        }
        break;
      }
    }

    // Performs final validation and tile occupation summary with visual 4x4 grid.
    {
      TaskDependencyGraph final_graph;
      final_graph.build(func, use_analytical);
      int final_total = final_graph.getTotalAllocatedCGRAs();

      // Assign each task a single character label for the combined grid.
      // Tasks are labelled '0','1','2',... ; free cells shown as '.'.
      // grid[row][col] == -1 means free.
      std::vector<std::vector<int>> combined_grid(
          kCgraGridRows, std::vector<int>(kCgraGridCols, -1));

      // Pack tasks onto the grid left-to-right, top-to-bottom.
      int next_col = 0, next_row = 0;
      int task_idx = 0;

      llvm::errs() << "\n=== Tile Occupation Summary (4x" << kCgraGridCols
                   << " CGRA Grid) ===\n";

      for (auto &node : final_graph.nodes) {
        auto shape = pickBestShape(node->cgra_count);
        int tile_rows = shape.rows * neura::getArchitecture().getPerCgraRows();
        int tile_cols = shape.cols * neura::getArchitecture().getPerCgraColumns();

        // Per-task grid (shape.rows x shape.cols bbox, filled up to cgra_count).
        llvm::errs() << "\n  [" << task_idx << "] " << node->op.getTaskName()
                     << "  cgra_count=" << node->cgra_count
                     << "  shape=" << shape.describe(node->cgra_count)
                     << "  tile_array=" << tile_rows << "x" << tile_cols
                     << "  ii=" << node->ii
                     << "  steps=" << node->steps
                     << "  trip_count=" << node->trip_count << "\n";

        // Draw a per-task bounding-box grid (shape.rows x shape.cols).
        int remaining = node->cgra_count;
        llvm::errs() << "      +" ;
        for (int c = 0; c < shape.cols; ++c) llvm::errs() << "---+";
        llvm::errs() << "\n";
        for (int r = 0; r < shape.rows; ++r) {
          llvm::errs() << "      |";
          for (int c = 0; c < shape.cols; ++c) {
            if (remaining > 0) {
              llvm::errs() << " # |";
              --remaining;
            } else {
              llvm::errs() << "   |";
            }
          }
          llvm::errs() << "\n";
          llvm::errs() << "      +";
          for (int c = 0; c < shape.cols; ++c) llvm::errs() << "---+";
          llvm::errs() << "\n";
        }

        // Place onto combined grid (pack sequentially).
        int placed = 0;
        for (int r = next_row; r < kCgraGridRows && placed < node->cgra_count; ++r) {
          for (int c = (r == next_row ? next_col : 0);
               c < kCgraGridCols && placed < node->cgra_count; ++c) {
            combined_grid[r][c] = task_idx;
            next_row = r;
            next_col = c + 1;
            if (next_col >= kCgraGridCols) { next_col = 0; next_row = r + 1; }
            ++placed;
          }
        }
        ++task_idx;
      }

      // Print combined 4xN grid.
      llvm::errs() << "\n  Combined 4x" << kCgraGridCols << " Grid"
                   << " (" << final_total << "/" << kTotalCGRAs << " used):\n";
      llvm::errs() << "  +";
      for (int c = 0; c < kCgraGridCols; ++c) llvm::errs() << "---+";
      llvm::errs() << "\n";
      for (int r = 0; r < kCgraGridRows; ++r) {
        llvm::errs() << "  |";
        for (int c = 0; c < kCgraGridCols; ++c) {
          int t = combined_grid[r][c];
          if (t < 0)
            llvm::errs() << " . |";
          else
            llvm::errs() << " " << (char)('0' + t) << " |";
        }
        llvm::errs() << "\n";
        llvm::errs() << "  +";
        for (int c = 0; c < kCgraGridCols; ++c) llvm::errs() << "---+";
        llvm::errs() << "\n";
      }
      llvm::errs() << "  (" << (kTotalCGRAs - final_total) << " free)\n";
      llvm::errs() << "================================================\n";

      llvm::errs() << "[ResourceAware] Final: " << final_graph.nodes.size()
                   << " tasks, " << final_total << " CGRAs\n";
      assert(final_total <= kTotalCGRAs &&
             "Total CGRA allocation exceeds 4x4 grid after optimization!");
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass>
mlir::taskflow::createResourceAwareTaskOptimizationPass() {
  return std::make_unique<ResourceAwareTaskOptimizationPass>();
}
