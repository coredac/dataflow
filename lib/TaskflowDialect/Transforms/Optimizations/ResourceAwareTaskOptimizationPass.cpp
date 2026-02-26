//===- ResourceAwareTaskOptimizationPass.cpp - Pipeline Balance & Fusion --===//
// This pass performs two-phase optimization on the task graph:
// 1. Utilization Fusion: merges independent (no-edge) tasks, selecting pairs
//    that minimize |trip_count_a - trip_count_b| for balanced utilization.
// 2. Pipeline Balance: allocates extra CGRAs to critical-path bottleneck tasks.
//    More CGRAs combine tile arrays into larger arrays for mapping, potentially
//    lowering compiled_ii.  Latency model: II * (trip_count - 1) + steps.
//
// Targets a 4x4 CGRA grid (16 CGRAs total). Validates shapes (rect, L, T, offset).
// Compiled_ii must come from the downstream pipeline (asserts on failure).
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "Conversion/ConversionPasses.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraAttributes.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
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

constexpr int kGridRows = 4;
constexpr int kGridCols = 4;
constexpr int kTotalCGRAs = kGridRows * kGridCols; // 16
constexpr int kMaxBalanceIterations = 100;
constexpr int kPerCgraRows = 4;  // Tile rows per single CGRA.
constexpr int kPerCgraCols = 4;  // Tile cols per single CGRA.

// Sentinel value: 0 means "not yet profiled". After profileTask() runs,
// both steps and ii MUST be > 0. An assert fires if profiling fails.
constexpr int64_t kUnprofiled = 0;
 
//===----------------------------------------------------------------------===//
// CGRA Shape Utilities
//===----------------------------------------------------------------------===//

/// Represents a tile shape on the 4×4 CGRA grid.
/// For rectangular shapes, rows × cols = cgra_count.
/// Non-rectangular shapes (L, T, diagonal offset) are also valid on the 4×4
/// grid but are represented by their bounding box with a note on layout.
struct CGRAShape {
  int rows;  // Number of CGRA rows in the bounding box.
  int cols;  // Number of CGRA columns in the bounding box.
  bool rectangular;  // True if this is a perfect rectangle.
  int area() const { return rows * cols; }

  /// Returns a human-readable description for log messages only (not IR).
  std::string describe(int cgra_count) const {
    std::string s = std::to_string(rows) + "x" + std::to_string(cols);
    if (!rectangular) {
      s += "(non-rect, " + std::to_string(cgra_count) + " CGRAs in " +
           std::to_string(rows) + "x" + std::to_string(cols) + " bbox)";
    }
    return s;
  }

  /// Returns the simple "NxM" string written into the IR tile_shape attribute.
  std::string irAttr() const {
    return std::to_string(rows) + "x" + std::to_string(cols);
  }
};

/// Returns all valid rectangular shapes for `cgra_count` CGRAs.
static SmallVector<CGRAShape> getRectangularShapes(int cgra_count) {
  SmallVector<CGRAShape> shapes;
  for (int r = 1; r <= kGridRows; ++r) {
    for (int c = 1; c <= kGridCols; ++c) {
      if (r * c == cgra_count) {
        shapes.push_back({r, c, /*rectangular=*/true});
      }
    }
  }
  return shapes;
}

/// Returns true if `cgra_count` CGRAs can fit on the 4×4 grid.
/// On a 4×4 grid (16 cells), any count 1..16 can always be arranged as
/// a connected shape (rectangle, L-shape, T-shape, diagonal offset, etc.).
static bool canFitOnGrid(int cgra_count) {
  return cgra_count >= 1 && cgra_count <= kTotalCGRAs;
}

/// Picks the best shape for display/profiling.
/// Prefers rectangular shapes (most square-ish). If no rectangle exists,
/// returns a non-rectangular bounding-box representation.
static CGRAShape pickBestShape(int cgra_count) {
  auto rect_shapes = getRectangularShapes(cgra_count);
  if (!rect_shapes.empty()) {
    return *std::min_element(rect_shapes.begin(), rect_shapes.end(),
        [](const CGRAShape &a, const CGRAShape &b) {
          return std::abs(a.rows - a.cols) < std::abs(b.rows - b.cols);
        });
  }
  // Non-rectangular: find smallest bounding box that contains cgra_count cells.
  CGRAShape best = {kGridRows, kGridCols, false};
  for (int r = 1; r <= kGridRows; ++r) {
    for (int c = 1; c <= kGridCols; ++c) {
      if (r * c >= cgra_count && r * c < best.area()) {
        best = {r, c, false};
      }
    }
  }
  return best;
}

/// Prints all valid shape options for a given cgra_count.
/// (Kept for optional debug use; not called in the main path.).
[[maybe_unused]] static void printShapeOptions(int cgra_count) {
  auto rect = getRectangularShapes(cgra_count);
  llvm::errs() << "    Valid shapes for " << cgra_count << " CGRAs: ";
  if (!rect.empty()) {
    for (size_t i = 0; i < rect.size(); ++i) {
      if (i > 0) llvm::errs() << ", ";
      llvm::errs() << rect[i].rows << "x" << rect[i].cols << "(rect)";
    }
  }
  if (rect.empty() || cgra_count > 4) {
    if (!rect.empty()) llvm::errs() << ", ";
    llvm::errs() << "L/T/offset shapes also valid";
  }
  llvm::errs() << "\n";
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
  CGRAShape shape = {1, 1, true};

  // Dependency edges (both SSA and memory).
  SmallVector<TaskGraphNode *> predecessors;
  SmallVector<TaskGraphNode *> successors;

  TaskGraphNode(size_t id, TaskflowTaskOp op) : id(id), op(op) {}

  /// Returns estimated task latency using the pipelined execution model:
  ///   latency = II * (trip_count - 1) + steps.
  int64_t estimatedLatency() const {
    return ii * (trip_count - 1) + steps;
  }
};

class TaskDependencyGraph {
public:
  SmallVector<std::unique_ptr<TaskGraphNode>> nodes;
  DenseMap<Operation *, TaskGraphNode *> op_to_node;

  void build(func::FuncOp func) {
    // 1. Creates TaskGraphNodes.
    size_t task_id = 0;
    func.walk([&](TaskflowTaskOp task) {
      auto node = std::make_unique<TaskGraphNode>(task_id++, task);
      
      // If the task already has profiling attributes (e.g., from fusion),
      // skip expensive speculative lowering and use those directly.
      bool has_precomputed = task->hasAttr("compiled_ii") && task->hasAttr("steps");
      if (!has_precomputed) {
        // Speculative lowering to Neura to get real metrics.
        profileTask(node.get(), task);
      }

      // Reads existing trip_count attribute if set by fusion.
      if (auto attr = task->getAttrOfType<IntegerAttr>("trip_count")) {
        node->trip_count = attr.getInt();
      } else {
        node->trip_count = computeTripCount(task);
      }
      
      // Override with explicit attributes if present (e.g. from manual tuning).
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
          if (auto *producer = op_to_node[producer_op]) {
            addEdge(producer, consumer.get());
          }
        }
      }
    }

    // 3. Builds memory edges (RAW, WAR, WAW).
    DenseMap<Value, SmallVector<TaskGraphNode *>> memref_writers;
    DenseMap<Value, SmallVector<TaskGraphNode *>> memref_readers;
    for (auto &node : nodes) {
      for (Value memref : node->op.getOriginalWriteMemrefs()) {
        memref_writers[memref].push_back(node.get());
      }
      for (Value memref : node->op.getOriginalReadMemrefs()) {
        memref_readers[memref].push_back(node.get());
      }
    }
    // RAW edges: writer -> reader.
    for (auto &node : nodes) {
      for (Value memref : node->op.getOriginalReadMemrefs()) {
        if (!memref_writers.count(memref)) continue;
        for (auto *writer : memref_writers[memref]) {
          if (writer != node.get() &&
              writer->op->isBeforeInBlock(node->op.getOperation())) {
            addEdge(writer, node.get());
          }
        }
      }
    }
    // WAR edges: reader -> writer (anti-dependency, preserves read-before-write).
    for (auto &[memref, writers] : memref_writers) {
      if (!memref_readers.count(memref)) continue;
      for (auto *reader : memref_readers[memref]) {
        for (auto *writer : writers) {
          if (reader != writer &&
              reader->op->isBeforeInBlock(writer->op.getOperation())) {
            addEdge(reader, writer);
          }
        }
      }
    }
    // WAW edges: earlier writer -> later writer (preserves write order).
    for (auto &[memref, writers] : memref_writers) {
      for (size_t i = 0; i < writers.size(); ++i) {
        for (size_t j = i + 1; j < writers.size(); ++j) {
          auto *a = writers[i];
          auto *b = writers[j];
          if (a->op->isBeforeInBlock(b->op.getOperation())) {
            addEdge(a, b);
          } else {
            addEdge(b, a);
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

  /// Returns true if there is any (direct or transitive) edge between a and b.
  bool hasPath(TaskGraphNode *from, TaskGraphNode *to) const {
    if (from == to) return true;
    DenseSet<TaskGraphNode *> visited;
    SmallVector<TaskGraphNode *> worklist;
    worklist.push_back(from);
    while (!worklist.empty()) {
      auto *current = worklist.pop_back_val();
      if (current == to) return true;
      if (!visited.insert(current).second) continue;
      for (auto *succ : current->successors) {
        worklist.push_back(succ);
      }
    }
    return false;
  }

  /// Returns true if a and b are completely independent (no path in either
  /// direction).
  bool areIndependent(TaskGraphNode *a, TaskGraphNode *b) const {
    return !hasPath(a, b) && !hasPath(b, a);
  }

  /// Returns total CGRAs allocated.
  int getTotalAllocatedCGRAs() const {
    int total = 0;
    for (auto &node : nodes) {
      total += node->cgra_count;
    }
    return total;
  }

  /// Public wrapper for profileTask — used by UtilizationFuser to re-profile
  /// fused tasks with the real downstream Neura pipeline.
  /// When skip_mapper=true, only ResMII/RecMII analytical estimates are used
  /// (no MapToAcceleratorPass). This is safe for speculative balance checks
  /// where the mapper may backtrack indefinitely on larger tile arrays.
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

  /// Performs speculative lowering of a single TaskflowTaskOp through the
  /// real downstream pipeline to extract compiled_ii and steps.
  ///
  /// Two-phase approach (inspired by PR #251 FuseKernelPass):
  ///   Phase 1: Taskflow->Neura conversion on cloned parent function
  ///     construct-hyperblock -> classify-counters -> convert-taskflow-to-neura
  ///     This produces neura.kernel ops (IsolatedFromAbove).
  ///
  ///   Phase 2: Clone each kernel body into a standalone func::FuncOp with
  ///     accelerator="neura" attribute, then run the full Neura lowering +
  ///     mapping pipeline to get real compiled_ii from MapToAcceleratorPass.
  /// ASSERTS if any phase fails — compiled_ii must come from the downstream
  /// pipeline, never from a silent fallback.
  ///
  /// skip_mapper: when true, skip MapToAcceleratorPass and use only
  ///   ResMII/RecMII analytical estimates. Safe for speculative balance probes
  ///   where the mapper may loop indefinitely on large tile arrays.
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
    // Split-Profiling for Multi-Body Tasks (e.g. after utilization fusion)
    // ================================================================
    // ConvertTaskflowToNeura asserts hyperblock_count==1. If a task body
    // contains more than one top-level op (loop nest), we split it back into
    // individual single-loop tasks, profile each independently, and take
    // max(ii) / sum(steps) as the combined metrics.
    SmallVector<Operation *> body_ops;
    Block &task_body = task.getBody().front();
    for (auto &op : task_body.getOperations()) {
      if (!isa<TaskflowYieldOp>(op))
        body_ops.push_back(&op);
    }

    if (body_ops.size() > 1) {
      int64_t total_ii = 1;
      int64_t total_steps = 0;

      for (Operation *loop_op : body_ops) {
        OpBuilder tmp_builder(task.getOperation());
        auto tmp_task = tmp_builder.create<TaskflowTaskOp>(
            task.getLoc(),
            task.getWriteOutputs().getTypes(),
            task.getValueOutputs().getTypes(),
            task.getReadMemrefs(),
            task.getWriteMemrefs(),
            task.getValueInputs(),
            (task.getTaskName().str() + "__split_profile__").c_str(),
            task.getOriginalReadMemrefs(),
            task.getOriginalWriteMemrefs());

        Block *tmp_body = new Block();
        tmp_task.getBody().push_back(tmp_body);
        for (BlockArgument arg : task_body.getArguments())
          tmp_body->addArgument(arg.getType(), arg.getLoc());

        OpBuilder body_builder = OpBuilder::atBlockEnd(tmp_body);
        IRMapping arg_map;
        for (auto [orig, repl] : llvm::zip(task_body.getArguments(),
                                           tmp_body->getArguments()))
          arg_map.map(orig, repl);

        body_builder.clone(*loop_op, arg_map);

        SmallVector<Value> yield_writes;
        for (size_t i = 0; i < task.getWriteMemrefs().size(); ++i)
          yield_writes.push_back(
              tmp_body->getArgument(task.getReadMemrefs().size() + i));
        SmallVector<Value> yield_vals;
        body_builder.create<TaskflowYieldOp>(task.getLoc(), yield_writes,
                                             yield_vals);

        // Inherit cgra_count so the split task profiles with the same tile array.
        tmp_task->setAttr("cgra_count",
                          tmp_builder.getI32IntegerAttr(node->cgra_count));

        TaskGraphNode tmp_node(/*id=*/0, tmp_task);
        tmp_node.cgra_count = node->cgra_count;
        tmp_node.shape = node->shape;
        this->profileTask(&tmp_node, tmp_task, skip_mapper);

        total_ii = std::max(total_ii, tmp_node.ii);
        total_steps += tmp_node.steps;

        tmp_task.erase();
      }

      node->ii = total_ii;
      node->steps = std::max(total_steps, (int64_t)1);
      llvm::errs() << "[profileTask] split-profile result for "
                   << task.getTaskName() << ": compiled_ii=" << node->ii
                   << ", steps=" << node->steps << "\n";
      return;
    }

    // ================================================================
    // Phase 1: Taskflow -> Neura conversion (get neura.kernel ops)
    // ================================================================
    // We clone the entire parent function but then strip all tasks EXCEPT the
    // one being profiled. This is critical because utilization-fused tasks
    // contain multiple hyperblocks, which causes ConvertTaskflowToNeura to
    // assert (hyperblock_count == 1). By keeping only the target task, Phase 1
    // processes just the single task we care about.
    auto phase1_module = ModuleOp::create(loc);
    {
      OpBuilder mod_builder(ctx);
      mod_builder.setInsertionPointToStart(phase1_module.getBody());
      IRMapping clone_mapping;
      mod_builder.clone(*parent_func, clone_mapping);

      // Finds the cloned copy of the target task and erase all others.
      Operation *cloned_target = clone_mapping.lookupOrNull(task.getOperation());
      func::FuncOp cloned_func = nullptr;
      phase1_module.walk([&](func::FuncOp f) { cloned_func = f; });
      if (cloned_func) {
        SmallVector<TaskflowTaskOp> to_erase;
        cloned_func.walk([&](TaskflowTaskOp t) {
          if (t.getOperation() != cloned_target) {
            to_erase.push_back(t);
          }
        });
        for (auto t : to_erase) {
          // Replaces all results with undef-like values so uses don't dangle.
          for (OpResult res : t->getResults()) {
            /// Creates a placeholder value so uses don't dangle.
            /// Uses UnrealizedConversionCastOp as a universal placeholder that
            /// works for any type (memref, index, integer, float, etc.)
            /// without needing type-specific logic. Verifier is disabled.
            OpBuilder b(t);
            Value placeholder =
                b.create<UnrealizedConversionCastOp>(t.getLoc(),
                                                     res.getType(),
                                                     ValueRange{})
                    .getResult(0);
            res.replaceAllUsesWith(placeholder);
          }
          t.erase();
        }
      }
    }

    {
      PassManager pm(ctx);
      pm.enableVerifier(false);
      pm.addNestedPass<func::FuncOp>(
          taskflow::createConstructHyperblockFromTaskPass());
      pm.addPass(taskflow::createClassifyCountersPass());
      pm.addPass(mlir::createConvertTaskflowToNeuraPass());

      if (failed(pm.run(phase1_module))) {
        phase1_module.erase();
        assert(false &&
               "[profileTask] FATAL: Phase 1 (Taskflow->Neura) failed. "
               "compiled_ii must come from downstream pipeline.");
        return;
      }
    }

    // ================================================================
    // Phase 2: For each kernel, clone body -> func -> run Neura pipeline
    // ================================================================
    // Collects all neura.kernel ops created by Phase 1.
    SmallVector<neura::KernelOp> kernels;
    phase1_module.walk([&](neura::KernelOp k) { kernels.push_back(k); });

    if (kernels.empty()) {
      phase1_module.erase();
      assert(false &&
             "[profileTask] FATAL: No kernels found after Phase 1. "
             "compiled_ii must come from downstream pipeline.");
      return;
    }

    int best_compiled_ii = 0;
    int best_cp_depth = 1;

    // Computes tile dimensions for the target CGRA shape.
    // For rectangular shapes: x_tiles = cols * per_cgra_cols,
    //                         y_tiles = rows * per_cgra_rows.
    // For non-rectangular (L/T): enumerate the exact tiles occupied.
    int x_tiles = node->shape.cols * neura::getArchitecture().getPerCgraColumns();
    int y_tiles = node->shape.rows * neura::getArchitecture().getPerCgraRows();
    std::string valid_tiles;
    if (!node->shape.rectangular) {
      // Builds an explicit tile list for the cgra_count CGRAs that actually fit.
      int cgras_added = 0;
      llvm::raw_string_ostream os(valid_tiles);
      for (int r = 0; r < node->shape.rows && cgras_added < node->cgra_count; ++r) {
        for (int c = 0; c < node->shape.cols && cgras_added < node->cgra_count; ++c) {
          for (int tr = 0; tr < neura::getArchitecture().getPerCgraRows(); ++tr) {
            for (int tc = 0; tc < neura::getArchitecture().getPerCgraColumns(); ++tc) {
              if (!os.str().empty()) os << ",";
              os << (c * neura::getArchitecture().getPerCgraColumns() + tc)
                 << "_"
                 << (r * neura::getArchitecture().getPerCgraRows() + tr);
            }
          }
          ++cgras_added;
        }
      }
    }

    for (neura::KernelOp kernel : kernels) {
      // Creates a fresh module with a standalone func containing the kernel
      // body. All downstream Neura passes walk func::FuncOp with
      // accelerator="neura", so we package the kernel body as such.
      auto phase2_module = ModuleOp::create(loc);
      int compiled_ii = 0;
      int cp_depth = 1;

      if (succeeded(
              runNeuraPipelineOnKernel(ctx, kernel, phase2_module,
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
                 << ", steps=" << node->steps << "\n";

    phase1_module.erase();
  }

  /// Clones a neura.kernel body into a standalone func::FuncOp inside
  /// dst_module, then runs the full Neura lowering + mapping pipeline.
  /// Returns success if MapToAccelerator ran and produced compiled_ii.
  ///
  /// x_tiles / y_tiles: total tile dimensions of the target CGRA array.
  ///   These are passed to MapToAcceleratorPass so it maps onto the correct
  ///   multi-CGRA tile grid rather than the default 1-CGRA singleton.
  /// valid_tiles: explicit comma-separated tile list for non-rectangular shapes.
  ///   Empty string means "use the full x_tiles × y_tiles rectangle".
  /// skip_mapper: when true, skip MapToAcceleratorPass entirely and rely only
  ///   on ResMII/RecMII analytical estimates. Used for speculative balance
  ///   probes to prevent infinite mapper backtracking on larger tile arrays.
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

    // Build function signature: all kernel inputs + iter_args as arguments.
    Region &kernel_body = kernel.getBody();
    if (kernel_body.empty())
      return failure();

    Block &entry = kernel_body.front();
    SmallVector<Type> arg_types;
    for (BlockArgument arg : entry.getArguments())
      arg_types.push_back(arg.getType());

    /// Result types from the kernel op.
    SmallVector<Type> result_types(kernel.getResultTypes());

    auto func_type = builder.getFunctionType(arg_types, result_types);
    auto wrapper_func = builder.create<func::FuncOp>(
        loc, "__speculative_kernel__", func_type);

    // Tag as neura accelerator — all downstream passes check this.
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

    /// Run the full Neura lowering + dataflow pipeline.
    /// Pipeline order follows the reference tests in
    /// test/multi-cgra/kernel_mapping/ (fir, relu, loop-in-kernel).
    PassManager pm(ctx);
    pm.enableVerifier(false);

    // Standard MLIR lowering: affine -> scf -> cf -> llvm.
    // Required because kernel body from Phase 1 may still contain scf.for,
    // affine.for, etc. These must become cf/llvm ops before Neura lowering.
    pm.addPass(mlir::createLowerAffinePass());        // affine -> scf
    pm.addPass(mlir::createConvertSCFToCFPass());      // scf -> cf
    pm.addPass(mlir::createConvertControlFlowToLLVMPass()); // cf -> llvm

    // Neura lowering passes (handle affine/arith/memref/llvm -> neura).
    pm.addPass(neura::createAssignAcceleratorPass());
    pm.addPass(mlir::createLowerMemRefToNeuraPass());
    pm.addPass(mlir::createLowerArithToNeuraPass());
    pm.addPass(mlir::createLowerBuiltinToNeuraPass());
    pm.addPass(mlir::createLowerLlvmToNeuraPass());

    // Neura canonicalization + optimization (production pipeline order from
    // test/multi-cgra/kernel_mapping/ reference tests).
    pm.addPass(neura::createPromoteInputArgToConstPass());

    // FoldConstantPass: skipped for speculative profiling.
    // pm.addPass(neura::createFoldConstantPass());
    pm.addPass(neura::createCanonicalizeCastPass());

    pm.addPass(neura::createCanonicalizeReturnPass());
    pm.addPass(neura::createCanonicalizeLiveInPass());
    pm.addPass(neura::createLeveragePredicatedValuePass());
    pm.addPass(neura::createTransformCtrlToDataFlowPass());
    // Pm.addPass(neura::createFoldConstantPass());

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
    // Falls back to the global singleton if tile dims are not specified.
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
    // Guards (Option C — safe default to prevent backtracking timeout):
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
      /// Runs MapToAcceleratorPass in a fresh pass manager on the already-lowered
      /// dst_module (pre-mapper pipeline already ran above).
      /// Pass the correct tile dimensions so the mapper uses the right array.
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
        // Read the true compiled_ii from mapping_info (overrides ResMII/RecMII).
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

    /// Falls back already computed via ResMII/RecMII above; nothing more to do.
    return success();
  }


  /// Extracts metrics from partially-lowered Neura IR when the full pipeline
  /// fails. Uses ResMII/RecMII analysis and critical path depth on whatever
  /// Neura ops were successfully created.
  ///
  /// x_tiles / y_tiles: if > 0, use a custom architecture sized to this tile
  ///   array so that ResMII reflects the real resource pool for multi-CGRA
  ///   shapes. Falls back to the global singleton (1-CGRA) otherwise.
  void extractMetricsFromPartialIR(ModuleOp tmp_module,
                                   int &out_ii, int &out_cp_depth,
                                   int x_tiles = 0, int y_tiles = 0) {
    // Builds architecture: use custom tile dimensions if provided.
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

  /// Computes total trip count for a task.
  ///
  /// Walks the task body and for each top-level affine.for, computes the
  /// product of the entire nested loop structure (perfectly-nested multiply).
  /// For multiple sequential top-level loops, sums their individual products
  /// (they execute sequentially, not as a combined iteration space).
  ///
  /// Examples:
  ///   for i=0..10 { for j=0..20 { } }  → 10 * 20 = 200
  ///   for i=0..10 { }; for j=0..5 { }  → 10 + 5 = 15
  static int64_t computeTripCount(TaskflowTaskOp task) {
    int64_t total = 0;
    // Only visit top-level affine.for ops in the task body (not nested ones).
    Block &body = task.getBody().front();
    for (Operation &op : body.getOperations()) {
      if (auto for_op = dyn_cast<affine::AffineForOp>(op)) {
        total += computeNestedTripCount(for_op);
      }
    }
    // If no affine.for found, default to 1.
    return (total > 0) ? total : 1;
  }

  /// Recursively computes the trip count of a nested loop structure.
  /// Multiplies the current loop's trip count by the maximum nested sub-loop
  /// trip count (for perfectly nested structures, this is the product of all
  /// loop bounds).
  static int64_t computeNestedTripCount(affine::AffineForOp for_op) {
    int64_t this_trip = 1;
    if (for_op.hasConstantBounds()) {
      int64_t lb = for_op.getConstantLowerBound();
      int64_t ub = for_op.getConstantUpperBound();
      int64_t step = for_op.getStepAsInt();
      int64_t count = (ub - lb + step - 1) / step;
      if (count > 0)
        this_trip = count;
    }
    // Recurse into nested loops: sum of all direct-child loop trip counts
    // (for sequential inner loops), then multiply by this loop's trip.
    int64_t inner_total = 0;
    for (Operation &inner_op : for_op.getBody()->getOperations()) {
      if (auto inner_for = dyn_cast<affine::AffineForOp>(inner_op)) {
        inner_total += computeNestedTripCount(inner_for);
      }
    }
    // If no inner loops, trip count is just this loop's count.
    // If inner loops exist, multiply: this_trip * inner_total.
    return (inner_total > 0) ? this_trip * inner_total : this_trip;
  }

};

//===----------------------------------------------------------------------===//
// Pipeline Balancer
//===----------------------------------------------------------------------===//
/// Identifies critical-path bottlenecks and allocates extra CGRAs.

class PipelineBalancer {
public:
  using ProfileFn = std::function<void(TaskGraphNode *, TaskflowTaskOp)>;

  /// Runs pipeline balance on the graph.
  ///
  /// For each iteration, speculatively increments the bottleneck task's
  /// cgra_count by 1 and re-profiles it via profile_fn. If the new estimated
  /// latency is lower, the change is accepted; otherwise it is reverted and
  /// the node is marked saturated (no further CGRA additions help it).
  ///
  /// This avoids blindly assigning more CGRAs without checking whether the
  /// larger array actually produces a better compiled_ii.
  ///
  /// Returns true if any changes were accepted.
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

      // Finds the bottleneck: the node on the critical path with highest
      // estimated latency. We recompute the critical path every iteration
      // because adding CGRAs to the previous bottleneck may shift the
      // critical path to a different node.
      TaskGraphNode *bottleneck = findBottleneck(graph, saturated_nodes);
      if (!bottleneck) {
        break;
      }

      int old_cgra_count = bottleneck->cgra_count;
      int new_cgra_count = old_cgra_count + 1;

      // Checks if incrementing cgra_count is feasible on the 4×4 grid.
      if (!canFitOnGrid(new_cgra_count)) {
        saturated_nodes.insert(bottleneck);
        continue;
      }

      // Save state for potential rollback.
      int64_t old_latency = bottleneck->estimatedLatency();
      int64_t old_ii     = bottleneck->ii;
      int64_t old_steps  = bottleneck->steps;
      CGRAShape old_shape = bottleneck->shape;

      // Speculatively apply the new CGRA count and re-profile.
      bottleneck->cgra_count = new_cgra_count;
      bottleneck->shape = pickBestShape(new_cgra_count);

      llvm::errs()
          << "  Balance: trying Task " << bottleneck->id << " ("
          << bottleneck->op.getTaskName().str()
          << ") cgra_count=" << old_cgra_count << "->" << new_cgra_count
          << ", shape=" << bottleneck->shape.describe(new_cgra_count)
          << ", tile_array=" << (bottleneck->shape.rows * kPerCgraRows)
          << "x" << (bottleneck->shape.cols * kPerCgraCols)
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
    /// Computes the weighted critical path length from a given node to any sink.
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

    /// Computes the longest path from any source to the given node
    /// (depth_from_source). Uses dynamic programming with memoization.
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

    /// Finds the bottleneck node on the critical path using full slack analysis.
    ///
    /// For each node, slack is defined as:
    ///   slack(node) = global_critical_path
    ///                 - depth_from_source(node)
    ///                 - depth_to_sink(node)
    ///                 + node->estimatedLatency()
    ///
    /// where depth_from_source includes the node's own latency, and
    /// depth_to_sink (computeCriticalPathFrom) also includes the node's own
    /// latency, so we add it back once to avoid double-counting.
    ///
    /// A node is on the critical path iff slack == 0.
    /// Among critical-path nodes, the one with highest individual latency
    /// is the bottleneck (reducing its latency most benefits the pipeline).
    TaskGraphNode *findBottleneck(TaskDependencyGraph &graph,
                                  const llvm::DenseSet<TaskGraphNode *> &ignored) {
      llvm::DenseMap<TaskGraphNode *, int64_t> to_sink_cache;
      llvm::DenseMap<TaskGraphNode *, int64_t> from_source_cache;

      // Computes depth_to_sink for all nodes (via computeCriticalPathFrom).
      int64_t global_critical_path = 0;
      for (auto &node : graph.nodes) {
        int64_t cp = computeCriticalPathFrom(node.get(), to_sink_cache);
        global_critical_path = std::max(global_critical_path, cp);
      }

      // Computes depth_from_source for all nodes.
      for (auto &node : graph.nodes) {
        computeDepthFromSource(node.get(), from_source_cache);
      }

      // Finds the critical-path node with highest individual latency.
      TaskGraphNode *bottleneck = nullptr;
      int64_t max_latency = -1;

      for (auto &node : graph.nodes) {
        if (ignored.count(node.get())) continue;
        if (node->cgra_count >= node->trip_count) continue;

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
/// Merges independent tasks (no edge in either direction) into a single task
/// to reduce total CGRA count.  Fusion candidates are chosen to minimize
/// |trip_count_a - trip_count_b| for balanced utilization.

class UtilizationFuser {
public:
  using ProfileFn = std::function<void(TaskGraphNode *, TaskflowTaskOp)>;

  /// Runs utilization fusion. Returns true if any fusions occurred.
  /// Only performs ONE fusion per call — the caller should rebuild the graph
  /// and call again if more fusions are desired.
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
  /// Finds the best pair of independent tasks to fuse.
  /// Selects the pair with the most balanced trip_count (minimizes
  /// |trip_count_a - trip_count_b|) to avoid wasting computation when
  /// the fused task executes both loop nests concurrently on the shared array.
  std::optional<std::pair<TaskGraphNode *, TaskGraphNode *>>
  findBestFusionCandidate(TaskDependencyGraph &graph) {
    TaskGraphNode *best_a = nullptr;
    TaskGraphNode *best_b = nullptr;
    int64_t best_cost = INT64_MAX;

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
      for (size_t j = i + 1; j < graph.nodes.size(); ++j) {
        auto *a = graph.nodes[i].get();
        auto *b = graph.nodes[j].get();

        // Skips tasks with value outputs (e.g. reduction loops with iter_args).
        // Sequential concatenation of loop bodies doesn't handle the cross-task
        // value flow required for these outputs.
        if (!a->op.getValueOutputs().empty() ||
            !b->op.getValueOutputs().empty()) {
          continue;
        }

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

  /// Checks whether fusing tasks a and b is safe w.r.t. dominance.
  /// Returns false if any other task positioned between a and b in the IR
  /// has a dependency (edge) on either a or b — because moving the fused
  /// task would break that intermediate dependency.
  bool canSafelyFuse(TaskGraphNode *a, TaskGraphNode *b,
                     TaskDependencyGraph &graph) {
    auto *task_a = a->op.getOperation();
    auto *task_b = b->op.getOperation();

    if (task_a->getBlock() != task_b->getBlock()) return false;

    // Ensures task_a is before task_b.
    if (!task_a->isBeforeInBlock(task_b)) {
      std::swap(task_a, task_b);
      std::swap(a, b);
    }

    // Checks: no other task between a and b should have an edge from/to a or b.
    for (auto &node : graph.nodes) {
      if (node.get() == a || node.get() == b) continue;

      auto *other_op = node->op.getOperation();
      if (other_op->getBlock() != task_a->getBlock()) continue;

      // Is this node between task_a and task_b?
      if (task_a->isBeforeInBlock(other_op) &&
          other_op->isBeforeInBlock(task_b)) {
        // Checks if this intermediate task has any dependency on a or b.
        if (!graph.areIndependent(a, node.get()) ||
            !graph.areIndependent(b, node.get())) {
          return false;
        }
      }
    }
    return true;
  }

  /// Performs IR-level fusion of two independent tasks.
  /// Creates a new task with sequential concatenation of both loop nests.
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

    /// Insert right after the latest operand definition.
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
    // Writes outputs = merged write memrefs (each becomes a result).
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

    // Step 5: Creates the body block with all operands as block arguments.
    Block *body = new Block();
    fused_task.getBody().push_back(body);
    // Block args order: read_memrefs, write_memrefs, value_inputs.
    for (Value v : merged_read_memrefs) {
      body->addArgument(v.getType(), fused_task.getLoc());
    }
    for (Value v : merged_write_memrefs) {
      body->addArgument(v.getType(), fused_task.getLoc());
    }
    for (Value v : merged_value_inputs) {
      body->addArgument(v.getType(), fused_task.getLoc());
    }

    // Step 6: Builds mapping from old block args to new block args.
    auto buildArgMapping = [&](TaskflowTaskOp orig_task, IRMapping &mapping) {
      Block &orig_body = orig_task.getBody().front();
      unsigned orig_arg_idx = 0;

      for (Value memref : orig_task.getReadMemrefs()) {
        unsigned new_idx = findOperandIndex(merged_read_memrefs, memref);
        mapping.map(orig_body.getArgument(orig_arg_idx++),
                    body->getArgument(new_idx));
      }

      for (Value memref : orig_task.getWriteMemrefs()) {
        unsigned new_idx = merged_read_memrefs.size() +
                           findOperandIndex(merged_write_memrefs, memref);
        mapping.map(orig_body.getArgument(orig_arg_idx++),
                    body->getArgument(new_idx));
      }

      for (Value val : orig_task.getValueInputs()) {
        unsigned new_idx = merged_read_memrefs.size() +
                           merged_write_memrefs.size() +
                           findOperandIndex(merged_value_inputs, val);
        mapping.map(orig_body.getArgument(orig_arg_idx++),
                    body->getArgument(new_idx));
      }
    };

    // Step 7: Clones task_a body into fused task.
    {
      IRMapping mapping_a;
      buildArgMapping(task_a, mapping_a);
      OpBuilder body_builder = OpBuilder::atBlockEnd(body);
      Block &src_body = task_a.getBody().front();
      for (auto &op : src_body.getOperations()) {
        if (!isa<TaskflowYieldOp>(op)) {
          body_builder.clone(op, mapping_a);
        }
      }
    }

    // Step 8: Clones task_b body into fused task (sequentially after task_a).
    {
      IRMapping mapping_b;
      buildArgMapping(task_b, mapping_b);
      OpBuilder body_builder = OpBuilder::atBlockEnd(body);
      Block &src_body = task_b.getBody().front();
      for (auto &op : src_body.getOperations()) {
        if (!isa<TaskflowYieldOp>(op)) {
          body_builder.clone(op, mapping_b);
        }
      }
    }

    // Step 9: Creates yield op with merged write + value outputs.
    {
      OpBuilder body_builder = OpBuilder::atBlockEnd(body);
      SmallVector<Value> yield_writes;
      for (size_t i = 0; i < merged_write_memrefs.size(); ++i) {
        yield_writes.push_back(
            body->getArgument(merged_read_memrefs.size() + i));
      }
      // Value outputs are empty for utilization fusion of independent tasks.
      SmallVector<Value> yield_values;
      body_builder.create<TaskflowYieldOp>(fused_task.getLoc(), yield_writes,
                                           yield_values);
    }

    // Step 10: Sets fused attributes for the latency model.
    // trip_count: max of both, since independent tasks execute concurrently
    // on the shared tile array.
    int64_t fused_trip = std::max(node_a->trip_count, node_b->trip_count);
    fused_task->setAttr("trip_count",
                        OpBuilder(fused_task).getI64IntegerAttr(fused_trip));

    // Profiles the fused task to get real ii and steps.
    // profileTask handles multi-body tasks by splitting them into per-loop-nest
    // temporary tasks internally, so we can call it directly here.
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
    replaceTaskResults(task_a, fused_task, merged_write_memrefs);
    replaceTaskResults(task_b, fused_task, merged_write_memrefs);

    // Step 12: Erases original tasks.
    task_a.erase();
    task_b.erase();

    return true;
  }

  /// Finds the index of a value in a list.
  unsigned findOperandIndex(const SmallVector<Value> &list, Value v) {
    for (unsigned i = 0; i < list.size(); ++i) {
      if (list[i] == v) return i;
    }
    llvm_unreachable("Value not found in operand list");
  }

  /// Replaces results of an original task with corresponding results from the
  /// fused task.
  void replaceTaskResults(TaskflowTaskOp orig_task, TaskflowTaskOp fused_task,
                          const SmallVector<Value> &merged_write_memrefs) {
    // Writes outputs first, then value outputs.
    for (unsigned i = 0; i < orig_task.getWriteOutputs().size(); ++i) {
      Value orig_result = orig_task.getWriteOutputs()[i];
      Value orig_write = orig_task.getWriteMemrefs()[i];
      unsigned fused_idx = findOperandIndex(merged_write_memrefs, orig_write);
      orig_result.replaceAllUsesWith(fused_task.getWriteOutputs()[fused_idx]);
    }
    // Value outputs: utilization fusion of independent tasks should not
    // produce value outputs. Assert to catch unexpected cases.
    assert(orig_task.getValueOutputs().empty() &&
           "Value outputs in utilization-fused independent tasks are "
           "unexpected; fusion logic needs extension to handle them.");
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

  StringRef getArgument() const override {
    return "resource-aware-task-optimization";
  }

  StringRef getDescription() const override {
    return "Balances pipeline latency and fuses independent tasks for CGRA "
           "utilization";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<neura::NeuraDialect>();
    registry.insert<taskflow::TaskflowDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    llvm::errs() << "=== ResourceAwareTaskOptimization on "
                 << func.getName() << " ===\n";

    constexpr int kMaxOuterIterations = 10;

    for (int outer = 0; outer < kMaxOuterIterations; ++outer) {
      // Rebuilds graph from current IR state.
      TaskDependencyGraph graph;
      graph.build(func);

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
      // Fuses independent tasks to free up CGRA budget for balance.
      UtilizationFuser fuser;
      // Exposes TaskDependencyGraph::profileTask to UtilizationFuser via a
      // lambda so fused tasks get real ResMII/RecMII profiling.
      auto profile_fn = [&graph](TaskGraphNode *node, TaskflowTaskOp task) {
        graph.profileTaskPublic(node, task);
      };
      bool fuse_changed = fuser.fuse(func, graph, profile_fn);

      llvm::errs() << "[ResourceAware] After fusion: total_cgras="
                   << graph.getTotalAllocatedCGRAs() << "\n";

      // Rebuilds graph after fusion (tasks may have been erased/created).
      if (fuse_changed) {
        graph = TaskDependencyGraph();
        graph.build(func);
      }

      // Phase 2: Latency-Aware Pipeline Balance.
      // Uses analytical-only profiling for speculative balance probes.
      // The mapper is skipped to prevent infinite backtracking on large tile
      // arrays. ResMII/RecMII estimates are sufficient to decide if adding a
      // CGRA reduces the bottleneck's II (RecMII-bound tasks will correctly
      // show no improvement; ResMII-bound tasks will show proportional gains).
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
            node->op->setAttr("compiled_ii", b.getI64IntegerAttr(node->ii));
          }
          if (node->steps != kUnprofiled) {
            node->op->setAttr("steps", b.getI64IntegerAttr(node->steps));
          }
          if (node->trip_count > 0) {
            node->op->setAttr("trip_count",
                              b.getI64IntegerAttr(node->trip_count));
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
        // Converged — write ALL attributes (cgra_count, ii, steps) to IR
        // for every task. Non-fused tasks only got cgra_count written during
        // intermediate iterations; ii, steps, and trip_count live only in the
        // graph node and must be persisted here.
        for (auto &node : graph.nodes) {
          OpBuilder b(node->op);
          node->shape = pickBestShape(node->cgra_count);
          node->op->setAttr("cgra_count",
                            b.getI32IntegerAttr(node->cgra_count));
          node->op->setAttr("compiled_ii",
                            b.getI64IntegerAttr(node->ii));
          node->op->setAttr("steps",
                            b.getI64IntegerAttr(node->steps));
          node->op->setAttr("trip_count",
                            b.getI64IntegerAttr(node->trip_count));
          // Writes tile_shape attribute: simple "NxM" bounding-box string.
          // The detailed occupancy diagram is printed in the summary below.
          std::string shape_str = node->shape.irAttr();
          node->op->setAttr("tile_shape", b.getStringAttr(shape_str));
        }
        break;
      }
    }

    // Final validation and tile occupation summary with visual 4x4 grid.
    {
      TaskDependencyGraph final_graph;
      final_graph.build(func);
      int final_total = final_graph.getTotalAllocatedCGRAs();

      /// Assigns each task a single character label for the combined grid.
      /// Tasks are labelled '0','1','2',... ; free cells shown as '.'.
      /// grid[row][col] == -1 means free.
      std::vector<std::vector<int>> combined_grid(
          kGridRows, std::vector<int>(kGridCols, -1));

      // Packs tasks onto the grid left-to-right, top-to-bottom.
      int next_col = 0, next_row = 0;
      int task_idx = 0;

      llvm::errs() << "\n=== Tile Occupation Summary (4x" << kGridCols
                   << " CGRA Grid) ===\n";

      for (auto &node : final_graph.nodes) {
        auto shape = pickBestShape(node->cgra_count);
        int tile_rows = shape.rows * kPerCgraRows;
        int tile_cols = shape.cols * kPerCgraCols;

        // Per-task grid (shape.rows x shape.cols bbox, filled up to cgra_count).
        llvm::errs() << "\n  [" << task_idx << "] " << node->op.getTaskName()
                     << "  cgra_count=" << node->cgra_count
                     << "  shape=" << shape.describe(node->cgra_count)
                     << "  tile_array=" << tile_rows << "x" << tile_cols
                     << "  ii=" << node->ii
                     << "  steps=" << node->steps
                     << "  trip_count=" << node->trip_count << "\n";

        // Draws a per-task bounding-box grid (shape.rows x shape.cols).
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

        // Places onto combined grid (pack sequentially).
        int placed = 0;
        for (int r = next_row; r < kGridRows && placed < node->cgra_count; ++r) {
          for (int c = (r == next_row ? next_col : 0);
               c < kGridCols && placed < node->cgra_count; ++c) {
            combined_grid[r][c] = task_idx;
            next_row = r;
            next_col = c + 1;
            if (next_col >= kGridCols) { next_col = 0; next_row = r + 1; }
            ++placed;
          }
        }
        ++task_idx;
      }

      // Prints combined 4xN grid.
      llvm::errs() << "\n  Combined 4x" << kGridCols << " Grid"
                   << " (" << final_total << "/" << kTotalCGRAs << " used):\n";
      llvm::errs() << "  +";
      for (int c = 0; c < kGridCols; ++c) llvm::errs() << "---+";
      llvm::errs() << "\n";
      for (int r = 0; r < kGridRows; ++r) {
        llvm::errs() << "  |";
        for (int c = 0; c < kGridCols; ++c) {
          int t = combined_grid[r][c];
          if (t < 0)
            llvm::errs() << " . |";
          else
            llvm::errs() << " " << (char)('0' + t) << " |";
        }
        llvm::errs() << "\n";
        llvm::errs() << "  +";
        for (int c = 0; c < kGridCols; ++c) llvm::errs() << "---+";
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
