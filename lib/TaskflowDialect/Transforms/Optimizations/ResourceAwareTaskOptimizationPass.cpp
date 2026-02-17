//===- ResourceAwareTaskOptimizationPass.cpp - Pipeline Balance & Fusion --===//
//
// This pass performs two-phase optimization on the task graph:
// 1. Utilization Fusion: merges independent (no-edge) tasks into a single task
//    to free up CGRA resources.
// 2. Latency-Aware Balance: allocates extra CGRAs to critical-path bottleneck
//    tasks using the pipelined latency model:
//      latency = II * (ceil(trip_count / cgra_count) - 1) + steps
//    where II and steps are obtained via speculative lowering to the Neura
//    dialect (running the full downstream pipeline on a cloned module).
//
// Targets a hardcoded 4x4 CGRA grid (16 CGRAs total).
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

// Sentinel value: 0 means "not yet profiled". After profileTask() runs,
// both steps and ii MUST be > 0. An assert fires if profiling fails silently.
constexpr int64_t kUnprofiled = 0;

//===----------------------------------------------------------------------===//
// Task Dependency Graph
//===----------------------------------------------------------------------===//

struct TaskGraphNode {
  size_t id;
  TaskflowTaskOp op;
  int64_t trip_count = 1;
  int64_t steps = kUnprofiled;  // Pipeline depth (critical path through DFG).
  int64_t ii = kUnprofiled;     // Initiation interval = max(ResMII, RecMII).
  int cgra_count = 1;

  // Dependency edges (both SSA and memory).
  SmallVector<TaskGraphNode *> predecessors;
  SmallVector<TaskGraphNode *> successors;

  TaskGraphNode(size_t id, TaskflowTaskOp op) : id(id), op(op) {}

  /// Returns estimated task latency using the pipelined execution model:
  ///   latency = II * (ceil(trip_count / cgra_count) - 1) + steps.
  int64_t estimatedLatency() const {
    int64_t iterations_per_cgra =
        (trip_count + cgra_count - 1) / cgra_count;
    return ii * (iterations_per_cgra - 1) + steps;
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
      bool has_precomputed = task->hasAttr("ii") && task->hasAttr("steps");
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
      if (auto attr = task->getAttrOfType<IntegerAttr>("ii")) {
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

    // 3. Builds memory edges (read-after-write and write-after-write).
    DenseMap<Value, SmallVector<TaskGraphNode *>> memref_writers;
    for (auto &node : nodes) {
      for (Value memref : node->op.getOriginalWriteMemrefs()) {
        memref_writers[memref].push_back(node.get());
      }
    }
    // RAW edges: writer -> reader.
    for (auto &node : nodes) {
      for (Value memref : node->op.getOriginalReadMemrefs()) {
        if (memref_writers.count(memref)) {
          for (auto *writer : memref_writers[memref]) {
            if (writer != node.get()) {
              addEdge(writer, node.get());
            }
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
  int totalCGRAs() const {
    int total = 0;
    for (auto &node : nodes) {
      total += node->cgra_count;
    }
    return total;
  }

  /// Public wrapper for profileTask — used by UtilizationFuser to re-profile
  /// fused tasks with the real downstream Neura pipeline.
  void profileTaskPublic(TaskGraphNode *node, TaskflowTaskOp task) {
    profileTask(node, task);
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
  void profileTask(TaskGraphNode *node, TaskflowTaskOp task) {
    MLIRContext *ctx = task.getContext();
    OpBuilder builder(ctx);
    Location loc = task.getLoc();

    auto parent_func = task->getParentOfType<func::FuncOp>();
    if (!parent_func) {
      llvm::errs() << "[profileTask] WARNING: task has no parent func, "
                      "skipping profiling\n";
      node->ii = 1;
      node->steps = 1;
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

      // Find the cloned copy of the target task and erase all others.
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
          // Replace all results with undef-like values so uses don't dangle.
          for (OpResult res : t->getResults()) {
            // Create a placeholder value so uses don't dangle.
            // Use UnrealizedConversionCastOp as a universal placeholder that
            // works for any type (memref, index, integer, float, etc.)
            // without needing type-specific logic. Verifier is disabled.
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
        llvm::errs() << "[profileTask] Phase 1 (Taskflow->Neura) failed for "
                     << task.getTaskName() << ", using defaults\n";
        node->ii = 1;
        node->steps = 1;
        phase1_module.erase();
        return;
      }
    }

    // ================================================================
    // Phase 2: For each kernel, clone body -> func -> run Neura pipeline
    // ================================================================
    // Collect all neura.kernel ops created by Phase 1.
    SmallVector<neura::KernelOp> kernels;
    phase1_module.walk([&](neura::KernelOp k) { kernels.push_back(k); });

    if (kernels.empty()) {
      llvm::errs() << "[profileTask] No kernels found after Phase 1 for "
                   << task.getTaskName() << ", using defaults\n";
      node->ii = 1;
      node->steps = 1;
      phase1_module.erase();
      return;
    }

    int best_compiled_ii = 0;
    int best_cp_depth = 1;

    for (neura::KernelOp kernel : kernels) {
      // Create a fresh module with a standalone func containing the kernel
      // body. All downstream Neura passes walk func::FuncOp with
      // accelerator="neura", so we package the kernel body as such.
      auto phase2_module = ModuleOp::create(loc);
      int compiled_ii = 0;
      int cp_depth = 1;

      if (succeeded(
              runNeuraPipelineOnKernel(ctx, kernel, phase2_module,
                                      compiled_ii, cp_depth))) {
        llvm::errs() << "[profileTask] kernel in " << task.getTaskName()
                     << ": compiled_ii=" << compiled_ii
                     << ", cp_depth=" << cp_depth << "\n";
      } else {
        llvm::errs() << "[profileTask] Phase 2 failed for kernel in "
                     << task.getTaskName() << ", extracting partial\n";
        extractMetricsFromPartialIR(phase2_module, compiled_ii, cp_depth);
      }

      best_compiled_ii = std::max(best_compiled_ii, compiled_ii);
      best_cp_depth = std::max(best_cp_depth, cp_depth);
      phase2_module.erase();
    }

    node->ii = (best_compiled_ii > 0) ? best_compiled_ii : 1;
    node->steps = std::max(best_cp_depth, 1);

    llvm::errs() << "[profileTask] " << task.getTaskName()
                 << ": compiled_ii=" << node->ii
                 << ", steps=" << node->steps << "\n";

    phase1_module.erase();
  }

  /// Clones a neura.kernel body into a standalone func::FuncOp inside
  /// dst_module, then runs the full Neura lowering + mapping pipeline.
  /// Returns success if MapToAccelerator ran and produced compiled_ii.
  LogicalResult runNeuraPipelineOnKernel(MLIRContext *ctx,
                                         neura::KernelOp kernel,
                                         ModuleOp dst_module,
                                         int &compiled_ii,
                                         int &cp_depth) {
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

    // Result types from the kernel op.
    SmallVector<Type> result_types(kernel.getResultTypes());

    auto func_type = builder.getFunctionType(arg_types, result_types);
    auto wrapper_func = builder.create<func::FuncOp>(
        loc, "__speculative_kernel__", func_type);

    // Tag as neura accelerator — all downstream passes check this.
    wrapper_func->setAttr("accelerator",
                          builder.getStringAttr("neura"));

    // Clone the entire kernel region (all blocks) into the func body.
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

    // Run the full Neura lowering + dataflow pipeline.
    // Pipeline order follows the reference tests in
    // test/multi-cgra/kernel_mapping/ (fir, relu, loop-in-kernel).
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
    // pm.addPass(neura::createFoldConstantPass());

    // InsertDataMov: wraps operands with neura.data_mov for the mapper.
    pm.addPass(neura::createInsertDataMovPass());

    if (failed(pm.run(dst_module))) {
      // Pre-mapper pipeline failed — extract best-effort metrics from partial
      // Neura IR using ResMII/RecMII analysis.
      extractMetricsFromPartialIR(dst_module, compiled_ii, cp_depth);
      return failure();
    }

    // Extract ResMII/RecMII from the post-InsertDataMov Neura IR. These are
    // the authoritative lower-bounds and the fallback metrics when the mapper
    // is skipped. We compute them now (before MapToAccelerator modifies the IR
    // with dfg_id attrs) so that the fallback always uses the same IR.
    {
      const neura::Architecture &architecture = neura::getArchitecture();
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
      });
    }

    // Optionally run MapToAcceleratorPass to get the true compiled_ii.
    //
    // Guards:
    //   1. All non-Reserve operand producers must be DataMovOp (mapper asserts
    //      otherwise).
    //   2. Kernel must be small enough (<= kMapperOpLimit ops) to avoid
    //      exponential backtracking blowup during speculative profiling.
    //
    // If either guard fails, we keep the ResMII/RecMII values computed above.
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
                 << " all_data_movs=" << all_data_movs_ok << "\n";

    if (all_data_movs_ok && total_mapped_ops <= kMapperOpLimit) {
      // Run MapToAcceleratorPass in a fresh pass manager on the already-lowered
      // dst_module (pre-mapper pipeline already ran above).
      PassManager pm2(ctx);
      pm2.enableVerifier(false);
      pm2.addPass(neura::createMapToAcceleratorPass());

      if (succeeded(pm2.run(dst_module))) {
        // Read the true compiled_ii from mapping_info (overrides ResMII/RecMII).
        // compiled_ii and cp_depth are already initialized from the pre-mapper
        // ResMII/RecMII analysis above; mapper result takes precedence.
        dst_module.walk([&](func::FuncOp fn) {
          if (!fn->hasAttr("accelerator")) return;
          if (auto mapping_info =
                  fn->getAttrOfType<DictionaryAttr>(neura::attr::kMappingInfo)) {
            if (auto ii_attr =
                    mapping_info.getAs<IntegerAttr>(neura::attr::kCompiledII))
              compiled_ii = (int)ii_attr.getInt(); // authoritative value
          }
        });
        return success();
      }
      // Mapper failed for all II values — keep ResMII/RecMII from above.
    }

    // Fallback already computed via ResMII/RecMII above; nothing more to do.
    return success();
  }


  /// Extracts metrics from partially-lowered Neura IR when the full pipeline
  /// fails. Uses ResMII/RecMII analysis and critical path depth on whatever
  /// Neura ops were successfully created.
  void extractMetricsFromPartialIR(ModuleOp tmp_module,
                                   int &out_ii, int &out_cp_depth) {
    const neura::Architecture &architecture = neura::getArchitecture();

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
  /// Runs pipeline balance on the graph.
  /// Returns true if any changes were made.
  bool balance(TaskDependencyGraph &graph) {
    bool changed = false;

    for (int iter = 0; iter < kMaxBalanceIterations; ++iter) {
      int total_cgras = graph.totalCGRAs();
      if (total_cgras >= kTotalCGRAs) {
        break;
      }

      // Finds the bottleneck: the node on the critical path with highest
      // estimated latency. We recompute the critical path every iteration
      // because adding CGRAs to the previous bottleneck may shift the
      // critical path to a different node.
      llvm::DenseSet<TaskGraphNode *> empty_ignored;
      TaskGraphNode *bottleneck = findBottleneck(graph, empty_ignored);
      if (!bottleneck) {
        break;
      }

      // Checks if incrementing cgra_count actually reduces latency
      // using pipelined model: II * (ceil(trip/cgra) - 1) + steps.
      int64_t current_latency = bottleneck->estimatedLatency();
      int new_cgra_count = bottleneck->cgra_count + 1;
      int64_t new_iterations =
          (bottleneck->trip_count + new_cgra_count - 1) / new_cgra_count;
      int64_t new_latency =
          bottleneck->ii * (new_iterations - 1) + bottleneck->steps;

      if (new_latency >= current_latency) {
        // No improvement — this bottleneck is saturated. Try skipping it.
        // Use a secondary search excluding saturated nodes.
        llvm::DenseSet<TaskGraphNode *> ignored_nodes;
        ignored_nodes.insert(bottleneck);
        bool found_alternative = false;

        while (graph.totalCGRAs() < kTotalCGRAs) {
          TaskGraphNode *alt = findBottleneck(graph, ignored_nodes);
          if (!alt) break;

          int64_t alt_lat = alt->estimatedLatency();
          int alt_new = alt->cgra_count + 1;
          int64_t alt_new_iters =
              (alt->trip_count + alt_new - 1) / alt_new;
          int64_t alt_new_lat =
              alt->ii * (alt_new_iters - 1) + alt->steps;

          if (alt_new_lat >= alt_lat) {
            ignored_nodes.insert(alt);
            continue;
          }

          alt->cgra_count = alt_new;
          changed = true;
          found_alternative = true;

          llvm::errs()
              << "  Balance: Task " << alt->id << " ("
              << alt->op.getTaskName().str()
              << ") cgra_count=" << alt_new
              << ", latency: " << alt_lat << " -> " << alt_new_lat
              << ", total_cgras=" << graph.totalCGRAs() << "\n";
          break; // Recompute critical path from the top.
        }

        if (!found_alternative) break;
        continue;
      }

      // Allocates one more CGRA to the bottleneck.
      bottleneck->cgra_count = new_cgra_count;
      changed = true;

      llvm::errs()
          << "  Balance: Task " << bottleneck->id << " ("
          << bottleneck->op.getTaskName().str()
          << ") cgra_count=" << new_cgra_count
          << ", latency: " << current_latency << " -> " << new_latency
          << ", total_cgras=" << graph.totalCGRAs() << "\n";
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
/// to reduce total CGRA count.

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
  /// Prioritizes tasks with smallest combined trip count.
  std::optional<std::pair<TaskGraphNode *, TaskGraphNode *>>
  findBestFusionCandidate(TaskDependencyGraph &graph) {
    TaskGraphNode *best_a = nullptr;
    TaskGraphNode *best_b = nullptr;
    int64_t best_cost = INT64_MAX;

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
      for (size_t j = i + 1; j < graph.nodes.size(); ++j) {
        auto *a = graph.nodes[i].get();
        auto *b = graph.nodes[j].get();

        // Skip tasks with value outputs (e.g. reduction loops with iter_args).
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

        // Fusing only saves if total CGRAs > kTotalCGRAs or we want to
        // minimize CGRA usage. Each fusion removes one CGRA slot.
        int64_t cost = a->trip_count + b->trip_count;
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

    // Compute the correct insertion point: must be after all operands of
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

    // Insert right after the latest operand definition.
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
    // trip_count: sum of both (sequential execution).
    int64_t fused_trip = node_a->trip_count + node_b->trip_count;
    fused_task->setAttr("trip_count",
                        OpBuilder(fused_task).getI64IntegerAttr(fused_trip));

    // Run profileTask on the fused task to get real ii and steps from the
    // merged loop body (ResMII/RecMII may differ after concatenation).
    //
    // A fused task body contains N sequential loop nests (one per original
    // task). ConvertTaskflowToNeura asserts hyperblock_count==1, so we cannot
    // profile the fused task as-is. Instead we split its body back into
    // individual single-loop tasks, profile each independently, and take
    // max(ii) / sum(steps) as the fused task's metrics.
    {
      // Count top-level ops (loop nests) inside the fused task body,
      // excluding the final TaskflowYieldOp.
      SmallVector<Operation *> body_ops;
      Block &fused_body = fused_task.getBody().front();
      for (auto &op : fused_body.getOperations()) {
        if (!isa<TaskflowYieldOp>(op))
          body_ops.push_back(&op);
      }

      int64_t total_ii = 1;
      int64_t total_steps = 0;

      // For each top-level op (loop nest), create a temporary wrapper task,
      // profile it, then discard it.
      for (Operation *loop_op : body_ops) {
        // Build a minimal temporary task wrapping just this one loop op.
        // Insert it right before the fused task so parent_func is valid.
        OpBuilder tmp_builder(fused_task.getOperation());

        // Use the same operand signature as the fused task (conservative).
        auto tmp_task = tmp_builder.create<TaskflowTaskOp>(
            fused_task.getLoc(),
            fused_task.getWriteOutputs().getTypes(),
            fused_task.getValueOutputs().getTypes(),
            fused_task.getReadMemrefs(),
            fused_task.getWriteMemrefs(),
            fused_task.getValueInputs(),
            (fused_task.getTaskName().str() + "__split_profile__").c_str(),
            fused_task.getOriginalReadMemrefs(),
            fused_task.getOriginalWriteMemrefs());

        // Build the body: clone just this one op, then yield.
        Block *tmp_body = new Block();
        tmp_task.getBody().push_back(tmp_body);
        // Mirror block arguments from the fused task body.
        for (BlockArgument arg : fused_body.getArguments())
          tmp_body->addArgument(arg.getType(), arg.getLoc());

        OpBuilder body_builder = OpBuilder::atBlockEnd(tmp_body);

        // Build a mapping: fused_body args -> tmp_body args.
        IRMapping arg_map;
        for (auto [orig, repl] : llvm::zip(fused_body.getArguments(),
                                           tmp_body->getArguments()))
          arg_map.map(orig, repl);

        body_builder.clone(*loop_op, arg_map);

        // Yield: pass back the write-memref args unchanged.
        SmallVector<Value> yield_writes;
        for (size_t i = 0; i < fused_task.getWriteMemrefs().size(); ++i)
          yield_writes.push_back(
              tmp_body->getArgument(fused_task.getReadMemrefs().size() + i));
        SmallVector<Value> yield_vals;
        body_builder.create<TaskflowYieldOp>(fused_task.getLoc(),
                                             yield_writes, yield_vals);

        // Profile this single-loop task.
        TaskGraphNode tmp_node(/*id=*/0, tmp_task);
        profile_fn(&tmp_node, tmp_task);

        total_ii = std::max(total_ii, tmp_node.ii);
        total_steps += tmp_node.steps;

        // Discard the temporary task.
        tmp_task.erase();
      }

      fused_task->setAttr("steps",
                          OpBuilder(fused_task).getI64IntegerAttr(total_steps));
      fused_task->setAttr("ii",
                          OpBuilder(fused_task).getI64IntegerAttr(total_ii));
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
    // Write outputs first, then value outputs.
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
      // Fuse independent tasks to free up CGRA budget for balance.
      UtilizationFuser fuser;
      // Expose TaskDependencyGraph::profileTask to UtilizationFuser via a
      // lambda so fused tasks get real ResMII/RecMII profiling.
      auto profile_fn = [&graph](TaskGraphNode *node, TaskflowTaskOp task) {
        graph.profileTaskPublic(node, task);
      };
      bool fuse_changed = fuser.fuse(func, graph, profile_fn);

      llvm::errs() << "[ResourceAware] After fusion: total_cgras="
                   << graph.totalCGRAs() << "\n";

      // Rebuild graph after fusion (tasks may have been erased/created).
      if (fuse_changed) {
        graph = TaskDependencyGraph();
        graph.build(func);
      }

      // Phase 2: Latency-Aware Pipeline Balance.
      PipelineBalancer balancer;
      bool balance_changed = balancer.balance(graph);

      // Writes cgra_count, ii, steps, and trip_count back to IR during
      // iterations so that the next iteration's graph.build() reads them
      // and skips expensive re-profiling for unchanged tasks.
      if (balance_changed || fuse_changed) {
        for (auto &node : graph.nodes) {
          OpBuilder b(node->op);
          node->op->setAttr(
              "cgra_count", b.getI32IntegerAttr(node->cgra_count));
          if (node->ii != kUnprofiled) {
            node->op->setAttr("ii", b.getI64IntegerAttr(node->ii));
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
                   << graph.totalCGRAs() << "\n";

      if (!balance_changed && !fuse_changed) {
        // Converged — write ALL attributes (cgra_count, ii, steps) to IR
        // for every task. Non-fused tasks only got cgra_count written during
        // intermediate iterations; ii, steps, and trip_count live only in the
        // graph node and must be persisted here.
        for (auto &node : graph.nodes) {
          OpBuilder b(node->op);
          node->op->setAttr("cgra_count",
                            b.getI32IntegerAttr(node->cgra_count));
          node->op->setAttr("ii",
                            b.getI64IntegerAttr(node->ii));
          node->op->setAttr("steps",
                            b.getI64IntegerAttr(node->steps));
          node->op->setAttr("trip_count",
                            b.getI64IntegerAttr(node->trip_count));
        }
        break;
      }
    }

    // Final validation.
    {
      TaskDependencyGraph final_graph;
      final_graph.build(func);
      int final_total = final_graph.totalCGRAs();
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
