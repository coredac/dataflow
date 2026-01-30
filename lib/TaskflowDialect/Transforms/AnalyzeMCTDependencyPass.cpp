//===- AnalyzeMCTDependencyPass.cpp - MCT Dependency Analysis Pass --------===//
//
// This pass analyzes dependencies between Minimized Canonicalized Tasks (MCTs)
// for multi-CGRA mapping optimization.
//
// Architecture context:
// - Our architecture can combine multiple CGRAs into one logical CGRA.
// - Task dependencies: SSA use-def AND memory access (RAW, WAR, WAW).
//
// This pass identifies:
// 1. SSA dependencies: Task output → Task input (data flow).
// 2. Memory dependencies: RAW, WAR, WAW via shared memrefs.
// 3. Same-header pairs: Fusion candidates for data forwarding.
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===----------------------------------------------------------------------===//
// Counter Chain Info
//===----------------------------------------------------------------------===//
/// Represents the counter chain (loop header bounds) of an MCT.
struct CounterChainInfo {
  SmallVector<int64_t> bounds; // e.g., {4, 8, 6} for 0→4, 0→8, 0→6.

  bool operator==(const CounterChainInfo &other) const {
    return bounds == other.bounds;
  }

  /// Prints the counter chain in readable format.
  void print(llvm::raw_ostream &os) const {
    os << "(";
    for (size_t i = 0; i < bounds.size(); ++i) {
      if (i > 0)
        os << "-";
      os << bounds[i];
    }
    os << ")";
  }
};

//===----------------------------------------------------------------------===//
// MCT Info
//===----------------------------------------------------------------------===//
/// Stores analysis results for an MCT.
struct MCTInfo {
  TaskflowTaskOp task;
  StringRef task_name;
  CounterChainInfo counter_chain;
  SetVector<Value> source_memref_reads;  // Source memrefs (function args or task outputs).
  SetVector<Value> source_memref_writes; // Source memrefs that are written.

  /// Analyzes the task and resolves block args to source memrefs.
  void analyze() {
    task_name = task.getTaskName();

    // Builds block arg to source mapping.
    Block *body = &task.getBody().front();
    auto mem_inputs = task.getMemoryInputs();
    auto mem_args = body->getArguments().take_front(mem_inputs.size());

    DenseMap<Value, Value> block_arg_to_source;
    for (auto [input, arg] : llvm::zip(mem_inputs, mem_args)) {
      block_arg_to_source[arg] = input;
    }

    // Collects memory accesses and resolves to source.
    task.walk([&](Operation *op) {
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        Value memref = load.getMemRef();
        auto it = block_arg_to_source.find(memref);
        if (it != block_arg_to_source.end()) {
          source_memref_reads.insert(it->second);
        }
      } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        Value memref = store.getMemRef();
        auto it = block_arg_to_source.find(memref);
        if (it != block_arg_to_source.end()) {
          source_memref_writes.insert(it->second);
        }
      }
    });

    // Extracts counter chain.
    task.walk([&](TaskflowCounterOp counter) {
      if (!counter.getParentIndex()) {
        collectCounterChain(counter);
      }
    });
  }

private:
  /// Recursively collects counter chain bounds from root to leaf.
  void collectCounterChain(TaskflowCounterOp counter) {
    auto upper = counter.getUpperBound();
    counter_chain.bounds.push_back(upper.getSExtValue());

    for (Operation *user : counter.getResult().getUsers()) {
      if (auto child = dyn_cast<TaskflowCounterOp>(user)) {
        collectCounterChain(child);
        break;
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// Memory Dependency Types
//===----------------------------------------------------------------------===//
enum class DepType { SSA, RAW, WAR, WAW };

/// Represents a dependency between two MCTs.
struct Dependency {
  DepType type;
  size_t producer_idx;
  size_t consumer_idx;
  bool same_header;
  Value via_memref; // The memref/SSA value that creates the dependency.
};

//===----------------------------------------------------------------------===//
// MCT Dependency Analyzer
//===----------------------------------------------------------------------===//
/// Analyzes dependencies between MCTs for multi-CGRA mapping.
class MCTDependencyAnalyzer {
public:
  /// Analyzes all tasks in the function and reports dependencies.
  void analyze(func::FuncOp func) {
    SmallVector<TaskflowTaskOp> tasks;
    func.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });

    if (tasks.empty()) {
      llvm::errs() << "No taskflow.task operations found.\n";
      return;
    }

    llvm::outs() << "=== MCT Dependency Analysis ===\n";
    llvm::outs() << "Found " << tasks.size() << " MCTs.\n\n";

    // Analyzes each task.
    SmallVector<MCTInfo> mct_infos;
    DenseMap<Value, size_t> output_to_producer; // Maps task output to producer index.

    for (size_t idx = 0; idx < tasks.size(); ++idx) {
      TaskflowTaskOp task = tasks[idx];
      MCTInfo info;
      info.task = task;
      info.analyze();
      mct_infos.push_back(info);

      // Records outputs for SSA dependency tracking.
      for (Value output : task.getMemoryOutputs()) {
        output_to_producer[output] = idx;
      }

      // Prints task info.
      llvm::outs() << "MCT " << idx << ": " << info.task_name << "\n";
      llvm::outs() << "  Counter Chain: ";
      info.counter_chain.print(llvm::outs());
      llvm::outs() << "\n";
      llvm::outs() << "  Source Reads: ";
      for (Value v : info.source_memref_reads) {
        if (auto arg = dyn_cast<BlockArgument>(v)) {
          llvm::outs() << "func_arg" << arg.getArgNumber() << " ";
        } else {
          llvm::outs() << v << " ";
        }
      }
      llvm::outs() << "\n";
      llvm::outs() << "  Source Writes: ";
      for (Value v : info.source_memref_writes) {
        if (auto arg = dyn_cast<BlockArgument>(v)) {
          llvm::outs() << "func_arg" << arg.getArgNumber() << " ";
        } else {
          llvm::outs() << v << " ";
        }
      }
      llvm::outs() << "\n\n";
    }

    // Detects dependencies.
    llvm::outs() << "=== Dependencies ===\n";
    SmallVector<Dependency> deps;

    for (size_t i = 0; i < mct_infos.size(); ++i) {
      TaskflowTaskOp task = mct_infos[i].task;

      // Checks SSA dependencies: if this task's input is another task's output.
      for (Value input : task.getMemoryInputs()) {
        auto it = output_to_producer.find(input);
        if (it != output_to_producer.end()) {
          size_t producer_idx = it->second;
          bool same_header = mct_infos[producer_idx].counter_chain ==
                             mct_infos[i].counter_chain;
          deps.push_back({DepType::SSA, producer_idx, i, same_header, input});
          llvm::outs() << mct_infos[producer_idx].task_name << " → "
                       << mct_infos[i].task_name << " : SSA";
          if (same_header) {
            llvm::outs() << " [SAME HEADER - FUSION CANDIDATE]";
          }
          llvm::outs() << "\n";
        }
      }

      // Checks RAW dependencies via shared function arguments.
      for (size_t j = 0; j < i; ++j) {
        for (Value w : mct_infos[j].source_memref_writes) {
          if (mct_infos[i].source_memref_reads.contains(w)) {
            bool same_header =
                mct_infos[j].counter_chain == mct_infos[i].counter_chain;
            deps.push_back({DepType::RAW, j, i, same_header, w});
            llvm::outs() << mct_infos[j].task_name << " → "
                         << mct_infos[i].task_name << " : RAW on ";
            if (auto arg = dyn_cast<BlockArgument>(w)) {
              llvm::outs() << "func_arg" << arg.getArgNumber();
            } else {
              llvm::outs() << w;
            }
            if (same_header) {
              llvm::outs() << " [SAME HEADER - FUSION CANDIDATE]";
            }
            llvm::outs() << "\n";
          }
        }

        // Checks WAR: j reads, i writes same memref.
        for (Value r : mct_infos[j].source_memref_reads) {
          if (mct_infos[i].source_memref_writes.contains(r)) {
            bool same_header =
                mct_infos[j].counter_chain == mct_infos[i].counter_chain;
            deps.push_back({DepType::WAR, j, i, same_header, r});
            llvm::outs() << mct_infos[j].task_name << " → "
                         << mct_infos[i].task_name << " : WAR on ";
            if (auto arg = dyn_cast<BlockArgument>(r)) {
              llvm::outs() << "func_arg" << arg.getArgNumber();
            } else {
              llvm::outs() << r;
            }
            if (same_header) {
              llvm::outs() << " [SAME HEADER]";
            }
            llvm::outs() << "\n";
          }
        }

        // Checks WAW: j writes, i writes same memref.
        for (Value w : mct_infos[j].source_memref_writes) {
          if (mct_infos[i].source_memref_writes.contains(w)) {
            bool same_header =
                mct_infos[j].counter_chain == mct_infos[i].counter_chain;
            deps.push_back({DepType::WAW, j, i, same_header, w});
            llvm::outs() << mct_infos[j].task_name << " → "
                         << mct_infos[i].task_name << " : WAW on ";
            if (auto arg = dyn_cast<BlockArgument>(w)) {
              llvm::outs() << "func_arg" << arg.getArgNumber();
            } else {
              llvm::outs() << w;
            }
            if (same_header) {
              llvm::outs() << " [SAME HEADER]";
            }
            llvm::outs() << "\n";
          }
        }
      }
    }

    // Prints summary by type.
    size_t ssa_count = 0, raw_count = 0, war_count = 0, waw_count = 0;
    size_t fusion_candidates = 0;
    for (const auto &dep : deps) {
      switch (dep.type) {
      case DepType::SSA: ssa_count++; break;
      case DepType::RAW: raw_count++; break;
      case DepType::WAR: war_count++; break;
      case DepType::WAW: waw_count++; break;
      }
      // Only SSA and RAW are considered fusion candidates because they involve
      // data flow dependencies (producer outputs data that consumer needs).
      // WAR/WAW are ordering dependencies without data forwarding opportunity.
      if (dep.same_header && (dep.type == DepType::SSA || dep.type == DepType::RAW)) {
        fusion_candidates++;
      }
    }
    llvm::outs() << "\n=== Summary ===\n";
    llvm::outs() << "Total dependencies: " << deps.size() << "\n";
    llvm::outs() << "  SSA: " << ssa_count << ", RAW: " << raw_count
                 << ", WAR: " << war_count << ", WAW: " << waw_count << "\n";
    llvm::outs() << "Fusion candidates (same-header SSA/RAW): " << fusion_candidates
                 << "\n";
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//
struct AnalyzeMCTDependencyPass
    : public PassWrapper<AnalyzeMCTDependencyPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeMCTDependencyPass)

  StringRef getArgument() const override { return "analyze-mct-dependency"; }

  StringRef getDescription() const override {
    return "Analyzes dependencies between MCTs for multi-CGRA mapping.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MCTDependencyAnalyzer analyzer;
    analyzer.analyze(func);
  }
};

} // namespace

namespace mlir {
namespace taskflow {

std::unique_ptr<Pass> createAnalyzeMCTDependencyPass() {
  return std::make_unique<AnalyzeMCTDependencyPass>();
}

} // namespace taskflow
} // namespace mlir
