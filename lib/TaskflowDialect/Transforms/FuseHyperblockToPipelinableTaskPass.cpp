#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {
//---------------------------------------------------------------------------
// Memory Access Analysis
//----------------------------------------------------------------------------
struct MemoryAccessInfo {
  SetVector<Value> read_memrefs;
  SetVector<Value> write_memrefs;
  SmallVector<Value> counter_indices;
};

//---------------------------------------------------------------------------
// Hyperblocks Grouping
//----------------------------------------------------------------------------

struct HyperblockGroup {
  SmallVector<TaskflowHyperblockOp> hyperblocks;
  SmallVector<Value> shared_indices;
  SetVector<Value> all_read_memrefs;
  SetVector<Value> all_write_memrefs;

  void addHyperblock(TaskflowHyperblockOp hb_op,
                     const MemoryAccessInfo &mem_info) {
    this->hyperblocks.push_back(hb_op);
    if (this->shared_indices.empty()) {
      this->shared_indices = mem_info.counter_indices;
    }
    all_read_memrefs.insert(mem_info.read_memrefs.begin(),
                            mem_info.read_memrefs.end());
    all_write_memrefs.insert(mem_info.write_memrefs.begin(),
                             mem_info.write_memrefs.end());
  }

  bool canAddHyperblock(const MemoryAccessInfo &mem_info) const {}
};

// Groups hyperblocks that can be fused together.
static SmallVector<HyperblockGroup>
groupHyperblocks(SmallVector<TaskflowHyperblockOp> &hyperblocks) {
  SmallVector<HyperblockGroup> groups;
  DenseMap<TaskflowHyperblockOp, MemoryAccessInfo> hb_to_meminfo_map;
}

struct FuseHyperblockToPipelinableTaskPass
    : public PassWrapper<FuseHyperblockToPipelinableTaskPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      FuseHyperblockToPipelinableTaskPass)

  StringRef getArgument() const final {
    return "fuse-hyperblock-to-pipelinable-task";
  }

  StringRef getDescription() const final {
    return "Conservative hyperblock fusion: split hyperblocks with memory "
           "access dependencies into different tasks, ensure each task has "
           "exactly one hyperblock";
  };

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<taskflow::TaskflowDialect, arith::ArithDialect,
                    memref::MemRefDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func_op = getOperation();

    // Collects all tasks.
    SmallVector<TaskflowTaskOp> tasks;
    func_op.walk([&](TaskflowTaskOp task_op) { tasks.push_back(task_op); });

    // Process each task.
    for (TaskflowTaskOp task_op : tasks) {
      SmallVector<TaskflowHyperblockOp> hyperblocks;
      task_op.walk(
          [&](TaskflowHyperblockOp hb_op) { hyperblocks.push_back(hb_op); });

      llvm::errs() << "Found " << hyperblocks.size() << " hyperblocks in task "
                   << task_op.getTaskName() << "\n";
      if (hyperblocks.size() <= 1) {
        llvm::errs() << "Task already has <=1 hyperblock, skip.\n";
        continue;
      }

      // Group hyperblocks that can be fused together (Do not have memory access
      // dependencies).
      auto hyperblock_groups = groupHyperblocks(hyperblocks);
    }
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::taskflow::createFuseHyperblockToPipelinableTaskPass() {
  return std::make_unique<FuseHyperblockToPipelinableTaskPass>();
}