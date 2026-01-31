///===- PartitionTaskByTarget.cpp - Partition tasks by hardware target --===//
//
// This pass analyzes taskflow.channel operations and annotates cross-boundary
// channels (channels connecting tasks on different hardware targets).
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Get the target hardware of a task operation
static StringRef getTaskTarget(TaskflowTaskOp taskOp) {
  if (auto targetAttr = taskOp->getAttrOfType<StringAttr>("target")) {
    return targetAttr.getValue();
  }
  return "CPU"; // Default target
}

/// Check if a value is produced by a TaskflowTaskOp
static TaskflowTaskOp getProducerTask(Value value) {
  if (auto taskOp = value.getDefiningOp<TaskflowTaskOp>()) {
    return taskOp;
  }
  // Handle block arguments (function parameters)
  return nullptr;
}

/// Get all consumer tasks of a value
static void getConsumerTasks(Value value, 
                            SmallVectorImpl<TaskflowTaskOp> &consumers) {
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    
    // Direct consumer
    if (auto taskOp = dyn_cast<TaskflowTaskOp>(owner)) {
      consumers.push_back(taskOp);
    }
    // Through channel
    else if (auto channelOp = dyn_cast<TaskflowChannelOp>(owner)) {
      getConsumerTasks(channelOp.getTarget(), consumers);
    }
  }
}

//===----------------------------------------------------------------------===//
// PartitionTaskByTarget Pass
//===----------------------------------------------------------------------===//

struct PartitionTaskByTargetPass
    : public PassWrapper<PartitionTaskByTargetPass,
                        OperationPass<func::FuncOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartitionTaskByTargetPass)

  StringRef getArgument() const final { return "partition-taskflow-by-target"; }
  
  StringRef getDescription() const final {
    return "Annotate cross-boundary channels in taskflow graph";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(&getContext());
    
    // Statistics
    unsigned totalChannels = 0;
    unsigned crossBoundaryChannels = 0;
    DenseMap<std::pair<StringRef, StringRef>, unsigned> transferStats;
    
    llvm::errs() << "\n";
    llvm::errs() << "========================================\n";
    llvm::errs() << "PartitionTaskByTarget Pass\n";
    llvm::errs() << "========================================\n";
    llvm::errs() << "Function: " << func.getName() << "\n\n";
    
    // Step 1: Collect all tasks and their targets
    SmallVector<TaskflowTaskOp> tasks;
    func.walk([&](TaskflowTaskOp taskOp) {
      tasks.push_back(taskOp);
      StringRef target = getTaskTarget(taskOp);
      llvm::errs() << "  Task: " << taskOp.getTaskName() 
                   << " -> " << target << "\n";
    });
    
    llvm::errs() << "\nTotal tasks: " << tasks.size() << "\n\n";
    
    // Step 2: Process all channels
    llvm::errs() << "Analyzing channels:\n";
    llvm::errs() << "----------------------------------------\n";
    
    func.walk([&](TaskflowChannelOp channelOp) {
      totalChannels++;
      
      Value source = channelOp.getSource();
      
      // Get producer task
      TaskflowTaskOp producerTask = getProducerTask(source);
      if (!producerTask) {
        llvm::errs() << "  Channel #" << totalChannels 
                     << ": skipped (no producer task)\n";
        return;
      }
      
      StringRef producerTarget = getTaskTarget(producerTask);
      
      // Get consumer tasks
      SmallVector<TaskflowTaskOp> consumerTasks;
      getConsumerTasks(channelOp.getTarget(), consumerTasks);
      
      if (consumerTasks.empty()) {
        llvm::errs() << "  Channel #" << totalChannels 
                     << ": " << producerTarget 
                     << " -> (no consumers)\n";
        return;
      }
      
      // Check all consumers
      bool isCrossBoundary = false;
      StringRef consumerTarget;
      
      for (auto consumerTask : consumerTasks) {
        consumerTarget = getTaskTarget(consumerTask);
        
        if (producerTarget != consumerTarget) {
          isCrossBoundary = true;
          
          // Annotate the channel
          channelOp->setAttr("cross_boundary", 
                            builder.getUnitAttr());
          channelOp->setAttr("from", 
                            builder.getStringAttr(producerTarget));
          channelOp->setAttr("to", 
                            builder.getStringAttr(consumerTarget));
          
          crossBoundaryChannels++;
          transferStats[{producerTarget, consumerTarget}]++;
          
          llvm::errs() << "  Channel #" << totalChannels << ": "
                       << producerTask.getTaskName() << " (" << producerTarget 
                       << ") -> "
                       << consumerTask.getTaskName() << " (" << consumerTarget 
                       << ") [CROSS-BOUNDARY]\n";
          
          break; // Only need to annotate once
        }
      }
      
      if (!isCrossBoundary) {
        llvm::errs() << "  Channel #" << totalChannels << ": "
                     << producerTarget << " -> " << producerTarget 
                     << " [same target]\n";
      }
    });
    
    // Step 3: Print summary
    llvm::errs() << "\n";
    llvm::errs() << "========================================\n";
    llvm::errs() << "Summary\n";
    llvm::errs() << "========================================\n";
    llvm::errs() << "Total channels:          " << totalChannels << "\n";
    llvm::errs() << "Cross-boundary channels: " << crossBoundaryChannels << "\n";
    llvm::errs() << "Same-target channels:    " 
                 << (totalChannels - crossBoundaryChannels) << "\n";
    
    if (!transferStats.empty()) {
      llvm::errs() << "\nCross-boundary transfer breakdown:\n";
      for (auto &entry : transferStats) {
        llvm::errs() << "  " << entry.first.first << " -> " 
                     << entry.first.second << ": " 
                     << entry.second << " transfer(s)\n";
      }
    }
    
    llvm::errs() << "========================================\n\n";
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace taskflow {

std::unique_ptr<Pass> createPartitionTaskByTargetPass() {
  return std::make_unique<PartitionTaskByTargetPass>();
}

} // namespace taskflow
} // namespace mlir
