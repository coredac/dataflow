// HyperblockDependencyAnalysis.h - Analyzes dependencies between hyperblocks.
//
// This file provides utilities for analyzing data dependencies between
// hyperblocks within a Taskflow task.

#ifndef TASKFLOW_ANALYSIS_HYPERBLOCK_DEPENDENCY_ANALYSIS_H
#define TASKFLOW_ANALYSIS_HYPERBLOCK_DEPENDENCY_ANALYSIS_H

#include "TaskflowDialect/TaskflowOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace taskflow {

/// Represents the type of data dependency between hyperblocks.
enum class DependencyType {
  None,
  RAW,  // Read-After-Write.
  WAR,  // Write-After-Read.
  WAW   // Write-After-Write.
};

/// Represents a dependency edge between two hyperblocks.
struct HyperblockDependencyEdge {
  TaskflowHyperblockOp source;
  TaskflowHyperblockOp target;
  DependencyType type;
  Value memref;  // The memory location causing the dependency.
};

/// Analyzes dependencies between hyperblocks within a task.
class HyperblockDependencyGraph {
public:
  /// Builds the dependency graph from a task operation.
  void buildFromTask(TaskflowTaskOp taskOp);

  /// Clears all stored dependency information.
  void clear();

  /// Returns true if there is any dependency from source to target.
  bool hasDependency(TaskflowHyperblockOp source,
                     TaskflowHyperblockOp target) const;

  /// Returns all dependencies from source to target.
  llvm::SmallVector<HyperblockDependencyEdge>
  getDependencies(TaskflowHyperblockOp source,
                  TaskflowHyperblockOp target) const;

  /// Returns all predecessors of a hyperblock (hyperblocks it depends on).
  llvm::SmallVector<TaskflowHyperblockOp>
  getPredecessors(TaskflowHyperblockOp op) const;

  /// Returns all successors of a hyperblock (hyperblocks that depend on it).
  llvm::SmallVector<TaskflowHyperblockOp>
  getSuccessors(TaskflowHyperblockOp op) const;

  /// Checks if two hyperblocks can be fused without creating circular deps.
  bool canFuse(TaskflowHyperblockOp a, TaskflowHyperblockOp b) const;

  /// Checks if two hyperblocks have compatible counter structures.
  bool areCountersCompatible(TaskflowHyperblockOp a, TaskflowHyperblockOp b,
                             int maxBoundDiff) const;

  /// Returns all hyperblocks in the analyzed task.
  const llvm::SmallVector<TaskflowHyperblockOp> &getHyperblocks() const {
    return hyperblocks_;
  }

private:
  /// Collects memory reads from a hyperblock.
  llvm::DenseSet<Value> collectReads(TaskflowHyperblockOp op) const;

  /// Collects memory writes from a hyperblock.
  llvm::DenseSet<Value> collectWrites(TaskflowHyperblockOp op) const;

  /// Adds a dependency edge to the graph.
  void addEdge(TaskflowHyperblockOp source, TaskflowHyperblockOp target,
               DependencyType type, Value memref);

  /// All hyperblocks in program order.
  llvm::SmallVector<TaskflowHyperblockOp> hyperblocks_;

  /// Maps each hyperblock to its predecessor edges.
  llvm::DenseMap<TaskflowHyperblockOp,
                 llvm::SmallVector<HyperblockDependencyEdge>>
      predecessorEdges_;

  /// Maps each hyperblock to its successor edges.
  llvm::DenseMap<TaskflowHyperblockOp,
                 llvm::SmallVector<HyperblockDependencyEdge>>
      successorEdges_;
};

} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_ANALYSIS_HYPERBLOCK_DEPENDENCY_ANALYSIS_H
