// HyperblockDependencyAnalysis.cpp - Implements hyperblock dependency analysis.

#include "TaskflowDialect/Analysis/HyperblockDependencyAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace mlir::taskflow;

void HyperblockDependencyGraph::buildFromTask(TaskflowTaskOp taskOp) {
  clear();

  // Collects all hyperblocks in program order.
  taskOp.getBody().walk([&](TaskflowHyperblockOp op) {
    hyperblocks_.push_back(op);
  });

  // Builds dependency edges between all pairs of hyperblocks.
  for (size_t i = 0; i < hyperblocks_.size(); ++i) {
    auto hbI = hyperblocks_[i];
    auto writesI = collectWrites(hbI);
    auto readsI = collectReads(hbI);

    for (size_t j = i + 1; j < hyperblocks_.size(); ++j) {
      auto hbJ = hyperblocks_[j];
      auto writesJ = collectWrites(hbJ);
      auto readsJ = collectReads(hbJ);

      // Checks RAW: I writes, J reads.
      for (Value memref : writesI) {
        if (readsJ.contains(memref)) {
          addEdge(hbI, hbJ, DependencyType::RAW, memref);
        }
      }

      // Checks WAR: I reads, J writes.
      for (Value memref : readsI) {
        if (writesJ.contains(memref)) {
          addEdge(hbI, hbJ, DependencyType::WAR, memref);
        }
      }

      // Checks WAW: I writes, J writes.
      for (Value memref : writesI) {
        if (writesJ.contains(memref)) {
          addEdge(hbI, hbJ, DependencyType::WAW, memref);
        }
      }
    }
  }
}

void HyperblockDependencyGraph::clear() {
  hyperblocks_.clear();
  predecessorEdges_.clear();
  successorEdges_.clear();
}

bool HyperblockDependencyGraph::hasDependency(
    TaskflowHyperblockOp source, TaskflowHyperblockOp target) const {
  auto it = successorEdges_.find(source);
  if (it == successorEdges_.end()) {
    return false;
  }
  for (const auto &edge : it->second) {
    if (edge.target == target) {
      return true;
    }
  }
  return false;
}

llvm::SmallVector<HyperblockDependencyEdge>
HyperblockDependencyGraph::getDependencies(TaskflowHyperblockOp source,
                                            TaskflowHyperblockOp target) const {
  llvm::SmallVector<HyperblockDependencyEdge> result;
  auto it = successorEdges_.find(source);
  if (it != successorEdges_.end()) {
    for (const auto &edge : it->second) {
      if (edge.target == target) {
        result.push_back(edge);
      }
    }
  }
  return result;
}

llvm::SmallVector<TaskflowHyperblockOp>
HyperblockDependencyGraph::getPredecessors(TaskflowHyperblockOp op) const {
  llvm::SmallVector<TaskflowHyperblockOp> result;
  llvm::DenseSet<TaskflowHyperblockOp> seen;

  auto it = predecessorEdges_.find(op);
  if (it != predecessorEdges_.end()) {
    for (const auto &edge : it->second) {
      if (!seen.contains(edge.source)) {
        seen.insert(edge.source);
        result.push_back(edge.source);
      }
    }
  }
  return result;
}

llvm::SmallVector<TaskflowHyperblockOp>
HyperblockDependencyGraph::getSuccessors(TaskflowHyperblockOp op) const {
  llvm::SmallVector<TaskflowHyperblockOp> result;
  llvm::DenseSet<TaskflowHyperblockOp> seen;

  auto it = successorEdges_.find(op);
  if (it != successorEdges_.end()) {
    for (const auto &edge : it->second) {
      if (!seen.contains(edge.target)) {
        seen.insert(edge.target);
        result.push_back(edge.target);
      }
    }
  }
  return result;
}

bool HyperblockDependencyGraph::canFuse(TaskflowHyperblockOp a,
                                         TaskflowHyperblockOp b) const {
  // Fusing two hyperblocks (A and B) is safe only if it does not violate 
  // intermediate dependencies. Specifically, if there is a block C between 
  // A and B in program order, we cannot fuse A and B if A -> C and C -> B.
  // Fusing A and B would effectively move B before C, breaking C -> B.

  // Finds positions in program order.
  int posA = -1, posB = -1;
  for (size_t i = 0; i < hyperblocks_.size(); ++i) {
    if (hyperblocks_[i] == a) posA = i;
    if (hyperblocks_[i] == b) posB = i;
  }

  if (posA < 0 || posB < 0) {
    return false;
  }

  // Ensures a comes before b for fusion (or they are adjacent).
  if (posA > posB) {
    std::swap(a, b);
    std::swap(posA, posB);
  }

  // Checks if there are any hyperblocks between a and b that depend on a
  // and b depends on them (would create cycle after fusion).
  for (size_t i = posA + 1; i < static_cast<size_t>(posB); ++i) {
    auto middle = hyperblocks_[i];
    if (hasDependency(a, middle) && hasDependency(middle, b)) {
      return false;  // Fusion would break dependency chain.
    }
  }

  return true;
}

bool HyperblockDependencyGraph::areCountersCompatible(
    TaskflowHyperblockOp a, TaskflowHyperblockOp b, int maxBoundDiff) const {
  auto indicesA = a.getIndices();
  auto indicesB = b.getIndices();

  // Requires same number of indices.
  if (indicesA.size() != indicesB.size()) {
    return false;
  }

  // Checks each counter pair.
  for (size_t i = 0; i < indicesA.size(); ++i) {
    auto counterA = indicesA[i].getDefiningOp<TaskflowCounterOp>();
    auto counterB = indicesB[i].getDefiningOp<TaskflowCounterOp>();

    if (!counterA || !counterB) {
      return false;
    }

    int64_t lowerA = counterA.getLowerBound().getSExtValue();
    int64_t upperA = counterA.getUpperBound().getSExtValue();
    int64_t stepA = counterA.getStep().getSExtValue();

    int64_t lowerB = counterB.getLowerBound().getSExtValue();
    int64_t upperB = counterB.getUpperBound().getSExtValue();
    int64_t stepB = counterB.getStep().getSExtValue();

    // Requires same lower bound and step.
    if (lowerA != lowerB || stepA != stepB) {
      return false;
    }

    // Checks upper bound difference.
    int diff = std::abs(static_cast<int>(upperA - upperB));
    if (diff > maxBoundDiff) {
      return false;
    }
  }

  return true;
}

llvm::DenseSet<Value>
HyperblockDependencyGraph::collectReads(TaskflowHyperblockOp op) const {
  llvm::DenseSet<Value> reads;
  op.getBody().walk([&](memref::LoadOp loadOp) {
    reads.insert(loadOp.getMemRef());
  });
  return reads;
}

llvm::DenseSet<Value>
HyperblockDependencyGraph::collectWrites(TaskflowHyperblockOp op) const {
  llvm::DenseSet<Value> writes;
  op.getBody().walk([&](memref::StoreOp storeOp) {
    writes.insert(storeOp.getMemRef());
  });
  return writes;
}

void HyperblockDependencyGraph::addEdge(TaskflowHyperblockOp source,
                                         TaskflowHyperblockOp target,
                                         DependencyType type, Value memref) {
  HyperblockDependencyEdge edge{source, target, type, memref};
  successorEdges_[source].push_back(edge);
  predecessorEdges_[target].push_back(edge);
}
