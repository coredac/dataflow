#ifndef CONVERSION_AFFINE_TO_NEURA_LOOP_NEST_ANALYSIS_H
#define CONVERSION_AFFINE_TO_NEURA_LOOP_NEST_ANALYSIS_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace mlir {
namespace neura {

/// Loop information structure - Stores all analysis information for a single loop.
struct LoopInfo {
  affine::AffineForOp loop;              // The loop operation itself.
  LoopInfo *parent = nullptr;            // Parent loop (nullptr if top-level).
  llvm::SmallVector<LoopInfo *, 4> children;  // Child loops list.
  unsigned depth = 0;                    // Nesting depth (0=top-level).
  bool is_perfect_nest = true;           // Whether it is a perfect nest.
  
  // Operations list for imperfect nesting.
  llvm::SmallVector<Operation *, 4> operations_before_child;  // Operations before child loops.
  llvm::SmallVector<Operation *, 4> operations_after_child;   // Operations after child loops.
  
  LoopInfo(affine::AffineForOp loop) : loop(loop) {}
};

/// Loop nest analysis class.
/// 
/// Purpose: Provides loop hierarchy information for AffineToNeura pass to support optimization decisions.
/// 
/// Usage example:
///   LoopNestAnalysis analysis(func_op);
///   analysis.dump();  // Prints analysis results.
///   LoopInfo *info = analysis.getLoopInfo(loop);
///   if (info && info->parent) {
///     // This is a nested loop, can reuse parent's valid signal.
///   }
class LoopNestAnalysis {
public:
  /// Constructor - Performs loop nest analysis on the given function.
  explicit LoopNestAnalysis(func::FuncOp func);
  
  /// Query interfaces.
  LoopInfo *getLoopInfo(affine::AffineForOp loop) const;  // Gets loop information.
  llvm::ArrayRef<LoopInfo *> getTopLevelLoops() const { return topLevelLoops; }  // Gets top-level loops.
  llvm::ArrayRef<std::unique_ptr<LoopInfo>> getAllLoops() const { return allLoops; }  // Gets all loops.
  bool isPerfectNest(affine::AffineForOp loop) const;  // Checks if perfect nest.
  LoopInfo *getParentLoop(affine::AffineForOp loop) const;  // Gets parent loop.
  llvm::ArrayRef<LoopInfo *> getChildLoops(affine::AffineForOp loop) const;  // Gets child loops.
  
  /// Debug interface - Prints analysis results.
  void dump() const;

private:
  /// Internal analysis methods.
  void buildLoopNestTree(func::FuncOp func);  // Builds loop hierarchy tree.
  void analyzePerfectNests();  // Analyzes perfect nest characteristics.
  
  /// Data members.
  llvm::DenseMap<Operation *, LoopInfo *> loopMap;  // Loop fast lookup table.
  llvm::SmallVector<std::unique_ptr<LoopInfo>, 8> allLoops;  // All loops (owns ownership).
  llvm::SmallVector<LoopInfo *, 4> topLevelLoops;  // Top-level loop pointers list.
};

} // namespace neura
} // namespace mlir

#endif
