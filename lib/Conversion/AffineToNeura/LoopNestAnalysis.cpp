#include "Conversion/AffineToNeura/LoopNestAnalysis.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::neura;

/// Constructor - Performs complete loop nest analysis.
LoopNestAnalysis::LoopNestAnalysis(func::FuncOp func) {
  llvm::errs() << "[LoopNestAnalysis] Starting analysis for function: " 
               << func.getName() << "\n";
  buildLoopNestTree(func);
  llvm::errs() << "[LoopNestAnalysis] Found " << allLoops.size() << " loops\n";
  analyzePerfectNests();
  llvm::errs() << "[LoopNestAnalysis] Analysis complete\n";
}

// Builds the loop hierarchy tree.
void LoopNestAnalysis::buildLoopNestTree(func::FuncOp func) {
  // Step 1: Collects all loops.
  func.walk([&](affine::AffineForOp loop) {
    auto loopInfo = std::make_unique<LoopInfo>(loop);
    loopMap[loop.getOperation()] = loopInfo.get();
    allLoops.push_back(std::move(loopInfo));
  });
  
  // Step 2: Establishes parent-child relationships.
  for (auto &loopInfoPtr : allLoops) {
    LoopInfo *loopInfo = loopInfoPtr.get();
    affine::AffineForOp loop = loopInfo->loop;
    
    // Searches upward for parent loop.
    Operation *parentOp = loop->getParentOp();
    while (parentOp && !isa<func::FuncOp>(parentOp)) {
      if (auto parentLoop = dyn_cast<affine::AffineForOp>(parentOp)) {
        auto it = loopMap.find(parentLoop.getOperation());
        if (it != loopMap.end()) {
          loopInfo->parent = it->second;
          loopInfo->depth = loopInfo->parent->depth + 1;  // depth = parent_depth + 1
          it->second->children.push_back(loopInfo);
        }
        break;
      }
      parentOp = parentOp->getParentOp();
    }
    
    // If no parent loop, this is a top-level loop.
    if (!loopInfo->parent) {
      topLevelLoops.push_back(loopInfo);
    }
  }
}

// Analyzes perfect nesting characteristics.
void LoopNestAnalysis::analyzePerfectNests() {
  for (auto &loopInfoPtr : allLoops) {
    LoopInfo *info = loopInfoPtr.get();
    
    // Leaf loops are automatically perfect.
    if (info->children.empty()) {
      info->isPerfectNest = true;
      continue;
    }
    
    Block &body = info->loop.getRegion().front();
    
    // Builds child loop operation set for fast lookup.
    llvm::DenseSet<Operation *> childLoopOps;
    for (LoopInfo *child : info->children) {
      childLoopOps.insert(child->loop.getOperation());
    }
    
    Operation *firstChild = info->children.front()->loop.getOperation();
    Operation *lastChild = info->children.back()->loop.getOperation();
    
    // Checks if operations exist before the first child loop.
    for (Operation &op : body.getOperations()) {
      if (&op == firstChild) break;
      if (isa<affine::AffineYieldOp>(&op)) continue;
      info->operationsBeforeChild.push_back(&op);
      info->isPerfectNest = false;  // Operations before child → imperfect
    }
    
    // Checks if operations exist after the last child loop.
    bool afterLastChild = false;
    for (Operation &op : body.getOperations()) {
      if (&op == lastChild) {
        afterLastChild = true;
        continue;
      }
      if (afterLastChild && !isa<affine::AffineYieldOp>(&op)) {
        info->operationsAfterChild.push_back(&op);
        info->isPerfectNest = false;  // Operations after child → imperfect
      }
    }
    
    // Checks if operations exist between sibling child loops.
    // Example: affine.for i { affine.for j1; op; affine.for j2 }
    if (info->children.size() > 1) {
      bool betweenChildren = false;
      Operation *prevChild = nullptr;
      
      for (Operation &op : body.getOperations()) {
        if (childLoopOps.contains(&op)) {
          if (prevChild && betweenChildren) {
            info->isPerfectNest = false;  // Operations between siblings → imperfect
            break;
          }
          prevChild = &op;
          betweenChildren = false;
        } else if (prevChild && !isa<affine::AffineYieldOp>(&op)) {
          betweenChildren = true;
        }
      }
    }
  }
}


// Query Interface Implementation

// Queries LoopInfo by loop operation.
LoopInfo *LoopNestAnalysis::getLoopInfo(affine::AffineForOp loop) const {
  auto it = loopMap.find(loop.getOperation());
  return it != loopMap.end() ? it->second : nullptr;
}

// Checks if the loop is a perfect nest.
bool LoopNestAnalysis::isPerfectNest(affine::AffineForOp loop) const {
  LoopInfo *info = getLoopInfo(loop);
  return info ? info->isPerfectNest : false;
}

// Gets the parent loop.
LoopInfo *LoopNestAnalysis::getParentLoop(affine::AffineForOp loop) const {
  LoopInfo *info = getLoopInfo(loop);
  return info ? info->parent : nullptr;
}

// Gets the list of child loops.
llvm::ArrayRef<LoopInfo *> 
LoopNestAnalysis::getChildLoops(affine::AffineForOp loop) const {
  LoopInfo *info = getLoopInfo(loop);
  return info ? llvm::ArrayRef<LoopInfo *>(info->children) 
              : llvm::ArrayRef<LoopInfo *>();
}


// Debug Output Implementation
void LoopNestAnalysis::dump() const {
  llvm::errs() << "=== Loop Nest Analysis ===\n";
  llvm::errs() << "Total loops: " << allLoops.size() << "\n";
  llvm::errs() << "Top-level loops: " << topLevelLoops.size() << "\n\n";
  
  // Recursive print function.
  std::function<void(LoopInfo *, unsigned)> printLoop;
  printLoop = [&](LoopInfo *info, unsigned indent) {
    // Prints indentation.
    for (unsigned i = 0; i < indent; ++i) llvm::errs() << "  ";
    
    // Prints basic loop information.
    llvm::errs() << "Loop (depth=" << info->depth 
                 << ", perfect=" << (info->isPerfectNest ? "yes" : "no")
                 << ", children=" << info->children.size() << ")";
    
    // If imperfect nest, prints detailed information.
    if (!info->isPerfectNest) {
      llvm::errs() << " [IMPERFECT: "
                   << "ops_before=" << info->operationsBeforeChild.size()
                   << ", ops_after=" << info->operationsAfterChild.size()
                   << "]";
    }
    llvm::errs() << "\n";
    
    // Prints location information.
    for (unsigned i = 0; i < indent; ++i) llvm::errs() << "  ";
    llvm::errs() << "  at: ";
    info->loop.getLoc().print(llvm::errs());
    llvm::errs() << "\n";
    
    // Recursively prints child loops.
    for (LoopInfo *child : info->children) {
      printLoop(child, indent + 1);
    }
  };
  
  for (LoopInfo *topLoop : topLevelLoops) {
    printLoop(topLoop, 0);
  }
  
  llvm::errs() << "=== End Loop Nest Analysis ===\n\n";
}
