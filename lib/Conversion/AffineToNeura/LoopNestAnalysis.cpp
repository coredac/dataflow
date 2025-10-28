#include "Conversion/AffineToNeura/LoopNestAnalysis.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::neura;

//===----------------------------------------------------------------------===//
// LoopNestAnalysis 实现
//===----------------------------------------------------------------------===//

/// 构造函数 - 执行完整的循环嵌套分析
LoopNestAnalysis::LoopNestAnalysis(func::FuncOp func) {
  llvm::errs() << "[LoopNestAnalysis] Starting analysis for function: " 
               << func.getName() << "\n";
  buildLoopNestTree(func);
  llvm::errs() << "[LoopNestAnalysis] Found " << allLoops.size() << " loops\n";
  analyzePerfectNests();
  llvm::errs() << "[LoopNestAnalysis] Analysis complete\n";
}

/// 构建循环层次树
/// 
/// 步骤1: 遍历所有循环，创建LoopInfo对象
/// 步骤2: 建立父子关系，计算嵌套深度
void LoopNestAnalysis::buildLoopNestTree(func::FuncOp func) {
  // 步骤1: 收集所有循环
  func.walk([&](affine::AffineForOp loop) {
    auto loopInfo = std::make_unique<LoopInfo>(loop);
    loopMap[loop.getOperation()] = loopInfo.get();
    allLoops.push_back(std::move(loopInfo));
  });
  
  // 步骤2: 建立父子关系
  for (auto &loopInfoPtr : allLoops) {
    LoopInfo *loopInfo = loopInfoPtr.get();
    affine::AffineForOp loop = loopInfo->loop;
    
    // 向上查找父循环
    Operation *parentOp = loop->getParentOp();
    while (parentOp && !isa<func::FuncOp>(parentOp)) {
      if (auto parentLoop = dyn_cast<affine::AffineForOp>(parentOp)) {
        auto it = loopMap.find(parentLoop.getOperation());
        if (it != loopMap.end()) {
          loopInfo->parent = it->second;
          loopInfo->depth = loopInfo->parent->depth + 1;  // 深度 = 父深度 + 1
          it->second->children.push_back(loopInfo);
        }
        break;
      }
      parentOp = parentOp->getParentOp();
    }
    
    // 如果没有父循环，则为顶层循环
    if (!loopInfo->parent) {
      topLevelLoops.push_back(loopInfo);
    }
  }
}

/// 分析完美嵌套特性
/// 
/// 完美嵌套定义：
/// - 叶子循环（无子循环）自动是完美嵌套
/// - 非叶子循环：子循环前后不能有其他操作（除了yield）
/// 
/// 非完美嵌套示例：
///   affine.for %i {
///     %x = arith.constant 0  // <- 这个操作使得嵌套不完美
///     affine.for %j { ... }
///   }
void LoopNestAnalysis::analyzePerfectNests() {
  for (auto &loopInfoPtr : allLoops) {
    LoopInfo *info = loopInfoPtr.get();
    
    // 叶子循环自动是完美嵌套
    if (info->children.empty()) {
      info->isPerfectNest = true;
      continue;
    }
    
    Block &body = info->loop.getRegion().front();
    
    // 构建子循环操作集合，用于快速查找
    llvm::DenseSet<Operation *> childLoopOps;
    for (LoopInfo *child : info->children) {
      childLoopOps.insert(child->loop.getOperation());
    }
    
    Operation *firstChild = info->children.front()->loop.getOperation();
    Operation *lastChild = info->children.back()->loop.getOperation();
    
    // 检查第一个子循环之前是否有操作
    for (Operation &op : body.getOperations()) {
      if (&op == firstChild) break;
      if (isa<affine::AffineYieldOp>(&op)) continue;
      info->operationsBeforeChild.push_back(&op);
      info->isPerfectNest = false;  // 有操作在子循环前 → 非完美嵌套
    }
    
    // 检查最后一个子循环之后是否有操作
    bool afterLastChild = false;
    for (Operation &op : body.getOperations()) {
      if (&op == lastChild) {
        afterLastChild = true;
        continue;
      }
      if (afterLastChild && !isa<affine::AffineYieldOp>(&op)) {
        info->operationsAfterChild.push_back(&op);
        info->isPerfectNest = false;  // 有操作在子循环后 → 非完美嵌套
      }
    }
    
    // 检查兄弟子循环之间是否有操作
    // 示例：affine.for i { affine.for j1; op; affine.for j2 }
    if (info->children.size() > 1) {
      bool betweenChildren = false;
      Operation *prevChild = nullptr;
      
      for (Operation &op : body.getOperations()) {
        if (childLoopOps.contains(&op)) {
          if (prevChild && betweenChildren) {
            info->isPerfectNest = false;  // 兄弟循环之间有操作 → 非完美嵌套
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

//===----------------------------------------------------------------------===//
// 查询接口实现
//===----------------------------------------------------------------------===//

/// 通过循环操作查询LoopInfo
LoopInfo *LoopNestAnalysis::getLoopInfo(affine::AffineForOp loop) const {
  auto it = loopMap.find(loop.getOperation());
  return it != loopMap.end() ? it->second : nullptr;
}

/// 检查循环是否为完美嵌套
bool LoopNestAnalysis::isPerfectNest(affine::AffineForOp loop) const {
  LoopInfo *info = getLoopInfo(loop);
  return info ? info->isPerfectNest : false;
}

/// 获取父循环
LoopInfo *LoopNestAnalysis::getParentLoop(affine::AffineForOp loop) const {
  LoopInfo *info = getLoopInfo(loop);
  return info ? info->parent : nullptr;
}

/// 获取子循环列表
llvm::ArrayRef<LoopInfo *> 
LoopNestAnalysis::getChildLoops(affine::AffineForOp loop) const {
  LoopInfo *info = getLoopInfo(loop);
  return info ? llvm::ArrayRef<LoopInfo *>(info->children) 
              : llvm::ArrayRef<LoopInfo *>();
}

//===----------------------------------------------------------------------===//
// 调试输出实现
//===----------------------------------------------------------------------===//

/// 打印分析结果（用于调试和验证）
/// 
/// 输出格式：
///   === Loop Nest Analysis ===
///   Total loops: 3
///   Top-level loops: 1
///   
///   Loop (depth=0, perfect=yes, children=2)
///     at: loc(...)
///     Loop (depth=1, perfect=yes, children=0)
///       at: loc(...)
void LoopNestAnalysis::dump() const {
  llvm::errs() << "=== Loop Nest Analysis ===\n";
  llvm::errs() << "Total loops: " << allLoops.size() << "\n";
  llvm::errs() << "Top-level loops: " << topLevelLoops.size() << "\n\n";
  
  // 递归打印函数
  std::function<void(LoopInfo *, unsigned)> printLoop;
  printLoop = [&](LoopInfo *info, unsigned indent) {
    // 打印缩进
    for (unsigned i = 0; i < indent; ++i) llvm::errs() << "  ";
    
    // 打印循环基本信息
    llvm::errs() << "Loop (depth=" << info->depth 
                 << ", perfect=" << (info->isPerfectNest ? "yes" : "no")
                 << ", children=" << info->children.size() << ")";
    
    // 如果是非完美嵌套，打印详细信息
    if (!info->isPerfectNest) {
      llvm::errs() << " [IMPERFECT: "
                   << "ops_before=" << info->operationsBeforeChild.size()
                   << ", ops_after=" << info->operationsAfterChild.size()
                   << "]";
    }
    llvm::errs() << "\n";
    
    // 打印位置信息
    for (unsigned i = 0; i < indent; ++i) llvm::errs() << "  ";
    llvm::errs() << "  at: ";
    info->loop.getLoc().print(llvm::errs());
    llvm::errs() << "\n";
    
    // 递归打印子循环
    for (LoopInfo *child : info->children) {
      printLoop(child, indent + 1);
    }
  };
  
  for (LoopInfo *topLoop : topLevelLoops) {
    printLoop(topLoop, 0);
  }
  
  llvm::errs() << "=== End Loop Nest Analysis ===\n\n";
}
