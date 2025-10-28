//===- LoopNestAnalysis.h - Analyze affine loop nests ----------*- C++ -*-===//
//
// 循环嵌套分析 - 用于分析affine循环的层次结构和完美嵌套特性
// 
// 功能：
// 1. 构建循环层次树（父子关系、嵌套深度）
// 2. 识别完美嵌套 vs 非完美嵌套
// 3. 支持循环valid信号重用优化
//
//===----------------------------------------------------------------------===//
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

/// 循环信息结构体 - 存储单个循环的所有分析信息
struct LoopInfo {
  affine::AffineForOp loop;              // 循环操作本身
  LoopInfo *parent = nullptr;            // 父循环（若为nullptr则是顶层循环）
  llvm::SmallVector<LoopInfo *, 4> children;  // 子循环列表
  unsigned depth = 0;                    // 嵌套深度（0=顶层）
  bool isPerfectNest = true;             // 是否为完美嵌套
  
  // 非完美嵌套的操作列表
  llvm::SmallVector<Operation *, 4> operationsBeforeChild;  // 子循环前的操作
  llvm::SmallVector<Operation *, 4> operationsAfterChild;   // 子循环后的操作
  
  LoopInfo(affine::AffineForOp loop) : loop(loop) {}
};

/// 循环嵌套分析类
/// 
/// 用途：为AffineToNeura pass提供循环层次结构信息，支持优化决策
/// 
/// 使用示例：
///   LoopNestAnalysis analysis(func_op);
///   analysis.dump();  // 打印分析结果
///   LoopInfo *info = analysis.getLoopInfo(loop);
///   if (info && info->parent) {
///     // 这是嵌套循环，可以重用父循环的valid信号
///   }
class LoopNestAnalysis {
public:
  /// 构造函数 - 对给定函数进行循环嵌套分析
  explicit LoopNestAnalysis(func::FuncOp func);
  
  /// 查询接口
  LoopInfo *getLoopInfo(affine::AffineForOp loop) const;  // 获取循环信息
  llvm::ArrayRef<LoopInfo *> getTopLevelLoops() const { return topLevelLoops; }  // 获取顶层循环
  llvm::ArrayRef<std::unique_ptr<LoopInfo>> getAllLoops() const { return allLoops; }  // 获取所有循环
  bool isPerfectNest(affine::AffineForOp loop) const;  // 检查是否完美嵌套
  LoopInfo *getParentLoop(affine::AffineForOp loop) const;  // 获取父循环
  llvm::ArrayRef<LoopInfo *> getChildLoops(affine::AffineForOp loop) const;  // 获取子循环
  
  /// 调试接口 - 打印分析结果
  void dump() const;

private:
  /// 内部分析方法
  void buildLoopNestTree(func::FuncOp func);  // 构建循环层次树
  void analyzePerfectNests();  // 分析完美嵌套特性
  
  /// 数据成员
  llvm::DenseMap<Operation *, LoopInfo *> loopMap;  // 循环快速查找表
  llvm::SmallVector<std::unique_ptr<LoopInfo>, 8> allLoops;  // 所有循环（拥有所有权）
  llvm::SmallVector<LoopInfo *, 4> topLevelLoops;  // 顶层循环指针列表
};

} // namespace neura
} // namespace mlir

#endif
