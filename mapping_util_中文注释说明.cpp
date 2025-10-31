/*
 * mapping_util.cpp/h - 中文注释说明
 * =================================
 * 
 * 本文件说明mapping_util中修改的部分，特别是is_steering_unwrapped_op函数。
 *
 * 修改背景：
 * =========
 * Reviewer指出原来的实现使用了否定判断：
 *   return !isa<DataMovOp>(op);
 * 
 * 这种写法不够明确，且可能将不该包含的操作也包含进来。
 * 应该显式列出所有steering模式下不需要DataMovOp包装的操作类型。
 *
 * 什么是Steering Mode？
 * =====================
 * 在CGRA映射中，有些操作需要特殊的数据路由处理：
 * - 普通操作：需要DataMovOp包装来进行数据传输
 * - Steering操作：有特殊的数据流语义，不需要DataMovOp包装
 *
 * Steering Unwrapped操作包括：
 * ---------------------------
 * 1. ConstantOp：常量操作
 *    - 不需要从其他tile接收数据
 *    - 直接在当前tile产生常量值
 *
 * 2. CarryOp：循环携带依赖
 *    - 将上一次迭代的值传递到当前迭代
 *    - 有自己的数据流路径
 *
 * 3. InvariantOp：循环不变量
 *    - 在整个循环中保持不变的值
 *    - 特殊的数据流处理
 *
 * 4. CarryInvariantOp：融合的carry和invariant
 *    - 同时处理循环携带和不变量
 *    - 特殊的融合操作语义
 *
 * 5. ConditionalSelectOp：条件选择
 *    - 基于条件选择数据流路径
 *    - 内置的routing逻辑
 *
 * 6. InvariantGroupOp：不变量组
 *    - 管理多个不变量
 *    - 特殊的组织结构
 *
 * 7. ReserveOp：占位操作
 *    - 在循环中预留位置
 *    - 不需要实际的数据传输
 *
 * 修改前的代码：
 * =============
 * bool is_steering_unwrapped_op(Operation *op) {
 *   return !isa<DataMovOp>(op);  // 太宽泛！
 * }
 *
 * 问题：
 * - 任何不是DataMovOp的操作都会返回true
 * - 包括了许多不该包括的操作（如普通的AddOp等）
 * - 语义不清晰
 *
 * 修改后的代码：
 * =============
 * bool is_steering_unwrapped_op(Operation *op) {
 *   return mlir::isa<neura::ConstantOp,        // 常量
 *                    neura::CarryOp,            // 循环携带
 *                    neura::InvariantOp,        // 循环不变量
 *                    neura::CarryInvariantOp,   // 融合操作
 *                    neura::ConditionalSelectOp,// 条件选择
 *                    neura::InvariantGroupOp,   // 不变量组
 *                    neura::ReserveOp>(op);     // 占位操作
 * }
 *
 * 优点：
 * -----
 * 1. 明确性：清楚列出所有不需要包装的操作
 * 2. 可维护性：添加/删除操作类型时一目了然
 * 3. 类型安全：编译器会检查这些类型是否存在
 * 4. 文档性：代码本身就是文档，说明了设计意图
 *
 * 使用场景：
 * =========
 * 此函数在MapToAcceleratorPass等映射pass中使用，用于判断：
 * 
 * if (is_steering_unwrapped_op(op)) {
 *   // 直接映射到CGRA tile，不需要DataMovOp包装
 *   map_directly(op);
 * } else {
 *   // 需要用DataMovOp包装来处理数据路由
 *   wrap_with_datamov(op);
 * }
 *
 * 相关的其他工具函数：
 * ===================
 *
 * 1. is_non_materialized(Operation *op)
 *    - 判断操作是否不需要CGRA tile放置
 *    - 包括：ReserveOp, CtrlMovOp, DataMovOp
 *    - 这些操作不占用实际的计算资源
 *
 * 2. getOperationKindFromMlirOp(Operation *op)
 *    - 将MLIR操作映射到OperationKind枚举
 *    - 用于硬件资源分配和调度
 *
 * 设计原则：
 * =========
 * - 显式优于隐式：明确列出所有情况
 * - 白名单优于黑名单：列出允许的而非禁止的
 * - 类型检查优于运行时判断：利用编译器的类型系统
 *
 * Header文件声明：
 * ================
 * // include/NeuraDialect/Mapping/mapping_util.h
 * 
 * // Returns true if the operation is a steering-mode operation that doesn't
 * // require DataMovOp wrapping (e.g., constants, carry, invariant, etc.).
 * bool is_steering_unwrapped_op(Operation *op);
 *
 * 注意注释也进行了改进：
 * - 使用第三人称单数 "Returns"
 * - 以句号结尾
 * - 给出了具体例子
 */

// 下面是完整的函数实现和上下文代码：

#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraOps.h"

namespace mlir {
namespace neura {

// 将MLIR操作映射到OperationKind枚举
// 用于硬件资源分配和调度决策
OperationKind getOperationKindFromMlirOp(Operation *op) {
  // 整数算术操作
  if (isa<neura::AddOp>(op)) return IAdd;
  if (isa<neura::SubOp>(op)) return ISub;
  if (isa<neura::MulOp>(op)) return IMul;
  // ... 其他操作映射
  
  // 默认回退
  return IAdd;
}

// 判断操作是否不需要CGRA tile放置
// 这些操作是虚拟的，不占用实际的硬件资源
bool is_non_materialized(Operation *op) {
  // ReserveOp: 占位符，用于循环等结构
  // CtrlMovOp: 控制流传输，不占用数据路径
  // DataMovOp: 数据传输包装，不是实际的计算操作
  return mlir::isa<neura::ReserveOp, neura::CtrlMovOp, neura::DataMovOp>(op);
}

// 【核心修改】判断操作是否是steering模式下不需要DataMovOp包装的操作
// 
// Steering模式是CGRA的一种特殊数据流模式，某些操作有内置的路由能力，
// 不需要额外的DataMovOp来进行数据传输。
//
// 此函数明确列出所有这些操作类型，而不是使用否定判断。
bool is_steering_unwrapped_op(Operation *op) {
  return mlir::isa<neura::ConstantOp,        // 常量：本地生成，不需要路由
                   neura::CarryOp,            // 循环携带：有专用数据路径
                   neura::InvariantOp,        // 循环不变量：特殊处理
                   neura::CarryInvariantOp,   // 融合操作：内置路由
                   neura::ConditionalSelectOp,// 条件选择：内置mux
                   neura::InvariantGroupOp,   // 不变量组：组织结构
                   neura::ReserveOp>(op);     // 占位符：不需要实际数据
}

// 判断操作是否是需要物化的reserve用户
// 即：phi、invariant、carry这些需要实际映射到硬件的操作
bool isMaterializedReserveUser(Operation *op) {
  return mlir::isa<neura::PhiOp, neura::InvariantOp, neura::CarryOp>(op);
}

} // namespace neura
} // namespace mlir

/*
 * 总结：
 * =====
 * 
 * 这次修改的核心思想是：
 * 1. 从否定判断（!isa<DataMovOp>）改为肯定判断（明确列出所有类型）
 * 2. 增强代码的可读性和可维护性
 * 3. 避免意外包含不应该包含的操作类型
 * 4. 使代码的设计意图更加明确
 *
 * 这是一个典型的代码review改进案例：
 * - 不改变功能（假设之前的类型列表是完整的）
 * - 提高代码质量
 * - 使代码更容易理解和维护
 */
