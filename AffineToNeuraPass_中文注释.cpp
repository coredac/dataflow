/*
 * AffineToNeura Pass - 中文注释详解版
 * 
 * 本文件提供了AffineToNeura pass实现的详细中文注释版本。
 * 它将Affine方言操作（循环、load/store）转换为Neura方言操作，
 * 用于CGRA（粗粒度可重构架构）执行。
 *
 * 核心概念：
 * ========
 * 
 * 1. 数据流语义：
 *    - Neura方言使用数据流执行模型
 *    - 操作在输入可用时触发
 *    - 循环控制使用valid信号而非命令式控制流
 *
 * 2. 循环控制模型：
 *    - affine.for（命令式） → neura.loop_control（数据流式）
 *    - 循环边界存储为属性（编译时常量）
 *    - Valid信号控制迭代
 *
 * 3. 模式重写：
 *    - 使用贪婪模式重写器（自底向上应用）
 *    - 内层循环先转换，然后是外层循环
 *    - 每个模式独立且可组合
 */

#include "Common/AcceleratorAttrs.h"
#include "Conversion/ConversionPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Memref/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;
using namespace mlir::neura;
using namespace mlir::func;

#define GEN_PASS_DEF_LOWERAFFINETONEURA
#include "Conversion/ConversionPasses.h.inc"

namespace {

/*
 * convertAffineMapToIndices - 将AffineMap转换为索引值列表
 * =======================================================
 * 
 * 将AffineMap转换为适用于neura.load_indexed/store_indexed操作的索引值列表。
 *
 * AffineMap结构：
 * --------------
 * AffineMap定义索引变换：
 *   map<(d0, d1)[s0] -> (d0 + s0, d1 * 2, 42)>
 *   - d0, d1: 维度操作数（循环归纳变量）
 *   - s0: 符号操作数（参数）
 *   - Results: 计算索引的表达式
 *
 * 转换策略：
 * ---------
 * 对于AffineMap中的每个结果表达式：
 *   1. 常量表达式 (42) → neura.constant
 *   2. 维度表达式 (d0) → 直接使用对应的操作数
 *   3. 符号表达式 (s0) → 使用对应的操作数
 *   4. 复杂表达式 (d0 + 1) → 创建affine.apply（由AffineApplyLowering处理）
 *
 * 为什么对复杂表达式使用affine.apply？
 * -----------------------------------
 * - 允许渐进式降低：affine.apply可以稍后被转换
 * - 分离关注点：每个模式处理一个转换
 * - 启用回退路径：复杂表达式可以通过affine→scf→neura路径
 *
 * 参数：
 * -----
 * @param map: 定义索引变换的AffineMap
 * @param map_operands: 维度和符号的值 (d0, d1, ..., s0, s1, ...)
 * @param loc: 新操作的源位置
 * @param rewriter: 用于创建操作的PatternRewriter
 * @param new_indices: [输出] 计算出的索引值
 *
 * 返回值：
 * -------
 * 如果所有表达式都成功转换则返回success()
 * 如果操作数索引越界则返回failure()
 */
LogicalResult convertAffineMapToIndices(AffineMap map, ValueRange map_operands,
                                        Location loc, PatternRewriter &rewriter,
                                        SmallVector<Value> &new_indices) {
  // 清空并预留空间以提高效率
  new_indices.clear();
  new_indices.reserve(map.getNumResults());
  
  // 处理AffineMap中的每个结果表达式
  // 示例：map<(d0, d1) -> (d0, d1 + 1, 0)> 有3个结果
  for (AffineExpr expr : map.getResults()) {
    
    // 情况1：常量表达式
    // -----------------
    // 示例：affine_map<() -> (42)>
    // 结果：创建值为42的neura.constant
    if (AffineConstantExpr const_expr = dyn_cast<AffineConstantExpr>(expr)) {
      IndexType index_type = rewriter.getIndexType();
      IntegerAttr value_attr =
          rewriter.getIntegerAttr(index_type, const_expr.getValue());
      new_indices.push_back(rewriter.create<neura::ConstantOp>(
          loc, index_type, value_attr));
    } 
    
    // 情况2：维度表达式
    // -----------------
    // 示例：affine_map<(d0, d1) -> (d0)>  // d0是维度0
    // 结果：直接使用第一个操作数（例如循环索引%i）
    else if (AffineDimExpr dim_expr = dyn_cast<AffineDimExpr>(expr)) {
      // 安全检查：维度索引必须有效
      if (dim_expr.getPosition() >= map.getNumDims() ||
          dim_expr.getPosition() >=
              map_operands
                  .size()) { // 检查mapOperands大小以确保安全
        return failure();
      }
      // 直接使用对应此维度的操作数
      new_indices.push_back(map_operands[dim_expr.getPosition()]);
    } 
    
    // 情况3：符号表达式
    // -----------------
    // 示例：affine_map<(d0)[s0] -> (s0)>  // s0是符号0
    // 结果：使用符号操作数（传递给map的参数）
    // 
    // 符号操作数在map_operands中位于维度操作数之后：
    //   map_operands = [dim0, dim1, ..., dimN, sym0, sym1, ..., symM]
    else if (AffineSymbolExpr sym_expr = dyn_cast<AffineSymbolExpr>(expr)) {
      unsigned symbol_operand_index = map.getNumDims() + sym_expr.getPosition();
      if (symbol_operand_index >= map_operands.size()) {
        return failure();
      }
      new_indices.push_back(map_operands[symbol_operand_index]);
    } 
    
    // 情况4：复杂表达式
    // -----------------
    // 示例：affine_map<(d0) -> (d0 + 1)>, affine_map<(d0, d1) -> (d0 * 2)>
    // 结果：创建affine.apply操作来计算结果
    //
    // 为什么不在这里展开复杂表达式？
    // -----------------------------
    // 1. 分离关注点：让AffineApplyLowering处理它
    // 2. 渐进式降低：affine.apply → neura操作逐步进行
    // 3. 回退路径：如果AffineApplyLowering也无法处理，用户可以手动使用两阶段降低
    //
    // 渐进式降低的三种可能结果：
    // -------------------------
    // 路径1（理想）：affine.apply在本pass的后续迭代中被AffineApplyLowering转换
    //   affine.apply affine_map<(d0) -> (d0 + 5)>
    //     ↓ [AffineApplyLowering匹配]
    //   neura.add(%d0, neura.constant(5))
    //
    // 路径2（部分支持）：简单表达式转换，复杂表达式保留为affine.apply
    //   如果AffineApplyLowering只支持加法，那么乘法表达式会保留：
    //   affine.apply affine_map<(d0) -> (d0 * 2)>  // 保留，等待进一步处理
    //
    // 路径3（手动回退）：用户需要显式使用SCF方言作为中间步骤
    //   第一步：mlir-opt input.mlir --lower-affine-to-scf
    //     affine.apply affine_map<(d0) -> (d0 * 2 + d1)>
    //       ↓
    //     %0 = arith.muli %d0, 2
    //     %1 = arith.addi %0, %d1
    //   
    //   第二步：mlir-opt --lower-scf-to-neura --lower-affine-to-neura
    //     %0 = arith.muli %d0, 2  →  %0 = neura.mul %d0, neura.constant(2)
    //     %1 = arith.addi %0, %d1 →  %1 = neura.add %0, %d1
    //
    // 注意：本pass并不自动执行SCF回退！
    // 这里只是创建affine.apply，期望：
    // - 要么被AffineApplyLowering处理（路径1）
    // - 要么用户手动介入使用SCF路径（路径3）
    else {
      // 对于更复杂的affine表达式（例如d0 + c1, d0 * 2, 等），
      // 使用affine.apply来具体化结果。
      //
      // 这不是"回退"而是"延迟处理"：
      // - 创建的affine.apply可能在贪婪重写器的后续迭代中被处理
      // - 如果仍然无法处理，最终会导致错误或需要用户介入
      //
      // TODO: 处理更多复杂表达式（mul, div, mod等）。
      llvm::errs() << "[affine2neura] 复杂affine表达式: " << expr << "\n";
      
      // 为这个表达式创建单结果AffineMap
      // 创建的affine.apply将在后续迭代中由AffineApplyLowering尝试转换
      AffineMap single_result_map = AffineMap::get(
          map.getNumDims(), map.getNumSymbols(), expr, rewriter.getContext());
      Value complexIndex = rewriter.create<affine::AffineApplyOp>(
          loc, single_result_map, map_operands);
      new_indices.push_back(complexIndex);
    }
  }
  return success();
}

/*
 * AffineLoadLowering - 将affine.load转换为neura.load_indexed
 * ===========================================================
 *
 * 用于将affine.load转换为neura.load_indexed的模式。
 *
 * 转换：
 * ------
 * 之前：
 *   %v = affine.load %memref[map(%i, %j)] : memref<10x10xf32>
 *
 * 之后：
 *   %idx0 = <从map计算>
 *   %idx1 = <从map计算>
 *   %v = neura.load_indexed %memref[%idx0, %idx1] : memref<10x10xf32>
 *
 * 关键区别：
 * ---------
 * - affine.load: 使用AffineMap进行索引计算
 * - neura.load_indexed: 使用显式索引值
 *
 * 为什么进行此转换？
 * -----------------
 * - Neura方言不支持AffineMap（数据流语义）
 * - 显式索引允许硬件独立调度操作
 * - 每个索引计算成为一个独立的数据流操作
 */
struct AffineLoadLowering : public OpRewritePattern<affine::AffineLoadOp> {
  using OpRewritePattern<affine::AffineLoadOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(affine::AffineLoadOp load_op,
                                PatternRewriter &rewriter) const override {
    Location loc = load_op.getLoc();
    auto memref = load_op.getMemref();
    AffineMap map = load_op.getAffineMap();
    ValueRange map_operands = load_op.getMapOperands();
    
    // 步骤1：将AffineMap转换为显式索引值
    // 获取load操作的索引。
    SmallVector<Value> new_indices;
    if (failed(convertAffineMapToIndices(map, map_operands, loc, rewriter,
                                         new_indices))) {
      return load_op.emitError(
          "[affine2neura] 转换affine map到索引失败");
    }

    // 步骤2：验证memref类型和索引
    // ---------------------------
    MemRefType memref_type = dyn_cast<MemRefType>(memref.getType());
    if (!memref_type) {
      return load_op.emitError(
          "[affine2neura] load的基址不是MemRefType");
    }
    
    // 索引数量必须匹配memref的秩
    // 示例：memref<10x20xf32>需要恰好2个索引
    if (new_indices.size() != static_cast<size_t>(memref_type.getRank())) {
      return load_op.emitError(
                 "[affine2neura] affine map的索引数量 (")
             << new_indices.size() << ") 与memref秩不匹配 ("
             << memref_type.getRank() << ")";
    }

    // 步骤3：创建neura.load_indexed操作
    // 创建neura.load_indexed操作。
    // 
    // neura.load_indexed语义：
    // - 当所有索引可用时触发（数据流）
    // - 无副作用（纯load）
    // - 内存访问完成时结果可用
   LoadIndexedOp new_load_op = rewriter.create<neura::LoadIndexedOp>(
        loc, load_op.getType(), memref, ValueRange{new_indices});

    // 步骤4：替换原始操作
    // load结果的所有使用都会自动更新
    rewriter.replaceOp(load_op, new_load_op.getResult());
    return success();
  }
};

/*
 * AffineStoreLowering - 将affine.store转换为neura.store_indexed
 * ==============================================================
 *
 * 用于将affine.store转换为neura.store_indexed的模式。
 *
 * 转换：
 * ------
 * 之前：
 *   affine.store %value, %memref[map(%i, %j)] : memref<10x10xf32>
 *
 * 之后：
 *   %idx0 = <从map计算>
 *   %idx1 = <从map计算>
 *   neura.store_indexed %value to %memref[%idx0, %idx1] : memref<10x10xf32>
 *
 * 类似于AffineLoadLowering但用于store。
 * 关键区别：store没有结果值。
 */
struct AffineStoreLowering : public OpRewritePattern<affine::AffineStoreOp> {
  using OpRewritePattern<affine::AffineStoreOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(affine::AffineStoreOp store_op,
                                PatternRewriter &rewriter) const override {
    Location loc = store_op.getLoc();
    auto memref = store_op.getMemref();
    Value value = store_op.getValueToStore();
    AffineMap map = store_op.getAffineMap();
    ValueRange mapOperands = store_op.getMapOperands();

    // 将AffineMap转换为显式索引
    SmallVector<Value> newIndices;
    if (failed(convertAffineMapToIndices(map, mapOperands, loc, rewriter,
                                         newIndices))) {
      return store_op.emitError(
          "[affine2neura] 转换affine map到索引失败");
    }

    // 验证memref和索引
    MemRefType memRefType = dyn_cast<MemRefType>(memref.getType());
    if (!memRefType) {
      return store_op.emitError(
          "[affine2neura] store的基址不是MemRefType");
    }
    if (newIndices.size() != static_cast<size_t>(memRefType.getRank())) {
      return store_op.emitError(
                 "[affine2neura] affine map的索引数量 (")
             << newIndices.size() << ") 与memref秩不匹配 ("
             << memRefType.getRank() << ")";
    }

    // 创建neura.store_indexed（无结果）
    rewriter.create<neura::StoreIndexedOp>(loc, value, memref,
                                           ValueRange{newIndices});
    // 删除原始store操作
    rewriter.eraseOp(store_op);
    return success();
  }
};

/*
 * AffineApplyLowering - 将affine.apply转换为neura操作（简单表达式）
 * =================================================================
 *
 * 用于将affine.apply转换为neura操作的模式（针对简单表达式）。
 *
 * 背景：
 * ------
 * affine.apply计算AffineMap并返回结果：
 *   %result = affine.apply affine_map<(d0) -> (d0 + 5)>(%i)
 *
 * 此模式处理可以直接降低到neura操作的简单情况。
 *
 * 支持的表达式：
 * -------------
 * 当前支持：d0 + 常量
 * 示例：affine_map<(d0) -> (d0 + 5)> → neura.add(%d0, neura.constant(5))
 *
 * 不支持（将失败）：
 * -----------------
 * - 乘法：d0 * 2
 * - 除法：d0 / 2
 * - 多维度：d0 + d1
 * - 取模：d0 mod 16
 *
 * 回退策略：
 * ---------
 * 当不支持时，用户应该：
 * 1. 首先使用--lower-affine-to-scf（affine → SCF方言）
 * 2. 然后使用--lower-scf-to-neura（SCF → Neura方言）
 * 这提供了完整的affine表达式支持。
 */
struct AffineApplyLowering : public OpRewritePattern<affine::AffineApplyOp> {
  using OpRewritePattern<affine::AffineApplyOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(affine::AffineApplyOp apply_op,
                                PatternRewriter &rewriter) const override {
    AffineMap map = apply_op.getAffineMap();
    ValueRange operands = apply_op.getMapOperands();
    Location loc = apply_op.getLoc();

    // 健全性检查：affine.apply总是只有一个结果
    // AffineMap在affine.for或affine.if中使用时可以有多个结果，
    // 但AffineApplyOp总是只有一个结果。
    // 多结果示例（在affine.for上下文中）：
    //   affine_map<(d0, d1) -> (d0 + 1, d1 * 2)>
    // 但是，AffineApplyOp会使用单结果map，如：
    //   affine_map<(d0) -> (d0 + 1)>
    if (map.getNumResults() != 1) {
      return apply_op.emitError(
          "[affine2neura] AffineApplyOp必须只有一个结果");
    }

    AffineExpr expr = map.getResult(0);
    
    // 支持表达式的模式匹配
    // 处理简单的affine表达式，如d0 + cst。
    // TODO: 处理更多复杂表达式。
    
    // 检查表达式是否为二元操作
    if (isa<AffineBinaryOpExpr>(expr)) {
      AffineBinaryOpExpr bin_expr = dyn_cast<AffineBinaryOpExpr>(expr);
      
      // 情况：加法（d0 + cst）
      // ----------------------
      if (bin_expr.getKind() == AffineExprKind::Add) {
        // 左侧应该是维度（例如d0）
        if (isa<AffineDimExpr>(bin_expr.getLHS())) {
          AffineDimExpr dim = dyn_cast<AffineDimExpr>(bin_expr.getLHS());
          
          // 右侧应该是常量（例如5）
          if (isa<AffineConstantExpr>(bin_expr.getRHS())) {
            AffineConstantExpr cst =
                dyn_cast<AffineConstantExpr>(bin_expr.getRHS());
            
            // 创建neura操作：constant + add
            // 示例：d0 + 5变成：
            //   %c5 = neura.constant 5 : index
            //   %result = neura.add %d0, %c5 : index
            neura::ConstantOp cstVal = rewriter.create<neura::ConstantOp>(
                loc, rewriter.getIndexType(),
                rewriter.getIntegerAttr(rewriter.getIndexType(),
                                        cst.getValue()));
            neura::AddOp addOp = rewriter.create<neura::AddOp>(
                loc, cstVal.getType(), operands[dim.getPosition()], cstVal);
            
            // 用add结果替换affine.apply
            rewriter.replaceOp(apply_op, addOp.getResult());
            return success();
          }
        }
      }
      
      // 可以在这里添加更多情况：
      // - 减法：d0 - cst
      // - 2的幂次乘法：d0 * 4（可以使用移位）
      // - 等等
    }

    // 不支持的表达式 - 失败并提供有用的消息
    // 可以在这里为不同的affine表达式添加更多情况。
    // 现在，我们只对不支持的表达式发出错误。
    return apply_op.emitError("[affine2neura] 不支持的复杂affine"
                              "表达式在AffineApplyOp中。\n")
           << "只支持简单的affine表达式，如d0 + cst。\n";
  }
};

/*
 * AffineForLowering - 将affine.for循环转换为neura数据流操作
 * =========================================================
 *
 * 用于将affine.for循环转换为neura数据流操作的模式。
 *
 * 命令式vs数据流循环模型：
 * -----------------------
 * 
 * Affine（命令式）：
 *   affine.for %i = 0 to N step 2 {
 *     %v = affine.load %A[%i]
 *     affine.store %v, %B[%i]
 *   }
 * 
 * 控制流：基于PC，顺序执行
 * 循环控制：比较、分支指令
 * 
 * Neura（数据流）：
 *   %grant = neura.grant_once            // 启动信号
 *   %i, %valid = neura.loop_control(%grant) <{start=0, end=N, step=2}>
 *   %v = neura.load_indexed %A[%i]      // 当%i可用时触发
 *   neura.store_indexed %v to %B[%i]    // 当%v, %i可用时触发
 * 
 * 控制流：基于令牌，操作在输入就绪时触发
 * 循环控制：Valid信号通过数据流图传播
 *
 * 转换策略：
 * ---------
 * 1. 创建grant_once：提供初始valid信号
 * 2. 创建loop_control：生成迭代索引和valid信号
 * 3. 内联循环体：操作以数据流方式执行
 * 4. 替换归纳变量：使用loop_control索引输出
 *
 * 循环控制语义：
 * -------------
 * neura.loop_control(%parent_valid) <{start, end, step, type}>
 *   → (%index, %valid)
 *
 * - 输入：
 *   * parent_valid: 指示何时开始/继续的信号
 * - 输出：
 *   * index: 当前迭代值
 *   * valid: 指示迭代活跃的信号
 * - 属性：
 *   * start, end, step: 循环边界（必须是常量）
 *   * type: "increment"或"decrement"
 *
 * 为什么边界使用属性？
 * -------------------
 * - 数据流调度：硬件需要静态循环边界
 * - 编译时分析：启用循环展开、流水线化
 * - 资源分配：计算缓冲区大小等
 *
 * 设计决策：不支持动态边界
 * -------------------------
 * 动态循环边界（运行时确定的边界）不被支持，因为：
 * 1. CGRA硬件配置需要编译时已知的循环结构
 * 2. 静态边界允许关键的硬件优化（流水线、展开等）
 * 3. 如果需要动态循环，应该：
 *    - 在host CPU上执行动态循环
 *    - 或者使用保守的最大边界并在运行时提前退出
 *
 * 嵌套循环处理：
 * -------------
 * 当前：每个循环获得独立的grant_once
 *   外层：grant_once → loop_control → body
 *   内层：grant_once → loop_control → body
 *
 * 这样可以工作但会创建冗余的控制信号。
 *
 * 未来优化：
 *   外层：grant_once → loop_control → body
 *                          ↓ （重用valid信号）
 *   内层：               loop_control → body
 *
 * TODO: 优化嵌套循环以重用父循环的valid信号。
 * 这需要数据流分析来识别父子关系。
 */
struct AffineForLowering : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(affine::AffineForOp for_op,
                                PatternRewriter &rewriter) const override {
    Location loc = for_op.getLoc();

    // 步骤1：提取并验证循环边界
    // --------------------------
    // 提取循环边界 - 必须是常量（设计决策）。
    // 
    // 为什么只支持常量边界？
    // -----------------------
    // 这不是临时限制，而是明确的设计决策：
    // - Neura loop_control使用属性（编译时常量）进行硬件配置
    // - CGRA架构需要在配置时知道循环结构以进行资源分配
    // - 静态边界允许关键优化：循环展开、流水线、并行化
    // 
    // 如果需要动态循环：
    // - 应在host CPU上执行（不在CGRA上）
    // - 或使用保守的最大边界，运行时条件提前退出
    if (!for_op.hasConstantLowerBound() || !for_op.hasConstantUpperBound()) {
      return for_op.emitError(
          "[affine2neura] 尚不支持非常量循环边界。"
          "循环边界必须是编译时常量以便进行CGRA硬件配置");
    }

    int64_t lower_bound = for_op.getConstantLowerBound();
    int64_t upper_bound = for_op.getConstantUpperBound();
    int64_t step = for_op.getStepAsInt();

    // 步骤2：创建父valid信号
    // ----------------------
    // 目前，总是为每个循环创建grant_once。
    // TODO: 优化嵌套循环以重用父循环的valid信号。
    //
    // grant_once语义：
    // - 在开始时触发一次
    // - 向loop_control提供初始valid信号
    // - 可以通过谓词门控（这里尚未使用）
    Type i1_type = rewriter.getI1Type();
    Value parent_valid = rewriter.create<neura::GrantOnceOp>(
        loc, i1_type, /*value=*/Value(), /*constant_value=*/nullptr);

    // 步骤3：创建loop_control操作
    // ---------------------------
    // 创建loop_control操作。
    //
    // 这是数据流循环执行的核心：
    // - 接受parent_valid作为输入
    // - 为每次迭代输出(index, valid)
    // - 边界指定为属性
    auto index_type = rewriter.getIndexType();
    
    auto loop_control = rewriter.create<neura::LoopControlOp>(
        loc,
        /*resultTypes=*/TypeRange{index_type, i1_type},
        /*parentValid=*/parent_valid,
        /*iterationType=*/rewriter.getStringAttr("increment"),
        /*start=*/rewriter.getI64IntegerAttr(lower_bound),
        /*end=*/rewriter.getI64IntegerAttr(upper_bound),
        /*step=*/rewriter.getI64IntegerAttr(step));

    Value loop_index = loop_control.getResult(0);
    // 注意：loop_control.getResult(1)返回loop_valid信号
    //
    // loop_valid的用途：
    // -----------------
    // loop_valid信号指示当前迭代是否有效，可以用于：
    // 1. 门控循环体内的操作（条件执行）
    // 2. 嵌套循环优化：内层循环的parent_valid应该使用外层的loop_valid
    //
    // 嵌套循环优化示例：
    // ----------------
    // 当前实现（每个循环独立）：
    //   外层：%outer_grant = grant_once
    //         %i, %outer_valid = loop_control(%outer_grant)
    //   内层：%inner_grant = grant_once  ← 冗余！
    //         %j, %inner_valid = loop_control(%inner_grant)
    //
    // 优化后（重用valid信号）：
    //   外层：%outer_grant = grant_once
    //         %i, %outer_valid = loop_control(%outer_grant)
    //   内层：%j, %inner_valid = loop_control(%outer_valid)  ← 重用外层valid！
    //
    // 实现优化需要：
    // - 数据流分析识别父子循环关系
    // - 在内层循环转换时能访问到外层的loop_valid
    // - 这需要在pass架构上做较大改动
    //
    // 目前：每个循环创建独立的grant_once（简单但有些冗余）

    // 步骤4：替换归纳变量
    // -------------------
    // 替换归纳变量的使用。
    //
    // 原始affine.for：
    //   affine.for %i = 0 to N {
    //     %v = affine.load %A[%i]  // 使用归纳变量%i
    //   }
    //
    // 转换后：
    //   %i, %valid = neura.loop_control(...)
    //   %v = neura.load_indexed %A[%i]  // 使用loop_control索引输出
    //
    // replaceAllUsesWith自动更新所有引用
    for_op.getInductionVar().replaceAllUsesWith(loop_index);

    // 步骤5：内联循环体
    // -----------------
    // 在for_op之前内联循环体操作。
    //
    // 原始结构：
    //   affine.for %i ... {
    //     ^bb0(%i: index):
    //       <body操作>
    //       affine.yield
    //   }
    //
    // 内联后：
    //   %grant = neura.grant_once
    //   %i, %valid = neura.loop_control(...)
    //   <body操作>  // 在这里内联
    //
    // 为什么内联而不是保留区域？
    // - Neura方言使用扁平结构（无命令式控制流）
    // - 操作基于数据可用性执行（数据流）
    // - 区域会暗示控制流边界
    //
    // 模式应用顺序确保正确性：
    // - 贪婪重写器自底向上应用模式
    // - 先转换内层循环（它们的操作已经被降低）
    // - 然后转换外层循环（内层neura操作已就位）
    Block &body_block = for_op.getRegion().front();
    Operation *terminator = body_block.getTerminator();
    rewriter.eraseOp(terminator);  // 首先移除affine.yield。
    
    // inlineBlockBefore：将操作从body_block移动到for_op之前
    // 这保持了SSA支配性：
    // - loop_control定义%i
    // - %i被内联的body操作使用
    // - 正确的支配性：loop_control在使用之前
    rewriter.inlineBlockBefore(&body_block, for_op.getOperation(),
                               body_block.getArguments());
    
    // 步骤6：移除原始for操作
    // ----------------------
    // 删除for_op。
    // 此时：
    // - Body操作已内联
    // - 归纳变量已替换
    // - 循环结构不再需要
    rewriter.eraseOp(for_op);

    return success();
  }
};

/*
 * LowerAffineToNeuraPass - Pass主实现
 * ====================================
 *
 * 编排所有模式应用的主pass实现。
 *
 * Pass架构：
 * ----------
 * MLIR使用pass流水线逐步降低IR：
 *   Affine方言（高级循环）
 *     ↓ [此pass]
 *   Neura方言（数据流操作）
 *     ↓ [后续pass]
 *   硬件配置（CGRA位流）
 *
 * 模式应用策略：
 * -------------
 * 使用贪婪模式重写器：
 * - 重复应用模式直到没有更多匹配
 * - 自底向上遍历（子节点先于父节点）
 * - 确保内层循环先于外层循环转换
 *
 * 为什么使用贪婪而不是一次性？
 * - 模式相互作用：循环内的load/store
 * - 顺序很重要：嵌套循环的内→外
 * - 灵活性：可以轻松添加/删除模式
 *
 * 目标函数：
 * ---------
 * 仅应用于目标Neura加速器的函数：
 * - 检查加速器属性
 * - 跳过目标其他加速器的函数
 * - 如果没有属性则应用于所有（用于测试）
 */
struct LowerAffineToNeuraPass
    : public PassWrapper<LowerAffineToNeuraPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAffineToNeuraPass)

  // 注册所需的方言
  // 此pass中使用的所有方言都必须注册
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<neura::NeuraDialect,    // 目标方言
                    arith::ArithDialect,     // 用于算术操作
                    memref::MemRefDialect,   // 用于内存操作
                    affine::AffineDialect>(); // 源方言
  }

  // Pass命令行接口
  StringRef getArgument() const override { return "lower-affine-to-neura"; }
  StringRef getDescription() const override {
    return "将affine操作降低到Neura方言操作";
  }

  // 主pass逻辑
  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    MLIRContext *context = module_op.getContext();

    // 遍历模块中的所有函数
    // 逐个函数应用转换
    module_op.walk([&](func::FuncOp func_op) {
      // 目标选择：转换哪些函数
      // 检查函数是否目标neura加速器，如果没有属性则应用于所有。
      if (func_op->hasAttr(mlir::accel::kAcceleratorAttr)) {
        auto target = func_op->getAttrOfType<StringAttr>(
            mlir::accel::kAcceleratorAttr);
        if (!target || target.getValue() != mlir::accel::kNeuraTarget) {
          return;  // 跳过此函数。
        }
      }
      // 如果没有加速器属性，仍然应用pass（用于测试）。
      
      // 注册所有重写模式
      // 顺序无关紧要 - 贪婪重写器处理顺序
      RewritePatternSet patterns(context);
      patterns.add<AffineForLowering,      // 转换循环
                   AffineLoadLowering,      // 转换load
                   AffineStoreLowering,     // 转换store
                   AffineApplyLowering>     // 转换索引计算
                  (context);

      // 贪婪应用模式
      // 持续直到没有模式匹配（不动点）
      if (failed(applyPatternsGreedily(func_op.getOperation(),
                                       std::move(patterns)))) {
        func_op.emitError("[affine2neura] 降低affine操作到Neura方言失败");
        signalPassFailure();
      }
    });
  }
};

} // namespace

/*
 * Pass工厂函数
 * ============
 *
 * 创建并返回pass的唯一实例。
 * 当构建pass流水线时由MLIR pass管理器调用。
 *
 * 用法：
 *   PassManager pm(...);
 *   pm.addPass(mlir::createLowerAffineToNeuraPass());
 *   pm.run(module);
 *
 * 或从命令行：
 *   mlir-neura-opt input.mlir --lower-affine-to-neura
 */
std::unique_ptr<mlir::Pass> mlir::createLowerAffineToNeuraPass() {
  return std::make_unique<LowerAffineToNeuraPass>();
}

/*
 * 关键设计决策总结：
 * ==================
 *
 * 1. 数据流优于控制流：
 *    - 操作在输入就绪时触发
 *    - Valid信号代替PC
 *    - 在CGRA上启用空间并行性
 *
 * 2. 基于属性的循环边界：
 *    - 编译时常量启用优化
 *    - 硬件调度器可以预先计算迭代
 *    - 设计决策：不支持动态边界（CGRA硬件限制）
 *
 * 3. 渐进式降低：
 *    - 对复杂表达式使用affine.apply
 *    - 可以回退到affine→scf→neura
 *    - 每个pass处理一个抽象级别
 *
 * 4. 每个循环独立的grant_once：
 *    - 简单且正确
 *    - 可优化：嵌套循环重用父valid（需要数据流分析）
 *    - 权衡：为了实现简单性而有一些冗余
 *
 * 5. 贪婪模式应用：
 *    - 自底向上确保内层先于外层
 *    - 不动点迭代直到稳定
 *    - 灵活：易于添加新模式
 *
 * 未来工作：
 * ==========
 * - 更多affine表达式（mul、div、mod等）直接转换
 * - 嵌套循环优化（重用父valid信号，需要数据流分析）
 * - 用于循环变换的多面体分析
 * - 支持affine.if（条件执行）
 * 
 * 明确不支持的特性：
 * ==================
 * - 动态循环边界：这是CGRA硬件的根本限制，不会支持
 *   需要动态循环的代码应该在host CPU上执行
 */
