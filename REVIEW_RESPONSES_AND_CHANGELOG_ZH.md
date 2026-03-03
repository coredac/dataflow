# Review 回复与变更说明文档

**分支：** `feature/resource-aware-task-optimization`  
**基准提交：** `8ee3421`（"fix(resource-aware-opt): restore affine/scf fallback in computeTripCount"）  
**HEAD 提交：**
- `099466f`：修复 performFusion 在多块任务体中的崩溃问题
- `c672e3a`：修复 computeTripCount 从后端 CF-lowered IR 中提取 trip count；在 RESOPT 流水线中添加 affine 序列化/完美化 pass

---

## 第一部分：Review 意见回复

### 意见 1 — `InsertDataMovPass.cpp`："为什么要跳过 ReserveOp？"

**位置：** `lib/NeuraDialect/Transforms/InsertDataMovPass.cpp`，第 31 行

```cpp
isa<neura::ReserveOp>(op) ||
```

**回复：**

`ReserveOp` 产生一个 `!neura.data<T, i1>` 类型的占位符，用于构成 dataflow 中的**循环回边（recurrence cycle）**：

```
%v        = neura.reserve : !neura.data<f32, i1>   // 占位符
%result   = neura.phi_start %init, %v              // 消费 %v
// ... 循环体计算出 %next ...
neura.ctrl_mov %next -> %v                         // 闭合回边
```

`ctrl_mov` 必须持有与 `phi_start` 使用的**完全相同的 SSA value**（即 `%v`）的直接引用。如果用 `DataMovOp` 包装 `%v`，`phi_start` 收到的是 `data_mov(%v)`，而 `ctrl_mov` 的 target 仍然是 `%v`——两者是不同的 SSA value，recurrence cycle 因此断裂。

跳过逻辑分两个层次：
1. **`isa<neura::ReserveOp>(op)`（第 31 行）：** 不处理 `ReserveOp` 本身——它没有任何 operand 需要包装。
2. **`isa<neura::ReserveOp>(producer)` 守卫（第 102 行）：** 当消费者 op（`phi_start` 等）将 `ReserveOp` 的结果作为 operand 使用时，**不**在中间插入 `DataMovOp`——否则会引入新的 SSA value，导致 `ctrl_mov` 与回边断开。

我们通过实验验证了这一点：移除该跳过逻辑会导致所有 4 个测试立即崩溃（core dump）。崩溃 trace 中出现 `<<UNKNOWN SSA VALUE>>`，证实 recurrence cycle 已经断裂。

---

### 意见 2 — `ResourceAwareTaskOptimizationPass.cpp:422-423`："我们不允许一个 task 包含多个 hyperblock，为什么还要处理这种情况？"

**位置：** 旧代码中 `profileTask()` 约第 422-423 行：
```
// are then profiled via Phase 2 (runNeuraPipelineOnKernel) and results
// are combined: max(ii) / sum(steps).
```

**回复：**

同意——架构上不允许多 hyperblock task。旧代码（`8ee3421`）在 `profileTask()` 内部有一个复杂的 Phase 1 流水线：
1. 检测 `is_multi_hb = (hyperblocks.size() > 1) || (preexisting_kernels.size() > 1)`
2. 对每个 hyperblock 分别克隆 + 执行 Phase 1（classify-counters → convert-taskflow-to-neura）
3. 用 `max(ii) / sum(steps)` 合并结果

**这部分已被完全删除。** 新代码中：
- `profileTask()` 通过 assert 确保 `neura.kernel` op 已经存在（pass 在 lowering 之后运行）
- 不做任何 hyperblock 级别的操作，直接提取 kernel
- 对于 fused task（来自 `performFusion`，可能包含多个 kernel），使用 `max(ii) / max(steps)`，因为 fused kernel 共享同一块 tile 阵列并发执行
- 在 `computeTripCount` 的 Strategy 2 中添加了 `assert(kernel_count <= 1)` 来强制执行单 kernel per task 的不变量

---

### 意见 3 — `ResourceAwareTaskOptimizationPass.cpp:999-1000`："多个独立的 counter chain 不是顺序执行的，它们是同时执行的"

**位置：** 旧代码：
```
// Multiple independent counter chains are summed (they are sequential).
```

**回复：**

同意——旧注释有误。独立的 counter chain 是并发的（它们驱动相互独立的循环维度，在 CGRA 上重叠执行）。

**新代码已修正。** 注释现在改为：
```
// Multiple independent counter chains execute concurrently, so the trip
// count is max(chain_product) across chains.
```

代码也正确使用了 `total = std::max(total, chain_product)`（而非 `total += chain_product`）。

---

### 意见 4 — `ResourceAwareTaskOptimizationPass.cpp`，affine.for："为什么还有 affine.for？运行这个 pass 时不应该存在 affine dialect"

**位置：** 旧代码的 `computeTripCount()` 中存在：
```cpp
if (auto affineFor = dyn_cast<affine::AffineForOp>(op)) { ... }
```

**回复：**

同意——`--resource-aware-task-optimization` 现在在 `--lower-affine` 之后运行，不应再有任何 `affine.for`。旧的 fallback 路径是 pass 在完整 lowering 之前运行时留下的残留。

**已完全删除。** 新的 `computeTripCount()` 有三种策略，均不引用 `affine::AffineForOp`：
1. **Strategy 1：** `taskflow.counter` ops（neura lowering 之前的 IR）
2. **Strategy 2：** `neura.counter` ops（convert-taskflow-to-neura 之后，ctrl-to-dataflow 之前）
3. **Strategy 3（新增）：** `arith.cmpi` + `neura.icmp`（CF lowering 之后，transform-ctrl-to-data-flow 之后）

相关头文件也随之删除：`mlir/Dialect/Affine/IR/AffineOps.h`、`mlir/Dialect/SCF/IR/SCF.h` 等。

---

### 意见 5 — `irregular-loop.mlir` RESOPT 流水线："这个 pass 应该在每个 task 都被转换为 neura.kernel & dataflow IR 之后才运行"

**位置：** 旧测试流水线：
```
// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --resource-aware-task-optimization \
```

**回复：**

同意——旧测试在 `--construct-hyperblock-from-task` 之后直接运行 `--resource-aware-task-optimization`，此时 task 尚未被 lower 到 `neura.kernel` + dataflow IR。这迫使 RESOPT 必须内嵌自己的 lowering 流水线（Phase 1 代码），既复杂又脆弱。

**已修复。** 所有 4 个测试文件现在都在 `--resource-aware-task-optimization` 之前包含完整的 lowering 链：

```
--affine-loop-tree-serialization
--affine-loop-perfection
--convert-affine-to-taskflow
--construct-hyperblock-from-task
--classify-counters
--convert-taskflow-to-neura
--lower-affine
--convert-scf-to-cf
--convert-cf-to-llvm
--assign-accelerator
--lower-memref-to-neura
--lower-arith-to-neura
--lower-builtin-to-neura
--lower-llvm-to-neura
--promote-input-arg-to-const
--fold-constant
--canonicalize-return
--canonicalize-live-in
--leverage-predicated-value
--transform-ctrl-to-data-flow
--fold-constant
--resource-aware-task-optimization
```

这确保 RESOPT 接收到的是包含 dataflow 形式的 `neura.kernel` op 的完整 lowered IR，满足预期的前置条件。

---

## 第二部分：详细变更说明

### 概述

这些变更将 `ResourceAwareTaskOptimizationPass` 改造为一个**后 lowering（post-lowering）pass**。原来它在内部嵌入了自己的 lowering 流水线（Phase 1：classify-counters → convert-taskflow-to-neura），从 `taskflow.hyperblock` op 产生 `neura.kernel` op。现在它期望输入 IR 已经包含 dataflow 形式的 `neura.kernel` op，由标准编译器流水线在 RESOPT 之前产生。

### 修改文件列表

| 文件 | 变更摘要 |
|------|---------|
| `include/TaskflowDialect/TaskflowOps.td` | 放宽 `SingleBlockImplicitTerminator` 和 `SizedRegion<1>` 约束 |
| `lib/NeuraDialect/Transforms/InsertDataMovPass.cpp` | 为 `ReserveOp` 跳过逻辑添加说明注释 |
| `lib/TaskflowDialect/Transforms/.../ResourceAwareTaskOptimizationPass.cpp` | 大规模简化（约 1400 行 diff） |
| `test/multi-cgra/taskflow/irregular-loop/irregular-loop.mlir` | 完整 lowering 流水线 + 更新 RESOPT 检查 |
| `test/multi-cgra/taskflow/multi-nested/multi-nested.mlir` | 完整 lowering 流水线 + 更新 RESOPT 检查 |
| `test/multi-cgra/taskflow/parallel-nested/parallel-nested.mlir` | 完整 lowering 流水线 + 更新 RESOPT 检查 |
| `test/multi-cgra/taskflow/resnet/simple_resnet_tosa.mlir` | 完整 lowering 流水线 + 更新 RESOPT 检查 |

---

### 1. `TaskflowOps.td` — 支持多块任务体

**问题：** 经过 CF lowering（`--convert-scf-to-cf`、`--convert-cf-to-llvm`）后，`taskflow.task` 和 `taskflow.hyperblock` 的体内包含通过 `llvm.br`/`llvm.cond_br` 连接的多个基本块。旧的 ODS 定义要求 `SizedRegion<1>`（恰好 1 个块）和 `SingleBlockImplicitTerminator`，这会拒绝合法的多块 IR。

**变更：**
```tablegen
// 之前：
SingleBlockImplicitTerminator<"TaskflowYieldOp">
let regions = (region SizedRegion<1>:$body);

// 之后：
// （删除了 SingleBlockImplicitTerminator）
let regions = (region AnyRegion:$body);
```

`taskflow.task` 和 `taskflow.hyperblock` 现在可以持有多块 region。`taskflow.yield` 终止符仍然是必需的，但现在由 lowering pass 负责放置，而非自动插入。

---

### 2. `ResourceAwareTaskOptimizationPass.cpp` — 主要变更

#### 2.1. 删除头文件（10 个）

删除了 RESOPT 内部不再直接操作的 dialect 依赖：
- `Conversion/ConversionPasses.h` — 不再在内部运行转换 pass
- `mlir/Dialect/Affine/IR/AffineOps.h` — 不再处理 `affine.for`
- `mlir/Dialect/Arith/IR/Arith.h`
- `mlir/Dialect/ControlFlow/IR/ControlFlowOps.h`
- `mlir/Dialect/LLVMIR/LLVMDialect.h`
- `mlir/Dialect/SCF/IR/SCF.h`
- `mlir/Dialect/Vector/IR/VectorOps.h`
- `mlir/Conversion/AffineToStandard/AffineToStandard.h`
- `mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h`
- `mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h`

`getDependentDialects()` 中同步删除：`AffineDialect`、`ArithDialect`、`ControlFlowDialect`、`LLVMDialect`、`SCFDialect`、`VectorDialect`。

#### 2.2. `profileTask()` — 从约 250 行简化到约 60 行

**之前（8ee3421）：**
```
profileTask:
  ├─ 检测 hyperblocks + 预存在的 kernels
  ├─ 如果 is_multi_hb：
  │   ├─ 对每个 hyperblock：
  │   │   ├─ 克隆父函数
  │   │   ├─ 只保留该 hyperblock（擦除其他 + 占位符）
  │   │   ├─ 运行 Phase 1 PM：classify-counters → convert-taskflow-to-neura
  │   │   ├─ 从 Phase 1 输出中提取 kernel
  │   │   └─ 收集 kernel
  │   ├─ 合并：ii = max，steps = sum(cp_depth)
  │   └─ 对每个 kernel 执行 Phase 2 profile
  ├─ 否则（单 hyperblock）：
  │   ├─ 克隆父函数
  │   ├─ 运行 Phase 1
  │   ├─ 提取 kernels
  │   └─ Phase 2 profile：ii = max，steps = max
  └─ 写入 node->ii, node->steps
```

**之后：**
```
profileTask:
  ├─ assert：neura.kernel op 必须存在（post-lowering 不变量）
  ├─ 将 task 克隆到临时 module
  ├─ 从克隆 task 中提取所有 kernels
  ├─ 对每个 kernel：运行 Phase 2（InsertDataMov + Mapper）
  ├─ 合并：ii = max，steps = max
  └─ 写入 node->ii, node->steps
```

主要简化：
- **RESOPT 内部不再运行 Phase 1。** 不再调用 `classify-counters` 或 `convert-taskflow-to-neura`，这些 pass 预期已在 RESOPT 之前的流水线中执行。
- **删除多 hyperblock 处理。** `is_multi_hb` 标志、per-hyperblock 克隆循环、`sum_cp_depth` 累加器全部删除。单 kernel per task 是不变量（通过 assert 强制执行）。
- **Fused task**（来自 `performFusion`）可能有 2 个 kernel；分别 profile 后用 `max(ii) / max(steps)` 合并，因为它们共享 tile 阵列。

#### 2.3. `computeTripCount()` — 新增 Strategy 3

**之前（8ee3421）：** 两种策略 + affine.for fallback
1. `taskflow.counter` ops → counter chain 范围之积（使用 `total += chain_product`，即对各 chain 求和）
2. `affine.for` 嵌套循环 → 递归计算 trip count 之积

**之后：** 三种策略，不含 affine
1. `taskflow.counter` ops → counter chain 范围之积（使用 `total = max(total, chain_product)`——并发 chain）
2. `neura.counter` ops → 每 kernel 内的乘积（含 `assert(kernel_count <= 1)`）
3. **（新增）** Post-CF-lowered IR：
   - **外层循环：** `arith.cmpi`（`predicate=slt`）——从 RHS 常量中提取上界
   - **内层 kernel 循环：** `neura.kernel` 内的 `neura.icmp`（`cmpType="slt"`，含 `rhs_value` 属性）
   - `trip_count = outer_product × inner_product`

Strategy 3 的必要性：`--transform-ctrl-to-data-flow` 消耗了 `neura.counter` op，将其 lower 为 dataflow 节点。此后，比较操作（外层循环对应 `arith.cmpi`，内层 kernel 循环对应 `neura.icmp`）是循环边界信息的唯一残留证据。

**Bug 修复：** Strategy 1 原先对独立 counter chain 使用 `total += chain_product`（求和）。独立 chain 在 CGRA 上并发执行，正确的聚合方式是 `total = max(total, chain_product)`。

#### 2.4. `performFusion()` — 支持多块的 Region 级融合

**之前（8ee3421）：**
```
performFusion:
  ├─ lowerTaskToPhase1(task_a) → 内部运行 Phase 1
  ├─ lowerTaskToPhase1(task_b)
  ├─ 创建单块体的 fused task
  ├─ 将 Phase 1 task_a 的 op 克隆到体中
  ├─ 将 Phase 1 task_b 的 op 克隆到体中
  ├─ 创建合并的 yield
  └─ 删除 Phase 1 module
```

**之后：**
```
performFusion:
  ├─ 创建 fused task（含入口块 + 合并的块参数）
  ├─ Region::cloneInto(task_a.getBody()) — 保留所有块
  ├─ Region::cloneInto(task_b.getBody())
  ├─ 将克隆的入口块合并到 fused 入口块
  ├─ 识别克隆的 kernels（每个 task 各一个）
  ├─ 创建合并的 fused kernel（含合并的 DFG）
  │   ├─ 合并 inputs、iter_args、result types
  │   ├─ 从两个源 kernel 克隆 DFG op
  │   └─ 创建合并的 neura.yield
  ├─ 替换旧 kernel 结果 → fused kernel 结果
  ├─ 合并 taskflow.yield op
  └─ 删除旧 kernels
```

主要变更：
- **不再有内部 Phase 1。** task 已经是 dataflow IR。
- **支持多块。** 使用 `Region::cloneInto()` 正确克隆 CF lowering 后含多块的 task 体，保留 `llvm.br`/`llvm.cond_br` 控制流。
- **DFG 合并。** 两个 task 各含一个 `neura.kernel`，融合将两个 kernel DFG 并排合并为一个 fused kernel（二者独立，共享 tile 阵列）。
- **Yield 合并。** 通过查找无终止符的块来处理单块和多块两种情况，并添加合并后的 yield。

---

### 3. 测试流水线变更

所有 4 个测试文件在 `--resource-aware-task-optimization` 之前都使用完整的 lowering 流水线：

```
--affine-loop-tree-serialization     ← 新增：序列化循环树以便 task 提取
--affine-loop-perfection             ← 新增：规范化循环结构
--convert-affine-to-taskflow         （原有，位置已调整）
--construct-hyperblock-from-task
--classify-counters                  ← 新增：从 RESOPT 内部移出
--convert-taskflow-to-neura          ← 新增：从 RESOPT 内部移出
--lower-affine
--convert-scf-to-cf
--convert-cf-to-llvm
--assign-accelerator
--lower-memref-to-neura
--lower-arith-to-neura
--lower-builtin-to-neura
--lower-llvm-to-neura
--promote-input-arg-to-const
--fold-constant
--canonicalize-return
--canonicalize-live-in
--leverage-predicated-value
--transform-ctrl-to-data-flow       ← counter op 在此被消耗
--fold-constant
--resource-aware-task-optimization   ← RESOPT 在完整 lowered IR 上运行
```

添加 `--verify-each=false` 是因为中间 pass 可能产生无法通过严格验证的 IR（例如，在放置 `taskflow.yield` 之前存在多块 region）。

### FileCheck 检查模式更新

RESOPT FileCheck 模式已更新以匹配完整 lowered 流水线的实际输出。与旧模式的主要差异：

| 测试 | 旧 cgra_count | 新 cgra_count | 旧 trip_count | 新 trip_count |
|------|--------------|---------------|--------------|---------------|
| irregular-loop Task_0_Task_1_utilfused | 2 | 1 | 32 | 32 |
| irregular-loop Task_2 | 1 | 1 | 32 | 32 |
| multi-nested Task_1 | 2 | 1 | 160 | 160 |
| multi-nested Task_0_Task_2_fused_Task_3_utilfused | 2 | 1 | 192 | 192 |
| multi-nested Task_4 | 2 | 1 | 36 | 36 |
| parallel-nested Task_0_Task_1_utilfused | 2 | 2 | 64 | 64 |
| resnet Task_6_Task_8_utilfused | 1→2 | 2 | 1→4096 | 4096 |

cgra_count 的变化反映了完整 lowered IR 提供更准确 profile 的结果（更小的 compiled_ii 允许平衡算法以更少的 CGRA 数量找到解）。

---

### 4. 设计决策总结

1. **RESOPT 是后 lowering pass。** 不应在内部嵌入转换流水线，所有 taskflow→neura lowering 在 RESOPT 之前完成。
2. **不需要多 hyperblock 支持。** 每个 task 恰好包含一个 `neura.kernel`，`computeTripCount` Strategy 2 中的 assert 强制执行此约束。
3. **Counter chain 是并发的。** CGRA 上的独立 counter chain 并行执行，因此 `trip_count = max(chain_product)`，不是 `sum`。
4. **Strategy 3 处理 post-dataflow IR。** `--transform-ctrl-to-data-flow` 消耗 counter op 后，循环边界信息只残留在比较操作中（外层循环对应 `arith.cmpi`，内层 kernel 循环对应 `neura.icmp`）。
5. **多块任务体是合法的。** CF lowering 后，task 体可能包含多个基本块，`TaskflowOps.td` 放宽为 `AnyRegion`，`performFusion` 使用 `Region::cloneInto()` 保证正确性。
