//===- FuseKernelPass.cpp - Kernel Fusion Pass for Neura Dialect ----------===//
//
// This pass implements kernel fusion for the Neura dialect:
// 1. Producer-Consumer Fusion: Fuses a producer kernel into its consumer.
// 2. Sibling Fusion: Fuses kernels that share inputs without data dependency.
//
//===----------------------------------------------------------------------===//

#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "Conversion/ConversionPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

#define GEN_PASS_DEF_FUSEKERNEL
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

// Represents metrics for evaluating fusion profitability.
struct FusionMetrics {
  int rec_mii = 1;
  int res_mii = 1;
  int max_fanout = 0;
  int num_ops = 0;
};

// Calculates the maximum fanout in a block.
int calculateMaxFanoutInBlock(Block &block) {
  int max_fanout = 0;
  for (Operation &op : block) {
    for (Value result : op.getResults()) {
      int fanout = std::distance(result.use_begin(), result.use_end());
      max_fanout = std::max(max_fanout, fanout);
    }
  }
  return max_fanout;
}

// Runs the neura transformation pipeline on a cloned module and computes MII metrics.
FusionMetrics computeRealMetrics(ModuleOp test_module, const neura::Architecture &architecture) {
  FusionMetrics metrics;
  auto cloned_module = test_module.clone();

  PassManager pm(cloned_module.getContext());
  pm.addPass(mlir::neura::createAssignAcceleratorPass());
  pm.addPass(mlir::createLowerArithToNeuraPass());
  pm.addPass(neura::createCanonicalizeReturnPass());
  pm.addPass(neura::createCanonicalizeCastPass());
  pm.addPass(neura::createPromoteFuncArgToConstPass());
  pm.addPass(neura::createCanonicalizeLiveInPass());
  pm.addPass(neura::createLeveragePredicatedValuePass());
  pm.addPass(neura::createTransformCtrlToDataFlowPass());
  pm.enableVerifier(true);

  if (failed(pm.run(cloned_module))) {
    metrics.rec_mii = 100;
    metrics.res_mii = 100;
    cloned_module.erase();
    return metrics;
  }

  cloned_module.walk([&](func::FuncOp func_op) {
    if (func_op.getName() != "test_fused_kernel") {
      return;
    }
    metrics.res_mii = neura::calculateResMii(func_op, architecture);
    auto cycles = neura::collectRecurrenceCycles(func_op);
    metrics.rec_mii = 1;
    for (const auto &cycle : cycles) {
      metrics.rec_mii = std::max(metrics.rec_mii, cycle.length);
    }
    int num_ops = 0;
    func_op.walk([&](Operation *op) {
      if (!isa<func::FuncOp>(op) && !op->hasTrait<OpTrait::IsTerminator>()) {
        ++num_ops;
      }
    });
    metrics.num_ops = num_ops;
    if (!func_op.getBody().empty()) {
      metrics.max_fanout = calculateMaxFanoutInBlock(func_op.getBody().front());
    }
  });

  cloned_module.erase();
  return metrics;
}

// Clones operations from a kernel block, collecting yield values.
void cloneKernelBlockOps(Block &source_block, OpBuilder &builder, IRMapping &mapping, SmallVectorImpl<Value> &yield_values) {
  for (Operation &op : source_block) {
    if (auto yield_op = dyn_cast<neura::YieldOp>(&op)) {
      for (Value v : yield_op.getOperands()) {
        yield_values.push_back(mapping.lookup(v));
      }
      continue;
    }
    builder.clone(op, mapping);
  }
}

// Creates a test function from a kernel's body and returns the function.
func::FuncOp cloneKernelToTestFunction(neura::KernelOp kernel, OpBuilder &builder, Location loc) {
  Block &kernel_block = kernel.getBody().front();

  SmallVector<Type> input_types;
  for (auto arg : kernel_block.getArguments()) {
    input_types.push_back(arg.getType());
  }
  SmallVector<Type> output_types(kernel.getResultTypes());

  if (input_types.empty()) {
    input_types.push_back(builder.getI64Type());
  }
  if (output_types.empty()) {
    output_types.push_back(builder.getI64Type());
  }

  auto func_type = builder.getFunctionType(input_types, output_types);
  auto func_op = builder.create<func::FuncOp>(loc, "test_fused_kernel", func_type);
  func_op->setAttr("accelerator", builder.getStringAttr("neura"));

  Block *entry_block = func_op.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);

  IRMapping mapping;
  for (auto [kernel_arg, func_arg] : llvm::zip(kernel_block.getArguments(), entry_block->getArguments())) {
    mapping.map(kernel_arg, func_arg);
  }

  SmallVector<Value> yield_values;
  cloneKernelBlockOps(kernel_block, builder, mapping, yield_values);

  if (yield_values.empty()) {
    yield_values.push_back(entry_block->getArgument(0));
  }
  auto return_op = builder.create<neura::ReturnOp>(loc, yield_values);
  return_op->setAttr("return_type", builder.getStringAttr("value"));

  return func_op;
}

// Computes metrics for a single kernel by creating a test module.
FusionMetrics computeSingleKernelMetrics(neura::KernelOp kernel, const neura::Architecture &architecture) {
  MLIRContext *ctx = kernel.getContext();
  OpBuilder builder(ctx);

  auto module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());
  cloneKernelToTestFunction(kernel, builder, builder.getUnknownLoc());

  FusionMetrics metrics = computeRealMetrics(module, architecture);
  module.erase();
  return metrics;
}

// Computes metrics for fused kernels by directly merging kernel bodies into a test function.
FusionMetrics computeFusedKernelMetrics(neura::KernelOp kernel1, neura::KernelOp kernel2, bool is_producer_consumer, Value fused_value, const neura::Architecture &architecture) {
  MLIRContext *ctx = kernel1.getContext();
  OpBuilder builder(ctx);
  Location loc = builder.getUnknownLoc();

  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());

  Block &k1_block = kernel1.getBody().front();
  Block &k2_block = kernel2.getBody().front();

  // Collects input types from both kernels.
  SmallVector<Type> input_types;
  for (auto arg : k1_block.getArguments()) {
    input_types.push_back(arg.getType());
  }

  // Finds which kernel2 arg corresponds to fused_value for producer-consumer fusion.
  int fused_value_arg_idx = -1;
  if (is_producer_consumer && fused_value) {
    for (auto [idx, input] : llvm::enumerate(kernel2.getInputs())) {
      if (input == fused_value) {
        fused_value_arg_idx = idx;
        break;
      }
    }
  }

  for (auto [idx, arg] : llvm::enumerate(k2_block.getArguments())) {
    if (static_cast<int>(idx) != fused_value_arg_idx) {
      input_types.push_back(arg.getType());
    }
  }

  // Determines output types based on fusion type.
  SmallVector<Type> output_types;
  if (is_producer_consumer) {
    output_types.append(kernel2.getResultTypes().begin(), kernel2.getResultTypes().end());
  } else {
    output_types.append(kernel1.getResultTypes().begin(), kernel1.getResultTypes().end());
    output_types.append(kernel2.getResultTypes().begin(), kernel2.getResultTypes().end());
  }
  if (input_types.empty()) {
    input_types.push_back(builder.getI64Type());
  }
  if (output_types.empty()) {
    output_types.push_back(builder.getI64Type());
  }

  // Creates test function.
  auto func_type = builder.getFunctionType(input_types, output_types);
  auto func_op = builder.create<func::FuncOp>(loc, "test_fused_kernel", func_type);
  func_op->setAttr("accelerator", builder.getStringAttr("neura"));
  Block *entry_block = func_op.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);

  // Maps kernel1's block arguments to function arguments.
  IRMapping mapping;
  unsigned func_arg_idx = 0;
  for (auto k1_arg : k1_block.getArguments()) {
    mapping.map(k1_arg, entry_block->getArgument(func_arg_idx++));
  }

  // Clones kernel1's operations.
  SmallVector<Value> k1_yields;
  cloneKernelBlockOps(k1_block, builder, mapping, k1_yields);

  // Maps kernel2's block arguments.
  for (auto [idx, k2_arg] : llvm::enumerate(k2_block.getArguments())) {
    if (is_producer_consumer && static_cast<int>(idx) == fused_value_arg_idx) {
      if (!k1_yields.empty()) {
        mapping.map(k2_arg, k1_yields[0]);
      }
    } else {
      mapping.map(k2_arg, entry_block->getArgument(func_arg_idx++));
    }
  }

  // Clones kernel2's operations.
  SmallVector<Value> k2_yields;
  cloneKernelBlockOps(k2_block, builder, mapping, k2_yields);

  // Creates return with appropriate yields.
  SmallVector<Value> return_values;
  if (is_producer_consumer) {
    return_values = k2_yields;
  } else {
    return_values.append(k1_yields.begin(), k1_yields.end());
    return_values.append(k2_yields.begin(), k2_yields.end());
  }
  if (return_values.empty()) {
    return_values.push_back(entry_block->getArgument(0));
  }
  auto return_op = builder.create<neura::ReturnOp>(loc, return_values);
  return_op->setAttr("return_type", builder.getStringAttr("value"));

  FusionMetrics metrics = computeRealMetrics(module, architecture);
  module.erase();
  return metrics;
}

int estimateMII(const FusionMetrics &metrics, int total_ops, int total_tiles) {
  const float alpha = 0.5;
  const float beta = 0.5;
  int mii = std::max(metrics.rec_mii, metrics.res_mii);
  return std::ceil((1.0 + alpha * (total_ops / float(total_tiles))) * (1 + beta * std::max(metrics.max_fanout - 4, 0)) * mii);
}

// Checks if fusion is profitable based on MII and fanout metrics.
bool isFusionProfitable(neura::KernelOp kernel1, neura::KernelOp kernel2, bool is_producer_consumer, Value fused_value = nullptr) {
  neura::Architecture architecture(1, 1, neura::BaseTopology::MESH, 4, 4, neura::BaseTopology::MESH);

  FusionMetrics m1 = computeSingleKernelMetrics(kernel1, architecture);
  FusionMetrics m2 = computeSingleKernelMetrics(kernel2, architecture);
  FusionMetrics fused = computeFusedKernelMetrics(kernel1, kernel2, is_producer_consumer, fused_value, architecture);
  
  return estimateMII(fused, fused.num_ops, architecture.getNumTiles()) <= std::max(estimateMII(m1, m1.num_ops, architecture.getNumTiles()), estimateMII(m2, m2.num_ops, architecture.getNumTiles()));

}

// Checks if two kernels can be fused (same block, producer before consumer).
bool canFuseKernels(neura::KernelOp producer, neura::KernelOp consumer) {
  if (!producer || !consumer || producer == consumer) {
    return false;
  }
  if (producer->getBlock() != consumer->getBlock()) {
    return false;
  }
  return producer->isBeforeInBlock(consumer);
}

// Returns true if consumer uses any of producer's results.
bool hasProducerConsumerRelation(neura::KernelOp producer, neura::KernelOp consumer) {
  for (Value result : producer.getOutputs()) {
    for (Value input : consumer.getInputs()) {
      if (result == input) {
        return true;
      }
    }
  }
  return false;
}

// Checks if two kernels are siblings (share inputs but no data dependency).
bool areSiblingKernels(neura::KernelOp kernel1, neura::KernelOp kernel2) {
  llvm::SmallPtrSet<Value, 8> kernel1_inputs(kernel1.getInputs().begin(), kernel1.getInputs().end());
  bool share_input = llvm::any_of(kernel2.getInputs(), [&](Value input) {
    return kernel1_inputs.contains(input);
  });
  return share_input && !hasProducerConsumerRelation(kernel1, kernel2) && !hasProducerConsumerRelation(kernel2, kernel1);
}

// Checks if any operation between producer and consumer uses producer's results.
bool hasInterveningUses(neura::KernelOp producer, neura::KernelOp consumer) {
  llvm::SmallPtrSet<Value, 8> producer_results(producer.getOutputs().begin(), producer.getOutputs().end());
  bool in_range = false;
  for (Operation &op : *producer->getBlock()) {
    if (&op == producer.getOperation()) {
      in_range = true;
      continue;
    }
    if (&op == consumer.getOperation()) {
      break;
    }
    if (in_range) {
      for (Value operand : op.getOperands()) {
        if (producer_results.contains(operand)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Collects inputs from two kernels, avoiding duplicates.
void collectFusedInputs(OperandRange inputs1, OperandRange inputs2, SmallVectorImpl<Value> &fused_inputs, SmallVectorImpl<Type> &fused_input_types, llvm::SmallDenseMap<Value, unsigned> &input_index_map) {
  for (Value input : inputs1) {
    input_index_map[input] = fused_inputs.size();
    fused_inputs.push_back(input);
    fused_input_types.push_back(input.getType());
  }
  for (Value input : inputs2) {
    if (!input_index_map.count(input)) {
      input_index_map[input] = fused_inputs.size();
      fused_inputs.push_back(input);
      fused_input_types.push_back(input.getType());
    }
  }
}

// Clones operations from a kernel block with input index mapping for sibling fusion.
void cloneKernelOpsWithIndexMap(Block &source_block, Block *fused_block, OpBuilder &builder, IRMapping &mapping, OperandRange kernel_inputs, const llvm::SmallDenseMap<Value, unsigned> &input_index_map, SmallVectorImpl<Value> *yield_values) {
  for (auto [idx, old_arg] : llvm::enumerate(source_block.getArguments())) {
    Value original_input = kernel_inputs[idx];
    mapping.map(old_arg, fused_block->getArgument(input_index_map.lookup(original_input)));
  }
  for (Operation &op : source_block) {
    if (auto yield_op = dyn_cast<neura::YieldOp>(&op)) {
      if (yield_values) {
        for (Value v : yield_op.getOperands()) {
          yield_values->push_back(mapping.lookup(v));
        }
      }
      continue;
    }
    builder.clone(op, mapping);
  }
}

// Fuses a producer kernel into its consumer and returns the fused kernel.
neura::KernelOp fuseProducerConsumerKernels(neura::KernelOp producer, neura::KernelOp consumer, Value fused_value, OpBuilder &builder) {
  Location loc = consumer.getLoc();

  SmallVector<Value> fused_inputs;
  SmallVector<Type> fused_input_types;
  for (Value input : producer.getInputs()) {
    fused_inputs.push_back(input);
    fused_input_types.push_back(input.getType());
  }
  for (Value input : consumer.getInputs()) {
    if (input != fused_value) {
      fused_inputs.push_back(input);
      fused_input_types.push_back(input.getType());
    }
  }

  SmallVector<Type> fused_output_types(consumer.getResultTypes());
  auto fused_kernel = builder.create<neura::KernelOp>(loc, fused_output_types, fused_inputs, consumer.getCgraIdAttr(), builder.getStringAttr("fused_producer_consumer"), consumer.getAcceleratorAttr());

  Block *fused_block = builder.createBlock(&fused_kernel.getBody());
  for (Type t : fused_input_types) {
    fused_block->addArgument(t, loc);
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(fused_block);

  // Maps and clones producer's operations.
  IRMapping producer_mapping;
  Block &producer_block = producer.getBody().front();
  for (auto [old_arg, new_arg] : llvm::zip(producer_block.getArguments(), fused_block->getArguments().take_front(producer.getInputs().size()))) {
    producer_mapping.map(old_arg, new_arg);
  }
  SmallVector<Value> producer_yields;
  cloneKernelBlockOps(producer_block, builder, producer_mapping, producer_yields);

  // Maps and clones consumer's operations with fused value mapped to producer's output.
  IRMapping consumer_mapping;
  Block &consumer_block = consumer.getBody().front();
  unsigned consumer_input_idx = producer.getInputs().size();
  for (auto [idx, old_arg] : llvm::enumerate(consumer_block.getArguments())) {
    Value original_input = consumer.getInputs()[idx];
    if (original_input == fused_value) {
      consumer_mapping.map(old_arg, producer_yields.empty() ? Value() : producer_yields[0]);
    } else {
      consumer_mapping.map(old_arg, fused_block->getArgument(consumer_input_idx++));
    }
  }
  SmallVector<Value> consumer_yields;
  cloneKernelBlockOps(consumer_block, builder, consumer_mapping, consumer_yields);

  builder.create<neura::YieldOp>(loc, consumer_yields);
  return fused_kernel;
}

// Fuses two sibling kernels and returns the fused kernel.
neura::KernelOp fuseSiblingKernels(neura::KernelOp kernel1, neura::KernelOp kernel2, OpBuilder &builder) {
  Location loc = kernel1.getLoc();

  SmallVector<Value> fused_inputs;
  SmallVector<Type> fused_input_types;
  llvm::SmallDenseMap<Value, unsigned> input_index_map;
  collectFusedInputs(kernel1.getInputs(), kernel2.getInputs(), fused_inputs, fused_input_types, input_index_map);

  SmallVector<Type> fused_output_types(kernel1.getResultTypes());
  fused_output_types.append(kernel2.getResultTypes().begin(), kernel2.getResultTypes().end());

  auto fused_kernel = builder.create<neura::KernelOp>(loc, fused_output_types, fused_inputs, kernel1.getCgraIdAttr(), builder.getStringAttr("fused_sibling"), kernel1.getAcceleratorAttr());

  Block *fused_block = builder.createBlock(&fused_kernel.getBody());
  for (Type t : fused_input_types) {
    fused_block->addArgument(t, loc);
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(fused_block);

  IRMapping mapping1;
  Block &block1 = kernel1.getBody().front();
  SmallVector<Value> kernel1_yields;
  cloneKernelOpsWithIndexMap(block1, fused_block, builder, mapping1, kernel1.getInputs(), input_index_map, &kernel1_yields);

  IRMapping mapping2;
  Block &block2 = kernel2.getBody().front();
  SmallVector<Value> kernel2_yields;
  cloneKernelOpsWithIndexMap(block2, fused_block, builder, mapping2, kernel2.getInputs(), input_index_map, &kernel2_yields);

  SmallVector<Value> all_yields(kernel1_yields);
  all_yields.append(kernel2_yields);
  builder.create<neura::YieldOp>(loc, all_yields);

  return fused_kernel;
}

// Pattern that fuses a producer kernel into its consumer.
struct ProducerConsumerFusion : public OpRewritePattern<neura::KernelOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::KernelOp consumer, PatternRewriter &rewriter) const override {
    neura::KernelOp producer = nullptr;
    Value fused_value;

    for (Value input : consumer.getInputs()) {
      auto def_op = input.getDefiningOp<neura::KernelOp>();
      if (!canFuseKernels(def_op, consumer)) {
        continue;
      }
      bool has_only_one_use = llvm::all_of(def_op.getOutputs(), [](Value result) {
        return result.hasOneUse() || result.use_empty();
      });
      if (!has_only_one_use || hasInterveningUses(def_op, consumer)) {
        continue;
      }
      if (!isFusionProfitable(def_op, consumer, true, input)) {
        continue;
      }
      producer = def_op;
      fused_value = input;
      break;
    }

    if (!producer) {
      return failure();
    }

    auto fused_kernel = fuseProducerConsumerKernels(producer, consumer, fused_value, rewriter);
    rewriter.replaceOp(consumer, fused_kernel.getOutputs());
    rewriter.eraseOp(producer);
    return success();
  }
};

// Pattern that fuses kernels sharing the same inputs without data dependencies.
struct SiblingFusion : public OpRewritePattern<neura::KernelOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::KernelOp kernel1, PatternRewriter &rewriter) const override {
    neura::KernelOp kernel2 = nullptr;

    for (Operation *op = kernel1->getNextNode(); op; op = op->getNextNode()) {
      if (auto next_kernel = dyn_cast<neura::KernelOp>(op)) {
        if (areSiblingKernels(kernel1, next_kernel) && canFuseKernels(kernel1, next_kernel) && isFusionProfitable(kernel1, next_kernel, false)) {
          kernel2 = next_kernel;
          break;
        }
      }
    }

    if (!kernel2) {
      return failure();
    }

    auto fused_kernel = fuseSiblingKernels(kernel1, kernel2, rewriter);

    SmallVector<Value> kernel1_results, kernel2_results;
    for (unsigned i = 0; i < kernel1.getNumResults(); ++i) {
      kernel1_results.push_back(fused_kernel.getResult(i));
    }
    for (unsigned i = 0; i < kernel2.getNumResults(); ++i) {
      kernel2_results.push_back(fused_kernel.getResult(kernel1.getNumResults() + i));
    }

    rewriter.replaceOp(kernel1, kernel1_results);
    rewriter.replaceOp(kernel2, kernel2_results);
    return success();
  }
};

// Pass that fuses neura.kernel operations using producer-consumer and sibling fusion.
struct FuseKernelPass : public PassWrapper<FuseKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseKernelPass)

  StringRef getArgument() const override { return "fuse-kernel"; }
  StringRef getDescription() const override { return "Fuses neura.kernel operations using producer-consumer and sibling fusion."; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<ProducerConsumerFusion>(&getContext(), 10);
    patterns.add<SiblingFusion>(&getContext(), 5);

    FrozenRewritePatternSet frozen(std::move(patterns));
    module.walk([&](func::FuncOp func_op) {
      if (failed(applyPatternsGreedily(func_op, frozen))) {
        signalPassFailure();
      }
    });

    unsigned num_kernels = 0;
    module.walk([&](neura::KernelOp) { ++num_kernels; });
    llvm::outs() << "[FuseKernelPass] Remaining kernels after fusion: " << num_kernels << "\n";
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createFuseKernelPass() {
  return std::make_unique<FuseKernelPass>();
}
} // namespace mlir::neura
