//===- FuseKernelPass.cpp - Kernel Fusion Pass for Neura Dialect ----------===//
//
// This pass implements kernel fusion for the Neura dialect, inspired by
// MLIR's linalg and affine loop fusion algorithms.
//
// The pass supports two types of fusion:
// 1. Producer-Consumer Fusion: Fuses a producer kernel into its consumer
//    when the producer's output is only used by the consumer.
// 2. Sibling Fusion: Fuses kernels that share the same input operands
//    and have no data dependencies between them.
//
//===----------------------------------------------------------------------===//

#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

#define GEN_PASS_DEF_FUSEKERNEL
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Check if two kernel operations can be fused based on their memory access
/// patterns. Returns true if there's no read-after-write or write-after-write
/// conflict that would prevent fusion.
static bool canFuseKernels(neura::KernelOp producer, neura::KernelOp consumer) {
  // Basic checks
  if (!producer || !consumer)
    return false;

  // Don't fuse if they're the same operation
  if (producer == consumer)
    return false;

  // Check that producer dominates consumer (producer comes before consumer)
  // This is a simplified check - in the same block, check operation order
  if (producer->getBlock() != consumer->getBlock())
    return false;

  // Verify producer comes before consumer
  bool producerBeforeConsumer = false;
  for (Operation &op : *producer->getBlock()) {
    if (&op == producer.getOperation()) {
      producerBeforeConsumer = true;
    }
    if (&op == consumer.getOperation()) {
      break;
    }
  }
  if (!producerBeforeConsumer)
    return false;

  return true;
}

/// Check if there's a producer-consumer relationship between two kernels.
/// Returns true if consumer uses any of producer's results.
static bool hasProducerConsumerRelation(neura::KernelOp producer,
                                         neura::KernelOp consumer) {
  for (Value result : producer.getOutputs()) {
    for (Value input : consumer.getInputs()) {
      if (result == input)
        return true;
    }
  }
  return false;
}

/// Check if two kernels are siblings (share inputs but no data dependency).
static bool areSiblingKernels(neura::KernelOp kernel1, neura::KernelOp kernel2) {
  // Check if they share any input operands
  llvm::SmallPtrSet<Value, 8> kernel1Inputs;
  for (Value input : kernel1.getInputs()) {
    kernel1Inputs.insert(input);
  }

  bool shareInput = false;
  for (Value input : kernel2.getInputs()) {
    if (kernel1Inputs.contains(input)) {
      shareInput = true;
      break;
    }
  }

  if (!shareInput)
    return false;

  // Check there's no producer-consumer relationship
  if (hasProducerConsumerRelation(kernel1, kernel2) ||
      hasProducerConsumerRelation(kernel2, kernel1))
    return false;

  return true;
}

/// Collect all operations between two operations in the same block.
static void collectOperationsBetween(Operation *start, Operation *end,
                                      SmallVectorImpl<Operation *> &ops) {
  if (start->getBlock() != end->getBlock())
    return;

  bool inRange = false;
  for (Operation &op : *start->getBlock()) {
    if (&op == start) {
      inRange = true;
      continue;
    }
    if (&op == end)
      break;
    if (inRange)
      ops.push_back(&op);
  }
}

/// Check if any operation between producer and consumer uses producer's results.
static bool hasInterveningUses(neura::KernelOp producer,
                                neura::KernelOp consumer) {
  SmallVector<Operation *> betweenOps;
  collectOperationsBetween(producer, consumer, betweenOps);

  llvm::SmallPtrSet<Value, 8> producerResults;
  for (Value result : producer.getOutputs()) {
    producerResults.insert(result);
  }

  for (Operation *op : betweenOps) {
    for (Value operand : op->getOperands()) {
      if (producerResults.contains(operand))
        return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Producer-Consumer Fusion Pattern
// Similar to linalg's tiled producer-consumer fusion
//===----------------------------------------------------------------------===//

struct ProducerConsumerFusionPattern
    : public OpRewritePattern<neura::KernelOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::KernelOp consumer,
                                PatternRewriter &rewriter) const override {
    // Find a producer kernel that can be fused into this consumer
    neura::KernelOp producer = nullptr;
    Value fusedValue;

    for (Value input : consumer.getInputs()) {
      if (auto defOp = input.getDefiningOp<neura::KernelOp>()) {
        // Check fusion legality
        if (!canFuseKernels(defOp, consumer))
          continue;

        // Check that producer has only one use (this consumer)
        bool hasOnlyOneUse = true;
        for (Value result : defOp.getOutputs()) {
          if (!result.hasOneUse()) {
            hasOnlyOneUse = false;
            break;
          }
        }
        if (!hasOnlyOneUse)
          continue;

        // Check no intervening uses
        if (hasInterveningUses(defOp, consumer))
          continue;

        producer = defOp;
        fusedValue = input;
        break;
      }
    }

    if (!producer)
      return failure();

    // Create the fused kernel
    Location loc = consumer.getLoc();

    // Collect inputs: producer's inputs + consumer's inputs (excluding fused value)
    SmallVector<Value> fusedInputs;
    SmallVector<Type> fusedInputTypes;

    // Add producer's inputs
    for (Value input : producer.getInputs()) {
      fusedInputs.push_back(input);
      fusedInputTypes.push_back(input.getType());
    }

    // Add consumer's inputs (excluding the fused value from producer)
    for (Value input : consumer.getInputs()) {
      if (input != fusedValue) {
        fusedInputs.push_back(input);
        fusedInputTypes.push_back(input.getType());
      }
    }

    // Output types from consumer
    SmallVector<Type> fusedOutputTypes;
    for (Type t : consumer.getResultTypes()) {
      fusedOutputTypes.push_back(t);
    }

    // Create fused kernel
    auto fusedKernel = rewriter.create<neura::KernelOp>(
        loc, fusedOutputTypes, fusedInputs,
        /*cgra_id=*/consumer.getCgraIdAttr(),
        /*kernel_name=*/rewriter.getStringAttr("fused_producer_consumer"),
        /*accelerator=*/consumer.getAcceleratorAttr());

    // Create the fused kernel body
    Block *fusedBlock = rewriter.createBlock(&fusedKernel.getBody());

    // Add block arguments for all inputs
    for (Type t : fusedInputTypes) {
      fusedBlock->addArgument(t, loc);
    }

    // Map old values to new block arguments
    IRMapping producerMapping;
    IRMapping consumerMapping;

    // Map producer's block arguments
    Block &producerBlock = producer.getBody().front();
    for (auto [oldArg, newArg] :
         llvm::zip(producerBlock.getArguments(),
                   fusedBlock->getArguments().take_front(
                       producer.getInputs().size()))) {
      producerMapping.map(oldArg, newArg);
    }

    // Clone producer's operations into fused kernel
    rewriter.setInsertionPointToStart(fusedBlock);
    Value producerYieldValue;
    for (Operation &op : producerBlock) {
      if (auto yieldOp = dyn_cast<neura::YieldOp>(&op)) {
        // Save the yielded value for the consumer to use
        if (!yieldOp.getOperands().empty()) {
          producerYieldValue = producerMapping.lookup(yieldOp.getOperands()[0]);
        }
        continue; // Don't clone yield yet
      }
      rewriter.clone(op, producerMapping);
    }

    // Map consumer's block arguments
    Block &consumerBlock = consumer.getBody().front();
    unsigned consumerInputIdx = producer.getInputs().size();
    for (auto [idx, oldArg] : llvm::enumerate(consumerBlock.getArguments())) {
      Value originalInput = consumer.getInputs()[idx];
      if (originalInput == fusedValue) {
        // Map to producer's output
        consumerMapping.map(oldArg, producerYieldValue);
      } else {
        // Map to corresponding fused block argument
        consumerMapping.map(oldArg, fusedBlock->getArgument(consumerInputIdx++));
      }
    }

    // Clone consumer's operations into fused kernel
    for (Operation &op : consumerBlock) {
      rewriter.clone(op, consumerMapping);
    }

    // Replace consumer with fused kernel
    rewriter.replaceOp(consumer, fusedKernel.getOutputs());

    // Erase the producer
    rewriter.eraseOp(producer);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Sibling Fusion Pattern
// Similar to affine's sibling loop fusion
//===----------------------------------------------------------------------===//

struct SiblingFusionPattern : public OpRewritePattern<neura::KernelOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::KernelOp kernel1,
                                PatternRewriter &rewriter) const override {
    // Find the next kernel operation in the same block
    neura::KernelOp kernel2 = nullptr;

    for (Operation *op = kernel1->getNextNode(); op; op = op->getNextNode()) {
      if (auto nextKernel = dyn_cast<neura::KernelOp>(op)) {
        if (areSiblingKernels(kernel1, nextKernel) &&
            canFuseKernels(kernel1, nextKernel)) {
          kernel2 = nextKernel;
          break;
        }
      }
    }

    if (!kernel2)
      return failure();

    // Create the fused kernel
    Location loc = kernel1.getLoc();

    // Merge inputs (avoid duplicates)
    SmallVector<Value> fusedInputs;
    SmallVector<Type> fusedInputTypes;
    llvm::SmallDenseMap<Value, unsigned> inputIndexMap;

    // Add kernel1's inputs
    for (Value input : kernel1.getInputs()) {
      inputIndexMap[input] = fusedInputs.size();
      fusedInputs.push_back(input);
      fusedInputTypes.push_back(input.getType());
    }

    // Add kernel2's unique inputs
    for (Value input : kernel2.getInputs()) {
      if (!inputIndexMap.count(input)) {
        inputIndexMap[input] = fusedInputs.size();
        fusedInputs.push_back(input);
        fusedInputTypes.push_back(input.getType());
      }
    }

    // Combine output types
    SmallVector<Type> fusedOutputTypes;
    for (Type t : kernel1.getResultTypes()) {
      fusedOutputTypes.push_back(t);
    }
    for (Type t : kernel2.getResultTypes()) {
      fusedOutputTypes.push_back(t);
    }

    // Create fused kernel
    auto fusedKernel = rewriter.create<neura::KernelOp>(
        loc, fusedOutputTypes, fusedInputs,
        /*cgra_id=*/kernel1.getCgraIdAttr(),
        /*kernel_name=*/rewriter.getStringAttr("fused_sibling"),
        /*accelerator=*/kernel1.getAcceleratorAttr());

    // Create the fused kernel body
    Block *fusedBlock = rewriter.createBlock(&fusedKernel.getBody());

    // Add block arguments
    for (Type t : fusedInputTypes) {
      fusedBlock->addArgument(t, loc);
    }

    // Map and clone kernel1's operations
    IRMapping mapping1;
    Block &block1 = kernel1.getBody().front();
    for (auto [idx, oldArg] : llvm::enumerate(block1.getArguments())) {
      Value originalInput = kernel1.getInputs()[idx];
      mapping1.map(oldArg, fusedBlock->getArgument(inputIndexMap[originalInput]));
    }

    rewriter.setInsertionPointToStart(fusedBlock);
    SmallVector<Value> kernel1Yields;
    for (Operation &op : block1) {
      if (auto yieldOp = dyn_cast<neura::YieldOp>(&op)) {
        for (Value v : yieldOp.getOperands()) {
          kernel1Yields.push_back(mapping1.lookup(v));
        }
        continue;
      }
      rewriter.clone(op, mapping1);
    }

    // Map and clone kernel2's operations
    IRMapping mapping2;
    Block &block2 = kernel2.getBody().front();
    for (auto [idx, oldArg] : llvm::enumerate(block2.getArguments())) {
      Value originalInput = kernel2.getInputs()[idx];
      mapping2.map(oldArg, fusedBlock->getArgument(inputIndexMap[originalInput]));
    }

    SmallVector<Value> kernel2Yields;
    for (Operation &op : block2) {
      if (auto yieldOp = dyn_cast<neura::YieldOp>(&op)) {
        for (Value v : yieldOp.getOperands()) {
          kernel2Yields.push_back(mapping2.lookup(v));
        }
        continue;
      }
      rewriter.clone(op, mapping2);
    }

    // Create combined yield
    SmallVector<Value> allYields;
    allYields.append(kernel1Yields);
    allYields.append(kernel2Yields);
    rewriter.create<neura::YieldOp>(loc, allYields);

    // Replace both kernels
    SmallVector<Value> kernel1Results, kernel2Results;
    for (unsigned i = 0; i < kernel1.getNumResults(); ++i) {
      kernel1Results.push_back(fusedKernel.getResult(i));
    }
    for (unsigned i = 0; i < kernel2.getNumResults(); ++i) {
      kernel2Results.push_back(
          fusedKernel.getResult(kernel1.getNumResults() + i));
    }

    rewriter.replaceOp(kernel1, kernel1Results);
    rewriter.replaceOp(kernel2, kernel2Results);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Elementwise Fusion Pattern
// Similar to linalg's elementwise fusion - fuses operations that can be
// computed element-by-element without changing the iteration domain
//===----------------------------------------------------------------------===//

struct ElementwiseFusionPattern : public OpRewritePattern<neura::FusedOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::FusedOp consumer,
                                PatternRewriter &rewriter) const override {
    // Look for a producer fused_op that feeds into this consumer
    for (Value input : consumer.getInputs()) {
      auto producer = input.getDefiningOp<neura::FusedOp>();
      if (!producer)
        continue;

      // Check single use
      bool singleUse = true;
      for (Value result : producer.getOutputs()) {
        if (!result.hasOneUse()) {
          singleUse = false;
          break;
        }
      }
      if (!singleUse)
        continue;

      // Fuse the two fused_ops
      Location loc = consumer.getLoc();

      // Collect inputs
      SmallVector<Value> fusedInputs;
      for (Value inp : producer.getInputs()) {
        fusedInputs.push_back(inp);
      }
      for (Value inp : consumer.getInputs()) {
        if (inp.getDefiningOp() != producer)
          fusedInputs.push_back(inp);
      }

      // Create new fused_op
      auto newFused = rewriter.create<neura::FusedOp>(
          loc, consumer.getResultTypes(), fusedInputs,
          rewriter.getI64IntegerAttr(consumer.getPatternId() * 100 +
                                     producer.getPatternId()),
          rewriter.getStringAttr(producer.getPatternName().str() + "->" +
                                 consumer.getPatternName().str()),
          rewriter.getI64IntegerAttr(
              std::min(producer.getFrequency(), consumer.getFrequency())));

      // Build the fused body
      Block *newBlock = rewriter.createBlock(&newFused.getBody());
      for (Value inp : fusedInputs) {
        newBlock->addArgument(inp.getType(), loc);
      }

      // Clone producer body
      IRMapping producerMapping;
      Block &producerBlock = producer.getBody().front();
      for (auto [idx, arg] : llvm::enumerate(producerBlock.getArguments())) {
        producerMapping.map(arg, newBlock->getArgument(idx));
      }

      rewriter.setInsertionPointToStart(newBlock);
      Value producerResult;
      for (Operation &op : producerBlock) {
        if (auto yield = dyn_cast<neura::YieldOp>(&op)) {
          if (!yield.getOperands().empty())
            producerResult = producerMapping.lookup(yield.getOperand(0));
          continue;
        }
        rewriter.clone(op, producerMapping);
      }

      // Clone consumer body
      IRMapping consumerMapping;
      Block &consumerBlock = consumer.getBody().front();
      unsigned argIdx = producer.getInputs().size();
      for (auto [idx, arg] : llvm::enumerate(consumerBlock.getArguments())) {
        Value origInput = consumer.getInputs()[idx];
        if (origInput.getDefiningOp() == producer) {
          consumerMapping.map(arg, producerResult);
        } else {
          consumerMapping.map(arg, newBlock->getArgument(argIdx++));
        }
      }

      for (Operation &op : consumerBlock) {
        rewriter.clone(op, consumerMapping);
      }

      rewriter.replaceOp(consumer, newFused.getOutputs());
      rewriter.eraseOp(producer);

      return success();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// FuseKernelPass
//===----------------------------------------------------------------------===//

struct FuseKernelPass
    : public PassWrapper<FuseKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseKernelPass)

  StringRef getArgument() const override { return "fuse-kernel"; }
  StringRef getDescription() const override {
    return "Fuse neura.kernel operations using producer-consumer and sibling "
           "fusion strategies (inspired by linalg/affine fusion).";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Apply fusion patterns with different priorities
    // Higher benefit = higher priority
    RewritePatternSet patterns(&getContext());

    // Producer-consumer fusion has highest priority as it reduces data movement
    patterns.add<ProducerConsumerFusionPattern>(&getContext(), /*benefit=*/10);

    // Sibling fusion is next
    patterns.add<SiblingFusionPattern>(&getContext(), /*benefit=*/5);

    // Elementwise fusion for fused_op
    patterns.add<ElementwiseFusionPattern>(&getContext(), /*benefit=*/8);

    FrozenRewritePatternSet frozen(std::move(patterns));

    // Apply patterns to IsolatedFromAbove operations (func.func, etc.)
    // applyPatternsGreedily requires IsolatedFromAbove trait
    module.walk([&](func::FuncOp funcOp) {
      if (failed(applyPatternsGreedily(funcOp, frozen))) {
        signalPassFailure();
      }
    });

    // Print statistics
    unsigned numKernels = 0;
    module.walk([&](neura::KernelOp) { ++numKernels; });
    llvm::outs() << "[FuseKernelPass] Remaining kernels after fusion: "
                 << numKernels << "\n";
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createFuseKernelPass() {
  return std::make_unique<FuseKernelPass>();
}
} // namespace mlir::neura
