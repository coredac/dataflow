#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace mlir;

#define GEN_PASS_DEF_TRANSFORMTOSTEERCONTROL
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
class OperationsToErase {
public:
  void markForErasure(Operation *op) {
    if (op) {
      ops_to_erase.insert(op);
    }
  }

  void eraseMarkedOperations() {
    for (auto it = this->ops_to_erase.rbegin(); it != this->ops_to_erase.rend();
         ++it) {
      if (!(*it)->use_empty()) {
        continue;
      }
      (*it)->erase();
    }
    ops_to_erase.clear();
  }

private:
  llvm::SetVector<Operation *> ops_to_erase;
};

class LoopAnalyzer {
public:
  struct LoopRecurrenceInfo {
    // The reserve operation that starts the loop.
    neura::ReserveOp reserve_op;
    // The phi operation that merges values from different iterations.
    neura::PhiOp phi_op;
    // The initial value before the loop starts.
    Value initial_value;
    // The condition that controls the loop continuation.
    Value condition;
    // The value that is passed back to the next iteration.
    Value backward_value;
    // Whether the loop is invariant (i.e., does not depend on the loop body).
    bool is_invariant;
  };

  LoopAnalyzer(func::FuncOp func) {
    // Map from values to their corresponding reserve operations.
    llvm::DenseMap<Value, neura::ReserveOp> value_to_reserve_map;

    func.walk([&](neura::ReserveOp op) {
      value_to_reserve_map[op.getResult()] = op;
    });

    llvm::DenseMap<Value, llvm::SmallVector<std::pair<Value, neura::CtrlMovOp>>>
        target_to_source_ctrl_mov_map;

    func.walk([&](neura::CtrlMovOp op) {
      target_to_source_ctrl_mov_map[op.getTarget()].push_back(
          {op.getValue(), op});
    });

    // Analyzes phi operations and the backward edges.
    func.walk([&](neura::PhiOp phi_op) {
      for (Value input : phi_op->getOperands()) {
        auto reserve_it = value_to_reserve_map.find(input);
        if (reserve_it == value_to_reserve_map.end()) {
          continue;
        }

        neura::ReserveOp reserve_op = reserve_it->second;
        auto ctrl_mov_it =
            target_to_source_ctrl_mov_map.find(reserve_op.getResult());
        assert(ctrl_mov_it != target_to_source_ctrl_mov_map.end() &&
               "Reserve output must be a target of a ctrl_mov operation");

        for (auto &[source_value, ctrl_mov_op] : ctrl_mov_it->second) {
          Value initial_value = nullptr;
          for (Value phi_input : phi_op->getOperands()) {
            if (phi_input != reserve_op.getResult()) {
              initial_value = phi_input;
              break;
            }
          }
          assert(initial_value != nullptr && "Phi must have an initial value");

          Value condition = nullptr;
          neura::GrantPredicateOp grant_op = nullptr;
          for (auto phi_user : phi_op->getUsers()) {
            if (isa<neura::GrantPredicateOp>(phi_user)) {
              auto def_op = llvm::dyn_cast<neura::GrantPredicateOp>(phi_user);
              if (def_op.getValue() == phi_op.getResult()) {
                condition = def_op.getPredicate();
                grant_op = def_op;
                break;
              }
            }
          }

          assert(condition && grant_op &&
                 "Phi must have a corresponding grant_predicate operation");

          // Checks if the source_value is a loop invariant.
          bool is_invariant = false;
          if (source_value == grant_op.getResult()) {
            is_invariant = true;
          }

          // Records the loop information.
          this->loop_recurrences.push_back({reserve_op, phi_op, initial_value,
                                            condition, source_value,
                                            is_invariant});

          // Maps the phi operation to its loop recurrence index.
          this->phi_to_loop_recurrences[phi_op.getResult()] =
              loop_recurrences.size() - 1;

          // Records the reserve operations that are part of loops.
          this->loop_reserves.insert(reserve_op.getResult());
        }
      }
    });
  }

  const llvm::SmallVector<LoopRecurrenceInfo> &getLoopRecurrences() const {
    return loop_recurrences;
  }

  bool isLoopReserve(Value value) const {
    return loop_reserves.contains(value);
  }

  bool isLoopPhi(Value value) const {
    return phi_to_loop_recurrences.contains(value);
  }

  const LoopRecurrenceInfo *getLoopRecurrenceInfo(Value phi_value) const {
    auto it = phi_to_loop_recurrences.find(phi_value);
    if (it != phi_to_loop_recurrences.end()) {
      return &loop_recurrences[it->second];
    }
    return nullptr;
  }

private:
  llvm::SmallVector<LoopRecurrenceInfo> loop_recurrences;
  llvm::DenseMap<Value, unsigned> phi_to_loop_recurrences;
  llvm::DenseSet<Value> loop_reserves;
};

class BackwardValueHandler {
public:
  BackwardValueHandler(PatternRewriter &rewriter) : rewriter(rewriter) {}

  Value createReserveForBackwardValue(Value backward_value,
                                      Operation *insertion_point) {
    auto it = backward_value_reserve_map.find(backward_value);
    if (it != backward_value_reserve_map.end()) {
      return it->second;
    }

    llvm::errs() << "[ctrl2steer] Creating reserve for backward value: "
                 << backward_value << "\n";

    // Creates the reserve operation for the backward value.
    this->rewriter.setInsertionPointToStart(insertion_point->getBlock());
    auto reserve_op = this->rewriter.create<neura::ReserveOp>(
        backward_value.getLoc(), backward_value.getType());
    this->backward_value_reserve_map[backward_value] = reserve_op.getResult();

    llvm::errs() << "[ctrl2steer] Creating ctrl_mov for backward value: "
                 << backward_value << "\n";

    // Creates a ctrl_mov operation to move the backward value into the reserve.
    this->rewriter.setInsertionPointAfter(backward_value.getDefiningOp());
    this->rewriter.create<neura::CtrlMovOp>(
        backward_value.getLoc(), backward_value, reserve_op.getResult());

    return reserve_op.getResult();
  }

private:
  PatternRewriter &rewriter;
  // Map from backward values to their corresponding reserve values.
  llvm::DenseMap<Value, Value> backward_value_reserve_map;
};

class PhiToCarryPattern : public OpRewritePattern<neura::PhiOp> {
public:
  PhiToCarryPattern(MLIRContext *context, const LoopAnalyzer &loop_analyzer,
                    BackwardValueHandler &backward_value_handler,
                    OperationsToErase &ops_to_erase)
      : OpRewritePattern<neura::PhiOp>(context), loop_analyzer(loop_analyzer),
        backward_value_handler(backward_value_handler),
        ops_to_erase(ops_to_erase) {}

  LogicalResult matchAndRewrite(neura::PhiOp phi_op,
                                PatternRewriter &rewriter) const override {
    // If the phi operation is not part of a loop, we do not handle it here.
    if (!loop_analyzer.isLoopPhi(phi_op.getResult())) {
      return failure();
    }

    const auto *loop_recurrence_info =
        loop_analyzer.getLoopRecurrenceInfo(phi_op.getResult());
    assert(loop_recurrence_info && "Loop recurrence info must be available");

    // Creates a reserve operation for the loop condition.
    Value condition = loop_recurrence_info->condition;
    assert(condition && "Loop condition must be available");
    Value condition_reserve =
        this->backward_value_handler.createReserveForBackwardValue(condition,
                                                                   phi_op);

    // Creates a carry or a invariant operation based on whether the loop
    // recurrence is invariant.
    rewriter.setInsertionPoint(phi_op);
    Value result;
    if (loop_recurrence_info->is_invariant) {
      auto invariant_op = rewriter.create<neura::InvariantOp>(
          phi_op.getLoc(), phi_op.getType(),
          loop_recurrence_info->initial_value, condition_reserve);
      result = invariant_op.getResult();
      // rewriter.replaceOp(phi_op, invariant_op.getResult());
    } else {
      Value backward_reserve =
          this->backward_value_handler.createReserveForBackwardValue(
              loop_recurrence_info->backward_value, phi_op);
      auto carry_op =
          rewriter.create<neura::CarryOp>(phi_op.getLoc(), phi_op.getType(),
                                          loop_recurrence_info->initial_value,
                                          condition_reserve, backward_reserve);
      result = carry_op.getResult();
      // rewriter.replaceOp(phi_op, carry_op.getResult());
    }

    llvm::SmallVector<neura::GrantPredicateOp> related_grant_ops;
    for (auto *user : phi_op->getUsers()) {
      if (auto grant_op = dyn_cast<neura::GrantPredicateOp>(user)) {
        if (grant_op.getValue() == phi_op.getResult()) {
          related_grant_ops.push_back(grant_op);
        }
      }
    }

    // Marks the related operations for erasure.
    for (auto grant_op : related_grant_ops) {
      rewriter.replaceAllOpUsesWith(grant_op, result);
      ops_to_erase.markForErasure(grant_op);
    }

    rewriter.replaceOp(phi_op, result);

    this->ops_to_erase.markForErasure(loop_recurrence_info->reserve_op);
    for (auto *user : loop_recurrence_info->reserve_op->getUsers()) {
      if (auto ctrl_mov_op = dyn_cast<neura::CtrlMovOp>(user)) {
        ops_to_erase.markForErasure(ctrl_mov_op);
      }
    }
    return success();
  }

private:
  const LoopAnalyzer &loop_analyzer;
  BackwardValueHandler &backward_value_handler;
  OperationsToErase &ops_to_erase;
};

class MergePatternFinder {
public:
  struct MergeCandidate {
    Value condition;
    Value true_value;
    Value false_value;
    neura::GrantPredicateOp true_grant;
    neura::GrantPredicateOp false_grant;
  };

  MergePatternFinder(func::FuncOp func) {
    // Collects all the not operations.
    llvm::DenseMap<Value, Value> condition_to_negated;
    llvm::DenseMap<Value, Value> negated_to_condition;

    func.walk([&](neura::NotOp not_op) {
      condition_to_negated[not_op.getInput()] = not_op.getResult();
      negated_to_condition[not_op.getResult()] = not_op.getInput();
    });

    // Collects all the grant_predicate operations based on their conditions.
    llvm::DenseMap<Value, llvm::SmallVector<neura::GrantPredicateOp>>
        condition_to_grants;

    func.walk([&](neura::GrantPredicateOp grant_op) {
      condition_to_grants[grant_op.getPredicate()].push_back(grant_op);
    });

    // Finds pairs of grant_predicate operations that can be merged.
    for (auto &entry : condition_to_grants) {
      Value condition = entry.first;
      auto &true_grants = entry.second;

      auto neg_it = condition_to_negated.find(condition);
      if (neg_it == condition_to_negated.end()) {
        continue;
      }

      Value negated_condition = neg_it->second;

      auto false_it = condition_to_grants.find(negated_condition);
      if (false_it == condition_to_grants.end()) {
        continue;
      }

      auto &false_grants = false_it->second;

      for (auto true_grant : true_grants) {
        Value true_value = true_grant.getValue();

        for (auto false_grant : false_grants) {
          Value false_value = false_grant.getValue();

          if (true_value.getType() == false_value.getType() &&
              true_grant.getResult().getType() ==
                  false_grant.getResult().getType()) {
            merge_candidates.push_back(
                {condition, true_value, false_value, true_grant, false_grant});
          }
        }
      }
    }
  }

  llvm::SmallVector<MergeCandidate> getMergeCandidates() const {
    return this->merge_candidates;
  }

private:
  llvm::SmallVector<MergeCandidate> merge_candidates;
};

class GrantPairToMergePattern : public RewritePattern {
public:
  GrantPairToMergePattern(MLIRContext *context,
                          const MergePatternFinder &merge_pattern_finder,
                          OperationsToErase &ops_to_erase)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context),
        merge_pattern_finder(merge_pattern_finder), ops_to_erase(ops_to_erase) {
  }
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto grant_op = dyn_cast<neura::GrantPredicateOp>(op);
    if (!grant_op) {
      return failure();
    }

    for (auto &candidate : this->merge_pattern_finder.getMergeCandidates()) {
      if (grant_op == candidate.true_grant) {
        rewriter.setInsertionPoint(grant_op);
        auto merge_op = rewriter.create<neura::MergeOp>(
            grant_op.getLoc(), grant_op.getResult().getType(),
            candidate.condition, candidate.true_value, candidate.false_value);

        if (auto not_op = candidate.false_grant.getPredicate()
                              .getDefiningOp<neura::NotOp>()) {
          ops_to_erase.markForErasure(not_op);
        }
        rewriter.replaceOp(candidate.true_grant, merge_op.getResult());
        rewriter.replaceOp(candidate.false_grant, merge_op.getResult());

        return success();
      } else if (grant_op == candidate.false_grant) {
        return failure();
      }
    }

    return failure();
  }

private:
  const MergePatternFinder &merge_pattern_finder;
  OperationsToErase &ops_to_erase;
};

class GrantPredicateToSteerPattern
    : public OpRewritePattern<neura::GrantPredicateOp> {
public:
  GrantPredicateToSteerPattern(MLIRContext *context,
                               OperationsToErase &ops_to_erase)
      : OpRewritePattern<neura::GrantPredicateOp>(context),
        ops_to_erase(ops_to_erase) {}

  LogicalResult matchAndRewrite(neura::GrantPredicateOp grant_op,
                                PatternRewriter &rewriter) const override {
    Value value = grant_op.getValue();
    Value condition = grant_op.getPredicate();

    if (auto not_op = condition.getDefiningOp<neura::NotOp>()) {
      // If the condition is a Not operation, we can transform it to the
      // false_steer.
      rewriter.setInsertionPoint(grant_op);
      auto false_steer = rewriter.create<neura::FalseSteerOp>(
          grant_op.getLoc(), value.getType(), value, not_op.getInput());
      rewriter.replaceOp(grant_op, false_steer.getResult());
      ops_to_erase.markForErasure(not_op);
    } else {
      // Otherwise, we transform it to the true_steer.
      rewriter.setInsertionPoint(grant_op);
      auto true_steer = rewriter.create<neura::TrueSteerOp>(
          grant_op.getLoc(), value.getType(), value, condition);
      rewriter.replaceOp(grant_op, true_steer.getResult());
    }

    return success();
  }

private:
  OperationsToErase &ops_to_erase;
};

class GrantOnceRemovalPattern : public OpRewritePattern<neura::GrantOnceOp> {
public:
  GrantOnceRemovalPattern(MLIRContext *context)
      : OpRewritePattern<neura::GrantOnceOp>(context) {}

  LogicalResult matchAndRewrite(neura::GrantOnceOp grant_once_op,
                                PatternRewriter &rewriter) const override {
    Value input = grant_once_op.getValue();

    rewriter.replaceOp(grant_once_op, input);
    return success();
  }
};

struct TransformToSteerControlPass
    : public PassWrapper<TransformToSteerControlPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransformToSteerControlPass)

  StringRef getArgument() const override {
    return "transform-to-steer-control";
  }

  StringRef getDescription() const override {
    return "Transform control flow into data flow using steer control "
           "operations.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext &context = getContext();
    PatternRewriter rewriter(&context);
    OperationsToErase ops_to_erase;

    RewritePatternSet grant_once_patterns(&context);
    grant_once_patterns.add<GrantOnceRemovalPattern>(&context);
    if (failed(applyPatternsGreedily(func, std::move(grant_once_patterns)))) {
      signalPassFailure();
    }

    LoopAnalyzer loop_analyzer(func);
    BackwardValueHandler backward_value_handler(rewriter);
    MergePatternFinder merge_pattern_finder(func);

    RewritePatternSet phi_patterns(&context);
    phi_patterns.add<PhiToCarryPattern>(&context, loop_analyzer,
                                        backward_value_handler, ops_to_erase);
    if (failed(applyPatternsGreedily(func, std::move(phi_patterns)))) {
      signalPassFailure();
    }
    // Erases the marked operations after processing all phi operations.
    ops_to_erase.eraseMarkedOperations();

    RewritePatternSet merge_patterns(&context);
    merge_patterns.add<GrantPairToMergePattern>(&context, merge_pattern_finder,
                                                ops_to_erase);
    if (failed(applyPatternsGreedily(func, std::move(merge_patterns)))) {
      signalPassFailure();
    }
    // Erases the marked operations after processing all merge patterns.
    ops_to_erase.eraseMarkedOperations();

    RewritePatternSet steer_patterns(&context);
    steer_patterns.add<GrantPredicateToSteerPattern>(&context, ops_to_erase);

    if (failed(applyPatternsGreedily(func, std::move(steer_patterns)))) {
      signalPassFailure();
    }
    // Erases the marked operations after processing all grant_predicate
    // operations.
    ops_to_erase.eraseMarkedOperations();

    // Checks if the function is now in predicate mode.
    auto dataflow_mode_attr = func->getAttrOfType<StringAttr>("dataflow_mode");
    if (!dataflow_mode_attr || dataflow_mode_attr.getValue() != "predicate") {
      func.emitError("transform-to-steer-control requires function to be in "
                     "predicate mode");
      signalPassFailure();
      return;
    }
    // Changes the dataflow_mode attribute to "steering".
    func->setAttr("dataflow_mode", StringAttr::get(&context, "steering"));
    llvm::errs()
        << "[ctrl2steer] Changed dataflow mode from predicate to steering "
           "for function: "
        << func.getName() << "\n";
  }
};
} // namespace

namespace mlir {
namespace neura {

std::unique_ptr<Pass> createTransformToSteerControlPass() {
  return std::make_unique<TransformToSteerControlPass>();
}

} // namespace neura
} // namespace mlir