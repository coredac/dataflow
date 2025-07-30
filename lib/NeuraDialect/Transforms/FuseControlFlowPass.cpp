#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <memory>

using namespace mlir;

#define GEN_PASS_DEF_FUSECONTROLFLOW
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
// A class to hold loop information for the control flow fusion pass.
class LoopInfo {
public:
  // Key operations in a loop.
  Value index_reserve_val; // Reserve values for index.
  Value index_phi_val;
  Value condition_val;
  Value not_condition_val;

  // Loop iteration parameters.
  Value start_val; // Start value for the loop index.
  Value end_val;   // End value for the loop index.
  Value step_val;  // Step value for the loop index.

  // Backward edge information.
  Operation *index_ctrl_mov = nullptr; // Initialized to nullptr.
  Operation *index_grant_op =
      nullptr; // The grant_predicate operation for the index.

  // Used for replace and update operations.
  llvm::SetVector<Operation *> ops_to_remove;
  llvm::MapVector<Value, SmallVector<std::pair<Operation *, unsigned>>>
      users_to_update;

  // Parent loop when handling nested loops, if any.
  LoopInfo *parent_loop = nullptr;

  // Adds operations to remove.
  void addOpToRemove(Operation *op) {
    if (op) {
      this->ops_to_remove.insert(op);
    }
  }

  // Checks if the loop info is complete.
  // There is no not_condition_val because it is derived from condition_val.
  bool isComplete() const {
    return index_reserve_val && index_phi_val && condition_val && start_val &&
           end_val && step_val && index_ctrl_mov;
  }

  // Records the users that use the loop index and (not-)condition values.
  void recordUsersToUpdate() {
    recordUsersFor(this->index_phi_val);
    recordUsersFor(this->index_reserve_val);
    recordUsersFor(this->index_grant_op->getResult(0));
    recordUsersFor(this->condition_val);
    if (this->not_condition_val) {
      recordUsersFor(this->not_condition_val);
    }
  }

private:
  // Records users of a value.
  void recordUsersFor(Value val) {
    if (!val) {
      return;
    }
    for (OpOperand &use : val.getUses()) {
      Operation *user = use.getOwner();
      // Records the user that will not be removed.
      if (!ops_to_remove.contains(user)) {
        users_to_update[val].push_back({user, use.getOperandNumber()});
      }
    }
  }
};

// Finds the original parameter for start value.
Value findOriginalConstant(Value val,
                           llvm::SetVector<Operation *> &ops_to_remove) {
  if (!val || !val.getDefiningOp())
    return val;

  Operation *def_op = val.getDefiningOp();

  // If the value is already a constant, return it.
  if (auto const_op = dyn_cast<neura::ConstantOp>(def_op)) {
    return val;
  }

  // Handle grant operations and add them to the removal list.
  if (auto grant_once_op = dyn_cast<neura::GrantOnceOp>(def_op)) {
    ops_to_remove.insert(def_op);
    return findOriginalConstant(grant_once_op.getValue(), ops_to_remove);
  }

  // For grant_predicate, only track value inputs and ignore condition inputs.
  if (auto grant_predicate_op = dyn_cast<neura::GrantPredicateOp>(def_op)) {
    ops_to_remove.insert(def_op);
    return findOriginalConstant(grant_predicate_op.getValue(), ops_to_remove);
  }

  return val;
}

// Identifies a simple loop.
// The pattern is: reserve -> phi -> icmp -> [not] -> grant_predicate ->
// ctrl_mov <- add
std::unique_ptr<LoopInfo> identifyLoop(Operation *index_reserve_op) {
  if (!isa<neura::ReserveOp>(index_reserve_op)) {
    return nullptr;
  }

  // Starts from the reserve operation for loop index.
  auto loop = std::make_unique<LoopInfo>();
  loop->index_reserve_val = index_reserve_op->getResult(0);
  loop->addOpToRemove(index_reserve_op);

  // Identifies the phi operation.
  neura::PhiOp index_phi_op = nullptr;
  for (Operation *user : loop->index_reserve_val.getUsers()) {
    if (auto phi = dyn_cast<neura::PhiOp>(user)) {
      index_phi_op = phi;
      break;
    }
  }

  if (!index_phi_op) {
    llvm::errs()
        << "[CtrlFlowFuse] No index phi operation found for the loop.\n";
    return nullptr; // No phi operation found.
  }

  loop->index_phi_val = index_phi_op.getResult();
  loop->addOpToRemove(index_phi_op);

  // Finds the start value for loop index.
  Value initial_value = nullptr;
  for (Value input : index_phi_op.getInputs()) {
    if (input != loop->index_reserve_val) {
      initial_value = input;
      break;
    }
  }

  if (!initial_value) {
    llvm::errs()
        << "[CtrlFlowFuse] No initial value found for the loop index.\n";
    return nullptr; // No start value found.
  }

  assert(initial_value && initial_value.getDefiningOp() &&
         isa<neura::GrantOnceOp>(initial_value.getDefiningOp()) &&
         "The initial_value should be defined by a GrantOnceOp.");
  loop->start_val = initial_value;

  // Identifies the phi->icmp->[not]->grant_predicate pattern.
  for (Operation *phi_user : index_phi_op->getUsers()) {
    if (neura::ICmpOp icmp_op = dyn_cast<neura::ICmpOp>(phi_user)) {
      if (icmp_op.getCmpType() == "slt" &&
          icmp_op.getLhs() == loop->index_phi_val) {
        loop->condition_val = icmp_op.getResult();
        loop->end_val = icmp_op.getRhs();
        loop->addOpToRemove(icmp_op);

        // Identifies the not operation if it exists.
        for (Operation *cond_user : icmp_op->getUsers()) {
          if (neura::NotOp not_op = dyn_cast<neura::NotOp>(cond_user)) {
            loop->not_condition_val = not_op.getResult();
            loop->addOpToRemove(not_op);
            break;
          }
        }

        // Identifies the grant_predicate operation for the index_phi_val.
        for (Operation *cond_user : icmp_op->getUsers()) {
          if (neura::GrantPredicateOp grant_predicate_op =
                  dyn_cast<neura::GrantPredicateOp>(cond_user)) {
            if (grant_predicate_op.getValue() == loop->index_phi_val &&
                grant_predicate_op.getPredicate() == loop->condition_val) {
              loop->index_grant_op = grant_predicate_op;
              loop->addOpToRemove(grant_predicate_op);
              break;
            }
          }
        }

        // Identifies the recurrence cycle of the end value.
        Operation *end_val_def_op = loop->end_val.getDefiningOp();
        if (auto end_phi_op = dyn_cast_or_null<neura::PhiOp>(end_val_def_op)) {
          // Identifies the end value's reserve operation.
          Value end_reserve_val = nullptr;
          for (Value input : end_phi_op.getInputs()) {
            if (auto reserve_op = input.getDefiningOp<neura::ReserveOp>()) {
              end_reserve_val = input;
              loop->addOpToRemove(reserve_op);
              break;
            }
          }

          if (end_reserve_val) {
            // Identifies the end ctrl_mov operation.
            for (Operation *user : end_reserve_val.getUsers()) {
              if (auto ctrl_mov_op = dyn_cast<neura::CtrlMovOp>(user)) {
                if (ctrl_mov_op.getTarget() == end_reserve_val) {
                  loop->addOpToRemove(ctrl_mov_op);
                  loop->addOpToRemove(end_phi_op);
                  if (isa<neura::GrantPredicateOp>(
                          ctrl_mov_op.getValue().getDefiningOp())) {
                    loop->addOpToRemove(ctrl_mov_op.getValue().getDefiningOp());
                  }
                  // Finds the actual end value from the inputs of the
                  // end_phi_op.
                  for (Value input : end_phi_op.getInputs()) {
                    if (input != end_reserve_val) {
                      loop->end_val =
                          findOriginalConstant(input, loop->ops_to_remove);
                      break;
                    }
                  }
                  break;
                }
              }
            }
          }
        }
        break;
      } else {
        // TODO: Adds support for other compare types if needed.
        if (icmp_op.getCmpType() != "slt") {
          llvm::errs() << "[CtrlFlowFuse] Unsupported compare type: "
                       << icmp_op.getCmpType() << "\n";
        } else {
          llvm::errs() << "[CtrlFlowFuse] Loop condition does not match "
                          "expected value.\n";
        }
        return nullptr; // Unsupported compare type.
      }
    }
  }

  if (!loop->condition_val || !loop->end_val || !loop->index_phi_val) {
    llvm::errs() << "[CtrlFlowFuse] Incomplete loop information.\n";
    return nullptr; // Incomplete loop.
  }

  // Identifies the ctrl_mov<-add pattern.
  for (Operation *user : loop->index_reserve_val.getUsers()) {
    if (neura::CtrlMovOp ctrl_mov_op = dyn_cast<neura::CtrlMovOp>(user)) {
      if (ctrl_mov_op.getTarget() == loop->index_reserve_val) {
        loop->index_ctrl_mov = ctrl_mov_op;
        loop->addOpToRemove(ctrl_mov_op);

        if (neura::AddOp add_op =
                ctrl_mov_op.getValue().getDefiningOp<neura::AddOp>()) {
          loop->addOpToRemove(add_op);
          Value granted_index = loop->index_grant_op->getResult(0);
          if (add_op.getLhs() == granted_index) {
            loop->step_val =
                findOriginalConstant(add_op.getRhs(), loop->ops_to_remove);
          } else if (add_op.getRhs() == granted_index) {
            loop->step_val =
                findOriginalConstant(add_op.getLhs(), loop->ops_to_remove);
          }
        }
        break;
      }
    }
  }

  if (!loop->index_ctrl_mov || !loop->step_val) {
    llvm::errs() << "[CtrlFlowFuse] Incomplete loop information: ctrl_mov or "
                    "step value not found.\n";
    return nullptr; // Incomplete loop.
  }

  if (loop->isComplete()) {
    loop->recordUsersToUpdate();
    return loop;
  }

  return nullptr; // Incomplete loop.
}

Value createConstantPredicate(PatternRewriter &rewriter, Location loc,
                              bool value) {
  auto predicated_type = rewriter.getType<neura::PredicatedValue>(
      rewriter.getI1Type(), rewriter.getI1Type());
  return rewriter.create<neura::ConstantOp>(loc, predicated_type,
                                            rewriter.getBoolAttr(value),
                                            rewriter.getBoolAttr(true));
}

Operation *findDefiningOp(Value value) {
  if (!value) {
    return nullptr;
  }
  return value.getDefiningOp();
}

LogicalResult replaceWithLoopController(LoopInfo *loop_info,
                                        PatternRewriter &rewriter) {
  if (!loop_info || !loop_info->isComplete()) {
    assert(false && "LoopInfo is incomplete or null.");
    return failure();
  }

  Location loc = loop_info->index_reserve_val.getLoc();

  Operation *start_def_op = findDefiningOp(loop_info->start_val);
  Operation *end_def_op = findDefiningOp(loop_info->end_val);
  Operation *step_def_op = findDefiningOp(loop_info->step_val);

  // Gets the insertion point for the new loop_controller operation.
  Operation *insertion_point = nullptr;

  // Compares the defining operations to find the latest one.
  auto updateLatestOp = [&](Operation *op1, Operation *op2) -> Operation * {
    if (!op1)
      return op2;
    if (!op2)
      return op1;
    // Returns the later operation in the block.
    return op2->isBeforeInBlock(op1) ? op1 : op2;
  };

  // Updates the insertion point based on the defining operations.
  if (start_def_op) {
    insertion_point = updateLatestOp(insertion_point, start_def_op);
  }
  if (end_def_op) {
    insertion_point = updateLatestOp(insertion_point, end_def_op);
  }
  if (step_def_op) {
    insertion_point = updateLatestOp(insertion_point, step_def_op);
  }

  // Sets the insertion point after the latest defining operation.
  if (insertion_point) {
    rewriter.setInsertionPointAfter(insertion_point);
  } else {
    assert(false && "No valid insertion point found for loop_controller");
    return failure();
  }

  // Creates the parentValid signal for loop_controller.
  auto true_val = createConstantPredicate(rewriter, loc, true);

  // Prepares the values and iter type for loop_controller.
  auto index_type = loop_info->index_phi_val.getType();
  rewriter.setInsertionPointAfter(true_val.getDefiningOp());

  // For start value, we use the grant_once for correctness.
  Value start_val = loop_info->start_val;
  if (!isa<neura::GrantOnceOp>(start_val.getDefiningOp())) {
    rewriter.setInsertionPointAfter(start_val.getDefiningOp());
    start_val = rewriter.create<neura::GrantOnceOp>(loc, index_type, start_val,
                                                    nullptr);
  }

  // For end value and step value, we create grant_always for correctness.
  Value end_val = loop_info->end_val;
  rewriter.setInsertionPointAfter(end_val.getDefiningOp());
  end_val =
      rewriter.create<neura::GrantAlwaysOp>(loc, index_type, end_val, nullptr);

  Value step_val = loop_info->step_val;
  rewriter.setInsertionPointAfter(end_val.getDefiningOp());
  step_val =
      rewriter.create<neura::GrantAlwaysOp>(loc, index_type, step_val, nullptr);

  rewriter.setInsertionPointAfter(true_val.getDefiningOp());

  StringAttr iter_type;
  if (neura::ICmpOp icmp_op =
          dyn_cast<neura::ICmpOp>(loop_info->condition_val.getDefiningOp())) {
    if (icmp_op.getCmpType() == "slt") {
      iter_type = rewriter.getStringAttr("increment");
    } else {
      assert(false && "Unsupported compare type");
      return failure(); // Unsupported compare type.
    }
  }

  // Creates the loop_controller operation.
  auto loop_controller = rewriter.create<neura::LoopControlOp>(
      loc, index_type, true_val.getType(), true_val, iter_type, start_val,
      end_val, step_val);

  Value new_index = loop_controller.getNextindex();
  Value new_valid = loop_controller.getValid();

  // Creates the replacement map for the loop info.
  DenseMap<Value, Value> replacement_map;

  // Creates the map for loop_info values (index_phi_val, condition_val, etc.)
  replacement_map[loop_info->index_phi_val] = new_index;
  if (loop_info->index_grant_op) {
    replacement_map[loop_info->index_grant_op->getResult(0)] = new_index;
  }
  replacement_map[loop_info->condition_val] = new_valid;
  neura::NotOp new_not;
  if (loop_info->not_condition_val) {
    rewriter.setInsertionPointAfter(loop_controller);
    new_not = rewriter.create<neura::NotOp>(
        loc, loop_info->not_condition_val.getType(), new_valid);
    replacement_map[loop_info->not_condition_val] = new_not.getResult();
  }

  // Replaces the index_reserve_val with the new index.
  if (!loop_info->users_to_update[loop_info->index_phi_val].empty()) {
    for (auto &user_info :
         loop_info->users_to_update[loop_info->index_phi_val]) {
      Operation *user = user_info.first;
      unsigned idx = user_info.second;
      user->setOperand(idx, new_index);
    }
  }

  // Replaces the granted index value with the new index.
  if (loop_info->index_grant_op &&
      !loop_info->users_to_update[loop_info->index_grant_op->getResult(0)]
           .empty()) {
    for (auto &user_info :
         loop_info->users_to_update[loop_info->index_grant_op->getResult(0)]) {
      Operation *user = user_info.first;
      unsigned idx = user_info.second;
      user->setOperand(idx, new_index);
    }
  }

  // Replaces the condition_val with the new_valid value.
  if (!loop_info->users_to_update[loop_info->condition_val].empty()) {
    for (auto &user_info :
         loop_info->users_to_update[loop_info->condition_val]) {
      Operation *user = user_info.first;
      unsigned idx = user_info.second;
      user->setOperand(idx, new_valid);
    }
  }

  // Handles not_condition_val if it exists
  if (loop_info->not_condition_val &&
      !loop_info->users_to_update[loop_info->not_condition_val].empty()) {
    // Replaces all uses of not_condition_val with the result of new_not
    for (auto &user_info :
         loop_info->users_to_update[loop_info->not_condition_val]) {
      Operation *user = user_info.first;
      unsigned idx = user_info.second;
      user->setOperand(idx, new_not.getResult());
    }
  }

  // Replaces the internal uses of the old loop info values.
  for (Operation *op : loop_info->ops_to_remove) {
    for (OpOperand &operand : op->getOpOperands()) {
      Value old_val = operand.get();
      if (replacement_map.count(old_val)) {
        operand.set(replacement_map[old_val]);
      }
    }
  }

  // Tracks erased operations.
  llvm::SmallPtrSet<Operation *, 16> erased_ops;
  bool made_progress = true;
  while (made_progress) {
    made_progress = false;
    for (Operation *op : loop_info->ops_to_remove) {
      if (!erased_ops.contains(op) && op->use_empty()) {
        rewriter.eraseOp(op);
        erased_ops.insert(op);
        made_progress = true;
      }
    }
  }

  // Checks if all operations were removed.
  for (Operation *op : loop_info->ops_to_remove) {
    if (!erased_ops.contains(op)) {
      llvm::errs() << "Warning: Could not remove operation: " << *op << "\n";
      llvm::errs() << "  Users: ";
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          llvm::errs() << *user << " ";
        }
      }
      llvm::errs() << "\n";
    }
  }

  return success();
}

struct FuseLoopControlFlowPattern : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp func_op,
                                PatternRewriter &rewriter) const override {
    auto accel_attr = func_op->getAttrOfType<StringAttr>("accelerator");
    if (!accel_attr || accel_attr.getValue() != "neura") {
      return failure();
    }
    // Saves all the identified loops.
    std::vector<std::unique_ptr<LoopInfo>> identified_loops;

    // Step 1: Identify loops in the function.
    func_op.walk([&](neura::ReserveOp reserveOp) {
      if (auto loop = identifyLoop(reserveOp)) {
        identified_loops.push_back(std::move(loop));
      }
    });

    if (identified_loops.empty()) {
      return failure();
    }

    for (auto &loop_info : identified_loops) {
      if (failed(replaceWithLoopController(loop_info.get(), rewriter))) {
        return failure();
      }
    }

    return success();
  }
};

struct FuseControlFlowPass
    : public PassWrapper<FuseControlFlowPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseControlFlowPass)

  StringRef getArgument() const override { return "fuse-control-flow"; }
  StringRef getDescription() const override {
    return "Fuses control flow operations into optimized neura dialect "
           "operations";
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    RewritePatternSet patterns(&getContext());

    patterns.add<FuseLoopControlFlowPattern>(&getContext());

    if (failed(applyPatternsGreedily(module_op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createFuseControlFlowPass() {
  return std::make_unique<FuseControlFlowPass>();
}
} // namespace mlir::neura