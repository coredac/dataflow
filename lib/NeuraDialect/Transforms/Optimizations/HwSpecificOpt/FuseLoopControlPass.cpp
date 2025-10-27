#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <memory>

using namespace mlir;

#define GEN_PASS_DEF_FUSELOOPCONTROL
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

  // Optional attributes for the parameters, if any.
  Attribute start_attr = nullptr;
  Attribute end_attr = nullptr;
  Attribute step_attr = nullptr;

  // Backward edge information.
  // The ctrl_mov operation for the index.
  Operation *index_ctrl_mov = nullptr;
  // The grant_predicate operation for the index.
  Operation *index_grant_op = nullptr;
  // The icmp operation for the loop condition.
  Operation *icmp_op = nullptr;
  // The add operation for the index.
  Operation *add_op = nullptr;

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
    return index_reserve_val && index_phi_val && index_grant_op &&
           condition_val && start_attr && end_attr && step_attr &&
           index_ctrl_mov && icmp_op && add_op;
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

// Finds the constant attribute for a value.
Attribute findConstantAttribute(Operation *op) {
  // Checks if the operation has a constant attribute.
  if (op && op->hasAttr("rhs_value")) {
    return op->getAttr("rhs_value");
  }

  // If the value is already a constant, return it.
  if (auto const_op = dyn_cast<neura::ConstantOp>(op)) {
    return const_op.getValueAttr();
  }

  // Handles grant operations and adds them to the removal list.
  if (auto grant_once_op = dyn_cast<neura::GrantOnceOp>(op)) {
    if (grant_once_op->hasAttr("constant_value")) {
      return grant_once_op->getAttr("constant_value");
    }
  }

  return nullptr;
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

  loop->start_attr = findConstantAttribute(initial_value.getDefiningOp());

  if (!loop->start_attr) {
    // Unable to determine start value or attribute.
    llvm::errs()
        << "[CtrlFlowFuse] Unable to determine start value or attribute.\n";
    return nullptr;
  }
  loop->addOpToRemove(initial_value.getDefiningOp());

  // Identifies the phi->icmp->[not]->grant_predicate pattern.
  for (Operation *phi_user : index_phi_op->getUsers()) {
    if (neura::ICmpOp icmp_op = dyn_cast<neura::ICmpOp>(phi_user)) {
      if (icmp_op.getCmpType() == "slt" &&
          icmp_op.getLhs() == loop->index_phi_val) {
        loop->condition_val = icmp_op.getResult();
        loop->icmp_op = icmp_op;

        loop->end_attr = findConstantAttribute(icmp_op);
        if (!loop->end_attr) {
          // Unable to determine end value or attribute.
          llvm::errs() << "[CtrlFlowFuse] Unable to determine end attribute.\n";
          return nullptr;
        }
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

  if (!loop->condition_val || !loop->icmp_op) {
    llvm::errs() << "[CtrlFlowFuse] Incomplete loop information, condition or "
                    "ICMP operation missing.\n";
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
          llvm::errs() << "[CtrlFlowFuse] Found add operation: " << *add_op
                       << "\n";
          loop->add_op = add_op;
          loop->step_attr = findConstantAttribute(add_op);
          if (!loop->step_attr) {
            // Unable to determine step attribute.
            llvm::errs()
                << "[CtrlFlowFuse] Unable to determine step attribute.\n";
            return nullptr;
          }
          loop->addOpToRemove(add_op);
          break;
        }
      }
    }
  }

  if (!loop->index_ctrl_mov || !loop->add_op || !loop->step_attr) {
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
                                            rewriter.getBoolAttr(value));
}

LogicalResult replaceWithLoopController(LoopInfo *loop_info,
                                        PatternRewriter &rewriter) {
  if (!loop_info || !loop_info->isComplete()) {
    assert(false && "LoopInfo is incomplete or null.");
    return failure();
  }

  Location loc = loop_info->index_reserve_val.getLoc();

  rewriter.setInsertionPointAfter(loop_info->index_phi_val.getDefiningOp());

  // Creates the parentValid signal for loop_controller.
  auto true_const = createConstantPredicate(rewriter, loc, true);
  rewriter.setInsertionPointAfter(true_const.getDefiningOp());
  auto true_val = rewriter
                      .create<neura::GrantAlwaysOp>(loc, true_const.getType(),
                                                    true_const, nullptr)
                      ->getResult(0);

  // Prepares the values and iter type for loop_controller.
  auto index_type = loop_info->index_phi_val.getType();

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

  rewriter.setInsertionPointAfter(true_val.getDefiningOp());

  auto loop_controller = rewriter.create<neura::LoopControlOp>(
      loc, index_type, true_val.getType(), true_val, iter_type,
      loop_info->start_attr, loop_info->end_attr, loop_info->step_attr);

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
      llvm::errs() << "[CtrlFlowFuse] No loops identified for fusion in "
                   << func_op.getName() << "\n";
      return failure();
    }

    llvm::errs() << "[CtrlFlowFuse] Identified " << identified_loops.size()
                 << " loops for fusion in function " << func_op.getName()
                 << "\n";

    for (auto &loop_info : identified_loops) {
      if (failed(replaceWithLoopController(loop_info.get(), rewriter))) {
        return failure();
      }
    }

    return success();
  }
};

struct FuseLoopControlPass
    : public PassWrapper<FuseLoopControlPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseLoopControlPass)

  StringRef getArgument() const override { return "fuse-loop-control"; }
  StringRef getDescription() const override {
    return "Fuses loop control operations into optimized neura dialect "
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
std::unique_ptr<Pass> createFuseLoopControlPass() {
  return std::make_unique<FuseLoopControlPass>();
}
} // namespace mlir::neura