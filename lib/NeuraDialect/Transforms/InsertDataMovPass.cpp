#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_DEF_InsertDataMov
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
struct InsertDataMovForNeuraOps : public RewritePattern {
  InsertDataMovForNeuraOps(MLIRContext *context)
      : RewritePattern(/*matchAnyOpTypeTag=*/MatchAnyOpTypeTag(), /*benefit=*/1,
                       context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getDialect()->getNamespace() != "neura" ||
        isa<neura::DataMovOp>(op)) {
      return failure();
    }

    bool all_inputs_are_mov_except_reserve =
        llvm::all_of(op->getOperands(), [](Value v) {
          Operation *def_op = v.getDefiningOp();
          return isa_and_nonnull<neura::DataMovOp>(def_op) ||
                 isa_and_nonnull<neura::ReserveOp>(def_op);
        });

    if (all_inputs_are_mov_except_reserve) {
      return failure(); // All operands are already handled
    }

    // // Skips ops that already being inserted mov on the operands.
    // bool all_inputs_are_mov = llvm::all_of(op->getOperands(), [](Value v) {
    //   return isa_and_nonnull<neura::DataMovOp>(v.getDefiningOp());
    // });
    // if (all_inputs_are_mov) {
    //   return failure();
    // }

    // // Special case: skips rewriting phi if any operand is from reserve.
    // if (isa<neura::PhiOp>(op)) {
    //   bool has_reserved_input = llvm::any_of(op->getOperands(), [](Value v)
    //   {
    //     return isa_and_nonnull<neura::ReserveOp>(v.getDefiningOp());
    //   });

    //   if (has_reserved_input)
    //     return failure();  // Skip entire phi if any operand is reserved.
    // }

    // Makes sure none of the operand has being processed.
    bool has_any_mov_input = llvm::any_of(op->getOperands(), [](Value v) {
      return isa_and_nonnull<neura::DataMovOp>(v.getDefiningOp());
    });
    if (has_any_mov_input) {
      llvm::errs() << "Warning: Operand already wrapped in neura.data_mov: "
                   << *op << "\n";
    }
    assert(!has_any_mov_input &&
           "Unexpected: operand already wrapped in neura.mov");

    Location loc = op->getLoc();

    // Skips adding mov if the consumer is ctrl_mov.
    if (isa<neura::CtrlMovOp>(op)) {
      return failure(); // do not rewrite
    }

    // Wraps operands in mov.
    SmallVector<Value> new_operands;
    for (Value operand : op->getOperands()) {
      Operation *producer = operand.getDefiningOp();
      // Skips adding mov for neura.reserve -> neura.phi.
      if (isa<neura::PhiOp>(op) && producer &&
          isa<neura::ReserveOp>(producer)) {
        new_operands.push_back(operand);
        continue;
      }

      auto mov =
          rewriter.create<neura::DataMovOp>(loc, operand.getType(), operand);
      new_operands.push_back(mov);
    }

    // Clones op with new operands.
    OperationState state(loc, op->getName());
    state.addOperands(new_operands);
    state.addTypes(op->getResultTypes());
    state.addAttributes(op->getAttrs());

    // Copies successors for terminator operations.
    if (op->hasTrait<OpTrait::IsTerminator>()) {
      for (Block *successor : op->getSuccessors()) {
        state.addSuccessors(successor);
      }
    }

    Operation *new_op = rewriter.create(state);
    rewriter.replaceOp(op, new_op->getResults());
    return success();
  }
};

struct InsertDataMovPass
    : public PassWrapper<InsertDataMovPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertDataMovPass)

  StringRef getArgument() const override { return "insert-data-mov"; }
  StringRef getDescription() const override {
    return "Insert neura.data_mov before all neura dialect operations.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InsertDataMovForNeuraOps>(&getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));

    ModuleOp module_op = getOperation();

    // Applies to every region inside the module (regardless of func type,
    // e.g., mlir func or llvm func).
    module_op.walk([&](Operation *op) {
      if (!op->getRegions().empty()) {
        for (Region &region : op->getRegions()) {
          if (failed(applyPatternsGreedily(region, frozen))) {
            signalPassFailure();
          }
        }
      }
    });
  }
};
} // namespace

namespace mlir {
namespace neura {

std::unique_ptr<Pass> createInsertDataMovPass() {
  return std::make_unique<InsertDataMovPass>();
}

} // namespace neura
} // namespace mlir
