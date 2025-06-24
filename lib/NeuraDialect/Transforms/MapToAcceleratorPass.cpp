#include <deque>

#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/Mapping/MappingState.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::neura;

#define GEN_PASS_DEF_MapToAccelerator
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

struct MapToAcceleratorPass
    : public PassWrapper<MapToAcceleratorPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapToAcceleratorPass)

  StringRef getArgument() const override { return "map-to-accelerator"; }
  StringRef getDescription() const override {
    return "Maps IR to the target accelerator.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp func) {
      // Skips functions not targeting the neura accelerator.
      auto accel_attr = func->getAttrOfType<StringAttr>("accelerator");
      if (!accel_attr || accel_attr.getValue() != "neura")
        return;

      // Collects and reports recurrence cycles found in the function.
      auto recurrence_cycles = collectRecurrenceCycles(func);
      RecurrenceCycle *longest = nullptr;
      int rec_mii = 1;
      for (auto &cycle : recurrence_cycles) {
        if (!longest || cycle.length > longest->length)
          longest = &cycle;
      }

      if (longest) {
        llvm::errs() << "[MapToAcceleratorPass] Longest recurrence cycle (length "
                    << longest->length << "):\n";
        for (Operation *op : longest->operations) {
          op->print(llvm::errs()), llvm::errs() << "\n";
        }
        rec_mii = longest->length;
        IntegerAttr rec_mii_attr = IntegerAttr::get(
            IntegerType::get(func.getContext(), 32), rec_mii);
        func->setAttr("RecMII", rec_mii_attr);
      }

      // AcceleratorConfig config{/*numTiles=*/8}; // Example
      Architecture architecture(2, 2);
      int res_mii = calculateResMii(func, architecture);
      IntegerAttr res_mii_attr = IntegerAttr::get(
          IntegerType::get(func.getContext(), 32), res_mii);
      func->setAttr("ResMII", res_mii_attr);

      const int minII = std::min(rec_mii, res_mii);
      constexpr int maxII = 5;
      for (int ii = minII; ii <= maxII; ++ii) {
        MappingState state(architecture, ii);
      }
    });
  }
};

} // namespace

namespace mlir::neura {

std::unique_ptr<Pass> createMapToAcceleratorPass() {
  return std::make_unique<MapToAcceleratorPass>();
}

} // namespace mlir::neura
