#include <deque>

#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/HeuristicMapping/HeuristicMapping.h"
#include "NeuraDialect/Mapping/MappingState.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

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

  MapToAcceleratorPass() = default;
  MapToAcceleratorPass(const MapToAcceleratorPass &pass)
      : PassWrapper<MapToAcceleratorPass, OperationPass<ModuleOp>>(pass) {}
  Option<std::string> mappingStrategy{
      *this, "mapping-strategy",
      llvm::cl::desc("Mapping strategy to use for mapping operations to the "
                     "accelerator. Options: greedy, exhaustive, "
                     "heuristic=max_loc,max_depth (default "
                     "max_loc=5, max_depth=3)"),
      llvm::cl::init("heuristic")};

  void runOnOperation() override {
    ModuleOp module = getOperation();

    StringRef mappingStrategy_stringRef(mappingStrategy.getValue());
    // Creates a mapping strategy based on the provided option.
    std::unique_ptr<MappingStrategy> mapping_strategy;
    if (mappingStrategy_stringRef == "greedy") {
      mapping_strategy = std::make_unique<HeuristicMapping>(INT_MAX, 1);
    } else if (mappingStrategy_stringRef == "exhaustive") {
      mapping_strategy = std::make_unique<HeuristicMapping>(INT_MAX, INT_MAX);
    } else if (mappingStrategy_stringRef == "heuristic") {
      mapping_strategy = std::make_unique<HeuristicMapping>(
          5, 3); // Randomly picked default values for max_loc and max_depth
    } else if (mappingStrategy_stringRef.starts_with("heuristic=")) {
      // Used for custom backtrack parameters.
      // Example: "heuristic=5,3" means max_loc=5, max_depth=3
      // Extracts the parameters after "heuristic=".
      StringRef paramsRef =
          mappingStrategy_stringRef.substr(strlen("heuristic="));
      size_t comma_pos = paramsRef.find(',');

      if (comma_pos != StringRef::npos) {
        StringRef max_loc_str = paramsRef.substr(0, comma_pos);
        StringRef max_depth_str = paramsRef.substr(comma_pos + 1);

        int max_loc, max_depth;
        if (!max_loc_str.getAsInteger(10, max_loc) &&
            !max_depth_str.getAsInteger(10, max_depth)) {
          mapping_strategy =
              std::make_unique<HeuristicMapping>(max_loc, max_depth);
          llvm::errs()
              << "[MapToAcceleratorPass] Use custom backtrack parameters: "
              << "max_location_to_try=" << max_loc
              << ", max_backtrack_depth=" << max_depth << "\n";
        } else {
          llvm::errs()
              << "[MapToAcceleratorPass] Illegal backtrack parameters format: "
              << mappingStrategy << "\n";
          return;
        }
      } else {
        llvm::errs()
            << "[MapToAcceleratorPass] Illegal backtrack parameters format: "
            << mappingStrategy << "\n";
        return;
      }
    } else {
      llvm::errs() << "[MapToAcceleratorPass] Unsupported mapping strategy: "
                   << mappingStrategy << "\n";
      return;
    }

    module.walk([&](func::FuncOp func) {
      // Skips functions not targeting the neura accelerator.
      auto accel_attr = func->getAttrOfType<StringAttr>("accelerator");
      if (!accel_attr || accel_attr.getValue() != "neura") {
        return;
      }

      // Collects and reports recurrence cycles found in the function.
      auto recurrence_cycles = collectRecurrenceCycles(func);
      RecurrenceCycle *longest = nullptr;
      int rec_mii = 1;
      for (auto &cycle : recurrence_cycles) {
        llvm::outs() << "[DEBUG] Recurrence cycle (length " << cycle.length
                     << "):\n";
        for (Operation *op : cycle.operations) {
          llvm::outs() << "  " << *op << "\n";
        }
        if (!longest || cycle.length > longest->length) {
          longest = &cycle;
        }
      }

      if (longest) {
        llvm::outs()
            << "[MapToAcceleratorPass] Longest recurrence cycle (length "
            << longest->length << "):\n";
        for (Operation *op : longest->operations) {
          op->print(llvm::outs()), llvm::outs() << "\n";
        }
        rec_mii = longest->length;
        IntegerAttr rec_mii_attr =
            IntegerAttr::get(IntegerType::get(func.getContext(), 32), rec_mii);
        func->setAttr("RecMII", rec_mii_attr);
      }

      // AcceleratorConfig config{/*numTiles=*/8}; // Example
      Architecture architecture(4, 4);
      int res_mii = calculateResMii(func, architecture);
      IntegerAttr res_mii_attr =
          IntegerAttr::get(IntegerType::get(func.getContext(), 32), res_mii);
      func->setAttr("ResMII", res_mii_attr);

      const int minII = std::min(rec_mii, res_mii);
      constexpr int maxII = 10;
      std::vector<Operation *> sorted_ops = getTopologicallySortedOps(func);
      for (Operation *op : sorted_ops) {
        llvm::outs() << "[MapToAcceleratorPass] sorted op: " << *op << "\n";
      }
      for (int ii = minII; ii <= maxII; ++ii) {
        MappingState mapping_state(architecture, ii);
        if (mapping_strategy->map(sorted_ops, architecture, mapping_state)) {
          // success
          llvm::errs() << "[MapToAcceleratorPass] Successfully mapped function "
                       << func.getName() << "' with II = " << ii << "\n";
          mapping_state.dumpOpToLocs(); // logs to stderr
          mapping_state.encodeMappingState();
          func->setAttr(
              "CompiledII",
              IntegerAttr::get(IntegerType::get(func.getContext(), 32), ii));
          break;
        }
        llvm::errs() << "[DEBUG] mapping failed for II = " << ii << "\n";
        mapping_state.dumpOpToLocs(); // logs to stderr
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
