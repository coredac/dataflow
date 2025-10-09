#include <deque>
#include <memory>

#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/HeuristicMapping/HeuristicMapping.h"
#include "NeuraDialect/Mapping/MappingState.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::neura;

#define GEN_PASS_DEF_MAPTOACCELERATOR
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
  Option<std::string> sortStrategy{
      *this, "sort-strategy",
      llvm::cl::desc("Strategy for sorting operations before mapping. "
                     "Options: topological, mixed (default)."),
      llvm::cl::init("mixed")};
  Option<std::string> mappingStrategy{
      *this, "mapping-strategy",
      llvm::cl::desc("Mapping strategy to use for mapping operations to the "
                     "accelerator. Options: heuristic (default)."),
      llvm::cl::init("heuristic")};
  Option<std::string> mappingMode{
      *this, "mapping-mode",
      llvm::cl::desc(
          "Mapping mode to use for mapping operations to the "
          "accelerator. Options: spatial-only, spatial-temporal (default)."),
      llvm::cl::init("spatial-temporal")};
  Option<std::string> backtrackConfig{
      *this, "backtrack-config",
      llvm::cl::desc(
          "Backtrack configuration used for mapping operations to the "
          "accelerator. Options: simple, greedy, exhaustive, "
          "customized=max_loc,max_depth (default "
          "max_loc=5, max_depth=3)"),
      llvm::cl::init("customized")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::unique_ptr<Mapping> mapping_strategy;
    StringRef mapping_strategy_string_ref(mappingStrategy.getValue());
    StringRef backtrack_config_string_ref(backtrackConfig.getValue());
    StringRef mapping_mode_string_ref(mappingMode.getValue());
    StringRef sort_strategy_string_ref(sortStrategy.getValue());
    bool is_spatial_only = (mapping_mode_string_ref == "spatial-only");
    if (is_spatial_only || mapping_mode_string_ref == "spatial-temporal" ||
        mapping_mode_string_ref.empty()) {
      if (mapping_mode_string_ref.empty()) {
        mapping_mode_string_ref = "spatial-temporal";
      }
      llvm::errs() << "[MapToAcceleratorPass] Using Mapping Mode: "
                   << mapping_mode_string_ref << "\n";
    } else {
      llvm::errs() << "[MapToAcceleratorPass] Unsupported mapping mode: "
                   << mapping_mode_string_ref << "\n";
      return;
    }

    if (mapping_strategy_string_ref == "heuristic" ||
        mapping_strategy_string_ref.empty()) {
      mapping_strategy_string_ref = "heuristic";

      if (backtrack_config_string_ref == "simple") {
        mapping_strategy = std::make_unique<HeuristicMapping>(1, 1);
      } else if (backtrack_config_string_ref == "greedy") {
        mapping_strategy = std::make_unique<HeuristicMapping>(INT_MAX, 1);
      } else if (backtrack_config_string_ref == "exhaustive") {
        mapping_strategy = std::make_unique<HeuristicMapping>(INT_MAX, INT_MAX);
      } else if (backtrack_config_string_ref == "customized") {
        mapping_strategy = std::make_unique<HeuristicMapping>(5, 3);
      } else if (backtrack_config_string_ref.starts_with("customized=")) {
        // Used for custom backtrack parameters.
        // Example: "customized=5,3" means max_loc=5, max_depth=3
        // Extracts the parameters after "customized=".
        StringRef params_ref =
            backtrack_config_string_ref.substr(strlen("customized="));
        size_t comma_pos = params_ref.find(',');

        if (comma_pos != StringRef::npos) {
          StringRef max_loc_str = params_ref.substr(0, comma_pos);
          StringRef max_depth_str = params_ref.substr(comma_pos + 1);

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
            llvm::errs() << "[MapToAcceleratorPass] Illegal customized "
                            "parameters format: "
                         << backtrack_config_string_ref << "\n";
            return;
          }
        } else {
          llvm::errs()
              << "[MapToAcceleratorPass] Illegal customized parameters format: "
              << backtrack_config_string_ref << "\n";
          return;
        }
      }
    } else {
      llvm::errs() << "[MapToAcceleratorPass] Unsupported mapping strategy: "
                   << mapping_strategy_string_ref << "\n";
      return;
    }

    module.walk([&](func::FuncOp func) {
      // Skips functions not targeting the neura accelerator.
      auto accel_attr = func->getAttrOfType<StringAttr>("accelerator");
      if (!accel_attr || accel_attr.getValue() != "neura") {
        return;
      }

      // Checks the dataflow IR mode.
      auto dataflow_mode_attr =
          func->getAttrOfType<StringAttr>("dataflow_mode");
      bool is_steering_mode =
          (dataflow_mode_attr && dataflow_mode_attr.getValue() == "steering");

      // If steering mode, enforce spatial-only mapping.
      if (is_steering_mode) {
        if (mapping_mode_string_ref != "spatial-only") {
          func.emitError() << "Steering IR mode requires spatial-only mapping, "
                           << "but got mapping mode: "
                           << mapping_mode_string_ref;
          signalPassFailure();
          return;
        }
        llvm::errs() << "[MapToAcceleratorPass] Using spatial-only mapping for "
                        "steering mode function: "
                     << func.getName() << "\n";
      }

      // Collects and reports recurrence cycles found in the function.
      auto recurrence_cycles = collectRecurrenceCycles(func);
      std::set<Operation *> critical_ops;
      RecurrenceCycle *longest = nullptr;
      int rec_mii = 1;
      for (auto &cycle : recurrence_cycles) {
        llvm::outs() << "[DEBUG] Recurrence cycle (length " << cycle.length
                     << "):\n";
        for (Operation *op : cycle.operations) {
          critical_ops.insert(op);
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
      } else if (!longest) {
        rec_mii = 1; // No recurrence cycles found, set MII to 1.
      }

      // AcceleratorConfig config{/*numTiles=*/8}; // Example
      Architecture architecture(4, 4);
      int res_mii = calculateResMii(func, architecture);

      const int possibleMinII = std::max(rec_mii, res_mii);
      constexpr int maxII = 20;
      std::vector<Operation *> topologically_sorted_ops =
          getTopologicallySortedOps(func);
      if (topologically_sorted_ops.empty()) {
        llvm::errs()
            << "[MapToAcceleratorPass] No operations to map in function "
            << func.getName() << "\n";
        assert(false && "Mapping aborted due to empty op list.");
      }
      for (Operation *op : topologically_sorted_ops) {
        llvm::outs() << "[MapToAcceleratorPass] Topologically sorted op: "
                     << *op << "\n";
      }

      // Two sorting strategies: pure topological order, or mixed ALAP + topo.
      std::vector<std::pair<Operation *, int>> sorted_ops_with_levels;
      if (sort_strategy_string_ref == "topological") {
        for (Operation *op : topologically_sorted_ops) {
          sorted_ops_with_levels.push_back({op, 0}); // Level 0 for all ops
        }
      } else if (sort_strategy_string_ref == "mixed") {
        std::vector<std::vector<Operation *>> level_buckets =
            getOpsInAlapLevels(topologically_sorted_ops, critical_ops);
        for (int level = 0; level < static_cast<int>(level_buckets.size());
             ++level) {
          llvm::outs() << "[MapToAcceleratorPass] ALAP Bucket Level " << level
                       << ": " << level_buckets[level].size() << " ops\n";
          for (Operation *op : level_buckets[level]) {
            llvm::outs() << "  " << *op << "\n";
          }
        }
        sorted_ops_with_levels = flatten_level_buckets(level_buckets);
        for (const auto &[op, level] : sorted_ops_with_levels) {
          llvm::outs() << "[MapToAcceleratorPass] ALAP sorted op: " << *op
                       << " (ALAP level: " << level << ")\n";
        }
      } else {
        llvm::errs() << "[MapToAcceleratorPass] Unsupported sort strategy: "
                     << sort_strategy_string_ref << "\n";
        return;
      }
      for (int ii = possibleMinII; ii <= maxII; ++ii) {
        llvm::errs()
            << "[MapToAcceleratorPass] Start mapping with target II of " << ii
            << "\n";
        // Creates a mapping state for the current II.
        MappingState mapping_state(architecture, ii, is_spatial_only);
        if (mapping_strategy->map(sorted_ops_with_levels, critical_ops,
                                  architecture, mapping_state)) {
          // success
          llvm::errs() << "[MapToAcceleratorPass] Successfully mapped function "
                       << func.getName() << "' with II = " << ii << "\n";
          mapping_state.dumpOpToLocs(); // logs to stderr
          mapping_state.encodeMappingState();

          // Sets the mapping_info attribute on the function.
          auto ctx = func.getContext();
          DictionaryAttr mapping_info = DictionaryAttr::get(
              ctx,
              {NamedAttribute(StringAttr::get(ctx, "x_tiles"),
                              IntegerAttr::get(IntegerType::get(ctx, 32),
                                               architecture.getWidth())),
               NamedAttribute(StringAttr::get(ctx, "y_tiles"),
                              IntegerAttr::get(IntegerType::get(ctx, 32),
                                               architecture.getHeight())),
               NamedAttribute(
                   StringAttr::get(ctx, "mapping_strategy"),
                   StringAttr::get(ctx, mapping_strategy_string_ref)),
               NamedAttribute(StringAttr::get(ctx, "mapping_mode"),
                              StringAttr::get(ctx, mapping_mode_string_ref)),
               NamedAttribute(StringAttr::get(ctx, "compiled_ii"),
                              IntegerAttr::get(IntegerType::get(ctx, 32), ii)),
               NamedAttribute(
                   StringAttr::get(ctx, "rec_mii"),
                   IntegerAttr::get(IntegerType::get(ctx, 32), rec_mii)),
               NamedAttribute(
                   StringAttr::get(ctx, "res_mii"),
                   IntegerAttr::get(IntegerType::get(ctx, 32), res_mii))});

          func->setAttr("mapping_info", mapping_info);
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
