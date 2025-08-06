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
#include <cstdlib>
#include <fstream>
#include <yaml-cpp/yaml.h>

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

  Option<std::string> archSpecPath{
      *this, "arch-spec",
      llvm::cl::desc("Path to the architecture specification YAML file. "
                     "If not specified, will use default 4x4 architecture."),
      llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp module = getOperation();

    StringRef mappingStrategy_stringRef(mappingStrategy.getValue());
    // Creates a mapping strategy based on the provided option.
    std::unique_ptr<MappingStrategy> mapping_strategy;
    if (mappingStrategy_stringRef == "simple") {
      mapping_strategy = std::make_unique<HeuristicMapping>(1, 1);
    } else if (mappingStrategy_stringRef == "greedy") {
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
        IntegerAttr rec_mii_attr =
            IntegerAttr::get(IntegerType::get(func.getContext(), 32), rec_mii);
        func->setAttr("RecMII", rec_mii_attr);
      }

      // AcceleratorConfig config{/*numTiles=*/8}; // Example
      // Read architecture specification from command line option
      YAML::Node config;
      bool use_default_arch = false;
      
      if (!archSpecPath.getValue().empty()) {
        try {
          std::ifstream file(archSpecPath.getValue());
          if (file.is_open()) {
            config = YAML::Load(file);
            if (config["architecture"]) {
              llvm::outs() << "\033[31m[MapToAcceleratorPass] Loaded architecture from " 
                          << archSpecPath.getValue() << "\033[0m\n";
            } else {
              llvm::errs() << "[MapToAcceleratorPass] Invalid YAML format in " 
                          << archSpecPath.getValue() << ", using default 4x4\n";
              use_default_arch = true;
            }
          } else {
            llvm::errs() << "[MapToAcceleratorPass] Could not open architecture file " 
                        << archSpecPath.getValue() << ", using default 4x4\n";
            use_default_arch = true;
          }
        } catch (const std::exception& e) {
          llvm::errs() << "[MapToAcceleratorPass] Error parsing YAML file " 
                      << archSpecPath.getValue() << ": " << e.what() << ", using default 4x4\n";
          use_default_arch = true;
        }
      } else {
        use_default_arch = true;
        llvm::errs() << "[MapToAcceleratorPass] No architecture specification provided, using default 4x4\n";
      }

      Architecture architecture = use_default_arch ? Architecture(4, 4) : Architecture(config);
      
      int res_mii = calculateResMii(func, architecture);
      IntegerAttr res_mii_attr =
          IntegerAttr::get(IntegerType::get(func.getContext(), 32), res_mii);
      func->setAttr("ResMII", res_mii_attr);

      const int possibleMinII = std::max(rec_mii, res_mii);
      constexpr int maxII = 10;
      std::vector<Operation *> topologically_sorted_ops =
          getTopologicallySortedOps(func);
      if (topologically_sorted_ops.empty()) {
        llvm::errs() << "[MapToAcceleratorPass] No operations to map in function "
                     << func.getName() << "\n";
        assert(false && "Mapping aborted due to empty op list.");
      }
      for (Operation *op : topologically_sorted_ops) {
        llvm::outs() << "[MapToAcceleratorPass] Topologically sorted op: "
                     << *op << "\n";
      }
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
      std::vector<std::pair<Operation *, int>> sorted_ops_with_alap_levels =
          flatten_level_buckets(level_buckets);
      for (const auto &[op, level] : sorted_ops_with_alap_levels) {
        llvm::outs() << "[MapToAcceleratorPass] ALAP sorted op: " << *op << " (ALAP level: " << level << ")\n";
      }
      // assert(false);
      for (int ii = possibleMinII; ii <= maxII; ++ii) {
        llvm::errs() << "[MapToAcceleratorPass] Start mapping with target II of "
                     << ii << "\n";
        // Creates a mapping state for the current II.
        MappingState mapping_state(architecture, ii);
        if (mapping_strategy->map(sorted_ops_with_alap_levels, critical_ops, architecture, mapping_state)) {
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
