#include <deque>
#include <fstream>
#include <memory>

#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Architecture/ArchitectureSpec.h"
#include "NeuraDialect/Mapping/HeuristicMapping/HeuristicMapping.h"
#include "NeuraDialect/Mapping/MappingState.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraTypes.h"
#include "NeuraDialect/Util/NeuraYamlKeys.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::neura;
using namespace mlir::neura::yamlkeys;

#define GEN_PASS_DEF_MAPTOACCELERATOR
#include "NeuraDialect/NeuraPasses.h.inc"

// -----------------------------------------------------------------------------
// Utility: Extracts an integer from a YAML ScalarNode. Returns true on success.
static bool parseYamlScalarInt(const llvm::yaml::Node *node, int &result) {
  auto *scalar = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(node);
  if (!scalar)
    return false;
  llvm::SmallString<64> value_string;
  llvm::StringRef value_ref = scalar->getValue(value_string);
  long long temp_value = 0;
  if (value_ref.getAsInteger(10, temp_value))
    return false;
  result = static_cast<int>(temp_value);
  return true;
}

// Utility: Extracts a string from a YAML ScalarNode. Returns true on success.
static bool parseYamlScalarString(const llvm::yaml::Node *node,
                                  std::string &result) {
  auto *scalar = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(node);
  if (!scalar)
    return false;
  llvm::SmallString<64> value_string;
  llvm::StringRef value_ref = scalar->getValue(value_string);
  result = value_ref.str();
  return true;
}

// Utility: Extracts a vector of strings from a YAML SequenceNode.
static void parseYamlStringSequence(llvm::yaml::Node *node,
                                    std::vector<std::string> &result) {
  auto *seq = llvm::dyn_cast_or_null<llvm::yaml::SequenceNode>(node);
  if (!seq)
    return;
  result.clear();
  for (auto &item : *seq) {
    std::string value;
    if (parseYamlScalarString(&item, value))
      result.push_back(value);
  }
}

// Utility: Print YAML parse error and return false.
static bool yamlParseError(const std::string &msg,
                           const std::string &file = "") {
  llvm::errs() << "[MapToAcceleratorPass] YAML parse error";
  if (!file.empty())
    llvm::errs() << " in: " << file;
  llvm::errs() << ": " << msg << "\n";
  return false;
}

// -----------------------------------------------------------------------------
// Helper function to parse tile defaults.
void parseTileDefaults(llvm::yaml::MappingNode *tile_defaults_map,
                       mlir::neura::TileDefaults &tile_defaults) {
  for (auto &key_value_pair : *tile_defaults_map) {
    auto *key_node =
        llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(key_value_pair.getKey());
    if (!key_node)
      continue;
    llvm::SmallString<64> key_string;
    llvm::StringRef key_ref = key_node->getValue(key_string);

    if (key_ref == kNumRegisters) {
      int temp_value = 0;
      if (parseYamlScalarInt(key_value_pair.getValue(), temp_value))
        tile_defaults.num_registers = temp_value;
    } else if (key_ref == kFuTypes) {
      parseYamlStringSequence(key_value_pair.getValue(),
                              tile_defaults.function_units);
    } else {
      llvm::errs() << "[MapToAcceleratorPass] Unknown tile_defaults key: "
                   << key_ref << "\n";
    }
  }
}

// Helper function to parse tile override operations and registers.
void parseTileOverrideOperations(llvm::yaml::MappingNode *override_map,
                                 mlir::neura::TileOverride &override) {
  for (auto &key_value_pair : *override_map) {
    auto *key_node =
        llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(key_value_pair.getKey());
    if (!key_node)
      continue;
    llvm::SmallString<64> key_string;
    llvm::StringRef key_ref = key_node->getValue(key_string);

    if (key_ref == kFuTypes) {
      parseYamlStringSequence(key_value_pair.getValue(), override.fu_types);
    } else if (key_ref == kNumRegisters) {
      int temp_value = 0;
      if (parseYamlScalarInt(key_value_pair.getValue(), temp_value))
        override.num_registers = temp_value;
    } else {
      llvm::errs() << "[MapToAcceleratorPass] Unknown tile_override key: "
                   << key_ref << "\n";
    }
  }
}

// Helper function to parse a single tile override.
void parseSingleTileOverride(llvm::yaml::MappingNode *override_map,
                             mlir::neura::TileOverride &override) {
  for (auto &key_value_pair : *override_map) {
    auto *key_node =
        llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(key_value_pair.getKey());
    if (!key_node)
      continue;
    llvm::SmallString<64> key_string;
    llvm::StringRef key_ref = key_node->getValue(key_string);

    int temp_value = 0;
    if (key_ref == kCgraX) {
      if (parseYamlScalarInt(key_value_pair.getValue(), temp_value))
        override.cgra_x = temp_value;
    } else if (key_ref == kCgraY) {
      if (parseYamlScalarInt(key_value_pair.getValue(), temp_value))
        override.cgra_y = temp_value;
    } else if (key_ref == kTileX) {
      if (parseYamlScalarInt(key_value_pair.getValue(), temp_value))
        override.tile_x = temp_value;
    } else if (key_ref == kTileY) {
      if (parseYamlScalarInt(key_value_pair.getValue(), temp_value))
        override.tile_y = temp_value;
    } else if (key_ref == kFuTypes) {
      parseYamlStringSequence(key_value_pair.getValue(), override.fu_types);
    } else if (key_ref == kNumRegisters) {
      if (parseYamlScalarInt(key_value_pair.getValue(), temp_value))
        override.num_registers = temp_value;
    } else if (key_ref == kExistence) {
      std::string value;
      if (parseYamlScalarString(key_value_pair.getValue(), value)) {
        override.existence = (value == "true" || value == "True" || value == "1");
      }
    } else {
      llvm::errs() << "[MapToAcceleratorPass] Unknown tile_override key: "
                   << key_ref << "\n";
    }
  }
}

// Helper function to parse tile overrides.
bool parseTileOverrides(
    llvm::yaml::SequenceNode *tile_overrides_seq,
    std::vector<mlir::neura::TileOverride> &tile_overrides) {
  for (auto &override_node : *tile_overrides_seq) {
    auto *override_map =
        llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(&override_node);
    if (!override_map)
      continue;
    mlir::neura::TileOverride override;
    parseSingleTileOverride(override_map, override);
    tile_overrides.push_back(override);
  }
  return true;
}

// Helper function to parse link defaults.
bool parseLinkDefaults(llvm::yaml::MappingNode *link_defaults_map,
                       mlir::neura::LinkDefaults &link_defaults) {
  for (auto &key_value_pair : *link_defaults_map) {
    auto *key_node =
        llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(key_value_pair.getKey());
    if (!key_node)
      continue;
    llvm::SmallString<64> key_string;
    llvm::StringRef key_ref = key_node->getValue(key_string);

    int temp_value = 0;
    if (key_ref == kLatency) {
      if (parseYamlScalarInt(key_value_pair.getValue(), temp_value))
        link_defaults.latency = temp_value;
    } else if (key_ref == kBandwidth) {
      if (parseYamlScalarInt(key_value_pair.getValue(), temp_value))
        link_defaults.bandwidth = temp_value;
    } else {
      llvm::errs() << "[MapToAcceleratorPass] Unknown link_defaults key: "
                   << key_ref << "\n";
    }
  }
  return true;
}

// Helper function to parse a single link override.
void parseSingleLinkOverride(llvm::yaml::MappingNode *override_map,
                             mlir::neura::LinkOverride &override) {
  for (auto &key_value_pair : *override_map) {
    auto *key_node =
        llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(key_value_pair.getKey());
    if (!key_node)
      continue;
    llvm::SmallString<64> key_string;
    llvm::StringRef key_ref = key_node->getValue(key_string);

    int temp_value = 0;
    if (key_ref == kLatency) {
      if (parseYamlScalarInt(key_value_pair.getValue(), temp_value))
        override.latency = temp_value;
    } else if (key_ref == kBandwidth) {
      if (parseYamlScalarInt(key_value_pair.getValue(), temp_value))
        override.bandwidth = temp_value;
    } else if (key_ref == kSrcTileX) {
      if (parseYamlScalarInt(key_value_pair.getValue(), temp_value))
        override.src_tile_x = temp_value;
    } else if (key_ref == kSrcTileY) {
      if (parseYamlScalarInt(key_value_pair.getValue(), temp_value))
        override.src_tile_y = temp_value;
    } else if (key_ref == kDstTileX) {
      if (parseYamlScalarInt(key_value_pair.getValue(), temp_value))
        override.dst_tile_x = temp_value;
    } else if (key_ref == kDstTileY) {
      if (parseYamlScalarInt(key_value_pair.getValue(), temp_value))
        override.dst_tile_y = temp_value;
    } else if (key_ref == kExistence) {
      std::string value;
      if (parseYamlScalarString(key_value_pair.getValue(), value)) {
        override.existence =
            (value == "true" || value == "True" || value == "1");
      }
    } else {
      llvm::errs() << "[MapToAcceleratorPass] Unknown link_override key: "
                   << key_ref << "\n";
    }
  }
}

// Helper function to parse link overrides.
bool parseLinkOverrides(
    llvm::yaml::SequenceNode *link_overrides_seq,
    std::vector<mlir::neura::LinkOverride> &link_overrides) {
  for (auto &override_node : *link_overrides_seq) {
    auto *override_map =
        llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(&override_node);
    if (!override_map)
      continue;
    mlir::neura::LinkOverride override;
    parseSingleLinkOverride(override_map, override);
    link_overrides.push_back(override);
  }
  return true;
}

// Helper function to parse topology string to BaseTopology enum
mlir::neura::BaseTopology parseTopologyString(const std::string &topology_str) {
  if (topology_str == kMesh) {
    return mlir::neura::BaseTopology::MESH;
  } else if (topology_str == kKingMesh || topology_str == kKingMeshAlt) {
    return mlir::neura::BaseTopology::KING_MESH;
  } else if (topology_str == kRing) {
    return mlir::neura::BaseTopology::RING;
  } else {
    // Default to mesh if unknown topology
    return mlir::neura::BaseTopology::MESH;
  }
}

// Helper function to parse architecture YAML configuration.
bool parseArchitectureYaml(
    llvm::yaml::Document &doc, int &multi_cgra_rows, int &multi_cgra_columns,
    mlir::neura::BaseTopology &multi_cgra_base_topology, int &per_cgra_rows,
    int &per_cgra_columns, mlir::neura::BaseTopology &per_cgra_base_topology,
    int &max_ctrl_mem_items, mlir::neura::TileDefaults &tile_defaults,
    std::vector<mlir::neura::TileOverride> &tile_overrides,
    mlir::neura::LinkDefaults &link_defaults,
    std::vector<mlir::neura::LinkOverride> &link_overrides) {
  auto *root = doc.getRoot();
  if (!root)
    return yamlParseError("Empty YAML document");
  auto *root_map = llvm::dyn_cast<llvm::yaml::MappingNode>(root);
  if (!root_map)
    return yamlParseError("YAML root is not a mapping");

  for (auto &key_value_pair : *root_map) {
    auto *key_node =
        llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(key_value_pair.getKey());
    if (!key_node)
      continue;
    llvm::SmallString<64> key_string;
    llvm::StringRef key_ref = key_node->getValue(key_string);

    if (key_ref == kArchitecture) {
      // Not used in this parser, but could be handled here.
      continue;
    } else if (key_ref == kMultiCgraDefaults) {
      auto *multi_cgra_map = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(
          key_value_pair.getValue());
      if (!multi_cgra_map)
        continue;
      for (auto &multi_cgra_map_key_value_pair : *multi_cgra_map) {
        auto *multi_cgra_map_key_node =
            llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(
                multi_cgra_map_key_value_pair.getKey());
        if (!multi_cgra_map_key_node)
          continue;
        llvm::SmallString<64> multi_cgra_map_key_string;
        llvm::StringRef multi_cgra_map_key_ref =
            multi_cgra_map_key_node->getValue(multi_cgra_map_key_string);
        int temp_value = 0;
        if (multi_cgra_map_key_ref == kRows) {
          if (parseYamlScalarInt(multi_cgra_map_key_value_pair.getValue(),
                                 temp_value))
            multi_cgra_rows = temp_value;
        } else if (multi_cgra_map_key_ref == kColumns) {
          if (parseYamlScalarInt(multi_cgra_map_key_value_pair.getValue(),
                                 temp_value))
            multi_cgra_columns = temp_value;
        } else if (multi_cgra_map_key_ref == kBaseTopology) {
          std::string topo_str;
          if (parseYamlScalarString(multi_cgra_map_key_value_pair.getValue(),
                                    topo_str))
            multi_cgra_base_topology = parseTopologyString(topo_str);
        }
      }
    } else if (key_ref == kPerCgraDefaults) {
      auto *per_cgra_map = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(
          key_value_pair.getValue());
      if (!per_cgra_map)
        continue;
      for (auto &per_cgra_map_key_value_pair : *per_cgra_map) {
        auto *per_cgra_map_key_node =
            llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(
                per_cgra_map_key_value_pair.getKey());
        if (!per_cgra_map_key_node)
          continue;
        llvm::SmallString<64> per_cgra_map_key_string;
        llvm::StringRef per_cgra_map_key_ref =
            per_cgra_map_key_node->getValue(per_cgra_map_key_string);
        int temp_value = 0;
        if (per_cgra_map_key_ref == kRows) {
          if (parseYamlScalarInt(per_cgra_map_key_value_pair.getValue(),
                                 temp_value))
            per_cgra_rows = temp_value;
        } else if (per_cgra_map_key_ref == kColumns) {
          if (parseYamlScalarInt(per_cgra_map_key_value_pair.getValue(),
                                 temp_value))
            per_cgra_columns = temp_value;
        } else if (per_cgra_map_key_ref == kBaseTopology) {
          std::string topo_str;
          if (parseYamlScalarString(per_cgra_map_key_value_pair.getValue(),
                                    topo_str))
            per_cgra_base_topology = parseTopologyString(topo_str);
        } else if (per_cgra_map_key_ref == kCtrlMemItems) {
          if (parseYamlScalarInt(per_cgra_map_key_value_pair.getValue(),
                                 temp_value))
            max_ctrl_mem_items = temp_value;
        }
      }
    } else if (key_ref == kTileDefaults) {
      auto *tile_defaults_map = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(
          key_value_pair.getValue());
      if (tile_defaults_map)
        parseTileDefaults(tile_defaults_map, tile_defaults);
    } else if (key_ref == kTileOverrides) {
      auto *tile_overrides_seq =
          llvm::dyn_cast_or_null<llvm::yaml::SequenceNode>(
              key_value_pair.getValue());
      if (tile_overrides_seq)
        parseTileOverrides(tile_overrides_seq, tile_overrides);
    } else if (key_ref == kLinkDefaults) {
      auto *link_defaults_map = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(
          key_value_pair.getValue());
      if (link_defaults_map)
        parseLinkDefaults(link_defaults_map, link_defaults);
    } else if (key_ref == kLinkOverrides) {
      auto *link_overrides_seq =
          llvm::dyn_cast_or_null<llvm::yaml::SequenceNode>(
              key_value_pair.getValue());
      if (link_overrides_seq)
        parseLinkOverrides(link_overrides_seq, link_overrides);
    } else {
      llvm::errs() << "[MapToAcceleratorPass] Unknown YAML root key: "
                   << key_ref << "\n";
    }
  }
  return true;
}

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
  Option<bool> dumpMappingTable{
      *this, "dump-mapping-table",
      llvm::cl::desc(
          "Dump the resource allocation table after mapping (default: true)"),
      llvm::cl::init(true)};

  // Configures mapping strategy and mode based on command-line options.
  bool configureMappingStrategy(StringRef mapping_strategy_opt,
                                StringRef backtrack_config_opt,
                                StringRef mapping_mode_opt,
                                std::unique_ptr<Mapping> &mapping_strategy,
                                std::string &resolved_mapping_mode,
                                std::string &resolved_mapping_strategy,
                                bool &is_spatial_only) {
    StringRef mapping_mode_str = mapping_mode_opt;
    if (mapping_mode_str.empty()) {
      mapping_mode_str = "spatial-temporal";
    }
    if (mapping_mode_str == "spatial-only" ||
        mapping_mode_str == "spatial-temporal") {
      llvm::errs() << "[MapToAcceleratorPass] Using Mapping Mode: "
                   << mapping_mode_str << "\n";
    } else {
      llvm::errs() << "[MapToAcceleratorPass] Unsupported mapping mode: "
                   << mapping_mode_str << "\n";
      return false;
    }
    resolved_mapping_mode = mapping_mode_str.str();
    is_spatial_only = (mapping_mode_str == "spatial-only");

    StringRef mapping_strategy_str = mapping_strategy_opt;
    if (mapping_strategy_str.empty()) {
      mapping_strategy_str = "heuristic";
    }
    StringRef backtrack_str = backtrack_config_opt;
    if (mapping_strategy_str.empty() || mapping_strategy_str == "heuristic") {
      if (backtrack_str.empty()) {
        backtrack_str = "heuristic";
      }
      if (backtrack_str == "simple") {
        mapping_strategy = std::make_unique<HeuristicMapping>(1, 1);
      } else if (backtrack_str == "greedy") {
        mapping_strategy = std::make_unique<HeuristicMapping>(INT_MAX, 1);
      } else if (backtrack_str == "exhaustive") {
        mapping_strategy = std::make_unique<HeuristicMapping>(INT_MAX, INT_MAX);
      } else if (backtrack_str == "customized") {
        mapping_strategy = std::make_unique<HeuristicMapping>(5, 3);
      } else if (backtrack_str.starts_with("customized=")) {
        StringRef params = backtrack_str.substr(strlen("customized="));
        size_t comma_pos = params.find(',');
        if (comma_pos != StringRef::npos) {
          StringRef max_loc_str = params.substr(0, comma_pos);
          StringRef max_depth_str = params.substr(comma_pos + 1);
          int max_loc = 0, max_depth = 0;
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
                         << backtrack_str << "\n";
            return false;
          }
        } else {
          llvm::errs() << "[MapToAcceleratorPass] Illegal customized "
                          "parameters format: "
                       << backtrack_str << "\n";
          return false;
        }
      } else {
        llvm::errs() << "[MapToAcceleratorPass] Unsupported backtrack config: "
                     << backtrack_str << "\n";
        return false;
      }
      resolved_mapping_strategy = mapping_strategy_str.str();
    } else {
      llvm::errs() << "[MapToAcceleratorPass] Unsupported mapping strategy: "
                   << mapping_strategy_str << "\n";
      return false;
    }
    return true;
  }

  // Assigns unique dfg_id to all operations in SSA topological order.
  void assignDfgIds(func::FuncOp func) {
    // Uses existing topological sort to get all operations in order.
    std::vector<Operation *> sorted_ops = getTopologicallySortedOps(func);

    auto ctx = func.getContext();
    int next_id = 0;

    // Assigns ID to each operation in topological order.
    for (Operation *op : sorted_ops) {
      op->setAttr("dfg_id",
                  IntegerAttr::get(IntegerType::get(ctx, 32), next_id));
      llvm::errs() << "[MapToAcceleratorPass] Assigned dfg_id=" << next_id
                   << " to " << *op << "\n";
      next_id++;
    }

    llvm::errs() << "[MapToAcceleratorPass] Assigned " << next_id
                 << " dfg_id(s) in total\n";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::errs() << "[MapToAcceleratorPass] Starting mapping pass...\n";
    std::unique_ptr<Mapping> mapping_strategy;
    std::string resolved_mapping_mode;
    std::string resolved_mapping_strategy;
    bool is_spatial_only = false;
    if (!configureMappingStrategy(
            mappingStrategy.getValue(), backtrackConfig.getValue(),
            mappingMode.getValue(), mapping_strategy, resolved_mapping_mode,
            resolved_mapping_strategy, is_spatial_only)) {
      return;
    }

    // Handle architecture specification file
    constexpr int kMultiCgraDefaultRows = 1;
    constexpr int kMultiCgraDefaultColumns = 1;
    constexpr int kPerCgraDefaultRows = 4;
    constexpr int kPerCgraDefaultColumns = 4;
    constexpr int kDefaultMaxCtrlMemItems = 20;

    std::string architecture_spec_file = mlir::neura::getArchitectureSpecFile();
    int multi_cgra_rows = kMultiCgraDefaultRows;
    int multi_cgra_columns = kMultiCgraDefaultColumns;
    int per_cgra_rows = kPerCgraDefaultRows;
    int per_cgra_columns = kPerCgraDefaultColumns;
    int max_ctrl_mem_items = kDefaultMaxCtrlMemItems;
    mlir::neura::TileDefaults tile_defaults;
    std::vector<mlir::neura::TileOverride> tile_overrides;
    mlir::neura::LinkDefaults link_defaults;
    std::vector<mlir::neura::LinkOverride> link_overrides;
    mlir::neura::BaseTopology multi_cgra_base_topology =
        mlir::neura::BaseTopology::MESH;
    mlir::neura::BaseTopology per_cgra_base_topology =
        mlir::neura::BaseTopology::MESH;

    if (!architecture_spec_file.empty()) {

      // Use LLVM YAML parser to validate the YAML syntax (no mapping yet)
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer_or_err =
          llvm::MemoryBuffer::getFile(architecture_spec_file);
      if (!buffer_or_err) {
        llvm::errs() << "[MapToAcceleratorPass] Failed to open architecture "
                        "specification file: "
                     << architecture_spec_file << "\n";
        return;
      }

      llvm::SourceMgr sm;
      sm.AddNewSourceBuffer(std::move(*buffer_or_err), llvm::SMLoc());
      llvm::yaml::Stream yaml_stream(
          sm.getMemoryBuffer(sm.getMainFileID())->getBuffer(), sm);

      bool parse_failed = false;
      llvm::yaml::Document &yaml_doc = *yaml_stream.begin();
      (void)yaml_doc; // ensure document is created
      if (yaml_stream.failed()) {
        parse_failed = true;
      }

      if (parse_failed) {
        llvm::errs() << "[MapToAcceleratorPass] YAML parse error in: "
                     << architecture_spec_file << "\n";
        return;
      }

      // Parse YAML configuration
      if (!parseArchitectureYaml(
              yaml_doc, multi_cgra_rows, multi_cgra_columns,
              multi_cgra_base_topology, per_cgra_rows, per_cgra_columns,
              per_cgra_base_topology, max_ctrl_mem_items, tile_defaults,
              tile_overrides, link_defaults, link_overrides)) {
        return;
      }
    } else {
      llvm::errs() << "[MapToAcceleratorPass] No architecture specification "
                      "file provided.\n";
    }
    // assert(false);
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
        if (!is_spatial_only) {
          func.emitError() << "Steering IR mode requires spatial-only mapping, "
                           << "but got mapping mode: " << resolved_mapping_mode;
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

      // Always use full constructor with YAML configuration
      Architecture architecture(
          multi_cgra_rows, multi_cgra_columns, multi_cgra_base_topology,
          per_cgra_rows, per_cgra_columns, per_cgra_base_topology,
          tile_defaults, tile_overrides, link_defaults, link_overrides);
      int res_mii = calculateResMii(func, architecture);

      const int possible_min_ii = std::max(rec_mii, res_mii);
      const int max_ii =
          max_ctrl_mem_items; // Use YAML config (default 20 if not specified)

      std::vector<Operation *> topologically_sorted_ops =
          getTopologicallySortedOps(func);
      if (topologically_sorted_ops.empty()) {
        llvm::errs()
            << "[MapToAcceleratorPass] No operations to map in function "
            << func.getName() << "\n";
        assert(false && "Mapping aborted due to empty op list.");
      }

      // Filter out operations inside fused_op regions
      // Only map the fused_op itself, not the operations within its region
      std::vector<Operation *> filtered_ops;
      int skipped_count = 0;
      for (Operation *op : topologically_sorted_ops) {
        Operation *parent_op = op->getParentOp();
        // Check if parent is a fused_op by checking operation name
        if (parent_op && parent_op->getName().getStringRef().contains("fused_op")) {
          // Skip operations inside fused_op region
          llvm::outs() << "[MapToAcceleratorPass] Skipping op inside fused_op: "
                       << *op << "\n";
          skipped_count++;
          continue;
        }
        filtered_ops.push_back(op);
      }
      topologically_sorted_ops = std::move(filtered_ops);
      
      if (skipped_count > 0) {
        llvm::errs() << "[MapToAcceleratorPass] Filtered out " << skipped_count
                     << " operations inside fused_op regions\n";
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
        llvm::outs() << "[MapToAcceleratorPass] ALAP sorted op: " << *op
                     << " (ALAP level: " << level << ")\n";
      }
      // assert(false);
      for (int ii = possible_min_ii; ii <= max_ii; ++ii) {
        llvm::errs()
            << "[MapToAcceleratorPass] Start mapping with target II of " << ii
            << "\n";
        // Creates a mapping state for the current II.
        MappingState mapping_state(architecture, ii, is_spatial_only);
        if (mapping_strategy->map(sorted_ops_with_alap_levels, critical_ops,
                                  architecture, mapping_state)) {
          // success
          if (dumpMappingTable) {
            // logs to stderr
            mapping_state.dumpOpToLocs();
          }
          mapping_state.encodeMappingState();

          // Assigns unique dfg_id to all operations in SSA topological order.
          assignDfgIds(func);

          // Sets the mapping_info attribute on the function.
          auto ctx = func.getContext();
          SmallVector<NamedAttribute, 8> mapping_attrs;
          mapping_attrs.push_back(NamedAttribute(
              StringAttr::get(ctx, "x_tiles"),
              IntegerAttr::get(IntegerType::get(ctx, 32),
                               architecture.getPerCgraColumns())));
          mapping_attrs.push_back(
              NamedAttribute(StringAttr::get(ctx, "y_tiles"),
                             IntegerAttr::get(IntegerType::get(ctx, 32),
                                              architecture.getPerCgraRows())));
          mapping_attrs.push_back(
              NamedAttribute(StringAttr::get(ctx, "mapping_strategy"),
                             StringAttr::get(ctx, resolved_mapping_strategy)));
          mapping_attrs.push_back(
              NamedAttribute(StringAttr::get(ctx, "mapping_mode"),
                             StringAttr::get(ctx, resolved_mapping_mode)));
          mapping_attrs.push_back(
              NamedAttribute(StringAttr::get(ctx, "compiled_ii"),
                             IntegerAttr::get(IntegerType::get(ctx, 32), ii)));
          mapping_attrs.push_back(NamedAttribute(
              StringAttr::get(ctx, "rec_mii"),
              IntegerAttr::get(IntegerType::get(ctx, 32), rec_mii)));
          mapping_attrs.push_back(NamedAttribute(
              StringAttr::get(ctx, "res_mii"),
              IntegerAttr::get(IntegerType::get(ctx, 32), res_mii)));
          DictionaryAttr mapping_info = DictionaryAttr::get(ctx, mapping_attrs);

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
