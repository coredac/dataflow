#include "NeuraDialect/Util/ParserUtils.h"

using namespace mlir::neura::yamlkeys;
using namespace mlir::neura;
namespace mlir {
namespace neura {
namespace util {
// Utility: Extracts an integer from a YAML ScalarNode. Returns true on success.
bool parseYamlScalarInt(const llvm::yaml::Node *node, int &result) {
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
bool parseYamlScalarString(const llvm::yaml::Node *node, std::string &result) {
  auto *scalar = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(node);
  if (!scalar)
    return false;
  llvm::SmallString<64> value_string;
  llvm::StringRef value_ref = scalar->getValue(value_string);
  result = value_ref.str();
  return true;
}

// Utility: Extracts a vector of strings from a YAML SequenceNode.
void parseYamlStringSequence(llvm::yaml::Node *node,
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
bool yamlParseError(const std::string &msg, const std::string &file) {
  llvm::errs() << "YAML parse error";
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
      llvm::errs() << "Unknown tile_defaults key: " << key_ref << "\n";
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
      llvm::errs() << "Unknown tile_override key: " << key_ref << "\n";
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
        override.existence =
            (value == "true" || value == "True" || value == "1");
      }
    } else {
      llvm::errs() << "Unknown tile_override key: " << key_ref << "\n";
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
      llvm::errs() << "Unknown link_defaults key: " << key_ref << "\n";
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
      llvm::errs() << "Unknown link_override key: " << key_ref << "\n";
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
BaseTopology parseTopologyString(const std::string &topology_str) {
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
} // namespace util
} // namespace neura
} // namespace mlir
