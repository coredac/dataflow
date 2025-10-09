#include <deque>
#include <memory>
#include <fstream>

#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Architecture/ArchitectureSpec.h"
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
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::neura;

#define GEN_PASS_DEF_MAPTOACCELERATOR
#include "NeuraDialect/NeuraPasses.h.inc"

// Use the TileOverride from ArchitectureSpec.h

// Helper function to parse tile defaults.
bool parseTileDefaults(llvm::yaml::MappingNode *tile_defaults_map, mlir::neura::TileDefaults &tile_defaults) {
  for (auto &key_value_pair : *tile_defaults_map) {
    auto *key_node = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(key_value_pair.getKey());
    if (!key_node) continue;
    
    llvm::SmallString<64> key_string;
    llvm::StringRef key_ref = key_node->getValue(key_string);
    
    if (key_ref == "num_registers") {
      auto *value_node = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(key_value_pair.getValue());
      if (value_node) {
        llvm::SmallString<64> value_string;
        llvm::StringRef value_ref = value_node->getValue(value_string);
        long long temp_value = 0;
        if (!value_ref.getAsInteger(10, temp_value)) {
          tile_defaults.num_registers = static_cast<int>(temp_value);
        }
      }
    } else if (key_ref == "operations") {
      auto *value_node = llvm::dyn_cast_or_null<llvm::yaml::SequenceNode>(key_value_pair.getValue());
      if (value_node) {
        tile_defaults.operations.clear();
        for (auto &operation_node : *value_node) {
          auto *operation_scalar = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(&operation_node);
          if (operation_scalar) {
            llvm::SmallString<64> operation_string;
            llvm::StringRef operation_ref = operation_scalar->getValue(operation_string);
            tile_defaults.operations.push_back(operation_ref.str());
          }
        }
      }
    } else if (key_ref == "default_ports") {
      auto *value_node = llvm::dyn_cast_or_null<llvm::yaml::SequenceNode>(key_value_pair.getValue());
      if (value_node) {
        tile_defaults.default_ports.clear();
        for (auto &port_node : *value_node) {
          auto *port_scalar = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(&port_node);
          if (port_scalar) {
            llvm::SmallString<64> port_string;
            llvm::StringRef port_ref = port_scalar->getValue(port_string);
            tile_defaults.default_ports.push_back(port_ref.str());
          }
        }
      }
    }
  }
  return true;
}

// Helper function to parse tile override coordinates and ID.
void parseTileOverrideCoordinates(llvm::yaml::MappingNode *overrideMap, mlir::neura::TileOverride &override) {
  for (auto &keyValuePair : *overrideMap) {
    auto *keyNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getKey());
    if (!keyNode) continue;
    
    llvm::SmallString<64> keyString;
    llvm::StringRef keyRef = keyNode->getValue(keyString);
    
    if (keyRef == "id") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.id = static_cast<int>(tempValue);
        }
      }
    } else if (keyRef == "x") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.x = static_cast<int>(tempValue);
        }
      }
    } else if (keyRef == "y") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.y = static_cast<int>(tempValue);
        }
      }
    }
  }
}

// Helper function to parse tile override operations and registers.
void parseTileOverrideOperations(llvm::yaml::MappingNode *overrideMap, mlir::neura::TileOverride &override) {
  for (auto &keyValuePair : *overrideMap) {
    auto *keyNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getKey());
    if (!keyNode) continue;
    
    llvm::SmallString<64> keyString;
    llvm::StringRef keyRef = keyNode->getValue(keyString);
    
    if (keyRef == "operations") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::SequenceNode>(keyValuePair.getValue());
      if (valueNode) {
        override.operations.clear();
        for (auto &operationNode : *valueNode) {
          auto *operationScalar = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(&operationNode);
          if (operationScalar) {
            llvm::SmallString<64> operationString;
            llvm::StringRef operationRef = operationScalar->getValue(operationString);
            override.operations.push_back(operationRef.str());
          }
        }
      }
    } else if (keyRef == "num_registers") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.num_registers = static_cast<int>(tempValue);
        }
      }
    }
  }
}

// Helper function to parse tile override ports and memory.
void parseTileOverridePortsAndMemory(llvm::yaml::MappingNode *overrideMap, mlir::neura::TileOverride &override) {
  for (auto &keyValuePair : *overrideMap) {
    auto *keyNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getKey());
    if (!keyNode) continue;
    
    llvm::SmallString<64> keyString;
    llvm::StringRef keyRef = keyNode->getValue(keyString);
    
    if (keyRef == "ports") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::SequenceNode>(keyValuePair.getValue());
      if (valueNode) {
        override.ports.clear();
        for (auto &portNode : *valueNode) {
          auto *portScalar = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(&portNode);
          if (portScalar) {
            llvm::SmallString<64> portString;
            llvm::StringRef portRef = portScalar->getValue(portString);
            override.ports.push_back(portRef.str());
          }
        }
      }
    } else if (keyRef == "memory") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(keyValuePair.getValue());
      if (valueNode) {
        for (auto &memoryKeyValuePair : *valueNode) {
          auto *memoryKeyNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(memoryKeyValuePair.getKey());
          if (!memoryKeyNode) continue;
          
          llvm::SmallString<64> memoryKeyString;
          llvm::StringRef memoryKeyRef = memoryKeyNode->getValue(memoryKeyString);
          
          if (memoryKeyRef == "capacity") {
            auto *memoryValueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(memoryKeyValuePair.getValue());
            if (memoryValueNode) {
              llvm::SmallString<64> memoryValueString;
              llvm::StringRef memoryValueRef = memoryValueNode->getValue(memoryValueString);
              long long tempValue = 0;
              if (!memoryValueRef.getAsInteger(10, tempValue)) {
                override.memory.capacity = static_cast<int>(tempValue);
              }
            }
          }
        }
      }
    }
  }
}

// Helper function to parse a single tile override.
void parseSingleTileOverride(llvm::yaml::MappingNode *overrideMap, mlir::neura::TileOverride &override) {
  for (auto &keyValuePair : *overrideMap) {
    auto *keyNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getKey());
    if (!keyNode) continue;
    
    llvm::SmallString<64> keyString;
    llvm::StringRef keyRef = keyNode->getValue(keyString);
    
    if (keyRef == "id") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.id = static_cast<int>(tempValue);
        }
      }
    } else if (keyRef == "x") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.x = static_cast<int>(tempValue);
        }
      }
    } else if (keyRef == "y") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.y = static_cast<int>(tempValue);
        }
      }
    } else if (keyRef == "operations") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::SequenceNode>(keyValuePair.getValue());
      if (valueNode) {
        override.operations.clear();
        for (auto &operationNode : *valueNode) {
          auto *operationScalar = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(&operationNode);
          if (operationScalar) {
            llvm::SmallString<64> operationString;
            llvm::StringRef operationRef = operationScalar->getValue(operationString);
            override.operations.push_back(operationRef.str());
          }
        }
      }
    } else if (keyRef == "num_registers") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.num_registers = static_cast<int>(tempValue);
        }
      }
    } else if (keyRef == "ports") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::SequenceNode>(keyValuePair.getValue());
      if (valueNode) {
        override.ports.clear();
        for (auto &portNode : *valueNode) {
          auto *portScalar = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(&portNode);
          if (portScalar) {
            llvm::SmallString<64> portString;
            llvm::StringRef portRef = portScalar->getValue(portString);
            override.ports.push_back(portRef.str());
          }
        }
      }
    } else if (keyRef == "memory") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(keyValuePair.getValue());
      if (valueNode) {
        for (auto &memoryKeyValuePair : *valueNode) {
          auto *memoryKeyNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(memoryKeyValuePair.getKey());
          if (!memoryKeyNode) continue;
          
          llvm::SmallString<64> memoryKeyString;
          llvm::StringRef memoryKeyRef = memoryKeyNode->getValue(memoryKeyString);
          
          if (memoryKeyRef == "capacity") {
            auto *memoryValueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(memoryKeyValuePair.getValue());
            if (memoryValueNode) {
              llvm::SmallString<64> memoryValueString;
              llvm::StringRef memoryValueRef = memoryValueNode->getValue(memoryValueString);
              long long tempValue = 0;
              if (!memoryValueRef.getAsInteger(10, tempValue)) {
                override.memory.capacity = static_cast<int>(tempValue);
              }
            }
          }
        }
      }
    }
  }
}

// Helper function to parse tile overrides.
bool parseTileOverrides(llvm::yaml::SequenceNode *tile_overrides_seq, std::vector<mlir::neura::TileOverride> &tile_overrides) {
  for (auto &override_node : *tile_overrides_seq) {
    auto *override_map = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(&override_node);
    if (!override_map) continue;

    mlir::neura::TileOverride override;
    parseSingleTileOverride(override_map, override);
    tile_overrides.push_back(override);
  }
  return true;
}

// Helper function to parse link defaults.
bool parseLinkDefaults(llvm::yaml::MappingNode *link_defaults_map, mlir::neura::LinkDefaults &link_defaults) {
  for (auto &key_value_pair : *link_defaults_map) {
    auto *key_node = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(key_value_pair.getKey());
    if (!key_node) continue;
    
    llvm::SmallString<64> key_string;
    llvm::StringRef key_ref = key_node->getValue(key_string);
    
    if (key_ref == "latency") {
      auto *value_node = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(key_value_pair.getValue());
      if (value_node) {
        llvm::SmallString<64> value_string;
        llvm::StringRef value_ref = value_node->getValue(value_string);
        long long temp_value = 0;
        if (!value_ref.getAsInteger(10, temp_value)) {
          link_defaults.latency = static_cast<int>(temp_value);
        }
      }
    } else if (key_ref == "bandwidth") {
      auto *value_node = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(key_value_pair.getValue());
      if (value_node) {
        llvm::SmallString<64> value_string;
        llvm::StringRef value_ref = value_node->getValue(value_string);
        long long temp_value = 0;
        if (!value_ref.getAsInteger(10, temp_value)) {
          link_defaults.bandwidth = static_cast<int>(temp_value);
        }
      }
    }
  }
  return true;
}

// Helper function to parse link override properties.
void parseLinkOverrideProperties(llvm::yaml::MappingNode *overrideMap, mlir::neura::LinkOverride &override) {
  for (auto &keyValuePair : *overrideMap) {
    auto *keyNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getKey());
    if (!keyNode) continue;
    
    llvm::SmallString<64> keyString;
    llvm::StringRef keyRef = keyNode->getValue(keyString);
    
    if (keyRef == "id") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.id = static_cast<int>(tempValue);
        }
      }
    } else if (keyRef == "latency") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.latency = static_cast<int>(tempValue);
        }
      }
    } else if (keyRef == "bandwidth") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.bandwidth = static_cast<int>(tempValue);
        }
      }
    }
  }
}

// Helper function to parse link override tile IDs and existence.
void parseLinkOverrideTilesAndExistence(llvm::yaml::MappingNode *overrideMap, mlir::neura::LinkOverride &override) {
  for (auto &keyValuePair : *overrideMap) {
    auto *keyNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getKey());
    if (!keyNode) continue;
    
    llvm::SmallString<64> keyString;
    llvm::StringRef keyRef = keyNode->getValue(keyString);
    
    if (keyRef == "src_tile_id") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.src_tile_id = static_cast<int>(tempValue);
        }
      }
    } else if (keyRef == "dst_tile_id") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.dst_tile_id = static_cast<int>(tempValue);
        }
      }
    } else if (keyRef == "existence") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        override.existence = (valueRef == "true" || valueRef == "True" || valueRef == "1");
      }
    }
  }
}

// Helper function to parse a single link override.
void parseSingleLinkOverride(llvm::yaml::MappingNode *overrideMap, mlir::neura::LinkOverride &override) {
  for (auto &keyValuePair : *overrideMap) {
    auto *keyNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getKey());
    if (!keyNode) continue;
    
    llvm::SmallString<64> keyString;
    llvm::StringRef keyRef = keyNode->getValue(keyString);
    
    if (keyRef == "id") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.id = static_cast<int>(tempValue);
        }
      }
    } else if (keyRef == "latency") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.latency = static_cast<int>(tempValue);
        }
      }
    } else if (keyRef == "bandwidth") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.bandwidth = static_cast<int>(tempValue);
        }
      }
    } else if (keyRef == "src_tile_id") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.src_tile_id = static_cast<int>(tempValue);
        }
      }
    } else if (keyRef == "dst_tile_id") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        long long tempValue = 0;
        if (!valueRef.getAsInteger(10, tempValue)) {
          override.dst_tile_id = static_cast<int>(tempValue);
        }
      }
    } else if (keyRef == "existence") {
      auto *valueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
      if (valueNode) {
        llvm::SmallString<64> valueString;
        llvm::StringRef valueRef = valueNode->getValue(valueString);
        override.existence = (valueRef == "true" || valueRef == "True" || valueRef == "1");
      }
    }
  }
}

// Helper function to parse link overrides.
bool parseLinkOverrides(llvm::yaml::SequenceNode *link_overrides_seq, std::vector<mlir::neura::LinkOverride> &link_overrides) {
  for (auto &override_node : *link_overrides_seq) {
    auto *override_map = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(&override_node);
    if (!override_map) continue;

    mlir::neura::LinkOverride override;
    parseSingleLinkOverride(override_map, override);
    link_overrides.push_back(override);
  }
  return true;
}

// Helper function to parse topology string to BaseTopology enum
mlir::neura::BaseTopology parseTopologyString(const std::string& topology_str) {
  if (topology_str == "mesh") {
    return mlir::neura::BaseTopology::MESH;
  } else if (topology_str == "king_mesh" || topology_str == "king mesh") {
    return mlir::neura::BaseTopology::KING_MESH;
  } else if (topology_str == "ring") {
    return mlir::neura::BaseTopology::RING;
  } else {
    // Default to mesh if unknown topology
    return mlir::neura::BaseTopology::MESH;
  }
}

// Helper function to parse architecture YAML configuration.
bool parseArchitectureYAML(llvm::yaml::Document &doc, int &width, int &height, 
                          mlir::neura::TileDefaults &tile_defaults,
                          std::vector<mlir::neura::TileOverride> &tile_overrides,
                          mlir::neura::LinkDefaults &link_defaults,
                          std::vector<mlir::neura::LinkOverride> &link_overrides,
                          mlir::neura::BaseTopology &base_topology) {
  auto *root = doc.getRoot();
  if (!root) {
    llvm::errs() << "[MapToAcceleratorPass] Empty YAML document\n";
    return false;
  }
  
  auto *rootMap = llvm::dyn_cast<llvm::yaml::MappingNode>(root);
  if (!rootMap) {
    llvm::errs() << "[MapToAcceleratorPass] YAML root is not a mapping\n";
    return false;
  }

  // Iterate root mapping ONCE; find 'architecture' and 'tile_defaults'.
  for (auto &keyValuePair : *rootMap) {
    auto *keyNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getKey());
    if (!keyNode) continue;
    
    llvm::SmallString<64> keyString;
    llvm::StringRef keyRef = keyNode->getValue(keyString);
    
    if (keyRef == "architecture") {
      auto *architectureMap = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(keyValuePair.getValue());
      if (!architectureMap) continue;

      // Iterate architecture mapping ONCE; read width/height in the same pass.
      for (auto &architectureKeyValuePair : *architectureMap) {
        auto *architectureKeyNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(architectureKeyValuePair.getKey());
        if (!architectureKeyNode) continue;
        
        llvm::SmallString<64> architectureKeyString;
        llvm::StringRef architectureKeyRef = architectureKeyNode->getValue(architectureKeyString);
        if (architectureKeyRef != "width" && architectureKeyRef != "height") continue;
        
        auto *architectureValueNode = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(architectureKeyValuePair.getValue());
        if (!architectureValueNode) continue;
        
        llvm::SmallString<64> architectureValueString;
        llvm::StringRef architectureValueRef = architectureValueNode->getValue(architectureValueString);
        long long tempValue = 0;
        if (!architectureValueRef.getAsInteger(10, tempValue)) {
          if (architectureKeyRef == "width") width = static_cast<int>(tempValue);
          if (architectureKeyRef == "height") height = static_cast<int>(tempValue);
        }
      }
        } else if (keyRef == "tile_defaults") {
          auto *tile_defaults_map = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(keyValuePair.getValue());
          if (tile_defaults_map) {
            parseTileDefaults(tile_defaults_map, tile_defaults);
          }
        } else if (keyRef == "tile_overrides") {
          auto *tile_overrides_seq = llvm::dyn_cast_or_null<llvm::yaml::SequenceNode>(keyValuePair.getValue());
          if (tile_overrides_seq) {
            parseTileOverrides(tile_overrides_seq, tile_overrides);
          }
        } else if (keyRef == "link_defaults") {
          auto *link_defaults_map = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(keyValuePair.getValue());
          if (link_defaults_map) {
            parseLinkDefaults(link_defaults_map, link_defaults);
          }
        } else if (keyRef == "link_overrides") {
          auto *link_overrides_seq = llvm::dyn_cast_or_null<llvm::yaml::SequenceNode>(keyValuePair.getValue());
          if (link_overrides_seq) {
            parseLinkOverrides(link_overrides_seq, link_overrides);
          }
        } else if (keyRef == "base_topology") {
          auto *topology_node = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(keyValuePair.getValue());
          if (topology_node) {
            llvm::SmallString<64> topology_string;
            llvm::StringRef topology_ref = topology_node->getValue(topology_string);
            base_topology = parseTopologyString(topology_ref.str());
          }
        }
  }

  if (width <= 0 || height <= 0) {
    width = -1;
    height = -1;
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

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::unique_ptr<Mapping> mapping_strategy;
    StringRef mappingStrategy_stringRef(mappingStrategy.getValue());
    StringRef backtrackConfig_stringRef(backtrackConfig.getValue());
    StringRef mappingMode_stringRef(mappingMode.getValue());
    bool is_spatial_only = (mappingMode_stringRef == "spatial-only");
    if (is_spatial_only || mappingMode_stringRef == "spatial-temporal" ||
        mappingMode_stringRef.empty()) {
      if (mappingMode_stringRef.empty()) {
        mappingMode_stringRef = "spatial-temporal";
      }
    } else {
      llvm::errs() << "[MapToAcceleratorPass] Unsupported mapping mode: "
                   << mappingMode_stringRef << "\n";
      return;
    }

    if (mappingStrategy_stringRef == "heuristic" ||
        mappingStrategy_stringRef.empty()) {
      mappingStrategy_stringRef = "heuristic";

      if (backtrackConfig_stringRef == "simple") {
        mapping_strategy = std::make_unique<HeuristicMapping>(1, 1);
      } else if (backtrackConfig_stringRef == "greedy") {
        mapping_strategy = std::make_unique<HeuristicMapping>(INT_MAX, 1);
      } else if (backtrackConfig_stringRef == "exhaustive") {
        mapping_strategy = std::make_unique<HeuristicMapping>(INT_MAX, INT_MAX);
      } else if (backtrackConfig_stringRef == "customized") {
        mapping_strategy = std::make_unique<HeuristicMapping>(5, 3);
      } else if (backtrackConfig_stringRef.starts_with("customized=")) {
        // Used for custom backtrack parameters.
        // Example: "customized=5,3" means max_loc=5, max_depth=3
        // Extracts the parameters after "customized=".
        StringRef paramsRef =
            backtrackConfig_stringRef.substr(strlen("customized="));
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
            llvm::errs() << "[MapToAcceleratorPass] Illegal customized "
                            "parameters format: "
                         << backtrackConfig_stringRef << "\n";
            return;
          }
        } else {
          llvm::errs()
              << "[MapToAcceleratorPass] Illegal customized parameters format: "
              << backtrackConfig_stringRef << "\n";
          return;
        }
      }
    } else {
      llvm::errs() << "[MapToAcceleratorPass] Unsupported mapping strategy: "
                   << mappingStrategy_stringRef << "\n";
      return;
    }

    // Handle architecture specification file
    std::string architecture_spec_file = mlir::neura::getArchitectureSpecFile();
    int yaml_width = -1;
    int yaml_height = -1;
    mlir::neura::TileDefaults yaml_tile_defaults;
    std::vector<mlir::neura::TileOverride> tile_overrides;
    mlir::neura::LinkDefaults yaml_link_defaults;
    std::vector<mlir::neura::LinkOverride> link_overrides;
    mlir::neura::BaseTopology base_topology = mlir::neura::BaseTopology::MESH; // Default to mesh
    if (!architecture_spec_file.empty()) {

      // Use LLVM YAML parser to validate the YAML syntax (no mapping yet)
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOrErr =
          llvm::MemoryBuffer::getFile(architecture_spec_file);
      if (!bufferOrErr) {
        llvm::errs() << "[MapToAcceleratorPass] Failed to open architecture specification file: "
                     << architecture_spec_file << "\n";
        return;
      }

      llvm::SourceMgr sm;
      sm.AddNewSourceBuffer(std::move(*bufferOrErr), llvm::SMLoc());
      llvm::yaml::Stream yamlStream(sm.getMemoryBuffer(sm.getMainFileID())->getBuffer(), sm);

      bool parseFailed = false;
      llvm::yaml::Document &firstDoc = *yamlStream.begin();
      (void)firstDoc; // ensure document is created
      if (yamlStream.failed()) {
        parseFailed = true;
      }

      if (parseFailed) {
        llvm::errs() << "[MapToAcceleratorPass] YAML parse error in: "
                     << architecture_spec_file << "\n";
        return;
      }

      // Parse YAML configuration
      if (!parseArchitectureYAML(firstDoc, yaml_width, yaml_height, yaml_tile_defaults, tile_overrides, yaml_link_defaults, link_overrides, base_topology)) {
        return;
      }

    } else {
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
      } else if (!longest) {
        rec_mii = 1; // No recurrence cycles found, set MII to 1.
      }

      // Construct architecture from YAML configuration
      int arch_w = (yaml_width > 0 ? yaml_width : 4);
      int arch_h = (yaml_height > 0 ? yaml_height : 4);
      
      // Always use full constructor with YAML configuration
      Architecture architecture(arch_w, arch_h, yaml_tile_defaults, tile_overrides, yaml_link_defaults, link_overrides, base_topology);
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
      for (int ii = possibleMinII; ii <= maxII; ++ii) {
        llvm::errs()
            << "[MapToAcceleratorPass] Start mapping with target II of " << ii
            << "\n";
        // Creates a mapping state for the current II.
        MappingState mapping_state(architecture, ii, is_spatial_only);
        if (mapping_strategy->map(sorted_ops_with_alap_levels, critical_ops,
                                  architecture, mapping_state)) {
          // success
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
               NamedAttribute(StringAttr::get(ctx, "mapping_strategy"),
                              StringAttr::get(ctx, mappingStrategy_stringRef)),
               NamedAttribute(StringAttr::get(ctx, "mapping_mode"),
                              StringAttr::get(ctx, mappingMode_stringRef)),
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
