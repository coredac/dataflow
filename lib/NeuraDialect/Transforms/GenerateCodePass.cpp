#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include <sstream>
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <set>
#include <algorithm>
#include <queue>


using namespace mlir;
using namespace mlir::neura;

#define GEN_PASS_DEF_GenerateCode
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

// Calculates direction of data movement from sender's perspective.
// For example: if data moves from (1,1) to (1,2), then PE(1,1) sends to North.
llvm::StringRef calculateSendDirection(int src_x, int src_y, int dst_x, int dst_y) {
  int dx = dst_x - src_x;
  int dy = dst_y - src_y;
  if (dx == 0 && dy == 0) return "Local";
  if (dx == 0 && dy > 0) return "North";
  if (dx == 0 && dy < 0) return "South";
  if (dx > 0 && dy == 0) return "East";
  if (dx < 0 && dy == 0) return "West";
  if (dx > 0 && dy > 0) return "NorthEast";
  if (dx > 0 && dy < 0) return "SouthEast";
  if (dx < 0 && dy > 0) return "NorthWest";
  if (dx < 0 && dy < 0) return "SouthWest";
  return "Unknown";
}

// Calculates direction of data movement from receiver's perspective.
// For example: if data moves from (1,1) to (1,2), then PE(1,2) receives from South.
llvm::StringRef calculateRecvDirection(int src_x, int src_y, int dst_x, int dst_y) {
  int dx = dst_x - src_x;
  int dy = dst_y - src_y;
  if (dx == 0 && dy == 0) return "Local";
  if (dx == 0 && dy > 0) return "South";  // data moves up, receiver gets from South
  if (dx == 0 && dy < 0) return "North";  // data moves down, receiver gets from North
  if (dx > 0 && dy == 0) return "West";   // data moves right, receiver gets from West
  if (dx < 0 && dy == 0) return "East";   // data moves left, receiver gets from East
  if (dx > 0 && dy > 0) return "SouthWest";
  if (dx > 0 && dy < 0) return "NorthWest";
  if (dx < 0 && dy > 0) return "SouthEast";
  if (dx < 0 && dy < 0) return "NorthEast";
  return "Unknown";
}

struct GenerateCodePass
    : public PassWrapper<GenerateCodePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateCodePass)

  StringRef getArgument() const override { return "generate-code"; }
  StringRef getDescription() const override {
    return "Generates YAML code from mapped Neura IR.";
  }

  // Simple YAML node class
  struct YAMLNode {
    enum Type { SCALAR, SEQUENCE, MAPPING };
    Type type;
    std::string scalar_value;
    std::vector<YAMLNode> sequence_value;
    std::map<std::string, YAMLNode> mapping_value;
    
    YAMLNode() : type(SCALAR) {}
    YAMLNode(const std::string& value) : type(SCALAR), scalar_value(value) {}
    
    static YAMLNode Scalar(const std::string& value) {
      return YAMLNode(value);
    }
    
    static YAMLNode Sequence() {
      YAMLNode node;
      node.type = SEQUENCE;
      return node;
    }
    
    static YAMLNode Mapping() {
      YAMLNode node;
      node.type = MAPPING;
      return node;
    }
    
    void push_back(const YAMLNode& node) {
      if (type != SEQUENCE) {
        type = SEQUENCE;
        sequence_value.clear();
      }
      sequence_value.push_back(node);
    }
    
    void set(const std::string& key, const YAMLNode& value) {
      if (type != MAPPING) {
        type = MAPPING;
        mapping_value.clear();
      }
      mapping_value[key] = value;
    }
    
    void set(const std::string& key, const std::string& value) {
      set(key, Scalar(value));
    }
    
    void set(const std::string& key, int value) {
      set(key, Scalar(std::to_string(value)));
    }
    
    std::string toString(int indent = 0) const {
      std::stringstream ss;
      std::string indent_str(indent * 2, ' ');
      
      switch (type) {
        case SCALAR:
          ss << scalar_value;
          break;
          
        case SEQUENCE:
          if (sequence_value.empty()) {
            ss << "[]";
          } else {
            ss << "\n";
            for (size_t i = 0; i < sequence_value.size(); ++i) {
              ss << indent_str << "- " << sequence_value[i].toString(indent + 1);
              if (i < sequence_value.size() - 1) ss << "\n";
            }
          }
          break;
          
        case MAPPING:
          if (mapping_value.empty()) {
            ss << "{}";
          } else {
            ss << "\n";
            size_t i = 0;
            for (const auto& pair : mapping_value) {
              ss << indent_str << pair.first << ": " << pair.second.toString(indent + 1);
              if (i < mapping_value.size() - 1) ss << "\n";
              ++i;
            }
          }
          break;
      }
      
      return ss.str();
    }
  };

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect, mlir::neura::NeuraDialect>();
  }

  // Set helper function to process tile mapping for an operation.
  void processTileMapping(mlir::Operation* op, mlir::DictionaryAttr loc,
                         std::map<Operation*, std::pair<int, int>>& op_to_tile,
                         std::map<Operation*, std::pair<int, int>>& op_to_final_tile,
                         std::map<Operation*, std::pair<int, int>>& op_to_source_tile,
                         std::map<std::pair<int, int>, int>& tile_coord_to_id,
                         int& tile_id_counter) {
    auto x_attr = loc.getAs<IntegerAttr>("x");
    auto y_attr = loc.getAs<IntegerAttr>("y");
    auto id_attr = mlir::dyn_cast<IntegerAttr>(loc.get("id"));
    
    int x, y;
    if (x_attr && y_attr) {
      x = x_attr.getInt();
      y = y_attr.getInt();
    } else {
      x = id_attr ? id_attr.getInt() : 0;
      y = 0;
    }
    
    op_to_tile[op] = std::make_pair(x, y);
    op_to_final_tile[op] = std::make_pair(x, y);
    op_to_source_tile[op] = std::make_pair(x, y);
    auto coord = std::make_pair(x, y);
    if (tile_coord_to_id.find(coord) == tile_coord_to_id.end()) {
      tile_coord_to_id[coord] = tile_id_counter++;
    }
  }

  // Set helper function to process link mapping for an operation.
  void processLinkMapping(mlir::Operation* op, mlir::DictionaryAttr loc,
                         std::map<Operation*, std::pair<int, int>>& op_to_final_tile,
                         std::map<Operation*, std::pair<int, int>>& op_to_source_tile) {
    auto id_attr = mlir::dyn_cast<IntegerAttr>(loc.get("id"));
    (void)id_attr; // Suppress unused variable warning.
    
    for (mlir::Value result : op->getResults()) {
      if (!result) continue;
      for (mlir::Operation *user : result.getUsers()) {
        if (!user) continue;
        auto user_attr_array = user->getAttrOfType<ArrayAttr>("mapping_locs");
        if (user_attr_array) {
          for (mlir::Attribute user_attr : user_attr_array) {
            if (auto user_loc = mlir::dyn_cast<DictionaryAttr>(user_attr)) {
              auto user_resource_attr = mlir::dyn_cast<StringAttr>(user_loc.get("resource"));
              if (user_resource_attr && user_resource_attr.getValue() == "tile") {
                auto user_x_attr = user_loc.getAs<IntegerAttr>("x");
                auto user_y_attr = user_loc.getAs<IntegerAttr>("y");
                if (user_x_attr && user_y_attr) {
                  int dst_x = user_x_attr.getInt();
                  int dst_y = user_y_attr.getInt();
                  op_to_final_tile[op] = std::make_pair(dst_x, dst_y);
                  
                  // Set find source from operands.
                  for (mlir::Value operand : op->getOperands()) {
                    if (!operand) continue;
                    if (auto defining_op = operand.getDefiningOp()) {
                      auto src_it = op_to_source_tile.find(defining_op);
                      if (src_it != op_to_source_tile.end()) {
                        op_to_source_tile[op] = src_it->second;
                        break;
                      }
                    }
                  }
                  return;
                }
              }
            }
          }
        }
      }
    }
    
    // Set fallback: use source location from operands.
    if (op_to_final_tile.find(op) == op_to_final_tile.end()) {
      for (mlir::Value operand : op->getOperands()) {
        if (!operand) continue;
        if (auto defining_op = operand.getDefiningOp()) {
          auto it = op_to_final_tile.find(defining_op);
          if (it != op_to_final_tile.end()) {
            op_to_final_tile[op] = it->second;
            auto src_it = op_to_source_tile.find(defining_op);
            if (src_it != op_to_source_tile.end()) {
              op_to_source_tile[op] = src_it->second;
            }
            break;
          }
        }
      }
    }
  }

  // Set helper function to process register mapping for an operation.
  void processRegisterMapping(mlir::Operation* op, mlir::DictionaryAttr loc,
                             std::map<Operation*, std::pair<int, int>>& op_to_final_tile,
                             std::map<Operation*, std::pair<int, int>>& op_to_source_tile) {
    auto id_attr = mlir::dyn_cast<IntegerAttr>(loc.get("id"));
    auto timestep_attr = mlir::dyn_cast<IntegerAttr>(loc.get("time_step"));
    
    if (!id_attr || !timestep_attr) return;
    
    int register_id = id_attr.getInt();
    int time_step = timestep_attr.getInt();
    
    // Set register operations are part of data movement operations
    // Set they don't have direct tile coordinates, but they belong to either source or destination tile
    
    // Set try to determine if this register belongs to source or destination tile
    // Set by looking at the operation's context (operands and users)
    
    (void)register_id; // Suppress unused variable warning
    (void)time_step;   // Suppress unused variable warning
    
    // Set first, try to find the source tile by looking at the operation's operands
    for (mlir::Value operand : op->getOperands()) {
      if (!operand) continue;
      if (auto defining_op = operand.getDefiningOp()) {
        auto def_attr_array = defining_op->getAttrOfType<ArrayAttr>("mapping_locs");
        if (def_attr_array) {
          for (mlir::Attribute def_attr : def_attr_array) {
            if (auto def_loc = mlir::dyn_cast<DictionaryAttr>(def_attr)) {
              auto def_resource_attr = mlir::dyn_cast<StringAttr>(def_loc.get("resource"));
              if (def_resource_attr && def_resource_attr.getValue() == "tile") {
                auto def_x_attr = def_loc.getAs<IntegerAttr>("x");
                auto def_y_attr = def_loc.getAs<IntegerAttr>("y");
                if (def_x_attr && def_y_attr) {
                  int src_x = def_x_attr.getInt();
                  int src_y = def_y_attr.getInt();
                  op_to_source_tile[op] = std::make_pair(src_x, src_y);
                  break;
                }
              }
            }
          }
        }
      }
    }
    
    // Set then, try to find the destination tile by looking at operations that use this operation's results
    for (mlir::Value result : op->getResults()) {
      if (!result) continue;
      for (mlir::Operation *user : result.getUsers()) {
        if (!user) continue;
        auto user_attr_array = user->getAttrOfType<ArrayAttr>("mapping_locs");
        if (user_attr_array) {
          for (mlir::Attribute user_attr : user_attr_array) {
            if (auto user_loc = mlir::dyn_cast<DictionaryAttr>(user_attr)) {
              auto user_resource_attr = mlir::dyn_cast<StringAttr>(user_loc.get("resource"));
              if (user_resource_attr && user_resource_attr.getValue() == "tile") {
                auto user_x_attr = user_loc.getAs<IntegerAttr>("x");
                auto user_y_attr = user_loc.getAs<IntegerAttr>("y");
                if (user_x_attr && user_y_attr) {
                  int dst_x = user_x_attr.getInt();
                  int dst_y = user_y_attr.getInt();
                  op_to_final_tile[op] = std::make_pair(dst_x, dst_y);
                  break;
                }
              }
            }
          }
        }
      }
    }
    
    // Set if we found both source and destination, and they're the same, this is a local register
    if (op_to_source_tile.find(op) != op_to_source_tile.end() && 
        op_to_final_tile.find(op) != op_to_final_tile.end()) {
      auto src = op_to_source_tile[op];
      auto dst = op_to_final_tile[op];
      if (src == dst) {
        // Set this is a local register operation within the same tile
        return;
      }
    }
  }

  // Set helper function to collect operation tile mappings.
  void collectOperationMappings(mlir::func::FuncOp func,
                               std::map<Operation*, std::pair<int, int>>& op_to_tile,
                               std::map<Operation*, std::pair<int, int>>& op_to_final_tile,
                               std::map<Operation*, std::pair<int, int>>& op_to_source_tile,
                               std::map<std::pair<int, int>, int>& tile_coord_to_id,
                               int& tile_id_counter) {
    func.walk([&](Operation *op) {
      auto attr_array = op->getAttrOfType<ArrayAttr>("mapping_locs");
      if (!attr_array || attr_array.size() == 0) {
        return;
      }
      
      for (mlir::Attribute attr : attr_array) {
        if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
          auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
          if (!resource_attr) continue;
          
          if (resource_attr.getValue() == "tile") {
            processTileMapping(op, loc, op_to_tile, op_to_final_tile, op_to_source_tile, 
                             tile_coord_to_id, tile_id_counter);
            break;
          } else if (resource_attr.getValue() == "link") {
            processLinkMapping(op, loc, op_to_final_tile, op_to_source_tile);
            break;
          } else if (resource_attr.getValue() == "register") {
            processRegisterMapping(op, loc, op_to_final_tile, op_to_source_tile);
            break;
          }
        }
      }
    });
  }

  // Set helper function to calculate source direction for an operation.
  void calculateSourceDirection(mlir::Operation* op, int x, int y, YAMLNode& inst_obj) {
    bool found_src = false;
    for (mlir::Value operand : op->getOperands()) {
      if (!operand) continue;
      if (auto defining_op = operand.getDefiningOp()) {
        // Set find the source tile by looking at the defining operation's tile mapping.
        std::pair<int, int> src_coord = std::make_pair(x, y); // Defaults to current location.
        
        auto def_attr_array = defining_op->getAttrOfType<ArrayAttr>("mapping_locs");
        if (def_attr_array) {
          for (mlir::Attribute attr : def_attr_array) {
            if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
              auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
              if (resource_attr && resource_attr.getValue() == "tile") {
                auto x_attr = loc.getAs<IntegerAttr>("x");
                auto y_attr = loc.getAs<IntegerAttr>("y");
                if (x_attr && y_attr) {
                  src_coord = std::make_pair(x_attr.getInt(), y_attr.getInt());
                  break;
                }
              } else if (resource_attr && resource_attr.getValue() == "register") {
                // Set for register operations, we need to trace back to find the source tile
                // Set this is handled in the ASM generation, not in YAML
                continue;
              }
            }
          }
        }
        
        // Set calculate the direction FROM source TO current tile.
        inst_obj["src_direction"] = calculateRecvDirection(src_coord.first, src_coord.second, x, y).str();
        inst_obj["src_tile"] = "(" + std::to_string(src_coord.first) + "," + std::to_string(src_coord.second) + ")";
        found_src = true;
        break;
      }
    }

    // Set if no source tile is found, sets to Local.
    if (!found_src) {
      inst_obj["src_direction"] = "Local";
      inst_obj["src_tile"] = "(" + std::to_string(x) + "," + std::to_string(y) + ")";
    }
  }

  // Set helper function to calculate destination direction for an operation.
  void calculateDestinationDirection(mlir::Operation* op, int x, int y, 
                                   const std::map<Operation*, std::pair<int, int>>& op_to_final_tile,
                                   YAMLNode& inst_obj) {
    bool found_dst = false;
    for (mlir::Value result : op->getResults()) {
      if (!result) continue;
      for (mlir::Operation *user : result.getUsers()) {
        if (!user) continue;
        
        // Set check if user is a register operation
        auto user_attr_array = user->getAttrOfType<ArrayAttr>("mapping_locs");
        if (user_attr_array) {
          bool is_register_user = false;
          for (mlir::Attribute user_attr : user_attr_array) {
            if (auto user_loc = mlir::dyn_cast<DictionaryAttr>(user_attr)) {
              auto user_resource_attr = mlir::dyn_cast<StringAttr>(user_loc.get("resource"));
              if (user_resource_attr && user_resource_attr.getValue() == "register") {
                is_register_user = true;
                break;
              }
            }
          }
          
          if (!is_register_user) {
            // Set regular user handling
            auto user_it = op_to_final_tile.find(user);
            if (user_it != op_to_final_tile.end()) {
              int dst_x = user_it->second.first;
              int dst_y = user_it->second.second;
              inst_obj["dst_direction"] = calculateSendDirection(x, y, dst_x, dst_y).str();
              inst_obj["dst_tile"] = "(" + std::to_string(dst_x) + "," + std::to_string(dst_y) + ")";
              found_dst = true;
              break;
            }
          }
        } else {
          // Set fallback for operations without mapping_locs
          auto user_it = op_to_final_tile.find(user);
          if (user_it != op_to_final_tile.end()) {
            int dst_x = user_it->second.first;
            int dst_y = user_it->second.second;
            inst_obj["dst_direction"] = calculateSendDirection(x, y, dst_x, dst_y).str();
            inst_obj["dst_tile"] = "(" + std::to_string(dst_x) + "," + std::to_string(dst_y) + ")";
            found_dst = true;
            break;
          }
        }
      }
      if (found_dst) break;
    }
    
    // Set if no destination tile is found, set to Local.
    if (!found_dst) {
      inst_obj["dst_direction"] = "Local";
      inst_obj["dst_tile"] = "(" + std::to_string(x) + "," + std::to_string(y) + ")";
    }
  }

  // Set helper function to process tile instruction.
  void processTileInstruction(mlir::Operation* op, mlir::DictionaryAttr loc,
                            const std::map<Operation*, std::pair<int, int>>& op_to_final_tile,
                            std::map<std::pair<int, int>, YAMLNode>& tile_instructions) {
    auto x_attr = loc.getAs<IntegerAttr>("x");
    auto y_attr = loc.getAs<IntegerAttr>("y");
    auto timestep_attr = mlir::dyn_cast<IntegerAttr>(loc.get("time_step"));
    
    if (x_attr && y_attr) {
      int x = x_attr.getInt();
      int y = y_attr.getInt();
      int time_step = timestep_attr ? timestep_attr.getInt() : -1;
      
      // Set create instruction object.
      YAMLNode inst_obj = createInstructionObject(op);
      inst_obj.set("time_step", time_step);
      inst_obj.set("dst_tile", "(" + std::to_string(x) + "," + std::to_string(y) + ")");
      
      // Set calculate directions.
      calculateSourceDirection(op, x, y, inst_obj);
      calculateDestinationDirection(op, x, y, op_to_final_tile, inst_obj);
      
      // Set special handling for return operations.
      llvm::StringRef fullOpName = op->getName().getStringRef();
      llvm::StringRef opcode = fullOpName;
      size_t dotPos = fullOpName.find_last_of('.');
      if (dotPos != llvm::StringRef::npos && dotPos + 1 < fullOpName.size()) {
        opcode = fullOpName.substr(dotPos + 1);
      }
      if (opcode == "return") {
        // Set return operations don't have destination direction.
        inst_obj.erase("dst_direction");
        inst_obj.erase("dst_tile");
      }
      
      // Set add instruction to corresponding tile.
      tile_instructions[std::make_pair(x, y)].push_back(std::move(inst_obj));
    }
  }

  // Set helper function to process link instruction.
  void processLinkInstruction(mlir::Operation* op, mlir::DictionaryAttr loc,
                            const std::map<Operation*, std::pair<int, int>>& op_to_final_tile,
                            std::map<std::pair<int, int>, YAMLNode>& tile_instructions) {
    auto id_attr = mlir::dyn_cast<IntegerAttr>(loc.get("id"));
    auto timestep_attr = mlir::dyn_cast<IntegerAttr>(loc.get("time_step"));
    
    if (!id_attr) return;
    
    int time_step = timestep_attr ? timestep_attr.getInt() : -1;
    
    // Set try to find the destination tile by looking at subsequent operations.
    std::pair<int, int> dst_tile = std::make_pair(-1, -1);
    
    if (op->getNumResults() > 0) {
      for (mlir::Operation *user : op->getResults()[0].getUsers()) {
        if (!user) continue;
        auto user_it = op_to_final_tile.find(user);
        if (user_it != op_to_final_tile.end()) {
          dst_tile = user_it->second;
          break;
        }
      }
    }
    
    // Set if we found a destination, create the instruction.
    if (dst_tile.first != -1 && dst_tile.second != -1) {
      YAMLNode inst_obj = createInstructionObject(op);
      inst_obj.set("time_step", time_step);
      inst_obj.set("dst_tile", "(" + std::to_string(dst_tile.first) + "," + std::to_string(dst_tile.second) + ")");
      
      // Set find source tile from operands.
      bool found_src = false;
      for (mlir::Value operand : op->getOperands()) {
        if (!operand) continue;
        if (auto defining_op = operand.getDefiningOp()) {
          auto it = op_to_final_tile.find(defining_op);
          if (it != op_to_final_tile.end()) {
            int src_x = it->second.first;
            int src_y = it->second.second;
            inst_obj["src_direction"] = calculateRecvDirection(src_x, src_y, dst_tile.first, dst_tile.second).str();
            inst_obj["src_tile"] = "(" + std::to_string(src_x) + "," + std::to_string(src_y) + ")";
            found_src = true;
            break;
          }
        }
      }
      
      if (!found_src) {
        inst_obj["src_direction"] = "Local";
        inst_obj["src_tile"] = "(" + std::to_string(dst_tile.first) + "," + std::to_string(dst_tile.second) + ")";
      }
      
      // Set calculate destination direction.
      calculateDestinationDirection(op, dst_tile.first, dst_tile.second, op_to_final_tile, inst_obj);
      
      // Set add instruction to corresponding tile.
      tile_instructions[dst_tile].push_back(std::move(inst_obj));
    }
  }

  // Set helper function to process function and generate YAML.
  void processFunction(mlir::func::FuncOp func, YAMLNode& functions_array) {
    YAMLNode func_obj;
    func_obj.set("func_name", func.getName().str());

    if (auto ii_attr = func->getAttrOfType<IntegerAttr>("CompiledII"))
      func_obj["CompiledII"] = ii_attr.getInt();
    if (auto recMII_attr = func->getAttrOfType<IntegerAttr>("RecMII"))
      func_obj["RecMII"] = recMII_attr.getInt();
    if (auto resMII_attr = func->getAttrOfType<IntegerAttr>("ResMII"))
      func_obj["ResMII"] = resMII_attr.getInt();

    // Set maps instructions organized by tile.
    std::map<std::pair<int, int>, YAMLNode> tile_instructions;
    
    // Set maps operation to its mapping location, for calculating data movement direction.
    std::map<Operation*, std::pair<int, int>> op_to_tile;
    
    // Set maps operation to its final tile location (after data movement).
    std::map<Operation*, std::pair<int, int>> op_to_final_tile;
    
    // Set maps operation to its original source location (for data flow tracking).
    std::map<Operation*, std::pair<int, int>> op_to_source_tile;
    
    // Set maps tile coordinates to ID.
    std::map<std::pair<int, int>, int> tile_coord_to_id;
    int tile_id_counter = 0;

    // Set first pass: collects all operation tile locations and assigns tile ID.
    collectOperationMappings(func, op_to_tile, op_to_final_tile, op_to_source_tile, 
                           tile_coord_to_id, tile_id_counter);

    // Set second pass: generates instructions and calculates direction.
    func.walk([&](Operation *op) {
      auto attr_array = op->getAttrOfType<ArrayAttr>("mapping_locs");
      if (!attr_array || attr_array.size() == 0) {
        return;
      }
      
      bool found = false;
      for (mlir::Attribute attr : attr_array) {
        if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
          auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
          if (!resource_attr) continue;
          
          if (resource_attr.getValue() == "tile") {
            processTileInstruction(op, loc, op_to_final_tile, tile_instructions);
            found = true;
            break;
          } else if (resource_attr.getValue() == "link") {
            processLinkInstruction(op, loc, op_to_final_tile, tile_instructions);
            found = true;
            break;
          } else if (resource_attr.getValue() == "register") {
            // Set register operations don't generate instructions directly
            // Set they are handled in data flow tracing
            found = true;
            break;
          }
        }
      }
      if (!found) return;
    });

    // Set organizes instructions by tile and generates YAML.
    YAMLNode tile_instructions_obj;
    for (std::pair<const std::pair<int, int>, YAMLNode>& tile_pair : tile_instructions) {
      int x = tile_pair.first.first;
      int y = tile_pair.first.second;
      YAMLNode& instructions = tile_pair.second;
      
      // Set gets tile ID.
      auto coord = std::make_pair(x, y);
      int tile_id = tile_coord_to_id[coord];
      
      YAMLNode tile_obj;
      tile_obj.set("id", tile_id);
      tile_obj.set("x", x);
      tile_obj.set("y", y);
      // Set adds instructions at the end, to ensure order.
      tile_obj.set("instructions", instructions);
      
      // Set uses "Tile(id)" as key.
      std::string tile_key = "Tile(" + std::to_string(tile_id) + ")";
      tile_instructions_obj[tile_key] = std::move(tile_obj);
    }

    func_obj["tile_instructions"] = std::move(tile_instructions_obj);
    functions_array.push_back(std::move(func_obj));
  }

  // Set helper function to create instruction object.
  YAMLNode createInstructionObject(Operation *op) {
    YAMLNode inst_obj;
    
    // Set extract operation name.
    llvm::StringRef fullOpName = op->getName().getStringRef();
    llvm::StringRef opcode = fullOpName;
    size_t dotPos = fullOpName.find_last_of('.');
    if (dotPos != llvm::StringRef::npos && dotPos + 1 < fullOpName.size()) {
      opcode = fullOpName.substr(dotPos + 1);
    }

    // Set name, operands, result_types.
    inst_obj["name"] = fullOpName.str();
    
    // Set operands.
    YAMLNode operands_array = YAMLNode::Sequence();
    for (mlir::Value operand : op->getOperands()) {
      if (auto defining_op = operand.getDefiningOp()) {
        operands_array.push_back(YAMLNode::Scalar(defining_op->getName().getStringRef().str()));
      } else {
        operands_array.push_back(YAMLNode::Scalar("block_arg"));
      }
    }
    inst_obj.set("operands", operands_array);
    
    // Set result_types.
    YAMLNode result_types_array = YAMLNode::Sequence();
    for (mlir::Value result : op->getResults()) {
      std::string type_str;
      llvm::raw_string_ostream os(type_str);
      result.getType().print(os);
      result_types_array.push_back(YAMLNode::Scalar(os.str()));
    }
    inst_obj.set("result_types", result_types_array);
    
    // Set time_step only (remove opcode).
    inst_obj["time_step"] = -1;

    // Set handles constant value.
    if (opcode == "constant") {
      if (op->getNumOperands() == 0 && op->getNumResults() == 1) {
        if (auto value_attr = op->getAttr("value")) {
          if (auto int_attr = mlir::dyn_cast<IntegerAttr>(value_attr)) {
            inst_obj["constant_value"] = std::to_string(int_attr.getInt());
          } else if (auto float_attr = mlir::dyn_cast<FloatAttr>(value_attr)) {
            inst_obj["constant_value"] = std::to_string(float_attr.getValueAsDouble());
          }
        }
      }
    }

    return inst_obj;
  }

  // Set helper function to handle return operation ASM.
  void handleReturnOperationASM(mlir::Operation* op, int x, int y, llvm::raw_fd_ostream& asm_out) {
    asm_out << "            RETURN";
    
    // Set handle operands (input directions).
    for (mlir::Value operand : op->getOperands()) {
      asm_out << ", [";
      if (auto defining_op = operand.getDefiningOp()) {
        // Set find the source PE by looking at the defining operation's tile mapping.
        std::pair<int, int> src_coord = std::make_pair(x, y); // Default to current location.
        
        auto def_attr_array = defining_op->getAttrOfType<ArrayAttr>("mapping_locs");
        if (def_attr_array) {
          for (mlir::Attribute attr : def_attr_array) {
            if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
              auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
              if (resource_attr && resource_attr.getValue() == "tile") {
                auto x_attr = loc.getAs<IntegerAttr>("x");
                auto y_attr = loc.getAs<IntegerAttr>("y");
                if (x_attr && y_attr) {
                  src_coord = std::make_pair(x_attr.getInt(), y_attr.getInt());
                  break;
                }
              }
            }
          }
        }
        
        std::string direction = calculateRecvDirection(src_coord.first, src_coord.second, x, y).str();
        asm_out << direction << ", R";
      } else {
        asm_out << "Local, R";
      }
      asm_out << "]";
    }
    
    // Set return operations don't have destination direction.
    asm_out << "\n";
    asm_out << "            NOP\n";
  }

  // Set helper function to handle constant operation ASM.
  void handleConstantOperationASM(mlir::Operation* op, int x, int y,
                                 const std::map<Operation*, std::pair<int, int>>& asm_op_to_final_tile,
                                 llvm::raw_fd_ostream& asm_out) {
    // Set for constants, output the value.
    if (auto value_attr = op->getAttr("value")) {
      if (auto int_attr = mlir::dyn_cast<IntegerAttr>(value_attr)) {
        asm_out << "            CONSTANT, IMM[" << int_attr.getInt() << "] -> ";
      } else if (auto float_attr = mlir::dyn_cast<FloatAttr>(value_attr)) {
        asm_out << "            CONSTANT, IMM[" << float_attr.getValueAsDouble() << "] -> ";
      }
    }
    
    // Set find destination direction.
    std::set<std::string> dst_directions;
    for (mlir::Value result : op->getResults()) {
      if (!result) continue;
      for (mlir::Operation *user : result.getUsers()) {
        if (!user) continue;
        auto user_it = asm_op_to_final_tile.find(user);
        if (user_it != asm_op_to_final_tile.end()) {
          int dst_x = user_it->second.first;
          int dst_y = user_it->second.second;
          std::string dst_direction = calculateSendDirection(x, y, dst_x, dst_y).str();
          dst_directions.insert(dst_direction);
        }
      }
    }
    
    if (dst_directions.empty()) {
      asm_out << "[Local, R]\n";
    } else {
      bool first = true;
      for (const std::string& direction : dst_directions) {
        if (!first) asm_out << ", ";
        asm_out << "[" << direction << ", R]";
        first = false;
      }
      asm_out << "\n";
    }
  }

  // Set helper function to handle regular operation ASM.
  void handleRegularOperationASM(mlir::Operation* op, int x, int y,
                                const std::map<Operation*, std::pair<int, int>>& asm_op_to_final_tile,
                                const std::map<Operation*, std::pair<int, int>>& asm_op_to_source_tile,
                                const std::string& upper_opcode, llvm::raw_fd_ostream& asm_out) {
    // Set for other operations, handle operands and results.
    asm_out << "            " << upper_opcode;
    
    // Set handle operands (input sources).
    for (mlir::Value operand : op->getOperands()) {
      asm_out << ", [";
      if (auto defining_op = operand.getDefiningOp()) {
        // Set try to find source through asm_op_to_source_tile first
        auto src_it = asm_op_to_source_tile.find(defining_op);
        if (src_it != asm_op_to_source_tile.end()) {
          int src_x = src_it->second.first;
          int src_y = src_it->second.second;
          std::string direction = calculateRecvDirection(src_x, src_y, x, y).str();
          asm_out << direction << ", R";
        } else {
          // Set fallback: try to find source through direct tile mapping or register operations.
          auto def_attr_array = defining_op->getAttrOfType<ArrayAttr>("mapping_locs");
          if (def_attr_array) {
            bool found_src = false;
            for (mlir::Attribute attr : def_attr_array) {
              if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
                auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
                if (resource_attr && resource_attr.getValue() == "tile") {
                  auto x_attr = loc.getAs<IntegerAttr>("x");
                  auto y_attr = loc.getAs<IntegerAttr>("y");
                  if (x_attr && y_attr) {
                    int src_x = x_attr.getInt();
                    int src_y = y_attr.getInt();
                    std::string direction = calculateRecvDirection(src_x, src_y, x, y).str();
                    asm_out << direction << ", R";
                    found_src = true;
                    break;
                  }
                } else if (resource_attr && resource_attr.getValue() == "register") {
                  // Set this is a register operation, use register format
                  auto register_id_attr = mlir::dyn_cast<IntegerAttr>(loc.get("id"));
                  if (register_id_attr) {
                    int register_id = register_id_attr.getInt();
                    asm_out << "$" << register_id;
                    found_src = true;
                  }
                  if (found_src) break;
                }
              }
            }
            if (!found_src) {
              asm_out << "Local, R";
            }
          } else {
            asm_out << "Local, R";
          }
        }
      } else {
        asm_out << "Local, R";
      }
      asm_out << "]";
    }
    
    // Set handle result (output destinations).
    asm_out << " -> ";
    std::set<std::string> dst_directions;
    std::set<std::string> dst_registers;
    
    // Set collect all destination directions and registers
    for (mlir::Value result : op->getResults()) {
      if (!result) continue;
      for (mlir::Operation *user : result.getUsers()) {
        if (!user) continue;
        
        // Set check if user is a register operation
        auto user_attr_array = user->getAttrOfType<ArrayAttr>("mapping_locs");
        if (user_attr_array) {
          bool is_register_user = false;
          for (mlir::Attribute user_attr : user_attr_array) {
            if (auto user_loc = mlir::dyn_cast<DictionaryAttr>(user_attr)) {
              auto user_resource_attr = mlir::dyn_cast<StringAttr>(user_loc.get("resource"));
              if (user_resource_attr && user_resource_attr.getValue() == "register") {
                is_register_user = true;
                // Set for register operations, add register as destination
                auto register_id_attr = mlir::dyn_cast<IntegerAttr>(user_loc.get("id"));
                if (register_id_attr) {
                  int register_id = register_id_attr.getInt();
                  dst_registers.insert("$" + std::to_string(register_id));
                }
                break;
              }
            }
          }
          
          if (!is_register_user) {
            // Set regular user handling
            auto user_it = asm_op_to_final_tile.find(user);
            if (user_it != asm_op_to_final_tile.end()) {
              int dst_x = user_it->second.first;
              int dst_y = user_it->second.second;
              std::string dst_direction = calculateSendDirection(x, y, dst_x, dst_y).str();
              dst_directions.insert(dst_direction);
            }
          }
        } else {
          // Set fallback for operations without mapping_locs
          auto user_it = asm_op_to_final_tile.find(user);
          if (user_it != asm_op_to_final_tile.end()) {
            int dst_x = user_it->second.first;
            int dst_y = user_it->second.second;
            std::string dst_direction = calculateSendDirection(x, y, dst_x, dst_y).str();
            dst_directions.insert(dst_direction);
          }
        }
      }
    }
    
    // Set if no destinations found through users, try asm_op_to_final_tile
    if (dst_directions.empty() && dst_registers.empty()) {
      auto op_dst_it = asm_op_to_final_tile.find(op);
      if (op_dst_it != asm_op_to_final_tile.end()) {
        int dst_x = op_dst_it->second.first;
        int dst_y = op_dst_it->second.second;
        std::string dst_direction = calculateSendDirection(x, y, dst_x, dst_y).str();
        dst_directions.insert(dst_direction);
      } else {
        dst_directions.insert("Local");
      }
    }
    
    // Set output all destination directions and registers
    bool first = true;
    for (const std::string& direction : dst_directions) {
      if (!first) asm_out << ", ";
      asm_out << "[" << direction << ", R]";
      first = false;
    }
    for (const std::string& register_dst : dst_registers) {
      if (!first) asm_out << ", ";
      asm_out << "[" << register_dst << "]";
      first = false;
    }
    asm_out << "\n";
  }

  // Set helper function to generate ASM for a single operation.
  void generateASMForOperation(mlir::Operation* op, int x, int y,
                              const std::map<Operation*, std::pair<int, int>>& asm_op_to_final_tile,
                              const std::map<Operation*, std::pair<int, int>>& asm_op_to_source_tile,
                              llvm::raw_fd_ostream& asm_out) {
    asm_out << "        {\n";
    
    // Set extract operation name.
    llvm::StringRef fullOpName = op->getName().getStringRef();
    llvm::StringRef opcode = fullOpName;
    size_t dotPos = fullOpName.find_last_of('.');
    if (dotPos != llvm::StringRef::npos && dotPos + 1 < fullOpName.size()) {
      opcode = fullOpName.substr(dotPos + 1);
    }
    
    // Set convert opcode to uppercase.
    std::string upper_opcode = opcode.str();
    std::transform(upper_opcode.begin(), upper_opcode.end(), upper_opcode.begin(), ::toupper);
    
    // Set handle different operation types.
    if (upper_opcode == "RETURN") {
      handleReturnOperationASM(op, x, y, asm_out);
    } else if (upper_opcode == "CONSTANT") {
      handleConstantOperationASM(op, x, y, asm_op_to_final_tile, asm_out);
    } else {
      handleRegularOperationASM(op, x, y, asm_op_to_final_tile, asm_op_to_source_tile, upper_opcode, asm_out);
    }
    
    asm_out << "        }\n";
  }

  // Set helper function to rebuild ASM mappings using thorough tracing.
  void rebuildASMMappings(mlir::func::FuncOp func,
                         std::map<Operation*, std::pair<int, int>>& asm_op_to_final_tile,
                         std::map<Operation*, std::pair<int, int>>& asm_op_to_source_tile) {
    // Set first pass: collect all operations and build value-to-operation mapping
    std::vector<std::pair<mlir::Value, mlir::Operation*>> value_source_pairs;
    func.walk([&](Operation *op) {
      // Set collect all value-to-operation pairs (for data flow tracing)
      for (mlir::Value result : op->getResults()) {
        value_source_pairs.push_back(std::make_pair(result, op));
      }
      
      // Set collect tile operations (only assign coordinates to operations with mapping_locs)
      auto attr_array = op->getAttrOfType<ArrayAttr>("mapping_locs");
      if (attr_array && attr_array.size() > 0) {
        for (mlir::Attribute attr : attr_array) {
          if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
            auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
            if (resource_attr && resource_attr.getValue() == "tile") {
              auto x_attr = loc.getAs<IntegerAttr>("x");
              auto y_attr = loc.getAs<IntegerAttr>("y");
              if (x_attr && y_attr) {
                int x = x_attr.getInt();
                int y = y_attr.getInt();
                asm_op_to_final_tile[op] = std::make_pair(x, y);
                asm_op_to_source_tile[op] = std::make_pair(x, y);
                break;
              }
            }
          }
        }
      }
      // Set note: operations without mapping_locs are still included in value_source_pairs
      // Set for data flow tracing, but they don't get direct tile assignments
    });
    
    // Set helper function to find source tile through thorough tracing
    std::function<std::pair<int, int>(mlir::Operation*, std::set<mlir::Operation*>&)> findSourceTile = 
      [&](mlir::Operation* op, std::set<mlir::Operation*>& visited) -> std::pair<int, int> {
        // Set prevent infinite recursion
        if (visited.find(op) != visited.end()) {
          return std::make_pair(-1, -1);
        }
        visited.insert(op);
        
        // Set check if this operation has a direct tile mapping
        auto attr_array = op->getAttrOfType<ArrayAttr>("mapping_locs");
        if (attr_array) {
          for (mlir::Attribute attr : attr_array) {
            if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
              auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
              if (resource_attr && resource_attr.getValue() == "tile") {
                auto x_attr = loc.getAs<IntegerAttr>("x");
                auto y_attr = loc.getAs<IntegerAttr>("y");
                if (x_attr && y_attr) {
                  return std::make_pair(x_attr.getInt(), y_attr.getInt());
                }
              }
            }
          }
        }
        
        // Set recursively trace through operands
        for (mlir::Value operand : op->getOperands()) {
          if (!operand) continue;
          for (const auto& pair : value_source_pairs) {
            if (pair.first == operand) {
              mlir::Operation* source_op = pair.second;
              auto source_tile = findSourceTile(source_op, visited);
              if (source_tile.first != -1 && source_tile.second != -1) {
                return source_tile;
              }
              break;
            }
          }
        }
        
        return std::make_pair(-1, -1); // No source found
      };
    
    // Set helper function to find destination tile through thorough tracing
    std::function<std::pair<int, int>(mlir::Operation*, std::set<mlir::Operation*>&)> findDestinationTile = 
      [&](mlir::Operation* op, std::set<mlir::Operation*>& visited) -> std::pair<int, int> {
        // Set prevent infinite recursion
        if (visited.find(op) != visited.end()) {
          return std::make_pair(-1, -1);
        }
        visited.insert(op);
        
        // Set check if this operation has a direct tile mapping
        auto attr_array = op->getAttrOfType<ArrayAttr>("mapping_locs");
        if (attr_array) {
          for (mlir::Attribute attr : attr_array) {
            if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
              auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
              if (resource_attr && resource_attr.getValue() == "tile") {
                auto x_attr = loc.getAs<IntegerAttr>("x");
                auto y_attr = loc.getAs<IntegerAttr>("y");
                if (x_attr && y_attr) {
                  return std::make_pair(x_attr.getInt(), y_attr.getInt());
                }
              }
            }
          }
        }
        
        // Set recursively trace through users
        for (mlir::Value result : op->getResults()) {
          if (!result) continue;
          for (mlir::Operation *user : result.getUsers()) {
            if (!user) continue;
            auto dest_tile = findDestinationTile(user, visited);
            if (dest_tile.first != -1 && dest_tile.second != -1) {
              return dest_tile;
            }
          }
        }
        
        return std::make_pair(-1, -1); // No destination found
      };
    
    // Set second pass: find source tiles for all operations
    func.walk([&](Operation *op) {
      if (asm_op_to_source_tile.find(op) == asm_op_to_source_tile.end()) {
        std::set<mlir::Operation*> visited;
        auto source_tile = findSourceTile(op, visited);
        if (source_tile.first != -1 && source_tile.second != -1) {
          asm_op_to_source_tile[op] = source_tile;
        }
      }
    });
    
    // Set third pass: find destination tiles for all operations
    func.walk([&](Operation *op) {
      if (asm_op_to_final_tile.find(op) == asm_op_to_final_tile.end()) {
        std::set<mlir::Operation*> visited;
        auto dest_tile = findDestinationTile(op, visited);
        if (dest_tile.first != -1 && dest_tile.second != -1) {
          asm_op_to_final_tile[op] = dest_tile;
        }
      }
    });
  }

  // Set helper function to group operations by PE and time step.
  void groupOperationsByPEAndTime(mlir::func::FuncOp func,
                                 std::map<std::pair<int, int>, std::map<int, std::vector<Operation*>>>& pe_time_ops) {
    func.walk([&](Operation *op) {
      auto attr_array = op->getAttrOfType<ArrayAttr>("mapping_locs");
      if (!attr_array || attr_array.size() == 0) {
        return; // Skip operations without mapping_locs for PE assignment
      }
      bool found = false;
      for (mlir::Attribute attr : attr_array) {
        if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
          auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
          auto timestep_attr = mlir::dyn_cast<IntegerAttr>(loc.get("time_step"));
          
          if (resource_attr && resource_attr.getValue() == "tile") {
            auto x_attr = loc.getAs<IntegerAttr>("x");
            auto y_attr = loc.getAs<IntegerAttr>("y");
            
            if (x_attr && y_attr && timestep_attr) {
              int x = x_attr.getInt();
              int y = y_attr.getInt();
              int time_step = timestep_attr.getInt();
              pe_time_ops[std::make_pair(x, y)][time_step].push_back(op);
              found = true;
              break;
            }
          }
          // Set link operations are not assigned to PEs, they are handled separately for data flow tracking
        }
      }
      if (!found) return;
    });
  }

  // Set helper function to collect input directions for a PE.
  void collectInputDirections(mlir::func::FuncOp func, int x, int y,
                             const std::map<int, std::vector<Operation*>>& time_ops,
                             std::set<std::string>& input_directions) {
    for (const std::pair<const int, std::vector<Operation*>>& time_pair : time_ops) {
      for (mlir::Operation* op : time_pair.second) {
        // Set find source directions for this operation.
        for (mlir::Value operand : op->getOperands()) {
          if (!operand) continue;
          if (auto defining_op = operand.getDefiningOp()) {
            // Set find the source tile by looking at the defining operation's tile mapping.
            std::pair<int, int> src_coord = std::make_pair(x, y); // Default to current location.
            
            auto def_attr_array = defining_op->getAttrOfType<ArrayAttr>("mapping_locs");
            if (def_attr_array) {
              for (mlir::Attribute attr : def_attr_array) {
                if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
                  auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
                  if (resource_attr && resource_attr.getValue() == "tile") {
                    auto x_attr = loc.getAs<IntegerAttr>("x");
                    auto y_attr = loc.getAs<IntegerAttr>("y");
                    if (x_attr && y_attr) {
                      src_coord = std::make_pair(x_attr.getInt(), y_attr.getInt());
                      break;
                    }
                  }
                }
              }
            }
            
            std::string direction = calculateRecvDirection(src_coord.first, src_coord.second, x, y).str();
            if (direction != "Local") {
              input_directions.insert(direction);
            }
          }
        }
      }
    }
  }

  // Set helper function to collect input directions from link operations.
  void collectLinkInputDirections(mlir::func::FuncOp func, int x, int y,
                                 std::set<std::string>& input_directions) {
    func.walk([&](Operation *op) {
      auto attr_array = op->getAttrOfType<ArrayAttr>("mapping_locs");
      if (!attr_array || attr_array.size() == 0) {
        return;
      }
      bool found = false;
      for (mlir::Attribute attr : attr_array) {
        if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
          auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
          if (resource_attr && resource_attr.getValue() == "link") {
            // Set check if this link operation targets the current tile.
            for (mlir::Value result : op->getResults()) {
              if (!result) continue;
              for (mlir::Operation *user : result.getUsers()) {
                if (!user) continue;
                auto user_attr_array = user->getAttrOfType<ArrayAttr>("mapping_locs");
                if (!user_attr_array) continue;
                for (mlir::Attribute user_attr : user_attr_array) {
                  if (auto user_loc = mlir::dyn_cast<DictionaryAttr>(user_attr)) {
                    auto user_resource_attr = mlir::dyn_cast<StringAttr>(user_loc.get("resource"));
                    if (user_resource_attr && user_resource_attr.getValue() == "tile") {
                      auto user_x_attr = user_loc.getAs<IntegerAttr>("x");
                      auto user_y_attr = user_loc.getAs<IntegerAttr>("y");
                      if (user_x_attr && user_y_attr && user_x_attr.getInt() == x && user_y_attr.getInt() == y) {
                        // Set this link operation targets the current tile, find its source.
                        for (mlir::Value operand : op->getOperands()) {
                          if (!operand) continue;
                          if (auto defining_op = operand.getDefiningOp()) {
                            auto def_attr_array = defining_op->getAttrOfType<ArrayAttr>("mapping_locs");
                            if (def_attr_array) {
                              for (mlir::Attribute def_attr : def_attr_array) {
                                if (auto def_loc = mlir::dyn_cast<DictionaryAttr>(def_attr)) {
                                  auto def_resource_attr = mlir::dyn_cast<StringAttr>(def_loc.get("resource"));
                                  if (def_resource_attr && def_resource_attr.getValue() == "tile") {
                                    auto def_x_attr = def_loc.getAs<IntegerAttr>("x");
                                    auto def_y_attr = def_loc.getAs<IntegerAttr>("y");
                                    if (def_x_attr && def_y_attr) {
                                      int src_x = def_x_attr.getInt();
                                      int src_y = def_y_attr.getInt();
                                      std::string direction = calculateRecvDirection(src_x, src_y, x, y).str();
                                      if (direction != "Local") {
                                        input_directions.insert(direction);
                                      }
                                      break;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      if (found) return;
    });
  }

  // Set helper function to generate ASM for a single PE.
  void generateASMForPE(int x, int y, const std::map<int, std::vector<Operation*>>& time_ops,
                       mlir::func::FuncOp func,
                       const std::map<Operation*, std::pair<int, int>>& asm_op_to_final_tile,
                       const std::map<Operation*, std::pair<int, int>>& asm_op_to_source_tile,
                       llvm::raw_fd_ostream& asm_out) {
    asm_out << "PE(" << x << "," << y << "):\n";
    asm_out << "{\n";
    
    // Set generate Entry conditions based on input directions.
    std::set<std::string> input_directions;
    collectInputDirections(func, x, y, time_ops, input_directions);
    collectLinkInputDirections(func, x, y, input_directions);
    
    // Set write Entry line.
    // asm_out << "    Entry ";
    // bool first = true;
    // for (const std::string& dir : input_directions) {
    //   if (!first) asm_out << ", ";
    //   asm_out << "[" << dir << ", R]";
    //   first = false;
    // }
    // if (input_directions.empty()) {
    //   asm_out << "[]";
    // }
    // asm_out << " => ";
    asm_out << "    Entry => ";
    // Set determine if it's Loop or Once based on time steps.
    if (time_ops.size() > 1) {
      asm_out << "Loop {\n";
    } else {
      asm_out << "Once {\n";
    }
    
    // Set generate operations for each time step.
    for (const std::pair<const int, std::vector<Operation*>>& time_pair : time_ops) {
      const std::vector<Operation*>& ops = time_pair.second;
      
      for (mlir::Operation* op : ops) {
        generateASMForOperation(op, x, y, asm_op_to_final_tile, asm_op_to_source_tile, asm_out);
      }
    }
    
    asm_out << "    }\n";
    asm_out << "}\n\n";
  }

  // Set helper function to generate ASM output.
  void generateASMOutput(ModuleOp module) {
    std::error_code asm_ec;
    llvm::raw_fd_ostream asm_out("generated-instructions.asm", asm_ec);
    if (asm_ec) {
        getOperation()->emitError("Failed to open 'generated-instructions.asm' for writing: " + asm_ec.message());
        return signalPassFailure();
    }
    
    // Set generate ASM for each function.
    for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
      auto accel_attr = func->getAttrOfType<StringAttr>("accelerator");
      if (!accel_attr || accel_attr.getValue() != "neura")
        continue;

      // Set rebuild op_to_final_tile mapping for ASM generation.
      std::map<Operation*, std::pair<int, int>> asm_op_to_final_tile;
      std::map<Operation*, std::pair<int, int>> asm_op_to_source_tile;
      rebuildASMMappings(func, asm_op_to_final_tile, asm_op_to_source_tile);
      
      // Set debug: print the size of mappings
      llvm::errs() << "DEBUG: asm_op_to_final_tile size: " << asm_op_to_final_tile.size() << "\n";
      llvm::errs() << "DEBUG: asm_op_to_source_tile size: " << asm_op_to_source_tile.size() << "\n";

      // Set group operations by PE and time step.
      std::map<std::pair<int, int>, std::map<int, std::vector<Operation*>>> pe_time_ops;
      groupOperationsByPEAndTime(func, pe_time_ops);
      
      // Set generate ASM for each PE.
      for (std::pair<const std::pair<int, int>, std::map<int, std::vector<Operation*>>>& pe_pair : pe_time_ops) {
        int x = pe_pair.first.first;
        int y = pe_pair.first.second;
        std::map<int, std::vector<Operation*>>& time_ops = pe_pair.second;
        
        generateASMForPE(x, y, time_ops, func, asm_op_to_final_tile, asm_op_to_source_tile, asm_out);
      }
    }
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    YAMLNode functions_array;

    for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
      auto accel_attr = func->getAttrOfType<StringAttr>("accelerator");
      if (!accel_attr || accel_attr.getValue() != "neura") {
        continue;
      }

      processFunction(func, functions_array);
    }

    // Set generate both YAML and ASM output.
    // Set YAML output.
    YAMLNode root;
    root.set("functions", functions_array);

    std::error_code ec;
    llvm::raw_fd_ostream yaml_out("generated-instructions.yaml", ec);
    if (ec) {
        getOperation()->emitError("Failed to open 'generated-instructions.yaml' for writing: " + ec.message());
        return signalPassFailure();
    }
    yaml_out << root.toString() << "\n";
    
    // Set generate ASM output.
    generateASMOutput(module);
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<mlir::Pass> createGenerateCodePass() {
  return std::make_unique<GenerateCodePass>();
}
} // namespace mlir::neura


