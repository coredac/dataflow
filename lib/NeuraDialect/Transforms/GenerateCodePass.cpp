#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/JSON.h"
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
    return "Generates JSON code from mapped Neura IR.";
  }

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
          }
        }
      }
    });
  }

  // Set helper function to calculate source direction for an operation.
  void calculateSourceDirection(mlir::Operation* op, int x, int y, llvm::json::Object& inst_obj) {
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
                                   llvm::json::Object& inst_obj) {
    bool found_dst = false;
    for (mlir::Value result : op->getResults()) {
      if (!result) continue;
      for (mlir::Operation *user : result.getUsers()) {
        if (!user) continue;
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
                            std::map<std::pair<int, int>, llvm::json::Array>& tile_instructions) {
    auto x_attr = loc.getAs<IntegerAttr>("x");
    auto y_attr = loc.getAs<IntegerAttr>("y");
    auto timestep_attr = mlir::dyn_cast<IntegerAttr>(loc.get("time_step"));
    
    if (x_attr && y_attr) {
      int x = x_attr.getInt();
      int y = y_attr.getInt();
      int time_step = timestep_attr ? timestep_attr.getInt() : -1;
      
      // Set create instruction object.
      llvm::json::Object inst_obj = createInstructionObject(op);
      inst_obj["time_step"] = time_step;
      inst_obj["dst_tile"] = "(" + std::to_string(x) + "," + std::to_string(y) + ")";
      
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
                            std::map<std::pair<int, int>, llvm::json::Array>& tile_instructions) {
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
      llvm::json::Object inst_obj = createInstructionObject(op);
      inst_obj["time_step"] = time_step;
      inst_obj["dst_tile"] = "(" + std::to_string(dst_tile.first) + "," + std::to_string(dst_tile.second) + ")";
      
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

  // Set helper function to process function and generate JSON.
  void processFunction(mlir::func::FuncOp func, llvm::json::Array& functions_array) {
    llvm::json::Object func_obj;
    func_obj["func_name"] = func.getName().str();

    if (auto ii_attr = func->getAttrOfType<IntegerAttr>("CompiledII"))
      func_obj["CompiledII"] = ii_attr.getInt();
    if (auto recMII_attr = func->getAttrOfType<IntegerAttr>("RecMII"))
      func_obj["RecMII"] = recMII_attr.getInt();
    if (auto resMII_attr = func->getAttrOfType<IntegerAttr>("ResMII"))
      func_obj["ResMII"] = resMII_attr.getInt();

    // Set maps instructions organized by tile.
    std::map<std::pair<int, int>, llvm::json::Array> tile_instructions;
    
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
          }
        }
      }
      if (!found) return;
    });

    // Set organizes instructions by tile and generates JSON.
    llvm::json::Object tile_instructions_obj;
    for (std::pair<const std::pair<int, int>, llvm::json::Array>& tile_pair : tile_instructions) {
      int x = tile_pair.first.first;
      int y = tile_pair.first.second;
      llvm::json::Array& instructions = tile_pair.second;
      
      // Set gets tile ID.
      auto coord = std::make_pair(x, y);
      int tile_id = tile_coord_to_id[coord];
      
      llvm::json::Object tile_obj;
      tile_obj["id"] = tile_id;
      tile_obj["x"] = x;
      tile_obj["y"] = y;
      // Set adds instructions at the end, to ensure order.
      tile_obj["instructions"] = std::move(instructions);
      
      // Set uses "Tile(id)" as key.
      std::string tile_key = "Tile(" + std::to_string(tile_id) + ")";
      tile_instructions_obj[tile_key] = std::move(tile_obj);
    }

    func_obj["tile_instructions"] = std::move(tile_instructions_obj);
    functions_array.push_back(std::move(func_obj));
  }

  // Set helper function to create instruction object.
  llvm::json::Object createInstructionObject(Operation *op) {
    llvm::json::Object inst_obj;
    
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
    llvm::json::Array operands_array;
    for (mlir::Value operand : op->getOperands()) {
      if (auto defining_op = operand.getDefiningOp()) {
        operands_array.push_back(defining_op->getName().getStringRef().str());
      } else {
        operands_array.push_back("block_arg");
      }
    }
    inst_obj["operands"] = std::move(operands_array);
    
    // Set result_types.
    llvm::json::Array result_types_array;
    for (mlir::Value result : op->getResults()) {
      std::string type_str;
      llvm::raw_string_ostream os(type_str);
      result.getType().print(os);
      result_types_array.push_back(os.str());
    }
    inst_obj["result_types"] = std::move(result_types_array);
    
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
    
    // Set handle operands (input directions).
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
          // Set fallback: try to find source through direct tile mapping or link operations.
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
                } else if (resource_attr && resource_attr.getValue() == "link") {
                  // Set this is a link operation, try to find its source through its operands
                  for (mlir::Value link_operand : defining_op->getOperands()) {
                    if (!link_operand) continue;
                    if (auto link_source_op = link_operand.getDefiningOp()) {
                      auto link_source_attr_array = link_source_op->getAttrOfType<ArrayAttr>("mapping_locs");
                      if (link_source_attr_array) {
                        for (mlir::Attribute link_source_attr : link_source_attr_array) {
                          if (auto link_source_loc = mlir::dyn_cast<DictionaryAttr>(link_source_attr)) {
                            auto link_source_resource_attr = mlir::dyn_cast<StringAttr>(link_source_loc.get("resource"));
                            if (link_source_resource_attr && link_source_resource_attr.getValue() == "tile") {
                              auto link_source_x_attr = link_source_loc.getAs<IntegerAttr>("x");
                              auto link_source_y_attr = link_source_loc.getAs<IntegerAttr>("y");
                              if (link_source_x_attr && link_source_y_attr) {
                                int src_x = link_source_x_attr.getInt();
                                int src_y = link_source_y_attr.getInt();
                                std::string direction = calculateRecvDirection(src_x, src_y, x, y).str();
                                
                                asm_out << direction << ", R";
                                
                                found_src = true;
                                break;
                              }
                            }
                          }
                        }
                      }
                      if (found_src) break;
                    }
                  }
                  if (found_src) break;
                }
              }
            }
                        if (!found_src) {
              // Set additional fallback: try to find source through operands even if no mapping_locs
              for (mlir::Value link_operand : defining_op->getOperands()) {
                if (!link_operand) continue;
                if (auto link_source_op = link_operand.getDefiningOp()) {
                  auto link_source_attr_array = link_source_op->getAttrOfType<ArrayAttr>("mapping_locs");
                  if (link_source_attr_array) {
                    for (mlir::Attribute link_source_attr : link_source_attr_array) {
                      if (auto link_source_loc = mlir::dyn_cast<DictionaryAttr>(link_source_attr)) {
                        auto link_source_resource_attr = mlir::dyn_cast<StringAttr>(link_source_loc.get("resource"));
                        if (link_source_resource_attr && link_source_resource_attr.getValue() == "tile") {
                          auto link_source_x_attr = link_source_loc.getAs<IntegerAttr>("x");
                          auto link_source_y_attr = link_source_loc.getAs<IntegerAttr>("y");
                          if (link_source_x_attr && link_source_y_attr) {
                            int src_x = link_source_x_attr.getInt();
                            int src_y = link_source_y_attr.getInt();
                            std::string direction = calculateRecvDirection(src_x, src_y, x, y).str();
                            asm_out << direction << ", R";
                            found_src = true;
                            break;
                          }
                        }
                      }
                    }
                  }
                  if (found_src) break;
                }
              }
            }
            if (!found_src) {
              // Set additional fallback: try to find source through operands even if no mapping_locs
              for (mlir::Value link_operand : defining_op->getOperands()) {
                if (!link_operand) continue;
                if (auto link_source_op = link_operand.getDefiningOp()) {
                  auto link_source_attr_array = link_source_op->getAttrOfType<ArrayAttr>("mapping_locs");
                  if (link_source_attr_array) {
                    for (mlir::Attribute link_source_attr : link_source_attr_array) {
                      if (auto link_source_loc = mlir::dyn_cast<DictionaryAttr>(link_source_attr)) {
                        auto link_source_resource_attr = mlir::dyn_cast<StringAttr>(link_source_loc.get("resource"));
                        if (link_source_resource_attr && link_source_resource_attr.getValue() == "tile") {
                          auto link_source_x_attr = link_source_loc.getAs<IntegerAttr>("x");
                          auto link_source_y_attr = link_source_loc.getAs<IntegerAttr>("y");
                          if (link_source_x_attr && link_source_y_attr) {
                            int src_x = link_source_x_attr.getInt();
                            int src_y = link_source_y_attr.getInt();
                            std::string direction = calculateRecvDirection(src_x, src_y, x, y).str();
                            asm_out << direction << ", R";
                            found_src = true;
                            break;
                          }
                        }
                      }
                    }
                  }
                  if (found_src) break;
                }
              }
            }
            if (!found_src) {
              // Set try to find source through asm_op_to_source_tile for this operation
              auto op_src_it = asm_op_to_source_tile.find(defining_op);
              if (op_src_it != asm_op_to_source_tile.end()) {
                int src_x = op_src_it->second.first;
                int src_y = op_src_it->second.second;
                std::string direction = calculateRecvDirection(src_x, src_y, x, y).str();
                
                asm_out << direction << ", R";
                              } else {
                  asm_out << "Local, R";
                }
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
    
    // Set handle result (output direction).
    asm_out << " -> ";
    std::set<std::string> dst_directions;
    
    // Set collect all destination directions
    for (mlir::Value result : op->getResults()) {
      if (!result) continue;
      for (mlir::Operation *user : result.getUsers()) {
        if (!user) continue;
        
        // Set special handling for ctrl_mov operations
        if (user->getName().getStringRef().contains("ctrl_mov")) {
          // Set ctrl_mov has link mapping, find its destination
          auto ctrl_attr_array = user->getAttrOfType<ArrayAttr>("mapping_locs");
          if (ctrl_attr_array) {
            for (mlir::Attribute ctrl_attr : ctrl_attr_array) {
              if (auto ctrl_loc = mlir::dyn_cast<DictionaryAttr>(ctrl_attr)) {
                auto ctrl_resource_attr = mlir::dyn_cast<StringAttr>(ctrl_loc.get("resource"));
                if (ctrl_resource_attr && ctrl_resource_attr.getValue() == "link") {
                  // Set this is a link operation, find where it goes
                  for (mlir::Value ctrl_result : user->getResults()) {
                    if (!ctrl_result) continue;
                    for (mlir::Operation *ctrl_user : ctrl_result.getUsers()) {
                      if (!ctrl_user) continue;
                      auto ctrl_user_it = asm_op_to_final_tile.find(ctrl_user);
                      if (ctrl_user_it != asm_op_to_final_tile.end()) {
                        int dst_x = ctrl_user_it->second.first;
                        int dst_y = ctrl_user_it->second.second;
                        std::string dst_direction = calculateSendDirection(x, y, dst_x, dst_y).str();
                        dst_directions.insert(dst_direction);
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          // Set regular user handling
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
    if (dst_directions.empty()) {
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
    
    // Set output all destination directions
    bool first = true;
    for (const std::string& direction : dst_directions) {
      if (!first) asm_out << ", ";
      asm_out << "[" << direction << ", R]";
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
    // Set first pass: collect all tile operations and build value-to-operation mapping
    std::vector<std::pair<mlir::Value, mlir::Operation*>> value_source_pairs;
    func.walk([&](Operation *op) {
      // Set collect all value-to-operation pairs
      for (mlir::Value result : op->getResults()) {
        value_source_pairs.push_back(std::make_pair(result, op));
      }
      
      // Set collect tile operations
      auto attr_array = op->getAttrOfType<ArrayAttr>("mapping_locs");
      if (!attr_array || attr_array.size() == 0) {
        return;
      }
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
    });
    
    // Set helper function to find source tile through thorough tracing
    std::function<std::pair<int, int>(mlir::Operation*)> findSourceTile = 
      [&](mlir::Operation* op) -> std::pair<int, int> {
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
              auto source_tile = findSourceTile(source_op);
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
    std::function<std::pair<int, int>(mlir::Operation*)> findDestinationTile = 
      [&](mlir::Operation* op) -> std::pair<int, int> {
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
            
            // Set check if user has a direct tile mapping
                  auto user_attr_array = user->getAttrOfType<ArrayAttr>("mapping_locs");
            if (user_attr_array) {
                  for (mlir::Attribute user_attr : user_attr_array) {
                    if (auto user_loc = mlir::dyn_cast<DictionaryAttr>(user_attr)) {
                      auto user_resource_attr = mlir::dyn_cast<StringAttr>(user_loc.get("resource"));
                      if (user_resource_attr && user_resource_attr.getValue() == "tile") {
                        auto user_x_attr = user_loc.getAs<IntegerAttr>("x");
                        auto user_y_attr = user_loc.getAs<IntegerAttr>("y");
                        if (user_x_attr && user_y_attr) {
                      return std::make_pair(user_x_attr.getInt(), user_y_attr.getInt());
                    }
                  }
                }
              }
            }
            
            // Set if user doesn't have direct tile mapping, recursively trace
            auto dest_tile = findDestinationTile(user);
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
        auto source_tile = findSourceTile(op);
        if (source_tile.first != -1 && source_tile.second != -1) {
          asm_op_to_source_tile[op] = source_tile;
        }
      }
    });
    
    // Set third pass: find destination tiles for all operations
    func.walk([&](Operation *op) {
            if (asm_op_to_final_tile.find(op) == asm_op_to_final_tile.end()) {
        auto dest_tile = findDestinationTile(op);
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
        return;
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

    llvm::json::Array functions_array;

    for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
      auto accel_attr = func->getAttrOfType<StringAttr>("accelerator");
      if (!accel_attr || accel_attr.getValue() != "neura") {
        continue;
      }

      processFunction(func, functions_array);
    }

    // Set generate both JSON and ASM output.
    // Set JSON output.
    llvm::json::Object root;
    root["functions"] = std::move(functions_array);

    std::error_code ec;
    llvm::raw_fd_ostream json_out("generated-instructions.json", ec);
    if (ec) {
        getOperation()->emitError("Failed to open 'generated-instructions.json' for writing: " + ec.message());
        return signalPassFailure();
    }
    json_out << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
    
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


