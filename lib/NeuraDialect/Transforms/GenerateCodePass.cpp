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

  void runOnOperation() override {
    ModuleOp module = getOperation();

    llvm::json::Array functions_array;

    for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
      auto accel_attr = func->getAttrOfType<StringAttr>("accelerator");
      if (!accel_attr || accel_attr.getValue() != "neura")
        continue;

      llvm::json::Object func_obj;
      func_obj["func_name"] = func.getName().str();

      if (auto ii_attr = func->getAttrOfType<IntegerAttr>("CompiledII"))
        func_obj["CompiledII"] = ii_attr.getInt();
      if (auto recMII_attr = func->getAttrOfType<IntegerAttr>("RecMII"))
        func_obj["RecMII"] = recMII_attr.getInt();
      if (auto resMII_attr = func->getAttrOfType<IntegerAttr>("ResMII"))
        func_obj["ResMII"] = resMII_attr.getInt();

      // Maps instructions organized by tile.
      std::map<std::pair<int, int>, llvm::json::Array> tile_instructions;
      
      // Maps operation to its mapping location, for calculating data movement direction.
      std::map<Operation*, std::pair<int, int>> op_to_tile;
      
      // Maps operation to its final tile location (after data movement).
      std::map<Operation*, std::pair<int, int>> op_to_final_tile;
      
      // Maps operation to its original source location (for data flow tracking).
      std::map<Operation*, std::pair<int, int>> op_to_source_tile;
      
      // Maps link ID to source and destination tiles.
      std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> link_to_tiles;
      
      // Maps tile coordinates to ID.
      std::map<std::pair<int, int>, int> tile_coord_to_id;
      int tile_id_counter = 0;

      // First pass: collects all operation tile locations and assigns tile ID.
      func.walk([&](Operation *op) {
        auto attr_array = op->getAttrOfType<ArrayAttr>("mapping_locs");
        if (!attr_array || attr_array.size() == 0) {
          return;
        }
        bool found = false;
        for (mlir::Attribute attr : attr_array) {
          if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
            auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
            auto id_attr = mlir::dyn_cast<IntegerAttr>(loc.get("id"));
            if (!resource_attr) continue;
            if (resource_attr.getValue() == "tile") {
              auto x_attr = loc.getAs<IntegerAttr>("x");
              auto y_attr = loc.getAs<IntegerAttr>("y");
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
              found = true;
              break;
            } else if (resource_attr.getValue() == "link" && id_attr) {
              int link_id = id_attr.getInt();
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
                            found = true;
                            goto link_found;
                          }
                        }
                      }
                    }
                  }
                }
              }
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
                      found = true;
                      break;
                    }
                  }
                }
              }
              link_found:;
            }
          }
        }
        if (!found) return;
      });

      // Second pass: generates instructions and calculates direction.
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
            if (resource_attr.getValue() == "tile" || resource_attr.getValue() == "link") {
              found = true;
              break;
            }
          }
        }
        if (!found) return;
        // Extracts operation code.
        llvm::StringRef fullOpName = op->getName().getStringRef();
        llvm::StringRef opcode = fullOpName;
        size_t dotPos = fullOpName.find_last_of('.');
        if (dotPos != llvm::StringRef::npos && dotPos + 1 < fullOpName.size()) {
          opcode = fullOpName.substr(dotPos + 1);
        }

        // Creates instruction object.
        llvm::json::Object inst_obj;
        // Set name, operands, result_types.
        inst_obj["name"] = fullOpName.str();
        // operands
        llvm::json::Array operands_array;
        for (mlir::Value operand : op->getOperands()) {
          if (auto defining_op = operand.getDefiningOp()) {
            operands_array.push_back(defining_op->getName().getStringRef().str());
          } else {
            operands_array.push_back("block_arg");
          }
        }
        inst_obj["operands"] = std::move(operands_array);
        // result_types
        llvm::json::Array result_types_array;
        for (mlir::Value result : op->getResults()) {
          std::string type_str;
          llvm::raw_string_ostream os(type_str);
          result.getType().print(os);
          result_types_array.push_back(os.str());
        }
        inst_obj["result_types"] = std::move(result_types_array);
        // Set time_step only (remove opcode)
        inst_obj["time_step"] = -1;

        // Handles constant value.
        if (opcode == "constant") {
          // Tries to get the constant value from the operation.
          if (op->getNumOperands() == 0 && op->getNumResults() == 1) {
            // For arith.constant, the value is stored as an attribute.
            if (auto value_attr = op->getAttr("value")) {
              if (auto int_attr = mlir::dyn_cast<IntegerAttr>(value_attr)) {
                inst_obj["constant_value"] = std::to_string(int_attr.getInt());
              } else if (auto float_attr = mlir::dyn_cast<FloatAttr>(value_attr)) {
                inst_obj["constant_value"] = std::to_string(float_attr.getValueAsDouble());
              }
            }
          }
        }

        // Handles mapping locations.
        auto mapping_attr_array = op->getAttrOfType<ArrayAttr>("mapping_locs");
        if (mapping_attr_array) {
          bool processed = false;
          for (mlir::Attribute attr : mapping_attr_array) {
            if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
              auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
              auto id_attr = mlir::dyn_cast<IntegerAttr>(loc.get("id"));
              auto timestep_attr = mlir::dyn_cast<IntegerAttr>(loc.get("time_step"));
              
              if (resource_attr && resource_attr.getValue() == "tile") {
                auto x_attr = loc.getAs<IntegerAttr>("x");
                auto y_attr = loc.getAs<IntegerAttr>("y");
                
                if (x_attr && y_attr) {
                  int x = x_attr.getInt();
                  int y = y_attr.getInt();
                  int time_step = timestep_attr ? timestep_attr.getInt() : -1;
                  
                  inst_obj["time_step"] = time_step;
                  inst_obj["dst_tile"] = "(" + std::to_string(x) + "," + std::to_string(y) + ")";
                  
                          // Calculates data movement directions based on tile coordinates.
        bool found_src = false;
        for (mlir::Value operand : op->getOperands()) {
          if (!operand) continue;
          if (auto defining_op = operand.getDefiningOp()) {
            // Finds the source tile by looking at the defining operation's tile mapping.
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
            
            // Calculates the direction FROM source TO current tile (this is the direction the current tile receives data from).
            // For example: if data moves from (1,1) to (1,2), then Tile(1,2) receives from South.
            inst_obj["src_direction"] = calculateRecvDirection(src_coord.first, src_coord.second, x, y).str();
            inst_obj["src_tile"] = "(" + std::to_string(src_coord.first) + "," + std::to_string(src_coord.second) + ")";
            found_src = true;
            break;
          }
        }

        // If no source tile is found, sets to Local.
        if (!found_src) {
          inst_obj["src_direction"] = "Local";
          inst_obj["src_tile"] = "(" + std::to_string(x) + "," + std::to_string(y) + ")";
        }
        
        // Calculates destination direction by looking at users of this operation.
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
        
        // if no destination tile is found, set to Local
        if (!found_dst) {
          inst_obj["dst_direction"] = "Local";
          inst_obj["dst_tile"] = "(" + std::to_string(x) + "," + std::to_string(y) + ")";
        }
        
        // Special handling for return operations
        if (opcode == "return") {
          // Return operations don't have destination direction
          inst_obj.erase("dst_direction");
          inst_obj.erase("dst_tile");
        }
                  
                  // Add instruction to corresponding tile
                  tile_instructions[std::make_pair(x, y)].push_back(std::move(inst_obj));
                  processed = true;
                  break; // only process the first tile mapping
                }
              } else if (resource_attr && resource_attr.getValue() == "link" && id_attr && !processed) {
                // For link mappings, we need to find the destination tile
                // Look for subsequent tile mappings or infer from operands
                int link_id = id_attr.getInt();
                int time_step = timestep_attr ? timestep_attr.getInt() : -1;
                
                // Try to find the destination tile by looking at subsequent operations
                // that use this operation's result
                std::pair<int, int> dst_tile = std::make_pair(-1, -1);
                
                // Look for operations that use this operation's result
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
                
                // If we found a destination, create the instruction
                if (dst_tile.first != -1 && dst_tile.second != -1) {
                  inst_obj["time_step"] = time_step;
                  inst_obj["dst_tile"] = "(" + std::to_string(dst_tile.first) + "," + std::to_string(dst_tile.second) + ")";
                  
                  // Find source tile from operands
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
                  
                  // calculate destination direction by looking at users of this operation
                  bool found_dst = false;
                  if (op->getNumResults() > 0) {
                    for (mlir::Value result : op->getResults()) {
                      if (!result) continue;
                      for (mlir::Operation *user : result.getUsers()) {
                        if (!user) continue;
                        auto user_it = op_to_final_tile.find(user);
                        if (user_it != op_to_final_tile.end()) {
                          int dst_x = user_it->second.first;
                          int dst_y = user_it->second.second;
                          inst_obj["dst_direction"] = calculateSendDirection(dst_tile.first, dst_tile.second, dst_x, dst_y).str();
                          inst_obj["dst_tile"] = "(" + std::to_string(dst_x) + "," + std::to_string(dst_y) + ")";
                          found_dst = true;
                          break;
                        }
                      }
                      if (found_dst) break;
                    }
                  }
                  
                  if (!found_dst) {
                    inst_obj["dst_direction"] = "Local";
                    inst_obj["dst_tile"] = "(" + std::to_string(dst_tile.first) + "," + std::to_string(dst_tile.second) + ")";
                  }
                  
                  // Add instruction to corresponding tile
                  tile_instructions[dst_tile].push_back(std::move(inst_obj));
                  processed = true;
                }
              }
            }
          }
        }
      });

      // organize instructions by PE and generate JSON
      // Organizes instructions by tile and generates JSON.
      llvm::json::Object tile_instructions_obj;
      for (std::pair<const std::pair<int, int>, llvm::json::Array>& tile_pair : tile_instructions) {
        int x = tile_pair.first.first;
        int y = tile_pair.first.second;
        llvm::json::Array& instructions = tile_pair.second;
        
        // Gets tile ID.
        auto coord = std::make_pair(x, y);
        int tile_id = tile_coord_to_id[coord];
        
        llvm::json::Object tile_obj;
        tile_obj["id"] = tile_id;
        tile_obj["x"] = x;
        tile_obj["y"] = y;
        // Adds instructions at the end, to ensure order.
        tile_obj["instructions"] = std::move(instructions);
        
        // Uses "Tile(id)" as key.
        std::string tile_key = "Tile(" + std::to_string(tile_id) + ")";
        tile_instructions_obj[tile_key] = std::move(tile_obj);
      }

      func_obj["tile_instructions"] = std::move(tile_instructions_obj);
      functions_array.push_back(std::move(func_obj));
    }

    // Generate both JSON and ASM output
    // JSON output
    llvm::json::Object root;
    root["functions"] = std::move(functions_array);

    // Create test/codegenerate directory if it doesn't exist.
    // if (std::error_code ec = llvm::sys::fs::create_directories("test/codegenerate")) {
    //     getOperation()->emitError("Failed to create 'test/codegenerate' directory: " + ec.message());
    //     return signalPassFailure();
    // }
    
    std::error_code ec;
    llvm::raw_fd_ostream json_out("generated-instructions.json", ec);
    if (ec) {
        getOperation()->emitError("Failed to open 'generated-instructions.json' for writing: " + ec.message());
        return signalPassFailure();
    }
    json_out << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
    
    // ASM output.
    std::error_code asm_ec;
    llvm::raw_fd_ostream asm_out("generated-instructions.asm", asm_ec);
    if (asm_ec) {
        getOperation()->emitError("Failed to open 'generated-instructions.asm' for writing: " + asm_ec.message());
        return signalPassFailure();
    }
    
    // Generate ASM for each function
    for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
      auto accel_attr = func->getAttrOfType<StringAttr>("accelerator");
      if (!accel_attr || accel_attr.getValue() != "neura")
        continue;

      // Rebuild op_to_final_tile mapping for ASM generation
      std::map<Operation*, std::pair<int, int>> asm_op_to_final_tile;
      std::map<Operation*, std::pair<int, int>> asm_op_to_source_tile;
      
      func.walk([&](Operation *op) {
        auto attr_array = op->getAttrOfType<ArrayAttr>("mapping_locs");
        if (!attr_array || attr_array.size() == 0) {
          return;
        }
        bool found = false;
        for (mlir::Attribute attr : attr_array) {
          if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
            auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
            auto id_attr = mlir::dyn_cast<IntegerAttr>(loc.get("id"));
            
            if (resource_attr && resource_attr.getValue() == "tile") {
              auto x_attr = loc.getAs<IntegerAttr>("x");
              auto y_attr = loc.getAs<IntegerAttr>("y");
              
              int x, y;
              if (x_attr && y_attr) {
                x = x_attr.getInt();
                y = y_attr.getInt();
              } else {
                x = id_attr ? id_attr.getInt() : 0;
                y = 0;
              }
              
              asm_op_to_final_tile[op] = std::make_pair(x, y);
              asm_op_to_source_tile[op] = std::make_pair(x, y);
              found = true;
              break;
            } else if (resource_attr && resource_attr.getValue() == "link" && id_attr) {
              // For link operations, we need to find the destination tile
              // Look for operations that use this operation's result
              if (op->getNumResults() > 0) {
                for (mlir::Value result : op->getResults()) {
                  if (!result) continue;
                  for (mlir::Operation *user : result.getUsers()) {
                    if (!user) continue;
                    auto user_attr_array = user->getAttrOfType<ArrayAttr>("mapping_locs");
                    if (!user_attr_array) continue;
                    for (mlir::Attribute user_attr : user_attr_array) {
                      if (!user_attr) continue;
                      if (auto user_loc = mlir::dyn_cast<DictionaryAttr>(user_attr)) {
                        auto user_resource_attr = mlir::dyn_cast<StringAttr>(user_loc.get("resource"));
                        if (user_resource_attr && user_resource_attr.getValue() == "tile") {
                          auto user_x_attr = user_loc.getAs<IntegerAttr>("x");
                          auto user_y_attr = user_loc.getAs<IntegerAttr>("y");
                          if (user_x_attr && user_y_attr) {
                            int dst_x = user_x_attr.getInt();
                            int dst_y = user_y_attr.getInt();
                            asm_op_to_final_tile[op] = std::make_pair(dst_x, dst_y);
                            // Set source location for link operation from its operand
                            for (mlir::Value operand : op->getOperands()) {
                              if (!operand) continue;
                              if (auto defining_op = operand.getDefiningOp()) {
                                auto src_it = asm_op_to_source_tile.find(defining_op);
                                if (src_it != asm_op_to_source_tile.end()) {
                                  asm_op_to_source_tile[op] = src_it->second;
                                  break;
                                }
                              }
                            }
                            found = true;
                            goto asm_link_found;
                          }
                        }
                      }
                    }
                  }
                }
              }
              // If no destination found, use source location as fallback
              if (asm_op_to_final_tile.find(op) == asm_op_to_final_tile.end()) {
                for (mlir::Value operand : op->getOperands()) {
                  if (!operand) continue;
                  if (auto defining_op = operand.getDefiningOp()) {
                    auto it = asm_op_to_final_tile.find(defining_op);
                    if (it != asm_op_to_final_tile.end()) {
                      asm_op_to_final_tile[op] = it->second;
                      break;
                    }
                  }
                }
              }
              asm_link_found:;
            }
          }
        }
        if (!found) return;
      });

      // Group operations by PE and time step
      std::map<std::pair<int, int>, std::map<int, std::vector<Operation*>>> pe_time_ops;
      
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
          }
        }
        if (!found) return;
      });
      
      // Generate ASM for each PE
      for (std::pair<const std::pair<int, int>, std::map<int, std::vector<Operation*>>>& pe_pair : pe_time_ops) {
        int x = pe_pair.first.first;
        int y = pe_pair.first.second;
        std::map<int, std::vector<Operation*>>& time_ops = pe_pair.second;
        
        asm_out << "PE(" << x << "," << y << "):\n";
        asm_out << "{\n";
        
        // Generate Entry conditions based on input directions.
        std::set<std::string> input_directions;
        for (std::pair<const int, std::vector<Operation*>>& time_pair : time_ops) {
          for (mlir::Operation* op : time_pair.second) {
            // Find source directions for this operation.
            for (mlir::Value operand : op->getOperands()) {
              if (!operand) continue;
              if (auto defining_op = operand.getDefiningOp()) {
                // Find the source tile by looking at the defining operation's tile mapping.
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
        
        // Also check for link operations that target this tile.
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
                // Check if this link operation targets the current tile.
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
                            // This link operation targets the current tile, find its source.
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
        
        // Write Entry line
        asm_out << "    Entry ";
        bool first = true;
        for (const std::string& dir : input_directions) {
          if (!first) asm_out << ", ";
          asm_out << "[" << dir << ", R]";
          first = false;
        }
        if (input_directions.empty()) {
          asm_out << "[]";
        }
        asm_out << " => ";
        
        // Determine if it's Loop or Once based on time steps
        if (time_ops.size() > 1) {
          asm_out << "Loop {\n";
        } else {
          asm_out << "Once {\n";
        }
        
        // Generate operations for each time step
        for (std::pair<const int, std::vector<Operation*>>& time_pair : time_ops) {
          int time_step = time_pair.first;
          std::vector<Operation*>& ops = time_pair.second;
          
          for (mlir::Operation* op : ops) {
            asm_out << "        {\n";
            
            // Extract operation name
            llvm::StringRef fullOpName = op->getName().getStringRef();
            llvm::StringRef opcode = fullOpName;
            size_t dotPos = fullOpName.find_last_of('.');
            if (dotPos != llvm::StringRef::npos && dotPos + 1 < fullOpName.size()) {
              opcode = fullOpName.substr(dotPos + 1);
            }
            
            // Convert opcode to uppercase
            std::string upper_opcode = opcode.str();
            std::transform(upper_opcode.begin(), upper_opcode.end(), upper_opcode.begin(), ::toupper);
            
            // Handle different operation types
            if (upper_opcode == "RETURN") {
              // For return operations, output without destination direction
              asm_out << "            " << upper_opcode;
              
              // Handle operands (input directions)
              for (mlir::Value operand : op->getOperands()) {
                asm_out << ", [";
                if (auto defining_op = operand.getDefiningOp()) {
                  // Find the source PE by looking at the defining operation's tile mapping
                  std::pair<int, int> src_coord = std::make_pair(x, y); // default to current location
                  
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
              
              // Return operations don't have destination direction
              asm_out << "\n";
              asm_out << "            NOP\n";
            } else if (upper_opcode == "CONSTANT") {
              // For constants, output the value
              if (auto value_attr = op->getAttr("value")) {
                if (auto int_attr = mlir::dyn_cast<IntegerAttr>(value_attr)) {
                  asm_out << "            " << upper_opcode << ", IMM[" << int_attr.getInt() << "] -> [";
                } else if (auto float_attr = mlir::dyn_cast<FloatAttr>(value_attr)) {
                  asm_out << "            " << upper_opcode << ", IMM[" << float_attr.getValueAsDouble() << "] -> [";
                }
              }
              
              // Find destination direction
              bool found_dst = false;
              for (mlir::Value result : op->getResults()) {
                if (!result) continue;
                for (mlir::Operation *user : result.getUsers()) {
                  if (!user) continue;
                  auto user_it = asm_op_to_final_tile.find(user);
                  if (user_it != asm_op_to_final_tile.end()) {
                    int dst_x = user_it->second.first;
                    int dst_y = user_it->second.second;
                    std::string dst_direction = calculateSendDirection(x, y, dst_x, dst_y).str();
                    asm_out << dst_direction << ", R]\n";
                    found_dst = true;
                    break;
                  }
                }
                if (found_dst) break;
              }
              if (!found_dst) {
                asm_out << "Local, R]\n";
              }
            } else {
              // For other operations, handle operands and results
              asm_out << "            " << upper_opcode;
              
              // Handle operands (input directions)
              for (mlir::Value operand : op->getOperands()) {
                asm_out << ", [";
                if (auto defining_op = operand.getDefiningOp()) {
                  // Use source location mapping to find the original source
                  auto src_it = asm_op_to_source_tile.find(defining_op);
                  if (src_it != asm_op_to_source_tile.end()) {
                    int src_x = src_it->second.first;
                    int src_y = src_it->second.second;
                    std::string direction = calculateRecvDirection(src_x, src_y, x, y).str();
                    asm_out << direction << ", R";
                  } else {
                    asm_out << "Local, R";
                  }
                } else {
                  asm_out << "Local, R";
                }
                asm_out << "]";
              }
              
              // Handle result (output direction)
              asm_out << " -> [";
              bool found_dst = false;
              for (mlir::Value result : op->getResults()) {
                if (!result) continue;
                for (mlir::Operation *user : result.getUsers()) {
                  if (!user) continue;
                  auto user_it = asm_op_to_final_tile.find(user);
                  if (user_it != asm_op_to_final_tile.end()) {
                    int dst_x = user_it->second.first;
                    int dst_y = user_it->second.second;
                    std::string dst_direction = calculateSendDirection(x, y, dst_x, dst_y).str();
                    asm_out << dst_direction << ", R";
                    found_dst = true;
                    break;
                  }
                }
                if (found_dst) break;
              }
              if (!found_dst) {
                asm_out << "Local, R";
              }
              asm_out << "]\n";
            }
            
            asm_out << "        }\n";
          }
        }
        
        asm_out << "    }\n";
        asm_out << "}\n\n";
      }
    }
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<mlir::Pass> createGenerateCodePass() {
  return std::make_unique<GenerateCodePass>();
}
} // namespace mlir::neura

