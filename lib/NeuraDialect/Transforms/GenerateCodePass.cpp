#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
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

// calculate direction of data movement
llvm::StringRef calculateDirection(int src_x, int src_y, int dst_x, int dst_y) {
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

    for (auto func : module.getOps<func::FuncOp>()) {
      auto accel_attr = func->getAttrOfType<StringAttr>("accelerator");
      if (!accel_attr || accel_attr.getValue() != "neura")
        continue;

      llvm::json::Object func_obj;
      func_obj["name"] = func.getName().str();

      if (auto ii_attr = func->getAttrOfType<IntegerAttr>("CompiledII"))
        func_obj["CompiledII"] = ii_attr.getInt();
      if (auto recMII_attr = func->getAttrOfType<IntegerAttr>("RecMII"))
        func_obj["RecMII"] = recMII_attr.getInt();
      if (auto resMII_attr = func->getAttrOfType<IntegerAttr>("ResMII"))
        func_obj["ResMII"] = resMII_attr.getInt();

      // map of instructions organized by PE
      std::map<std::pair<int, int>, llvm::json::Array> pe_instructions;
      
      // map of operation to its mapping location, for calculating data movement direction
      std::map<Operation*, std::pair<int, int>> op_to_tile;
      
      // map of operation to its final tile location (after data movement)
      std::map<Operation*, std::pair<int, int>> op_to_final_tile;
      
      // map of operation to its original source location (for data flow tracking)
      std::map<Operation*, std::pair<int, int>> op_to_source_tile;
      
      // map of link ID to source and destination tiles
      std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> link_to_tiles;
      
      // map of PE coordinates to ID
      std::map<std::pair<int, int>, int> pe_coord_to_id;
      int pe_id_counter = 0;

      // first pass: collect all operation tile locations and assign PE ID
      func.walk([&](Operation *op) {
        if (isa<func::ReturnOp>(op))
          return;

        auto attr_array = op->getAttrOfType<ArrayAttr>("mapping_locs");
        if (!attr_array || attr_array.size() == 0) {
          // skip if no mapping information
          return;
        }
        for (Attribute attr : attr_array) {
          if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
            auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
            auto id_attr = mlir::dyn_cast<IntegerAttr>(loc.get("id"));
            
            if (resource_attr && resource_attr.getValue() == "tile") {
              auto xAttr = loc.getAs<IntegerAttr>("x");
              auto yAttr = loc.getAs<IntegerAttr>("y");
              
              int x, y;
              if (xAttr && yAttr) {
                x = xAttr.getInt();
                y = yAttr.getInt();
              } else {
                // if no x, y coordinates, use tile ID as coordinates
                x = id_attr ? id_attr.getInt() : 0;
                y = 0;
              }
              
              op_to_tile[op] = std::make_pair(x, y);
              op_to_final_tile[op] = std::make_pair(x, y);
              op_to_source_tile[op] = std::make_pair(x, y);
              
              // assign ID to new PE coordinates
              auto coord = std::make_pair(x, y);
              if (pe_coord_to_id.find(coord) == pe_coord_to_id.end()) {
                pe_coord_to_id[coord] = pe_id_counter++;
              }
              break; // only process the first tile mapping
            } else if (resource_attr && resource_attr.getValue() == "link" && id_attr) {
              int link_id = id_attr.getInt();
              // For link operations, we need to find the destination tile
              // Look for operations that use this operation's result
              for (Value result : op->getResults()) {
                for (auto user : result.getUsers()) {
                  auto user_attr_array = user->getAttrOfType<ArrayAttr>("mapping_locs");
                  if (user_attr_array) {
                    for (Attribute user_attr : user_attr_array) {
                      if (auto user_loc = mlir::dyn_cast<DictionaryAttr>(user_attr)) {
                        auto user_resource_attr = mlir::dyn_cast<StringAttr>(user_loc.get("resource"));
                        if (user_resource_attr && user_resource_attr.getValue() == "tile") {
                          auto user_xAttr = user_loc.getAs<IntegerAttr>("x");
                          auto user_yAttr = user_loc.getAs<IntegerAttr>("y");
                          if (user_xAttr && user_yAttr) {
                            int dst_x = user_xAttr.getInt();
                            int dst_y = user_yAttr.getInt();
                            op_to_final_tile[op] = std::make_pair(dst_x, dst_y);
                            // Set source location for link operation from its operand
                            for (Value operand : op->getOperands()) {
                              if (auto defining_op = operand.getDefiningOp()) {
                                auto src_it = op_to_source_tile.find(defining_op);
                                if (src_it != op_to_source_tile.end()) {
                                  op_to_source_tile[op] = src_it->second;
                                  break;
                                }
                              }
                            }
                            goto link_found;
                          }
                        }
                      }
                    }
                  }
                }
              }
              // If no destination found, use source location as fallback
              if (op_to_final_tile.find(op) == op_to_final_tile.end()) {
                for (Value operand : op->getOperands()) {
                  if (auto defining_op = operand.getDefiningOp()) {
                    auto it = op_to_final_tile.find(defining_op);
                    if (it != op_to_final_tile.end()) {
                      op_to_final_tile[op] = it->second;
                      // Set source location for link operation
                      auto src_it = op_to_source_tile.find(defining_op);
                      if (src_it != op_to_source_tile.end()) {
                        op_to_source_tile[op] = src_it->second;
                      }
                      break;
                    }
                  }
                }
              }
              link_found:;
            }
          }
        }
      });

      // second pass: generate instructions and calculate direction
      func.walk([&](Operation *op) {
        if (isa<func::ReturnOp>(op))
          return;

        auto attr_array = op->getAttrOfType<ArrayAttr>("mapping_locs");
        if (!attr_array || attr_array.size() == 0) {
          // skip if no mapping information
          return;
        }
        // extract operation code
        llvm::StringRef fullOpName = op->getName().getStringRef();
        llvm::StringRef opcode = fullOpName;
        size_t dotPos = fullOpName.find_last_of('.');
        if (dotPos != llvm::StringRef::npos && dotPos + 1 < fullOpName.size()) {
          opcode = fullOpName.substr(dotPos + 1);
        }

        // create instruction object
        llvm::json::Object inst_obj;
        inst_obj["opcode"] = opcode.str();
        inst_obj["time_step"] = -1;

        // handle constant value
        if (opcode == "constant") {
          // Try to get the constant value from the operation
          if (op->getNumOperands() == 0 && op->getNumResults() == 1) {
            // For arith.constant, the value is stored as an attribute
            if (auto value_attr = op->getAttr("value")) {
              if (auto int_attr = mlir::dyn_cast<IntegerAttr>(value_attr)) {
                inst_obj["constant_value"] = std::to_string(int_attr.getInt());
              } else if (auto float_attr = mlir::dyn_cast<FloatAttr>(value_attr)) {
                inst_obj["constant_value"] = std::to_string(float_attr.getValueAsDouble());
              }
            }
          }
        }

        // handle operands
        llvm::json::Array operands_array;
        for (Value operand : op->getOperands()) {
          if (auto defining_op = operand.getDefiningOp()) {
            operands_array.push_back(defining_op->getName().getStringRef().str());
          } else {
            operands_array.push_back("block_arg");
          }
        }
        inst_obj["operands"] = std::move(operands_array);

        // handle mapping locations
        auto mapping_attr_array = op->getAttrOfType<ArrayAttr>("mapping_locs");
        if (mapping_attr_array) {
          bool processed = false;
          for (Attribute attr : mapping_attr_array) {
            if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
              auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
              auto id_attr = mlir::dyn_cast<IntegerAttr>(loc.get("id"));
              auto timestep_attr = mlir::dyn_cast<IntegerAttr>(loc.get("time_step"));
              
              if (resource_attr && resource_attr.getValue() == "tile") {
                auto xAttr = loc.getAs<IntegerAttr>("x");
                auto yAttr = loc.getAs<IntegerAttr>("y");
                
                if (xAttr && yAttr) {
                  int x = xAttr.getInt();
                  int y = yAttr.getInt();
                  int time_step = timestep_attr ? timestep_attr.getInt() : -1;
                  
                  inst_obj["time_step"] = time_step;
                  inst_obj["dst_tile"] = "(" + std::to_string(x) + "," + std::to_string(y) + ")";
                  
                          // calculate data movement directions
        bool found_src = false;
        for (Value operand : op->getOperands()) {
          if (auto defining_op = operand.getDefiningOp()) {
            // Use source location mapping to find the original source
            auto src_it = op_to_source_tile.find(defining_op);
            if (src_it != op_to_source_tile.end()) {
              int src_x = src_it->second.first;
              int src_y = src_it->second.second;
              inst_obj["src_direction"] = calculateDirection(src_x, src_y, x, y).str();
              inst_obj["src_tile"] = "(" + std::to_string(src_x) + "," + std::to_string(src_y) + ")";
              found_src = true;
              break;
            }
          }
        }
        
        // if no source tile is found, set to Local
        if (!found_src) {
          inst_obj["src_direction"] = "Local";
          inst_obj["src_tile"] = "(" + std::to_string(x) + "," + std::to_string(y) + ")";
        }
        
        // calculate destination direction by looking at users of this operation
        bool found_dst = false;
        for (Value result : op->getResults()) {
          for (auto user : result.getUsers()) {
            auto user_it = op_to_final_tile.find(user);
            if (user_it != op_to_final_tile.end()) {
              int dst_x = user_it->second.first;
              int dst_y = user_it->second.second;
              inst_obj["dst_direction"] = calculateDirection(x, y, dst_x, dst_y).str();
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
                  
                  // Add instruction to corresponding PE
                  pe_instructions[std::make_pair(x, y)].push_back(std::move(inst_obj));
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
                for (auto user : op->getResults()[0].getUsers()) {
                  auto user_it = op_to_final_tile.find(user);
                  if (user_it != op_to_final_tile.end()) {
                    dst_tile = user_it->second;
                    break;
                  }
                }
                
                // If we found a destination, create the instruction
                if (dst_tile.first != -1 && dst_tile.second != -1) {
                  inst_obj["time_step"] = time_step;
                  inst_obj["dst_tile"] = "(" + std::to_string(dst_tile.first) + "," + std::to_string(dst_tile.second) + ")";
                  
                  // Find source tile from operands
                  bool found_src = false;
                  for (Value operand : op->getOperands()) {
                    if (auto defining_op = operand.getDefiningOp()) {
                      auto it = op_to_final_tile.find(defining_op);
                      if (it != op_to_final_tile.end()) {
                        int src_x = it->second.first;
                        int src_y = it->second.second;
                        inst_obj["src_direction"] = calculateDirection(src_x, src_y, dst_tile.first, dst_tile.second).str();
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
                  for (Value result : op->getResults()) {
                    for (auto user : result.getUsers()) {
                      auto user_it = op_to_final_tile.find(user);
                      if (user_it != op_to_final_tile.end()) {
                        int dst_x = user_it->second.first;
                        int dst_y = user_it->second.second;
                        inst_obj["dst_direction"] = calculateDirection(dst_tile.first, dst_tile.second, dst_x, dst_y).str();
                        inst_obj["dst_tile"] = "(" + std::to_string(dst_x) + "," + std::to_string(dst_y) + ")";
                        found_dst = true;
                        break;
                      }
                    }
                    if (found_dst) break;
                  }
                  
                  if (!found_dst) {
                    inst_obj["dst_direction"] = "Local";
                    inst_obj["dst_tile"] = "(" + std::to_string(dst_tile.first) + "," + std::to_string(dst_tile.second) + ")";
                  }
                  
                  // Add instruction to corresponding PE
                  pe_instructions[dst_tile].push_back(std::move(inst_obj));
                  processed = true;
                }
              }
            }
          }
        }
      });

      // organize instructions by PE and generate JSON
      llvm::json::Object pe_instructions_obj;
      for (auto& pe_pair : pe_instructions) {
        int x = pe_pair.first.first;
        int y = pe_pair.first.second;
        auto& instructions = pe_pair.second;
        
        // get PE ID
        auto coord = std::make_pair(x, y);
        int pe_id = pe_coord_to_id[coord];
        
        llvm::json::Object pe_obj;
        pe_obj["id"] = pe_id;
        pe_obj["x"] = x;
        pe_obj["y"] = y;
        // add instructions at the end, to ensure order
        pe_obj["instructions"] = std::move(instructions);
        
        // use "PE(id)" as key
        std::string pe_key = "PE(" + std::to_string(pe_id) + ")";
        pe_instructions_obj[pe_key] = std::move(pe_obj);
      }
      
      func_obj["pe_instructions"] = std::move(pe_instructions_obj);
      functions_array.push_back(std::move(func_obj));
    }

    // Generate both JSON and ASM output
    // JSON output
    llvm::json::Object root;
    root["functions"] = std::move(functions_array);

    std::error_code ec;
    llvm::raw_fd_ostream json_out("generated-instructions.json", ec);
    if (ec) {
        getOperation()->emitError("Failed to open 'generated-instructions.json' for writing: " + ec.message());
        return signalPassFailure();
    }
    json_out << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
    
    // ASM output
    std::error_code asm_ec;
    llvm::raw_fd_ostream asm_out("generated-instructions.asm", asm_ec);
    if (asm_ec) {
        getOperation()->emitError("Failed to open 'generated-instructions.asm' for writing: " + asm_ec.message());
        return signalPassFailure();
    }
    
    // Generate ASM for each function
    for (auto func : module.getOps<func::FuncOp>()) {
      auto accel_attr = func->getAttrOfType<StringAttr>("accelerator");
      if (!accel_attr || accel_attr.getValue() != "neura")
        continue;

      // Rebuild op_to_final_tile mapping for ASM generation
      std::map<Operation*, std::pair<int, int>> asm_op_to_final_tile;
      std::map<Operation*, std::pair<int, int>> asm_op_to_source_tile;
      
      func.walk([&](Operation *op) {
        if (isa<func::ReturnOp>(op))
          return;

        auto attr_array = op->getAttrOfType<ArrayAttr>("mapping_locs");
        if (!attr_array || attr_array.size() == 0) {
          return;
        }
        for (Attribute attr : attr_array) {
          if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
            auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
            auto id_attr = mlir::dyn_cast<IntegerAttr>(loc.get("id"));
            
            if (resource_attr && resource_attr.getValue() == "tile") {
              auto xAttr = loc.getAs<IntegerAttr>("x");
              auto yAttr = loc.getAs<IntegerAttr>("y");
              
              int x, y;
              if (xAttr && yAttr) {
                x = xAttr.getInt();
                y = yAttr.getInt();
              } else {
                x = id_attr ? id_attr.getInt() : 0;
                y = 0;
              }
              
              asm_op_to_final_tile[op] = std::make_pair(x, y);
              asm_op_to_source_tile[op] = std::make_pair(x, y);
              break;
            } else if (resource_attr && resource_attr.getValue() == "link" && id_attr) {
              // For link operations, we need to find the destination tile
              // Look for operations that use this operation's result
              for (Value result : op->getResults()) {
                for (auto user : result.getUsers()) {
                  auto user_attr_array = user->getAttrOfType<ArrayAttr>("mapping_locs");
                  if (user_attr_array) {
                    for (Attribute user_attr : user_attr_array) {
                      if (auto user_loc = mlir::dyn_cast<DictionaryAttr>(user_attr)) {
                        auto user_resource_attr = mlir::dyn_cast<StringAttr>(user_loc.get("resource"));
                        if (user_resource_attr && user_resource_attr.getValue() == "tile") {
                          auto user_xAttr = user_loc.getAs<IntegerAttr>("x");
                          auto user_yAttr = user_loc.getAs<IntegerAttr>("y");
                          if (user_xAttr && user_yAttr) {
                            int dst_x = user_xAttr.getInt();
                            int dst_y = user_yAttr.getInt();
                            asm_op_to_final_tile[op] = std::make_pair(dst_x, dst_y);
                            // Set source location for link operation from its operand
                            for (Value operand : op->getOperands()) {
                              if (auto defining_op = operand.getDefiningOp()) {
                                auto src_it = asm_op_to_source_tile.find(defining_op);
                                if (src_it != asm_op_to_source_tile.end()) {
                                  asm_op_to_source_tile[op] = src_it->second;
                                  break;
                                }
                              }
                            }
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
                for (Value operand : op->getOperands()) {
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
      });

      // Group operations by PE and time step
      std::map<std::pair<int, int>, std::map<int, std::vector<Operation*>>> pe_time_ops;
      
      func.walk([&](Operation *op) {
        if (isa<func::ReturnOp>(op))
          return;

        auto attr_array = op->getAttrOfType<ArrayAttr>("mapping_locs");
        if (!attr_array || attr_array.size() == 0) {
          return;
        }
        
        for (Attribute attr : attr_array) {
          if (auto loc = mlir::dyn_cast<DictionaryAttr>(attr)) {
            auto resource_attr = mlir::dyn_cast<StringAttr>(loc.get("resource"));
            auto timestep_attr = mlir::dyn_cast<IntegerAttr>(loc.get("time_step"));
            
            if (resource_attr && resource_attr.getValue() == "tile") {
              auto xAttr = loc.getAs<IntegerAttr>("x");
              auto yAttr = loc.getAs<IntegerAttr>("y");
              
              if (xAttr && yAttr && timestep_attr) {
                int x = xAttr.getInt();
                int y = yAttr.getInt();
                int time_step = timestep_attr.getInt();
                pe_time_ops[std::make_pair(x, y)][time_step].push_back(op);
                break;
              }
            }
          }
        }
      });
      
      // Generate ASM for each PE
      for (auto& pe_pair : pe_time_ops) {
        int x = pe_pair.first.first;
        int y = pe_pair.first.second;
        auto& time_ops = pe_pair.second;
        
        asm_out << "PE(" << x << "," << y << "):\n";
        asm_out << "{\n";
        
        // Generate Entry conditions based on input directions
        std::set<std::string> input_directions;
        for (auto& time_pair : time_ops) {
          for (Operation* op : time_pair.second) {
            // Find source directions for this operation
            for (Value operand : op->getOperands()) {
              if (auto defining_op = operand.getDefiningOp()) {
                // Use source location mapping to find the original source
                auto src_it = asm_op_to_source_tile.find(defining_op);
                if (src_it != asm_op_to_source_tile.end()) {
                  int src_x = src_it->second.first;
                  int src_y = src_it->second.second;
                  std::string direction = calculateDirection(src_x, src_y, x, y).str();
                  if (direction != "Local") {
                    input_directions.insert(direction);
                  }
                }
              }
            }
          }
        }
        
        // Write Entry line
        asm_out << "    Entry ";
        bool first = true;
        for (const auto& dir : input_directions) {
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
        for (auto& time_pair : time_ops) {
          int time_step = time_pair.first;
          auto& ops = time_pair.second;
          
          for (Operation* op : ops) {
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
            if (upper_opcode == "CONSTANT") {
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
              for (Value result : op->getResults()) {
                for (auto user : result.getUsers()) {
                  auto user_it = asm_op_to_final_tile.find(user);
                  if (user_it != asm_op_to_final_tile.end()) {
                    int dst_x = user_it->second.first;
                    int dst_y = user_it->second.second;
                    std::string dst_direction = calculateDirection(x, y, dst_x, dst_y).str();
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
              for (Value operand : op->getOperands()) {
                asm_out << ", [";
                if (auto defining_op = operand.getDefiningOp()) {
                  // Use source location mapping to find the original source
                  auto src_it = asm_op_to_source_tile.find(defining_op);
                  if (src_it != asm_op_to_source_tile.end()) {
                    int src_x = src_it->second.first;
                    int src_y = src_it->second.second;
                    std::string direction = calculateDirection(src_x, src_y, x, y).str();
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
              for (Value result : op->getResults()) {
                for (auto user : result.getUsers()) {
                  auto user_it = asm_op_to_final_tile.find(user);
                  if (user_it != asm_op_to_final_tile.end()) {
                    int dst_x = user_it->second.first;
                    int dst_y = user_it->second.second;
                    std::string dst_direction = calculateDirection(x, y, dst_x, dst_y).str();
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

