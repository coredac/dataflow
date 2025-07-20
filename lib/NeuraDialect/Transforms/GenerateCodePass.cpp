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
  if (dx == 0 && dy > 0) return "N";
  if (dx == 0 && dy < 0) return "S";
  if (dx > 0 && dy == 0) return "E";
  if (dx < 0 && dy == 0) return "W";
  if (dx > 0 && dy > 0) return "NE";
  if (dx > 0 && dy < 0) return "SE";
  if (dx < 0 && dy > 0) return "NW";
  if (dx < 0 && dy < 0) return "SW";
  
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
              
              // assign ID to new PE coordinates
              auto coord = std::make_pair(x, y);
              if (pe_coord_to_id.find(coord) == pe_coord_to_id.end()) {
                pe_coord_to_id[coord] = pe_id_counter++;
              }
              break; // only process the first tile mapping
            } else if (resource_attr && resource_attr.getValue() == "link" && id_attr) {
              int link_id = id_attr.getInt();
              if (op_to_final_tile.find(op) == op_to_final_tile.end()) {
                for (Value operand : op->getOperands()) {
                  if (auto defining_op = operand.getDefiningOp()) {
                    auto it = op_to_final_tile.find(defining_op);
                    if (it != op_to_final_tile.end()) {
                      op_to_final_tile[op] = it->second;
                      break;
                    }
                  }
                }
              }
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
        if (auto const_op = mlir::dyn_cast<neura::ConstantOp>(op)) {
          auto val_attr = const_op.getValue();
          if (val_attr) {
            if (auto int_attr = mlir::dyn_cast<IntegerAttr>(val_attr)) {
              inst_obj["constant_value"] = std::to_string(int_attr.getInt());
            } else if (auto float_attr = mlir::dyn_cast<FloatAttr>(val_attr)) {
              inst_obj["constant_value"] = std::to_string(float_attr.getValueAsDouble());
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
                  
                  // calculate data movement direction
                  bool found_src = false;
                  for (Value operand : op->getOperands()) {
                    if (auto defining_op = operand.getDefiningOp()) {
                      // For data_mov, we need to find the source tile of the operand
                      auto it = op_to_final_tile.find(defining_op);
                      if (it != op_to_final_tile.end()) {
                        int src_x = it->second.first;
                        int src_y = it->second.second;
                        inst_obj["direction"] = calculateDirection(src_x, src_y, x, y).str();
                        inst_obj["src_tile"] = "(" + std::to_string(src_x) + "," + std::to_string(src_y) + ")";
                        found_src = true;
                        break;
                      }
                    }
                  }
                  
                  // if no source tile is found, set to Local
                  if (!found_src) {
                    inst_obj["direction"] = "Local";
                    inst_obj["src_tile"] = "(" + std::to_string(x) + "," + std::to_string(y) + ")";
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
                        inst_obj["direction"] = calculateDirection(src_x, src_y, dst_tile.first, dst_tile.second).str();
                        inst_obj["src_tile"] = "(" + std::to_string(src_x) + "," + std::to_string(src_y) + ")";
                        found_src = true;
                        break;
                      }
                    }
                  }
                  
                  if (!found_src) {
                    inst_obj["direction"] = "Local";
                    inst_obj["src_tile"] = "(" + std::to_string(dst_tile.first) + "," + std::to_string(dst_tile.second) + ")";
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

    // Final JSON object.
    llvm::json::Object root;
    root["functions"] = std::move(functions_array);

    // llvm::outs() << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
    std::error_code ec;
    llvm::raw_fd_ostream json_out("generated-instructions.json", ec);
    if (ec) {
        getOperation()->emitError("Failed to open 'generated-instructions.json' for writing: " + ec.message());
        return signalPassFailure();
    }
    json_out << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<mlir::Pass> createGenerateCodePass() {
  return std::make_unique<GenerateCodePass>();
}
} // namespace mlir::neura

