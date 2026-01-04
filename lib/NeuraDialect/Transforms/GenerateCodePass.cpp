#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <unordered_set>
#include <tuple>
#include <sstream>

#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/NeuraOps.h"

using namespace mlir;
using namespace neura;

namespace {

struct Operand {
  std::string operand;
  std::string color;
  Operand(const std::string &op, const std::string &c = "RED")
      : operand(op), color(c) {}
};

struct Instruction {
  std::string opcode;
  std::vector<Operand> src_operands;
  std::vector<Operand> dst_operands;
  int id = -1;       // Unique instruction ID for debug/DFG.
  int time_step = -1; // original scheduling timestep.
  int index_per_ii = -1; // grouping key (typically time_step % compiled_ii).
  int invalid_iterations = 0; // prologue length before the op becomes valid.
  Instruction(const std::string &op) : opcode(op) {}
};

// A single-entry model per tile for now. Can be extended later.
struct Entry {
  std::string entry_id;
  std::string type;
  std::vector<Instruction> instructions;
  Entry(const std::string &id, const std::string &t = "loop")
      : entry_id(id), type(t) {}
};

struct Tile {
  int col_idx, row_idx;
  int core_id;
  Entry entry; // single entry per tile.
  Tile(int c, int r, int id) : col_idx(c), row_idx(r), core_id(id), entry("entry0", "loop") {}
};

struct ArrayConfig {
  int columns;
  int rows;
  int compiled_ii = -1;
  std::vector<Tile> cores;
};

struct TileLocation {
  int col_idx = -1, row_idx = -1, time_step = -1;
  int index_per_ii = -1;
  int invalid_iterations = 0;
  bool has_tile = false;
};

// ---- Operation kind helpers ----.
static bool isDataMov(Operation *op) { return dyn_cast<DataMovOp>(op) != nullptr; }
static bool isCtrlMov(Operation *op) { return dyn_cast<CtrlMovOp>(op) != nullptr; }
static bool isPhiStart(Operation *op) { return dyn_cast<PhiStartOp>(op) != nullptr; }
static bool isReserve(Operation *op) { return dyn_cast<ReserveOp>(op) != nullptr; }
static bool isConstant(Operation *op) { return dyn_cast<ConstantOp>(op) != nullptr; }
// ---- Constant for phi_start operation ----.
static constexpr unsigned kReserveOpIndex = 1;

// Returns the reserve operand for phi_start (operand #1). Guards to ReserveOp.
static Value getReserveOperand(Operation *op) {
  if (auto phi_start = dyn_cast<PhiStartOp>(op)) {
    assert(op->getNumOperands() > kReserveOpIndex &&
           "phi_start must have a reserve at operand #1");
    Value candidate = phi_start->getOperand(kReserveOpIndex);
    assert((!candidate || isa<ReserveOp>(candidate.getDefiningOp())) &&
           "phi_start operand #1 must be a ReserveOp");
    return candidate;
  }
  return Value();
}

namespace mapping_utils {
// ----- placement helpers -----.
static TileLocation getTileLocation(Operation *op) {
  TileLocation tile_location;
  if (auto arr = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
    for (Attribute a : arr) {
      auto d = dyn_cast<DictionaryAttr>(a);
      if (!d) continue;
      auto resource = dyn_cast_or_null<StringAttr>(d.get("resource"));
      if (auto timestep = dyn_cast_or_null<IntegerAttr>(d.get("time_step")))
        tile_location.time_step = timestep.getInt();
      if (auto index_attr = dyn_cast_or_null<IntegerAttr>(d.get("index_per_ii")))
        tile_location.index_per_ii = index_attr.getInt();
      if (auto invalid_attr = dyn_cast_or_null<IntegerAttr>(d.get("invalid_iterations")))
        tile_location.invalid_iterations = invalid_attr.getInt();
      if (resource && resource.getValue() == "tile") {
        if (auto x_coord = dyn_cast_or_null<IntegerAttr>(d.get("x"))) tile_location.col_idx = x_coord.getInt();
        if (auto y_coord = dyn_cast_or_null<IntegerAttr>(d.get("y"))) tile_location.row_idx = y_coord.getInt();
        tile_location.has_tile = true;
      }
    }
  }
  // If tile mappings exist, x/y must be valid.
  assert(!tile_location.has_tile || (tile_location.col_idx >= 0 && tile_location.row_idx >= 0));
  return tile_location;
}

// Helper to get mapping_locations arrays.
static ArrayAttr getMappingLocations(Operation *op) {
  return op->getAttrOfType<ArrayAttr>("mapping_locs");
}

static std::optional<int> getMappedRegId(Operation *op) {
  if (auto mapping_locations = getMappingLocations(op)) {
    for (Attribute location_attr : mapping_locations) {
      auto location_dict = dyn_cast<DictionaryAttr>(location_attr);
      if (!location_dict) continue;
      auto resource_attr = dyn_cast_or_null<StringAttr>(location_dict.get("resource"));
      if (!resource_attr) continue;
      if (resource_attr.getValue() == "register" || resource_attr.getValue() == "reg") {
        if (auto per_tile_register_id =
                dyn_cast_or_null<IntegerAttr>(location_dict.get("per_tile_register_id"))) {
          return per_tile_register_id.getInt();
        }
      }
    }
  }
  return std::nullopt;
}

} // namespace mapping_utils

static std::string getOpcode(Operation *op) {
  std::string opcode = op->getName().getStringRef().str();
  if (opcode.rfind("neura.", 0) == 0) opcode = opcode.substr(6);
  if (isConstant(op)) return "CONSTANT";
  std::transform(opcode.begin(), opcode.end(), opcode.begin(), ::toupper);
  
  // For comparison operations, appends the comparison type to the opcode.
  if (auto icmp_op = dyn_cast<ICmpOp>(op)) {
    std::string cmp_type = icmp_op.getCmpType().str();
    std::transform(cmp_type.begin(), cmp_type.end(), cmp_type.begin(), ::toupper);
    return opcode + "_" + cmp_type;
  }
  if (auto fcmp_op = dyn_cast<FCmpOp>(op)) {
    std::string cmp_type = fcmp_op.getCmpType().str();
    std::transform(cmp_type.begin(), cmp_type.end(), cmp_type.begin(), ::toupper);
    return opcode + "_" + cmp_type;
  }
  
  // For cast operations, appends the cast type to the opcode.
  if (auto cast_op = dyn_cast<CastOp>(op)) {
    std::string cast_type = cast_op.getCastType().str();
    std::transform(cast_type.begin(), cast_type.end(), cast_type.begin(), ::toupper);
    return opcode + "_" + cast_type;
  }
  
  return opcode;
}

// Extracts constant literal from an attribute.
// Returns formatted string like "#10" or "#3.0", or "arg0" for "%arg0", or empty string if not found.
static std::string extractConstantLiteralFromAttr(Attribute attr) {
  if (!attr) return "";
  
  if (auto integer_attr = dyn_cast<IntegerAttr>(attr))
    return "#" + std::to_string(integer_attr.getInt());
  if (auto float_attr = dyn_cast<FloatAttr>(attr))
    return "#" + std::to_string(float_attr.getValueAsDouble());
  
  // Handles string attributes like "%arg0" -> "arg0".
  if (auto string_attr = dyn_cast<StringAttr>(attr)) {
    std::string value = string_attr.getValue().str();
    // Checks if the string starts with "%arg" followed by digits.
    if (value.size() > 4 && value.substr(0, 4) == "%arg") {
      return value.substr(1);
    }
  }
  
  return "";
}

// Literals for CONSTANT operations, e.g. "#10" / "#0" / "#3.0".
static std::string getConstantLiteral(Operation *op) {
  if (isConstant(op)) {
    if (auto value_attr = op->getAttr("value")) {
      std::string result = extractConstantLiteralFromAttr(value_attr);
      if (!result.empty()) return result;
    }
    return "#0";
  }
  
  // Checks for constant_value attribute in non-CONSTANT operations.
  if (auto constant_value_attr = op->getAttr("constant_value")) {
    std::string result = extractConstantLiteralFromAttr(constant_value_attr);
    if (!result.empty()) return result;
  }
  
  // Checks for rhs_value attribute (for binary operations with constant RHS).
  if (auto rhs_value_attr = op->getAttr("rhs_value")) {
    std::string result = extractConstantLiteralFromAttr(rhs_value_attr);
    if (!result.empty()) return result;
  }
  
  return "";
}

namespace mapping_utils {
// ----- Topology from Architecture -----.
struct Topology {
  DenseMap<int, std::pair<int,int>> link_ends;      // link_id -> (srcTileId, dstTileId).
  DenseMap<int, std::pair<int,int>> tile_location;  // tileId -> (x,y).
  DenseMap<std::pair<int,int>, int> coord_to_tile;  // (x,y) -> tileId.

  StringRef getDirBetween(int src_tile_id, int dst_tile_id) const {
    auto [src_x, src_y] = tile_location.lookup(src_tile_id);
    auto [dst_x, dst_y] = tile_location.lookup(dst_tile_id);
    int dc = dst_x - src_x, dr = dst_y - src_y;
    if (dc == 1 && dr == 0) return "EAST";
    if (dc == -1 && dr == 0) return "WEST";
    if (dc == 0 && dr == 1) return "NORTH";
    if (dc == 0 && dr == -1) return "SOUTH";
    return "LOCAL";
  }
  StringRef dirFromLink(int link_id) const {
    auto it = link_ends.find(link_id);
    if (it == link_ends.end()) return "LOCAL";
    return getDirBetween(it->second.first, it->second.second);
  }
  StringRef invertDir(StringRef d) const {
    if (d == "EAST") return "WEST";
    if (d == "WEST") return "EAST";
    if (d == "NORTH") return "SOUTH";
    if (d == "SOUTH") return "NORTH";
    return "LOCAL";
  }
  int srcTileOfLink(int link_id) const { return link_ends.lookup(link_id).first; }
  int dstTileOfLink(int link_id) const { return link_ends.lookup(link_id).second; }
  int tileIdAt(int x, int y) const {
    auto it = coord_to_tile.find({x,y});
    return (it == coord_to_tile.end()) ? -1 : it->second;
  }
};

static Topology getTopologyFromArchitecture(int per_cgra_rows, int per_cgra_columns) {
  Topology topo;
  mlir::neura::Architecture architecture(1,
                                         1,
                                         mlir::neura::BaseTopology::MESH,
                                         per_cgra_rows,
                                         per_cgra_columns,
                                         mlir::neura::BaseTopology::MESH,
                                         mlir::neura::TileDefaults{},
                                         std::vector<mlir::neura::TileOverride>{},
                                         mlir::neura::LinkDefaults{},
                                         std::vector<mlir::neura::LinkOverride>{});

  for (auto *tile : architecture.getAllTiles()) {
    topo.tile_location[tile->getId()] = {tile->getX(), tile->getY()};
    topo.coord_to_tile[{tile->getX(), tile->getY()}] = tile->getId();
  }
  for (auto *link : architecture.getAllLinks()) {
    auto *src_tile = link->getSrcTile();
    auto *dst_tile = link->getDstTile();
    topo.link_ends[link->getId()] = {src_tile->getId(), dst_tile->getId()};
  }
  return topo;
}

// ----- Extract mapping steps (sorted by time) -----.
struct LinkStep { int link_id; int time_step; };
struct RegStep  { int regId;  int time_step; };

static SmallVector<LinkStep, 8> collectLinkSteps(Operation *op) {
  SmallVector<LinkStep, 8> steps;
  if (auto mapping_locations = getMappingLocations(op)) {
    for (Attribute location_attr : mapping_locations) {
      auto location_dict = dyn_cast<DictionaryAttr>(location_attr);
      if (!location_dict) continue;
      auto resource_attr = dyn_cast_or_null<StringAttr>(location_dict.get("resource"));
      if (!resource_attr || resource_attr.getValue() != "link") continue;
      auto link_id = dyn_cast_or_null<IntegerAttr>(location_dict.get("id"));
      auto time_step = dyn_cast_or_null<IntegerAttr>(location_dict.get("time_step"));
      if (!link_id || !time_step) continue;
      steps.push_back({(int)link_id.getInt(), (int)time_step.getInt()});
    }
  }
  llvm::sort(steps, [](const LinkStep &a, const LinkStep &b){ return a.time_step < b.time_step; });
  return steps;
}

static SmallVector<RegStep, 4> collectRegSteps(Operation *op) {
  SmallVector<RegStep, 4> steps;
  if (auto mapping_locations = getMappingLocations(op)) {
    for (Attribute location_attr : mapping_locations) {
      auto location_dict = dyn_cast<DictionaryAttr>(location_attr);
      if (!location_dict) continue;
      auto resource_attr = dyn_cast_or_null<StringAttr>(location_dict.get("resource"));
      if (!resource_attr) continue;
      if (resource_attr.getValue() == "register" || resource_attr.getValue() == "reg") {
        auto per_tile_register_id = dyn_cast_or_null<IntegerAttr>(location_dict.get("per_tile_register_id"));
        auto time_step = dyn_cast_or_null<IntegerAttr>(location_dict.get("time_step"));
        if (!per_tile_register_id || !time_step) continue;
        steps.push_back({(int)per_tile_register_id.getInt(), (int)time_step.getInt()});
      }
    }
  }
  llvm::sort(steps, [](const RegStep &a, const RegStep &b){ return a.time_step < b.time_step; });
  return steps;
}
} // namespace mapping_utils

// Keep existing call sites stable (byte-identical behavior) by re-exporting the names.
using mapping_utils::Topology;
using mapping_utils::LinkStep;
using mapping_utils::RegStep;
using mapping_utils::collectLinkSteps;
using mapping_utils::collectRegSteps;
using mapping_utils::getMappedRegId;
using mapping_utils::getMappingLocations;
using mapping_utils::getTileLocation;
using mapping_utils::getTopologyFromArchitecture;

// ----- Pass -----.
struct InstructionReference { int col_idx, row_idx, t, idx; };

struct GenerateCodePass
    : public PassWrapper<GenerateCodePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateCodePass)

  StringRef getArgument() const override { return "generate-code"; }
  StringRef getDescription() const override {
    return "CGRA YAML/ASM gen (multi-hop routers + endpoint register deposit + timing-aware rewiring, with CTRL_MOV kept).";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
  }

  DenseMap<Operation*, TileLocation>   operation_placements;

  // Maps of tile coordinates (x,y) -> time_step -> vector of Instructions.
  std::map<std::pair<int,int>, std::map<int, std::vector<Instruction>>> tile_time_instructions;

  // Back references from IR operations to emitted instructions.
  DenseMap<Operation*, InstructionReference>    operation_to_instruction_reference;
  DenseMap<Operation*, SmallVector<Value>>      operation_to_operands;
  // Map dfg_id -> op for later adjustments.
  DenseMap<int, Operation*>                     dfg_id_to_op;
  // Map (col,row,index_per_ii,local_idx_in_bucket) -> global instruction id.
  std::map<std::tuple<int,int,int,int>, int>    instruction_id_map;
  int next_instruction_id = 0;
  int current_compiled_ii = -1;
  bool timing_field_error = false;

  // De-dup sets.
  std::unordered_set<uint64_t> hop_signatures;     // (midTileId, time_step, link_id).
  std::unordered_set<uint64_t> deposit_signatures; // (dstTileId, time_step, regId).
  std::unordered_set<uint64_t> egress_signatures;  // (srcTileId, time_step, regId, out_dir).

  // ---------- helpers to place materialized instructions ----------.
  void placeRouterHop(const Topology &topology, int tile_id, int time_step,
                      StringRef input_direction, StringRef output_direction,
                      bool asCtrlMov = false, int assigned_id = -1) {
    auto [tile_x, tile_y] = topology.tile_location.lookup(tile_id);
    Instruction instruction(asCtrlMov ? "CTRL_MOV" : "DATA_MOV");
    instruction.id = assigned_id;
    instruction.time_step = time_step;
    instruction.index_per_ii = indexPerIiFromTimeStep(time_step);
    instruction.invalid_iterations = syntheticInvalidIterations(time_step);
    instruction.src_operands.emplace_back(input_direction.str(), "RED");
    instruction.dst_operands.emplace_back(output_direction.str(), "RED");
    tile_time_instructions[{tile_x, tile_y}][instruction.index_per_ii].push_back(std::move(instruction));
  }

  // ---------- initialization helpers ----------.
  void clearState() {
    operation_placements.clear();
    tile_time_instructions.clear();
    operation_to_instruction_reference.clear();
    operation_to_operands.clear();
    dfg_id_to_op.clear();
    hop_signatures.clear();
    deposit_signatures.clear();
    egress_signatures.clear();
    instruction_id_map.clear();
    next_instruction_id = 0;
    timing_field_error = false;
  }

  std::pair<int, int> getArrayDimensions(func::FuncOp function) {
    int columns = 4, rows = 4; // default 4x4 CGRA.
    if (auto mapping_info = function->getAttrOfType<DictionaryAttr>("mapping_info")) {
      if (auto x_tiles = dyn_cast_or_null<IntegerAttr>(mapping_info.get("x_tiles"))) columns = x_tiles.getInt();
      if (auto y_tiles = dyn_cast_or_null<IntegerAttr>(mapping_info.get("y_tiles"))) rows   = y_tiles.getInt();
    }
    return {columns, rows};
  }

  int getCompiledII(func::FuncOp function) {
    if (auto mapping_info = function->getAttrOfType<DictionaryAttr>("mapping_info")) {
      if (auto compiled_ii = dyn_cast_or_null<IntegerAttr>(mapping_info.get("compiled_ii"))) {
        return compiled_ii.getInt();
      }
    }
    return -1;
  }

  // Derives index_per_ii from an absolute time_step; falls back to time_step when II is unknown.
  int indexPerIiFromTimeStep(int time_step) const {
    return current_compiled_ii > 0 ? time_step % current_compiled_ii : time_step;
  }

  int syntheticInvalidIterations(int time_step) const {
    return current_compiled_ii > 0 ? time_step / current_compiled_ii : 0;
  }

  // Extracts index_per_ii and invalid_iterations from mapping_locs.
  bool getIndexAndInvalid(Operation *op, int &index_per_ii, int &invalid_iterations) {
    index_per_ii = -1;
    invalid_iterations = 0;
    bool has_index = false, has_invalid = false;
    if (auto arr = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
      for (Attribute a : arr) {
        auto dict = dyn_cast<DictionaryAttr>(a);
        if (!dict) continue;
        if (auto idx_attr = dyn_cast_or_null<IntegerAttr>(dict.get("index_per_ii"))) {
          index_per_ii = idx_attr.getInt();
          has_index = true;
        }
        if (auto inv_attr = dyn_cast_or_null<IntegerAttr>(dict.get("invalid_iterations"))) {
          invalid_iterations = inv_attr.getInt();
          has_invalid = true;
        }
      }
    }
    if (!has_index || !has_invalid) {
      std::string loc_str;
      llvm::raw_string_ostream rso(loc_str);
      rso << op->getLoc();
      rso.flush();

      std::string op_name = op->getName().getStringRef().str();
      std::stringstream errMsg;
      errMsg << "Operation '" << op_name << "' at " << loc_str
             << " missing index_per_ii or invalid_iterations in mapping_locs";
      op->emitError(errMsg.str());
      timing_field_error = true;
      return false;
    }
    return true;
  }

  // ---------- Single-walk indexing ----------.
  // Do everything that needs walks in a single pass:.
  //   - record operation_placements.
  //   - materialize compute/phi/const instructions.
  //   - collect DATA_MOV and CTRL_MOV ops.
  //   - collect reserve_to_phi_maps (PHI's operand#0 is the reserve).
  void indexIR(func::FuncOp function,
               SmallVector<Operation*> &data_movs,
               SmallVector<Operation*> &ctrl_movs,
               DenseMap<Value, Operation*> &reserve_to_phi_map) {
    function.walk([&](Operation *op) {
      // placement for every op (even for mov/reserve).
      operation_placements[op] = getTileLocation(op);

      // build reserve -> phi mapping.
      if (isPhiStart(op)) {
        if (Value reserve = getReserveOperand(op)) {
          reserve_to_phi_map[reserve] = op;
        }
      }

      // collect forwarders.
      if (isDataMov(op)) { data_movs.push_back(op); return; }
      if (isCtrlMov(op)) { ctrl_movs.push_back(op); return; }

      // skip Reserve from materialization.
      if (isReserve(op)) return;

      // materialize all other ops placed on tiles (compute/phi/const/etc.).
      TileLocation placement = operation_placements[op];
      if (!placement.has_tile) return;

      std::string opcode = getOpcode(op);
      Instruction inst(opcode);
      inst.id = getDfgId(op);
      inst.time_step = placement.time_step;

      if (isConstant(op)) {
        inst.src_operands.emplace_back(getConstantLiteral(op), "RED");
      } else if (op->getAttr("constant_value")) {
        // Checks if operation has constant_value attribute (for non-CONSTANT operations).
        inst.src_operands.emplace_back(getConstantLiteral(op), "RED");
      } else {
        // Handles normal operands, including operations with rhs_value attribute.
        SmallVector<Value> operands; operands.reserve(op->getNumOperands());
        
        // Processes actual Value operands (if any).
        for (Value v : op->getOperands()) {
          operands.push_back(v);
          inst.src_operands.emplace_back("UNRESOLVED", "RED");
        }
        
        // Handles cases where binary operations have the RHS constant stored as an attribute.
        if (auto rhs_value_attr = op->getAttr("rhs_value")) {
          std::string rhs_literal = extractConstantLiteralFromAttr(rhs_value_attr);
          if (!rhs_literal.empty()) {
            inst.src_operands.emplace_back(rhs_literal, "RED");
          }
        }
        
        operation_to_operands[op] = std::move(operands);
      }

      if (auto mapped_register_id = getMappedRegId(op))
        inst.dst_operands.emplace_back("$" + std::to_string(*mapped_register_id), "RED");

      int index_per_ii = -1, invalid_iterations = 0;
      if (!getIndexAndInvalid(op, index_per_ii, invalid_iterations)) return;

      inst.index_per_ii = index_per_ii;
      inst.invalid_iterations = invalid_iterations;

      auto &bucket = getInstructionBucket(placement.col_idx, placement.row_idx, index_per_ii);
      bucket.push_back(std::move(inst));
      operation_to_instruction_reference[op] =
          InstructionReference{placement.col_idx, placement.row_idx, index_per_ii,
                               (int)bucket.size() - 1};
    });
  }

  // ---------- unified forwarder expansion helpers ----------.
  static SmallVector<LinkStep, 8> getLinkChain(Operation *forwarder) { return collectLinkSteps(forwarder); }
  static SmallVector<RegStep, 4>  getRegisterSteps(Operation *forwarder) { return collectRegSteps(forwarder); }

  // Validates forwarder op arities: DATA_MOV: at least 1 in/1 out; CTRL_MOV: at least 2 inputs (src,reserve).
  template<bool IsCtrl>
  bool validateForwarderShape(Operation *forwarder) {
    if constexpr (!IsCtrl) {
      return forwarder->getNumOperands() >= 1 && forwarder->getNumResults() >= 1;
    } else {
      return forwarder->getNumOperands() >= 2;
    }
  }

  // Computes producer first-hop directions and consumer last-hop directions (or LOCAL if link-less).
  std::pair<StringRef, StringRef> computeDirections(const SmallVector<LinkStep, 8> &links, const Topology &topo) {
    StringRef producer_direction("LOCAL");
    StringRef consumer_direction("LOCAL");
    if (!links.empty()) {
      producer_direction = topo.dirFromLink(links.front().link_id);
      consumer_direction = topo.invertDir(topo.dirFromLink(links.back().link_id));
    }
    return {producer_direction, consumer_direction};
  }

  // Adds producer endpoints:
  // - If src_reg_step exists: producer writes ONLY to $src_reg (directional port is emitted by egress at first_link_ts).
  // - Else: producer writes to producer_direction if link-based, or $reg for reg-only paths.
  void setProducerDestination(Operation *producer, StringRef producer_direction,
                              const SmallVector<RegStep, 4> &regs,
                              std::optional<RegStep> src_reg_step) {
    if (auto *pi = getInstructionPointer(producer)) {
      if (src_reg_step) {
        // This mov uses src_reg + egress to send on link.time_step. We still must NOT
        // delete existing directional outputs, because the same producer may
        // fan-out to other consumers via other mov paths.
        setUniqueDestination(pi, "$" + std::to_string(src_reg_step->regId));
        return;
      }

      if (!producer_direction.empty() && producer_direction != "LOCAL")
        setUniqueDestination(pi, producer_direction.str());
      else if (!regs.empty())
        setUniqueDestination(pi, "$" + std::to_string(regs.back().regId));
    }
  }

  // Egress: on source tile at first_link_ts, move [$src_reg] -> [out_dir].
  void placeSrcEgress(const Topology &topology, int src_tile_id, int time_step,
                      StringRef out_dir, int reg_id,
                      bool asCtrlMov = false, int assigned_id = -1) {
    // Signature must be stable and support arbitrary direction strings (arch-spec dependent).
    uint64_t signature =
        static_cast<uint64_t>(llvm::hash_combine(src_tile_id, time_step, reg_id, out_dir));
    if (!egress_signatures.insert(signature).second) return;

    auto [tile_x, tile_y] = topology.tile_location.lookup(src_tile_id);
    Instruction inst(asCtrlMov ? "CTRL_MOV" : "DATA_MOV");
    inst.id = assigned_id;
    inst.time_step = time_step;
    inst.index_per_ii = indexPerIiFromTimeStep(time_step);
    inst.invalid_iterations = syntheticInvalidIterations(time_step);
    inst.src_operands.emplace_back("$" + std::to_string(reg_id), "RED");
    inst.dst_operands.emplace_back(out_dir.str(), "RED");
    tile_time_instructions[{tile_x, tile_y}][inst.index_per_ii].push_back(std::move(inst));
  }

  struct MovRegSplit {
    std::optional<RegStep> src_reg_step; // last reg with time_step < first_link_time_step
    std::optional<RegStep> dst_reg_step; // first reg with time_step > last_link_time_step
  };

  MovRegSplit splitMovRegs(const SmallVector<RegStep, 4> &regs,
                           const SmallVector<LinkStep, 8> &links) const {
    MovRegSplit out;
    if (regs.empty() || links.empty()) return out;
    int first_link_time_step = links.front().time_step;
    int last_link_time_step = links.back().time_step;
    for (const RegStep &r : regs) {
      if (r.time_step < first_link_time_step) out.src_reg_step = r;
      if (!out.dst_reg_step && r.time_step > last_link_time_step) out.dst_reg_step = r;
    }
    return out;
  }

  // Emits router hops for multi-hop paths (from the second hop onwards). CTRL_MOV emits CTRL_MOV hops.
  template<bool IsCtrl>
  void generateIntermediateHops(const SmallVector<LinkStep, 8> &links, const Topology &topo,
                                int base_mov_id, size_t &hop_counter) {
    for (size_t i = 1; i < links.size(); ++i) {
      int prev_link = links[i - 1].link_id;
      int cur_link  = links[i].link_id;
      int time_step = links[i].time_step;

      int mid_tile = topo.srcTileOfLink(cur_link);
      StringRef in  = topo.invertDir(topo.dirFromLink(prev_link));
      StringRef out = topo.dirFromLink(cur_link);

      uint64_t sig = static_cast<uint64_t>(llvm::hash_combine(mid_tile, time_step, cur_link));
      if (hop_signatures.insert(sig).second) {
        int hop_id = base_mov_id >= 0 ? base_mov_id * 10000 + static_cast<int>(hop_counter) : -1;
        ++hop_counter;
        placeRouterHop(topo, mid_tile, time_step, in, out, /*asCtrlMov=*/IsCtrl, hop_id);
      }
    }
  }

  // Consumers for DATA_MOV: all users of forwarder results(0).
  SmallVector<std::pair<Operation*, Value>, 2> collectDataMovConsumers(Operation *forwarder) {
    SmallVector<std::pair<Operation*, Value>, 2> consumers;
    Value out = forwarder->getResult(0);
    for (OpOperand &use : out.getUses())
      consumers.push_back({use.getOwner(), use.get()});
    return consumers;
  }

  // Consumers for CTRL_MOV: find PHI via reserve->phi maps; wire the PHI's *data* inputs (sources).
  SmallVector<std::pair<Operation*, Value>, 2> collectCtrlMovConsumers(Operation *forwarder,
                                                                      const DenseMap<Value, Operation*> &reserve2phi) {
    SmallVector<std::pair<Operation*, Value>, 2> consumers;
    Value reserve = forwarder->getOperand(1);
    Value source  = forwarder->getOperand(0);
    if (Operation *phi = reserve2phi.lookup(reserve))
      consumers.push_back({phi, source});
    else
      forwarder->emitWarning("ctrl_mov dest is not consumed by a PHI operand#0; skipping.");
    return consumers;
  }

  // Try register-based rewiring. If cross-tile, emit deposits [incoming_dir]->[$reg] at earliest reg time_step.
  // Returns true if rewiring to $reg was applied to consumers.
  template<bool IsCtrl>
  bool handleRegisterRewiring(Operation *consumer_operation, Value value_at_consumer,
                              const SmallVector<RegStep, 4> &regs,
                              std::optional<RegStep> dst_reg_step,
                              const SmallVector<LinkStep, 8> &links, const Topology &topo, int mov_dfg_id) {
    if (!links.empty()) {
      // Cross-tile: deposit on destination tile at dst_reg_step (post-link register).
      if (!dst_reg_step) return false;
      int deposit_time_step = dst_reg_step->time_step;
      int register_id = dst_reg_step->regId;

      int dst_tile = topo.dstTileOfLink(links.back().link_id);
      // Computes incoming direction from destination tile's perspective.
      StringRef incoming_dir = topo.invertDir(topo.dirFromLink(links.back().link_id));
      placeDstDeposit(topo, dst_tile, deposit_time_step, incoming_dir, register_id,
                      /*asCtrlMov=*/IsCtrl, mov_dfg_id);

      TileLocation consumer_placement = operation_placements.lookup(consumer_operation);
      if (consumer_placement.has_tile && consumer_placement.time_step > deposit_time_step) {
        setConsumerSourceExact(consumer_operation, value_at_consumer, "$" + std::to_string(register_id));
        return true;
      }
    } else {
      // Same-tile: must go via register.
      if (regs.empty()) return false;
      int register_id = regs.back().regId;
      setConsumerSourceExact(consumer_operation, value_at_consumer, "$" + std::to_string(register_id));
      return true;
    }
    return false;
  }

  template<bool IsCtrl>
  void handleDirectionRewiring(Operation *consumer_operation, Value value_at_consumer, StringRef consumer_direction,
                               const SmallVector<LinkStep, 8> &links, const Topology &topo,
                               Operation *forwarder) {
    if (!links.empty()) {
      // Computes the direction from the link destination tile to the consumer tile.
      TileLocation consumer_placement = operation_placements.lookup(consumer_operation);
      if (consumer_placement.has_tile) {
        int dst_tile_id = topo.dstTileOfLink(links.back().link_id);
        int consumer_tile_id = topo.tileIdAt(consumer_placement.col_idx, consumer_placement.row_idx);
        
        // If consumer is on the link destination tile, use the incoming direction.
        if (consumer_tile_id == dst_tile_id) {
          setConsumerSourceExact(consumer_operation, value_at_consumer, consumer_direction.str());
        } else {
          // Computes direction from link destination tile to consumer tile.
          StringRef actual_dir = topo.invertDir(topo.getDirBetween(dst_tile_id, consumer_tile_id));
          setConsumerSourceExact(consumer_operation, value_at_consumer, actual_dir.str());
        }
      } else {
        // Falls back to consumer_direction if consumer placement is unknown.
        setConsumerSourceExact(consumer_operation, value_at_consumer, consumer_direction.str());
      }
    } else {
      forwarder->emitError(IsCtrl
          ? "same-tile ctrl_mov without register mapping is illegal. Provide a register in mapping_locs."
          : "same-tile data_mov without register mapping is illegal. Provide a register in mapping_locs.");
      assert(false && "same-tile mov without register mapping");
    }
  }

  template<bool IsCtrl>
  struct MovBasics {
    int mov_dfg_id = -1;
    Operation *producer = nullptr;
    SmallVector<LinkStep, 8> links;
    SmallVector<RegStep, 4> regs;
    MovRegSplit reg_split;
    StringRef producer_direction;
    StringRef consumer_direction;
  };

  template<bool IsCtrl>
  MovBasics<IsCtrl> buildMovBasics(Operation *forwarder, const Topology &topo) {
    MovBasics<IsCtrl> basics;
    basics.mov_dfg_id = getDfgId(forwarder);

    // Basic info from forwarders.
    Value source = forwarder->getOperand(0);
    basics.producer = source.getDefiningOp();
    basics.links = getLinkChain(forwarder);
    basics.regs = getRegisterSteps(forwarder);
    basics.reg_split = splitMovRegs(basics.regs, basics.links);
    std::pair<StringRef, StringRef> directions = computeDirections(basics.links, topo);
    basics.producer_direction = directions.first;
    basics.consumer_direction = directions.second;
    return basics;
  }

  template<bool IsCtrl>
  void emitMovRoutingInstructions(Operation *forwarder, const MovBasics<IsCtrl> &basics,
                                 const Topology &topo) {
    // Producer endpoints & intermediate hops.
    setProducerDestination(basics.producer, basics.producer_direction, basics.regs, basics.reg_split.src_reg_step);
    if (!basics.links.empty() && basics.reg_split.src_reg_step) {
      int src_tile = topo.srcTileOfLink(basics.links.front().link_id);
      int first_link_time_step = basics.links.front().time_step;
      int egress_instruction_id = basics.mov_dfg_id >= 0 ? basics.mov_dfg_id * 10000 : -1;
      placeSrcEgress(topo, src_tile, first_link_time_step, basics.producer_direction,
                     basics.reg_split.src_reg_step->regId,
                     /*asCtrlMov=*/IsCtrl, egress_instruction_id);
    }
    size_t hop_counter = 1;
    generateIntermediateHops<IsCtrl>(basics.links, topo, basics.mov_dfg_id, hop_counter);
  }

  template<bool IsCtrl>
  SmallVector<std::pair<Operation*, Value>, 2>
  collectMovConsumers(Operation *forwarder, const DenseMap<Value, Operation*> &reserve2phi) {
    if constexpr (IsCtrl) {
      return collectCtrlMovConsumers(forwarder, reserve2phi);
    } else {
      return collectDataMovConsumers(forwarder);
    }
  }

  template<bool IsCtrl>
  void rewriteMovConsumers(Operation *forwarder,
                           const MovBasics<IsCtrl> &basics,
                           const SmallVector<std::pair<Operation*, Value>, 2> &consumers,
                           const Topology &topo) {
    // Wires each consumer: prefer register rewiring; fallback to direction rewiring.
    for (const std::pair<Operation*, Value> &consumer_pair : consumers) {
      Operation *consumer_operation = consumer_pair.first;
      Value value_at_consumer = consumer_pair.second;
      if (!handleRegisterRewiring<IsCtrl>(consumer_operation, value_at_consumer, basics.regs,
                                          basics.reg_split.dst_reg_step,
                                          basics.links, topo, basics.mov_dfg_id))
        handleDirectionRewiring<IsCtrl>(consumer_operation, value_at_consumer, basics.consumer_direction,
                                        basics.links, topo, forwarder);
    }
  }

  template<bool IsCtrl>
  void expandMovImpl(Operation *forwarder, const Topology &topo,
                     const DenseMap<Value, Operation*> &reserve2phi) {
    if (!validateForwarderShape<IsCtrl>(forwarder)) return;

    MovBasics<IsCtrl> basics = buildMovBasics<IsCtrl>(forwarder, topo);

    emitMovRoutingInstructions<IsCtrl>(forwarder, basics, topo);

    SmallVector<std::pair<Operation*, Value>, 2> consumers =
        collectMovConsumers<IsCtrl>(forwarder, reserve2phi);
    if constexpr (IsCtrl) {
      if (consumers.empty()) return;
    }

    rewriteMovConsumers<IsCtrl>(forwarder, basics, consumers, topo);
  }


  // ---------- output generation ----------.
  void logUnresolvedOperands() {
    unsigned unsrc = 0, undst = 0;
    for (auto &tile_entry : tile_time_instructions) {
      std::pair<int,int> tile_key = tile_entry.first;
      int column = tile_key.first, row = tile_key.second;
      for (auto &timestep_entry : tile_entry.second) {
        int time_step = timestep_entry.first;
        std::vector<Instruction> &vec = timestep_entry.second;
        for (size_t i = 0; i < vec.size(); ++i) {
          Instruction &inst = vec[i];
          for (size_t si = 0; si < inst.src_operands.size(); ++si) {
            Operand &s = inst.src_operands[si];
            if (s.operand == "UNRESOLVED") {
              s.color = "ERROR"; ++unsrc;
              llvm::errs() << "[UNRESOLVED SRC] tile("<<column<<","<<row<<") t="<<time_step
                           << " inst#" << i << " op=" << inst.opcode
                           << " src_idx=" << si << "\n";
            }
          }
          inst.dst_operands.erase(
            std::remove_if(inst.dst_operands.begin(), inst.dst_operands.end(),
              [](const Operand &o){ return o.operand.empty() || o.operand=="UNKNOWN"; }),
            inst.dst_operands.end());
          for (size_t di = 0; di < inst.dst_operands.size(); ++di) {
            Operand &d = inst.dst_operands[di];
            if (d.operand == "UNRESOLVED") {
              d.color = "ERROR"; ++undst;
              llvm::errs() << "[UNRESOLVED DST] tile("<<column<<","<<row<<") t="<< time_step
                           << " inst#" << i << " op=" << inst.opcode
                           << " dst_idx=" << di << "\n";
            }
          }
        }
      }
    }
    if (unsrc + undst) {
      ModuleOp module = getOperation();
      auto diag = module.emitWarning("GenerateCodePass: UNRESOLVED operands kept for debugging");
      diag << " (src=" << unsrc << ", dst=" << undst << "); they are highlighted with color=ERROR in YAML.";
    }
  }

  // Assigns unique IDs to all materialized instructions (including data/ctrl mov hops).
  void assignInstructionIds(std::unordered_set<int> &materialized_ids) {
    instruction_id_map.clear();
    int max_assigned = -1;
    for (auto &[tile_key, timestep_map] : tile_time_instructions) {
      for (auto &[time_step, inst_vec] : timestep_map) {
        for (Instruction &inst : inst_vec) {
          if (inst.id >= 0) max_assigned = std::max(max_assigned, inst.id);
        }
      }
    }
    next_instruction_id = max_assigned + 1;
    for (auto &[tile_key, timestep_map] : tile_time_instructions) {
      int col = tile_key.first;
      int row = tile_key.second;
      for (auto &[time_step, inst_vec] : timestep_map) {
        for (size_t idx = 0; idx < inst_vec.size(); ++idx) {
          Instruction &inst = inst_vec[idx];
          if (inst.id < 0) inst.id = next_instruction_id++;
          instruction_id_map[{col, row, time_step, (int)idx}] = inst.id;
          materialized_ids.insert(inst.id);
        }
      }
    }
  }

  // Looks up instruction ID by InstructionReference (col,row,time_step,idx in bucket).
  int lookupInstructionId(const InstructionReference &ref) const {
    auto it = instruction_id_map.find({ref.col_idx, ref.row_idx, ref.t, ref.idx});
    if (it == instruction_id_map.end()) return -1;
    return it->second;
  }

  // Helper to escape strings for DOT/JSON.
  static std::string escape(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
      if (c == '"') out += "\\\"";
      else if (c == '\\') out += "\\\\";
      else if (c == '\n') out += "\\n";
      else out += c;
    }
    return out;
  }

  // Helper to extract dfg_id from operation.
  static int getDfgId(Operation *op) {
    if (auto id_attr = op->getAttrOfType<IntegerAttr>("dfg_id")) {
      return id_attr.getInt();
    }
    return -1;
  }

  // Helper to extract tile coordinates and time_step from mapping_locs.
  struct LocationInfo {
    int tile_x = -1;
    int tile_y = -1;
    int time_step = -1;
    int index_per_ii = -1;
    int invalid_iterations = 0;
    bool has_tile = false;
  };

  static LocationInfo getLocationInfo(Operation *op) {
    LocationInfo info;
    if (auto arr = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
      for (Attribute a : arr) {
        auto d = dyn_cast<DictionaryAttr>(a);
        if (!d) continue;
        auto resource = dyn_cast_or_null<StringAttr>(d.get("resource"));
        
        // Extracts time_step from any resource type.
        if (auto ts_attr = dyn_cast_or_null<IntegerAttr>(d.get("time_step"))) {
          info.time_step = ts_attr.getInt();
        }
        if (auto index_attr = dyn_cast_or_null<IntegerAttr>(d.get("index_per_ii"))) {
          info.index_per_ii = index_attr.getInt();
        }
        if (auto invalid_attr = dyn_cast_or_null<IntegerAttr>(d.get("invalid_iterations"))) {
          info.invalid_iterations = invalid_attr.getInt();
        }
        
        // Only tile resources have x/y coordinates.
        if (resource && resource.getValue() == "tile") {
          if (auto x_attr = dyn_cast_or_null<IntegerAttr>(d.get("x"))) {
            info.tile_x = x_attr.getInt();
          }
          if (auto y_attr = dyn_cast_or_null<IntegerAttr>(d.get("y"))) {
            info.tile_y = y_attr.getInt();
          }
          info.has_tile = true;
        }
      }
    }
    return info;
  }

  struct DfgNodeInfo {
    std::string opcode;
    int tile_x = -1;
    int tile_y = -1;
    int time_step = -1;
  };
  using DfgNodeMap = std::map<int, DfgNodeInfo>;
  using DfgEdgeList = std::vector<std::pair<int, int>>;

  struct HopRewriteInfo {
    int mov_id = -1;
    int producer_id = -1;
    SmallVector<int, 4> consumer_ids;
    SmallVector<std::pair<int, int>, 8> hop_tiles;
    SmallVector<int, 8> hop_time_steps;
  };

  void collectCtrlMovReserves(func::FuncOp func,
                              DenseMap<Value, SmallVector<Operation *, 2>> &reserve_to_ctrl_movs) const {
    func.walk([&](Operation *operation) {
      if (auto ctrl_mov_op = dyn_cast<CtrlMovOp>(operation)) {
        if (ctrl_mov_op->getNumOperands() >= 2) {
          Value reserve_value = ctrl_mov_op->getOperand(1);
          reserve_to_ctrl_movs[reserve_value].push_back(operation);
        }
      }
    });
  }

  void addEdge(DfgEdgeList &edges, int src, int dst) const {
    if (src < 0 || dst < 0) return;
    if (src == dst) return; // Avoids self-loop.
    edges.emplace_back(src, dst);
  }

  void buildSsaNodesAndEdges(func::FuncOp func,
                             const DenseMap<Value, SmallVector<Operation *, 2>> &reserve_to_ctrl_movs,
                             DfgNodeMap &nodes,
                             DfgEdgeList &edges) {
    func.walk([&](Operation *operation) {
      if (operation == func.getOperation()) return;  // Skips function itself.
      if (isReserve(operation)) return; // Skips reserve nodes entirely (bypass later).
      if (isa<YieldOp>(operation)) return; // Skips yield nodes entirely (bypass later).

      int dfg_id = getDfgId(operation);
      if (dfg_id < 0) {
        llvm::errs() << "[WARN] Operation without dfg_id: " << *operation << "\n";
        return;
      }
      dfg_id_to_op[dfg_id] = operation;

      std::string opcode = getOpcode(operation);
      LocationInfo location_info = getLocationInfo(operation);

      if ((isDataMov(operation) || isCtrlMov(operation)) && !location_info.has_tile) {
        nodes[dfg_id] = DfgNodeInfo{opcode, -1, -1, location_info.time_step};
      } else {
        nodes[dfg_id] = DfgNodeInfo{opcode, location_info.tile_x, location_info.tile_y, location_info.time_step};
      }

      for (Value operand : operation->getOperands()) {
        Operation *producer_op = operand.getDefiningOp();
        if (producer_op && isReserve(producer_op)) {
          auto it = reserve_to_ctrl_movs.find(operand);
          if (it != reserve_to_ctrl_movs.end()) {
            for (Operation *ctrl_mov_op : it->second) {
              if (ctrl_mov_op == operation) continue;
              int producer_id = getDfgId(ctrl_mov_op);
              if (producer_id >= 0) addEdge(edges, producer_id, dfg_id);
            }
          }
          continue;
        }
        if (producer_op) {
          int producer_id = getDfgId(producer_op);
          addEdge(edges, producer_id, dfg_id);
        }
      }
    });
  }

  void collectHopRewrites(func::FuncOp func, const Topology &topology,
                          DfgNodeMap &nodes,
                          const DfgEdgeList &original_edges,
                          DfgEdgeList &edges,
                          llvm::SmallDenseSet<std::pair<int,int>, 32> &skip_edges,
                          std::vector<HopRewriteInfo> &rewrites) {
    func.walk([&](Operation *operation) {
      if (!isDataMov(operation) && !isCtrlMov(operation)) return;
      int mov_dfg_id = getDfgId(operation);
      if (mov_dfg_id < 0) return;

      int producer_id = -1;
      if (operation->getNumOperands() >= 1) {
        if (Operation *producer = operation->getOperand(0).getDefiningOp())
          producer_id = getDfgId(producer);
      }

      SmallVector<int, 4> consumer_ids;
      if (operation->getNumResults() >= 1) {
        for (OpOperand &use : operation->getResult(0).getUses()) {
          Operation *user = use.getOwner();
          while (user && isReserve(user)) {
            if (user->getNumResults() == 0) { user = nullptr; break; }
            bool forwarded = false;
            for (OpOperand &reserve_use : user->getResult(0).getUses()) {
              Operation *forward_user = reserve_use.getOwner();
              if (!forward_user) continue;
              int consumer_id = getDfgId(forward_user);
              if (consumer_id >= 0) consumer_ids.push_back(consumer_id);
              forwarded = true;
            }
            user = nullptr;
            if (forwarded) break;
          }
          if (user) {
            int consumer_id = getDfgId(user);
            if (consumer_id >= 0) consumer_ids.push_back(consumer_id);
          }
        }
      }

      SmallVector<LinkStep, 8> link_steps = collectLinkSteps(operation);
      SmallVector<std::pair<int,int>, 8> hop_tiles;
      SmallVector<int, 8> hop_time_steps;
      // Detect whether this mov has a src_reg_step (reg.time_step < first_link_time_step). If so,
      // an egress instruction exists with id=mov_id*10000 and must participate in edges.
      bool has_src_reg_step = false;
      if (!link_steps.empty()) {
        SmallVector<RegStep, 4> regs = collectRegSteps(operation);
        MovRegSplit reg_split = splitMovRegs(regs, link_steps);
        has_src_reg_step = reg_split.src_reg_step.has_value();
        if (has_src_reg_step) {
          int base = mov_dfg_id * 10000;
          int egress_id = base;
          int src_tile_id = topology.srcTileOfLink(link_steps.front().link_id);
          auto coord = topology.tile_location.lookup(src_tile_id);
          // Add placeholder node now (will be filled/confirmed by injectMaterializedInstructionNodes).
          if (!nodes.count(egress_id)) {
            nodes[egress_id] = DfgNodeInfo{
                isCtrlMov(operation) ? "CTRL_MOV" : "DATA_MOV",
                coord.first, coord.second, link_steps.front().time_step};
          }
        }
      }

      // Build hop tiles directly from link steps (mirrors router hop emission).
      if (link_steps.size() > 1) {
        for (size_t i = 1; i < link_steps.size(); ++i) {
          int middle_tile_id = topology.srcTileOfLink(link_steps[i].link_id);
          auto coord = topology.tile_location.lookup(middle_tile_id);
          hop_tiles.push_back(coord);
          hop_time_steps.push_back(link_steps[i].time_step);
        }
      }

      if (hop_tiles.empty()) {
        auto it = nodes.find(mov_dfg_id);
        if (it != nodes.end()) {
          it->second.tile_x = -1;
          it->second.tile_y = -1;
        }
      } else {
        int base = mov_dfg_id * 10000;
        for (size_t i = 0; i < hop_tiles.size(); ++i) {
          int node_id = base + static_cast<int>(i) + 1;
          DfgNodeInfo hop_node;
          hop_node.opcode = isCtrlMov(operation) ? "CTRL_MOV" : "DATA_MOV";
          hop_node.tile_x = hop_tiles[i].first;
          hop_node.tile_y = hop_tiles[i].second;
          hop_node.time_step = (i < hop_time_steps.size()) ? hop_time_steps[i] : -1;
          nodes[node_id] = hop_node;
        }

        if (producer_id >= 0)
          skip_edges.insert({producer_id, mov_dfg_id});
        for (int consumer_id : consumer_ids)
          skip_edges.insert({mov_dfg_id, consumer_id});

        rewrites.push_back(
            HopRewriteInfo{mov_dfg_id, producer_id, consumer_ids, hop_tiles, hop_time_steps});
      }

      // Also rewrite edges for single-hop (no intermediate hop nodes) when egress exists:
      // producer -> egress(base) -> mov_id -> consumers
      // so that DFG edges align with instruction ids.
      if (has_src_reg_step && hop_tiles.empty()) {
        if (producer_id >= 0)
          skip_edges.insert({producer_id, mov_dfg_id});
        for (int consumer_id : consumer_ids)
          skip_edges.insert({mov_dfg_id, consumer_id});
        // Store a rewrite with empty hop list; later we will special-case it.
        rewrites.push_back(HopRewriteInfo{mov_dfg_id, producer_id, consumer_ids, /*hop_tiles*/{}, /*hop_time_steps*/{}});
      }
    });

    for (const auto &edge : original_edges) {
      if (skip_edges.count(edge)) continue;
      edges.push_back(edge);
    }

    for (const HopRewriteInfo &rewrite : rewrites) {
      int base = rewrite.mov_id * 10000;
      int egress_id = base;
      int previous = rewrite.producer_id;

      // If an egress instruction exists (id=mov_id*10000), route through it.
      // This keeps DFG edges aligned with instruction ids for reg-before-link paths.
      if (nodes.count(egress_id)) {
        if (previous >= 0) edges.emplace_back(previous, egress_id);
        previous = egress_id;
      }

      for (size_t i = 0; i < rewrite.hop_tiles.size(); ++i) {
        int node_id = base + static_cast<int>(i) + 1;
        if (previous >= 0) edges.emplace_back(previous, node_id);
        previous = node_id;
      }
      // Always connect the chain to mov_id; if mov_id is later pruned (non-materialized),
      // pruneMovNodesWithoutCoords will bypass it and connect previous directly to consumers.
      if (previous >= 0) edges.emplace_back(previous, rewrite.mov_id);
      for (int consumer_id : rewrite.consumer_ids)
        edges.emplace_back(rewrite.mov_id, consumer_id);
    }
  }

  void adjustRegisterOnlyMovCoords(DfgNodeMap &nodes, const DfgEdgeList &edges) {
    std::unordered_map<int, std::vector<int>> successors;
    for (const auto &edge : edges) successors[edge.first].push_back(edge.second);

    auto scanMappingKinds = [](Operation *op, bool &has_link, bool &has_register, int &min_register_ts) {
      has_link = false; has_register = false; min_register_ts = -1;
      if (auto arr = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
        for (Attribute attr : arr) {
          auto dict = dyn_cast<DictionaryAttr>(attr);
          if (!dict) continue;
          auto res = dyn_cast_or_null<StringAttr>(dict.get("resource"));
          if (!res) continue;
          if (res.getValue() == "link") {
            has_link = true;
          } else if (res.getValue() == "register" || res.getValue() == "reg") {
            has_register = true;
            if (auto ts_attr = dyn_cast_or_null<IntegerAttr>(dict.get("time_step"))) {
              int time_step = ts_attr.getInt();
              if (min_register_ts < 0 || time_step < min_register_ts) min_register_ts = time_step;
            }
          }
        }
      }
    };

    for (auto &entry : nodes) {
      int node_id = entry.first;
      DfgNodeInfo &node = entry.second;
      if (node.tile_x != -1 || node.tile_y != -1) continue;
      if (node.opcode != "DATA_MOV" && node.opcode != "CTRL_MOV") continue;

      Operation *op = dfg_id_to_op.lookup(node_id);
      if (!op) continue;

      bool has_link = false, has_register = false;
      int min_register_ts = -1;
      scanMappingKinds(op, has_link, has_register, min_register_ts);
      if (!has_register) continue; // if register not present, keep as is; if present, fill coords from consumer

      auto succ_it = successors.find(node_id);
      if (succ_it == successors.end()) continue;
      for (int consumer_id : succ_it->second) {
        auto consumer_it = nodes.find(consumer_id);
        if (consumer_it == nodes.end()) continue;
        const DfgNodeInfo &consumer_node = consumer_it->second;
        if (consumer_node.tile_x == -1 || consumer_node.tile_y == -1) continue;
        node.tile_x = consumer_node.tile_x;
        node.tile_y = consumer_node.tile_y;
        if (node.time_step < 0 && min_register_ts >= 0) node.time_step = min_register_ts;
        break;
      }
    }
  }

  void pruneMovNodesWithoutCoords(DfgNodeMap &nodes, DfgEdgeList &edges,
                                  const std::unordered_set<int> &materialized_ids) {
    std::unordered_set<int> nodes_to_remove;
    for (const auto &entry : nodes) {
      int node_id = entry.first;
      const DfgNodeInfo &node = entry.second;
      if ((node.opcode == "DATA_MOV" || node.opcode == "CTRL_MOV") &&
          (node.tile_x == -1 && node.tile_y == -1)) {
        nodes_to_remove.insert(node_id);
      }
      if ((node.opcode == "DATA_MOV" || node.opcode == "CTRL_MOV") &&
          !materialized_ids.count(node_id)) {
        nodes_to_remove.insert(node_id);
      }
    }
    if (nodes_to_remove.empty()) return;

    std::unordered_map<int, std::vector<int>> predecessors, successors;
    for (const auto &edge : edges) {
      successors[edge.first].push_back(edge.second);
      predecessors[edge.second].push_back(edge.first);
    }
    std::unordered_set<uint64_t> dedup_edge_set;
    auto encode = [](int from, int to)->uint64_t { return (static_cast<uint64_t>(from) << 32) ^ static_cast<uint32_t>(to); };

    std::vector<std::pair<int,int>> new_edges;
    for (const auto &edge : edges) {
      if (nodes_to_remove.count(edge.first) || nodes_to_remove.count(edge.second))
        continue;
      uint64_t key = encode(edge.first, edge.second);
      if (dedup_edge_set.insert(key).second) new_edges.push_back(edge);
    }

    for (int removed : nodes_to_remove) {
      const auto &preds = predecessors[removed];
      const auto &succs = successors[removed];
      for (int pred : preds) {
        for (int succ : succs) {
          if (pred == succ) continue;
          uint64_t key = encode(pred, succ);
          if (dedup_edge_set.insert(key).second)
            new_edges.emplace_back(pred, succ);
        }
      }
    }
    edges.swap(new_edges);
    for (int removed : nodes_to_remove) nodes.erase(removed);
  }

  void emitDotOutput(const DfgNodeMap &nodes, const DfgEdgeList &edges) const {
    std::error_code ec;
    llvm::raw_fd_ostream dot_out("tmp-generated-dfg.dot", ec);
    if (ec) return;

    dot_out << "digraph DFG {\n  rankdir=LR;\n  node [shape=box, style=filled];\n";
    auto color_for = [](const std::string &op) {
      if (op == "DATA_MOV") return "lightgreen";
      if (op == "CTRL_MOV") return "lightyellow";
      if (op == "CONSTANT") return "lightblue";
      return "white";
    };
    for (const auto &entry : nodes) {
      int id = entry.first;
      const DfgNodeInfo &node = entry.second;
      dot_out << "  n" << id << " [label=\""
              << escape(node.opcode) << "\\nID=" << id;
      if (node.tile_x >= 0 && node.tile_y >= 0) {
        dot_out << "\\n(" << node.tile_x << "," << node.tile_y << ")";
      }
      if (node.time_step >= 0) {
        dot_out << " t=" << node.time_step;
      }
      dot_out << "\", fillcolor=" << color_for(node.opcode) << "];\n";
    }
    for (const auto &edge : edges) {
      dot_out << "  n" << edge.first << " -> n" << edge.second << ";\n";
    }
    dot_out << "}\n";
  }

  void emitYamlOutput(const DfgNodeMap &nodes, const DfgEdgeList &edges) const {
    std::error_code ec;
    llvm::raw_fd_ostream yaml_out("tmp-generated-dfg.yaml", ec);
    if (ec) return;

    yaml_out << "nodes:\n";
    for (const auto &entry : nodes) {
      int id = entry.first;
      const DfgNodeInfo &node = entry.second;
      yaml_out << "  - id: " << id << "\n"
               << "    opcode: \"" << escape(node.opcode) << "\"\n"
               << "    tile_x: " << node.tile_x << "\n"
               << "    tile_y: " << node.tile_y << "\n"
               << "    time_step: " << node.time_step << "\n";
    }
    yaml_out << "edges:\n";
    for (const auto &edge : edges) {
      yaml_out << "  - from: " << edge.first << "\n"
               << "    to: " << edge.second << "\n";
    }
  }

  // Ensures every materialized instruction (from tmp-generated-instructions.yaml)
  // has a corresponding DFG node keyed by instruction.id. This is important for
  // synthetic instructions that don't exist as IR ops (e.g., egress DATA_MOV with
  // id=mov_id*10000).
  void injectMaterializedInstructionNodes(DfgNodeMap &nodes) const {
    for (const auto &[tile_key, timestep_map] : tile_time_instructions) {
      int tile_x = tile_key.first;
      int tile_y = tile_key.second;
      for (const auto &[idx_per_ii, inst_vec] : timestep_map) {
        (void)idx_per_ii;
        for (const Instruction &inst : inst_vec) {
          if (inst.id < 0) continue;
          // Only inject if absent; for IR-backed ops (id == dfg_id) the existing
          // node is fine, but injecting is also harmless.
          auto it = nodes.find(inst.id);
          if (it == nodes.end()) {
            nodes[inst.id] = DfgNodeInfo{inst.opcode, tile_x, tile_y, inst.time_step};
          } else {
            // If the existing node has no coords, prefer concrete instruction coords.
            if (it->second.tile_x < 0 || it->second.tile_y < 0) {
              it->second.tile_x = tile_x;
              it->second.tile_y = tile_y;
            }
            if (it->second.time_step < 0 && inst.time_step >= 0)
              it->second.time_step = inst.time_step;
          }
        }
      }
    }
  }

  struct DfgBuilder {
    GenerateCodePass &pass;
    func::FuncOp func;
    const Topology &topology;
    const std::unordered_set<int> &materialized_ids;

    void run() {
      DfgNodeMap nodes;
      DfgEdgeList edges;

      DenseMap<Value, SmallVector<Operation *, 2>> reserve_to_ctrl_movs;
      pass.collectCtrlMovReserves(func, reserve_to_ctrl_movs);

      pass.buildSsaNodesAndEdges(func, reserve_to_ctrl_movs, nodes, edges);

      std::vector<std::pair<int,int>> original_edges = edges;
      edges.clear();
      llvm::SmallDenseSet<std::pair<int,int>, 32> edges_to_skip;
      std::vector<HopRewriteInfo> hop_rewrites;

      pass.collectHopRewrites(func, topology, nodes, original_edges, edges, edges_to_skip, hop_rewrites);

      // Bring in synthetic instruction nodes (egress/deposit/hops) by instruction id.
      pass.injectMaterializedInstructionNodes(nodes);

      pass.adjustRegisterOnlyMovCoords(nodes, edges);

      pass.pruneMovNodesWithoutCoords(nodes, edges, materialized_ids);

      pass.emitDotOutput(nodes, edges);
      pass.emitYamlOutput(nodes, edges);

      llvm::outs() << "[generate-code] DFG (SSA-based) emitted: nodes=" << nodes.size()
                   << ", edges=" << edges.size()
                   << " -> tmp-generated-dfg.dot, tmp-generated-dfg.yaml\n";
    }
  };

  // Writes DOT and JSON DFG outputs based on SSA and dfg_id attributes.
  // Applies hop-aware coordinates/middle-node insertion for DATA_MOV / CTRL_MOV,
  // and bypasses reserve nodes (no node for reserve, edges direct from producer to consumer).
  void writeDfgOutputSSA(func::FuncOp func, const Topology &topology,
                         const std::unordered_set<int> &materialized_ids) {
    DfgBuilder{*this, func, topology, materialized_ids}.run();
  }

  ArrayConfig buildArrayConfig(int columns, int rows, int compiled_ii = -1) {
    ArrayConfig config{columns, rows, compiled_ii, {}};
    std::map<std::pair<int,int>, std::vector<Instruction>> tile_insts;

    // Flattens and sorts by timesteps.
    for (auto &[tile_key, timestep_map] : tile_time_instructions) {
      auto &flat = tile_insts[tile_key];
      for (auto &[timestep, instruction_vec] : timestep_map) for (Instruction &inst : instruction_vec) flat.push_back(inst);
      std::stable_sort(flat.begin(), flat.end(),
        [](const Instruction &a, const Instruction &b){ return a.time_step < b.time_step; });
    }

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < columns; ++c) {
        auto it = tile_insts.find({c,r});
        if (it == tile_insts.end()) continue;
        Tile tile(c, r, r*columns + c);
        for (Instruction &inst : it->second) tile.entry.instructions.push_back(inst);
        config.cores.push_back(std::move(tile));
      }
    }
    return config;
  }

  using IndexGroups = std::map<int, std::vector<const Instruction*>>;

  static IndexGroups groupByIndexPerIi(const std::vector<Instruction> &instructions) {
    IndexGroups index_groups;
    for (const Instruction &inst : instructions)
      index_groups[inst.index_per_ii].push_back(&inst);
    return index_groups;
  }

  void emitYamlForTile(llvm::raw_fd_ostream &yaml_out, const Tile &tile) {
    yaml_out << "    - column: " << tile.col_idx << "\n      row: " << tile.row_idx
             << "\n      core_id: \"" << tile.core_id << "\"\n      entries:\n";

    // Groups instructions by index_per_ii.
    IndexGroups index_groups = groupByIndexPerIi(tile.entry.instructions);

    yaml_out << "        - entry_id: \"entry0\"\n          instructions:\n";
    for (const auto &index_pair : index_groups) {
      int index_per_ii = index_pair.first;
      auto operations = index_pair.second;
      std::stable_sort(operations.begin(), operations.end(),
                       [](const Instruction *a, const Instruction *b) {
                         return a->time_step < b->time_step;
                       });

      yaml_out << "            - index_per_ii: " << index_per_ii << "\n              operations:\n";
      for (const Instruction *inst : operations) {
        yaml_out << "                - opcode: \"" << inst->opcode << "\"\n";
        if (inst->id >= 0)
          yaml_out << "                  id: " << inst->id << "\n";
        yaml_out << "                  time_step: " << inst->time_step << "\n"
                 << "                  invalid_iterations: " << inst->invalid_iterations << "\n";
        // sources.
        if (!inst->src_operands.empty()) {
          yaml_out << "                  src_operands:\n";
          for (const Operand &opnd : inst->src_operands)
            yaml_out << "                    - operand: \"" << opnd.operand << "\"\n                      color: \"" << opnd.color << "\"\n";
        }
        // destinations.
        if (!inst->dst_operands.empty()) {
          yaml_out << "                  dst_operands:\n";
          for (const Operand &opnd : inst->dst_operands)
            yaml_out << "                    - operand: \"" << opnd.operand << "\"\n                      color: \"" << opnd.color << "\"\n";
        }
      }
    }
  }

  void writeYAMLOutput(const ArrayConfig &config) {
    std::error_code ec;
    llvm::raw_fd_ostream yaml_out("tmp-generated-instructions.yaml", ec);
    if (ec) return;

    yaml_out << "array_config:\n  columns: " << config.columns << "\n  rows: " << config.rows;
    if (config.compiled_ii >= 0) {
      yaml_out << "\n  compiled_ii: " << config.compiled_ii;
    }
    yaml_out << "\n  cores:\n";
    for (const Tile &core : config.cores) {
      emitYamlForTile(yaml_out, core);
    }
    yaml_out.close();
  }

  // Direction vs const/reg helpers.
  static bool isDirectionalOperand(const std::string &operand) {
    // Non-directional operands start with $ (registers), # (constants), or arg (function arguments).
    if (operand.empty()) return false;
    if (operand[0] == '$' || operand[0] == '#') return false;
    // Checks if the operand starts with "arg" followed by digits (e.g., "arg0", "arg1").
    if (operand.size() >= 4 && operand.substr(0, 3) == "arg") {
      // Verifies that the rest is digits.
      bool all_digits = true;
      for (size_t i = 3; i < operand.size(); ++i) {
        if (!std::isdigit(operand[i])) {
          all_digits = false;
          break;
        }
      }
      if (all_digits) return false;
    }
    return true;
  }

  static std::string formatOperand(const Operand &operand) {
    std::string result = "[" + operand.operand;
    if (isDirectionalOperand(operand.operand)) {
      result += ", " + operand.color;
    }
    result += "]";
    return result;
  }

  void emitAsmForTile(llvm::raw_fd_ostream &asm_out, const Tile &tile) {
    asm_out << "PE(" << tile.col_idx << "," << tile.row_idx << "):\n";

    // Groups instructions by index_per_ii.
    IndexGroups index_groups = groupByIndexPerIi(tile.entry.instructions);

    for (const auto &index_pair : index_groups) {
      int index_per_ii = index_pair.first;
      auto instructions = index_pair.second;
      std::stable_sort(instructions.begin(), instructions.end(),
                       [](const Instruction *a, const Instruction *b) {
                         return a->time_step < b->time_step;
                       });

      asm_out << "{\n";
      for (size_t i = 0; i < instructions.size(); ++i) {
        const Instruction *inst = instructions[i];
        asm_out << "  " << inst->opcode;
        for (const Operand &operand : inst->src_operands) asm_out << ", " << formatOperand(operand);
        if (!inst->dst_operands.empty()) {
          asm_out << " -> ";
          for (size_t j = 0; j < inst->dst_operands.size(); ++j) {
            if (j > 0) asm_out << ", ";
            asm_out << formatOperand(inst->dst_operands[j]);
          }
        }
        asm_out << " (t=" << inst->time_step
                << ", inv_iters=" << inst->invalid_iterations << ")\n";
      }
      asm_out << "} (idx_per_ii=" << index_per_ii << ")\n";
    }
    asm_out << "\n";
  }

  void writeAsmOutput(const ArrayConfig &config) {
    std::error_code ec;
    llvm::raw_fd_ostream asm_out("tmp-generated-instructions.asm", ec);
    if (ec) return;

    if (config.compiled_ii >= 0) {
      asm_out << "# Compiled II: " << config.compiled_ii << "\n\n";
    }

    for (const Tile &core : config.cores) {
      emitAsmForTile(asm_out, core);
    }
    asm_out.close();
  }

  // Endpoint deposits: on destination tiles at earliest reg time_step, move [incoming_dir] -> [$reg].
  // CTRL_MOV paths emit CTRL_MOV deposits; DATA_MOV paths emit DATA_MOV deposits.
  void placeDstDeposit(const Topology &topo, int dst_tile_id, int time_step,
                       StringRef incoming_dir, int reg_id, bool asCtrlMov = false, int assigned_id = -1) {
    uint64_t signature = static_cast<uint64_t>(llvm::hash_combine(dst_tile_id, time_step, reg_id));
    if (!deposit_signatures.insert(signature).second) return; // already placed.
    auto [tile_x, tile_y] = topo.tile_location.lookup(dst_tile_id);
    Instruction inst(asCtrlMov ? "CTRL_MOV" : "DATA_MOV");
    inst.id = assigned_id;
    inst.time_step = time_step;
    inst.index_per_ii = indexPerIiFromTimeStep(time_step);
    inst.invalid_iterations = syntheticInvalidIterations(time_step);
    inst.src_operands.emplace_back(incoming_dir.str(), "RED");
    inst.dst_operands.emplace_back("$" + std::to_string(reg_id), "RED");
    tile_time_instructions[{tile_x, tile_y}][inst.index_per_ii].push_back(std::move(inst));
  }

  // Utilities to access instruction buckets/pointers.
  std::vector<Instruction> &getInstructionBucket(int column, int row, int time_step) {
    return tile_time_instructions[{column,row}][time_step];
  }
  Instruction* getInstructionPointer(Operation *operation) {
    auto it = operation_to_instruction_reference.find(operation);
    if (it == operation_to_instruction_reference.end()) return nullptr;
    auto [c, r, t, idx] = it->second;
    auto &vec = tile_time_instructions[{c,r}][t];
    if (idx < 0 || idx >= (int)vec.size()) return nullptr;
    return &vec[idx];
  }

  // Replaces the exact source slots in consumers that correspond to `value_at_consumer`,
  // or fills the first UNRESOLVED placeholder if a 1:1 match wasn't found.
  void setConsumerSourceExact(Operation *consumer, Value value_at_consumer, const std::string &text) {
    Instruction *ci = getInstructionPointer(consumer);
    if (!ci) return;
    auto it = operation_to_operands.find(consumer);
    if (it == operation_to_operands.end()) return;
    auto &ops = it->second;
    for (size_t i = 0; i < ops.size() && i < ci->src_operands.size(); ++i) {
      if (ops[i] == value_at_consumer) { ci->src_operands[i].operand = text; return; }
    }
    for (Operand &src : ci->src_operands)
      if (src.operand == "UNRESOLVED") { src.operand = text; return; }
  }

  // Appends a destination only once.
  static void setUniqueDestination(Instruction *inst, const std::string &text){
    for (Operand &d : inst->dst_operands) if (d.operand == text) return;
    inst->dst_operands.emplace_back(text, "RED");
  }

  // ---------- entry point ----------.
  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (auto func : module.getOps<func::FuncOp>()) {
      auto accel = func->getAttrOfType<StringAttr>("accelerator");
      if (!accel || accel.getValue() != "neura") continue;

      auto [columns, rows] = getArrayDimensions(func);
      Topology topo = getTopologyFromArchitecture(columns, rows);
      current_compiled_ii = getCompiledII(func);

      clearState();

      // Single function-level walks: index + materialize + collect.
      SmallVector<Operation*> data_movs;
      SmallVector<Operation*> ctrl_movs;
      DenseMap<Value, Operation*> reserve_to_phi_map;
      indexIR(func, data_movs, ctrl_movs, reserve_to_phi_map);

      // Expands forwarders without re-walking IR.
      for (Operation *op : data_movs)
        expandMovImpl<false>(op, topo, /*unused*/reserve_to_phi_map);
      for (Operation *op : ctrl_movs)
        expandMovImpl<true>(op,  topo, reserve_to_phi_map);
      logUnresolvedOperands();

      std::unordered_set<int> materialized_ids;
      assignInstructionIds(materialized_ids);
      ArrayConfig config = buildArrayConfig(columns, rows, current_compiled_ii);
      writeYAMLOutput(config);
      writeAsmOutput(config);
      writeDfgOutputSSA(func, topo, materialized_ids);
      if (timing_field_error) signalPassFailure();
    }
  }
};

} // namespace.

namespace mlir::neura {
std::unique_ptr<Pass> createGenerateCodePass() {
  return std::make_unique<GenerateCodePass>();
}
} // namespace mlir::neura.
