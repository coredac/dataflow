#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
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
#include "NeuraDialect/Architecture/ArchitectureSpec.h"
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
  int time_step = -1; // for ordering.
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
  bool has_tile = false;
};

// ---- Operation kind helpers ----.
static bool isDataMov(Operation *op) { return dyn_cast<DataMovOp>(op) != nullptr; }
static bool isCtrlMov(Operation *op) { return dyn_cast<CtrlMovOp>(op) != nullptr; }
static bool isPhi(Operation *op) { return dyn_cast<PhiOp>(op) != nullptr; }
static bool isReserve(Operation *op) { return dyn_cast<ReserveOp>(op) != nullptr; }
static bool isConstant(Operation *op) { return dyn_cast<ConstantOp>(op) != nullptr; }

// ----- placement helpers -----.
static TileLocation getTileLocation(Operation *op) {
  TileLocation tile_location;
  if (auto arr = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
    for (Attribute a : arr) {
      auto d = dyn_cast<DictionaryAttr>(a);
      if (!d) continue;
      auto resource = dyn_cast_or_null<StringAttr>(d.get("resource"));
      if (!resource || resource.getValue() != "tile") continue;
      if (auto x_coord = dyn_cast_or_null<IntegerAttr>(d.get("x"))) tile_location.col_idx = x_coord.getInt();
      if (auto y_coord = dyn_cast_or_null<IntegerAttr>(d.get("y"))) tile_location.row_idx = y_coord.getInt();
      if (auto timestep = dyn_cast_or_null<IntegerAttr>(d.get("time_step"))) tile_location.time_step = timestep.getInt();
      tile_location.has_tile = true;
      break;
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
        if (auto per_tile_register_id = dyn_cast_or_null<IntegerAttr>(location_dict.get("per_tile_register_id"))) {
          return per_tile_register_id.getInt();
        }
      }
    }
  }
  return std::nullopt;
}

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
struct LinkStep { int link_id; int ts; };
struct RegStep  { int regId;  int ts; };

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
  llvm::sort(steps, [](const LinkStep &a, const LinkStep &b){ return a.ts < b.ts; });
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
  llvm::sort(steps, [](const RegStep &a, const RegStep &b){ return a.ts < b.ts; });
  return steps;
}

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
  DenseMap<Value, Operation*>                   reserve_to_phi_for_ctrl;
  // Map dfg_id -> op for later adjustments.
  DenseMap<int, Operation*>                     dfg_id_to_op;
  // Map (col,row,ts,local_idx_in_bucket) -> global instruction id.
  std::map<std::tuple<int,int,int,int>, int>    instruction_id_map;
  int next_instruction_id = 0;

  // De-dup sets.
  std::unordered_set<uint64_t> hop_signatures;     // (midTileId, ts, link_id).
  std::unordered_set<uint64_t> deposit_signatures; // (dstTileId, ts, regId).

  // ---------- helpers to place materialized instructions ----------.
  void placeRouterHop(const Topology &topology, int tile_id, int time_step,
                      StringRef input_direction, StringRef output_direction,
                      bool asCtrlMov = false, int assigned_id = -1) {
    auto [tile_x, tile_y] = topology.tile_location.lookup(tile_id);
    Instruction instruction(asCtrlMov ? "CTRL_MOV" : "DATA_MOV");
    instruction.id = assigned_id;
    instruction.time_step = time_step;
    instruction.src_operands.emplace_back(input_direction.str(), "RED");
    instruction.dst_operands.emplace_back(output_direction.str(), "RED");
    tile_time_instructions[{tile_x, tile_y}][time_step].push_back(std::move(instruction));
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
    function.walk([&](Operation *operation) {
      // Records placement for every operation (even for mov/reserve).
      operation_placements[operation] = getTileLocation(operation);

      // Builds reserve -> phi mapping.
      if (isPhi(operation) && operation->getNumOperands() >= 1) {
        reserve_to_phi_map[operation->getOperand(0)] = operation;
      }

      // Collects forwarders.
      if (isDataMov(operation)) { data_movs.push_back(operation); return; }
      if (isCtrlMov(operation)) { ctrl_movs.push_back(operation); return; }

      // Skips Reserve from materialization.
      if (isReserve(operation)) return;

      // Materializes all other operations placed on tiles (compute/phi/const/etc.).
      TileLocation placement = operation_placements[operation];
      if (!placement.has_tile) return;

      std::string opcode = getOpcode(operation);
      Instruction inst(opcode);
      inst.id = getDfgId(operation);
      inst.time_step = placement.time_step;

      if (isConstant(operation)) {
        inst.src_operands.emplace_back(getConstantLiteral(operation), "RED");
      } else if (operation->getAttr("constant_value")) {
        // Checks if operation has constant_value attribute (for non-CONSTANT operations).
        inst.src_operands.emplace_back(getConstantLiteral(operation), "RED");
      } else {
        // Handles normal operands, including operations with rhs_value attribute.
        SmallVector<Value> operands; operands.reserve(operation->getNumOperands());
        
        // Processes actual Value operands (if any).
        for (Value v : operation->getOperands()) {
          operands.push_back(v);
          inst.src_operands.emplace_back("UNRESOLVED", "RED");
        }
        
        // Handles cases where binary operations have the RHS constant stored as an attribute.
        if (auto rhs_value_attr = operation->getAttr("rhs_value")) {
          std::string rhs_literal = extractConstantLiteralFromAttr(rhs_value_attr);
          if (!rhs_literal.empty()) {
            inst.src_operands.emplace_back(rhs_literal, "RED");
          }
        }
        
        operation_to_operands[operation] = std::move(operands);
      }

      if (auto mapped_register_id = getMappedRegId(operation))
        inst.dst_operands.emplace_back("$" + std::to_string(*mapped_register_id), "RED");

      auto &bucket = getInstructionBucket(placement.col_idx, placement.row_idx, placement.time_step);
      bucket.push_back(std::move(inst));
      operation_to_instruction_reference[operation] =
          InstructionReference{placement.col_idx, placement.row_idx, placement.time_step,
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

  // Adds producer endpoints (first-hop directions or local $reg when using same-tile register paths).
  void setProducerDestination(Operation *producer, StringRef producer_direction, const SmallVector<RegStep, 4> &regs) {
    if (auto *pi = getInstructionPointer(producer)) {
      if (!producer_direction.empty() && producer_direction != "LOCAL") {
        setUniqueDestination(pi, producer_direction.str());
      } else if (!regs.empty()) {
        setUniqueDestination(pi, "$" + std::to_string(regs.back().regId));
      }
    }
  }

  // Emits router hops for multi-hop paths (from the second hop onwards). CTRL_MOV emits CTRL_MOV hops.
  template<bool IsCtrl>
  void generateIntermediateHops(const SmallVector<LinkStep, 8> &links, const Topology &topo,
                                int base_mov_id, size_t &hop_counter) {
    for (size_t i = 1; i < links.size(); ++i) {
      int prev_link = links[i - 1].link_id;
      int cur_link  = links[i].link_id;
      int ts        = links[i].ts;

      int mid_tile = topo.srcTileOfLink(cur_link);
      StringRef in  = topo.invertDir(topo.dirFromLink(prev_link));
      StringRef out = topo.dirFromLink(cur_link);

      uint64_t sig = (uint64_t)mid_tile << 32 ^ (uint64_t)ts << 16 ^ (uint64_t)cur_link;
      if (hop_signatures.insert(sig).second) {
        int hop_id = base_mov_id >= 0 ? base_mov_id * 10000 + static_cast<int>(hop_counter) : -1;
        ++hop_counter;
        placeRouterHop(topo, mid_tile, ts, in, out, /*asCtrlMov=*/IsCtrl, hop_id);
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

  // Try register-based rewiring. If cross-tile, emit deposits [incoming_dir]->[$reg] at earliest reg ts.
  // Returns true if rewiring to $reg was applied to consumers.
  template<bool IsCtrl>
  bool handleRegisterRewiring(Operation *consumer_operation, Value value_at_consumer, const SmallVector<RegStep, 4> &regs,
                              const SmallVector<LinkStep, 8> &links, const Topology &topo, int mov_dfg_id) {
    if (regs.empty()) return false;

    int timestep_0 = regs.front().ts;
    int register_id = regs.back().regId;

    if (!links.empty()) {
      // Cross-tile: deposit on destination tile at earliest register ts.
      int dst_tile = topo.dstTileOfLink(links.back().link_id);
      // Computes incoming direction from destination tile's perspective.
      StringRef incoming_dir = topo.invertDir(topo.dirFromLink(links.back().link_id));
      placeDstDeposit(topo, dst_tile, timestep_0, incoming_dir, register_id, /*asCtrlMov=*/IsCtrl, mov_dfg_id);

      TileLocation consumer_placement = operation_placements.lookup(consumer_operation);
      if (consumer_placement.has_tile && consumer_placement.time_step > timestep_0) {
        setConsumerSourceExact(consumer_operation, value_at_consumer, "$" + std::to_string(register_id));
        return true;
      }
    } else {
      // Same-tile: must go via register.
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
  void expandMovImpl(Operation *forwarder, const Topology &topo,
                     const DenseMap<Value, Operation*> &reserve2phi) {
    if (!validateForwarderShape<IsCtrl>(forwarder)) return;

    int mov_dfg_id = getDfgId(forwarder);

    // Basic info from forwarders.
    Value source = forwarder->getOperand(0);
    Operation *producer = source.getDefiningOp();
    SmallVector<LinkStep, 8> links = getLinkChain(forwarder);
    SmallVector<RegStep, 4> regs = getRegisterSteps(forwarder);
    std::pair<StringRef, StringRef> directions = computeDirections(links, topo);
    StringRef producer_direction = directions.first;
    StringRef consumer_direction = directions.second;

    // Producer endpoints & intermediate hops.
    setProducerDestination(producer, producer_direction, regs);
    size_t hop_counter = 1;
    generateIntermediateHops<IsCtrl>(links, topo, mov_dfg_id, hop_counter);

    // Gather consumers.
    SmallVector<std::pair<Operation*, Value>, 2> consumers;
    if constexpr (IsCtrl) {
      consumers = collectCtrlMovConsumers(forwarder, reserve2phi);
      if (consumers.empty()) return;
    } else {
      consumers = collectDataMovConsumers(forwarder);
    }

    // Wires each consumer: prefer register rewiring; fallback to direction rewiring.
    for (std::pair<Operation*, Value> &consumer_pair : consumers) {
      Operation *consumer_operation = consumer_pair.first;
      Value value_at_consumer = consumer_pair.second;
      if (!handleRegisterRewiring<IsCtrl>(consumer_operation, value_at_consumer, regs, links, topo, mov_dfg_id))
        handleDirectionRewiring<IsCtrl>(consumer_operation, value_at_consumer, consumer_direction, links, topo, forwarder);
    }
  }


  // ---------- output generation ----------.
  void logUnresolvedOperands() {
    unsigned unsrc = 0, undst = 0;
    for (auto &tile_entry : tile_time_instructions) {
      std::pair<int,int> tile_key = tile_entry.first;
      int column = tile_key.first, row = tile_key.second;
      for (auto &timestep_entry : tile_entry.second) {
        int ts = timestep_entry.first;
        std::vector<Instruction> &vec = timestep_entry.second;
        for (size_t i = 0; i < vec.size(); ++i) {
          Instruction &inst = vec[i];
          for (size_t si = 0; si < inst.src_operands.size(); ++si) {
            Operand &s = inst.src_operands[si];
            if (s.operand == "UNRESOLVED") {
              s.color = "ERROR"; ++unsrc;
              llvm::errs() << "[UNRESOLVED SRC] tile("<<column<<","<<row<<") t="<<ts
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
              llvm::errs() << "[UNRESOLVED DST] tile("<<column<<","<<row<<") t="<< ts
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

  // Assigns unique IDs to all materialized instructions (including data/ctrl mov hops),
  // and records their timesteps.
  void assignInstructionIds(std::unordered_set<int> &materialized_ids,
                            std::unordered_map<int, int> &materialized_timesteps) {
    instruction_id_map.clear();
    int max_assigned = -1;
    for (auto &[tile_key, timestep_map] : tile_time_instructions) {
      for (auto &[ts, inst_vec] : timestep_map) {
        for (Instruction &inst : inst_vec) {
          if (inst.id >= 0) max_assigned = std::max(max_assigned, inst.id);
        }
      }
    }
    next_instruction_id = max_assigned + 1;
    for (auto &[tile_key, timestep_map] : tile_time_instructions) {
      int col = tile_key.first;
      int row = tile_key.second;
      for (auto &[ts, inst_vec] : timestep_map) {
        for (size_t idx = 0; idx < inst_vec.size(); ++idx) {
          Instruction &inst = inst_vec[idx];
          if (inst.id < 0) inst.id = next_instruction_id++;
          instruction_id_map[{col, row, ts, (int)idx}] = inst.id;
          materialized_ids.insert(inst.id);
          materialized_timesteps[inst.id] = ts;
        }
      }
    }
  }

  // Looks up instruction ID by InstructionReference (col,row,ts,idx in bucket).
  int lookupInstructionId(const InstructionReference &ref) const {
    auto it = instruction_id_map.find({ref.col_idx, ref.row_idx, ref.t, ref.idx});
    if (it == instruction_id_map.end()) return -1;
    return it->second;
  }

  // Gets instruction ID for a materialized operation.
  int getInstructionId(Operation *op) const {
    auto it = operation_to_instruction_reference.find(op);
    if (it == operation_to_instruction_reference.end()) return -1;
    return lookupInstructionId(it->second);
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

  // Finds hop instruction ID for a link step (data or ctrl).
  int findHopId(const LinkStep &step, const Topology &topo, bool isCtrl) const {
    int mid_tile = topo.srcTileOfLink(step.link_id);
    auto [x, y] = topo.tile_location.lookup(mid_tile);
    auto tile_it = tile_time_instructions.find({x, y});
    if (tile_it == tile_time_instructions.end()) return -1;
    auto ts_it = tile_it->second.find(step.ts);
    if (ts_it == tile_it->second.end()) return -1;
    const auto &vec = ts_it->second;
    for (size_t i = 0; i < vec.size(); ++i) {
      const Instruction &inst = vec[i];
      if (inst.opcode == (isCtrl ? "CTRL_MOV" : "DATA_MOV")) {
        auto id_it = instruction_id_map.find({x, y, step.ts, (int)i});
        if (id_it != instruction_id_map.end()) return id_it->second;
      }
    }
    return -1;
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
        
        // Only tile resources have x/y coordinates.
        if (resource && resource.getValue() == "tile") {
          if (auto x_attr = dyn_cast_or_null<IntegerAttr>(d.get("x"))) {
            info.tile_x = x_attr.getInt();
          }
          if (auto y_attr = dyn_cast_or_null<IntegerAttr>(d.get("y"))) {
            info.tile_y = y_attr.getInt();
          }
          info.has_tile = true;
          break;  // Takes first tile location.
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

  void collectHopTiles(const SmallVector<LinkStep, 8> &link_steps,
                       TileLocation producer_loc,
                       const Topology &topology,
                       SmallVector<std::pair<int, int>, 8> &out_tiles,
                       SmallVector<int, 8> &out_time_steps) const {
    out_tiles.clear();
    out_time_steps.clear();
    if (link_steps.empty()) return;

    int producer_tile_id = producer_loc.has_tile
        ? topology.tileIdAt(producer_loc.col_idx, producer_loc.row_idx) : -1;
    int consumer_tile_id = topology.dstTileOfLink(link_steps.back().link_id);

    for (size_t i = 0; i < link_steps.size(); ++i) {
      int middle_tile_id = topology.srcTileOfLink(link_steps[i].link_id);
      if (middle_tile_id == producer_tile_id || middle_tile_id == consumer_tile_id) continue;
      auto coord = topology.tile_location.lookup(middle_tile_id);
      if (!out_tiles.empty() && out_tiles.back().first == coord.first &&
          out_tiles.back().second == coord.second) {
        continue; // Skips duplicates.
      }
      out_tiles.push_back(coord);
      out_time_steps.push_back(link_steps[i].ts);
    }
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
      // Build hop tiles directly from link steps (mirrors router hop emission).
      if (link_steps.size() > 1) {
        for (size_t i = 1; i < link_steps.size(); ++i) {
          int middle_tile_id = topology.srcTileOfLink(link_steps[i].link_id);
          auto coord = topology.tile_location.lookup(middle_tile_id);
          hop_tiles.push_back(coord);
          hop_time_steps.push_back(link_steps[i].ts);
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
    });

    for (const auto &edge : original_edges) {
      if (skip_edges.count(edge)) continue;
      edges.push_back(edge);
    }

    for (const HopRewriteInfo &rewrite : rewrites) {
      if (rewrite.hop_tiles.empty()) continue;
      int previous = rewrite.producer_id;
      int base = rewrite.mov_id * 10000;
      for (size_t i = 0; i < rewrite.hop_tiles.size(); ++i) {
        int node_id = base + static_cast<int>(i) + 1;
        if (previous >= 0) edges.emplace_back(previous, node_id);
        previous = node_id;
      }
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
              int ts = ts_attr.getInt();
              if (min_register_ts < 0 || ts < min_register_ts) min_register_ts = ts;
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

  // Writes DOT and JSON DFG outputs based on SSA and dfg_id attributes.
  // Applies hop-aware coordinates/middle-node insertion for DATA_MOV / CTRL_MOV,
  // and bypasses reserve nodes (no node for reserve, edges direct from producer to consumer).
  void writeDFGOutputSSA(func::FuncOp func, const Topology &topology,
                         const std::unordered_set<int> &materialized_ids,
                         const std::unordered_map<int, int> &materialized_timesteps) {
    DfgNodeMap nodes;
    DfgEdgeList edges;

    DenseMap<Value, SmallVector<Operation *, 2>> reserve_to_ctrl_movs;
    collectCtrlMovReserves(func, reserve_to_ctrl_movs);

    buildSsaNodesAndEdges(func, reserve_to_ctrl_movs, nodes, edges);

    std::vector<std::pair<int,int>> original_edges = edges;
    edges.clear();
    llvm::SmallDenseSet<std::pair<int,int>, 32> edges_to_skip;
    std::vector<HopRewriteInfo> hop_rewrites;

    collectHopRewrites(func, topology, nodes, original_edges, edges, edges_to_skip, hop_rewrites);

    std::unordered_set<int> reg_only_movs;
    adjustRegisterOnlyMovCoords(nodes, edges);

    // Align mov/hop timesteps to materialized instruction timesteps when available.
    for (auto &entry : nodes) {
      int node_id = entry.first;
      DfgNodeInfo &node = entry.second;
      if (node.opcode == "DATA_MOV" || node.opcode == "CTRL_MOV") {
        auto it_ts = materialized_timesteps.find(node_id);
        if (it_ts != materialized_timesteps.end()) {
          node.time_step = it_ts->second;
        }
      }
    }

    pruneMovNodesWithoutCoords(nodes, edges, materialized_ids);

    emitDotOutput(nodes, edges);
    emitYamlOutput(nodes, edges);

    llvm::outs() << "[generate-code] DFG (SSA-based) emitted: nodes=" << nodes.size()
                 << ", edges=" << edges.size()
                 << " -> tmp-generated-dfg.dot, tmp-generated-dfg.yaml\n";
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
      yaml_out << "    - column: " << core.col_idx << "\n      row: " << core.row_idx
               << "\n      core_id: \"" << core.core_id << "\"\n      entries:\n";
      
      // Groups instructions by timestep.
      std::map<int, std::vector<const Instruction*>> timestep_groups;
      for (const Instruction &inst : core.entry.instructions) {
        timestep_groups[inst.time_step].push_back(&inst);
      }
      
      yaml_out << "        - entry_id: \"entry0\"\n          instructions:\n";
      for (const auto &timestep_pair : timestep_groups) {
        int timestep = timestep_pair.first;
        const auto &operations = timestep_pair.second;
        
        yaml_out << "            - timestep: " << timestep << "\n              operations:\n";
        for (const Instruction *inst : operations) {
          yaml_out << "                - opcode: \"" << inst->opcode << "\"\n";
          if (inst->id >= 0)
            yaml_out << "                  id: " << inst->id << "\n";
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

  void writeASMOutput(const ArrayConfig &config) {
    std::error_code ec;
    llvm::raw_fd_ostream asm_out("tmp-generated-instructions.asm", ec);
    if (ec) return;

    if (config.compiled_ii >= 0) {
      asm_out << "# Compiled II: " << config.compiled_ii << "\n\n";
    }

    for (const Tile &core : config.cores) {
      asm_out << "PE(" << core.col_idx << "," << core.row_idx << "):\n";
      
      // Groups instructions by timestep.
      std::map<int, std::vector<const Instruction*>> timestep_groups;
      for (const Instruction &inst : core.entry.instructions) {
        timestep_groups[inst.time_step].push_back(&inst);
      }
      
      for (const auto &timestep_pair : timestep_groups) {
        int timestep = timestep_pair.first;
        const auto &instructions = timestep_pair.second;
        
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
          asm_out << "\n";
        }
        asm_out << "} (t=" << timestep << ")\n";
      }
      asm_out << "\n";
    }
    asm_out.close();
  }

  // Endpoint deposits: on destination tiles at earliest reg ts, move [incoming_dir] -> [$reg].
  // CTRL_MOV paths emit CTRL_MOV deposits; DATA_MOV paths emit DATA_MOV deposits.
  void placeDstDeposit(const Topology &topo, int dst_tile_id, int ts,
                       StringRef incoming_dir, int reg_id, bool asCtrlMov = false, int assigned_id = -1) {
    uint64_t signature = (uint64_t)dst_tile_id << 32 ^ (uint64_t)ts << 16 ^ (uint64_t)reg_id;
    if (!deposit_signatures.insert(signature).second) return; // already placed.
    auto [tile_x, tile_y] = topo.tile_location.lookup(dst_tile_id);
    Instruction inst(asCtrlMov ? "CTRL_MOV" : "DATA_MOV");
    inst.id = assigned_id;
    inst.time_step = ts;
    inst.src_operands.emplace_back(incoming_dir.str(), "RED");
    inst.dst_operands.emplace_back("$" + std::to_string(reg_id), "RED");
    tile_time_instructions[{tile_x, tile_y}][ts].push_back(std::move(inst));
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

      clearState();

      // Single function-level walks: index + materialize + collect.
      SmallVector<Operation*> data_movs;
      SmallVector<Operation*> ctrl_movs;
      DenseMap<Value, Operation*> reserve_to_phi_map;
      indexIR(func, data_movs, ctrl_movs, reserve_to_phi_map);
      reserve_to_phi_for_ctrl = reserve_to_phi_map;

      // Expands forwarders without re-walking IR.
      for (Operation *op : data_movs)
        expandMovImpl<false>(op, topo, /*unused*/reserve_to_phi_map);
      for (Operation *op : ctrl_movs)
        expandMovImpl<true>(op,  topo, reserve_to_phi_map);
      logUnresolvedOperands();

      int compiled_ii = getCompiledII(func);
    std::unordered_set<int> materialized_ids;
    std::unordered_map<int, int> materialized_timesteps;
    assignInstructionIds(materialized_ids, materialized_timesteps);
      ArrayConfig config = buildArrayConfig(columns, rows, compiled_ii);
      writeYAMLOutput(config);
      writeASMOutput(config);
    writeDFGOutputSSA(func, topo, materialized_ids, materialized_timesteps);
    }
  }
};

} // namespace.

namespace mlir::neura {
std::unique_ptr<Pass> createGenerateCodePass() {
  return std::make_unique<GenerateCodePass>();
}
} // namespace mlir::neura.
