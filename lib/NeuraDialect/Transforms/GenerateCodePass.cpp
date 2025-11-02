#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
// Returns formatted string like "#10" or "#3.0", or empty string if not found.
static std::string extractConstantLiteralFromAttr(Attribute attr) {
  if (!attr) return "";
  
  if (auto integer_attr = dyn_cast<IntegerAttr>(attr))
    return "#" + std::to_string(integer_attr.getInt());
  if (auto float_attr = dyn_cast<FloatAttr>(attr))
    return "#" + std::to_string(float_attr.getValueAsDouble());
  
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

  // De-dup sets.
  std::unordered_set<uint64_t> hop_signatures;     // (midTileId, ts, link_id).
  std::unordered_set<uint64_t> deposit_signatures; // (dstTileId, ts, regId).

  // ---------- helpers to place materialized instructions ----------.
  void placeRouterHop(const Topology &topology, int tile_id, int time_step,
                      StringRef input_direction, StringRef output_direction, bool asCtrlMov = false) {
    auto [tile_x, tile_y] = topology.tile_location.lookup(tile_id);
    Instruction instruction(asCtrlMov ? "CTRL_MOV" : "DATA_MOV");
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
    function.walk([&](Operation *op) {
      // placement for every op (even for mov/reserve).
      operation_placements[op] = getTileLocation(op);

      // build reserve -> phi mapping.
      if (isPhi(op) && op->getNumOperands() >= 1) {
        reserve_to_phi_map[op->getOperand(0)] = op;
      }

      // collect forwarders.
      if (isDataMov(op)) { data_movs.push_back(op); return; }
      if (isCtrlMov(op)) { ctrl_movs.push_back(op); return; }

      // skip Reserve from materialization.
      if (isReserve(op)) return;

      // materialize all other ops placed on tiles (compute/phi/const/etc.).
      auto placement = operation_placements[op];
      if (!placement.has_tile) return;

      std::string opcode = getOpcode(op);
      Instruction inst(opcode);
      inst.time_step = placement.time_step;

      if (isConstant(op)) {
        inst.src_operands.emplace_back(getConstantLiteral(op), "RED");
      } else if (op->getAttr("constant_value")) {
        // Checks if operation has constant_value attribute (for non-CONSTANT operations).
        inst.src_operands.emplace_back(getConstantLiteral(op), "RED");
      } else {
        // Handles normal operands, including operations with rhs_value attribute.
        SmallVector<Value> operands; operands.reserve(op->getNumOperands());
        
        // Process actual Value operands (if any).
        for (Value v : op->getOperands()) {
          operands.push_back(v);
          inst.src_operands.emplace_back("UNRESOLVED", "RED");
        }
        
        // If operation has rhs_value attribute, add it as an additional source operand.
        // This handles cases where binary operations have the RHS constant stored as an attribute.
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

      auto &bucket = getInstructionBucket(placement.col_idx, placement.row_idx, placement.time_step);
      bucket.push_back(std::move(inst));
      operation_to_instruction_reference[op] =
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
  void generateIntermediateHops(const SmallVector<LinkStep, 8> &links, const Topology &topo) {
    for (size_t i = 1; i < links.size(); ++i) {
      int prev_link = links[i - 1].link_id;
      int cur_link  = links[i].link_id;
      int ts        = links[i].ts;

      int mid_tile = topo.srcTileOfLink(cur_link);
      StringRef in  = topo.invertDir(topo.dirFromLink(prev_link));
      StringRef out = topo.dirFromLink(cur_link);

      uint64_t sig = (uint64_t)mid_tile << 32 ^ (uint64_t)ts << 16 ^ (uint64_t)cur_link;
      if (hop_signatures.insert(sig).second)
        placeRouterHop(topo, mid_tile, ts, in, out, /*asCtrlMov=*/IsCtrl);
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
  bool handleRegisterRewiring(Operation *consOp, Value atVal, const SmallVector<RegStep, 4> &regs,
                              const SmallVector<LinkStep, 8> &links, const Topology &topo) {
    if (regs.empty()) return false;

    int timestep_0 = regs.front().ts;
    int register_id = regs.back().regId;

    if (!links.empty()) {
      // Cross-tile: deposit on destination tile at earliest register ts.
      int dst_tile = topo.dstTileOfLink(links.back().link_id);
      StringRef incoming_dir = topo.dirFromLink(links.back().link_id);
      placeDstDeposit(topo, dst_tile, timestep_0, incoming_dir, register_id, /*asCtrlMov=*/IsCtrl);

      auto cp = operation_placements.lookup(consOp);
      if (cp.has_tile && cp.time_step > timestep_0) {
        setConsumerSourceExact(consOp, atVal, "$" + std::to_string(register_id));
        return true;
      }
    } else {
      // Same-tile: must go via register.
      setConsumerSourceExact(consOp, atVal, "$" + std::to_string(register_id));
      return true;
    }
    return false;
  }

  template<bool IsCtrl>
  void handleDirectionRewiring(Operation *consOp, Value atVal, StringRef consumer_direction,
                               const SmallVector<LinkStep, 8> &links, Operation *forwarder) {
    if (!links.empty()) {
      setConsumerSourceExact(consOp, atVal, consumer_direction.str());
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

    // Basic info from forwarders.
    Value source = forwarder->getOperand(0);
    Operation *producer = source.getDefiningOp();
    auto links = getLinkChain(forwarder);
    auto regs  = getRegisterSteps(forwarder);
    auto [producer_direction, consumer_direction] = computeDirections(links, topo);

    // Producer endpoints & intermediate hops.
    setProducerDestination(producer, producer_direction, regs);
    generateIntermediateHops<IsCtrl>(links, topo);

    // Gather consumers.
    SmallVector<std::pair<Operation*, Value>, 2> consumers;
    if constexpr (IsCtrl) {
      consumers = collectCtrlMovConsumers(forwarder, reserve2phi);
      if (consumers.empty()) return;
    } else {
      consumers = collectDataMovConsumers(forwarder);
    }

    // Wires each consumer: prefer register rewiring; fallback to direction rewiring.
    for (auto &[consOp, atVal] : consumers) {
      if (!handleRegisterRewiring<IsCtrl>(consOp, atVal, regs, links, topo))
        handleDirectionRewiring<IsCtrl>(consOp, atVal, consumer_direction, links, forwarder);
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
    // Only non-directional operands start with $ (registers) or # (constants).
    return !operand.empty() && operand[0] != '$' && operand[0] != '#';
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
  void placeDstDeposit(const Topology &topo, int dstTileId, int ts,
                       StringRef incomingDir, int regId, bool asCtrlMov = false) {
    uint64_t signature = (uint64_t)dstTileId << 32 ^ (uint64_t)ts << 16 ^ (uint64_t)regId;
    if (!deposit_signatures.insert(signature).second) return; // already placed.
    auto [tile_x, tile_y] = topo.tile_location.lookup(dstTileId);
    Instruction inst(asCtrlMov ? "CTRL_MOV" : "DATA_MOV");
    inst.time_step = ts;
    inst.src_operands.emplace_back(incomingDir.str(), "RED");
    inst.dst_operands.emplace_back("$" + std::to_string(regId), "RED");
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

      // Expands forwarders without re-walking IR.
      for (Operation *op : data_movs)
        expandMovImpl<false>(op, topo, /*unused*/reserve_to_phi_map);
      for (Operation *op : ctrl_movs)
        expandMovImpl<true>(op,  topo, reserve_to_phi_map);
      logUnresolvedOperands();

      int compiled_ii = getCompiledII(func);
      ArrayConfig config = buildArrayConfig(columns, rows, compiled_ii);
      writeYAMLOutput(config);
      writeASMOutput(config);
    }
  }
};

} // namespace.

namespace mlir::neura {
std::unique_ptr<Pass> createGenerateCodePass() {
  return std::make_unique<GenerateCodePass>();
}
} // namespace mlir::neura.
