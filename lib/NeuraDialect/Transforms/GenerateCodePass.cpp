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
  int time_step = -1; // for ordering
  Instruction(const std::string &op) : opcode(op) {}
};

// Entry represents a basic block or execution context within a tile.
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
  Entry entry; // single entry per tile
  Tile(int c, int r, int id) : col_idx(c), row_idx(r), core_id(id), entry("entry0", "loop") {}
};

struct ArrayConfig {
  int columns;
  int rows;
  std::vector<Tile> cores;
};

struct TileLocation {
  int col_idx = -1, row_idx = -1, time_step = -1;
  bool has_tile = false;
};

// ---- Operation kind helpers ----
static bool isDataMov(Operation *op) { return dyn_cast<DataMovOp>(op) != nullptr; }
static bool isCtrlMov(Operation *op) { return dyn_cast<CtrlMovOp>(op) != nullptr; }
static bool isPhi(Operation *op) { return dyn_cast<PhiOp>(op) != nullptr; }
static bool isReserve(Operation *op) { return dyn_cast<ReserveOp>(op) != nullptr; }
static bool isConstant(Operation *op) { return dyn_cast<ConstantOp>(op) != nullptr; }

// ----- placement helpers -----
static TileLocation getTileLocation(Operation *op) {
  TileLocation tile_location;
  if (auto arr = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
    for (Attribute a : arr) {
      auto d = dyn_cast<DictionaryAttr>(a);
      if (!d) continue;
      auto res = dyn_cast_or_null<StringAttr>(d.get("resource"));
      if (!res || res.getValue() != "tile") continue;
      if (auto x = dyn_cast_or_null<IntegerAttr>(d.get("x"))) tile_location.col_idx = x.getInt();
      if (auto y = dyn_cast_or_null<IntegerAttr>(d.get("y"))) tile_location.row_idx = y.getInt();
      if (auto ts = dyn_cast_or_null<IntegerAttr>(d.get("time_step"))) tile_location.time_step = ts.getInt();
      tile_location.has_tile = true;
      break;
    }
  }
  // If tile mapping exists, x/y must be valid.
  assert(!tile_location.has_tile || (tile_location.col_idx >= 0 && tile_location.row_idx >= 0));
  return tile_location;
}

static std::optional<int> getMappedRegId(Operation *op) {
  if (auto mapping_locations = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
    for (Attribute location_attr : mapping_locations) {
      auto location_dict = dyn_cast<DictionaryAttr>(location_attr);
      if (!location_dict) continue;
      auto resource_attr = dyn_cast_or_null<StringAttr>(location_dict.get("resource"));
      if (!resource_attr) continue;
      if (resource_attr.getValue() == "register" || resource_attr.getValue() == "reg") {
        if (auto register_id = dyn_cast_or_null<IntegerAttr>(location_dict.get("id")))
          return register_id.getInt();
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
  return opcode;
}

// Literal for CONSTANT's source
static std::string getConstantLiteral(Operation *op) {
  if (!isConstant(op)) return "";
  if (auto value_attr = op->getAttr("value")) {
    if (auto integer_attr = dyn_cast<IntegerAttr>(value_attr))
      return "#" + std::to_string(integer_attr.getInt());
    if (auto float_attr = dyn_cast<FloatAttr>(value_attr))
      return "#" + std::to_string(float_attr.getValueAsDouble());
  }
  return "#0";
}

// ----- Topology from Architecture -----
struct Topology {
  DenseMap<int, std::pair<int,int>> link_ends;      // link_id -> (srcTileId, dstTileId)
  DenseMap<int, std::pair<int,int>> tile_location;  // tileId -> (x,y)
  DenseMap<std::pair<int,int>, int> coord_to_tile;  // (x,y) -> tileId

  StringRef getDirBetween(int srcTid, int dstTid) const {
    auto [src_x, src_y] = tile_location.lookup(srcTid);
    auto [dst_x, dst_y] = tile_location.lookup(dstTid);
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

static Topology getTopologyFromArchitecture(int columns, int rows) {
  Topology topo;
  mlir::neura::Architecture arch(columns, rows);

  for (auto *tile : arch.getAllTiles()) {
    topo.tile_location[tile->getId()] = {tile->getX(), tile->getY()};
    topo.coord_to_tile[{tile->getX(), tile->getY()}] = tile->getId();
  }
  for (auto *link : arch.getAllLinks()) {
    auto *src_tile = link->getSrcTile();
    auto *dst_tile = link->getDstTile();
    topo.link_ends[link->getId()] = {src_tile->getId(), dst_tile->getId()};
  }
  return topo;
}

// ----- Extract mapping steps (sorted by time) -----
struct LinkStep { int link_id; int ts; };
struct RegStep  { int regId;  int ts; };

static SmallVector<LinkStep, 8> collectLinkSteps(Operation *op) {
  SmallVector<LinkStep, 8> steps;
  if (auto mapping_locations = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
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
  if (auto mapping_locations = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
    for (Attribute location_attr : mapping_locations) {
      auto location_dict = dyn_cast<DictionaryAttr>(location_attr);
      if (!location_dict) continue;
      auto resource_attr = dyn_cast_or_null<StringAttr>(location_dict.get("resource"));
      if (!resource_attr) continue;
      if (resource_attr.getValue() == "register" || resource_attr.getValue() == "reg") {
        auto register_id = dyn_cast_or_null<IntegerAttr>(location_dict.get("id"));
        auto time_step = dyn_cast_or_null<IntegerAttr>(location_dict.get("time_step"));
        if (!register_id || !time_step) continue;
        steps.push_back({(int)register_id.getInt(), (int)time_step.getInt()});
      }
    }
  }
  llvm::sort(steps, [](const RegStep &a, const RegStep &b){ return a.ts < b.ts; });
  return steps;
}

// ----- Pass -----
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

  // Map of tile coordinates (x,y) -> time_step -> vector of Instructions
  std::map<std::pair<int,int>, std::map<int, std::vector<Instruction>>> tile_time_instructions;

  // Back references from IR operation to emitted instruction
  DenseMap<Operation*, InstructionReference>    operation_to_instruction_reference;
  DenseMap<Operation*, SmallVector<Value>>      operation_to_operands;

  // De-dup sets
  std::unordered_set<uint64_t> hop_signatures;     // (midTileId, ts, link_id)
  std::unordered_set<uint64_t> deposit_signatures; // (dstTileId, ts, regId)

  // ---------- helpers to place materialized instructions ----------
  void placeRouterHop(const Topology &topology, int tile_id, int time_step,
                      StringRef input_direction, StringRef output_direction, bool asCtrlMov = false) {
    auto [tile_x, tile_y] = topology.tile_location.lookup(tile_id);
    Instruction instruction(asCtrlMov ? "CTRL_MOV" : "DATA_MOV");
    instruction.time_step = time_step;
    instruction.src_operands.emplace_back(input_direction.str(), "RED");
    instruction.dst_operands.emplace_back(output_direction.str(), "RED");
    tile_time_instructions[{tile_x, tile_y}][time_step].push_back(std::move(instruction));
  }

  // ---------- initialization helpers ----------
  void clearState() {
    operation_placements.clear();
    tile_time_instructions.clear();
    operation_to_instruction_reference.clear();
    operation_to_operands.clear();
    hop_signatures.clear();
    deposit_signatures.clear();
  }

  void setTilePlacements(func::FuncOp function) {
    function.walk([&](Operation *operation){ operation_placements[operation] = getTileLocation(operation); });
  }

  std::pair<int, int> getArrayDimensions(func::FuncOp function) {
    int columns = 4, rows = 4; // Default 4x4 CGRA configuration
    if (auto mapping_info = function->getAttrOfType<DictionaryAttr>("mapping_info")) {
      if (auto x_tiles = dyn_cast_or_null<IntegerAttr>(mapping_info.get("x_tiles"))) columns = x_tiles.getInt();
      if (auto y_tiles = dyn_cast_or_null<IntegerAttr>(mapping_info.get("y_tiles"))) rows   = y_tiles.getInt();
    }
    return {columns, rows};
  }

  // ---------- compute operations materialization ----------
  void materializeComputeOps(func::FuncOp function) {
    function.walk([&](Operation *operation){
      auto placement = operation_placements[operation];
      if (!placement.has_tile) return;
      if (isReserve(operation) || isCtrlMov(operation) || isDataMov(operation)) return;

      std::string opcode = getOpcode(operation);
      Instruction instruction(opcode);
      instruction.time_step = placement.time_step;

      if (isConstant(operation)) {
        instruction.src_operands.emplace_back(getConstantLiteral(operation), "RED");
      } else {
        SmallVector<Value> operands; operands.reserve(operation->getNumOperands());
        for (Value input_value : operation->getOperands()) {
          operands.push_back(input_value);
          instruction.src_operands.emplace_back("UNRESOLVED", "RED");
        }
        operation_to_operands[operation] = std::move(operands);
      }

      // If an op maps a register by itself, keep it as a destination.
      if (auto mapped_register_id = getMappedRegId(operation))
        instruction.dst_operands.emplace_back("$" + std::to_string(*mapped_register_id), "RED");

      auto &bucket = getInstructionBucket(placement.col_idx, placement.row_idx, placement.time_step);
      bucket.push_back(std::move(instruction));
      operation_to_instruction_reference[operation] =
          InstructionReference{placement.col_idx, placement.row_idx, placement.time_step,
                               (int)bucket.size() - 1};
    });
  }

  // ---------- wiring via unified forwarder expansion ----------
  static SmallVector<LinkStep, 8> getLinkChain(Operation *forwarder) { return collectLinkSteps(forwarder); }
  static SmallVector<RegStep, 4>  getRegisterSteps(Operation *forwarder) { return collectRegSteps(forwarder); }

  // Validate forwarder op arity: DATA_MOV: at least 1 in/1 out; CTRL_MOV: at least 2 inputs (src,reserve)
  template<bool IsCtrl>
  bool validateForwarderShape(Operation *forwarder) {
    if constexpr (!IsCtrl) {
      return forwarder->getNumOperands() >= 1 && forwarder->getNumResults() >= 1;
    } else {
      return forwarder->getNumOperands() >= 2;
    }
  }

  // Compute producer first-hop direction and consumer last-hop direction (or LOCAL if link-less)
  std::pair<StringRef, StringRef> computeDirections(const SmallVector<LinkStep, 8> &links, const Topology &topo) {
    StringRef producer_direction("LOCAL");
    StringRef consumer_direction("LOCAL");
    if (!links.empty()) {
      producer_direction = topo.dirFromLink(links.front().link_id);
      consumer_direction = topo.invertDir(topo.dirFromLink(links.back().link_id));
    }
    return {producer_direction, consumer_direction};
  }

  // Add producer endpoint (first-hop direction or local $reg when using same-tile register path)
  void setProducerDestination(Operation *producer, StringRef producer_direction, const SmallVector<RegStep, 4> &regs) {
    if (auto *pi = getInstructionPointer(producer)) {
      if (!producer_direction.empty() && producer_direction != "LOCAL") {
        setUniqueDestination(pi, producer_direction.str());
      } else if (!regs.empty()) {
        setUniqueDestination(pi, "$" + std::to_string(regs.back().regId));
      }
    }
  }

  // Emit router hops for multi-hop path (from the second hop onwards). CTRL_MOV emits CTRL_MOV hops.
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

  // Consumers for DATA_MOV: all users of forwarder result(0)
  SmallVector<std::pair<Operation*, Value>, 2> collectDataMovConsumers(Operation *forwarder) {
    SmallVector<std::pair<Operation*, Value>, 2> consumers;
    Value out = forwarder->getResult(0);
    for (OpOperand &use : out.getUses())
      consumers.push_back({use.getOwner(), use.get()});
    return consumers;
  }

  // Consumers for CTRL_MOV: PHI found by reserve->phi map, but we must wire the PHI's *data* input.
  // So we pair {phi, source}, not {phi, reserve}.
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

  // Try register-based rewiring. If cross-tile, emits a deposit [incoming_dir]->[$reg] at earliest reg ts.
  // Returns true if rewiring to $reg was applied to the consumer.
  template<bool IsCtrl>
  bool handleRegisterRewiring(Operation *consOp, Value atVal, const SmallVector<RegStep, 4> &regs,
                              const SmallVector<LinkStep, 8> &links, const Topology &topo) {
    if (regs.empty()) return false;

    int ts0 = regs.front().ts;
    int rId = regs.back().regId;

    if (!links.empty()) {
      // Cross-tile: deposit on destination tile at earliest register ts.
      int dst_tile = topo.dstTileOfLink(links.back().link_id);
      StringRef incoming_dir = topo.dirFromLink(links.back().link_id);
      placeDstDeposit(topo, dst_tile, ts0, incoming_dir, rId, /*asCtrlMov=*/IsCtrl);

      auto cp = operation_placements.lookup(consOp);
      if (cp.has_tile && cp.time_step > ts0) {
        setConsumerSourceExact(consOp, atVal, "$" + std::to_string(rId));
        return true;
      }
    } else {
      // Same-tile: must go via register
      setConsumerSourceExact(consOp, atVal, "$" + std::to_string(rId));
      return true;
    }
    return false;
  }

  // If not using a register, wire the consumer to the final incoming direction.
  template<bool IsCtrl>
  void handleDirectionRewiring(Operation *consOp, Value atVal, StringRef consumer_direction,
                               const SmallVector<LinkStep, 8> &links, Operation *forwarder) {
    if (!links.empty()) {
      setConsumerSourceExact(consOp, atVal, consumer_direction.str());
    } else {
      forwarder->emitError(IsCtrl
          ? "same-tile ctrl_mov without register mapping is illegal (LOCAL forbidden). Provide a register in mapping_locs."
          : "same-tile data_mov without register mapping is illegal (LOCAL forbidden). Provide a register in mapping_locs.");
      assert(false && "same-tile mov without register mapping");
    }
  }

  // Core implementation:
  //   IsCtrl=false -> DATA_MOV
  //   IsCtrl=true  -> CTRL_MOV (uses CTRL_MOV for hops/deposits and targets PHI's data input)
  template<bool IsCtrl>
  void expandMovImpl(Operation *forwarder, const Topology &topo,
                     const DenseMap<Value, Operation*> &reserve2phi) {
    if (!validateForwarderShape<IsCtrl>(forwarder)) return;

    // Basic info from forwarder
    Value source = forwarder->getOperand(0);
    Operation *producer = source.getDefiningOp();
    auto links = getLinkChain(forwarder);
    auto regs  = getRegisterSteps(forwarder);
    auto [producer_direction, consumer_direction] = computeDirections(links, topo);

    // Producer endpoint & intermediate hops
    setProducerDestination(producer, producer_direction, regs);
    generateIntermediateHops<IsCtrl>(links, topo);

    // Gather consumers
    SmallVector<std::pair<Operation*, Value>, 2> consumers;
    if constexpr (IsCtrl) {
      consumers = collectCtrlMovConsumers(forwarder, reserve2phi);
      if (consumers.empty()) return;
    } else {
      consumers = collectDataMovConsumers(forwarder);
    }

    // Wire each consumer: prefer register rewiring; fallback to direction rewiring
    for (auto &[consOp, atVal] : consumers) {
      if (!handleRegisterRewiring<IsCtrl>(consOp, atVal, regs, links, topo))
        handleDirectionRewiring<IsCtrl>(consOp, atVal, consumer_direction, links, forwarder);
    }
  }

  // Thin wrappers to keep the original interface
  void expandDataMov(Operation *forwarder, const Topology &topology) {
    static const DenseMap<Value, Operation*> kEmpty;
    expandMovImpl<false>(forwarder, topology, kEmpty);
  }

  void expandCtrlMov(Operation *forwarder, const Topology &topology,
                     const DenseMap<Value, Operation*> &reserve_to_phi_map) {
    expandMovImpl<true>(forwarder, topology, reserve_to_phi_map);
  }

  // ---------- output generation ----------
  void logUnresolvedOperands(ModuleOp module) {
    unsigned unsrc = 0, undst = 0;
    for (auto &tileKV : tile_time_instructions) {
      std::pair<int,int> tile_key = tileKV.first;
      int column = tile_key.first, row = tile_key.second;
      for (auto &tsKV : tileKV.second) {
        int ts = tsKV.first;
        std::vector<Instruction> &vec = tsKV.second;
        for (size_t i = 0; i < vec.size(); ++i) {
          Instruction &inst = vec[i];
          for (size_t si = 0; si < inst.src_operands.size(); ++si) {
            Operand &s = inst.src_operands[si];
            if (s.operand == "UNRESOLVED") {
              s.color = "YELLOW"; ++unsrc;
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
              d.color = "YELLOW"; ++undst;
              llvm::errs() << "[UNRESOLVED DST] tile("<<column<<","<<row<<") t="<< ts
                           << " inst#" << i << " op=" << inst.opcode
                           << " dst_idx=" << di << "\n";
            }
          }
        }
      }
    }
    if (unsrc + undst) {
      auto diag = module.emitWarning("GenerateCodePass: UNRESOLVED operands kept for debugging");
      diag << " (src=" << unsrc << ", dst=" << undst << "); they are highlighted with color=YELLOW in YAML.";
    }
  }

  ArrayConfig buildArrayConfig(int columns, int rows) {
    ArrayConfig config{columns, rows, {}};
    std::map<std::pair<int,int>, std::vector<Instruction>> tile_insts;
    
    // Flatten and sort by timestep
    for (auto &[tile_key, tsmap] : tile_time_instructions) {
      auto &flat = tile_insts[tile_key];
      for (auto &[ts, vec] : tsmap) for (Instruction &inst : vec) flat.push_back(inst);
      std::stable_sort(flat.begin(), flat.end(), [](const Instruction &a, const Instruction &b){ return a.time_step < b.time_step; });
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
    llvm::raw_fd_ostream yaml_out("generated-instructions.yaml", ec);
    if (ec) return;

    yaml_out << "array_config:\n  columns: " << config.columns << "\n  rows: " << config.rows << "\n  cores:\n";
    for (const Tile &core : config.cores) {
      yaml_out << "    - column: " << core.col_idx << "\n      row: " << core.row_idx 
               << "\n      core_id: \"" << core.core_id << "\"\n      entries:\n";
      int entry_id = 0;
      for (const Instruction &inst : core.entry.instructions) {
        yaml_out << "        - entry_id: \"entry" << entry_id++ << "\"\n          instructions:\n"
                 << "            - opcode: \"" << inst.opcode << "\"\n              timestep: " << inst.time_step << "\n";
        for (const auto &[operands, prefix] : {std::make_pair(inst.src_operands, "src_operands"), 
                                               std::make_pair(inst.dst_operands, "dst_operands")}) {
          if (!operands.empty()) {
            yaml_out << "              " << prefix << ":\n";
            for (const Operand &opnd : operands)
              yaml_out << "                - operand: \"" << opnd.operand << "\"\n                  color: \"" << opnd.color << "\"\n";
          }
        }
      }
    }
    yaml_out.close();
  }

  // Check if an operand token is a direction (vs constant/register)
  static bool isDirectionalOperand(const std::string &operand) {
    // Only non-directional operands start with $ (registers) or # (constants)
    return !operand.empty() && operand[0] != '$' && operand[0] != '#';
  }

  // Print one operand as [TOKEN, COLOR] (omit color for non-directionals in YAML; in ASM we keep it)
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
    llvm::raw_fd_ostream asm_out("generated-instructions.asm", ec);
    if (ec) return;

    for (const Tile &core : config.cores) {
      asm_out << "PE(" << core.col_idx << "," << core.row_idx << "):\n";
      for (const Instruction &inst : core.entry.instructions) {
        asm_out << "{\n  " << inst.opcode;
        for (const Operand &operand : inst.src_operands) asm_out << ", " << formatOperand(operand);
        if (!inst.dst_operands.empty()) {
          asm_out << " -> ";
          for (size_t i = 0; i < inst.dst_operands.size(); ++i) {
            if (i > 0) asm_out << ", ";
            asm_out << formatOperand(inst.dst_operands[i]);
          }
        }
        asm_out << " (t=" << inst.time_step << ")\n}\n";
      }
      asm_out << "\n";
    }
    asm_out.close();
  }

  // Endpoint deposit: on destination tile at earliest reg ts, move [incoming_dir] -> [$reg].
  // CTRL_MOV path emits a CTRL_MOV deposit; DATA_MOV path emits a DATA_MOV deposit.
  void placeDstDeposit(const Topology &topo, int dstTileId, int ts,
                       StringRef incomingDir, int regId, bool asCtrlMov = false) {
    uint64_t signature = (uint64_t)dstTileId << 32 ^ (uint64_t)ts << 16 ^ (uint64_t)regId;
    if (!deposit_signatures.insert(signature).second) return; // already placed
    auto [tile_x, tile_y] = topo.tile_location.lookup(dstTileId);
    Instruction inst(asCtrlMov ? "CTRL_MOV" : "DATA_MOV");
    inst.time_step = ts;
    inst.src_operands.emplace_back(incomingDir.str(), "RED");
    inst.dst_operands.emplace_back("$" + std::to_string(regId), "RED");
    tile_time_instructions[{tile_x, tile_y}][ts].push_back(std::move(inst));
  }

  // Utilities to access instruction buckets/pointers
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

  // Replace the exact source slot in the consumer that corresponds to `value_at_consumer`,
  // or fill the first UNRESOLVED placeholder if a 1:1 match wasn't found.
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

  // Append a destination only once.
  static void setUniqueDestination(Instruction *inst, const std::string &text){
    for (Operand &d : inst->dst_operands) if (d.operand == text) return;
    inst->dst_operands.emplace_back(text, "RED");
  }

  // ---------- ctrl_mov helpers ----------
  // Build mapping from reserve SSA value -> PHI op (PHI operand#0 = reserve)
  DenseMap<Value, Operation*> buildReserveToPhiMap(func::FuncOp function) {
    DenseMap<Value, Operation*> m;
    function.walk([&](Operation *op){
      if (isPhi(op) && op->getNumOperands() >= 1)
        m[op->getOperand(0)] = op;
    });
    return m;
  }

  // ---------- entry points ----------
  void wireEdges(func::FuncOp function, const Topology &topology) {
    function.walk([&](Operation *op){ if (isDataMov(op)) expandDataMov(op, topology); });
  }

  void handleCtrlMov(func::FuncOp function, const Topology &topology) {
    auto reserve_to_phi_map = buildReserveToPhiMap(function);
    function.walk([&](Operation *op){ if (isCtrlMov(op)) expandCtrlMov(op, topology, reserve_to_phi_map); });
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (auto func : module.getOps<func::FuncOp>()) {
      auto accel = func->getAttrOfType<StringAttr>("accelerator");
      if (!accel || accel.getValue() != "neura") continue;

      auto [columns, rows] = getArrayDimensions(func);
      Topology topo = getTopologyFromArchitecture(columns, rows);

      clearState();
      setTilePlacements(func);

      // 1) Materialize compute/logic ops
      materializeComputeOps(func);

      // 2) Wire by expanding forwarders (data/control mov)
      wireEdges(func, topo);
      handleCtrlMov(func, topo);

      // 3) Log unresolved operands
      logUnresolvedOperands(module);

      // 4) Generate outputs
      ArrayConfig config = buildArrayConfig(columns, rows);
      writeYAMLOutput(config);
      writeASMOutput(config);
    }
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createGenerateCodePass() {
  return std::make_unique<GenerateCodePass>();
}
} // namespace mlir::neura
