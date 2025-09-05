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
#include <queue>
#include <set>
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

// Entry represents a basic block or execution context within a tile. Currently, each instruction is in a single entry.
// This is an option to extend to multiple entries per tile for context switching in the future.
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

// Use dyn_cast for more efficient and type-safe operation checking
static bool isDataMov(Operation *op) { 
  return dyn_cast<DataMovOp>(op) != nullptr; 
}
static bool isCtrlMov(Operation *op) { 
  return dyn_cast<CtrlMovOp>(op) != nullptr; 
}
static bool isPhi(Operation *op) { 
  return dyn_cast<PhiOp>(op) != nullptr; 
}
static bool isReserve(Operation *op) { 
  return dyn_cast<ReserveOp>(op) != nullptr; 
}
static bool isConstant(Operation *op) { 
  return dyn_cast<ConstantOp>(op) != nullptr; 
}
// Only neura.data_mov is treated as "forwarder" (pure routing step) for value chasing.
static bool isForwarder(Operation *op) { 
  return isDataMov(op); 
}

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
  // Assert that tile_location was properly assigned if we found tile mapping
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

// Literal for CONSTANT's source: "#10" / "#0" / "#3.0"
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

// Chases up the operation chain to find the nearest ancestor that has been assigned to a tile.
static Operation *getFirstAncestorMappedOnTile(Operation *op,
                                               const DenseMap<Operation*, TileLocation> &op_placements) {
  SmallPtrSet<Operation*, 32> visited_operations;  // Prevents infinite loops in cyclic graphs
  Operation *current_operation = op;
  while (current_operation && !op_placements.lookup(current_operation).has_tile && !visited_operations.count(current_operation)) {
    visited_operations.insert(current_operation);
    if (current_operation->getNumOperands() == 0) break;  // No more operands to follow
    current_operation = current_operation->getOperand(0).getDefiningOp();  // Follows the first operand
  }
  return (current_operation && op_placements.lookup(current_operation).has_tile) ? current_operation : op;
}

// ----- Topology built from Architecture -----

struct Topology {
  DenseMap<int, std::pair<int,int>> link_ends;      // link_id -> (srcTileId, dstTileId)
  DenseMap<int, std::pair<int,int>> tile_location;  // tileId -> (x,y)

  StringRef getDirBetween(int srcTid, int dstTid) const {
    auto [src_x, src_y] = tile_location.lookup(srcTid);
    auto [dst_x, dst_y] = tile_location.lookup(dstTid);
    int dist_cols = dst_x - src_x, dist_rows = dst_y - src_y;
    if (dist_cols == 1 && dist_rows == 0) return "EAST";
    if (dist_cols == -1 && dist_rows == 0) return "WEST";
    if (dist_cols == 0 && dist_rows == 1) return "NORTH";
    if (dist_cols == 0 && dist_rows == -1) return "SOUTH";
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
};

static Topology getTopologyFromArchitecture(int columns, int rows) {
  Topology topo;
  mlir::neura::Architecture arch(columns, rows);

  for (auto *tile : arch.getAllTiles()) {
    topo.tile_location[tile->getId()] = {tile->getX(), tile->getY()};
  }
  for (auto *link : arch.getAllLinks()) {
    auto *src_tile = link->getSrcTile();
    auto *dst_tile = link->getDstTile();
    topo.link_ends[link->getId()] = {src_tile->getId(), dst_tile->getId()};
  }
  return topo;
}

// ----- Extract mapping steps (sorted by time) from mapping_locs -----

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

// Only checks for register id on the last forwarder (op) at consumer side or on op itself.
static std::optional<int> getRegisterIdOnBackwardPath(Value value_at_consumer) {
  Operation *defining_operation = value_at_consumer.getDefiningOp();
  if (!defining_operation) return std::nullopt;
  if (isDataMov(defining_operation) || isCtrlMov(defining_operation)) {
    if (auto register_steps = collectRegSteps(defining_operation); !register_steps.empty())
      return register_steps.back().regId; // In your IR, ids are identical, so takes the last one
  }
  return getMappedRegId(defining_operation);
}

// Mandatory requirement: same-tile cross-step must have assigned register; otherwise error and assert.
static std::string getAssignedRegisterOrFail(Value value_at_consumer, Operation *producer) {
  if (auto register_id = getRegisterIdOnBackwardPath(value_at_consumer))
    return "$" + std::to_string(*register_id);
  if (auto register_id = getMappedRegId(producer))
    return "$" + std::to_string(*register_id);

  producer->emitError("same-tile cross-step edge without assigned register id (mapping pass must assign one).");
  assert(false && "register id must be assigned by mapping pass for same-tile cross-step");
  return "$0"; // unreachable, silences compiler warning
}

// ----- Pass -----

struct InstructionReference { int col_idx, row_idx, t, idx; };

struct ProducerSummary {
  std::set<std::string> dirs;   // directions to remote consumers (keep for dst operands)
  bool hasSameTileConsumer = false;
};

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

  // to get Instruction* back when wiring operands
  DenseMap<Operation*, InstructionReference>            operation_to_instruction_reference;
  DenseMap<Operation*, SmallVector<Value>> operation_to_operands;

  // de-dup sets
  std::unordered_set<uint64_t> hop_signatures;     // (midTileId, ts, link_id)
  std::unordered_set<uint64_t> deposit_signatures; // (dstTileId, ts, regId)

  // ---------- helpers to place materialized instructions ----------
  void placeRouterHop(const Topology &topology, int tile_id, int time_step,
                      StringRef input_direction, StringRef output_direction) {
    auto [tile_x, tile_y] = topology.tile_location.lookup(tile_id);
    Instruction instruction("DATA_MOV");
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

      // If operation itself mapped a register (rare), keeps as destination register.
      if (auto mapped_register_id = getMappedRegId(operation))
        instruction.dst_operands.emplace_back("$" + std::to_string(*mapped_register_id), "RED");

      auto &instruction_vector = getInstructionBucket(placement.col_idx, placement.row_idx, placement.time_step);
      instruction_vector.push_back(std::move(instruction));
      operation_to_instruction_reference[operation] = InstructionReference{placement.col_idx, placement.row_idx, placement.time_step, (int)instruction_vector.size() - 1};
    });
  }

  // ---------- wiring logic ----------
  SmallVector<std::pair<Operation*, Value>, 8> findConsumers(Value result) {
    SmallVector<std::pair<Operation*, Value>, 8> consumers;
    SmallPtrSet<Value, 32> seen_values;
    std::queue<Value> value_queue; value_queue.push(result); seen_values.insert(result);
    while (!value_queue.empty()) {
      Value current_value = value_queue.front(); value_queue.pop();
      for (OpOperand &use : current_value.getUses()) {
        Operation *use_operation = use.getOwner();
        if (operation_placements.lookup(use_operation).has_tile) {
          consumers.emplace_back(use_operation, current_value);
        } else if (isForwarder(use_operation)) {
          if (use_operation->getNumResults() > 0) {
            Value next_value = use_operation->getResult(0);
            if (!seen_values.count(next_value)) { seen_values.insert(next_value); value_queue.push(next_value); }
          }
        }
      }
    }
    return consumers;
  }

  void wireSameTileEdge(Operation *producer, Operation *consumer, Value value_at_consumer, 
                       DenseMap<Operation*, ProducerSummary> &producer_summary) {
    std::string register_name = getAssignedRegisterOrFail(value_at_consumer, producer);
    setConsumerSourceExact(consumer, value_at_consumer, register_name);
    if (Instruction *producer_instruction = getInstructionPointer(producer)) setUniqueDestination(producer_instruction, register_name);
    producer_summary[producer].hasSameTileConsumer = true;
  }

  void wireCrossTileEdge(Operation *producer, Operation *consumer, Value value_at_consumer,
                        const Topology &topology, DenseMap<Operation*, ProducerSummary> &producer_summary) {
    auto producer_placement = operation_placements.lookup(getFirstAncestorMappedOnTile(producer, operation_placements));
    auto consumer_placement = operation_placements.lookup(consumer);
    if (!producer_placement.has_tile || !consumer_placement.has_tile) return;

    auto link_chain = getLinkChainForValueAtConsumer(value_at_consumer);
    auto register_steps = getRegisterStepsForValueAtConsumer(value_at_consumer);

    std::string producer_direction = "LOCAL";
    std::string consumer_direction = "LOCAL";

    if (!link_chain.empty()) {
      int first_link = link_chain.front().link_id;
      int last_link  = link_chain.back().link_id;
      producer_direction = topology.dirFromLink(first_link).str();
      consumer_direction = topology.invertDir(topology.dirFromLink(last_link)).str();
    } else {
      int column_delta = consumer_placement.col_idx - producer_placement.col_idx;
      int row_delta = consumer_placement.row_idx - producer_placement.row_idx;
      if (column_delta > 0 && row_delta == 0) { producer_direction = "EAST"; consumer_direction = "WEST"; }
      else if (column_delta < 0 && row_delta == 0) { producer_direction = "WEST"; consumer_direction = "EAST"; }
      else if (column_delta == 0 && row_delta > 0) { producer_direction = "NORTH"; consumer_direction = "SOUTH"; }
      else if (column_delta == 0 && row_delta < 0) { producer_direction = "SOUTH"; consumer_direction = "NORTH"; }
    }

    // record producer endpoint directions
    if (Instruction *producer_instruction = getInstructionPointer(producer)) setUniqueDestination(producer_instruction, producer_direction);
    producer_summary[producer].dirs.insert(producer_direction);

    // endpoint deposit if the forwarder before consumer contains register mapping
    bool rewired_to_register = false;
    if (!link_chain.empty() && !register_steps.empty()) {
      int last_link  = link_chain.back().link_id;
      int destination_tile_id = topology.dstTileOfLink(last_link);
      auto tile_location_iterator = topology.tile_location.find(destination_tile_id);
      if (tile_location_iterator != topology.tile_location.end()) {
        auto [destination_tile_x, destination_tile_y] = tile_location_iterator->second;
        if (destination_tile_x == consumer_placement.col_idx && destination_tile_y == consumer_placement.row_idx) {
          const int earliest_register_time_step = register_steps.front().ts;
          const int register_id = register_steps.back().regId;
          placeDstDeposit(topology, destination_tile_id, earliest_register_time_step, consumer_direction, register_id, /*asCtrlMov=*/false);
          if (consumer_placement.time_step > earliest_register_time_step) {
            setConsumerSourceExact(consumer, value_at_consumer, "$" + std::to_string(register_id));
            rewired_to_register = true;
          }
        }
      }
    }
    if (!rewired_to_register) setConsumerSourceExact(consumer, value_at_consumer, consumer_direction);
  }

  void wireEdges(func::FuncOp function, const Topology &topology) {
    DenseMap<Operation*, ProducerSummary> producer_summary;
    
    function.walk([&](Operation *producer){
      if (isReserve(producer) || isCtrlMov(producer)) return;
      for (Value result : producer->getResults()) {
        auto consumers = findConsumers(result);
        for (auto &consumer_pair : consumers) {
          Operation *consumer = consumer_pair.first;
          Value value_at_consumer = consumer_pair.second;

          Operation *tiled_producer = getFirstAncestorMappedOnTile(producer, operation_placements);
          auto producer_placement = operation_placements.lookup(tiled_producer);
          auto consumer_placement = operation_placements.lookup(consumer);
          if (!producer_placement.has_tile || !consumer_placement.has_tile) continue;

          if (producer_placement.col_idx == consumer_placement.col_idx && producer_placement.row_idx == consumer_placement.row_idx) {
            wireSameTileEdge(producer, consumer, value_at_consumer, producer_summary);
          } else {
            wireCrossTileEdge(producer, consumer, value_at_consumer, topology, producer_summary);
          }
        }
      }
    });

    // Producer post-process
    for (auto &key_value_pair : producer_summary) {
      Operation *operation = key_value_pair.first;
      Instruction *producer_instruction = getInstructionPointer(operation);
      if (!producer_instruction) continue;
      for (const auto &direction : key_value_pair.second.dirs) setUniqueDestination(producer_instruction, direction);

      const bool has_mapped_register = getMappedRegId(operation).has_value();
      if (!key_value_pair.second.hasSameTileConsumer && !has_mapped_register) {
        producer_instruction->dst_operands.erase(
          std::remove_if(producer_instruction->dst_operands.begin(), producer_instruction->dst_operands.end(),
            [](const Operand &operand){ return !operand.operand.empty() && operand.operand[0] == '$'; }),
          producer_instruction->dst_operands.end()
        );
      }
    }
  }

  // ---------- ctrl_mov handling ----------
  DenseMap<Value, Operation*> buildReserveToPhiMap(func::FuncOp function) {
    DenseMap<Value, Operation*> reserve_to_phi_map;
    function.walk([&](Operation *operation){
      if (isPhi(operation) && operation->getNumOperands() >= 1)
        reserve_to_phi_map[operation->getOperand(0)] = operation;
    });
    return reserve_to_phi_map;
  }

  void handleSameTileCtrlMov(Operation *operation, Value source, Value reserve, 
                            const DenseMap<Value, Operation*> &reserve_to_phi_map,
                            DenseMap<Operation*, ProducerSummary> &producer_summary) {
    Operation *phi_operation = reserve_to_phi_map.lookup(reserve);
    if (!phi_operation) {
      operation->emitWarning("ctrl_mov dest is not consumed by a PHI operand#0; skipping ctrl_mov-specific wiring.");
      return;
    }
    Operation *producer = source.getDefiningOp();
    if (!producer) return;

    auto producer_placement = operation_placements.lookup(getFirstAncestorMappedOnTile(producer, operation_placements));
    auto phi_placement = operation_placements.lookup(phi_operation);
    if (!producer_placement.has_tile || !phi_placement.has_tile) return;

    auto register_steps = collectRegSteps(operation);
    std::string register_name;
    if (!register_steps.empty()) register_name = "$" + std::to_string(register_steps.back().regId);
    else if (auto mapped_register_id = getMappedRegId(operation)) register_name = "$" + std::to_string(*mapped_register_id);
    else {
      operation->emitError("same-tile ctrl_mov without assigned register id (mapping pass must assign one).");
      assert(false && "register id must be assigned by mapping pass for same-tile ctrl_mov");
    }
    setConsumerSourceExact(phi_operation, source, register_name);
    if (Instruction *producer_instruction = getInstructionPointer(producer)) setUniqueDestination(producer_instruction, register_name);
    producer_summary[producer].hasSameTileConsumer = true;
  }

  void handleCrossTileCtrlMov(Operation *operation, Value source, Value reserve,
                             const DenseMap<Value, Operation*> &reserve_to_phi_map,
                             const Topology &topology, DenseMap<Operation*, ProducerSummary> &producer_summary) {
    Operation *phi_operation = reserve_to_phi_map.lookup(reserve);
    if (!phi_operation) {
      operation->emitWarning("ctrl_mov dest is not consumed by a PHI operand#0; skipping ctrl_mov-specific wiring.");
      return;
    }
    Operation *producer = source.getDefiningOp();
    if (!producer) return;

    auto producer_placement = operation_placements.lookup(getFirstAncestorMappedOnTile(producer, operation_placements));
    auto phi_placement = operation_placements.lookup(phi_operation);
    if (!producer_placement.has_tile || !phi_placement.has_tile) return;

    auto link_chain  = collectLinkSteps(operation);
    auto register_steps = collectRegSteps(operation);

    std::string producer_direction = "LOCAL";
    std::string consumer_direction = "LOCAL";
    if (!link_chain.empty()) {
      producer_direction = topology.dirFromLink(link_chain.front().link_id).str();
      consumer_direction = topology.invertDir(topology.dirFromLink(link_chain.back().link_id)).str();
    } else {
      int column_delta = phi_placement.col_idx - producer_placement.col_idx;
      int row_delta = phi_placement.row_idx - producer_placement.row_idx;
      if (column_delta > 0 && row_delta == 0) { producer_direction = "EAST"; consumer_direction = "WEST"; }
      else if (column_delta < 0 && row_delta == 0) { producer_direction = "WEST"; consumer_direction = "EAST"; }
      else if (column_delta == 0 && row_delta > 0) { producer_direction = "NORTH"; consumer_direction = "SOUTH"; }
      else if (column_delta == 0 && row_delta < 0) { producer_direction = "SOUTH"; consumer_direction = "NORTH"; }
    }
    if (Instruction *producer_instruction = getInstructionPointer(producer)) setUniqueDestination(producer_instruction, producer_direction);
    producer_summary[producer].dirs.insert(producer_direction);

    bool rewired_to_register = false;
    if (!link_chain.empty() && !register_steps.empty()) {
      int last_link  = link_chain.back().link_id;
      int destination_tile_id = topology.dstTileOfLink(last_link);
      auto tile_location_iterator = topology.tile_location.find(destination_tile_id);
      if (tile_location_iterator != topology.tile_location.end()) {
        auto [destination_tile_x, destination_tile_y] = tile_location_iterator->second;
        if (destination_tile_x == phi_placement.col_idx && destination_tile_y == phi_placement.row_idx) {
          const int earliest_register_time_step = register_steps.front().ts;
          const int register_id = register_steps.back().regId;
          placeDstDeposit(topology, destination_tile_id, earliest_register_time_step, consumer_direction, register_id, /*asCtrlMov=*/true);
          if (phi_placement.time_step > earliest_register_time_step) {
            setConsumerSourceExact(phi_operation, source, "$" + std::to_string(register_id));
            rewired_to_register = true;
          }
        }
      }
    }
    if (!rewired_to_register) setConsumerSourceExact(phi_operation, source, consumer_direction);
  }

  void handleCtrlMov(func::FuncOp function, const Topology &topology) {
    auto reserve_to_phi_map = buildReserveToPhiMap(function);
    DenseMap<Operation*, ProducerSummary> producer_summary;

    function.walk([&](Operation *operation){
      if (!isCtrlMov(operation) || operation->getNumOperands() < 2) return;
      Value source = operation->getOperand(0);
      Value reserve = operation->getOperand(1);
      Operation *phi_operation = reserve_to_phi_map.lookup(reserve);
      if (!phi_operation) return;

      Operation *producer = source.getDefiningOp();
      if (!producer) return;

      auto producer_placement = operation_placements.lookup(getFirstAncestorMappedOnTile(producer, operation_placements));
      auto phi_placement = operation_placements.lookup(phi_operation);
      if (!producer_placement.has_tile || !phi_placement.has_tile) return;

      if (producer_placement.col_idx == phi_placement.col_idx && producer_placement.row_idx == phi_placement.row_idx) {
        handleSameTileCtrlMov(operation, source, reserve, reserve_to_phi_map, producer_summary);
      } else {
        handleCrossTileCtrlMov(operation, source, reserve, reserve_to_phi_map, topology, producer_summary);
      }
    });
  }

  // ---------- router generation ----------
  void generateRouters(func::FuncOp function, const Topology &topology) {
    function.walk([&](Operation *operation){
      if (isDataMov(operation) || isCtrlMov(operation)) generateIntermediateRoutersForOperation(operation, topology);
    });
  }

  // ---------- output generation ----------
  void logUnresolvedOperands(ModuleOp module) {
    unsigned unresolved_source_count = 0, unresolved_destination_count = 0;
    for (auto &tile_key_value_pair : tile_time_instructions) {
      auto tile_key = tile_key_value_pair.first;
      int column = tile_key.first, row = tile_key.second;
      for (auto &time_step_key_value_pair : tile_key_value_pair.second) {
        int time_step = time_step_key_value_pair.first;
        auto &instruction_vector = time_step_key_value_pair.second;
        for (size_t instruction_index = 0; instruction_index < instruction_vector.size(); ++instruction_index) {
          auto &instruction = instruction_vector[instruction_index];
          for (size_t source_operand_index = 0; source_operand_index < instruction.src_operands.size(); ++source_operand_index) {
            auto &source_operand = instruction.src_operands[source_operand_index];
            if (source_operand.operand == "UNRESOLVED") {
              source_operand.color = "YELLOW";
              ++unresolved_source_count;
              llvm::errs() << "[UNRESOLVED SRC] tile("<<column<<","<<row<<") t="<<time_step
                           << " inst#" << instruction_index << " op=" << instruction.opcode
                           << " src_idx=" << source_operand_index << "\n";
            }
          }
          instruction.dst_operands.erase(
            std::remove_if(instruction.dst_operands.begin(), instruction.dst_operands.end(),
              [](const Operand &operand){
                return operand.operand.empty() || operand.operand=="UNKNOWN";
              }),
            instruction.dst_operands.end());
          for (size_t destination_operand_index = 0; destination_operand_index < instruction.dst_operands.size(); ++destination_operand_index) {
            auto &destination_operand = instruction.dst_operands[destination_operand_index];
            if (destination_operand.operand == "UNRESOLVED") {
              destination_operand.color = "YELLOW";
              ++unresolved_destination_count;
              llvm::errs() << "[UNRESOLVED DST] tile("<<column<<","<<row<<") t="<<time_step
                           << " inst#" << instruction_index << " op=" << instruction.opcode
                           << " dst_idx=" << destination_operand_index << "\n";
            }
          }
        }
      }
    }
    if (unresolved_source_count + unresolved_destination_count) {
      auto diagnostic = module.emitWarning("GenerateCodePass: UNRESOLVED operands kept for debugging");
      diagnostic << " (src=" << unresolved_source_count << ", dst=" << unresolved_destination_count
           << "); they are highlighted with color=YELLOW in YAML.";
    }
  }

  ArrayConfig buildArrayConfig(int columns, int rows) {
    ArrayConfig config;
    config.columns = columns;
    config.rows = rows;

    // flatten & sort by timestep
    std::map<std::pair<int,int>, std::vector<Instruction>> tile_instructions;
    for (auto &tile_key_value_pair : tile_time_instructions) {
      auto tile_key = tile_key_value_pair.first;
      auto &time_step_map = tile_key_value_pair.second;
      auto &flattened_instructions = tile_instructions[tile_key];
      for (auto &time_step_key_value_pair : time_step_map) for (auto &instruction : time_step_key_value_pair.second) flattened_instructions.push_back(instruction);
      std::stable_sort(flattened_instructions.begin(), flattened_instructions.end(),
        [](const Instruction &instruction_a, const Instruction &instruction_b){ return instruction_a.time_step < instruction_b.time_step; });
    }

    for (int row = 0; row < rows; ++row) {
      for (int column = 0; column < columns; ++column) {
        auto tile_iterator = tile_instructions.find({column,row});
        if (tile_iterator == tile_instructions.end()) continue;
        Tile tile(column, row, row*columns + column);
        for (auto &instruction : tile_iterator->second) tile.entry.instructions.push_back(instruction);
        config.cores.push_back(std::move(tile));
      }
    }
    return config;
  }

  void writeYAMLOutput(const ArrayConfig &config) {
    std::error_code ec;
    llvm::raw_fd_ostream yaml_out("generated-instructions.yaml", ec);
    if (ec) {
      // Error handling would need to be passed up
      return;
    }

    yaml_out << "array_config:\n";
    yaml_out << "  columns: " << config.columns << "\n";
    yaml_out << "  rows: " << config.rows << "\n";
    yaml_out << "  cores:\n";
    for (const auto &core : config.cores) {
      yaml_out << "    - column: " << core.col_idx << "\n";
      yaml_out << "      row: " << core.row_idx << "\n";
      yaml_out << "      core_id: \"" << core.core_id << "\"\n";
      yaml_out << "      entries:\n";
      int entry_id = 0;
      for (const auto &inst : core.entry.instructions) {
        yaml_out << "        - entry_id: \"entry" << entry_id << "\"\n";
        yaml_out << "          instructions:\n";
        yaml_out << "            - opcode: \"" << inst.opcode << "\"\n";
        yaml_out << "              timestep: " << inst.time_step << "\n";
        if (!inst.src_operands.empty()) {
          yaml_out << "              src_operands:\n";
          for (const auto &opnd : inst.src_operands) {
            yaml_out << "                - operand: \"" << opnd.operand << "\"\n";
            yaml_out << "                  color: \"" << opnd.color << "\"\n";
          }
        }
        if (!inst.dst_operands.empty()) {
          yaml_out << "              dst_operands:\n";
          for (const auto &opnd : inst.dst_operands) {
            yaml_out << "                - operand: \"" << opnd.operand << "\"\n";
            yaml_out << "                  color: \"" << opnd.color << "\"\n";
          }
        }
        entry_id++;
      }
    }
    yaml_out.close();
  }

  void writeASMOutput(const ArrayConfig &config) {
    std::error_code ec;
    llvm::raw_fd_ostream asm_out("generated-instructions.asm", ec);
    if (ec) {
      // Error handling would need to be passed up
      return;
    }
    for (const auto &core : config.cores) {
      asm_out << "PE(" << core.col_idx << "," << core.row_idx << "):\n";
      for (const auto &inst : core.entry.instructions) {
        asm_out << "{\n";
        asm_out << "  " << inst.opcode;
        if (!inst.src_operands.empty()) {
          for (size_t i=0;i<inst.src_operands.size();++i) {
            asm_out << ", [" << inst.src_operands[i].operand << "]";
          }
        }
        if (!inst.dst_operands.empty()) {
          asm_out << " -> ";
          for (size_t i=0;i<inst.dst_operands.size();++i) {
            if (i > 0) asm_out << ", ";
            asm_out << "[" << inst.dst_operands[i].operand << "]";
          }
        }
        asm_out << " (t=" << inst.time_step << ")\n";
        asm_out << "}\n";
      }
      asm_out << "\n";
    }
    asm_out.close();
  }

  // endpoint deposit: on destination tile at earliest reg ts, move [incoming_dir] -> [$reg]
  // If `asCtrlMov=true`, encode the deposit as "CTRL_MOV" instead of "DATA_MOV".
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

  void generateIntermediateRoutersForOperation(Operation *operation, const Topology &topology) {
    auto steps = collectLinkSteps(operation);
    if (steps.size() < 2) return; // only multi-hop
    for (size_t i = 1; i < steps.size(); ++i) {
      int prev_link = steps[i - 1].link_id;
      int current_link  = steps[i].link_id;
      int time_step    = steps[i].ts;

      int middle_tile = topology.srcTileOfLink(current_link);
      int previous_destination = topology.dstTileOfLink(prev_link);
      if (previous_destination != middle_tile) {
        operation->emitWarning() << "discontinuous multi-link chain: dst(prev)="
                          << previous_destination << ", src(cur)=" << middle_tile
                          << " (placing hop at src(cur))";
      }

      StringRef input_direction  = topology.invertDir(topology.dirFromLink(prev_link));
      StringRef output_direction = topology.dirFromLink(current_link);

      uint64_t signature = (uint64_t)middle_tile << 32 ^ (uint64_t)time_step << 16 ^ (uint64_t)current_link;
      if (hop_signatures.insert(signature).second) placeRouterHop(topology, middle_tile, time_step, input_direction, output_direction);
    }
  }

  // utilities to access instruction bucket and pointer
  std::vector<Instruction> &getInstructionBucket(int column, int row, int time_step) {
    return tile_time_instructions[{column,row}][time_step];
  }
  Instruction* getInstructionPointer(Operation *operation) {
    auto iterator = operation_to_instruction_reference.find(operation);
    if (iterator == operation_to_instruction_reference.end()) return nullptr;
    auto [column, row, time_step, index] = iterator->second;
    auto &instruction_vector = tile_time_instructions[{column,row}][time_step];
    if (index < 0 || index >= (int)instruction_vector.size()) return nullptr;
    return &instruction_vector[index];
  }

  void setConsumerSourceExact(Operation *consumer, Value value_at_consumer, const std::string &text) {
    Instruction *consumer_instruction = getInstructionPointer(consumer);
    if (!consumer_instruction) return;
    auto operand_iterator = operation_to_operands.find(consumer);
    if (operand_iterator == operation_to_operands.end()) return;
    auto &operands = operand_iterator->second;
    for (size_t i = 0; i < operands.size() && i < consumer_instruction->src_operands.size(); ++i) {
      if (operands[i] == value_at_consumer) { consumer_instruction->src_operands[i].operand = text; return; }
    }
    for (auto &source_operand : consumer_instruction->src_operands)
      if (source_operand.operand == "UNRESOLVED") { source_operand.operand = text; return; }
  }

  static void setUniqueDestination(Instruction *instruction, const std::string &text){
    for (auto &destination_operand : instruction->dst_operands) if (destination_operand.operand == text) return;
    instruction->dst_operands.emplace_back(text, "RED");
  }

  // Reads link/reg steps sitting on the last forwarder (if any) before the consumer.
  static SmallVector<LinkStep, 8> getLinkChainForValueAtConsumer(Value value_at_consumer) {
    Operation *defining_operation = value_at_consumer.getDefiningOp();
    if (!defining_operation) return {};
    if (isDataMov(defining_operation) || isCtrlMov(defining_operation)) return collectLinkSteps(defining_operation);
    return {};
  }
  static SmallVector<RegStep, 4> getRegisterStepsForValueAtConsumer(Value value_at_consumer) {
    Operation *defining_operation = value_at_consumer.getDefiningOp();
    if (!defining_operation) return {};
    if (isDataMov(defining_operation) || isCtrlMov(defining_operation)) return collectRegSteps(defining_operation);
    return {};
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

      // 2) Wire edges
      wireEdges(func, topo);

      // 2.1) Handle ctrl_mov
      handleCtrlMov(func, topo);

      // 3) Generate routers
      generateRouters(func, topo);

      // 4) Log unresolved operands
      logUnresolvedOperands(module);

      // 5) Generate outputs
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
