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
  bool hasTile = false;
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
      tile_location.hasTile = true;
      break;
    }
  }
  // Assert that tile_location was properly assigned if we found tile mapping
  assert(!tile_location.hasTile || (tile_location.col_idx >= 0 && tile_location.row_idx >= 0));
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

static std::string mapOpcode(Operation *op) {
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

// Chase up the operation chain to find the nearest ancestor that has been assigned to a tile.
static Operation *chaseToTiledProducer(Operation *op,
                                       const DenseMap<Operation*, TileLocation> &opPlace) {
  SmallPtrSet<Operation*, 32> visited_ops;  // Prevent infinite loops in cyclic graphs
  Operation *current_op = op;
  while (current_op && !opPlace.lookup(current_op).hasTile && !visited_ops.count(current_op)) {
    visited_ops.insert(current_op);
    if (current_op->getNumOperands() == 0) break;  // No more operands to follow
    current_op = current_op->getOperand(0).getDefiningOp();  // Follow the first operand
  }
  return (current_op && opPlace.lookup(current_op).hasTile) ? current_op : op;
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

static Topology buildTopologyFromArchitecture(int columns, int rows) {
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

// Only check for register id on the last forwarder(op) at consumer side or on op itself.
static std::optional<int> regOnBackwardPath(Value vAtConsumer) {
  Operation *def = vAtConsumer.getDefiningOp();
  if (!def) return std::nullopt;
  if (isDataMov(def) || isCtrlMov(def)) {
    if (auto steps = collectRegSteps(def); !steps.empty())
      return steps.back().regId; // In your IR, ids are identical, so take the last one
  }
  return getMappedRegId(def);
}

// Mandatory requirement: same-tile cross-step must have assigned register; otherwise error and assert.
static std::string pickAssignedRegisterOrDie(Value vAtConsumer, Operation *producer) {
  if (auto id = regOnBackwardPath(vAtConsumer))
    return "$" + std::to_string(*id);
  if (auto id = getMappedRegId(producer))
    return "$" + std::to_string(*id);

  producer->emitError("same-tile cross-step edge without assigned register id (mapping pass must assign one).");
  assert(false && "register id must be assigned by mapping pass for same-tile cross-step");
  return "$0"; // unreachable, silences compiler warning
}

// ----- Pass -----

struct InstRef { int col_idx, row_idx, t, idx; };

struct ProdSummary {
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

  DenseMap<Operation*, TileLocation>   opPlace;

  // Map of tile coordinates (x,y) -> time_step -> vector of Instructions
  std::map<std::pair<int,int>, std::map<int, std::vector<Instruction>>> tile_time_insts;

  // to get Instruction* back when wiring operands
  DenseMap<Operation*, InstRef>            op2ref;
  DenseMap<Operation*, SmallVector<Value>> op2Operands;

  // de-dup sets
  std::unordered_set<uint64_t> hopSig;     // (midTileId, ts, link_id)
  std::unordered_set<uint64_t> depositSig; // (dstTileId, ts, regId)

  // ---------- helpers to place materialized instructions ----------
  void placeRouterHop(const Topology &topo, int tileId, int ts,
                      StringRef inDir, StringRef outDir) {
    auto [tile_x, tile_y] = topo.tile_location.lookup(tileId);
    Instruction inst("DATA_MOV");
    inst.time_step = ts;
    inst.src_operands.emplace_back(inDir.str(), "RED");
    inst.dst_operands.emplace_back(outDir.str(), "RED");
    tile_time_insts[{tile_x, tile_y}][ts].push_back(std::move(inst));
  }

  // endpoint deposit: on destination tile at earliest reg ts, move [incoming_dir] -> [$reg]
  // If `asCtrlMov=true`, encode the deposit as "CTRL_MOV" instead of "DATA_MOV".
  void placeDstDeposit(const Topology &topo, int dstTileId, int ts,
                       StringRef incomingDir, int regId, bool asCtrlMov = false) {
    uint64_t signature = (uint64_t)dstTileId << 32 ^ (uint64_t)ts << 16 ^ (uint64_t)regId;
    if (!depositSig.insert(signature).second) return; // already placed
    auto [tile_x, tile_y] = topo.tile_location.lookup(dstTileId);
    Instruction inst(asCtrlMov ? "CTRL_MOV" : "DATA_MOV");
    inst.time_step = ts;
    inst.src_operands.emplace_back(incomingDir.str(), "RED");
    inst.dst_operands.emplace_back("$" + std::to_string(regId), "RED");
    tile_time_insts[{tile_x, tile_y}][ts].push_back(std::move(inst));
  }

  void emitIntermediateRoutersForOp(Operation *op, const Topology &topo) {
    auto steps = collectLinkSteps(op);
    if (steps.size() < 2) return; // only multi-hop
    for (size_t i = 1; i < steps.size(); ++i) {
      int prevL = steps[i - 1].link_id;
      int curL  = steps[i].link_id;
      int ts    = steps[i].ts;

      int midTile = topo.srcTileOfLink(curL);
      int prevDst = topo.dstTileOfLink(prevL);
      if (prevDst != midTile) {
        op->emitWarning() << "discontinuous multi-link chain: dst(prev)="
                          << prevDst << ", src(cur)=" << midTile
                          << " (placing hop at src(cur))";
      }

      StringRef inDir  = topo.invertDir(topo.dirFromLink(prevL));
      StringRef outDir = topo.dirFromLink(curL);

      uint64_t sig = (uint64_t)midTile << 32 ^ (uint64_t)ts << 16 ^ (uint64_t)curL;
      if (hopSig.insert(sig).second) placeRouterHop(topo, midTile, ts, inDir, outDir);
    }
  }

  // utilities to access instruction bucket and pointer
  std::vector<Instruction> &bucketOf(int c, int r, int t) {
    return tile_time_insts[{c,r}][t];
  }
  Instruction* getInstPtr(Operation *op) {
    auto it = op2ref.find(op);
    if (it == op2ref.end()) return nullptr;
    auto [c,r,t,idx] = it->second;
    auto &vec = tile_time_insts[{c,r}][t];
    if (idx < 0 || idx >= (int)vec.size()) return nullptr;
    return &vec[idx];
  }

  void setConsumerSrcExact(Operation *consumer, Value vAtConsumer, const std::string &text) {
    Instruction *ci = getInstPtr(consumer);
    if (!ci) return;
    auto itOps = op2Operands.find(consumer);
    if (itOps == op2Operands.end()) return;
    auto &ops = itOps->second;
    for (size_t i = 0; i < ops.size() && i < ci->src_operands.size(); ++i) {
      if (ops[i] == vAtConsumer) { ci->src_operands[i].operand = text; return; }
    }
    for (auto &s : ci->src_operands)
      if (s.operand == "UNRESOLVED") { s.operand = text; return; }
  }

  static void addUniqueDst(Instruction *inst, const std::string &text){
    for (auto &d : inst->dst_operands) if (d.operand == text) return;
    inst->dst_operands.emplace_back(text, "RED");
  }

  // Read link/reg steps sitting on the last forwarder (if any) before the consumer.
  static SmallVector<LinkStep, 8> linkChainForValueAtConsumer(Value vAtCons) {
    Operation *def = vAtCons.getDefiningOp();
    if (!def) return {};
    if (isDataMov(def) || isCtrlMov(def)) return collectLinkSteps(def);
    return {};
  }
  static SmallVector<RegStep, 4> regStepsForValueAtConsumer(Value vAtCons) {
    Operation *def = vAtCons.getDefiningOp();
    if (!def) return {};
    if (isDataMov(def) || isCtrlMov(def)) return collectRegSteps(def);
    return {};
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (auto func : module.getOps<func::FuncOp>()) {
      auto accel = func->getAttrOfType<StringAttr>("accelerator");
      if (!accel || accel.getValue() != "neura") continue;

      int columns = 4, rows = 4;
      if (auto mi = func->getAttrOfType<DictionaryAttr>("mapping_info")) {
        if (auto xv = dyn_cast_or_null<IntegerAttr>(mi.get("x_tiles"))) columns = xv.getInt();
        if (auto yv = dyn_cast_or_null<IntegerAttr>(mi.get("y_tiles"))) rows   = yv.getInt();
      }

      Topology topo = buildTopologyFromArchitecture(columns, rows);

      // clear state
      opPlace.clear();
      tile_time_insts.clear();
      op2ref.clear();
      op2Operands.clear();
      hopSig.clear();
      depositSig.clear();

      // cache tile placements
      func.walk([&](Operation *op){ opPlace[op] = getTileLocation(op); });

      // 1) Materialize compute/logic ops (CONSTANT/ADD/FADD/PHI/ICMP/NOT/GRANT_* etc.)
      func.walk([&](Operation *op){
        auto p = opPlace[op];
        if (!p.hasTile) return;
        if (isReserve(op) || isCtrlMov(op) || isDataMov(op)) return;

        std::string opc = mapOpcode(op);
        Instruction inst(opc);
        inst.time_step = p.time_step;

        if (isConstant(op)) {
          inst.src_operands.emplace_back(getConstantLiteral(op), "RED");
        } else {
          SmallVector<Value> operands; operands.reserve(op->getNumOperands());
          for (Value in : op->getOperands()) {
            operands.push_back(in);
            inst.src_operands.emplace_back("UNRESOLVED", "RED");
          }
          op2Operands[op] = std::move(operands);
        }

        // If op itself mapped a register (rare), keep as dst reg.
        if (auto mr = getMappedRegId(op))
          inst.dst_operands.emplace_back("$" + std::to_string(*mr), "RED");

        auto &vec = bucketOf(p.col_idx, p.row_idx, p.time_step);
        vec.push_back(std::move(inst));
        op2ref[op] = InstRef{p.col_idx, p.row_idx, p.time_step, (int)vec.size() - 1};
      });

      // Summaries for producers
      DenseMap<Operation*, ProdSummary> prodSum;

      // 2) Wire edges (same tile: $reg; cross tile: direction, plus endpoint deposit if reg-steps exist)
      func.walk([&](Operation *prod){
        if (isReserve(prod) || isCtrlMov(prod)) return; // ctrl_mov handled later
        for (Value res : prod->getResults()) {
          // BFS to final tiled consumers, skipping neura.data_mov
          SmallVector<std::pair<Operation*, Value>, 8> consumers;
          {
            SmallPtrSet<Value, 32> seenV;
            std::queue<Value> q; q.push(res); seenV.insert(res);
            while (!q.empty()) {
              Value v = q.front(); q.pop();
              for (OpOperand &use : v.getUses()) {
                Operation *u = use.getOwner();
                if (opPlace.lookup(u).hasTile) {
                  consumers.emplace_back(u, v);
                } else if (isForwarder(u)) {
                  if (u->getNumResults() > 0) {
                    Value nv = u->getResult(0);
                    if (!seenV.count(nv)) { seenV.insert(nv); q.push(nv); }
                  }
                }
              }
            }
          }

          for (auto &cc : consumers) {
            Operation *cons = cc.first;
            Value vAtCons   = cc.second;

            Operation *tp = chaseToTiledProducer(prod, opPlace);
            auto pp = opPlace.lookup(tp);
            auto pc = opPlace.lookup(cons);
            if (!pp.hasTile || !pc.hasTile) continue;

            // steps on the forwarder directly before consumer (if any)
            auto chain = linkChainForValueAtConsumer(vAtCons);
            auto rsteps= regStepsForValueAtConsumer(vAtCons);

            if (pp.col_idx == pc.col_idx && pp.row_idx == pc.row_idx) {
              // same-tile: MUST use assigned register id
              std::string rn = pickAssignedRegisterOrDie(vAtCons, prod);
              setConsumerSrcExact(cons, vAtCons, rn);
              if (Instruction *pi = getInstPtr(prod)) addUniqueDst(pi, rn);
              prodSum[prod].hasSameTileConsumer = true;
            } else {
              // cross-tile: compute endpoint directions
              std::string prodDir = "LOCAL";
              std::string consDir = "LOCAL";

              if (!chain.empty()) {
                int firstLink = chain.front().link_id;
                int lastLink  = chain.back().link_id;
                prodDir = topo.dirFromLink(firstLink).str();
                consDir = topo.invertDir(topo.dirFromLink(lastLink)).str();
              } else {
                int dc = pc.col_idx - pp.col_idx, dr = pc.row_idx - pp.row_idx;
                if (dc > 0 && dr == 0) { prodDir = "EAST"; consDir = "WEST"; }
                else if (dc < 0 && dr == 0) { prodDir = "WEST"; consDir = "EAST"; }
                else if (dc == 0 && dr > 0) { prodDir = "NORTH"; consDir = "SOUTH"; }
                else if (dc == 0 && dr < 0) { prodDir = "SOUTH"; consDir = "NORTH"; }
              }

              // record producer endpoint directions
              if (Instruction *pi = getInstPtr(prod)) addUniqueDst(pi, prodDir);
              prodSum[prod].dirs.insert(prodDir);

              // endpoint deposit if the forwarder before consumer contains register mapping
              bool rewiredToReg = false;
              if (!chain.empty() && !rsteps.empty()) {
                int lastLink  = chain.back().link_id;
                int dstTid    = topo.dstTileOfLink(lastLink);
                auto tile_location_iter = topo.tile_location.find(dstTid);
                if (tile_location_iter != topo.tile_location.end()) {
                  auto [dst_tile_x, dst_tile_y] = tile_location_iter->second;
                  if (dst_tile_x == pc.col_idx && dst_tile_y == pc.row_idx) {
                    const int earliestRegTs = rsteps.front().ts;
                    const int rid = rsteps.back().regId; // ids are identical in your IR
                    // For data_mov-origin edges, deposit as DATA_MOV.
                    placeDstDeposit(topo, dstTid, earliestRegTs, consDir, rid, /*asCtrlMov=*/false);
                    if (pc.time_step > earliestRegTs) {
                      setConsumerSrcExact(cons, vAtCons, "$" + std::to_string(rid));
                      rewiredToReg = true;
                    }
                  }
                }
              }
              if (!rewiredToReg) setConsumerSrcExact(cons, vAtCons, consDir);
            }
          }
        }
      });

      // Map reserve -> phi for ctrl_mov loop-carried (IR invariant)
      DenseMap<Value, Operation*> reserveToPhi;
      func.walk([&](Operation *op){
        if (isPhi(op) && op->getNumOperands() >= 1)
          reserveToPhi[op->getOperand(0)] = op;
      });

      // 2.1) ctrl_mov endpoints: deposit to its declared register (CTRL_MOV), then timing-aware rewiring.
      func.walk([&](Operation *op){
        if (!isCtrlMov(op) || op->getNumOperands() < 2) return;
        Value src = op->getOperand(0);
        Value reserve = op->getOperand(1);
        Operation *phi = reserveToPhi.lookup(reserve);
        if (!phi) {
          op->emitWarning("ctrl_mov dest is not consumed by a PHI operand#0; skipping ctrl_mov-specific wiring.");
          return;
        }
        Operation *producer = src.getDefiningOp();
        if (!producer) return;

        auto pp = opPlace.lookup(chaseToTiledProducer(producer, opPlace));
        auto pc = opPlace.lookup(phi);
        if (!pp.hasTile || !pc.hasTile) return;

        auto chain  = collectLinkSteps(op);
        auto rsteps = collectRegSteps(op);

        // Same-tile ctrl_mov: no deposit; just honor reg if present, else die.
        if (pp.col_idx == pc.col_idx && pp.row_idx == pc.row_idx) {
          std::string rn;
          if (!rsteps.empty()) rn = "$" + std::to_string(rsteps.back().regId);
          else if (auto mr = getMappedRegId(op)) rn = "$" + std::to_string(*mr);
          else {
            op->emitError("same-tile ctrl_mov without assigned register id (mapping pass must assign one).");
            assert(false && "register id must be assigned by mapping pass for same-tile ctrl_mov");
          }
          setConsumerSrcExact(phi, src, rn);
          if (Instruction *pi = getInstPtr(producer)) addUniqueDst(pi, rn);
          prodSum[producer].hasSameTileConsumer = true;
          return;
        }

        // Cross-tile ctrl_mov
        std::string prodDir = "LOCAL";
        std::string consDir = "LOCAL";
        if (!chain.empty()) {
          prodDir = topo.dirFromLink(chain.front().link_id).str();
          consDir = topo.invertDir(topo.dirFromLink(chain.back().link_id)).str();
        } else {
          int dc = pc.col_idx - pp.col_idx, dr = pc.row_idx - pp.row_idx;
          if (dc > 0 && dr == 0) { prodDir = "EAST"; consDir = "WEST"; }
          else if (dc < 0 && dr == 0) { prodDir = "WEST"; consDir = "EAST"; }
          else if (dc == 0 && dr > 0) { prodDir = "NORTH"; consDir = "SOUTH"; }
          else if (dc == 0 && dr < 0) { prodDir = "SOUTH"; consDir = "NORTH"; }
        }
        if (Instruction *pi = getInstPtr(producer)) addUniqueDst(pi, prodDir);
        prodSum[producer].dirs.insert(prodDir);

        bool rewiredToReg = false;
        if (!chain.empty() && !rsteps.empty()) {
          int lastLink  = chain.back().link_id;
          int dstTid    = topo.dstTileOfLink(lastLink);
          auto tile_location_iter = topo.tile_location.find(dstTid);
          if (tile_location_iter != topo.tile_location.end()) {
            auto [dst_tile_x, dst_tile_y] = tile_location_iter->second;
            if (dst_tile_x == pc.col_idx && dst_tile_y == pc.row_idx) {
              const int earliestRegTs = rsteps.front().ts;
              const int rid = rsteps.back().regId;
              // For ctrl_mov, encode the endpoint deposit as CTRL_MOV (not DATA_MOV).
              placeDstDeposit(topo, dstTid, earliestRegTs, consDir, rid, /*asCtrlMov=*/true);
              if (pc.time_step > earliestRegTs) {
                setConsumerSrcExact(phi, src, "$" + std::to_string(rid));
                rewiredToReg = true;
              }
            }
          }
        }
        if (!rewiredToReg) setConsumerSrcExact(phi, src, consDir);
      });

      // 3) Producer post-process (applies to ALL producers)
      for (auto &kv : prodSum) {
        Operation *op = kv.first;
        Instruction *pi = getInstPtr(op);
        if (!pi) continue;
        for (const auto &d : kv.second.dirs) addUniqueDst(pi, d);

        const bool hasMappedReg = getMappedRegId(op).has_value();
        if (!kv.second.hasSameTileConsumer && !hasMappedReg) {
          pi->dst_operands.erase(
            std::remove_if(pi->dst_operands.begin(), pi->dst_operands.end(),
              [](const Operand &o){ return !o.operand.empty() && o.operand[0] == '$'; }),
            pi->dst_operands.end()
          );
        }
      }

      // 4) MATERIALIZE routers ONLY for multi-link ops (no single-hop)
      func.walk([&](Operation *op){
        if (isDataMov(op) || isCtrlMov(op)) emitIntermediateRoutersForOp(op, topo);
      });

      // 5) Keep UNRESOLVED and log them (color=YELLOW)
      unsigned unresolvedSrc = 0, unresolvedDst = 0;
      for (auto &tk : tile_time_insts) {
        auto key = tk.first;
        int c = key.first, r = key.second;
        for (auto &tt : tk.second) {
          int t = tt.first;
          auto &vec = tt.second;
          for (size_t i = 0; i < vec.size(); ++i) {
            auto &ins = vec[i];
            for (size_t k = 0; k < ins.src_operands.size(); ++k) {
              auto &s = ins.src_operands[k];
              if (s.operand == "UNRESOLVED") {
                s.color = "YELLOW";
                ++unresolvedSrc;
                llvm::errs() << "[UNRESOLVED SRC] tile("<<c<<","<<r<<") t="<<t
                             << " inst#" << i << " op=" << ins.opcode
                             << " src_idx=" << k << "\n";
              }
            }
            ins.dst_operands.erase(
              std::remove_if(ins.dst_operands.begin(), ins.dst_operands.end(),
                [](const Operand &o){
                  return o.operand.empty() || o.operand=="UNKNOWN";
                }),
              ins.dst_operands.end());
            for (size_t k = 0; k < ins.dst_operands.size(); ++k) {
              auto &d = ins.dst_operands[k];
              if (d.operand == "UNRESOLVED") {
                d.color = "YELLOW";
                ++unresolvedDst;
                llvm::errs() << "[UNRESOLVED DST] tile("<<c<<","<<r<<") t="<<t
                             << " inst#" << i << " op=" << ins.opcode
                             << " dst_idx=" << k << "\n";
              }
            }
          }
        }
      }
      if (unresolvedSrc + unresolvedDst) {
        auto diag = module.emitWarning("GenerateCodePass: UNRESOLVED operands kept for debugging");
        diag << " (src=" << unresolvedSrc << ", dst=" << unresolvedDst
             << "); they are highlighted with color=YELLOW in YAML.";
      }

      // 6) Assemble YAML (time-ordered per tile)
      ArrayConfig config;
      config.columns = columns;
      config.rows = rows;

      // flatten & sort by timestep
      std::map<std::pair<int,int>, std::vector<Instruction>> tileInstrs;
      for (auto &tileKV : tile_time_insts) {
        auto key = tileKV.first;
        auto &byT  = tileKV.second;
        auto &flat = tileInstrs[key];
        for (auto &kv : byT) for (auto &ins : kv.second) flat.push_back(ins);
        std::stable_sort(flat.begin(), flat.end(),
          [](const Instruction &a, const Instruction &b){ return a.time_step < b.time_step; });
      }

      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < columns; ++c) {
          auto it = tileInstrs.find({c,r});
          if (it == tileInstrs.end()) continue;
          Tile tile(c, r, r*columns + c);
          for (auto &ins : it->second) tile.entry.instructions.push_back(ins);
          config.cores.push_back(std::move(tile));
        }
      }

      std::error_code ec;
      llvm::raw_fd_ostream yaml_out("generated-instructions.yaml", ec);
      if (ec) { module.emitError("open yaml failed: " + ec.message()); return; }

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

      // Optional ASM output
      llvm::raw_fd_ostream asm_out("generated-instructions.asm", ec);
      if (ec) { module.emitError("open asm failed: " + ec.message()); return; }
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
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createGenerateCodePass() {
  return std::make_unique<GenerateCodePass>();
}
} // namespace mlir::neura
