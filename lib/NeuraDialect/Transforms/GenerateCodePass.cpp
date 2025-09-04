#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include <unordered_set>

#include "NeuraDialect/Architecture/Architecture.h"

using namespace mlir;

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

struct Entry {
  std::string entry_id;
  std::string type;
  std::vector<Instruction> instructions;
  Entry(const std::string &id, const std::string &t = "loop")
      : entry_id(id), type(t) {}
};

struct Core {
  int col, row;
  int core_id;
  Entry entry; // single entry per tile
  Core(int c, int r, int id) : col(c), row(r), core_id(id), entry("entry0", "loop") {}
};

struct ArrayConfig {
  int columns;
  int rows;
  std::vector<Core> cores;
};

struct OpLocation {
  int col = -1, row = -1, time_step = -1;
  bool hasTile = false;
};

static bool isDataMov(Operation *op) { return op->getName().getStringRef() == "neura.data_mov"; }
static bool isCtrlMov(Operation *op) { return op->getName().getStringRef() == "neura.ctrl_mov"; }
static bool isPhi(Operation *op)     { return op->getName().getStringRef() == "neura.phi"; }
static bool isReserve(Operation *op) { return op->getName().getStringRef() == "neura.reserve"; }
static bool isConstant(Operation *op){ return op->getName().getStringRef() == "neura.constant"; }
// Only neura.data_mov is treated as "forwarder" (pure routing step) for value chasing.
static bool isForwarder(Operation *op){ return isDataMov(op); }

// ----- placement helpers -----
static OpLocation getTilePlace(Operation *op) {
  OpLocation p;
  if (auto arr = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
    for (Attribute a : arr) {
      auto d = dyn_cast<DictionaryAttr>(a);
      if (!d) continue;
      auto res = dyn_cast_or_null<StringAttr>(d.get("resource"));
      if (!res || res.getValue() != "tile") continue;
      if (auto x = dyn_cast_or_null<IntegerAttr>(d.get("x"))) p.col = x.getInt();
      if (auto y = dyn_cast_or_null<IntegerAttr>(d.get("y"))) p.row = y.getInt();
      if (auto ts = dyn_cast_or_null<IntegerAttr>(d.get("time_step"))) p.time_step = ts.getInt();
      p.hasTile = true;
      break;
    }
  }
  return p;
}

static std::optional<int> getMappedReg(Operation *op) {
  if (auto arr = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
    for (Attribute a : arr) {
      auto d = dyn_cast<DictionaryAttr>(a);
      if (!d) continue;
      auto res = dyn_cast_or_null<StringAttr>(d.get("resource"));
      if (!res) continue;
      if (res.getValue() == "register" || res.getValue() == "reg") {
        if (auto id = dyn_cast_or_null<IntegerAttr>(d.get("id")))
          return id.getInt();
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
static std::string constantLiteral(Operation *op) {
  if (!isConstant(op)) return "";
  if (auto valAttr = op->getAttr("value")) {
    if (auto ia = dyn_cast<IntegerAttr>(valAttr))
      return "#" + std::to_string(ia.getInt());
    if (auto fa = dyn_cast<FloatAttr>(valAttr))
      return "#" + std::to_string(fa.getValueAsDouble());
  }
  return "#0";
}

// Chase through forwarders backwards to root producer value.
static Value chaseBackToRootValue(Value v) {
  Operation *d = v.getDefiningOp();
  while (d && isForwarder(d)) {
    if (d->getNumOperands() == 0) break;
    v = d->getOperand(0);
    d = v.getDefiningOp();
  }
  return v;
}

// nearest tiled ancestor on the input chain if producer is not tiled.
static Operation *chaseToTiledProducer(Operation *op,
                                       const DenseMap<Operation*, OpLocation> &opPlace) {
  SmallPtrSet<Operation*, 32> vis;
  Operation *cur = op;
  while (cur && !opPlace.lookup(cur).hasTile && !vis.count(cur)) {
    vis.insert(cur);
    if (cur->getNumOperands() == 0) break;
    cur = cur->getOperand(0).getDefiningOp();
  }
  return (cur && opPlace.lookup(cur).hasTile) ? cur : op;
}

// Use chain scanning to find a preferred register id for same-tile case.
static std::optional<int> regOnBackwardPath(Value vAtConsumer) {
  Operation *d = vAtConsumer.getDefiningOp();
  while (d) {
    if (auto r = getMappedReg(d)) return r;
    if (!isForwarder(d)) break;
    if (auto arr = d->getAttrOfType<ArrayAttr>("mapping_locs")) {
      for (Attribute a : arr) {
        auto dict = dyn_cast<DictionaryAttr>(a);
        if (!dict) continue;
        auto res = dyn_cast_or_null<StringAttr>(dict.get("resource"));
        if (res && (res.getValue() == "register" || res.getValue() == "reg")) {
          if (auto id = dyn_cast_or_null<IntegerAttr>(dict.get("id")))
            return id.getInt();
        }
      }
    }
    if (d->getNumOperands() == 0) break;
    vAtConsumer = d->getOperand(0);
    d = vAtConsumer.getDefiningOp();
  }
  return std::nullopt;
}

// ----- Topology built from Architecture -----

struct Topology {
  DenseMap<int, std::pair<int,int>> linkEnds;   // linkId -> (srcTileId, dstTileId)
  DenseMap<int, std::pair<int,int>> tileXY;     // tileId -> (x,y)

  StringRef dirBetween(int srcTid, int dstTid) const {
    auto [x0,y0] = tileXY.lookup(srcTid);
    auto [x1,y1] = tileXY.lookup(dstTid);
    int dc = x1 - x0, dr = y1 - y0;
    if (dc == 1 && dr == 0) return "EAST";
    if (dc == -1 && dr == 0) return "WEST";
    if (dc == 0 && dr == 1) return "NORTH";
    if (dc == 0 && dr == -1) return "SOUTH";
    return "LOCAL";
  }
  StringRef dirFromLink(int linkId) const {
    auto it = linkEnds.find(linkId);
    if (it == linkEnds.end()) return "LOCAL";
    return dirBetween(it->second.first, it->second.second);
  }
  StringRef invertDir(StringRef d) const {
    if (d == "EAST") return "WEST";
    if (d == "WEST") return "EAST";
    if (d == "NORTH") return "SOUTH";
    if (d == "SOUTH") return "NORTH";
    return "LOCAL";
  }
  int srcTileOfLink(int linkId) const { return linkEnds.lookup(linkId).first; }
  int dstTileOfLink(int linkId) const { return linkEnds.lookup(linkId).second; }
};

static Topology buildTopologyFromArchitecture(int columns, int rows) {
  Topology topo;
  mlir::neura::Architecture arch(columns, rows);

  for (auto *tile : arch.getAllTiles()) {
    topo.tileXY[tile->getId()] = {tile->getX(), tile->getY()};
  }
  for (auto *link : arch.getAllLinks()) {
    auto *s = link->getSrcTile();
    auto *d = link->getDstTile();
    topo.linkEnds[link->getId()] = {s->getId(), d->getId()};
  }
  return topo;
}

// ----- Extract mapping steps (sorted by time) from mapping_locs -----

struct LinkStep { int linkId; int ts; };
struct RegStep  { int regId;  int ts; };

static SmallVector<LinkStep, 8> collectLinkSteps(Operation *op) {
  SmallVector<LinkStep, 8> steps;
  if (auto arr = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
    for (Attribute a : arr) {
      auto d = dyn_cast<DictionaryAttr>(a);
      if (!d) continue;
      auto res = dyn_cast_or_null<StringAttr>(d.get("resource"));
      if (!res || res.getValue() != "link") continue;
      auto id = dyn_cast_or_null<IntegerAttr>(d.get("id"));
      auto ts = dyn_cast_or_null<IntegerAttr>(d.get("time_step"));
      if (!id || !ts) continue;
      steps.push_back({(int)id.getInt(), (int)ts.getInt()});
    }
  }
  llvm::sort(steps, [](const LinkStep &a, const LinkStep &b){ return a.ts < b.ts; });
  return steps;
}

static SmallVector<RegStep, 4> collectRegSteps(Operation *op) {
  SmallVector<RegStep, 4> steps;
  if (auto arr = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
    for (Attribute a : arr) {
      auto d = dyn_cast<DictionaryAttr>(a);
      if (!d) continue;
      auto res = dyn_cast_or_null<StringAttr>(d.get("resource"));
      if (!res) continue;
      if (res.getValue() == "register" || res.getValue() == "reg") {
        auto id = dyn_cast_or_null<IntegerAttr>(d.get("id"));
        auto ts = dyn_cast_or_null<IntegerAttr>(d.get("time_step"));
        if (!id || !ts) continue;
        steps.push_back({(int)id.getInt(), (int)ts.getInt()});
      }
    }
  }
  llvm::sort(steps, [](const RegStep &a, const RegStep &b){ return a.ts < b.ts; });
  return steps;
}

// ----- Pass -----

struct InstRef { int col, row, t, idx; };

struct ProdSummary {
  std::set<std::string> dirs;   // directions to remote consumers (keep for dst operands)
  bool hasSameTileConsumer = false;
};

struct GenerateCodePass
    : public PassWrapper<GenerateCodePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateCodePass)

  StringRef getArgument() const override { return "generate-code"; }
  StringRef getDescription() const override {
    return "CGRA YAML gen (multi-hop routers + endpoint register deposit + timing-aware rewiring, with CTRL_MOV kept).";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
  }

  // per-tile round-robin $1..$32
  DenseMap<std::pair<int,int>, int> tileNextReg;
  // root SSA value -> chosen $N (sticky across same-tile uses)
  DenseMap<Value, std::string>       localRegOf;
  DenseMap<Operation*, OpLocation>   opPlace;

  std::string allocTileReg(std::pair<int,int> tileKey) {
    int &n = tileNextReg[tileKey];
    if (n <= 0) n = 1;
    std::string name = "$" + std::to_string(n);
    n = (n % 32) + 1;
    return name;
  }

  // choose register for same-tile edges honoring provided ids.
  std::string pickRegisterForEdge(Operation *producer, Value vAtConsumer,
                                  std::optional<int> preferredId = std::nullopt) {
    Value root = chaseBackToRootValue(vAtConsumer);
    auto it = localRegOf.find(root);
    if (it != localRegOf.end()) return it->second;

    std::string rn;
    if (preferredId) {
      rn = "$" + std::to_string(*preferredId);
    } else if (auto id = regOnBackwardPath(vAtConsumer)) {
      rn = "$" + std::to_string(*id);
    } else if (auto mr = getMappedReg(producer)) {
      rn = "$" + std::to_string(*mr);
    } else {
      Operation *tp = chaseToTiledProducer(producer, opPlace);
      auto pp = opPlace.lookup(tp);
      rn = allocTileReg({pp.col, pp.row});
    }
    localRegOf[root] = rn;
    return rn;
  }

  // map of tile(x,y) -> time_step -> vector<Instruction>
  std::map<std::pair<int,int>, std::map<int, std::vector<Instruction>>> pe_time_insts;

  // to get Instruction* back when wiring operands
  DenseMap<Operation*, InstRef>            op2ref;
  DenseMap<Operation*, SmallVector<Value>> op2Operands;

  // de-dup sets
  std::unordered_set<uint64_t> hopSig;     // (midTileId, ts, linkId)
  std::unordered_set<uint64_t> depositSig; // (dstTileId, ts, regId)

  // ---------- helpers to place materialized instructions ----------
  void placeRouterHop(const Topology &topo, int tileId, int ts,
                      StringRef inDir, StringRef outDir) {
    auto [x,y] = topo.tileXY.lookup(tileId);
    Instruction inst("DATA_MOV");
    inst.time_step = ts;
    inst.src_operands.emplace_back(inDir.str(), "RED");
    inst.dst_operands.emplace_back(outDir.str(), "RED");
    pe_time_insts[{x,y}][ts].push_back(std::move(inst));
  }

  // endpoint deposit: on destination tile at earliest reg ts, move [incoming_dir] -> [$reg]
  // If `asCtrlMov=true`, encode the deposit as "CTRL_MOV" instead of "DATA_MOV".
  void placeDstDeposit(const Topology &topo, int dstTileId, int ts,
                       StringRef incomingDir, int regId, bool asCtrlMov = false) {
    uint64_t sig = (uint64_t)dstTileId << 32 ^ (uint64_t)ts << 16 ^ (uint64_t)regId;
    if (!depositSig.insert(sig).second) return; // already placed
    auto [x,y] = topo.tileXY.lookup(dstTileId);
    Instruction inst(asCtrlMov ? "CTRL_MOV" : "DATA_MOV");
    inst.time_step = ts;
    inst.src_operands.emplace_back(incomingDir.str(), "RED");
    inst.dst_operands.emplace_back("$" + std::to_string(regId), "RED");
    pe_time_insts[{x,y}][ts].push_back(std::move(inst));
  }

  void emitIntermediateRoutersForOp(Operation *op, const Topology &topo) {
    auto steps = collectLinkSteps(op);
    if (steps.size() < 2) return; // only multi-hop
    for (size_t i = 1; i < steps.size(); ++i) {
      int prevL = steps[i - 1].linkId;
      int curL  = steps[i].linkId;
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
    return pe_time_insts[{c,r}][t];
  }
  Instruction* getInstPtr(Operation *op) {
    auto it = op2ref.find(op);
    if (it == op2ref.end()) return nullptr;
    auto [c,r,t,idx] = it->second;
    auto &vec = pe_time_insts[{c,r}][t];
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
      tileNextReg.clear();
      localRegOf.clear();
      pe_time_insts.clear();
      op2ref.clear();
      op2Operands.clear();
      hopSig.clear();
      depositSig.clear();

      // cache tile placements
      func.walk([&](Operation *op){ opPlace[op] = getTilePlace(op); });

      // 1) Materialize compute/logic ops (CONSTANT/ADD/FADD/PHI/ICMP/NOT/GRANT_* etc.)
      //    Drop reserve/ctrl_mov/data_mov here; they are for wiring & router hops only.
      func.walk([&](Operation *op){
        auto p = opPlace[op];
        if (!p.hasTile) return;
        if (isReserve(op) || isCtrlMov(op) || isDataMov(op)) return;

        std::string opc = mapOpcode(op);
        Instruction inst(opc);
        inst.time_step = p.time_step;

        if (isConstant(op)) {
          inst.src_operands.emplace_back(constantLiteral(op), "RED");
        } else {
          SmallVector<Value> operands; operands.reserve(op->getNumOperands());
          for (Value in : op->getOperands()) {
            operands.push_back(in);
            inst.src_operands.emplace_back("UNRESOLVED", "RED");
          }
          op2Operands[op] = std::move(operands);
        }

        // If op itself mapped a register (rare), keep as dst reg.
        if (auto mr = getMappedReg(op))
          inst.dst_operands.emplace_back("$" + std::to_string(*mr), "RED");

        auto &vec = bucketOf(p.col, p.row, p.time_step);
        vec.push_back(std::move(inst));
        op2ref[op] = InstRef{p.col, p.row, p.time_step, (int)vec.size() - 1};
      });

      // Summaries for producers
      DenseMap<Operation*, ProdSummary> prodSum;

      auto stickRootReg = [&](Value vAtConsumer, const std::string &rn){
        Value root = chaseBackToRootValue(vAtConsumer);
        localRegOf[root] = rn;
      };

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

            if (pp.col == pc.col && pp.row == pc.row) {
              // same-tile: use $reg
              std::string rn = pickRegisterForEdge(prod, vAtCons, std::nullopt);
              setConsumerSrcExact(cons, vAtCons, rn);
              if (Instruction *pi = getInstPtr(prod)) addUniqueDst(pi, rn);
              stickRootReg(vAtCons, rn);
              prodSum[prod].hasSameTileConsumer = true;
            } else {
              // cross-tile: compute endpoint directions
              std::string prodDir = "LOCAL";
              std::string consDir = "LOCAL";

              if (!chain.empty()) {
                int firstLink = chain.front().linkId;
                int lastLink  = chain.back().linkId;
                prodDir = topo.dirFromLink(firstLink).str();
                consDir = topo.invertDir(topo.dirFromLink(lastLink)).str();
              } else {
                int dc = pc.col - pp.col, dr = pc.row - pp.row;
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
                int lastLink  = chain.back().linkId;
                int dstTid    = topo.dstTileOfLink(lastLink);
                auto itXY     = topo.tileXY.find(dstTid);
                if (itXY != topo.tileXY.end()) {
                  auto [dx,dy] = itXY->second;
                  if (dx == pc.col && dy == pc.row) {
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

        // Same-tile ctrl_mov: no deposit; just honor reg if present.
        if (pp.col == pc.col && pp.row == pc.row) {
          std::string rn;
          if (!rsteps.empty()) rn = "$" + std::to_string(rsteps.back().regId);
          else if (auto mr = getMappedReg(op)) rn = "$" + std::to_string(*mr);
          else rn = pickRegisterForEdge(producer, src, std::nullopt);
          setConsumerSrcExact(phi, src, rn);
          if (Instruction *pi = getInstPtr(producer)) addUniqueDst(pi, rn);
          prodSum[producer].hasSameTileConsumer = true;
          return;
        }

        // Cross-tile ctrl_mov
        std::string prodDir = "LOCAL";
        std::string consDir = "LOCAL";
        if (!chain.empty()) {
          prodDir = topo.dirFromLink(chain.front().linkId).str();
          consDir = topo.invertDir(topo.dirFromLink(chain.back().linkId)).str();
        } else {
          int dc = pc.col - pp.col, dr = pc.row - pp.row;
          if (dc > 0 && dr == 0) { prodDir = "EAST"; consDir = "WEST"; }
          else if (dc < 0 && dr == 0) { prodDir = "WEST"; consDir = "EAST"; }
          else if (dc == 0 && dr > 0) { prodDir = "NORTH"; consDir = "SOUTH"; }
          else if (dc == 0 && dr < 0) { prodDir = "SOUTH"; consDir = "NORTH"; }
        }
        if (Instruction *pi = getInstPtr(producer)) addUniqueDst(pi, prodDir);
        prodSum[producer].dirs.insert(prodDir);

        bool rewiredToReg = false;
        if (!chain.empty() && !rsteps.empty()) {
          int lastLink  = chain.back().linkId;
          int dstTid    = topo.dstTileOfLink(lastLink);
          auto itXY     = topo.tileXY.find(dstTid);
          if (itXY != topo.tileXY.end()) {
            auto [dx,dy] = itXY->second;
            if (dx == pc.col && dy == pc.row) {
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
      //    - Keep producer endpoint dirs recorded earlier.
      //    - Remove any $ on producer dst IF it has NO same-tile consumers AND
      //      the op itself does NOT have a mapped register.
      for (auto &kv : prodSum) {
        Operation *op = kv.first;
        Instruction *pi = getInstPtr(op);
        if (!pi) continue;
        for (const auto &d : kv.second.dirs) addUniqueDst(pi, d);

        const bool hasMappedReg = getMappedReg(op).has_value();
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
      for (auto &tk : pe_time_insts) {
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
      for (auto &tileKV : pe_time_insts) {
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
          Core core(c, r, r*columns + c);
          for (auto &ins : it->second) core.entry.instructions.push_back(ins);
          config.cores.push_back(std::move(core));
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
        yaml_out << "    - column: " << core.col << "\n";
        yaml_out << "      row: " << core.row << "\n";
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

      // Optional ASM output (human friendly)
      llvm::raw_fd_ostream asm_out("generated-instructions.asm", ec);
      if (ec) { module.emitError("open asm failed: " + ec.message()); return; }
      asm_out << "; ASM (multi-hop routers + endpoint deposit (DATA/CTRL_MOV) + timing-aware src wiring)\n\n";
      for (const auto &core : config.cores) {
        asm_out << "PE(" << core.col << "," << core.row << "):\n";
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
