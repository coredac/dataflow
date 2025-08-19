#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include <map>
#include <vector>
#include <string>
#include <set>
#include <tuple>
#include <optional>
#include <algorithm>
#include <queue>
#include <cmath>

using namespace mlir;

namespace {

struct Operand {
  std::string operand;
  std::string color;
  Operand(const std::string &op, const std::string &c = "RED")
      : operand(op), color(c) {}
};

struct RouteHint {
  int toX = -1, toY = -1;
  std::string firstHop; // EAST/WEST/NORTH/... (first hop direction)
  int hops = 0;         // Chebyshev distance
};

struct Instruction {
  std::string opcode;
  std::vector<Operand> src_operands;
  std::vector<Operand> dst_operands;
  int                  time_step = -1; // for ordering
  std::vector<RouteHint> route_hints;  // only on multi-hop producer insts
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
  Core(int c, int r, int id) : col(c), row(r), core_id(id), entry("entry0","loop") {}
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

// Sets whether op is neura.data_mov.
static bool isDataMov(Operation *op) { return op->getName().getStringRef() == "neura.data_mov"; }
// Sets whether op is neura.ctrl_mov.
static bool isCtrlMov(Operation *op) { return op->getName().getStringRef() == "neura.ctrl_mov"; }
// Sets whether op is neura.phi.
static bool isPhi(Operation *op)     { return op->getName().getStringRef() == "neura.phi"; }
// Sets whether op is neura.reserve.
static bool isReserve(Operation *op) { return op->getName().getStringRef() == "neura.reserve"; }
// Sets whether op is neura.constant.
static bool isConstant(Operation *op){ return op->getName().getStringRef() == "neura.constant"; }
// Sets whether op is a forwarder (only neura.data_mov now).
static bool isForwarder(Operation *op){ return isDataMov(op); }

static OpLocation getTilePlace(Operation *op) {
  OpLocation p;
  if (ArrayAttr arr = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
    for (Attribute a : arr) {
      DictionaryAttr d = dyn_cast<DictionaryAttr>(a);
      if (!d) continue;
      StringAttr res = dyn_cast_or_null<StringAttr>(d.get("resource"));
      if (!res || res.getValue() != "tile") continue;
      if (IntegerAttr x = dyn_cast_or_null<IntegerAttr>(d.get("x"))) p.col = x.getInt();
      if (IntegerAttr y = dyn_cast_or_null<IntegerAttr>(d.get("y"))) p.row = y.getInt();
      if (IntegerAttr ts= dyn_cast_or_null<IntegerAttr>(d.get("time_step"))) p.time_step = ts.getInt();
      p.hasTile = true;
      break;
    }
  }
  return p;
}

static std::optional<int> getMappedReg(Operation *op) {
  if (ArrayAttr arr = op->getAttrOfType<ArrayAttr>("mapping_locs")) {
    for (Attribute a : arr) {
      DictionaryAttr d = dyn_cast<DictionaryAttr>(a);
      if (!d) continue;
      StringAttr res = dyn_cast_or_null<StringAttr>(d.get("resource"));
      if (!res) continue;
      if (res.getValue() == "register" || res.getValue() == "reg") {
        if (IntegerAttr id = dyn_cast_or_null<IntegerAttr>(d.get("id")))
          return id.getInt();
      }
    }
  }
  return std::nullopt;
}

static std::string mapOpcode(Operation *op) {
  std::string opcode = op->getName().getStringRef().str();
  if (opcode.rfind("neura.", 0) == 0) opcode = opcode.substr(6);
  if (isConstant(op)) return "CONSTANT"; // keep as CONSTANT
  std::transform(opcode.begin(), opcode.end(), opcode.begin(), ::toupper);
  return opcode;
}

// Sets the first-hop direction from (c0,r0) to (c1,r1).
static std::string dirFrom(int c0, int r0, int c1, int r1) {
  int dc = c1 - c0;
  int dr = r1 - r0;
  // clamp to {-1, 0, 1}
  int sdc = (dc > 0) ? 1 : (dc < 0 ? -1 : 0);
  int sdr = (dr > 0) ? 1 : (dr < 0 ? -1 : 0);

  if (sdc == 0 && sdr == 0) return "LOCAL";
  if (sdc == 1 && sdr == 0) return "EAST";
  if (sdc == -1 && sdr == 0) return "WEST";
  if (sdc == 0 && sdr == 1) return "NORTH";
  if (sdc == 0 && sdr == -1) return "SOUTH";
  if (sdc == 1 && sdr == 1) return "NORTHEAST";
  if (sdc == -1 && sdr == 1) return "NORTHWEST";
  if (sdc == 1 && sdr == -1) return "SOUTHEAST";
  if (sdc == -1 && sdr == -1) return "SOUTHWEST";
  return "LOCAL";
}

// Sets the opposite direction of d.
static std::string oppDir(StringRef d) {
  if (d == "EAST") return "WEST";
  if (d == "WEST") return "EAST";
  if (d == "NORTH") return "SOUTH";
  if (d == "SOUTH") return "NORTH";
  if (d == "NORTHEAST") return "SOUTHWEST";
  if (d == "NORTHWEST") return "SOUTHEAST";
  if (d == "SOUTHEAST") return "NORTHWEST";
  if (d == "SOUTHWEST") return "NORTHEAST";
  return "LOCAL";
}



static int hopCount(int c0, int r0, int c1, int r1) {
  return std::max(std::abs(c1 - c0), std::abs(r1 - r0));
}

// Sets the literal for CONSTANT's source: "#10"/"#0"/"#3.0".
static std::string constantLiteral(Operation *op) {
  if (!isConstant(op)) return "";
  if (Attribute valAttr = op->getAttr("value")) {
    if (IntegerAttr ia = dyn_cast<IntegerAttr>(valAttr))
      return "#" + std::to_string(ia.getInt());
    if (FloatAttr fa = dyn_cast<FloatAttr>(valAttr))
      return "#" + std::to_string(fa.getValueAsDouble());
  }
  return "#0";
}

// Sets the root value by walking back through forwarders (data_mov).
static Value chaseBackToRootValue(Value v) {
  Operation *d = v.getDefiningOp();
  while (d && isForwarder(d)) {
    if (d->getNumOperands() == 0) break;
    v = d->getOperand(0);
    d = v.getDefiningOp();
  }
  return v;
}

// Sets the preferred register id along the backward path from vAtConsumer.
static std::optional<int> regOnBackwardPath(Value vAtConsumer) {
  Operation *d = vAtConsumer.getDefiningOp();
  while (d) {
    if (std::optional<int> r = getMappedReg(d)) return r; // op itself
    if (!isForwarder(d)) break;

    if (ArrayAttr arr = d->getAttrOfType<ArrayAttr>("mapping_locs")) {
      for (Attribute a : arr) {
        DictionaryAttr dict = dyn_cast<DictionaryAttr>(a);
        if (!dict) continue;
        StringAttr res = dyn_cast_or_null<StringAttr>(dict.get("resource"));
        if (res && (res.getValue() == "register" || res.getValue() == "reg")) {
          if (IntegerAttr id = dyn_cast_or_null<IntegerAttr>(dict.get("id")))
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

// Sets the register ids that appear on the forwarder chain that produces vAtConsumer.
static SmallVector<int, 4> regsOnForwarderChainTo(Value vAtConsumer) {
  SmallVector<int, 4> ids;
  Operation *d = vAtConsumer.getDefiningOp();
  while (d && isForwarder(d)) {
    if (ArrayAttr arr = d->getAttrOfType<ArrayAttr>("mapping_locs")) {
      for (Attribute a : arr) {
        DictionaryAttr dict = dyn_cast<DictionaryAttr>(a);
        if (!dict) continue;
        StringAttr res = dyn_cast_or_null<StringAttr>(dict.get("resource"));
        if (res && (res.getValue() == "register" || res.getValue() == "reg")) {
          if (IntegerAttr id = dyn_cast_or_null<IntegerAttr>(dict.get("id"))) {
            int v = id.getInt();
            if (std::find(ids.begin(), ids.end(), v) == ids.end())
              ids.push_back(v);
          }
        }
      }
    }
    if (d->getNumOperands() == 0) break;
    Value prev = d->getOperand(0);
    d = prev.getDefiningOp();
  }
  return ids;
}

// Sets the nearest tiled ancestor by walking to the nearest tiled ancestor on its input chain if producer is not tiled.
static Operation* chaseToTiledProducer(Operation *op,
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

// Sets the final tiled consumers by skipping data_mov forwarders.
static SmallVector<std::pair<Operation*, Value>, 8>
findTiledConsumers(Value startV,
                   const DenseMap<Operation*, OpLocation> &opPlace) {
  SmallVector<std::pair<Operation*, Value>, 8> out;
  SmallPtrSet<Value, 32> seenV;
  std::queue<Value> q;
  q.push(startV);
  seenV.insert(startV);

  while (!q.empty()) {
    Value v = q.front(); q.pop();

    for (OpOperand &use : v.getUses()) {
      Operation *u = use.getOwner();
      if (opPlace.lookup(u).hasTile) {
        out.emplace_back(u, v);
        continue;
      }
      if (isForwarder(u)) {
        if (u->getNumResults() > 0) {
          Value nv = u->getResult(0);
          if (!seenV.count(nv)) { seenV.insert(nv); q.push(nv); }
        }
        continue;
      }
    }
  }
  return out;
}

struct InstRef { int col, row, t, idx; };

struct ProdSummary {
  std::set<std::string> dirs;   // directions to remote consumers (producer-side)
  bool hasSameTileConsumer = false;
};

struct GenerateCodePass
    : public PassWrapper<GenerateCodePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateCodePass)

  StringRef getArgument() const override { return "generate-code"; }
  StringRef getDescription() const override {
    return "CGRA YAML gen (opposite consumer ports; multi-hop route_hints on producers; honor provided register ids; keep UNRESOLVED).";
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

  // Sets a register for a same-tile edge by honoring provided ids.
  std::string pickRegisterForEdge(Operation *producer, Value vAtConsumer,
                                  std::optional<int> preferredId = std::nullopt) {
    Value root = chaseBackToRootValue(vAtConsumer);
    DenseMap<Value, std::string>::iterator it = localRegOf.find(root);
    if (it != localRegOf.end()) return it->second;

    std::string rn;
    if (preferredId) {
      rn = "$" + std::to_string(*preferredId);
    } else if (std::optional<int> id = regOnBackwardPath(vAtConsumer)) {
      rn = "$" + std::to_string(*id);
    } else if (std::optional<int> mr = getMappedReg(producer)) {
      rn = "$" + std::to_string(*mr);
    } else {
      Operation *tp = chaseToTiledProducer(producer, opPlace);
              OpLocation pp = opPlace.lookup(tp);
      rn = allocTileReg({pp.col, pp.row});
    }
    localRegOf[root] = rn;
    return rn;
  }

  // Sets a route hint on producer instruction if this edge is multi-hop.
  static void addRouteHintIfMultiHop(Instruction *producerInst,
                                     int fromC,int fromR,int toC,int toR,
                                     StringRef firstHopDir) {
    if (!producerInst) return;
    int hops = hopCount(fromC, fromR, toC, toR);
    if (hops <= 1) return;
    RouteHint rh;
    rh.toX = toC; rh.toY = toR;
    rh.firstHop = firstHopDir.str();
    rh.hops = hops;
    producerInst->route_hints.push_back(std::move(rh));
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      StringAttr accel = func->getAttrOfType<StringAttr>("accelerator");
      if (!accel || accel.getValue() != "neura") continue;

      int columns = 4, rows = 4;
      if (DictionaryAttr mi = func->getAttrOfType<DictionaryAttr>("mapping_info")) {
        if (IntegerAttr xv = dyn_cast_or_null<IntegerAttr>(mi.get("x_tiles"))) columns = xv.getInt();
        if (IntegerAttr yv = dyn_cast_or_null<IntegerAttr>(mi.get("y_tiles"))) rows   = yv.getInt();
      }

      opPlace.clear();
      tileNextReg.clear();
      localRegOf.clear();

      // collect placements
      func.walk([&](Operation *op){ opPlace[op] = getTilePlace(op); });

      // tile -> time -> instructions
      std::map<std::pair<int,int>, std::map<int, std::vector<Instruction>>> pe_time_insts;

      std::function<std::vector<Instruction>&(int, int, int)> bucketOf = [&](int c, int r, int t) -> std::vector<Instruction>& {
        return pe_time_insts[{c,r}][t];
      };

      DenseMap<Operation*, InstRef>            op2ref;
      DenseMap<Operation*, SmallVector<Value>> op2Operands;

      // 1) Materialize compute ops (incl. CONSTANT & GRANT_ONCE). Drop reserve/ctrl_mov/data_mov.
      func.walk([&](Operation *op){
        OpLocation p = opPlace[op];
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

        // If this op maps its result to a register, keep it (multiple dsts allowed later).
        if (std::optional<int> mr = getMappedReg(op))
          inst.dst_operands.emplace_back("$" + std::to_string(*mr), "RED");

        std::vector<Instruction> &vec = bucketOf(p.col, p.row, p.time_step);
        vec.push_back(std::move(inst));
        op2ref[op] = InstRef{p.col, p.row, p.time_step, (int)vec.size() - 1};
      });

      std::function<Instruction*(Operation*)> getInstPtr = [&](Operation *op) -> Instruction* {
        DenseMap<Operation*, InstRef>::iterator it = op2ref.find(op);
        if (it == op2ref.end()) return nullptr;
        auto [c,r,t,idx] = it->second; // 这是结构化绑定，保留auto
        std::vector<Instruction> &vec = pe_time_insts[{c,r}][t];
        if (idx < 0 || idx >= (int)vec.size()) return nullptr;
        return &vec[idx];
      };

      std::function<void(Operation*, Value, const std::string&)> setConsumerSrcExact = [&](Operation *consumer, Value vAtConsumer, const std::string &text) {
        Instruction *ci = getInstPtr(consumer);
        if (!ci) return;
        DenseMap<Operation*, SmallVector<Value>>::iterator itOps = op2Operands.find(consumer);
        if (itOps == op2Operands.end()) return;
        SmallVector<Value> &ops = itOps->second;
        for (size_t i = 0; i < ops.size() && i < ci->src_operands.size(); ++i) {
          if (ops[i] == vAtConsumer) { ci->src_operands[i].operand = text; return; }
        }
        for (Operand &s : ci->src_operands)
          if (s.operand == "UNRESOLVED") { s.operand = text; return; }
      };

      std::function<void(Instruction*, const std::string&)> addUniqueDst = [](Instruction *inst, const std::string &text){
        for (Operand &d : inst->dst_operands) if (d.operand == text) return;
        inst->dst_operands.emplace_back(text, "RED");
      };

      std::function<void(Value, const std::string&)> stickRootReg = [&](Value vAtConsumer, const std::string &rn){
        Value root = chaseBackToRootValue(vAtConsumer);
        localRegOf[root] = rn;
      };

      // 2) Edges + producer summaries
      DenseMap<Operation*, ProdSummary> prodSum;

      func.walk([&](Operation *prod){
        if (isReserve(prod) || isCtrlMov(prod)) return;
        for (Value res : prod->getResults()) {
          SmallVector<std::pair<Operation*, Value>, 8> consumers = findTiledConsumers(res, opPlace);
          for (std::pair<Operation*, Value> &cc : consumers) {
            Operation *cons = cc.first;
            Value vAtCons   = cc.second;

            Operation *tp = chaseToTiledProducer(prod, opPlace);
            OpLocation pp = opPlace.lookup(tp);
            OpLocation pc = opPlace.lookup(cons);
            if (!pp.hasTile || !pc.hasTile) continue;

            ProdSummary &sum = prodSum[prod];
            if (pp.col == pc.col && pp.row == pc.row) {
              // same tile: pick register with backward/producer hints
              std::string rn = pickRegisterForEdge(prod, vAtCons, std::nullopt);
              setConsumerSrcExact(cons, vAtCons, rn);
              if (Instruction *pi = getInstPtr(prod)) addUniqueDst(pi, rn);
              stickRootReg(vAtCons, rn);
              sum.hasSameTileConsumer = true;
            } else {
              // cross tile: consumer reads "opposite direction"
              std::string dir = dirFrom(pp.col, pp.row, pc.col, pc.row);
              setConsumerSrcExact(cons, vAtCons, oppDir(dir));      // <-- key change
              if (Instruction *pi = getInstPtr(prod)) {
                addUniqueDst(pi, dir);                               // producer sends to "dir"
                addRouteHintIfMultiHop(pi, pp.col, pp.row, pc.col, pc.row, dir);
              }
              sum.dirs.insert(dir);

              // Attach destination register holds on the consumer if chain shows them
              if (Instruction *ci = getInstPtr(cons)) {
                SmallVector<int, 4> ids = regsOnForwarderChainTo(vAtCons);
                for (int id : ids) addUniqueDst(ci, "$" + std::to_string(id));
              }
            }
          }
        }
      });

      // Map reserve -> phi for loop-carried
      DenseMap<Value, Operation*> reserveToPhi;
      func.walk([&](Operation *op){
        if (isPhi(op) && op->getNumOperands() >= 1)
          reserveToPhi[op->getOperand(0)] = op;
      });

      // 2.1) Handle ctrl_mov feeding reserve (loop-carried). Prefer ctrl_mov's register id.
      func.walk([&](Operation *op){
        if (!isCtrlMov(op) || op->getNumOperands() < 2) return;
        Value src = op->getOperand(0);
        Value reserve = op->getOperand(1);
        Operation *phi = reserveToPhi.lookup(reserve);
        if (!phi) return;
        Operation *producer = src.getDefiningOp();
        if (!producer) return;

        Operation *tp = chaseToTiledProducer(producer, opPlace);
        OpLocation pp = opPlace.lookup(tp);
        OpLocation pc = opPlace.lookup(phi);
        if (!pp.hasTile || !pc.hasTile) return;

        std::optional<int> ctrlReg = getMappedReg(op);

        ProdSummary &sum = prodSum[producer];
        if (pp.col == pc.col && pp.row == pc.row) {
          std::string rn = pickRegisterForEdge(producer, src, ctrlReg);
          setConsumerSrcExact(phi, src, rn);
          if (Instruction *pi = getInstPtr(producer)) addUniqueDst(pi, rn);
          stickRootReg(src, rn);
          sum.hasSameTileConsumer = true;
        } else {
          std::string dir = dirFrom(pp.col, pp.row, pc.col, pc.row);
          setConsumerSrcExact(phi, src, oppDir(dir));                 // consumer src uses opposite
          if (Instruction *pi = getInstPtr(producer)) {
            addUniqueDst(pi, dir);
            addRouteHintIfMultiHop(pi, pp.col, pp.row, pc.col, pc.row, dir);
          }
          sum.dirs.insert(dir);

          // If ctrl_mov shows a reg id on dest, attach to phi (consumer)
          if (Instruction *ci = getInstPtr(phi)) {
            if (ctrlReg) addUniqueDst(ci, "$" + std::to_string(*ctrlReg));
          }
        }
      });

      // 3) PHI post-process (keep producer fanout dirs; clean stray $ if no same-tile consumer and no mapped reg)
      for (std::pair<Operation*, ProdSummary> &kv : prodSum) {
        Operation *op = kv.first;
        if (!isPhi(op)) continue;
        Instruction *pi = getInstPtr(op);
        if (!pi) continue;

        for (const std::string &d : kv.second.dirs) addUniqueDst(pi, d);

        bool hasMappedReg = getMappedReg(op).has_value();
        if (!kv.second.hasSameTileConsumer && !hasMappedReg) {
          pi->dst_operands.erase(
            std::remove_if(pi->dst_operands.begin(), pi->dst_operands.end(),
              [](const Operand &o){ return !o.operand.empty() && o.operand[0] == '$'; }),
            pi->dst_operands.end()
          );
        }
      }

      // 4) Keep UNRESOLVED and log them (color=YELLOW)
      unsigned unresolvedSrc = 0, unresolvedDst = 0;
      for (std::pair<const std::pair<int,int>, std::map<int, std::vector<Instruction>>> &tk : pe_time_insts) {
        std::pair<int,int> key = tk.first;
        int c = key.first, r = key.second;
        for (std::pair<const int, std::vector<Instruction>> &tt : tk.second) {
          int t = tt.first;
          std::vector<Instruction> &vec = tt.second;

          for (size_t i = 0; i < vec.size(); ++i) {
            Instruction &ins = vec[i];

            for (size_t k = 0; k < ins.src_operands.size(); ++k) {
              Operand &s = ins.src_operands[k];
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
              Operand &d = ins.dst_operands[k];
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
        InFlightDiagnostic diag = module.emitWarning("GenerateCodePass: UNRESOLVED operands kept for debugging");
        diag << " (src=" << unresolvedSrc << ", dst=" << unresolvedDst
             << "); they are highlighted with color=YELLOW in YAML.";
      }

      // 5) Assemble YAML (single entry0 per tile, time-ordered)
      ArrayConfig config;
      config.columns = columns;
      config.rows = rows;

      // flatten & sort
      std::map<std::pair<int,int>, std::vector<Instruction>> tileInstrs;
      for (std::pair<const std::pair<int,int>, std::map<int, std::vector<Instruction>>> &tileKV : pe_time_insts) {
        std::pair<int,int> key = tileKV.first;
        std::map<int, std::vector<Instruction>> &byT  = tileKV.second;
        std::vector<Instruction> &flat = tileInstrs[key];
        for (std::pair<const int, std::vector<Instruction>> &kv : byT) for (Instruction &ins : kv.second) flat.push_back(ins);
        std::stable_sort(flat.begin(), flat.end(),
          [](const Instruction &a, const Instruction &b){ return a.time_step < b.time_step; });
      }

      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < columns; ++c) {
          std::map<std::pair<int,int>, std::vector<Instruction>>::iterator it = tileInstrs.find({c,r});
          if (it == tileInstrs.end()) continue;
          Core core(c, r, r*columns + c);
          for (Instruction &ins : it->second) core.entry.instructions.push_back(ins);
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
      for (const Core &core : config.cores) {
        yaml_out << "    - column: " << core.col << "\n";
        yaml_out << "      row: " << core.row << "\n";
        yaml_out << "      core_id: \"" << core.core_id << "\"\n";
        yaml_out << "      entries:\n";
        yaml_out << "        - entry_id: \"entry0\"\n";
        yaml_out << "          type: \"loop\"\n";
        yaml_out << "          instructions:\n";
        for (const Instruction &inst : core.entry.instructions) {
          yaml_out << "            - opcode: \"" << inst.opcode << "\"\n";
          yaml_out << "              timestep: " << inst.time_step << "\n";
          if (!inst.src_operands.empty()) {
            yaml_out << "              src_operands:\n";
            for (const Operand &opnd : inst.src_operands) {
              yaml_out << "                - operand: \"" << opnd.operand << "\"\n";
              yaml_out << "                  color: \"" << opnd.color << "\"\n";
            }
          }
          if (!inst.dst_operands.empty()) {
            yaml_out << "              dst_operands:\n";
            for (const Operand &opnd : inst.dst_operands) {
              yaml_out << "                - operand: \"" << opnd.operand << "\"\n";
              yaml_out << "                  color: \"" << opnd.color << "\"\n";
            }
          }
          // route_hints only when present (multi-hop producers)
          if (!inst.route_hints.empty()) {
            yaml_out << "              route_hints:\n";
            for (const RouteHint &rh : inst.route_hints) {
              yaml_out << "                - to: { x: " << rh.toX << ", y: " << rh.toY << " }\n";
              yaml_out << "                  first_hop: \"" << rh.firstHop << "\"\n";
              yaml_out << "                  hops: " << rh.hops << "\n";
            }
          }
        }
      }
      yaml_out.close();

      // Optional ASM
      llvm::raw_fd_ostream asm_out("generated-instructions.asm", ec);
      if (ec) { module.emitError("open asm failed: " + ec.message()); return; }
      asm_out << "; ASM (opposite consumer ports; route_hints only on multi-hop producers; dest $regs attached on consumers when present; UNRESOLVED kept)\n\n";
      for (const Core &core : config.cores) {
        asm_out << "PE(" << core.col << "," << core.row << "):\n{\n";
        asm_out << "  Entry => Loop {\n";
        for (const Instruction &inst : core.entry.instructions) {
          asm_out << "    {\n";
          asm_out << "      " << inst.opcode;
          
          // Print source operands first
          if (!inst.src_operands.empty()) {
            // Check if first operand needs color (direction-based)
            if (inst.src_operands[0].operand == "NORTH" || inst.src_operands[0].operand == "SOUTH" || 
                inst.src_operands[0].operand == "EAST" || inst.src_operands[0].operand == "WEST" ||
                inst.src_operands[0].operand == "NORTHEAST" || inst.src_operands[0].operand == "NORTHWEST" ||
                inst.src_operands[0].operand == "SOUTHEAST" || inst.src_operands[0].operand == "SOUTHWEST") {
              asm_out << ", [" << inst.src_operands[0].operand << ", R]";
            } else {
              asm_out << ", [" << inst.src_operands[0].operand << "]";
            }
            
            for (size_t i=1;i<inst.src_operands.size();++i) {
              if (inst.src_operands[i].operand == "NORTH" || inst.src_operands[i].operand == "SOUTH" || 
                  inst.src_operands[i].operand == "EAST" || inst.src_operands[i].operand == "WEST" ||
                  inst.src_operands[i].operand == "NORTHEAST" || inst.src_operands[i].operand == "NORTHWEST" ||
                  inst.src_operands[i].operand == "SOUTHEAST" || inst.src_operands[i].operand == "SOUTHWEST") {
                asm_out << ", [" << inst.src_operands[i].operand << ", R]";
              } else {
                asm_out << ", [" << inst.src_operands[i].operand << "]";
              }
            }
          }
          
          // Add arrow if there are destination operands
          if (!inst.dst_operands.empty()) {
            asm_out << " -> ";
            // Check if first destination operand needs color
            if (inst.dst_operands[0].operand == "NORTH" || inst.dst_operands[0].operand == "SOUTH" || 
                inst.dst_operands[0].operand == "EAST" || inst.dst_operands[0].operand == "WEST" ||
                inst.dst_operands[0].operand == "NORTHEAST" || inst.dst_operands[0].operand == "NORTHWEST" ||
                inst.dst_operands[0].operand == "SOUTHEAST" || inst.dst_operands[0].operand == "SOUTHWEST") {
              asm_out << "[" << inst.dst_operands[0].operand << ", R]";
            } else {
              asm_out << "[" << inst.dst_operands[0].operand << "]";
            }
            
            for (size_t i=1;i<inst.dst_operands.size();++i) {
              if (inst.dst_operands[i].operand == "NORTH" || inst.dst_operands[i].operand == "SOUTH" || 
                  inst.dst_operands[i].operand == "EAST" || inst.dst_operands[i].operand == "WEST" ||
                  inst.dst_operands[i].operand == "NORTHEAST" || inst.dst_operands[i].operand == "NORTHWEST" ||
                  inst.dst_operands[i].operand == "SOUTHEAST" || inst.dst_operands[i].operand == "SOUTHWEST") {
                asm_out << ", [" << inst.dst_operands[i].operand << ", R]";
              } else {
                asm_out << ", [" << inst.dst_operands[i].operand << "]";
              }
            }
          }
          
          // print route hints inline (for quick glance)
          if (!inst.route_hints.empty()) {
            asm_out << "  ; route_hints: ";
            for (size_t i=0;i<inst.route_hints.size();++i) {
              const RouteHint &rh = inst.route_hints[i];
              if (i) asm_out << " | ";
              asm_out << "to=("<<rh.toX<<","<<rh.toY<<"),first="<<rh.firstHop<<",hops="<<rh.hops;
            }
          }
          asm_out << "\n    }\n";
        }
        asm_out << "  }\n}\n\n";
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
