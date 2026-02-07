//===- FuseTaskPass.cpp - Task fusion for Taskflow dialect ----------------===//
//
// Fuses taskflow.task ops by merging counter chains and hyperblock bodies.
// Supports producer-consumer fusion (with intermediate store/load elimination)
// and sibling fusion (with shared input deduplication).
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "Conversion/ConversionPasses.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===----------------------------------------------------------------------===//
// Operand helpers
//===----------------------------------------------------------------------===//

/// Collects unique values from two ranges into a deduped vector with index map.
static void collectUnique(ValueRange a, ValueRange b,
                          SmallVectorImpl<Value> &result,
                          llvm::SmallDenseMap<Value, unsigned> &idx_map) {
  for (Value v : a) { idx_map[v] = result.size(); result.push_back(v); }
  for (Value v : b)
    if (!idx_map.count(v)) { idx_map[v] = result.size(); result.push_back(v); }
}

//===----------------------------------------------------------------------===//
// Profitability
//===----------------------------------------------------------------------===//

/// Placeholder: always permits fusion. MII-based profitability analysis is
/// disabled pending pipeline support for scf.for inside hyperblocks.
static bool isFusionProfitable(TaskflowTaskOp, TaskflowTaskOp,
                               bool, Value = nullptr) {
  return true;
}

//===----------------------------------------------------------------------===//
// Counter chain & hyperblock utilities
//===----------------------------------------------------------------------===//

/// Extracts all TaskflowCounterOps from a task body in definition order.
static SmallVector<TaskflowCounterOp>
extractCounterChain(TaskflowTaskOp task) {
  SmallVector<TaskflowCounterOp> chain;
  for (Operation &op : task.getBody().front())
    if (auto c = dyn_cast<TaskflowCounterOp>(&op))
      chain.push_back(c);
  return chain;
}

/// Returns true if two counter chains have identical bounds and steps.
static bool counterChainsMatch(SmallVectorImpl<TaskflowCounterOp> &a,
                               SmallVectorImpl<TaskflowCounterOp> &b) {
  if (a.size() != b.size()) return false;
  for (unsigned i = 0; i < a.size(); ++i) {
    auto get_int = [](Operation *op, StringRef name) -> int64_t {
      return op->getAttrOfType<IntegerAttr>(name).getInt();
    };
    if (get_int(a[i], "lower_bound") != get_int(b[i], "lower_bound") ||
        get_int(a[i], "upper_bound") != get_int(b[i], "upper_bound") ||
        get_int(a[i], "step")        != get_int(b[i], "step"))
      return false;
  }
  return true;
}

/// Finds the single TaskflowHyperblockOp in a task body. Returns nullptr
/// if none or multiple exist.
static TaskflowHyperblockOp findHyperblock(TaskflowTaskOp task) {
  TaskflowHyperblockOp result = nullptr;
  for (Operation &op : task.getBody().front()) {
    if (auto hb = dyn_cast<TaskflowHyperblockOp>(&op)) {
      if (result) return nullptr; // multiple hyperblocks
      result = hb;
    }
  }
  return result;
}

/// Finds the single scf::ForOp in a hyperblock body. Returns nullptr
/// if none or multiple exist.
static scf::ForOp findInnerForOp(TaskflowHyperblockOp hb) {
  scf::ForOp result = nullptr;
  for (auto &op : hb.getBody().front()) {
    if (auto f = dyn_cast<scf::ForOp>(&op)) {
      if (result) return nullptr;
      result = f;
    }
  }
  return result;
}

/// Returns true if two scf::ForOp loops have identical constant bounds
/// and step.
static bool scfForBoundsMatch(scf::ForOp a, scf::ForOp b) {
  auto get_const = [](Value v) -> std::optional<int64_t> {
    if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
      return cst.value();
    return std::nullopt;
  };
  auto al = get_const(a.getLowerBound()), bl = get_const(b.getLowerBound());
  auto au = get_const(a.getUpperBound()), bu = get_const(b.getUpperBound());
  auto as = get_const(a.getStep()),       bs = get_const(b.getStep());
  return al && bl && au && bu && as && bs &&
         *al == *bl && *au == *bu && *as == *bs;
}

//===----------------------------------------------------------------------===//
// Legality checks
//===----------------------------------------------------------------------===//

/// Returns true if task1 dominates task2 in the same block.
static bool canFuseTasks(TaskflowTaskOp t1, TaskflowTaskOp t2) {
  return t1 && t2 && t1 != t2 &&
         t1->getBlock() == t2->getBlock() &&
         t1->isBeforeInBlock(t2);
}

/// Returns true if any write output of the producer feeds the consumer.
static bool hasProducerConsumerRelation(TaskflowTaskOp producer,
                                       TaskflowTaskOp consumer) {
  for (Value r : producer.getWriteOutputs()) {
    for (Value in : consumer.getReadMemrefs())  if (r == in) return true;
    for (Value in : consumer.getWriteMemrefs()) if (r == in) return true;
  }
  for (Value r : producer.getValueOutputs())
    for (Value in : consumer.getValueInputs()) if (r == in) return true;
  return false;
}

/// Returns true if tasks share at least one input with no data dependency.
static bool areSiblingTasks(TaskflowTaskOp t1, TaskflowTaskOp t2) {
  llvm::SmallPtrSet<Value, 8> t1_mem;
  for (Value v : t1.getReadMemrefs())  t1_mem.insert(v);
  for (Value v : t1.getWriteMemrefs()) t1_mem.insert(v);
  llvm::SmallPtrSet<Value, 8> t1_val(t1.getValueInputs().begin(),
                                     t1.getValueInputs().end());
  bool share = false;
  for (Value v : t2.getReadMemrefs())  if (t1_mem.count(v)) { share = true; break; }
  if (!share)
    for (Value v : t2.getWriteMemrefs()) if (t1_mem.count(v)) { share = true; break; }
  if (!share)
    for (Value v : t2.getValueInputs())  if (t1_val.count(v)) { share = true; break; }
  return share && !hasProducerConsumerRelation(t1, t2) &&
         !hasProducerConsumerRelation(t2, t1);
}

/// Returns true if any op between producer and consumer uses producer's results.
static bool hasInterveningUses(TaskflowTaskOp producer,
                               TaskflowTaskOp consumer) {
  llvm::SmallPtrSet<Value, 8> results;
  for (Value v : producer.getWriteOutputs()) results.insert(v);
  for (Value v : producer.getValueOutputs()) results.insert(v);
  bool in_range = false;
  for (Operation &op : *producer->getBlock()) {
    if (&op == producer.getOperation()) { in_range = true; continue; }
    if (&op == consumer.getOperation()) break;
    if (in_range)
      for (Value v : op.getOperands())
        if (results.count(v)) return true;
  }
  return false;
}

/// Returns true if all outputs of the task have at most one use.
static bool hasOnlySingleUseOutputs(TaskflowTaskOp task) {
  for (Value r : task.getWriteOutputs())
    if (!r.hasOneUse() && !r.use_empty()) return false;
  for (Value r : task.getValueOutputs())
    if (!r.hasOneUse() && !r.use_empty()) return false;
  return true;
}

/// Returns true if two tasks have compatible loop structures for merging.
/// Handles both counter-chain and scf.for-inside-hyperblock cases.
static bool haveMatchingLoopStructure(TaskflowTaskOp t1, TaskflowTaskOp t2) {
  auto hb1 = findHyperblock(t1), hb2 = findHyperblock(t2);
  if (!hb1 || !hb2) return false;
  if (hb1.getIterArgs().size() > 0 || hb2.getIterArgs().size() > 0)
    return false;

  auto c1 = extractCounterChain(t1), c2 = extractCounterChain(t2);
  if (!c1.empty() && !c2.empty())
    return counterChainsMatch(c1, c2);

  // No counter chains: compares scf.for loops in hyperblock bodies.
  if (c1.empty() && c2.empty()) {
    auto f1 = findInnerForOp(hb1), f2 = findInnerForOp(hb2);
    if (f1 && f2) return scfForBoundsMatch(f1, f2);
    // Both have no loops: trivially compatible.
    if (!f1 && !f2) return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Producer-consumer fusion
//===----------------------------------------------------------------------===//

/// Fuses a producer into its consumer by merging their counter chains and
/// hyperblock bodies into a single task with direct SSA value forwarding
/// for the intermediate memref.
static TaskflowTaskOp
fuseProducerConsumerTasks(TaskflowTaskOp producer, TaskflowTaskOp consumer,
                         Value intermediate_memref, OpBuilder &builder) {
  Location loc = consumer.getLoc();
  auto prod_read  = producer.getReadMemrefs();
  auto prod_write = producer.getWriteMemrefs();
  auto prod_val   = producer.getValueInputs();
  auto cons_read  = consumer.getReadMemrefs();
  auto cons_write = consumer.getWriteMemrefs();
  auto cons_val   = consumer.getValueInputs();

  // Collects fused inputs, keeping intermediate in write_memrefs for
  // block arg availability but excluding it from consumer's read side.
  SmallVector<Value> fused_read, fused_write, fused_val;
  for (Value v : prod_read)  fused_read.push_back(v);
  for (Value v : cons_read)
    if (v != intermediate_memref) fused_read.push_back(v);
  for (Value v : prod_write) fused_write.push_back(v);
  for (Value v : cons_write)
    if (v != intermediate_memref) fused_write.push_back(v);
  for (Value v : prod_val) fused_val.push_back(v);
  for (Value v : cons_val) fused_val.push_back(v);

  // Output types come from the consumer only.
  SmallVector<Type> write_out_types(consumer.getWriteOutputs().getTypes());
  SmallVector<Type> val_out_types(consumer.getValueOutputs().getTypes());

  SmallVector<Value> orig_reads, orig_writes;
  for (Value v : producer.getOriginalReadMemrefs())  orig_reads.push_back(v);
  for (Value v : consumer.getOriginalReadMemrefs())  orig_reads.push_back(v);
  for (Value v : producer.getOriginalWriteMemrefs()) orig_writes.push_back(v);
  for (Value v : consumer.getOriginalWriteMemrefs()) orig_writes.push_back(v);

  auto fused = builder.create<TaskflowTaskOp>(
      loc, write_out_types, val_out_types,
      fused_read, fused_write, fused_val,
      builder.getStringAttr("fused_pc"),
      orig_reads, orig_writes);

  Block *body = new Block();
  fused.getBody().push_back(body);
  for (Value v : fused_read)  body->addArgument(v.getType(), loc);
  for (Value v : fused_write) body->addArgument(v.getType(), loc);
  for (Value v : fused_val)   body->addArgument(v.getType(), loc);

  unsigned fused_read_n  = fused_read.size();
  unsigned fused_write_n = fused_write.size();

  // --- Maps producer block args to fused block args ---
  IRMapping mapping;
  Block &prod_body = producer.getBody().front();
  for (unsigned i = 0; i < prod_read.size(); ++i)
    mapping.map(prod_body.getArgument(i), body->getArgument(i));
  for (unsigned i = 0; i < prod_write.size(); ++i)
    mapping.map(prod_body.getArgument(prod_read.size() + i),
                body->getArgument(fused_read_n + i));
  for (unsigned i = 0; i < prod_val.size(); ++i)
    mapping.map(prod_body.getArgument(prod_read.size() + prod_write.size() + i),
                body->getArgument(fused_read_n + fused_write_n + i));

  // Identifies the producer's task-body block arg for the intermediate write
  // memref. The intermediate_memref is the producer's write OUTPUT result;
  // finds which write output index it corresponds to, then gets the matching
  // write memref block arg.
  BlockArgument prod_intermediate_arg = nullptr;
  for (unsigned i = 0; i < producer.getWriteOutputs().size(); ++i)
    if (producer.getWriteOutputs()[i] == intermediate_memref)
      prod_intermediate_arg = prod_body.getArgument(prod_read.size() + i);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  // --- Maps consumer block args to fused block args ---
  Block &cons_body = consumer.getBody().front();
  unsigned cons_read_fused = prod_read.size();
  for (unsigned i = 0; i < cons_read.size(); ++i) {
    if (cons_read[i] == intermediate_memref) {
      if (prod_intermediate_arg)
        mapping.map(cons_body.getArgument(i),
                    mapping.lookupOrDefault(prod_intermediate_arg));
    } else {
      mapping.map(cons_body.getArgument(i),
                  body->getArgument(cons_read_fused++));
    }
  }
  unsigned cons_write_fused = fused_read_n + prod_write.size();
  for (unsigned i = 0; i < cons_write.size(); ++i) {
    if (cons_write[i] == intermediate_memref) {
      if (prod_intermediate_arg)
        mapping.map(cons_body.getArgument(cons_read.size() + i),
                    mapping.lookupOrDefault(prod_intermediate_arg));
    } else {
      mapping.map(cons_body.getArgument(cons_read.size() + i),
                  body->getArgument(cons_write_fused++));
    }
  }
  unsigned cons_val_fused = fused_read_n + fused_write_n + prod_val.size();
  for (unsigned i = 0; i < cons_val.size(); ++i)
    mapping.map(cons_body.getArgument(cons_read.size() + cons_write.size() + i),
                body->getArgument(cons_val_fused + i));

  // Clones counter chain from producer.
  auto prod_counters = extractCounterChain(producer);
  for (auto ctr : prod_counters) builder.clone(*ctr, mapping);

  // Clones non-counter, non-hyperblock, non-yield ops from both task bodies
  // (e.g. arith.constant for loop bounds).
  for (Operation &op : prod_body) {
    if (isa<TaskflowCounterOp, TaskflowHyperblockOp, TaskflowYieldOp>(&op))
      continue;
    builder.clone(op, mapping);
  }
  for (Operation &op : cons_body) {
    if (isa<TaskflowCounterOp, TaskflowHyperblockOp, TaskflowYieldOp>(&op))
      continue;
    builder.clone(op, mapping);
  }

  // Locates both hyperblocks.
  auto prod_hb = findHyperblock(producer);
  auto cons_hb = findHyperblock(consumer);
  Block &prod_hb_body = prod_hb.getBody().front();
  Block &cons_hb_body = cons_hb.getBody().front();

  // Creates fused hyperblock.
  SmallVector<Value> triggers;
  for (Value v : prod_hb.getIndices())
    triggers.push_back(mapping.lookupOrDefault(v));
  auto fused_hb = builder.create<TaskflowHyperblockOp>(
      loc, TypeRange{}, triggers, ValueRange{});
  Block *hb_body = &fused_hb.getBody().emplaceBlock();
  for (auto arg : prod_hb_body.getArguments())
    hb_body->addArgument(arg.getType(), loc);

  // Maps hyperblock block args.
  for (unsigned i = 0; i < prod_hb_body.getNumArguments(); ++i)
    mapping.map(prod_hb_body.getArgument(i), hb_body->getArgument(i));
  for (unsigned i = 0; i < cons_hb_body.getNumArguments(); ++i)
    mapping.map(cons_hb_body.getArgument(i), hb_body->getArgument(i));

  // Identifies the consumer's intermediate read block arg for load skipping.
  BlockArgument cons_intermediate_arg = nullptr;
  for (unsigned i = 0; i < cons_read.size(); ++i)
    if (cons_read[i] == intermediate_memref)
      cons_intermediate_arg = cons_body.getArgument(i);

  {
    OpBuilder::InsertionGuard hb_guard(builder);
    builder.setInsertionPointToStart(hb_body);

    auto prod_for = findInnerForOp(prod_hb);
    auto cons_for = findInnerForOp(cons_hb);

    if (prod_for && cons_for) {
      // --- SCF loop fusion: merges for-body ops into a single scf.for ---

      // Clones non-for, non-yield ops from both hyperblock bodies.
      for (Operation &op : prod_hb_body) {
        if (isa<TaskflowHyperblockYieldOp, scf::ForOp>(&op)) continue;
        builder.clone(op, mapping);
      }
      for (Operation &op : cons_hb_body) {
        if (isa<TaskflowHyperblockYieldOp, scf::ForOp>(&op)) continue;
        builder.clone(op, mapping);
      }

      // Creates merged scf.for with producer's bounds.
      Value lb   = mapping.lookupOrDefault(prod_for.getLowerBound());
      Value ub   = mapping.lookupOrDefault(prod_for.getUpperBound());
      Value step = mapping.lookupOrDefault(prod_for.getStep());
      auto merged_for = builder.create<scf::ForOp>(loc, lb, ub, step);
      mapping.map(prod_for.getInductionVar(), merged_for.getInductionVar());
      mapping.map(cons_for.getInductionVar(), merged_for.getInductionVar());

      OpBuilder::InsertionGuard for_guard(builder);
      builder.setInsertionPoint(merged_for.getBody()->getTerminator());

      // Clones producer for-body; skips store to intermediate, records
      // the forwarded value.
      Value forwarded = nullptr;
      for (Operation &op : *prod_for.getBody()) {
        if (isa<scf::YieldOp>(&op)) continue;
        if (auto store = dyn_cast<memref::StoreOp>(&op)) {
          if (prod_intermediate_arg &&
              store.getMemRef() == prod_intermediate_arg) {
            forwarded = mapping.lookupOrDefault(store.getValueToStore());
            continue;
          }
        }
        builder.clone(op, mapping);
      }

      // Clones consumer for-body; replaces loads from intermediate
      // with the forwarded value.
      for (Operation &op : *cons_for.getBody()) {
        if (isa<scf::YieldOp>(&op)) continue;
        if (auto load = dyn_cast<memref::LoadOp>(&op)) {
          if (cons_intermediate_arg && forwarded &&
              load.getMemRef() == cons_intermediate_arg) {
            mapping.map(load.getResult(), forwarded);
            continue;
          }
        }
        builder.clone(op, mapping);
      }

    } else {
      // --- Counter-chain path: clones hyperblock bodies sequentially ---
      Value forwarded = nullptr;
      for (Operation &op : prod_hb_body) {
        if (isa<TaskflowHyperblockYieldOp>(&op)) continue;
        if (auto store = dyn_cast<memref::StoreOp>(&op)) {
          if (prod_intermediate_arg &&
              store.getMemRef() == prod_intermediate_arg) {
            forwarded = mapping.lookupOrDefault(store.getValueToStore());
            continue;
          }
        }
        builder.clone(op, mapping);
      }
      for (Operation &op : cons_hb_body) {
        if (isa<TaskflowHyperblockYieldOp>(&op)) continue;
        if (auto load = dyn_cast<memref::LoadOp>(&op)) {
          if (cons_intermediate_arg && forwarded &&
              load.getMemRef() == cons_intermediate_arg) {
            mapping.map(load.getResult(), forwarded);
            continue;
          }
        }
        builder.clone(op, mapping);
      }
    }

    builder.create<TaskflowHyperblockYieldOp>(loc);
  }

  // Creates the task yield from consumer's yield operands.
  auto cons_yield = cast<TaskflowYieldOp>(consumer.getBody().front().getTerminator());
  SmallVector<Value> yield_mem, yield_val;
  for (Value v : cons_yield.getMemoryResults())
    yield_mem.push_back(mapping.lookupOrDefault(v));
  for (Value v : cons_yield.getValueResults())
    yield_val.push_back(mapping.lookupOrDefault(v));
  builder.create<TaskflowYieldOp>(loc, yield_mem, yield_val);
  return fused;
}

//===----------------------------------------------------------------------===//
// Sibling fusion
//===----------------------------------------------------------------------===//

/// Fuses two sibling tasks by merging their counter chains and hyperblock
/// bodies into a single task with deduplicated inputs.
static TaskflowTaskOp fuseSiblingTasks(TaskflowTaskOp t1, TaskflowTaskOp t2,
                                       OpBuilder &builder) {
  Location loc = t1.getLoc();

  // Deduplicates inputs across both tasks.
  SmallVector<Value> fused_read, fused_write, fused_val;
  llvm::SmallDenseMap<Value, unsigned> read_idx, write_idx, val_idx;
  collectUnique(t1.getReadMemrefs(),  t2.getReadMemrefs(),  fused_read,  read_idx);
  collectUnique(t1.getWriteMemrefs(), t2.getWriteMemrefs(), fused_write, write_idx);
  collectUnique(t1.getValueInputs(),  t2.getValueInputs(),  fused_val,   val_idx);

  // Combined output types.
  SmallVector<Type> write_out_types, val_out_types;
  write_out_types.append(t1.getWriteOutputs().getTypes().begin(),
                         t1.getWriteOutputs().getTypes().end());
  write_out_types.append(t2.getWriteOutputs().getTypes().begin(),
                         t2.getWriteOutputs().getTypes().end());
  val_out_types.append(t1.getValueOutputs().getTypes().begin(),
                       t1.getValueOutputs().getTypes().end());
  val_out_types.append(t2.getValueOutputs().getTypes().begin(),
                       t2.getValueOutputs().getTypes().end());

  SmallVector<Value> orig_reads, orig_writes;
  for (Value v : t1.getOriginalReadMemrefs())  orig_reads.push_back(v);
  for (Value v : t2.getOriginalReadMemrefs())  orig_reads.push_back(v);
  for (Value v : t1.getOriginalWriteMemrefs()) orig_writes.push_back(v);
  for (Value v : t2.getOriginalWriteMemrefs()) orig_writes.push_back(v);

  auto fused = builder.create<TaskflowTaskOp>(
      loc, write_out_types, val_out_types,
      fused_read, fused_write, fused_val,
      builder.getStringAttr("fused_sibling"),
      orig_reads, orig_writes);

  Block *body = new Block();
  fused.getBody().push_back(body);
  for (Value v : fused_read)  body->addArgument(v.getType(), loc);
  for (Value v : fused_write) body->addArgument(v.getType(), loc);
  for (Value v : fused_val)   body->addArgument(v.getType(), loc);

  unsigned rn = fused_read.size(), wn = fused_write.size();

  // Lambda to map a task's block args to fused block args via index maps.
  auto mapBlockArgs = [&](TaskflowTaskOp task, IRMapping &m) {
    Block &tb = task.getBody().front();
    auto r = task.getReadMemrefs();
    auto w = task.getWriteMemrefs();
    auto v = task.getValueInputs();
    for (unsigned i = 0; i < r.size(); ++i)
      m.map(tb.getArgument(i), body->getArgument(read_idx[r[i]]));
    for (unsigned i = 0; i < w.size(); ++i)
      m.map(tb.getArgument(r.size() + i),
            body->getArgument(rn + write_idx[w[i]]));
    for (unsigned i = 0; i < v.size(); ++i)
      m.map(tb.getArgument(r.size() + w.size() + i),
            body->getArgument(rn + wn + val_idx[v[i]]));
  };

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  // Clones counter chain from t1.
  IRMapping mapping;
  mapBlockArgs(t1, mapping);
  auto counters = extractCounterChain(t1);
  for (auto ctr : counters) builder.clone(*ctr, mapping);

  // Clones non-counter, non-hyperblock, non-yield ops from both task bodies.
  Block &t1_body = t1.getBody().front();
  Block &t2_body = t2.getBody().front();
  IRMapping mapping2;
  mapBlockArgs(t2, mapping2);
  for (Operation &op : t1_body) {
    if (isa<TaskflowCounterOp, TaskflowHyperblockOp, TaskflowYieldOp>(&op))
      continue;
    builder.clone(op, mapping);
  }
  for (Operation &op : t2_body) {
    if (isa<TaskflowCounterOp, TaskflowHyperblockOp, TaskflowYieldOp>(&op))
      continue;
    builder.clone(op, mapping2);
  }

  // Locates both hyperblocks.
  auto hb1 = findHyperblock(t1), hb2 = findHyperblock(t2);
  Block &hb1_body = hb1.getBody().front();
  Block &hb2_body = hb2.getBody().front();

  // Creates fused hyperblock.
  SmallVector<Value> triggers;
  for (Value v : hb1.getIndices())
    triggers.push_back(mapping.lookupOrDefault(v));
  auto fused_hb = builder.create<TaskflowHyperblockOp>(
      loc, TypeRange{}, triggers, ValueRange{});
  Block *hb_body = &fused_hb.getBody().emplaceBlock();
  for (auto arg : hb1_body.getArguments())
    hb_body->addArgument(arg.getType(), loc);

  // Maps hyperblock block args.
  for (unsigned i = 0; i < hb1_body.getNumArguments(); ++i)
    mapping.map(hb1_body.getArgument(i), hb_body->getArgument(i));
  for (unsigned i = 0; i < hb2_body.getNumArguments(); ++i)
    mapping2.map(hb2_body.getArgument(i), hb_body->getArgument(i));

  {
    OpBuilder::InsertionGuard hb_guard(builder);
    builder.setInsertionPointToStart(hb_body);

    auto for1 = findInnerForOp(hb1), for2 = findInnerForOp(hb2);

    if (for1 && for2) {
      // --- SCF loop fusion: merges for-body ops into a single scf.for ---

      // Clones non-for, non-yield ops from both hyperblock bodies.
      for (Operation &op : hb1_body) {
        if (isa<TaskflowHyperblockYieldOp, scf::ForOp>(&op)) continue;
        builder.clone(op, mapping);
      }
      for (Operation &op : hb2_body) {
        if (isa<TaskflowHyperblockYieldOp, scf::ForOp>(&op)) continue;
        builder.clone(op, mapping2);
      }

      // Creates merged scf.for with t1's bounds.
      Value lb   = mapping.lookupOrDefault(for1.getLowerBound());
      Value ub   = mapping.lookupOrDefault(for1.getUpperBound());
      Value step = mapping.lookupOrDefault(for1.getStep());
      auto merged_for = builder.create<scf::ForOp>(loc, lb, ub, step);
      mapping.map(for1.getInductionVar(), merged_for.getInductionVar());
      mapping2.map(for2.getInductionVar(), merged_for.getInductionVar());

      OpBuilder::InsertionGuard for_guard(builder);
      builder.setInsertionPoint(merged_for.getBody()->getTerminator());

      // Clones t1 for-body.
      for (Operation &op : *for1.getBody()) {
        if (isa<scf::YieldOp>(&op)) continue;
        builder.clone(op, mapping);
      }
      // Clones t2 for-body.
      for (Operation &op : *for2.getBody()) {
        if (isa<scf::YieldOp>(&op)) continue;
        builder.clone(op, mapping2);
      }

    } else {
      // --- Counter-chain path: clones hyperblock bodies sequentially ---
      for (Operation &op : hb1_body) {
        if (isa<TaskflowHyperblockYieldOp>(&op)) continue;
        builder.clone(op, mapping);
      }
      for (Operation &op : hb2_body) {
        if (isa<TaskflowHyperblockYieldOp>(&op)) continue;
        builder.clone(op, mapping2);
      }
    }

    builder.create<TaskflowHyperblockYieldOp>(loc);
  }

  // Creates combined task yield.
  auto t1_yield = cast<TaskflowYieldOp>(t1.getBody().front().getTerminator());
  auto t2_yield = cast<TaskflowYieldOp>(t2.getBody().front().getTerminator());
  SmallVector<Value> all_mem, all_val;
  for (Value v : t1_yield.getMemoryResults())
    all_mem.push_back(mapping.lookupOrDefault(v));
  for (Value v : t1_yield.getValueResults())
    all_val.push_back(mapping.lookupOrDefault(v));
  for (Value v : t2_yield.getMemoryResults())
    all_mem.push_back(mapping2.lookupOrDefault(v));
  for (Value v : t2_yield.getValueResults())
    all_val.push_back(mapping2.lookupOrDefault(v));
  builder.create<TaskflowYieldOp>(loc, all_mem, all_val);
  return fused;
}

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

/// Fuses a producer task into its consumer when the producer's output feeds
/// directly into the consumer and the loop structures match.
struct ProducerConsumerTaskFusion
    : public OpRewritePattern<TaskflowTaskOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TaskflowTaskOp consumer,
                                PatternRewriter &rewriter) const override {
    TaskflowTaskOp producer = nullptr;
    Value intermediate;

    // Searches consumer's read and write memrefs for a fusible producer.
    auto tryFindProducer = [&](ValueRange inputs) -> bool {
      for (Value in : inputs) {
        auto def = in.getDefiningOp<TaskflowTaskOp>();
        if (!canFuseTasks(def, consumer) ||
            !hasOnlySingleUseOutputs(def) ||
            hasInterveningUses(def, consumer) ||
            !haveMatchingLoopStructure(def, consumer) ||
            !isFusionProfitable(def, consumer, true, in))
          continue;
        producer = def;
        intermediate = in;
        return true;
      }
      return false;
    };
    if (!tryFindProducer(consumer.getReadMemrefs()) &&
        !tryFindProducer(consumer.getWriteMemrefs()))
      return failure();

    auto fused = fuseProducerConsumerTasks(producer, consumer,
                                           intermediate, rewriter);
    for (auto [o, n] : llvm::zip(consumer.getWriteOutputs(),
                                 fused.getWriteOutputs()))
      o.replaceAllUsesWith(n);
    for (auto [o, n] : llvm::zip(consumer.getValueOutputs(),
                                 fused.getValueOutputs()))
      o.replaceAllUsesWith(n);
    rewriter.eraseOp(consumer);
    rewriter.eraseOp(producer);
    return success();
  }
};

/// Fuses sibling tasks that share inputs and have matching loop structures.
struct SiblingTaskFusion : public OpRewritePattern<TaskflowTaskOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TaskflowTaskOp t1,
                                PatternRewriter &rewriter) const override {
    TaskflowTaskOp t2 = nullptr;
    for (Operation *op = t1->getNextNode(); op; op = op->getNextNode()) {
      auto next = dyn_cast<TaskflowTaskOp>(op);
      if (!next) continue;
      if (areSiblingTasks(t1, next) && canFuseTasks(t1, next) &&
          haveMatchingLoopStructure(t1, next) &&
          isFusionProfitable(t1, next, false)) {
        t2 = next;
        break;
      }
    }
    if (!t2) return failure();

    auto fused = fuseSiblingTasks(t1, t2, rewriter);
    unsigned t1_wo = t1.getWriteOutputs().size();
    unsigned t1_vo = t1.getValueOutputs().size();
    for (unsigned i = 0; i < t1_wo; ++i)
      t1.getWriteOutputs()[i].replaceAllUsesWith(fused.getWriteOutputs()[i]);
    for (unsigned i = 0; i < t1_vo; ++i)
      t1.getValueOutputs()[i].replaceAllUsesWith(fused.getValueOutputs()[i]);
    for (unsigned i = 0; i < t2.getWriteOutputs().size(); ++i)
      t2.getWriteOutputs()[i].replaceAllUsesWith(
          fused.getWriteOutputs()[t1_wo + i]);
    for (unsigned i = 0; i < t2.getValueOutputs().size(); ++i)
      t2.getValueOutputs()[i].replaceAllUsesWith(
          fused.getValueOutputs()[t1_vo + i]);
    rewriter.eraseOp(t1);
    rewriter.eraseOp(t2);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

/// Applies producer-consumer and sibling task fusion using MII-based
/// profitability analysis and loop body merging.
struct FuseTaskPass
    : public PassWrapper<FuseTaskPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseTaskPass)

  StringRef getArgument() const override { return "fuse-task"; }
  StringRef getDescription() const override {
    return "Fuses taskflow.task ops via counter chain and hyperblock merging.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, func::FuncDialect,
                    arith::ArithDialect, memref::MemRefDialect,
                    neura::NeuraDialect, taskflow::TaskflowDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ProducerConsumerTaskFusion>(&getContext(), /*benefit=*/10);
    patterns.add<SiblingTaskFusion>(&getContext(), /*benefit=*/5);
    if (failed(applyPatternsGreedily(getOperation(),
                                     FrozenRewritePatternSet(std::move(patterns)))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::taskflow {
std::unique_ptr<Pass> createFuseTaskPass() {
  return std::make_unique<FuseTaskPass>();
}
} // namespace mlir::taskflow
