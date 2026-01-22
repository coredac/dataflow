//===- HardwareTemplate.cpp - Hardware Template Data Structures and Helpers -===//
//
// This file contains data structures and helper functions for hardware template
// merging, including pattern extraction, template creation, and cost calculation.
//
//===----------------------------------------------------------------------===//

#include "NeuraDialect/Transforms/GraphMining/HardwareTemplate.h"
#include "NeuraDialect/NeuraOps.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/ADT/DenseMap.h"
#include <vector>
#include <map>
#include <set>
#include <string>
#include <algorithm>
#include <cmath>

using namespace mlir;

namespace mlir::neura {

// Initializes the operation cost model with default costs for each operation type.
// TODO: The cost model should be imported from the input file.
OperationCostModel::OperationCostModel() {
  costs["neura.div"] = 100; costs["neura.fdiv"] = 100; costs["neura.rem"] = 80;
  costs["neura.load"] = 50; costs["neura.store"] = 50;
  costs["neura.load_indexed"] = 55; costs["neura.store_indexed"] = 55;
  costs["neura.mul"] = 30; costs["neura.fmul"] = 35;
  costs["neura.gep"] = 20;
  costs["neura.add"] = 10; costs["neura.sub"] = 10;
  costs["neura.fadd"] = 12; costs["neura.fsub"] = 12;
  costs["neura.icmp"] = 15; costs["neura.fcmp"] = 15;
  costs["neura.and"] = 5; costs["neura.or"] = 5; costs["neura.not"] = 5;
  costs["neura.sel"] = 6; costs["neura.phi"] = 3;
  costs["neura.grant_predicate"] = 3; costs["neura.grant_once"] = 3;
  costs["neura.cast"] = 2; costs["neura.sext"] = 2; costs["neura.zext"] = 2;
  costs["neura.data_mov"] = 1; costs["neura.constant"] = 1;
}

// Returns the cost for a given operation, or 5.0 as default if not found.
double OperationCostModel::get(const std::string& op) const {
  auto it = costs.find(op);
  return it != costs.end() ? it->second : 5.0;
}

// Returns the maximum cost among all operations in a slot.
double OperationCostModel::slotCost(const std::set<std::string>& ops) const {
  double mx = 0;
  for (const auto& op : ops) mx = std::max(mx, get(op));
  return mx;
}

// Returns the total cost for a pattern by summing costs of all its operations.
double OperationCostModel::patternCost(const std::vector<std::string>& ops) const {
  double sum = 0;
  for (const auto& op : ops) sum += get(op);
  return sum;
}


// Constructs a HardwarePattern with the given id, name, and frequency.
HardwarePattern::HardwarePattern(int64_t i, const std::string& n, int64_t f) 
    : id(i), name(n), freq(f), cost(0) {}

// Constructs a HardwareSlot with the given id.
HardwareSlot::HardwareSlot(int i) : id(i) {}

// Constructs a HardwareTemplate with the given id and one instance.
HardwareTemplate::HardwareTemplate(int i) : id(i), instances(1) {}

// Adds a new slot at the end of the template.
void HardwareTemplate::addSlot() {
  int newId = slots.size();
  slots.emplace_back(newId);
}
  
// Inserts a new slot at the front of the template, shifting all existing slot IDs.
void HardwareTemplate::insertSlotAtFront() {
  for (auto& slot : slots) {
    slot.id++;
  }
  
  slots.insert(slots.begin(), HardwareSlot(0));
  
  for (auto& pair : mapping) {
    for (int& slotIdx : pair.second) {
      slotIdx++;
    }
  }
}
  
// Checks if a connection can be made between two slots.
bool HardwareTemplate::canRoute(int from, int to) const {
  if (from < 0) return true;
  return from < to;
}

// Finds a mapping for a pattern into the existing template.
std::vector<int> HardwareTemplate::findMapping(const std::vector<std::string>& patOps) const {
  std::vector<int> bestMapping;
  int bestScore = -1;
  
  dfsWithScoring(patOps, 0, -1, {}, bestMapping, bestScore);
  return bestMapping;
}
  
// Performs DFS with scoring to find the best mapping for a pattern.
void HardwareTemplate::dfsWithScoring(const std::vector<std::string>& patOps, size_t opIdx, int prevSlot, std::vector<int> cur, std::vector<int>& bestMapping, int& bestScore) const {
  if (opIdx >= patOps.size()) {
    int score = 0;
    for (size_t i = 0; i < patOps.size(); ++i) {
      int slot = cur[i];
      const std::string& op = patOps[i];
      
      if (slotCanHandle(slot, op) && !slots[slot].ops.empty()) {
        score += 10;
      }
    }
    if (score > bestScore) {
      bestScore = score;
      bestMapping = cur;
    }
    return;
  }
  
  const std::string& op = patOps[opIdx];
  
  std::vector<std::pair<int, size_t>> candidates;
  for (size_t s = 0; s < slots.size(); ++s) {
    if (!slotCanHandle(s, op)) continue;
    
    if (!slots[s].ops.empty()) {
      candidates.push_back({100, s});
    } else {
      candidates.push_back({50, s});
    }
  }
  
  std::sort(candidates.rbegin(), candidates.rend());
  
  for (const auto& cand : candidates) {
    size_t s = cand.second;
    if (!canRoute(prevSlot, (int)s)) continue;
    if (std::find(cur.begin(), cur.end(), (int)s) != cur.end()) continue;
    
    auto next = cur;
    next.push_back((int)s);
    dfsWithScoring(patOps, opIdx + 1, (int)s, next, bestMapping, bestScore);
  }
}
  
// Performs DFS search to find a mapping, wrapping dfsWithScoring.
std::vector<int> HardwareTemplate::dfs(const std::vector<std::string>& patOps, size_t opIdx, int prevSlot, std::vector<int> cur) const {
  std::vector<int> best;
  int score = -1;
  dfsWithScoring(patOps, opIdx, prevSlot, cur, best, score);
  return best;
}
  
// Checks if a slot can handle an operation.
bool HardwareTemplate::slotCanHandle(size_t s, const std::string& op) const {
  if (slots[s].ops.count(op)) return true;
  
  if (slots[s].ops.empty()) return true;
  
  for (const auto& existing : slots[s].ops) {
    if (!compatible(existing, op)) return false;
  }
  return true;
}
  
// Checks if two operations are compatible and can share the same slot.
bool HardwareTemplate::compatible(const std::string& a, const std::string& b) {
  if (a == b) return true;
  
  auto inGroup = [](const std::string& op, const std::vector<std::string>& group) -> bool {
    for (const auto& keyword : group) {
      if (op.find(keyword) != std::string::npos) return true;
    }
    return false;
  };
  
  static const std::vector<std::vector<std::string>> compatibleGroups = {
    {"add", "sub"},
    {"grant_once", "grant_predicate", "grant_always"},
    {"and", "or", "not", "xor"},
    {"load", "store"}
  };
  
  for (const auto& group : compatibleGroups) {
    if (inGroup(a, group) && inGroup(b, group)) {
      return true;
    }
  }
  
  return false;
}

// Tries to accommodate a pattern into the existing template.
bool HardwareTemplate::tryAccommodate(const HardwarePattern& pat, const OperationCostModel& cm, std::vector<int>& outMapping, double& outCostIncrease) {
  auto m = findMapping(pat.ops);
  if (!m.empty()) {
    double oldCost = computeCost(cm);
    
    for (size_t i = 0; i < m.size(); ++i) {
      slots[m[i]].ops.insert(pat.ops[i]);
    }
    double newCost = computeCost(cm);
    
    for (size_t i = 0; i < m.size(); ++i) {
      slots[m[i]].ops.erase(pat.ops[i]);
    }
    
    outMapping = m;
    outCostIncrease = newCost - oldCost;
    return true;
  }

  int origSize = slots.size();
  int needed = pat.ops.size();
  
  bool shouldInsertAtFront = false;
  if (!pat.ops.empty() && !slots.empty() && slotCanHandle(0, pat.ops[0])) {
    shouldInsertAtFront = true;
  }
  
  int slotsToAdd = std::max(0, needed + 2 - (int)slots.size());
  
  if (shouldInsertAtFront && slotsToAdd > 0) {
    for (int i = 0; i < slotsToAdd; ++i) {
      insertSlotAtFront();
    }
  } else { 
    while ((int)slots.size() < needed + 2) {
      addSlot();
    }
  }
  
  m = findMapping(pat.ops);
  if (!m.empty()) {
    double oldCost = computeCost(cm);
    for (size_t i = 0; i < m.size(); ++i) {
      slots[m[i]].ops.insert(pat.ops[i]);
    }
    double newCost = computeCost(cm);
    for (size_t i = 0; i < m.size(); ++i) {
      slots[m[i]].ops.erase(pat.ops[i]);
    }
    
    outMapping = m;
    outCostIncrease = newCost - oldCost;
    return true;
  }
  
  while ((int)slots.size() > origSize) {
    slots.pop_back();
  }
  
  return false;
}
  
// Applies a mapping to the template, adding the pattern's operations to slots.
void HardwareTemplate::applyMapping(const HardwarePattern& pat, const std::vector<int>& m) {
  patterns.push_back(pat.id);
  mapping[pat.id] = m;
  for (size_t i = 0; i < m.size(); ++i) {
    if (!slotCanHandle(m[i], pat.ops[i])) {
      continue;
    }
    slots[m[i]].ops.insert(pat.ops[i]);
  }
}
  
// Computes the total cost of the template based on operations in each slot.
double HardwareTemplate::computeCost(const OperationCostModel& cm) const {
  double sum = 0;
  
  std::map<std::string, int> opSlotCount;
  for (const auto& slot : slots) {
    for (const auto& op : slot.ops) {
      opSlotCount[op]++;
    }
  }
  
  std::set<std::string> countedOps;
  for (const auto& slot : slots) {
    for (const auto& op : slot.ops) {
      if (countedOps.count(op)) continue;
      countedOps.insert(op);
      sum += cm.get(op) * opSlotCount[op];
    }
  }
  
  return sum;
}

// Extracts operations from fused op and linearizes them via topological sort.
void extractPatternOps(neura::FusedOp fop, HardwarePattern& pat) {
  Region& body = fop.getBody();
  if (body.empty()) return;
  Block& blk = body.front();
  
  llvm::DenseMap<Operation*, int> opId;
  std::vector<std::string> opNames;
  std::vector<std::vector<int>> preds;
  
  int id = 0;
  for (Operation& op : blk.getOperations()) {
    std::string name = op.getName().getStringRef().str();
    if (name == "neura.yield") continue;
    opId[&op] = id++;
    opNames.push_back(name);
    preds.push_back({});
  }
  
  int idx = 0;
  for (Operation& op : blk.getOperations()) {
    std::string name = op.getName().getStringRef().str();
    if (name == "neura.yield") continue;
    for (Value v : op.getOperands()) {
      if (Operation* def = v.getDefiningOp()) {
        if (opId.count(def)) {
          preds[idx].push_back(opId[def]);
        }
      }
    }
    idx++;
  }
  
  int n = opNames.size();
  std::vector<int> level(n, 0);
  std::vector<int> inDeg(n, 0);
  for (int i = 0; i < n; ++i) inDeg[i] = preds[i].size();
  
  std::vector<int> q;
  for (int i = 0; i < n; ++i) if (inDeg[i] == 0) q.push_back(i);
  
  for (size_t h = 0; h < q.size(); ++h) {
    int cur = q[h];
    for (int i = 0; i < n; ++i) {
      for (int p : preds[i]) {
        if (p == cur) {
          level[i] = std::max(level[i], level[cur] + 1);
          if (--inDeg[i] == 0) q.push_back(i);
        }
      }
    }
  }
  
  std::vector<int> order;
  for (int i = 0; i < n; ++i) order.push_back(i);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return level[a] < level[b];
  });
  
  // Build mapping from old index to new index after reordering.
  std::vector<int> oldToNew(n);
  for (int newIdx = 0; newIdx < n; ++newIdx) {
    oldToNew[order[newIdx]] = newIdx;
  }
  
  for (int i : order) {
    pat.ops.push_back(opNames[i]);
    pat.opLevels.push_back(level[i]);
    
    std::vector<int> remappedPreds;
    for (int p : preds[i]) {
      remappedPreds.push_back(oldToNew[p]);
    }
    pat.opPreds.push_back(remappedPreds);
  }
}

// Extracts all patterns from module.
void extractPatterns(ModuleOp module, std::vector<HardwarePattern>& patterns, OperationCostModel& costModel) {
  module.walk([&](neura::FusedOp fop) {
    int64_t pid = fop.getPatternId();
    if (std::find_if(patterns.begin(), patterns.end(), 
        [pid](const HardwarePattern& p) { return p.id == pid; }) != patterns.end()) return;
    
    HardwarePattern pat(pid, fop.getPatternName().str(), fop.getFrequency());
    extractPatternOps(fop, pat);
    pat.cost = costModel.patternCost(pat.ops);
    patterns.push_back(pat);
  });
}

// Extracts all standalone operations from module.
void extractAllStandaloneOps(ModuleOp module, std::set<std::string>& allOps) {
  module.walk([&](Operation* op) {
    if (isa<neura::FusedOp>(op)) return;
    
    Operation* parent = op->getParentOp();
    while (parent) {
      if (isa<neura::FusedOp>(parent)) return;
      parent = parent->getParentOp();
    }
    
    std::string opName = op->getName().getStringRef().str();
    if (opName.find("neura.") == 0) {
      if (opName == "neura.yield" || opName == "neura.fused_op") return;
      allOps.insert(opName);
    }
  });
}

// Creates hardware templates from patterns.
void createHardwareTemplates(const std::vector<HardwarePattern>& patterns, std::vector<HardwareTemplate>& templates, OperationCostModel& costModel) {
  auto countDistinctOps = [](const HardwarePattern& p) -> int {
    std::set<std::string> distinctOps;
    for (const auto& op : p.ops) {
      distinctOps.insert(op);
    }
    return distinctOps.size();
  };
  
  std::vector<int> order;
  for (size_t i = 0; i < patterns.size(); ++i) order.push_back(i);
  std::sort(order.begin(), order.end(), [&patterns, &countDistinctOps](int a, int b) {
    int distA = countDistinctOps(patterns[a]);
    int distB = countDistinctOps(patterns[b]);
    if (distA != distB) {
      return distA > distB;
    }
    return patterns[a].cost > patterns[b].cost;
  });
  
  for (int idx : order) {
    const HardwarePattern& pat = patterns[idx];
    
    int bestT = -1;
    std::vector<int> bestM;
    double bestInc = 1e18;
    
    for (size_t t = 0; t < templates.size(); ++t) {
      HardwareTemplate tempTmpl = templates[t];
      std::vector<int> m;
      double inc;
      if (tempTmpl.tryAccommodate(pat, costModel, m, inc)) {
        if (inc < bestInc) {
          bestInc = inc;
          bestT = t;
          bestM = m;
        }
      }
    }
    
    double newTmplCost = pat.cost;
    
    if (bestT >= 0 && bestInc <= newTmplCost * 0.5) {
      while ((int)templates[bestT].slots.size() < *std::max_element(bestM.begin(), bestM.end()) + 1) {
        templates[bestT].addSlot();
      }
      templates[bestT].applyMapping(pat, bestM);
    } else {
      HardwareTemplate t(templates.size());
      for (size_t i = 0; i < pat.ops.size(); ++i) {
        t.addSlot();
      }
      std::vector<int> m;
      for (size_t i = 0; i < pat.ops.size(); ++i) m.push_back(i);
      t.applyMapping(pat, m);
      templates.push_back(t);
    }
  }
  
  for (auto& t : templates) {
    int64_t totalFreq = 0;
    for (int64_t pid : t.patterns) {
      for (const auto& p : patterns) {
        if (p.id == pid) { totalFreq += p.freq; break; }
      }
    }
    t.instances = std::max(1, (int)std::ceil(totalFreq / 10.0));
  }
}

// Generates slot connections for all templates based on pattern mappings.
// For each pattern, creates connections between consecutive slots in its mapping.
// Handles bypass cases where slots are skipped (e.g., mapping [0, 2] creates connection 0->2).
void generateConnections(const std::vector<HardwarePattern>& patterns, std::vector<HardwareTemplate>& templates) {
  for (auto& tmpl : templates) {
    tmpl.connections.clear();
    
    // For each pattern mapped to this template, generate connections based on its slot mapping.
    for (const auto& [pid, slotMapping] : tmpl.mapping) {
      if (slotMapping.size() < 2) continue;  // Need at least 2 slots to form a connection
      
      // Creates connections between consecutive slots in the pattern's execution order.
      // For example, if mapping is [0, 1, 2], creates connections 0->1 and 1->2.
      // If mapping is [0, 2] (bypassing slot 1), creates connection 0->2.
      for (size_t i = 0; i < slotMapping.size() - 1; ++i) {
        int from = slotMapping[i];
        int to = slotMapping[i + 1];
        
        // Only adds connection if from < to (satisfies canRoute constraint).
        if (from >= 0 && to >= 0 && from < to) {
          tmpl.connections.insert({from, to});
        }
      }
    }
  }
}

// Checks if slot 'from' can reach slot 'to' through existing connections.
static bool canReachViaBypass(const std::set<std::pair<int, int>>& connections, int from, int to, int numSlots) {
  if (from >= to) return false;
  
  std::vector<bool> visited(numSlots, false);
  std::vector<int> queue;
  queue.push_back(from);
  visited[from] = true;
  
  for (size_t h = 0; h < queue.size(); ++h) {
    int cur = queue[h];
    for (const auto& conn : connections) {
      if (conn.first == cur && !visited[conn.second]) {
        if (conn.second == to) return true;
        visited[conn.second] = true;
        queue.push_back(conn.second);
      }
    }
  }
  return false;
}

// Generates optimized slot connections for all templates based on pattern dependencies.
void generateOptimizedConnections(const std::vector<HardwarePattern>& patterns, std::vector<HardwareTemplate>& templates) {
  std::map<int64_t, const HardwarePattern*> patternMap;
  for (const auto& p : patterns) {
    patternMap[p.id] = &p;
  }
  
  for (auto& tmpl : templates) {
    std::set<std::pair<int, int>> requiredConnections;
    
    for (const auto& [pid, slotMapping] : tmpl.mapping) {
      auto patIt = patternMap.find(pid);
      if (patIt == patternMap.end()) continue;
      const HardwarePattern* pat = patIt->second;
      
      if (slotMapping.size() < 2) continue;
      
      if (!pat->opPreds.empty()) {
        for (size_t opIdx = 0; opIdx < pat->opPreds.size() && opIdx < slotMapping.size(); ++opIdx) {
          int toSlot = slotMapping[opIdx];
          
          for (int predOpIdx : pat->opPreds[opIdx]) {
            if (predOpIdx >= 0 && predOpIdx < (int)slotMapping.size()) {
              int fromSlot = slotMapping[predOpIdx];
              if (fromSlot >= 0 && toSlot >= 0 && fromSlot < toSlot) {
                requiredConnections.insert({fromSlot, toSlot});
              }
            }
          }
        }
      } else {
        for (size_t i = 0; i < slotMapping.size() - 1; ++i) {
          int from = slotMapping[i];
          int to = slotMapping[i + 1];
          if (from >= 0 && to >= 0 && from < to) {
            requiredConnections.insert({from, to});
          }
        }
      }
    }
    
    std::vector<std::pair<int, int>> sortedConnections(requiredConnections.begin(), requiredConnections.end());
    std::sort(sortedConnections.begin(), sortedConnections.end(), 
              [](const auto& a, const auto& b) {
                return (a.second - a.first) < (b.second - b.first);
              });
    
    // Build optimized connections - add a connection only if it's not already reachable.
    tmpl.connections.clear();
    int numSlots = tmpl.slots.size();
    
    for (const auto& conn : sortedConnections) {
      // Check if we can already reach conn.second from conn.first via existing connections.
      if (!canReachViaBypass(tmpl.connections, conn.first, conn.second, numSlots)) {
        tmpl.connections.insert(conn);
      }
    }
  }
}

// Generates execution plans for all patterns on their assigned templates.
void generateExecutionPlans(const std::vector<HardwarePattern>& patterns, const std::vector<HardwareTemplate>& templates, std::vector<PatternExecutionPlan>& plans) {
  std::map<int64_t, const HardwareTemplate*> patternToTemplate;
  
  for (const auto& t : templates) {
    for (int64_t pid : t.patterns) {
      patternToTemplate[pid] = &t;
    }
  }
  
  for (const auto& pat : patterns) {
    PatternExecutionPlan plan;
    plan.patternId = pat.id;
    plan.patternName = pat.name;
    
    auto it = patternToTemplate.find(pat.id);
    if (it == patternToTemplate.end()) continue;
    
    const HardwareTemplate* tmpl = it->second;
    auto mappingIt = tmpl->mapping.find(pat.id);
    if (mappingIt == tmpl->mapping.end()) continue;
    
    const std::vector<int>& slotMapping = mappingIt->second;
    
    std::map<int, std::vector<std::pair<int, std::pair<int, std::string>>>> levelToOps;
    
    for (size_t i = 0; i < pat.ops.size() && i < slotMapping.size(); ++i) {
      int level = (i < pat.opLevels.size()) ? pat.opLevels[i] : (int)i;
      int slot = slotMapping[i];
      levelToOps[level].push_back({(int)i, {slot, pat.ops[i]}});
    }
    
    for (const auto& [level, opsAtLevel] : levelToOps) {
      ExecutionStage stage;
      for (const auto& [opIdx, slotAndOp] : opsAtLevel) {
        stage.slots.push_back(slotAndOp.first);
        stage.ops.push_back(slotAndOp.second);
      }
      plan.stages.push_back(stage);
    }
    
    plans.push_back(plan);
  }
}

// Collects supported operations for each template.
void collectSupportedOperations(const std::vector<HardwarePattern>& patterns, const std::vector<HardwareTemplate>& templates, const std::set<std::string>& allDfgOps, std::vector<TemplateSupportedOps>& supportedOps) {
  for (const auto& tmpl : templates) {
    TemplateSupportedOps ops;
    ops.templateId = tmpl.id;
    
    std::set<std::string> templateOps;
    for (const auto& slot : tmpl.slots) {
      for (const auto& op : slot.ops) {
        templateOps.insert(op);
      }
    }
    
    for (const std::string& dfgOp : allDfgOps) {
      bool canSupport = false;
      
      if (templateOps.count(dfgOp)) {
        canSupport = true;
      } else {
        for (const auto& slot : tmpl.slots) {
          if (!slot.ops.empty()) {
            bool compatible = true;
            for (const auto& existingOp : slot.ops) {
              if (!HardwareTemplate::compatible(existingOp, dfgOp)) {
                compatible = false;
                break;
              }
            }
            if (compatible) {
              canSupport = true;
              break;
            }
          } else {
            canSupport = true;
            break;
          }
        }
      }
      
      if (canSupport) {
        ops.singleOps.insert(dfgOp);
      }
    }
    
    ops.compositeOps = tmpl.patterns;
    
    supportedOps.push_back(ops);
  }
}

// Calculates total cost of templates.
double calculateTotalCost(const std::vector<HardwareTemplate>& templates, const OperationCostModel& costModel) {
  double totalCost = 0;
  for (const auto& t : templates) {
    totalCost += t.computeCost(costModel) * t.instances;
  }
  return totalCost;
}

// Escapes string for JSON output.
std::string escapeJsonString(const std::string& s) {
  std::string r;
  for (char c : s) {
    if (c == '"') r += "\\\"";
    else if (c == '\\') r += "\\\\";
    else r += c;
  }
  return r;
}

// Writes hardware configuration to JSON file (extended version with execution plans and supported ops).
void writeHardwareConfigJson(const std::string& path, const std::vector<HardwarePattern>& patterns, const std::vector<HardwareTemplate>& templates, const OperationCostModel& costModel, const std::vector<PatternExecutionPlan>& executionPlans, const std::vector<TemplateSupportedOps>& supportedOps) {
  std::error_code EC;
  llvm::raw_fd_ostream os(path, EC, llvm::sys::fs::OF_Text);
  if (EC) return;
  
  // Build pattern name lookup.
  std::map<int64_t, std::string> patternNames;
  for (const auto& p : patterns) patternNames[p.id] = p.name;
  
  os << "{\n  \"hardware_configuration\": {\n";
  os << "    \"summary\": {\n";
  os << "      \"total_templates\": " << templates.size() << "\n";
  os << "    },\n";
  
  os << "    \"hardware_templates\": [\n";
  for (size_t t = 0; t < templates.size(); ++t) {
    const auto& tmpl = templates[t];
    if (t) os << ",\n";
    
    os << "      {\n";
    os << "        \"template_id\": " << tmpl.id << ",\n";
    os << "        \"instance_count\": " << tmpl.instances << ",\n";
    
    const TemplateSupportedOps* tmplSupportedOps = nullptr;
    for (const auto& sop : supportedOps) {
      if (sop.templateId == tmpl.id) {
        tmplSupportedOps = &sop;
        break;
      }
    }
    
    if (tmplSupportedOps) {
      os << "        \"supported_single_ops\": [";
      bool first = true;
      for (const auto& op : tmplSupportedOps->singleOps) {
        if (!first) os << ", ";
        first = false;
        os << "\"" << op << "\"";
      }
      os << "],\n";
      
      os << "        \"supported_composite_ops\": [\n";
      for (size_t i = 0; i < tmplSupportedOps->compositeOps.size(); ++i) {
        if (i) os << ",\n";
        int64_t pid = tmplSupportedOps->compositeOps[i];
        auto nameIt = patternNames.find(pid);
        std::string pname = (nameIt != patternNames.end()) ? nameIt->second : "";
        os << "          {\"pattern_id\": " << pid << ", \"name\": \"" << escapeJsonString(pname) << "\"}";
      }
      os << "\n        ],\n";
    }
    
    os << "        \"slots\": [\n";
    for (size_t s = 0; s < tmpl.slots.size(); ++s) {
      const auto& slot = tmpl.slots[s];
      if (s) os << ",\n";
      os << "          {\"slot_id\": " << slot.id << ", \"supported_ops\": [";
      bool first = true;
      for (const auto& op : slot.ops) {
        if (!first) os << ", ";
        first = false;
        os << "\"" << op << "\"";
      }
      os << "]}";
    }
    os << "\n        ],\n";
    
    os << "        \"slot_connections\": {\n";
    os << "          \"connections\": [";
    bool firstConn = true;
    for (const auto& conn : tmpl.connections) {
      if (!firstConn) os << ", ";
      firstConn = false;
      os << "{\"from\": " << conn.first << ", \"to\": " << conn.second << "}";
    }
    os << "]\n";
    os << "        },\n";
    
    os << "        \"pattern_execution_plans\": [\n";
    bool firstPlan = true;
    for (const auto& plan : executionPlans) {
      auto mappingIt = tmpl.mapping.find(plan.patternId);
      if (mappingIt == tmpl.mapping.end()) continue;
      
      if (!firstPlan) os << ",\n";
      firstPlan = false;
      
      os << "          {\n";
      os << "            \"pattern_id\": " << plan.patternId << ",\n";
      os << "            \"pattern_name\": \"" << escapeJsonString(plan.patternName) << "\",\n";
      os << "            \"slot_mapping\": [";
      const auto& m = mappingIt->second;
      for (size_t i = 0; i < m.size(); ++i) {
        if (i) os << ", ";
        os << m[i];
      }
      os << "],\n";
      os << "            \"execution_stages\": [\n";
      for (size_t stageIdx = 0; stageIdx < plan.stages.size(); ++stageIdx) {
        const auto& stage = plan.stages[stageIdx];
        if (stageIdx) os << ",\n";
        os << "              {\n";
        os << "                \"stage\": " << stageIdx << ",\n";
        os << "                \"parallel_slots\": [";
        for (size_t i = 0; i < stage.slots.size(); ++i) {
          if (i) os << ", ";
          os << stage.slots[i];
        }
        os << "],\n";
        os << "                \"parallel_ops\": [";
        for (size_t i = 0; i < stage.ops.size(); ++i) {
          if (i) os << ", ";
          os << "\"" << stage.ops[i] << "\"";
        }
        os << "]\n";
        os << "              }";
      }
      os << "\n            ]\n";
      os << "          }";
    }
    os << "\n        ]\n";
    os << "      }";
  }
  os << "\n    ]\n";
  
  os << "  }\n}\n";
}

// Legacy version for backward compatibility.
void writeHardwareConfigJson(const std::string& path, const std::vector<HardwarePattern>& patterns, const std::vector<HardwareTemplate>& templates, const OperationCostModel& costModel) {
  std::vector<PatternExecutionPlan> plans;
  generateExecutionPlans(patterns, templates, plans);
  
  std::set<std::string> allDfgOps;
  for (const auto& p : patterns) {
    for (const auto& op : p.ops) {
      allDfgOps.insert(op);
    }
  }
  
  std::vector<TemplateSupportedOps> supportedOps;
  collectSupportedOperations(patterns, templates, allDfgOps, supportedOps);
  
  writeHardwareConfigJson(path, patterns, templates, costModel, plans, supportedOps);
}

} // namespace mlir::neura

