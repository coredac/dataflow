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

// Cost Model Implementation
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

double OperationCostModel::get(const std::string& op) const {
  auto it = costs.find(op);
  return it != costs.end() ? it->second : 5.0;
}

double OperationCostModel::slotCost(const std::set<std::string>& ops) const {
  double mx = 0;
  for (const auto& op : ops) mx = std::max(mx, get(op));
  return mx;
}

double OperationCostModel::patternCost(const std::vector<std::string>& ops) const {
  double sum = 0;
  for (const auto& op : ops) sum += get(op);
  return sum;
}


// Hardware Pattern Implementation
HardwarePattern::HardwarePattern(int64_t i, const std::string& n, int64_t f) 
    : id(i), name(n), freq(f), cost(0) {}

HardwareSlot::HardwareSlot(int i) : id(i) {}

HardwareTemplate::HardwareTemplate(int i) : id(i), instances(1) {}

void HardwareTemplate::addSlot() {
  int newId = slots.size();
  slots.emplace_back(newId);
}
  
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
  
// Checks if a connection can be made between two slots (simplified: only checks index order).
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
  
// DFS with scoring to find the best mapping for a pattern into the existing template.
// opIdx: current operation index in the pattern; prevSlot: previous slot index
void HardwareTemplate::dfsWithScoring(const std::vector<std::string>& patOps, size_t opIdx, int prevSlot, std::vector<int> cur, std::vector<int>& bestMapping, int& bestScore) const {
  // Finishs processing all the pattern operations.
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
    if (!slotCanHandle(s, op)) continue;  // Skip slots that cannot handle this op
    
    // Non-empty slots with compatible ops: highest priority (reuses hardware, no cost increase)
    if (!slots[s].ops.empty()) {
      candidates.push_back({100, s});
    }
    // Empty slots: lower priority (needs new hardware)
    else {
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
  
std::vector<int> HardwareTemplate::dfs(const std::vector<std::string>& patOps, 
                                       size_t opIdx, int prevSlot,
                                       std::vector<int> cur) const {
  std::vector<int> best;
  int score = -1;
  dfsWithScoring(patOps, opIdx, prevSlot, cur, best, score);
  return best;
}
  
// Checks if a slot can handle an operation.
// A slot can only contain ops from the same compatible group.
// Empty slots can accept any op; non-empty slots can only accept ops compatible with all existing ops.
bool HardwareTemplate::slotCanHandle(size_t s, const std::string& op) const {
  // If slot already contains this exact op, it can handle it.
  if (slots[s].ops.count(op)) return true;
  
  // If slot is empty, it can accept any op.
  if (slots[s].ops.empty()) return true;
  
  // If slot has existing ops, the new op must be compatible with ALL existing ops.
  // This ensures the slot only contains ops from the same compatible group.
  for (const auto& existing : slots[s].ops) {
    if (!compatible(existing, op)) return false;
  }
  return true;
}
  
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
// outMapping: mapping of pattern operations to template slots; outCostIncrease: cost increase for the mapping this pattern.
bool HardwareTemplate::tryAccommodate(const HardwarePattern& pat, const OperationCostModel& cm, std::vector<int>& outMapping, double& outCostIncrease) {
  // Tries to find a mapping wihout extending current templates.
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

  // Checks if we need to insert some slots at the front of the template.
  int origSize = slots.size();
  int needed = pat.ops.size();
  
  // Simplified heuristic: inserts at front if pattern's first op can be handled by slot[0].
  // This encourages more patterns to share early slots.
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
  
  // Fails to find a mapping, so we need to remove new slots.
  while ((int)slots.size() > origSize) {
    slots.pop_back();
  }
  
  return false;
}
  
void HardwareTemplate::applyMapping(const HardwarePattern& pat, const std::vector<int>& m) {
  patterns.push_back(pat.id);
  mapping[pat.id] = m;
  for (size_t i = 0; i < m.size(); ++i) {
    // Verifies the op can be handled by the slot (satisfies compatible group constraint).
    // If not, this indicates the slot state changed since tryAccommodate was called.
    // In this case, we skip adding the op (it may already be in the slot from a previous pattern).
    if (!slotCanHandle(m[i], pat.ops[i])) {
      // The op may already be in the slot, or the slot state changed.
      // If it's already there, we're fine; otherwise, this is a logic error.
      if (!slots[m[i]].ops.count(pat.ops[i])) {
        llvm::errs() << "Warning: Cannot apply op " << pat.ops[i] 
                     << " to slot " << m[i] << " (incompatible with existing ops)\n";
        // Continue anyway - the mapping may still be valid for other ops
      }
      continue;
    }
    slots[m[i]].ops.insert(pat.ops[i]);
  }
}
  
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
  
  for (int i : order) pat.ops.push_back(opNames[i]);
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

// Helper function for DFS with slot preference.
void dfsWithSlotPreferenceHelper(const HardwareTemplate& tmpl, const std::vector<std::string>& ops, size_t opIdx, int prevSlot, std::vector<int> cur, const std::map<std::string, int>& preferred, std::vector<int>& bestMapping, int& bestScore) {
  if (opIdx >= ops.size()) {
    int score = 0;
    for (size_t i = 0; i < ops.size(); ++i) {
      int slot = cur[i];
      const std::string& op = ops[i];
      
      // Slot already has compatible ops (reuses hardware, no cost increase)
      if (tmpl.slotCanHandle(slot, op) && !tmpl.slots[slot].ops.empty()) {
        score += 50;
      }
      // Empty slot (needs new hardware) gets no score
    }
    if (score > bestScore) {
      bestScore = score;
      bestMapping = cur;
    }
    return;
  }
  
  const std::string& op = ops[opIdx];
  
  std::vector<size_t> candidates;
  auto prefIt = preferred.find(op);
  if (prefIt != preferred.end()) {
    int prefSlot = prefIt->second;
    if (prefSlot >= 0 && prefSlot < (int)tmpl.slots.size() && tmpl.slotCanHandle(prefSlot, op) && tmpl.canRoute(prevSlot, prefSlot)) {
      candidates.push_back(prefSlot);
    }
  }
  
  for (size_t s = 0; s < tmpl.slots.size(); ++s) {
    if (tmpl.slots[s].ops.count(op) && 
        std::find(candidates.begin(), candidates.end(), s) == candidates.end() &&
        tmpl.canRoute(prevSlot, (int)s)) {
      candidates.push_back(s);
    }
  }
  
  for (size_t s = 0; s < tmpl.slots.size(); ++s) {
    if (tmpl.slotCanHandle(s, op) &&
        std::find(candidates.begin(), candidates.end(), s) == candidates.end() &&
        tmpl.canRoute(prevSlot, (int)s)) {
      candidates.push_back(s);
    }
  }
  
  for (size_t s : candidates) {
    if (std::find(cur.begin(), cur.end(), (int)s) != cur.end()) continue;
    
    auto next = cur;
    next.push_back((int)s);
    dfsWithSlotPreferenceHelper(tmpl, ops, opIdx + 1, (int)s, next, preferred, 
                               bestMapping, bestScore);
  }
}

// Finds mapping with strong preference for slots that already have same ops.
std::vector<int> findMappingWithSlotPreference(HardwareTemplate& tmpl, const HardwarePattern& pat,
                                               const std::map<std::string, int>& preferred) {
  std::vector<int> bestMapping;
  int bestScore = -1;
  
  dfsWithSlotPreferenceHelper(tmpl, pat.ops, 0, -1, {}, preferred, bestMapping, bestScore);
  return bestMapping;
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
  // Sorts patterns by distinct ops and cost. We prefer to solve the most diverse patterns first.
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
    
    // Reuses existing template if the cost increase is less than half of the pattern cost.
    if (bestT >= 0 && bestInc <= newTmplCost * 0.5) {
      while ((int)templates[bestT].slots.size() < *std::max_element(bestM.begin(), bestM.end()) + 1) {
        templates[bestT].addSlot();
      }
      templates[bestT].applyMapping(pat, bestM);
      
      llvm::errs() << "  P" << pat.id << " → T" << bestT << " slots[";
      for (size_t i = 0; i < bestM.size(); ++i) {
        if (i) llvm::errs() << ",";
        llvm::errs() << bestM[i];
      }
      llvm::errs() << "] +cost=" << bestInc << "\n";
    } else {
      // Creates a new template
      HardwareTemplate t(templates.size());
      for (size_t i = 0; i < pat.ops.size(); ++i) {
        t.addSlot();
      }
      std::vector<int> m;
      for (size_t i = 0; i < pat.ops.size(); ++i) m.push_back(i);
      t.applyMapping(pat, m);
      templates.push_back(t);
      
      llvm::errs() << "  P" << pat.id << " → NEW T" << t.id << "\n";
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
  
  llvm::errs() << "\n[HardwareMerge] " << templates.size() << " templates\n";
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

void IncreaseHardwareComponent(std::map<std::string, int>* hardware_components) {
  for (const auto& [op, count] : *hardware_components) {
    (*hardware_components)[op]++;
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

// Writes hardware configuration to JSON file.
void writeHardwareConfigJson(const std::string& path, const std::vector<HardwarePattern>& patterns, const std::vector<HardwareTemplate>& templates, const OperationCostModel& costModel) {
  std::error_code EC;
  llvm::raw_fd_ostream os(path, EC, llvm::sys::fs::OF_Text);
  if (EC) return;
  
  double costNoShare = 0;
  for (const auto& p : patterns) costNoShare += p.cost;
  double totalCost = calculateTotalCost(templates, costModel);
  
  os << "{\n  \"hardware_configuration\": {\n";
  os << "    \"description\": \"Maximally-merged hardware templates\",\n";
  os << "    \"summary\": {\n";
  os << "      \"total_templates\": " << templates.size() << ",\n";
  os << "      \"total_cost\": " << totalCost << ",\n";
  os << "      \"cost_without_sharing\": " << costNoShare << ",\n";
  os << "      \"cost_reduction_percent\": " << (1 - totalCost/costNoShare)*100 << "\n";
  os << "    },\n";
  
  os << "    \"hardware_templates\": [\n";
  for (size_t t = 0; t < templates.size(); ++t) {
    const auto& tmpl = templates[t];
    if (t) os << ",\n";
    
    os << "      {\n";
    os << "        \"template_id\": " << tmpl.id << ",\n";
    os << "        \"instance_count\": " << tmpl.instances << ",\n";
    os << "        \"cost_per_instance\": " << tmpl.computeCost(costModel) << ",\n";
    os << "        \"total_cost\": " << tmpl.computeCost(costModel) * tmpl.instances << ",\n";
    
    os << "        \"patterns\": [";
    for (size_t i = 0; i < tmpl.patterns.size(); ++i) {
      if (i) os << ", ";
      os << tmpl.patterns[i];
    }
    os << "],\n";
    
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
      os << "], \"cost\": " << costModel.slotCost(slot.ops) << "}";
    }
    os << "\n        ],\n";
    
    os << "        \"connections\": [";
    bool firstConn = true;
    for (const auto& conn : tmpl.connections) {
      if (!firstConn) os << ", ";
      firstConn = false;
      os << "[" << conn.first << ", " << conn.second << "]";
    }
    os << "],\n";
    
    os << "        \"pattern_configs\": [\n";
    bool first = true;
    for (const auto& [pid, m] : tmpl.mapping) {
      if (!first) os << ",\n";
      first = false;
      std::string pname;
      for (const auto& p : patterns) if (p.id == pid) { pname = p.name; break; }
      os << "          {\"pattern_id\": " << pid << ", \"name\": \"" << escapeJsonString(pname) << "\", \"slots\": [";
      for (size_t i = 0; i < m.size(); ++i) {
        if (i) os << ", ";
        os << m[i];
      }
      os << "]}";
    }
    os << "\n        ]\n";
    os << "      }";
  }
  os << "\n    ],\n";
  
  os << "    \"pattern_to_template\": [\n";
  for (size_t i = 0; i < patterns.size(); ++i) {
    const auto& p = patterns[i];
    if (i) os << ",\n";
    int tid = -1;
    std::vector<int> m;
    for (const auto& t : templates) {
      auto it = t.mapping.find(p.id);
      if (it != t.mapping.end()) {
        tid = t.id;
        m = it->second;
        break;
      }
    }
    os << "      {\"pattern_id\": " << p.id << ", \"name\": \"" << escapeJsonString(p.name)
       << "\", \"freq\": " << p.freq << ", \"template\": " << tid << ", \"slots\": [";
    for (size_t j = 0; j < m.size(); ++j) {
      if (j) os << ", ";
      os << m[j];
    }
    os << "], \"ops\": [";
    for (size_t j = 0; j < p.ops.size(); ++j) {
      if (j) os << ", ";
      os << "\"" << p.ops[j] << "\"";
    }
    os << "]}";
  }
  os << "\n    ],\n";
  
  os << "    \"cost_model\": {";
  std::set<std::string> allOps;
  for (const auto& p : patterns) for (const auto& op : p.ops) allOps.insert(op);
  bool first = true;
  for (const auto& op : allOps) {
    if (!first) os << ", ";
    first = false;
    os << "\"" << op << "\": " << costModel.get(op);
  }
  os << "}\n";
  
  os << "  }\n}\n";
  
  llvm::errs() << "[HardwareMerge] Output: " << path << "\n";
}

} // namespace mlir::neura

