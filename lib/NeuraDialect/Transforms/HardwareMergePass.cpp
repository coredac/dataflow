//===- HardwareMergePass.cpp - Aggressive Hardware Template Merging -------===//
//
// Goal: Maximize pattern coverage with minimum hardware cost
//
// Key Features:
// 1. Every slot can be BYPASSED
// 2. Connections can be ADDED between any slots
// 3. Templates can be EXTENDED with new slots
// 4. Process patterns by COST (most expensive first)
// 5. Aggressively merge templates when possible
//
//===----------------------------------------------------------------------===//

#include "NeuraDialect/NeuraOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <algorithm>
#include <cmath>

using namespace mlir;

#define GEN_PASS_DEF_HARDWAREMERGE
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Cost Model
//===----------------------------------------------------------------------===//

class CostModel {
public:
  CostModel() {
    c["neura.div"] = 100; c["neura.fdiv"] = 100; c["neura.rem"] = 80;
    c["neura.load"] = 50; c["neura.store"] = 50;
    c["neura.load_indexed"] = 55; c["neura.store_indexed"] = 55;
    c["neura.mul"] = 30; c["neura.fmul"] = 35;
    c["neura.gep"] = 20;
    c["neura.add"] = 10; c["neura.sub"] = 10;
    c["neura.fadd"] = 12; c["neura.fsub"] = 12;
    c["neura.icmp"] = 15; c["neura.fcmp"] = 15;
    c["neura.and"] = 5; c["neura.or"] = 5; c["neura.not"] = 5;
    c["neura.sel"] = 6; c["neura.phi"] = 3;
    c["neura.grant_predicate"] = 3; c["neura.grant_once"] = 3;
    c["neura.cast"] = 2; c["neura.sext"] = 2; c["neura.zext"] = 2;
    c["neura.data_mov"] = 1; c["neura.constant"] = 1;
  }
  
  double get(const std::string& op) const {
    auto it = c.find(op);
    return it != c.end() ? it->second : 5.0;
  }
  
  double slotCost(const std::set<std::string>& ops) const {
    double mx = 0;
    for (const auto& op : ops) mx = std::max(mx, get(op));
    return mx;
  }
  
  double patternCost(const std::vector<std::string>& ops) const {
    double sum = 0;
    for (const auto& op : ops) sum += get(op);
    return sum;
  }

private:
  std::map<std::string, double> c;
};

//===----------------------------------------------------------------------===//
// Data Structures  
//===----------------------------------------------------------------------===//

struct Pattern {
  int64_t id;
  std::string name;
  int64_t freq;
  std::vector<std::string> ops;  // Linearized operation sequence
  double cost;
  
  Pattern(int64_t i, const std::string& n, int64_t f) 
      : id(i), name(n), freq(f), cost(0) {}
};

struct Slot {
  int id;
  std::set<std::string> ops;
  Slot(int i) : id(i) {}
};

struct Template {
  int id;
  std::vector<Slot> slots;
  std::set<std::pair<int,int>> edges;  // All possible connections
  std::vector<int64_t> patterns;
  std::map<int64_t, std::vector<int>> mapping;  // pattern -> active slots
  int instances;
  
  Template(int i) : id(i), instances(1) {}
  
  void addSlot() {
    int newId = slots.size();
    slots.emplace_back(newId);
    // Connect to all previous slots (full flexibility)
    for (int i = 0; i < newId; ++i) {
      edges.insert({i, newId});
    }
  }
  
  // Insert a slot at the beginning and update all existing mappings
  void insertSlotAtFront() {
    // Renumber all existing slots: slot id increases by 1
    for (auto& slot : slots) {
      slot.id++;
    }
    
    // Renumber all edges: both from and to indices increase by 1
    std::set<std::pair<int,int>> newEdges;
    for (const auto& e : edges) {
      newEdges.insert({e.first + 1, e.second + 1});
    }
    edges = newEdges;
    
    // Insert new slot at position 0
    slots.insert(slots.begin(), Slot(0));
    
    // Connect new slot to all existing slots (now starting from index 1)
    for (size_t i = 1; i < slots.size(); ++i) {
      edges.insert({0, (int)i});
    }
    
    // Update all existing mappings: shift all slot indices by +1
    for (auto& pair : mapping) {
      for (int& slotIdx : pair.second) {
        slotIdx++;
      }
    }
  }
  
  void addEdge(int from, int to) {
    if (from >= 0 && from < (int)slots.size() && 
        to >= 0 && to < (int)slots.size() && from != to) {
      edges.insert({from, to});
    }
  }
  
  // Check if we can route from slot 'from' to slot 'to'
  bool canRoute(int from, int to) const {
    if (from < 0) return true;
    if (edges.count({from, to})) return true;
    
    // BFS for reachability
    std::vector<bool> vis(slots.size(), false);
    std::vector<int> q = {from};
    vis[from] = true;
    for (size_t h = 0; h < q.size(); ++h) {
      int cur = q[h];
      for (const auto& e : edges) {
        if (e.first == cur && !vis[e.second]) {
          if (e.second == to) return true;
          vis[e.second] = true;
          q.push_back(e.second);
        }
      }
    }
    return false;
  }
  
  // Find mapping with preference for slot consistency (same op → same slot)
  // Returns best mapping and its cost (considering operation reuse)
  std::vector<int> findMapping(const std::vector<std::string>& patOps) const {
    // Try all possible mappings, score by slot reuse
    std::vector<int> bestMapping;
    int bestScore = -1;
    
    dfsWithScoring(patOps, 0, -1, {}, bestMapping, bestScore);
    return bestMapping;
  }
  
  void dfsWithScoring(const std::vector<std::string>& patOps,
                      size_t opIdx, int prevSlot, std::vector<int> cur,
                      std::vector<int>& bestMapping, int& bestScore) const {
    if (opIdx >= patOps.size()) {
      // Calculate score: prefer mappings that reuse slots with same ops
      int score = 0;
      for (size_t i = 0; i < patOps.size(); ++i) {
        int slot = cur[i];
        const std::string& op = patOps[i];
        
        // Strong bonus if slot already has this exact operation
        if (slots[slot].ops.count(op)) {
          score += 10;  // Strong preference for reuse
        }
        // Bonus if slot has compatible operations from same family
        else if (slotCanHandle(slot, op) && !slots[slot].ops.empty()) {
          score += 1;  // Weak preference
        }
      }
      if (score > bestScore) {
        bestScore = score;
        bestMapping = cur;
      }
      return;
    }
    
    const std::string& op = patOps[opIdx];
    
    // Collect candidate slots, prioritizing:
    // 1. Slots that already have this exact op (for reuse)
    // 2. Empty slots that can handle this op (prefer earlier slots)
    // 3. Slots with compatible ops
    std::vector<std::pair<int, size_t>> candidates; // priority, slotId
    for (size_t s = 0; s < slots.size(); ++s) {
      if (slots[s].ops.count(op)) {
        candidates.push_back({100, s});  // Highest priority: exact match
      } else if (slots[s].ops.empty() && slotCanHandle(s, op)) {
        candidates.push_back({50 - (int)s, s});  // Prefer earlier empty slots
      } else if (slotCanHandle(s, op)) {
        candidates.push_back({10, s});  // Compatible ops
      }
    }
    
    // Sort by priority (descending)
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
  
  // Legacy DFS for compatibility
  std::vector<int> dfs(const std::vector<std::string>& patOps, 
                       size_t opIdx, int prevSlot,
                       std::vector<int> cur) const {
    std::vector<int> best;
    int score = -1;
    dfsWithScoring(patOps, opIdx, prevSlot, cur, best, score);
    return best;
  }
  
  bool slotCanHandle(size_t s, const std::string& op) const {
    if (slots[s].ops.count(op)) return true;
    // Check compatibility
    for (const auto& existing : slots[s].ops) {
      if (compatible(existing, op)) return true;
    }
    return slots[s].ops.empty();  // Empty slot accepts anything
  }
  
  static bool compatible(const std::string& a, const std::string& b) {
    if (a == b) return true;
    auto fam = [](const std::string& x) {
      if (x.find("add") != std::string::npos || x.find("sub") != std::string::npos) return 1;
      if (x.find("mul") != std::string::npos) return 2;
      if (x.find("div") != std::string::npos || x.find("rem") != std::string::npos) return 3;
      if (x.find("cmp") != std::string::npos) return 4;
      if (x.find("load") != std::string::npos) return 5;
      if (x.find("store") != std::string::npos) return 6;
      if (x.find("phi") != std::string::npos) return 7;
      if (x.find("grant") != std::string::npos) return 8;
      if (x.find("gep") != std::string::npos) return 9;
      if (x.find("cast") != std::string::npos || x.find("ext") != std::string::npos) return 10;
      return 0;
    };
    return fam(a) == fam(b) && fam(a) != 0;
  }
  
  // Try to accommodate pattern, possibly extending template
  bool tryAccommodate(const Pattern& pat, const CostModel& cm, 
                      std::vector<int>& outMapping, double& outCostIncrease) {
    // First try without extension - use scoring to prefer slot reuse
    auto m = findMapping(pat.ops);
    if (!m.empty()) {
      // Calculate cost increase considering operation distribution
      double oldCost = computeCost(cm);
      
      // Track which ops we're adding to which slots
      std::map<std::string, std::set<int>> opToSlots;
      for (size_t i = 0; i < m.size(); ++i) {
        opToSlots[pat.ops[i]].insert(m[i]);
      }
      
      // Simulate adding ops
      for (size_t i = 0; i < m.size(); ++i) {
        slots[m[i]].ops.insert(pat.ops[i]);
      }
      double newCost = computeCost(cm);
      
      // Revert
      for (size_t i = 0; i < m.size(); ++i) {
        slots[m[i]].ops.erase(pat.ops[i]);
      }
      
      outMapping = m;
      outCostIncrease = newCost - oldCost;
      return true;
    }
    
    // Try extending with more slots
    // Strategy: if pattern starts with operations that match the beginning of template,
    // try inserting slots at the front for better alignment. Otherwise, append at the end.
    int origSize = slots.size();
    int needed = pat.ops.size();
    
    // Check if we should insert at front:
    // Pattern starts with an op that matches/needs to be before the first slot's op
    bool shouldInsertAtFront = false;
    if (!pat.ops.empty() && !slots.empty()) {
      const std::string& firstOp = pat.ops[0];
      // Check if first slot can handle the first op
      if (slotCanHandle(0, firstOp)) {
        // Check if we need additional slots of the same type before
        // (e.g., pattern has [phi, phi, ...] and template starts with [phi, ...])
        int sameOpCount = 1;
        for (size_t i = 1; i < pat.ops.size() && pat.ops[i] == firstOp; ++i) {
          sameOpCount++;
        }
        // Count how many consecutive slots at the start can handle this op
        int templateCanHandle = 0;
        for (size_t s = 0; s < slots.size() && s < (size_t)sameOpCount + 2; ++s) {
          if (slotCanHandle(s, firstOp)) {
            templateCanHandle++;
          } else {
            break;
          }
        }
        // If pattern needs more slots of this type than template has at the start
        if (sameOpCount > templateCanHandle) {
          shouldInsertAtFront = true;
        }
      }
    }
    
    // Calculate how many slots we need
    int slotsToAdd = std::max(0, needed + 2 - (int)slots.size());
    
    if (shouldInsertAtFront && slotsToAdd > 0) {
      // Insert slots at the front (this updates existing mappings automatically)
      for (int i = 0; i < slotsToAdd; ++i) {
        insertSlotAtFront();
      }
    } else {
      // Append slots at the end
      while ((int)slots.size() < needed + 2) {
        addSlot();
      }
    }
    
    // Add full connectivity: ensure all slots can reach each other
    // (bidirectional edges for maximum flexibility)
    for (size_t i = 0; i < slots.size(); ++i) {
      for (size_t j = 0; j < slots.size(); ++j) {
        if (i != j) {
          addEdge(i, j);  // Add edge if not already present
        }
      }
    }
    
    m = findMapping(pat.ops);
    if (!m.empty()) {
      double oldCost = computeCost(cm);
      for (size_t i = 0; i < m.size(); ++i) {
        slots[m[i]].ops.insert(pat.ops[i]);
      }
      double newCost = computeCost(cm);
      // Revert ops but keep slots
      for (size_t i = 0; i < m.size(); ++i) {
        slots[m[i]].ops.erase(pat.ops[i]);
      }
      
      outMapping = m;
      outCostIncrease = newCost - oldCost;
      return true;
    }
    
    // Revert slot additions if still can't fit
    while ((int)slots.size() > origSize) {
      int lastId = slots.size() - 1;
      slots.pop_back();
      // Remove edges involving last slot
      for (auto it = edges.begin(); it != edges.end(); ) {
        if (it->first == lastId || it->second == lastId) {
          it = edges.erase(it);
        } else {
          ++it;
        }
      }
    }
    
    return false;
  }
  
  void applyMapping(const Pattern& pat, const std::vector<int>& m) {
    patterns.push_back(pat.id);
    mapping[pat.id] = m;
    for (size_t i = 0; i < m.size(); ++i) {
      slots[m[i]].ops.insert(pat.ops[i]);
    }
    // Add edges between consecutive active slots
    for (size_t i = 0; i + 1 < m.size(); ++i) {
      addEdge(m[i], m[i+1]);
    }
  }
  
  double computeCost(const CostModel& cm) const {
    double sum = 0;
    
    // Count how many slots each operation type appears in
    std::map<std::string, int> opSlotCount;
    for (const auto& slot : slots) {
      for (const auto& op : slot.ops) {
        opSlotCount[op]++;
      }
    }
    
    // Calculate cost: each operation type that appears in N slots costs N times
    // the operation cost (since we need N hardware units)
    std::set<std::string> countedOps;
    for (const auto& slot : slots) {
      for (const auto& op : slot.ops) {
        if (countedOps.count(op)) continue;
        countedOps.insert(op);
        // Cost = operation_cost * number_of_slots_it_appears_in
        sum += cm.get(op) * opSlotCount[op];
      }
    }
    
    return sum;
  }
};

//===----------------------------------------------------------------------===//
// Algorithm
//===----------------------------------------------------------------------===//

class Algorithm {
public:
  void extract(ModuleOp module) {
    module.walk([&](neura::FusedOp fop) {
      int64_t pid = fop.getPatternId();
      if (std::find_if(pats.begin(), pats.end(), 
          [pid](const Pattern& p) { return p.id == pid; }) != pats.end()) return;
      
      Pattern pat(pid, fop.getPatternName().str(), fop.getFrequency());
      extractOps(fop, pat);
      pat.cost = cm.patternCost(pat.ops);
      pats.push_back(pat);
    });
    
    llvm::errs() << "[HardwareMerge] " << pats.size() << " patterns:\n";
    for (auto& p : pats) {
      llvm::errs() << "  P" << p.id << " cost=" << p.cost << ": ";
      for (size_t i = 0; i < p.ops.size(); ++i) {
        if (i) llvm::errs() << "→";
        auto s = p.ops[i];
        if (s.find("neura.") == 0) s = s.substr(6);
        llvm::errs() << s;
      }
      llvm::errs() << "\n";
    }
  }
  
  void createTemplates() {
    llvm::errs() << "\n[HardwareMerge] Creating templates (diversity-first order)...\n";
    
    // Helper function to count distinct operation types in a pattern
    auto countDistinctOps = [](const Pattern& p) -> int {
      std::set<std::string> distinctOps;
      for (const auto& op : p.ops) {
        distinctOps.insert(op);
      }
      return distinctOps.size();
    };
    
    // Sort by distinct operation count (highest first), then by cost as tiebreaker
    std::vector<int> order;
    for (size_t i = 0; i < pats.size(); ++i) order.push_back(i);
    std::sort(order.begin(), order.end(), [this, &countDistinctOps](int a, int b) {
      int distA = countDistinctOps(pats[a]);
      int distB = countDistinctOps(pats[b]);
      if (distA != distB) {
        return distA > distB;  // More distinct ops first
      }
      return pats[a].cost > pats[b].cost;  // Tiebreaker: higher cost first
    });
    
    llvm::errs() << "Processing order (by distinct ops):\n";
    for (int idx : order) {
      llvm::errs() << "  P" << pats[idx].id << " has " << countDistinctOps(pats[idx]) 
                   << " distinct ops, cost=" << pats[idx].cost << "\n";
    }
    llvm::errs() << "\n";
    
    for (int idx : order) {
      Pattern& pat = pats[idx];
      
      // Try to fit into existing template with minimum cost increase
      int bestT = -1;
      std::vector<int> bestM;
      double bestInc = 1e18;
      
      for (size_t t = 0; t < tmpls.size(); ++t) {
        std::vector<int> m;
        double inc;
        if (tmpls[t].tryAccommodate(pat, cm, m, inc)) {
          if (inc < bestInc) {
            bestInc = inc;
            bestT = t;
            bestM = m;
          }
        }
      }
      
      // Compare: merge vs new template
      double newTmplCost = pat.cost;
      
      if (bestT >= 0 && bestInc <= newTmplCost * 0.5) {
        // Merge is beneficial
        // Need to re-extend if needed
        while ((int)tmpls[bestT].slots.size() < *std::max_element(bestM.begin(), bestM.end()) + 1) {
          tmpls[bestT].addSlot();
        }
        tmpls[bestT].applyMapping(pat, bestM);
        
        llvm::errs() << "  P" << pat.id << " → T" << bestT << " slots[";
        for (size_t i = 0; i < bestM.size(); ++i) {
          if (i) llvm::errs() << ",";
          llvm::errs() << bestM[i];
        }
        llvm::errs() << "] +cost=" << bestInc << "\n";
      } else {
        // Create new template
        Template t(tmpls.size());
        for (size_t i = 0; i < pat.ops.size(); ++i) {
          t.addSlot();
        }
        // Full connectivity
        for (size_t i = 0; i < t.slots.size(); ++i) {
          for (size_t j = i + 1; j < t.slots.size(); ++j) {
            t.addEdge(i, j);
          }
        }
        std::vector<int> m;
        for (size_t i = 0; i < pat.ops.size(); ++i) m.push_back(i);
        t.applyMapping(pat, m);
        tmpls.push_back(t);
        
        llvm::errs() << "  P" << pat.id << " → NEW T" << t.id << "\n";
      }
    }
    
    // Try to merge templates with each other
    // mergeTemplates();
    
    // Compute instances
    for (auto& t : tmpls) {
      int64_t totalFreq = 0;
      for (int64_t pid : t.patterns) {
        for (auto& p : pats) {
          if (p.id == pid) { totalFreq += p.freq; break; }
        }
      }
      t.instances = std::max(1, (int)std::ceil(totalFreq / 10.0));
    }
    
    // Calculate total cost
    totalCost = 0;
    for (auto& t : tmpls) {
      totalCost += t.computeCost(cm) * t.instances;
    }
    
    llvm::errs() << "\n[HardwareMerge] " << tmpls.size() << " templates, cost=" << totalCost << "\n";
  }
  
  void mergeTemplates() {
    // Try to merge smaller templates into larger ones
    bool changed = true;
    while (changed) {
      changed = false;
      
      for (size_t i = 0; i < tmpls.size() && !changed; ++i) {
        for (size_t j = i + 1; j < tmpls.size() && !changed; ++j) {
          // Try merging j into i
          if (canMergeTemplates(i, j)) {
            doMergeTemplates(i, j);
            tmpls.erase(tmpls.begin() + j);
            // Renumber
            for (size_t k = j; k < tmpls.size(); ++k) {
              tmpls[k].id = k;
            }
            changed = true;
          }
        }
      }
    }
  }
  
  bool canMergeTemplates(size_t i, size_t j) {
    Template& ti = tmpls[i];
    Template& tj = tmpls[j];
    
    // Prerequisite: must have at least one common operation type
    std::set<std::string> opsI;
    for (const auto& slot : ti.slots) {
      for (const auto& op : slot.ops) {
        opsI.insert(op);
      }
    }
    
    std::set<std::string> opsJ;
    for (const auto& slot : tj.slots) {
      for (const auto& op : slot.ops) {
        opsJ.insert(op);
      }
    }
    
    // Check for common operations
    bool hasCommonOp = false;
    for (const auto& op : opsI) {
      if (opsJ.count(op)) {
        hasCommonOp = true;
        break;
      }
      // Also check compatibility
      for (const auto& opJ : opsJ) {
        if (Template::compatible(op, opJ)) {
          hasCommonOp = true;
          break;
        }
      }
      if (hasCommonOp) break;
    }
    
    if (!hasCommonOp) {
      return false;  // No common operation, cannot merge
    }
    
    // Check if all patterns in template j can fit into template i
    for (int64_t pid : tj.patterns) {
      Pattern* pat = nullptr;
      for (auto& p : pats) {
        if (p.id == pid) { pat = &p; break; }
      }
      if (!pat) continue;
      
      std::vector<int> m;
      double inc;
      if (!ti.tryAccommodate(*pat, cm, m, inc)) {
        return false;
      }
    }
    return true;
  }
  
  void doMergeTemplates(size_t i, size_t j) {
    Template& ti = tmpls[i];
    Template& tj = tmpls[j];
    
    llvm::errs() << "  Merging T" << j << " into T" << i << "\n";
    
    // Build op-to-slot mapping for template i (preferred slots for each op)
    std::map<std::string, int> opToPreferredSlot;
    for (size_t s = 0; s < ti.slots.size(); ++s) {
      for (const auto& op : ti.slots[s].ops) {
        // Prefer slot that already has this exact operation
        if (opToPreferredSlot.count(op) == 0) {
          opToPreferredSlot[op] = s;
        }
      }
    }
    
    // Extend template if needed
    int maxNeededSlots = (int)ti.slots.size();
    for (int64_t pid : tj.patterns) {
      Pattern* pat = nullptr;
      for (auto& p : pats) {
        if (p.id == pid) { pat = &p; break; }
      }
      if (!pat) continue;
      maxNeededSlots = std::max(maxNeededSlots, (int)pat->ops.size() + 2);
    }
    while ((int)ti.slots.size() < maxNeededSlots) {
      ti.addSlot();
    }
    // Add full connectivity
    for (size_t a = 0; a < ti.slots.size(); ++a) {
      for (size_t b = a + 1; b < ti.slots.size(); ++b) {
        ti.addEdge(a, b);
      }
    }
    
    // Now merge patterns with preference for slot reuse
    for (int64_t pid : tj.patterns) {
      Pattern* pat = nullptr;
      for (auto& p : pats) {
        if (p.id == pid) { pat = &p; break; }
      }
      if (!pat) continue;
      
      // Find mapping with preference for slots that already have the same ops
      std::vector<int> m = findMappingWithSlotPreference(ti, *pat, opToPreferredSlot);
      
      if (m.empty()) {
        // Fallback: use tryAccommodate
        double inc;
        ti.tryAccommodate(*pat, cm, m, inc);
      }
      
      // Update preferred slots after mapping
      for (size_t idx = 0; idx < pat->ops.size() && idx < m.size(); ++idx) {
        const std::string& op = pat->ops[idx];
        int slot = m[idx];
        if (opToPreferredSlot.count(op) == 0) {
          opToPreferredSlot[op] = slot;
        } else if (ti.slots[slot].ops.count(op)) {
          // This slot already has the op, prefer it
          opToPreferredSlot[op] = slot;
        }
      }
      
      ti.applyMapping(*pat, m);
    }
  }
  
  // Find mapping with strong preference for slots that already have same ops
  std::vector<int> findMappingWithSlotPreference(Template& tmpl, 
                                                  const Pattern& pat,
                                                  const std::map<std::string, int>& preferred) {
    std::vector<int> bestMapping;
    int bestScore = -1;
    
    dfsWithSlotPreference(tmpl, pat.ops, 0, -1, {}, preferred, bestMapping, bestScore);
    return bestMapping;
  }
  
  void dfsWithSlotPreference(const Template& tmpl,
                              const std::vector<std::string>& ops,
                              size_t opIdx, int prevSlot, std::vector<int> cur,
                              const std::map<std::string, int>& preferred,
                              std::vector<int>& bestMapping, int& bestScore) const {
    if (opIdx >= ops.size()) {
      // Calculate score: strongly prefer slots that match preferred slots
      int score = 0;
      for (size_t i = 0; i < ops.size(); ++i) {
        int slot = cur[i];
        const std::string& op = ops[i];
        
        // Strong bonus if slot matches preferred slot
        auto it = preferred.find(op);
        if (it != preferred.end() && it->second == slot) {
          score += 100;
        }
        // Bonus if slot already has this exact operation
        if (tmpl.slots[slot].ops.count(op)) {
          score += 50;
        }
        // Bonus if slot has compatible operations
        else if (tmpl.slotCanHandle(slot, op) && !tmpl.slots[slot].ops.empty()) {
          score += 10;
        }
      }
      if (score > bestScore) {
        bestScore = score;
        bestMapping = cur;
      }
      return;
    }
    
    const std::string& op = ops[opIdx];
    
    // First try preferred slot if exists
    std::vector<size_t> candidates;
    auto prefIt = preferred.find(op);
    if (prefIt != preferred.end()) {
      int prefSlot = prefIt->second;
      if (prefSlot >= 0 && prefSlot < (int)tmpl.slots.size() && 
          tmpl.slotCanHandle(prefSlot, op) &&
          tmpl.canRoute(prevSlot, prefSlot)) {
        candidates.push_back(prefSlot);
      }
    }
    
    // Then try slots that already have this op
    for (size_t s = 0; s < tmpl.slots.size(); ++s) {
      if (tmpl.slots[s].ops.count(op) && 
          std::find(candidates.begin(), candidates.end(), s) == candidates.end() &&
          tmpl.canRoute(prevSlot, (int)s)) {
        candidates.push_back(s);
      }
    }
    
    // Finally try other compatible slots
    for (size_t s = 0; s < tmpl.slots.size(); ++s) {
      if (tmpl.slotCanHandle(s, op) &&
          std::find(candidates.begin(), candidates.end(), s) == candidates.end() &&
          tmpl.canRoute(prevSlot, (int)s)) {
        candidates.push_back(s);
      }
    }
    
    // Try each candidate
    for (size_t s : candidates) {
      if (std::find(cur.begin(), cur.end(), (int)s) != cur.end()) continue;
      
      auto next = cur;
      next.push_back((int)s);
      dfsWithSlotPreference(tmpl, ops, opIdx + 1, (int)s, next, preferred, 
                           bestMapping, bestScore);
    }
  }
  
  void writeJson(const std::string& path) {
    std::error_code EC;
    llvm::raw_fd_ostream os(path, EC, llvm::sys::fs::OF_Text);
    if (EC) return;
    
    double costNoShare = 0;
    for (auto& p : pats) costNoShare += p.cost;
    
    os << "{\n  \"hardware_configuration\": {\n";
    os << "    \"description\": \"Maximally-merged hardware templates\",\n";
    os << "    \"summary\": {\n";
    os << "      \"total_templates\": " << tmpls.size() << ",\n";
    os << "      \"total_cost\": " << totalCost << ",\n";
    os << "      \"cost_without_sharing\": " << costNoShare << ",\n";
    os << "      \"cost_reduction_percent\": " << (1 - totalCost/costNoShare)*100 << "\n";
    os << "    },\n";
    
    os << "    \"hardware_templates\": [\n";
    for (size_t t = 0; t < tmpls.size(); ++t) {
      auto& tmpl = tmpls[t];
      if (t) os << ",\n";
      
      os << "      {\n";
      os << "        \"template_id\": " << tmpl.id << ",\n";
      os << "        \"instance_count\": " << tmpl.instances << ",\n";
      os << "        \"cost_per_instance\": " << tmpl.computeCost(cm) << ",\n";
      os << "        \"total_cost\": " << tmpl.computeCost(cm) * tmpl.instances << ",\n";
      
      os << "        \"patterns\": [";
      for (size_t i = 0; i < tmpl.patterns.size(); ++i) {
        if (i) os << ", ";
        os << tmpl.patterns[i];
      }
      os << "],\n";
      
      os << "        \"slots\": [\n";
      for (size_t s = 0; s < tmpl.slots.size(); ++s) {
        auto& slot = tmpl.slots[s];
        if (s) os << ",\n";
        os << "          {\"slot_id\": " << slot.id << ", \"supported_ops\": [";
        bool first = true;
        for (auto& op : slot.ops) {
          if (!first) os << ", ";
          first = false;
          os << "\"" << op << "\"";
        }
        os << "], \"cost\": " << cm.slotCost(slot.ops) << "}";
      }
      os << "\n        ],\n";
      
      // Connection matrix
      int n = tmpl.slots.size();
      os << "        \"connections\": [\n";
      for (int i = 0; i < n; ++i) {
        os << "          [";
        for (int j = 0; j < n; ++j) {
          if (j) os << ", ";
          os << (tmpl.edges.count({i,j}) ? 1 : 0);
        }
        os << "]" << (i < n-1 ? "," : "") << "\n";
      }
      os << "        ],\n";
      
      os << "        \"pattern_configs\": [\n";
      bool first = true;
      for (auto& [pid, m] : tmpl.mapping) {
        if (!first) os << ",\n";
        first = false;
        std::string pname;
        for (auto& p : pats) if (p.id == pid) { pname = p.name; break; }
        os << "          {\"pattern_id\": " << pid << ", \"name\": \"" << esc(pname) << "\", \"slots\": [";
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
    for (size_t i = 0; i < pats.size(); ++i) {
      auto& p = pats[i];
      if (i) os << ",\n";
      int tid = -1;
      std::vector<int> m;
      for (auto& t : tmpls) {
        if (t.mapping.count(p.id)) {
          tid = t.id;
          m = t.mapping[p.id];
          break;
        }
      }
      os << "      {\"pattern_id\": " << p.id << ", \"name\": \"" << esc(p.name) 
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
    for (auto& p : pats) for (auto& op : p.ops) allOps.insert(op);
    bool first = true;
    for (auto& op : allOps) {
      if (!first) os << ", ";
      first = false;
      os << "\"" << op << "\": " << cm.get(op);
    }
    os << "}\n";
    
    os << "  }\n}\n";
    
    llvm::errs() << "[HardwareMerge] Output: " << path << "\n";
  }
  
  size_t numTemplates() const { return tmpls.size(); }
  double cost() const { return totalCost; }

private:
  std::vector<Pattern> pats;
  std::vector<Template> tmpls;
  CostModel cm;
  double totalCost = 0;
  
  void extractOps(neura::FusedOp fop, Pattern& pat) {
    Region& body = fop.getBody();
    if (body.empty()) return;
    Block& blk = body.front();
    
    // Build DAG
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
    
    // Topo sort
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
    
    // Sort by level
    std::vector<int> order;
    for (int i = 0; i < n; ++i) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return level[a] < level[b];
    });
    
    for (int i : order) pat.ops.push_back(opNames[i]);
  }
  
  std::string esc(const std::string& s) {
    std::string r;
    for (char c : s) {
      if (c == '"') r += "\\\"";
      else if (c == '\\') r += "\\\\";
      else r += c;
    }
    return r;
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct HardwareMergePass
    : public PassWrapper<HardwareMergePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HardwareMergePass)
  
  HardwareMergePass() = default;
  HardwareMergePass(const HardwareMergePass &pass)
      : PassWrapper<HardwareMergePass, OperationPass<ModuleOp>>(pass) {}
  
  StringRef getArgument() const override { return "hardware-merge"; }
  StringRef getDescription() const override {
    return "Maximally merge hardware templates with bypass support.";
  }
  
  Option<std::string> outputFile{*this, "output", 
    llvm::cl::desc("Output JSON"), llvm::cl::init("hardware_config.json")};
  
  void runOnOperation() override {
    llvm::errs() << "\n========================================\n";
    llvm::errs() << "HardwareMergePass: Aggressive Merging\n";
    llvm::errs() << "========================================\n";
    
    Algorithm algo;
    algo.extract(getOperation());
    algo.createTemplates();
    algo.writeJson(outputFile.getValue());
    
    llvm::errs() << "\nResult: " << algo.numTemplates() << " templates, "
                 << "cost: " << algo.cost() << "\n";
    llvm::errs() << "========================================\n\n";
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createHardwareMergePass() {
  return std::make_unique<HardwareMergePass>();
}
} // namespace mlir::neura
