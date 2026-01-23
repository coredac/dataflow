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
double OperationCostModel::slot_cost(const std::set<std::string>& ops) const {
  double mx = 0;
  for (const auto& op : ops) mx = std::max(mx, get(op));
  return mx;
}

// Returns the total cost for a pattern by summing costs of all its operations.
double OperationCostModel::pattern_cost(const std::vector<std::string>& ops) const {
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
void HardwareTemplate::add_slot() {
  int new_id = slots.size();
  slots.emplace_back(new_id);
}
  
// Inserts a new slot at the front of the template, shifting all existing slot IDs.
void HardwareTemplate::insert_slot_at_front() {
  for (auto& slot : slots) {
    slot.id++;
  }
  
  slots.insert(slots.begin(), HardwareSlot(0));
  
  for (auto& pair : mapping) {
    for (int& slot_idx : pair.second) {
      slot_idx++;
    }
  }
}
  
// Checks if a connection can be made between two slots.
bool HardwareTemplate::can_route(int from, int to) const {
  if (from < 0) return true;
  return from < to;
}

// Finds a mapping for a pattern into the existing template.
std::vector<int> HardwareTemplate::find_mapping(const std::vector<std::string>& pat_ops) const {
  std::vector<int> best_mapping;
  int best_score = -1;
  
  dfs_with_scoring(pat_ops, 0, -1, {}, best_mapping, best_score);
  return best_mapping;
}
  
// Performs DFS with scoring to find the best mapping for a pattern.
void HardwareTemplate::dfs_with_scoring(const std::vector<std::string>& pat_ops, size_t op_idx, int prev_slot, std::vector<int> cur, std::vector<int>& best_mapping, int& best_score) const {
  if (op_idx >= pat_ops.size()) {
    int score = 0;
    for (size_t i = 0; i < pat_ops.size(); ++i) {
      int slot = cur[i];
      const std::string& op = pat_ops[i];
      
      if (slot_can_handle(slot, op) && !slots[slot].ops.empty()) {
        score += 10;
      }
    }
    if (score > best_score) {
      best_score = score;
      best_mapping = cur;
    }
    return;
  }
  
  const std::string& op = pat_ops[op_idx];
  
  std::vector<std::pair<int, size_t>> candidates;
  for (size_t s = 0; s < slots.size(); ++s) {
    if (!slot_can_handle(s, op)) continue;
    
    if (!slots[s].ops.empty()) {
      candidates.push_back({100, s});
    } else {
      candidates.push_back({50, s});
    }
  }
  
  std::sort(candidates.rbegin(), candidates.rend());
  
  for (const auto& cand : candidates) {
    size_t s = cand.second;
    if (!can_route(prev_slot, (int)s)) continue;
    if (std::find(cur.begin(), cur.end(), (int)s) != cur.end()) continue;
    
    auto next = cur;
    next.push_back((int)s);
    dfs_with_scoring(pat_ops, op_idx + 1, (int)s, next, best_mapping, best_score);
  }
}
  
// Performs DFS search to find a mapping, wrapping dfs_with_scoring.
std::vector<int> HardwareTemplate::dfs(const std::vector<std::string>& pat_ops, size_t op_idx, int prev_slot, std::vector<int> cur) const {
  std::vector<int> best;
  int score = -1;
  dfs_with_scoring(pat_ops, op_idx, prev_slot, cur, best, score);
  return best;
}
  
// Checks if a slot can handle an operation.
bool HardwareTemplate::slot_can_handle(size_t s, const std::string& op) const {
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
  
  auto in_group = [](const std::string& op, const std::vector<std::string>& group) -> bool {
    for (const auto& keyword : group) {
      if (op.find(keyword) != std::string::npos) return true;
    }
    return false;
  };
  
  static const std::vector<std::vector<std::string>> compatible_groups = {
    {"add", "sub"},
    {"grant_once", "grant_predicate", "grant_always"},
    {"and", "or", "not", "xor"},
    {"load", "store"}
  };
  
  for (const auto& group : compatible_groups) {
    if (in_group(a, group) && in_group(b, group)) {
      return true;
    }
  }
  
  return false;
}

// Tries to accommodate a pattern into the existing template.
bool HardwareTemplate::try_accommodate(const HardwarePattern& pat, const OperationCostModel& cm, std::vector<int>& out_mapping, double& out_cost_increase) {
  auto m = find_mapping(pat.ops);
  if (!m.empty()) {
    double old_cost = compute_cost(cm);
    
    for (size_t i = 0; i < m.size(); ++i) {
      slots[m[i]].ops.insert(pat.ops[i]);
    }
    double new_cost = compute_cost(cm);
    
    for (size_t i = 0; i < m.size(); ++i) {
      slots[m[i]].ops.erase(pat.ops[i]);
    }
    
    out_mapping = m;
    out_cost_increase = new_cost - old_cost;
    return true;
  }

  int orig_size = slots.size();
  int needed = pat.ops.size();
  
  bool should_insert_at_front = false;
  if (!pat.ops.empty() && !slots.empty() && slot_can_handle(0, pat.ops[0])) {
    should_insert_at_front = true;
  }
  
  int slots_to_add = std::max(0, needed + 2 - (int)slots.size());
  
  if (should_insert_at_front && slots_to_add > 0) {
    for (int i = 0; i < slots_to_add; ++i) {
      insert_slot_at_front();
    }
  } else { 
    while ((int)slots.size() < needed + 2) {
      add_slot();
    }
  }
  
  m = find_mapping(pat.ops);
  if (!m.empty()) {
    double old_cost = compute_cost(cm);
    for (size_t i = 0; i < m.size(); ++i) {
      slots[m[i]].ops.insert(pat.ops[i]);
    }
    double new_cost = compute_cost(cm);
    for (size_t i = 0; i < m.size(); ++i) {
      slots[m[i]].ops.erase(pat.ops[i]);
    }
    
    out_mapping = m;
    out_cost_increase = new_cost - old_cost;
    return true;
  }
  
  while ((int)slots.size() > orig_size) {
    slots.pop_back();
  }
  
  return false;
}
  
// Applies a mapping to the template, adding the pattern's operations to slots.
void HardwareTemplate::apply_mapping(const HardwarePattern& pat, const std::vector<int>& m) {
  patterns.push_back(pat.id);
  mapping[pat.id] = m;
  for (size_t i = 0; i < m.size(); ++i) {
    if (!slot_can_handle(m[i], pat.ops[i])) {
      continue;
    }
    slots[m[i]].ops.insert(pat.ops[i]);
  }
}
  
// Computes the total cost of the template based on operations in each slot.
double HardwareTemplate::compute_cost(const OperationCostModel& cm) const {
  double sum = 0;
  
  std::map<std::string, int> op_slot_count;
  for (const auto& slot : slots) {
    for (const auto& op : slot.ops) {
      op_slot_count[op]++;
    }
  }
  
  std::set<std::string> counted_ops;
  for (const auto& slot : slots) {
    for (const auto& op : slot.ops) {
      if (counted_ops.count(op)) continue;
      counted_ops.insert(op);
      sum += cm.get(op) * op_slot_count[op];
    }
  }
  
  return sum;
}

// Extracts operations from fused op and linearizes them via topological sort.
void extract_pattern_ops(neura::FusedOp fop, HardwarePattern& pat) {
  Region& body = fop.getBody();
  if (body.empty()) return;
  Block& blk = body.front();
  
  llvm::DenseMap<Operation*, int> op_id;
  std::vector<std::string> op_names;
  std::vector<std::vector<int>> preds;
  
  int id = 0;
  for (Operation& op : blk.getOperations()) {
    std::string name = op.getName().getStringRef().str();
    if (name == "neura.yield") continue;
    op_id[&op] = id++;
    op_names.push_back(name);
    preds.push_back({});
  }
  
  int idx = 0;
  for (Operation& op : blk.getOperations()) {
    std::string name = op.getName().getStringRef().str();
    if (name == "neura.yield") continue;
    for (Value v : op.getOperands()) {
      if (Operation* def = v.getDefiningOp()) {
        if (op_id.count(def)) {
          preds[idx].push_back(op_id[def]);
        }
      }
    }
    idx++;
  }
  
  int n = op_names.size();
  std::vector<int> level(n, 0);
  std::vector<int> in_deg(n, 0);
  for (int i = 0; i < n; ++i) in_deg[i] = preds[i].size();
  
  std::vector<int> q;
  for (int i = 0; i < n; ++i) if (in_deg[i] == 0) q.push_back(i);
  
  for (size_t h = 0; h < q.size(); ++h) {
    int cur = q[h];
    for (int i = 0; i < n; ++i) {
      for (int p : preds[i]) {
        if (p == cur) {
          level[i] = std::max(level[i], level[cur] + 1);
          if (--in_deg[i] == 0) q.push_back(i);
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
  std::vector<int> old_to_new(n);
  for (int new_idx = 0; new_idx < n; ++new_idx) {
    old_to_new[order[new_idx]] = new_idx;
  }
  
  for (int i : order) {
    pat.ops.push_back(op_names[i]);
    pat.op_levels.push_back(level[i]);
    
    std::vector<int> remapped_preds;
    for (int p : preds[i]) {
      remapped_preds.push_back(old_to_new[p]);
    }
    pat.op_preds.push_back(remapped_preds);
  }
}

// Extracts all patterns from module.
void extract_patterns(ModuleOp module, std::vector<HardwarePattern>& patterns, OperationCostModel& cost_model) {
  module.walk([&](neura::FusedOp fop) {
    int64_t pid = fop.getPatternId();
    if (std::find_if(patterns.begin(), patterns.end(), 
        [pid](const HardwarePattern& p) { return p.id == pid; }) != patterns.end()) return;
    
    HardwarePattern pat(pid, fop.getPatternName().str(), fop.getFrequency());
    extract_pattern_ops(fop, pat);
    pat.cost = cost_model.pattern_cost(pat.ops);
    patterns.push_back(pat);
  });
}

// Extracts all standalone operations from module.
void extract_all_standalone_ops(ModuleOp module, std::set<std::string>& all_ops) {
  module.walk([&](Operation* op) {
    if (isa<neura::FusedOp>(op)) return;
    
    Operation* parent = op->getParentOp();
    while (parent) {
      if (isa<neura::FusedOp>(parent)) return;
      parent = parent->getParentOp();
    }
    
    std::string op_name = op->getName().getStringRef().str();
    if (op_name.find("neura.") == 0) {
      if (op_name == "neura.yield" || op_name == "neura.fused_op") return;
      all_ops.insert(op_name);
    }
  });
}

// Creates hardware templates from patterns.
void create_hardware_templates(const std::vector<HardwarePattern>& patterns, std::vector<HardwareTemplate>& templates, OperationCostModel& cost_model) {
  auto count_distinct_ops = [](const HardwarePattern& p) -> int {
    std::set<std::string> distinct_ops;
    for (const auto& op : p.ops) {
      distinct_ops.insert(op);
    }
    return distinct_ops.size();
  };
  
  std::vector<int> order;
  for (size_t i = 0; i < patterns.size(); ++i) order.push_back(i);
  std::sort(order.begin(), order.end(), [&patterns, &count_distinct_ops](int a, int b) {
    int dist_a = count_distinct_ops(patterns[a]);
    int dist_b = count_distinct_ops(patterns[b]);
    if (dist_a != dist_b) {
      return dist_a > dist_b;
    }
    return patterns[a].cost > patterns[b].cost;
  });
  
  for (int idx : order) {
    const HardwarePattern& pat = patterns[idx];
    
    int best_t = -1;
    std::vector<int> best_m;
    double best_inc = 1e18;
    
    for (size_t t = 0; t < templates.size(); ++t) {
      HardwareTemplate temp_tmpl = templates[t];
      std::vector<int> m;
      double inc;
      if (temp_tmpl.try_accommodate(pat, cost_model, m, inc)) {
        if (inc < best_inc) {
          best_inc = inc;
          best_t = t;
          best_m = m;
        }
      }
    }
    
    double new_tmpl_cost = pat.cost;
    
    if (best_t >= 0 && best_inc <= new_tmpl_cost * 0.5) {
      while ((int)templates[best_t].slots.size() < *std::max_element(best_m.begin(), best_m.end()) + 1) {
        templates[best_t].add_slot();
      }
      templates[best_t].apply_mapping(pat, best_m);
    } else {
      HardwareTemplate t(templates.size());
      for (size_t i = 0; i < pat.ops.size(); ++i) {
        t.add_slot();
      }
      std::vector<int> m;
      for (size_t i = 0; i < pat.ops.size(); ++i) m.push_back(i);
      t.apply_mapping(pat, m);
      templates.push_back(t);
    }
  }
  
  for (auto& t : templates) {
    int64_t total_freq = 0;
    for (int64_t pid : t.patterns) {
      for (const auto& p : patterns) {
        if (p.id == pid) { total_freq += p.freq; break; }
      }
    }
    t.instances = std::max(1, (int)std::ceil(total_freq / 10.0));
  }
}

// Generates slot connections for all templates based on pattern mappings.
// For each pattern, creates connections between consecutive slots in its mapping.
// Handles bypass cases where slots are skipped (e.g., mapping [0, 2] creates connection 0->2).
void generate_connections(const std::vector<HardwarePattern>& patterns, std::vector<HardwareTemplate>& templates) {
  for (auto& tmpl : templates) {
    tmpl.connections.clear();
    
    // For each pattern mapped to this template, generate connections based on its slot mapping.
    for (const auto& [pid, slot_mapping] : tmpl.mapping) {
      if (slot_mapping.size() < 2) continue;  // Need at least 2 slots to form a connection
      
      // Creates connections between consecutive slots in the pattern's execution order.
      // For example, if mapping is [0, 1, 2], creates connections 0->1 and 1->2.
      // If mapping is [0, 2] (bypassing slot 1), creates connection 0->2.
      for (size_t i = 0; i < slot_mapping.size() - 1; ++i) {
        int from = slot_mapping[i];
        int to = slot_mapping[i + 1];
        
        // Only adds connection if from < to (satisfies can_route constraint).
        if (from >= 0 && to >= 0 && from < to) {
          tmpl.connections.insert({from, to});
        }
      }
    }
  }
}

// Checks if slot 'from' can reach slot 'to' through existing connections.
static bool can_reach_via_bypass(const std::set<std::pair<int, int>>& connections, int from, int to, int num_slots) {
  if (from >= to) return false;
  
  std::vector<bool> visited(num_slots, false);
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
void generate_optimized_connections(const std::vector<HardwarePattern>& patterns, std::vector<HardwareTemplate>& templates) {
  std::map<int64_t, const HardwarePattern*> pattern_map;
  for (const auto& p : patterns) {
    pattern_map[p.id] = &p;
  }
  
  for (auto& tmpl : templates) {
    std::set<std::pair<int, int>> required_connections;
    
    for (const auto& [pid, slot_mapping] : tmpl.mapping) {
      auto pat_it = pattern_map.find(pid);
      if (pat_it == pattern_map.end()) continue;
      const HardwarePattern* pat = pat_it->second;
      
      if (slot_mapping.size() < 2) continue;
      
      if (!pat->op_preds.empty()) {
        for (size_t op_idx = 0; op_idx < pat->op_preds.size() && op_idx < slot_mapping.size(); ++op_idx) {
          int to_slot = slot_mapping[op_idx];
          
          for (int pred_op_idx : pat->op_preds[op_idx]) {
            if (pred_op_idx >= 0 && pred_op_idx < (int)slot_mapping.size()) {
              int from_slot = slot_mapping[pred_op_idx];
              if (from_slot >= 0 && to_slot >= 0 && from_slot < to_slot) {
                required_connections.insert({from_slot, to_slot});
              }
            }
          }
        }
      } else {
        for (size_t i = 0; i < slot_mapping.size() - 1; ++i) {
          int from = slot_mapping[i];
          int to = slot_mapping[i + 1];
          if (from >= 0 && to >= 0 && from < to) {
            required_connections.insert({from, to});
          }
        }
      }
    }
    
    std::vector<std::pair<int, int>> sorted_connections(required_connections.begin(), required_connections.end());
    std::sort(sorted_connections.begin(), sorted_connections.end(), 
              [](const auto& a, const auto& b) {
                return (a.second - a.first) < (b.second - b.first);
              });
    
    // Build optimized connections - add a connection only if it's not already reachable.
    tmpl.connections.clear();
    int num_slots = tmpl.slots.size();
    
    for (const auto& conn : sorted_connections) {
      // Check if we can already reach conn.second from conn.first via existing connections.
      if (!can_reach_via_bypass(tmpl.connections, conn.first, conn.second, num_slots)) {
        tmpl.connections.insert(conn);
      }
    }
  }
}

// Generates execution plans for all patterns on their assigned templates.
void generate_execution_plans(const std::vector<HardwarePattern>& patterns, const std::vector<HardwareTemplate>& templates, std::vector<PatternExecutionPlan>& plans) {
  std::map<int64_t, const HardwareTemplate*> pattern_to_template;
  
  for (const auto& t : templates) {
    for (int64_t pid : t.patterns) {
      pattern_to_template[pid] = &t;
    }
  }
  
  for (const auto& pat : patterns) {
    PatternExecutionPlan plan;
    plan.pattern_id = pat.id;
    plan.pattern_name = pat.name;
    
    auto it = pattern_to_template.find(pat.id);
    if (it == pattern_to_template.end()) continue;
    
    const HardwareTemplate* tmpl = it->second;
    auto mapping_it = tmpl->mapping.find(pat.id);
    if (mapping_it == tmpl->mapping.end()) continue;
    
    const std::vector<int>& slot_mapping = mapping_it->second;
    
    std::map<int, std::vector<std::pair<int, std::pair<int, std::string>>>> level_to_ops;
    
    for (size_t i = 0; i < pat.ops.size() && i < slot_mapping.size(); ++i) {
      int level = (i < pat.op_levels.size()) ? pat.op_levels[i] : (int)i;
      int slot = slot_mapping[i];
      level_to_ops[level].push_back({(int)i, {slot, pat.ops[i]}});
    }
    
    for (const auto& [level, ops_at_level] : level_to_ops) {
      ExecutionStage stage;
      for (const auto& [op_idx, slot_and_op] : ops_at_level) {
        stage.slots.push_back(slot_and_op.first);
        stage.ops.push_back(slot_and_op.second);
      }
      plan.stages.push_back(stage);
    }
    
    plans.push_back(plan);
  }
}

// Collects supported operations for each template.
void collect_supported_operations(const std::vector<HardwarePattern>& patterns, const std::vector<HardwareTemplate>& templates, const std::set<std::string>& all_dfg_ops, std::vector<TemplateSupportedOps>& supported_ops) {
  for (const auto& tmpl : templates) {
    TemplateSupportedOps ops;
    ops.template_id = tmpl.id;
    
    std::set<std::string> template_ops;
    for (const auto& slot : tmpl.slots) {
      for (const auto& op : slot.ops) {
        template_ops.insert(op);
      }
    }
    
    for (const std::string& dfg_op : all_dfg_ops) {
      bool can_support = false;
      
      if (template_ops.count(dfg_op)) {
        can_support = true;
      } else {
        for (const auto& slot : tmpl.slots) {
          if (!slot.ops.empty()) {
            bool compatible = true;
            for (const auto& existing_op : slot.ops) {
              if (!HardwareTemplate::compatible(existing_op, dfg_op)) {
                compatible = false;
                break;
              }
            }
            if (compatible) {
              can_support = true;
              break;
            }
          } else {
            can_support = true;
            break;
          }
        }
      }
      
      if (can_support) {
        ops.single_ops.insert(dfg_op);
      }
    }
    
    ops.composite_ops = tmpl.patterns;
    
    supported_ops.push_back(ops);
  }
}

// Calculates total cost of templates.
double calculate_total_cost(const std::vector<HardwareTemplate>& templates, const OperationCostModel& cost_model) {
  double total_cost = 0;
  for (const auto& t : templates) {
    total_cost += t.compute_cost(cost_model) * t.instances;
  }
  return total_cost;
}

// Escapes string for JSON output.
std::string escape_json_string(const std::string& s) {
  std::string r;
  for (char c : s) {
    if (c == '"') r += "\\\"";
    else if (c == '\\') r += "\\\\";
    else r += c;
  }
  return r;
}

// Writes hardware configuration to JSON file (extended version with execution plans and supported ops).
void write_hardware_config_json(const std::string& path, const std::vector<HardwarePattern>& patterns, const std::vector<HardwareTemplate>& templates, const OperationCostModel& cost_model, const std::vector<PatternExecutionPlan>& execution_plans, const std::vector<TemplateSupportedOps>& supported_ops) {
  std::error_code EC;
  llvm::raw_fd_ostream os(path, EC, llvm::sys::fs::OF_Text);
  if (EC) return;
  
  // Build pattern name lookup.
  std::map<int64_t, std::string> pattern_names;
  for (const auto& p : patterns) pattern_names[p.id] = p.name;
  
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
    
    const TemplateSupportedOps* tmpl_supported_ops = nullptr;
    for (const auto& sop : supported_ops) {
      if (sop.template_id == tmpl.id) {
        tmpl_supported_ops = &sop;
        break;
      }
    }
    
    if (tmpl_supported_ops) {
      os << "        \"supported_single_ops\": [";
      bool first = true;
      for (const auto& op : tmpl_supported_ops->single_ops) {
        if (!first) os << ", ";
        first = false;
        os << "\"" << op << "\"";
      }
      os << "],\n";
      
      os << "        \"supported_composite_ops\": [\n";
      for (size_t i = 0; i < tmpl_supported_ops->composite_ops.size(); ++i) {
        if (i) os << ",\n";
        int64_t pid = tmpl_supported_ops->composite_ops[i];
        auto name_it = pattern_names.find(pid);
        std::string pname = (name_it != pattern_names.end()) ? name_it->second : "";
        os << "          {\"pattern_id\": " << pid << ", \"name\": \"" << escape_json_string(pname) << "\"}";
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
    bool first_conn = true;
    for (const auto& conn : tmpl.connections) {
      if (!first_conn) os << ", ";
      first_conn = false;
      os << "{\"from\": " << conn.first << ", \"to\": " << conn.second << "}";
    }
    os << "]\n";
    os << "        },\n";
    
    os << "        \"pattern_execution_plans\": [\n";
    bool first_plan = true;
    for (const auto& plan : execution_plans) {
      auto mapping_it = tmpl.mapping.find(plan.pattern_id);
      if (mapping_it == tmpl.mapping.end()) continue;
      
      if (!first_plan) os << ",\n";
      first_plan = false;
      
      os << "          {\n";
      os << "            \"pattern_id\": " << plan.pattern_id << ",\n";
      os << "            \"pattern_name\": \"" << escape_json_string(plan.pattern_name) << "\",\n";
      os << "            \"slot_mapping\": [";
      const auto& m = mapping_it->second;
      for (size_t i = 0; i < m.size(); ++i) {
        if (i) os << ", ";
        os << m[i];
      }
      os << "],\n";
      os << "            \"execution_stages\": [\n";
      for (size_t stage_idx = 0; stage_idx < plan.stages.size(); ++stage_idx) {
        const auto& stage = plan.stages[stage_idx];
        if (stage_idx) os << ",\n";
        os << "              {\n";
        os << "                \"stage\": " << stage_idx << ",\n";
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
void write_hardware_config_json(const std::string& path, const std::vector<HardwarePattern>& patterns, const std::vector<HardwareTemplate>& templates, const OperationCostModel& cost_model) {
  std::vector<PatternExecutionPlan> plans;
  generate_execution_plans(patterns, templates, plans);
  
  std::set<std::string> all_dfg_ops;
  for (const auto& p : patterns) {
    for (const auto& op : p.ops) {
      all_dfg_ops.insert(op);
    }
  }
  
  std::vector<TemplateSupportedOps> supported_ops;
  collect_supported_operations(patterns, templates, all_dfg_ops, supported_ops);
  
  write_hardware_config_json(path, patterns, templates, cost_model, plans, supported_ops);
}

} // namespace mlir::neura

