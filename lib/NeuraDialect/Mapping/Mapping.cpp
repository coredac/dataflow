#include "NeuraDialect/Mapping/Mapping.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include <memory>

// TODO: Move common functions from mapping_util.cpp to Mapping.cpp
// Issue Link: https://github.com/coredac/dataflow/issues/107

namespace mlir {
namespace neura {
void Mapping::loadDfg(
    const std::vector<std::pair<Operation *, int>> &sorted_ops_with_levels,
    const std::set<Operation *> &critical_ops) {
  this->sorted_ops_with_levels = sorted_ops_with_levels;
  this->critical_ops = critical_ops;
  for (auto [op, level] : sorted_ops_with_levels) {
    if (!is_non_materialized(op)) {
      this->materialized_ops_with_levels.emplace_back(op, level);
    }
  }
}
} // namespace neura

} // namespace mlir