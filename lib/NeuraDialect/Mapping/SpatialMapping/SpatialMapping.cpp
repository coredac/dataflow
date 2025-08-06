#include "NeuraDialect/Mapping/SpatialMapping/SpatialMapping.h"
#include "NeuraDialect/Mapping/MappingState.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraOps.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <vector>

namespace mlir {
namespace neura {
bool SpatialMapping::map(
    std::vector<std::pair<Operation *, int>> &sorted_ops_with_levels,
    std::set<Operation *> &critical_ops, const Architecture &architecture,
    MappingState &mapping_state) {
  return mapWithBacktrack(sorted_ops_with_levels, critical_ops, architecture,
                          mapping_state, 0, 0);
}

bool SpatialMapping::mapWithBacktrack(
    std::vector<std::pair<Operation *, int>> &sorted_ops_with_levels,
    std::set<Operation *> &critical_ops, const Architecture &architecture,
    MappingState &mapping_state, size_t op_index, int backtrack_depth) {
  assert(false && "Not implemented yet");
}

std::vector<MappingLoc> SpatialMapping::caculateSpatialAward(
    Operation *op, std::set<Operation *> &critical_ops, int target_level,
    const Architecture &architecture, const MappingState &mapping_state) {
  assert(false && "Not implemented yet");
}

} // namespace neura
} // namespace mlir