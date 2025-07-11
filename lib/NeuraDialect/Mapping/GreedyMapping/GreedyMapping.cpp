#include "NeuraDialect/Mapping/GreedyMapping/GreedyMapping.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraOps.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace neura {

bool GreedyMapping::map(std::vector<Operation*>& sorted_ops,
                               const Architecture& architecture,
                               MappingState& mapping_state) {
  // This is just a wrapper around the existing tryHeuristicMapping function
  return tryHeuristicMapping(sorted_ops, architecture, mapping_state);
}

} // namespace neura
} // namespace mlir