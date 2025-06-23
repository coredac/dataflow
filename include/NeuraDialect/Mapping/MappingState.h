#ifndef NEURA_MAPPING_STATE_H
#define NEURA_MAPPING_STATE_H

#include "mlir/IR/Operation.h"
#include "NeuraDialect/Architecture/Architecture.h"  // for BasicResource
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <optional>

namespace mlir {
namespace neura {

// Represents a spatial-temporal location: (resource, timeStep)
using MappingLoc = std::pair<BasicResource*, int>;

// Tracks placement and routing of ops on the CGRA.
class MappingState {
public:
  // Binds a (tile/link, timeStep) location to an operation.
  void bindOp(MappingLoc loc, Operation *op);

  // Checks if a (tile/link, timeStep) is available (unoccupied).
  bool isAvailable(const MappingLoc &loc) const;

  // Gets the operation at a specific (tile/link, timeStep) location.
  std::optional<Operation*> getOpAt(MappingLoc loc) const;

private:
  std::unordered_map<MappingLoc, Operation*> loc_to_op;
  std::unordered_set<MappingLoc> occupied_locs;
};

} // namespace neura
} // namespace mlir

#endif // NEURA_MAPPING_STATE_H
