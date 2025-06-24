#include "NeuraDialect/Mapping/MappingState.h"

using namespace mlir;
using namespace mlir::neura;

void MappingState::bindOp(MappingLoc loc, Operation *op) {
  loc_to_op[loc] = op;
  occupied_locs.insert(loc);
}

bool MappingState::isAvailable(const MappingLoc &loc) const {
  return !occupied_locs.contains(loc);
}
