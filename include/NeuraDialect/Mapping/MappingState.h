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

// Represents a spatial-temporal location: (resource, time_step)
struct MappingLoc {
  BasicResource* resource;
  int time_step;

  bool operator==(const MappingLoc &other) const {
    return resource == other.resource && time_step == other.time_step;
  }
};

} // namespace neura
} // namespace mlir

namespace std {
template <>
struct hash<mlir::neura::MappingLoc> {
  std::size_t operator()(const mlir::neura::MappingLoc& loc) const {
    std::size_t h1 = std::hash<mlir::neura::BasicResource*>()(loc.resource);
    std::size_t h2 = std::hash<int>()(loc.time_step);
    return h1 ^ (h2 << 1);
  }
};
}

namespace mlir {
namespace neura {

// Tracks placement and routing of ops on the CGRA.
class MappingState {
public:
  MappingState(const Architecture &arch, int II);
  // Binds a (tile/link, time_step) location to an operation.
  void bindOp(MappingLoc loc, Operation *op);

  // Checks if a (tile/link, time_step) is available (unoccupied).
  bool isAvailable(const MappingLoc &loc) const;

  // Gets the operation at a specific (tile/link, time_step) location.
  std::optional<Operation*> getOpAt(MappingLoc loc) const;

  // Gets all MRRG nodes.
  const std::unordered_set<MappingLoc> &getAllLocs() const;

  // Gets neighboring tiles on next step of a given MappingLoc.
  const std::vector<MappingLoc> &getNextStepTiles(MappingLoc loc) const;

  // Gets neighboring links on next step of a given MappingLoc.
  const std::vector<MappingLoc> &getNextStepLinks(MappingLoc loc) const;

  // Gets neighboring tiles on current step of a given MappingLoc.
  const std::vector<MappingLoc> &getCurrentStepTiles(MappingLoc loc) const;

  // Gets neighboring links on current step of a given MappingLoc.
  const std::vector<MappingLoc> &getCurrentStepLinks(MappingLoc loc) const;

  int getII() const { return II; }

private:
  // Initiation interval.
  int II;
  std::unordered_set<MappingLoc> all_locs;
  // current and next step tiles and links for a given MappingLoc. Note that
  // the key MappingLoc is either a pair of (tile, time_step) or (link, time_step).
  std::unordered_map<MappingLoc, std::vector<MappingLoc>> next_step_tiles;
  std::unordered_map<MappingLoc, std::vector<MappingLoc>> next_step_links;
  std::unordered_map<MappingLoc, std::vector<MappingLoc>> current_step_tiles;
  std::unordered_map<MappingLoc, std::vector<MappingLoc>> current_step_links;

  std::unordered_map<MappingLoc, Operation*> loc_to_op;
  std::unordered_set<MappingLoc> occupied_locs;
};

} // namespace neura
} // namespace mlir

#endif // NEURA_MAPPING_STATE_H
