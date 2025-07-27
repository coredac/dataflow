#ifndef NEURA_MAPPING_STATE_H
#define NEURA_MAPPING_STATE_H

#include "NeuraDialect/Architecture/Architecture.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mlir {
namespace neura {

// Represents a spatial-temporal location: (resource, time_step)
struct MappingLoc {
  BasicResource *resource;
  int time_step;

  bool operator==(const MappingLoc &other) const {
    return resource == other.resource && time_step == other.time_step;
  }

  bool operator<(const MappingLoc &other) const {
    if (time_step != other.time_step)
      return time_step < other.time_step;
    return resource->getId() < other.resource->getId();
  }
};

} // namespace neura
} // namespace mlir

namespace std {
template <> struct hash<mlir::neura::MappingLoc> {
  std::size_t operator()(const mlir::neura::MappingLoc &loc) const {
    std::size_t h1 = std::hash<mlir::neura::BasicResource *>()(loc.resource);
    std::size_t h2 = std::hash<int>()(loc.time_step);
    return h1 ^ (h2 << 1);
  }
};
} // namespace std

namespace mlir {
namespace neura {

// Tracks placement and routing of ops on the CGRA.
class MappingState {
public:
  MappingState(const Architecture &arch, int II);
  // Binds a (tile/link, time_step) location to an operation.
  bool bindOp(const MappingLoc &loc, Operation *op);

  // Unbinds an operation from its (tile/link, time_step) location,
  // which is useful for backtracking.
  void unbindOp(Operation *op);

  // Checks if a (tile/link, time_step) is available (unoccupied).
  // Note that the check is performed in II granularity.
  // For example, if II is 4, and we want to check (tile 2, step 5), then
  // it will check (tile 2, step 1), (tile 2, step 5), (tile 2, step 9), etc.
  bool isAvailableAcrossTime(const MappingLoc &loc) const;

  // Checks if a hardware resource is available across a time range.
  // This function leverages the isAvailableAcrossTime function in each
  // time step.
  bool isAvailableAcrossTimeInRange(BasicResource *resource,
                                    int start_time,
                                    int exclusive_end_time) const;

  // Gets the operation at a specific (tile/link, time_step) location.
  std::optional<Operation *> getOpAt(MappingLoc loc) const;

  // Counts the number of operations at a specific resource across time steps.
  int countOpsAtResource(BasicResource *resource) const;

  // Gets all MRRG nodes allocated to a given op.
  const std::vector<MappingLoc> &getAllLocsOfOp(Operation *op) const;

  // Reserves links for an move operation.
  void reserveRoute(Operation *op, ArrayRef<MappingLoc> path);

  // Releases links for an move operation.
  void releaseRoute(Operation *op);

  // Gets neighboring tiles on next step of a given MappingLoc.
  std::vector<MappingLoc> getNextStepTiles(MappingLoc loc) const;

  //   // Gets neighboring links on next step of a given MappingLoc.
  //   const std::vector<MappingLoc> &getNextStepLinks(MappingLoc loc) const;

  //   // Gets neighboring tiles on current step of a given MappingLoc.
  //   const std::vector<MappingLoc> &getCurrentStepTiles(MappingLoc loc) const;

  // Gets neighboring links on current step of a given MappingLoc.
  std::vector<MappingLoc> getCurrentStepLinks(MappingLoc loc) const;

  // Gets the target initiation interval (II) for the mapping.
  int getII() const { return II; }

  // Embeds the mapping states onto the mapped operations.
  void encodeMappingState();

  void dumpOpToLocs(llvm::raw_ostream &os = llvm::errs()) const;

  // Getters for state information.
  const std::set<MappingLoc> &getOccupiedLocs() const {
    return this->occupied_locs;
  }
  const std::map<MappingLoc, Operation *> &getLocToOp() const {
    return this->loc_to_op;
  }
  const std::map<Operation *, std::vector<MappingLoc>> &getOpToLocs() const {
    return this->op_to_locs;
  }

  // Setters for state information.
  void setOccupiedLocs(const std::set<MappingLoc> &locs) {
    this->occupied_locs = locs;
  }
  void setLocToOp(const std::map<MappingLoc, Operation *> &loc_to_op) {
    this->loc_to_op = loc_to_op;
  }
  void setOpToLocs(const std::map<Operation *, std::vector<MappingLoc>> &op_to_locs) {
    this->op_to_locs = op_to_locs;
  }

private:
  // Initiation interval.
  int II;
  static constexpr int kMaxSteps = 10;

  std::set<MappingLoc> occupied_locs;
  std::map<MappingLoc, Operation *> loc_to_op;
  std::map<Operation *, std::vector<MappingLoc>> op_to_locs;
};

} // namespace neura
} // namespace mlir

#endif // NEURA_MAPPING_STATE_H
