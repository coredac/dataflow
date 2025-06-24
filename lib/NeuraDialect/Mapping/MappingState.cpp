#include "NeuraDialect/Mapping/MappingState.h"

using namespace mlir;
using namespace mlir::neura;

MappingState::MappingState(const Architecture &arch, int II) : II(II) {
  for (Tile* tile : arch.getAllTiles()) {
    for (int t = 0; t < II; ++t) {
      MappingLoc loc = {tile, t};
      all_locs.insert(loc);

      // Create edges to neighboring tiles at t+1.
      for (Tile* dst : tile->getDstTiles()) {
        MappingLoc next_step_dst_tile_loc = {dst, (t + 1) % II}; // modulo II for reuse
        next_step_tiles[loc].push_back(next_step_dst_tile_loc);
      }

      // TODO: Not sure whether we need the link on t or t+1.
      // Creates edges to neighboring links at t.
      for (Link* dst : tile->getOutLinks()) {
        MappingLoc current_step_dst_link_loc = {dst, t % II};
        next_step_tiles[loc].push_back(current_step_dst_link_loc);
      }
    }
  }
}

void MappingState::bindOp(MappingLoc loc, Operation *op) {
  loc_to_op[loc] = op;
  occupied_locs.insert(loc);
}

bool MappingState::isAvailable(const MappingLoc &loc) const {
  return occupied_locs.find(loc) == occupied_locs.end();
}

std::optional<Operation*> MappingState::getOpAt(MappingLoc loc) const {
  auto it = loc_to_op.find(loc);
  if (it == loc_to_op.end()) return std::nullopt;
  return it->second;
}

const std::unordered_set<MappingLoc> &MappingState::getAllLocs() const {
  return all_locs;
}

const std::vector<MappingLoc> &MappingState::getNextStepTiles(MappingLoc loc) const {
  static const std::vector<MappingLoc> empty;
  auto it = next_step_tiles.find(loc);
  return it != next_step_tiles.end() ? it->second : empty;
}

const std::vector<MappingLoc> &MappingState::getNextStepLinks(MappingLoc loc) const {
  static const std::vector<MappingLoc> empty;
  auto it = next_step_links.find(loc);
  return it != next_step_links.end() ? it->second : empty;
}

const std::vector<MappingLoc> &MappingState::getCurrentStepTiles(MappingLoc loc) const {
  static const std::vector<MappingLoc> empty;
  auto it = current_step_tiles.find(loc);
  return it != current_step_tiles.end() ? it->second : empty;
}

const std::vector<MappingLoc> &MappingState::getCurrentStepLinks(MappingLoc loc) const {
  static const std::vector<MappingLoc> empty;
  auto it = current_step_links.find(loc);
  return it != current_step_links.end() ? it->second : empty;
}