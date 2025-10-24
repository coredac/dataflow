#include "NeuraDialect/Mapping/MappingState.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::neura;

MappingState::MappingState(const Architecture &arch, int II,
                           bool is_spatial_only)
    : II(II), is_spatial_only(is_spatial_only) {}

bool MappingState::bindOp(const MappingLoc &loc, Operation *op) {
  loc_to_op[loc] = op;
  occupied_locs.insert(loc);
  auto it = op_to_locs.find(op);
  assert(it == op_to_locs.end() && "Operation already has reserved locations");
  op_to_locs[op].push_back(loc);
  return true;
}

void MappingState::unbindOp(Operation *op) {
  auto it = op_to_locs.find(op);
  if (it == op_to_locs.end()) {
    return;
  }

  for (const MappingLoc &loc : it->second) {
    loc_to_op.erase(loc);
    occupied_locs.erase(loc);
  }

  op_to_locs.erase(it);
}

bool MappingState::isAvailableAcrossTime(const MappingLoc &loc) const {
  // For spatial mapping, checks if the location is available across all time.
  if (this->is_spatial_only) {
    for (int t = 0; t < II * kMaxSteps; ++t) {
      MappingLoc check_loc = {loc.resource, t};
      if (occupied_locs.find(check_loc) != occupied_locs.end()) {
        return false;
      }
    }
    return true;
  } else {

    // Checks the availability across time domain.
    for (int t = loc.time_step % II; t < II * kMaxSteps; t += II) {
      MappingLoc check_loc = {loc.resource, t};
      if (occupied_locs.find(check_loc) != occupied_locs.end()) {
        return false;
      }
    }
    return true;
  }
}

bool MappingState::isAvailableAcrossTimeInRange(BasicResource *resource,
                                                int start_time,
                                                int exclusive_end_time) const {
  // Checks the availability for each time step across time domain.
  for (int t = start_time; t < exclusive_end_time; ++t) {
    MappingLoc check_loc = {resource, t};
    // Checks the availability across time domain.
    if (!isAvailableAcrossTime(check_loc)) {
      return false;
    }
  }
  return true;
}

std::optional<Operation *> MappingState::getOpAt(MappingLoc loc) const {
  auto it = loc_to_op.find(loc);
  if (it == loc_to_op.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<Operation *>
MappingState::getOpAtLocAcrossTime(MappingLoc loc) const {
  for (int t = loc.time_step % II; t < II * kMaxSteps; t += II) {
    MappingLoc check_loc = {loc.resource, t};
    auto it = loc_to_op.find(check_loc);
    if (it != loc_to_op.end()) {
      return it->second;
    }
  }
  return std::nullopt;
}

int MappingState::countOpsAtResource(BasicResource *resource) const {
  int count = 0;
  for (const auto &[loc, op] : loc_to_op) {
    if (loc.resource == resource) {
      count++;
    }
  }
  return count;
}

const std::vector<MappingLoc> &
MappingState::getAllLocsOfOp(Operation *op) const {
  auto it = op_to_locs.find(op);
  if (it != op_to_locs.end()) {
    return it->second;
  }

  static const std::vector<MappingLoc> empty;
  return empty;
}

std::vector<MappingLoc> MappingState::getNextStepTiles(MappingLoc loc) const {
  std::vector<MappingLoc> next_step_tiles;
  const int next_step = loc.time_step + 1;
  assert(next_step < II * kMaxSteps && "Next step exceeds max steps");
  // Collects neighboring tiles at t+1 for both tile and link.
  if (loc.resource->getKind() == ResourceKind::Tile) {
    Tile *tile = dyn_cast<Tile>(loc.resource);
    for (Tile *dst : tile->getDstTiles()) {
      MappingLoc next_step_dst_tile_loc = {dst, next_step};
      next_step_tiles.push_back(next_step_dst_tile_loc);
    }
    // Includes self for reuse.
    next_step_tiles.push_back({tile, next_step});
  } else if (loc.resource->getKind() == ResourceKind::Link) {
    Link *link = dyn_cast<Link>(loc.resource);
    Tile *dst = link->getDstTile();
    MappingLoc next_step_dst_tile_loc = {dst, next_step};
    next_step_tiles.push_back(next_step_dst_tile_loc);
  }
  return next_step_tiles;
}

// const std::vector<MappingLoc> &MappingState::getNextStepLinks(MappingLoc loc)
// const {
//   static const std::vector<MappingLoc> empty;
//   auto it = next_step_links.find(loc);
//   return it != next_step_links.end() ? it->second : empty;
// }

// const std::vector<MappingLoc> &MappingState::getCurrentStepTiles(MappingLoc
// loc) const {
//   static const std::vector<MappingLoc> empty;
//   auto it = current_step_tiles.find(loc);
//   return it != current_step_tiles.end() ? it->second : empty;
// }

std::vector<MappingLoc>
MappingState::getCurrentStepLinks(MappingLoc loc) const {
  assert((loc.resource->getKind() == ResourceKind::Tile) &&
         "Current step links can only be queried for tiles");
  std::vector<MappingLoc> current_step_links;
  const int current_step = loc.time_step;
  if (!(current_step < II * kMaxSteps)) {
    llvm::errs() << "Current step exceeds max steps: " << current_step
                 << ", max steps: " << II * kMaxSteps << "\n";
    return current_step_links; // Return empty if step exceeds max.
  }
  // Collects neighboring tiles at t for given tile.
  Tile *tile = dyn_cast<Tile>(loc.resource);
  for (Link *out_link : tile->getOutLinks()) {
    MappingLoc current_step_out_link_loc = {out_link, current_step};
    current_step_links.push_back(current_step_out_link_loc);
  }
  return current_step_links;
}

void MappingState::reserveRoute(Operation *op, ArrayRef<MappingLoc> path) {

  // Records all mapping locations.
  assert(op_to_locs.find(op) == op_to_locs.end() &&
         "Operation already has reserved locations");
  op_to_locs[op] = std::vector<MappingLoc>(path.begin(), path.end());

  for (const MappingLoc &loc : path) {
    assert(occupied_locs.find(loc) == occupied_locs.end() &&
           "Mapping location already occupied");
    loc_to_op[loc] = op;
    assert(occupied_locs.find(loc) == occupied_locs.end() &&
           "Mapping location already occupied in occupied_locs");
    occupied_locs.insert(loc);
  }
}

void MappingState::releaseRoute(Operation *op) {
  auto it = op_to_locs.find(op);
  if (it == op_to_locs.end()) {
    return;
  }

  const std::vector<MappingLoc> &route = it->second;

  for (const MappingLoc &loc : route) {
    loc_to_op.erase(loc);
    occupied_locs.erase(loc);
  }

  op_to_locs.erase(it);
}

void MappingState::dumpOpToLocs(llvm::raw_ostream &os) const {
  os << "=== MappingState: op_to_locs ===\n";

  for (const auto &[op, locs] : op_to_locs) {
    os << "  - " << op->getName();
    if (auto name_attr = op->getAttrOfType<StringAttr>("sym_name")) {
      os << " @" << name_attr;
    }
    os << "\n";

    for (const MappingLoc &loc : locs) {
      auto *res = loc.resource;
      os << "      -> " << res->getType() << "#" << res->getId()
         << " @t=" << loc.time_step << "\n";
    }
  }
  os << "=== End ===\n";
}

void MappingState::encodeMappingState() {
  for (const auto &[op, locs] : op_to_locs) {
    llvm::SmallVector<mlir::Attribute, 4> mapping_entries;
    auto ctx = op->getContext();
    for (const MappingLoc &loc : locs) {
      std::string kind_str;
      if (loc.resource->getKind() == ResourceKind::Tile) {
        kind_str = "tile";
        Tile *tile = dyn_cast<Tile>(loc.resource);
        auto dict = mlir::DictionaryAttr::get(
            ctx, {mlir::NamedAttribute(mlir::StringAttr::get(ctx, "resource"),
                                       mlir::StringAttr::get(ctx, kind_str)),
                  mlir::NamedAttribute(
                      mlir::StringAttr::get(ctx, "id"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             loc.resource->getId())),
                  mlir::NamedAttribute(
                      mlir::StringAttr::get(ctx, "time_step"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             loc.time_step)),
                  mlir::NamedAttribute(
                      mlir::StringAttr::get(ctx, "x"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             tile->getX())),
                  mlir::NamedAttribute(
                      mlir::StringAttr::get(ctx, "y"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             tile->getY()))});
        mapping_entries.push_back(dict);
      } else if (loc.resource->getKind() == ResourceKind::Link) {
        kind_str = "link";
        auto dict = mlir::DictionaryAttr::get(
            ctx, {mlir::NamedAttribute(mlir::StringAttr::get(ctx, "resource"),
                                       mlir::StringAttr::get(ctx, kind_str)),
                  mlir::NamedAttribute(
                      mlir::StringAttr::get(ctx, "id"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             loc.resource->getId())),
                  mlir::NamedAttribute(
                      mlir::StringAttr::get(ctx, "time_step"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             loc.time_step))});
        mapping_entries.push_back(dict);
      } else if (loc.resource->getKind() == ResourceKind::Register) {
        kind_str = "register";
        Register *reg = static_cast<Register *>(loc.resource);
        int global_id = loc.resource->getId();
        int per_tile_register_id = reg->getPerTileId();
        
        auto dict = mlir::DictionaryAttr::get(
            ctx, {mlir::NamedAttribute(mlir::StringAttr::get(ctx, "resource"),
                                       mlir::StringAttr::get(ctx, kind_str)),
                  mlir::NamedAttribute(
                      mlir::StringAttr::get(ctx, "id"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             global_id)),
                  mlir::NamedAttribute(
                      mlir::StringAttr::get(ctx, "per_tile_register_id"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             per_tile_register_id)),
                  mlir::NamedAttribute(
                      mlir::StringAttr::get(ctx, "time_step"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             loc.time_step))});
        mapping_entries.push_back(dict);
      } else {
        kind_str = "unknown";
        auto dict = mlir::DictionaryAttr::get(
            ctx, {mlir::NamedAttribute(mlir::StringAttr::get(ctx, "resource"),
                                       mlir::StringAttr::get(ctx, kind_str)),
                  mlir::NamedAttribute(
                      mlir::StringAttr::get(ctx, "id"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             loc.resource->getId())),
                  mlir::NamedAttribute(
                      mlir::StringAttr::get(ctx, "time_step"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             loc.time_step))});
        mapping_entries.push_back(dict);
      }
    }
    op->setAttr("mapping_locs", mlir::ArrayAttr::get(ctx, mapping_entries));
  }
}

MappingStateSnapshot::MappingStateSnapshot(const MappingState &mapping_state) {
  this->occupied_locs = mapping_state.getOccupiedLocs();
  this->loc_to_op = mapping_state.getLocToOp();
  this->op_to_locs = mapping_state.getOpToLocs();
}

void MappingStateSnapshot::restore(MappingState &mapping_state) {
  mapping_state.setOccupiedLocs(this->occupied_locs);
  mapping_state.setLocToOp(this->loc_to_op);
  mapping_state.setOpToLocs(this->op_to_locs);
}
