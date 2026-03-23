#include "NeuraDialect/Mapping/MappingState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::neura;

// Constants for table formatting in dumpOpToLocs.
// Total column width including separators.
constexpr int kKeyMaxLen = 36;
// Actual cell content width (35).
constexpr int kCellWidth = kKeyMaxLen - 1;
// Threshold to distinguish single-digit from double-digit numbers.
constexpr int kTwoDigitThreshold = 10;
// Threshold to distinguish double-digit from triple-digit numbers.
constexpr int kThreeDigitThreshold = 100;
// Length of time slot header prefix "t%N=" for single-digit II (e.g., "t%3=").
constexpr int kHeaderPrefixLenSingleDigit = 5;
// Length of time slot header prefix "t%N=" for double-digit II (e.g., "t%10=").
constexpr int kHeaderPrefixLenDoubleDigit = 6;
// Number of digits for single-digit slot numbers.
constexpr int kSingleDigitLen = 1;
// Number of digits for double-digit slot numbers.
constexpr int kDoubleDigitLen = 2;

MappingState::MappingState(const Architecture &arch, int II,
                           bool is_spatial_only)
    : II(II), is_spatial_only(is_spatial_only) {}

bool MappingState::bindOp(const MappingLoc &loc, Operation *op) {
  // Default to SINGLE_OCCUPY for backward compatibility
  return bindOp(loc, op, SINGLE_OCCUPY);
}

bool MappingState::bindOp(const MappingLoc &loc, Operation *op,
                          int occupy_status) {
  // Check if the location is available for the specified occupy status
  if (!isAvailableForOccupyStatus(loc, occupy_status)) {
    return false;
  }

  loc_to_op[loc] = op;
  occupied_locs[loc].push_back({occupy_status, op});
  auto it = op_to_locs.find(op);
  assert(it == op_to_locs.end() && "Operation already has reserved locations");
  op_to_locs[op].push_back(loc);
  return true;
}

bool MappingState::bindMultiCycleOp(BasicResource *resource, int start_time,
                                    int latency, Operation *op) {
  // First check if all locations are available
  for (int t = start_time; t < start_time + latency; ++t) {
    MappingLoc check_loc = {resource, t};
    int status;
    if (t == start_time) {
      status = START_PIPE_OCCUPY;
    } else if (t == start_time + latency - 1) {
      status = END_PIPE_OCCUPY;
    } else {
      status = IN_PIPE_OCCUPY;
    }
    if (!isAvailableForOccupyStatus(check_loc, status)) {
      return false;
    }
  }

  // Now bind all locations
  for (int t = start_time; t < start_time + latency; ++t) {
    MappingLoc loc = {resource, t};
    int status;
    if (t == start_time) {
      status = START_PIPE_OCCUPY;
    } else if (t == start_time + latency - 1) {
      status = END_PIPE_OCCUPY;
    } else {
      status = IN_PIPE_OCCUPY;
    }

    loc_to_op[loc] = op;
    occupied_locs[loc].push_back({status, op});
    op_to_locs[op].push_back(loc);
  }
  return true;
}

void MappingState::unbindOp(Operation *op) {
  auto it = op_to_locs.find(op);
  if (it == op_to_locs.end()) {
    return;
  }

  for (const MappingLoc &loc : it->second) {
    loc_to_op.erase(loc);
    // Remove entries for this op from occupied_locs
    auto occ_it = occupied_locs.find(loc);
    if (occ_it != occupied_locs.end()) {
      auto &entries = occ_it->second;
      entries.erase(
          std::remove_if(entries.begin(), entries.end(),
                         [op](const std::pair<int, Operation *> &entry) {
                           return entry.second == op;
                         }),
          entries.end());
      // Remove the location entirely if no more entries
      if (entries.empty()) {
        occupied_locs.erase(occ_it);
      }
    }
  }

  op_to_locs.erase(it);
}

bool MappingState::isAvailableAcrossTime(const MappingLoc &loc) const {
  // For spatial mapping, checks if the location is available across all time.
  if (this->is_spatial_only) {
    for (int t = 0; t < II * kMaxSteps; ++t) {
      MappingLoc check_loc = {loc.resource, t};
      auto it = occupied_locs.find(check_loc);
      if (it != occupied_locs.end()) {
        // Check if all existing occupy statuses allow new single-cycle op
        for (const auto &entry : it->second) {
          if (entry.first != IN_PIPE_OCCUPY) {
            return false;
          }
        }
      }
    }
    return true;
  } else {
    // Checks the availability across time domain.
    for (int t = loc.time_step % II; t < II * kMaxSteps; t += II) {
      MappingLoc check_loc = {loc.resource, t};
      auto it = occupied_locs.find(check_loc);
      if (it != occupied_locs.end()) {
        // Check if all existing occupy statuses allow new single-cycle op
        for (const auto &entry : it->second) {
          if (entry.first != IN_PIPE_OCCUPY) {
            return false;
          }
        }
      }
    }
    return true;
  }
}

bool MappingState::isAvailableForOccupyStatus(const MappingLoc &loc,
                                              int new_occupy_status) const {
  // Helper lambda to check a single location against all existing entries
  auto checkSingleLoc = [this, new_occupy_status](const MappingLoc &check_loc) -> bool {
    auto it = occupied_locs.find(check_loc);
    if (it == occupied_locs.end() || it->second.empty()) {
      // Location is free, always available
      return true;
    }

    // Check against all existing entries at this location
    for (const auto &entry : it->second) {
      int existing_status = entry.first;

      // Implement the pipeline-aware availability rules:
      // - SINGLE_OCCUPY (0): exclusive, no other op can share
      // - START_PIPE_OCCUPY (1): cannot coexist with SINGLE or another START
      // - END_PIPE_OCCUPY (2): cannot coexist with SINGLE or another END
      // - IN_PIPE_OCCUPY (3): can coexist with any status except SINGLE

      if (existing_status == SINGLE_OCCUPY) {
        // SINGLE_OCCUPY blocks everything
        return false;
      }

      if (new_occupy_status == SINGLE_OCCUPY) {
        // SINGLE_OCCUPY cannot be placed if anything is there
        return false;
      }

      if (new_occupy_status == START_PIPE_OCCUPY) {
        // START cannot coexist with another START
        if (existing_status == START_PIPE_OCCUPY) {
          return false;
        }
      }

      if (new_occupy_status == END_PIPE_OCCUPY) {
        // END cannot coexist with another END
        if (existing_status == END_PIPE_OCCUPY) {
          return false;
        }
      }

      // IN_PIPE_OCCUPY can coexist with START, END, or other IN_PIPE
    }
    return true;
  };

  // For spatial mapping, check all time steps
  if (this->is_spatial_only) {
    for (int t = 0; t < II * kMaxSteps; ++t) {
      MappingLoc check_loc = {loc.resource, t};
      if (!checkSingleLoc(check_loc)) {
        return false;
      }
    }
    return true;
  } else {
    // Check across time domain (modulo II)
    for (int t = loc.time_step % II; t < II * kMaxSteps; t += II) {
      MappingLoc check_loc = {loc.resource, t};
      if (!checkSingleLoc(check_loc)) {
        return false;
      }
    }
    return true;
  }
}

int MappingState::getOccupyStatusAcrossTime(const MappingLoc &loc) const {
  // For spatial mapping, check all time steps
  if (this->is_spatial_only) {
    for (int t = 0; t < II * kMaxSteps; ++t) {
      MappingLoc check_loc = {loc.resource, t};
      auto it = occupied_locs.find(check_loc);
      if (it != occupied_locs.end() && !it->second.empty()) {
        // Return the first status found (most restrictive)
        return it->second[0].first;
      }
    }
    return -1;
  } else {
    // Check across time domain (modulo II)
    for (int t = loc.time_step % II; t < II * kMaxSteps; t += II) {
      MappingLoc check_loc = {loc.resource, t};
      auto it = occupied_locs.find(check_loc);
      if (it != occupied_locs.end() && !it->second.empty()) {
        // Return the first status found (most restrictive)
        return it->second[0].first;
      }
    }
    return -1;
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


  llvm::errs() << "Reserving route for operation: " << *op << "\n";
  llvm::errs() << "Path: ";
  for (const MappingLoc &loc : path) {
    llvm::errs() << loc.resource->getType() << "#" << loc.resource->getId() << " @t=" << loc.time_step << " ";
  }
  llvm::errs() << "\n";
  assert(op_to_locs.find(op) == op_to_locs.end() &&
         "Operation already has reserved locations");
  op_to_locs[op] = std::vector<MappingLoc>(path.begin(), path.end());

  for (const MappingLoc &loc : path) {
    loc_to_op[loc] = op;
    // Use SINGLE_OCCUPY for route reservations (links/registers)
    occupied_locs[loc].push_back({SINGLE_OCCUPY, op});
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
    // Remove entries for this op from occupied_locs
    auto occ_it = occupied_locs.find(loc);
    if (occ_it != occupied_locs.end()) {
      auto &entries = occ_it->second;
      entries.erase(
          std::remove_if(entries.begin(), entries.end(),
                         [op](const std::pair<int, Operation *> &entry) {
                           return entry.second == op;
                         }),
          entries.end());
      // Remove the location entirely if no more entries
      if (entries.empty()) {
        occupied_locs.erase(occ_it);
      }
    }
  }

  op_to_locs.erase(it);
}

void MappingState::dumpOpToLocs(llvm::raw_ostream &os) const {
  os << "=== MappingState: Resource Allocation Table ===\n";

  // Collects all tiles and time steps (modulo II).
  std::set<int> tile_ids;
  // Time slots range from 0 to II-1.
  std::set<int> time_slots;
  // Maps (tile_id, time_slot) to list of (operation, actual_time_step).
  std::map<std::pair<int, int>, std::vector<std::pair<Operation *, int>>>
      tile_slot_to_ops;

  for (const auto &[op, locs] : op_to_locs) {
    for (const MappingLoc &loc : locs) {
      auto *res = loc.resource;
      // Only shows tiles in the table.
      if (res->getType() == "tile") {
        tile_ids.insert(res->getId());
        // Computes modulo II.
        int time_slot = loc.time_step % II;
        time_slots.insert(time_slot);
        tile_slot_to_ops[{res->getId(), time_slot}].push_back(
            {op, loc.time_step});
      }
    }
  }

  if (tile_ids.empty() || time_slots.empty()) {
    os << "No tile operations mapped.\n";
    os << "=== End ===\n";
    return;
  }

  os << "II = " << II << "\n";

  // Prints header - time slots (0 to II-1) as columns.
  os << "\nTile     | ";
  for (int slot : time_slots) {
    os << "t%" << II << "=" << slot;
    int padding =
        kKeyMaxLen -
        (II < kTwoDigitThreshold ? kHeaderPrefixLenSingleDigit
                                 : kHeaderPrefixLenDoubleDigit) -
        (slot < kTwoDigitThreshold ? kSingleDigitLen : kDoubleDigitLen);
    for (int i = 0; i < padding; ++i)
      os << " ";
    os << " | ";
  }
  os << "\n";

  // Prints separator line.
  os << "---------+";
  for (size_t i = 0; i < time_slots.size(); ++i) {
    for (int j = 0; j < kKeyMaxLen + 1; ++j)
      os << "-";
    os << "+";
  }
  os << "\n";

  // Prints each tile as a row.
  for (int tile_id : tile_ids) {
    os << "Tile#" << tile_id;
    if (tile_id < kTwoDigitThreshold)
      os << "  ";
    else if (tile_id < kThreeDigitThreshold)
      os << " ";
    os << " | ";

    for (int slot : time_slots) {
      auto it = tile_slot_to_ops.find({tile_id, slot});
      if (it != tile_slot_to_ops.end() && !it->second.empty()) {
        // Multiple operations may exist in the same slot (from different
        // iterations). Shows the first one.
        Operation *op = it->second[0].first;
        int actual_time = it->second[0].second;

        // Builds operation string: %result = op_name(%operand1, %operand2,
        // ...).
        std::string op_str;
        llvm::raw_string_ostream op_stream(op_str);
        mlir::OpPrintingFlags flags;

        // Prints result (if exists).
        if (op->getNumResults() > 0) {
          op->getResult(0).printAsOperand(op_stream, flags);
          op_stream << " = ";
        }

        // Prints operation name (removes "neura." prefix).
        std::string op_name = op->getName().getStringRef().str();
        if (op_name.rfind("neura.", 0) == 0) {
          op_name = op_name.substr(6);
        }
        op_stream << op_name;

        // Prints operands.
        if (op->getNumOperands() > 0) {
          op_stream << "(";
          for (unsigned i = 0; i < op->getNumOperands(); ++i) {
            if (i > 0)
              op_stream << ", ";
            op->getOperand(i).printAsOperand(op_stream, flags);
          }
          op_stream << ")";
        }

        // Adds time annotation if not in [0, II).
        if (actual_time >= II) {
          op_stream << " (t=" << actual_time << ")";
        }

        op_stream.flush();

        // Truncates string if too long to fit in the cell.
        if (op_str.length() > kCellWidth) {
          op_str = op_str.substr(0, kCellWidth - 3) + "...";
        }

        // Pads to fixed width (kCellWidth chars).
        os << op_str;
        int padding = kCellWidth - op_str.length();
        for (int i = 0; i < padding; ++i)
          os << " ";
      } else {
        // Renders empty cell.
        for (int i = 0; i < kCellWidth; ++i)
          os << " ";
      }
      os << " | ";
    }
    os << "\n";
  }

  os << "\n=== Legend ===\n";
  os << "- Table shows operations mapped to tiles (modulo II scheduling)\n";
  os << "- Column headers: t%II=X means time slot X (t=X, X+II, X+2*II, ...)\n";
  os << "- Operations with (t=Y) annotation are scheduled at actual time step "
        "Y\n";
  os << "- Operations without annotation are scheduled at t=0 to t=" << (II - 1)
     << "\n";
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
        int invalid_iterations = loc.time_step / II;
        int index_per_ii = loc.time_step % II;
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
                      mlir::StringAttr::get(ctx, "invalid_iterations"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             invalid_iterations)),
                  mlir::NamedAttribute(
                      mlir::StringAttr::get(ctx, "index_per_ii"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             index_per_ii)),
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
        int invalid_iterations = loc.time_step / II;
        int index_per_ii = loc.time_step % II;
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
                      mlir::StringAttr::get(ctx, "invalid_iterations"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             invalid_iterations)),
                  mlir::NamedAttribute(
                      mlir::StringAttr::get(ctx, "index_per_ii"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             index_per_ii))});
        mapping_entries.push_back(dict);
      } else if (loc.resource->getKind() == ResourceKind::Register) {
        kind_str = "register";
        Register *reg = static_cast<Register *>(loc.resource);
        int global_id = loc.resource->getId();
        int per_tile_register_id = reg->getPerTileId();
        int invalid_iterations = loc.time_step / II;
        int index_per_ii = loc.time_step % II;

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
                                             loc.time_step)),
                  mlir::NamedAttribute(
                      mlir::StringAttr::get(ctx, "invalid_iterations"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             invalid_iterations)),
                  mlir::NamedAttribute(
                      mlir::StringAttr::get(ctx, "index_per_ii"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             index_per_ii))});
        mapping_entries.push_back(dict);
      } else {
        kind_str = "unknown";
        int invalid_iterations = loc.time_step / II;
        int index_per_ii = loc.time_step % II;
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
                      mlir::StringAttr::get(ctx, "invalid_iterations"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             invalid_iterations)),
                  mlir::NamedAttribute(
                      mlir::StringAttr::get(ctx, "index_per_ii"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             index_per_ii))});
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
