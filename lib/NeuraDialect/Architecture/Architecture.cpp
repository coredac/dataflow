#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Architecture/ArchitectureSpec.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace mlir;
using namespace mlir::neura;

// Configures all supported operations for a function unit.
void configureSupportedOperations(CustomizableFunctionUnit *function_unit, const std::string &operation) {
  // Integer arithmetic operations.
  if (operation == "add") {
    function_unit->addSupportedOperation(IAdd);
  } else if (operation == "sub") {
    function_unit->addSupportedOperation(ISub);
  } else if (operation == "mul") {
    function_unit->addSupportedOperation(IMul);
  } else if (operation == "div") {
    function_unit->addSupportedOperation(IDiv);
  } else if (operation == "rem") {
    function_unit->addSupportedOperation(IRem);
  }
  // Floating-point arithmetic operations.
  else if (operation == "fadd") {
    function_unit->addSupportedOperation(FAdd);
  } else if (operation == "fsub") {
    function_unit->addSupportedOperation(FSub);
  } else if (operation == "fmul") {
    function_unit->addSupportedOperation(FMul);
  } else if (operation == "fdiv") {
    function_unit->addSupportedOperation(FDiv);
  }
  // Memory operations.
  else if (operation == "load") {
    function_unit->addSupportedOperation(ILoad);
  } else if (operation == "store") {
    function_unit->addSupportedOperation(IStore);
  } else if (operation == "load_indexed") {
    function_unit->addSupportedOperation(ILoadIndexed);
  } else if (operation == "store_indexed") {
    function_unit->addSupportedOperation(IStoreIndexed);
  } else if (operation == "alloca") {
    function_unit->addSupportedOperation(IAlloca);
  }
  // Logical operations.
  else if (operation == "or") {
    function_unit->addSupportedOperation(IOr);
  } else if (operation == "not") {
    function_unit->addSupportedOperation(INot);
  } else if (operation == "icmp") {
    function_unit->addSupportedOperation(ICmp);
  } else if (operation == "fcmp") {
    function_unit->addSupportedOperation(FCmp);
  } else if (operation == "sel") {
    function_unit->addSupportedOperation(ISel);
  }
  // Type conversion operations.
  else if (operation == "cast") {
    function_unit->addSupportedOperation(ICast);
  } else if (operation == "sext") {
    function_unit->addSupportedOperation(ISExt);
  } else if (operation == "zext") {
    function_unit->addSupportedOperation(IZExt);
  } else if (operation == "shl") {
    function_unit->addSupportedOperation(IShl);
  }
  // Vector and fused operations.
  else if (operation == "vfmul") {
    function_unit->addSupportedOperation(VFMul);
  } else if (operation == "fadd_fadd") {
    function_unit->addSupportedOperation(FAddFAdd);
  } else if (operation == "fmul_fadd") {
    function_unit->addSupportedOperation(FMulFAdd);
  }
  // Control flow operations.
  else if (operation == "return") {
    function_unit->addSupportedOperation(IReturn);
  } else if (operation == "phi") {
    function_unit->addSupportedOperation(IPhi);
  } else if (operation == "loop_control") {
    function_unit->addSupportedOperation(ILoopControl);
  }
  // Data movement operations.
  else if (operation == "data_mov") {
    function_unit->addSupportedOperation(IDataMov);
  } else if (operation == "ctrl_mov") {
    function_unit->addSupportedOperation(ICtrlMov);
  }
  // Predicate and reservation operations.
  else if (operation == "reserve") {
    function_unit->addSupportedOperation(IReserve);
  } else if (operation == "grant_predicate") {
    function_unit->addSupportedOperation(IGrantPredicate);
  } else if (operation == "grant_once") {
    function_unit->addSupportedOperation(IGrantOnce);
  } else if (operation == "grant_always") {
    function_unit->addSupportedOperation(IGrantAlways);
  }
  // Constant operations.
  else if (operation == "constant") {
    function_unit->addSupportedOperation(IConstant);
  }
}

// Creates a function unit for a specific operation.
// Maps YAML operation names to OperationKind enum values and creates appropriate function units.
void createFunctionUnitForOperation(Tile *tile, const std::string &operation, int &function_unit_id) {
  auto function_unit = std::make_unique<CustomizableFunctionUnit>(function_unit_id++);
  
  // Configures all supported operations using the unified function.
  configureSupportedOperations(function_unit.get(), operation);
  
  // TODO: Add support for unknown operations with warning instead of silent failure.
  // This would help users identify typos in their YAML configuration.
  
  tile->addFunctionUnit(std::move(function_unit));
}

// Configures tile function units based on operations.
void configureTileFunctionUnits(Tile *tile, const std::vector<std::string> &operations, bool clear_existing = true) {
  // Configures function units based on YAML operations specification.
  // If clear_existing is true, this replaces any existing function units with the specified ones.
  
  if (clear_existing) {
    tile->clearFunctionUnits();
  }
  
  int function_unit_id = 0;
  for (const auto &operation : operations) {
    createFunctionUnitForOperation(tile, operation, function_unit_id);
  }
}

//===----------------------------------------------------------------------===//
// Tile
//===----------------------------------------------------------------------===//

Tile::Tile(int id, int x, int y) {
  this->id = id;
  this->x = x;
  this->y = y;

}

int Tile::getId() const { return id; }

int Tile::getX() const { return x; }

int Tile::getY() const { return y; }

void Tile::linkDstTile(Link *link, Tile *tile) {
  assert(tile && "Cannot link to a null tile");
  dst_tiles.insert(tile);
  out_links.insert(link);
  tile->src_tiles.insert(this);
  tile->in_links.insert(link);
}

void Tile::unlinkDstTile(Link *link, Tile *tile) {
  assert(tile && "Cannot unlink from a null tile");
  dst_tiles.erase(tile);
  out_links.erase(link);
  tile->src_tiles.erase(this);
  tile->in_links.erase(link);
}

const std::set<Tile *> &Tile::getDstTiles() const { return dst_tiles; }

const std::set<Tile *> &Tile::getSrcTiles() const { return src_tiles; }

const std::set<Link *> &Tile::getOutLinks() const { return out_links; }

const std::set<Link *> &Tile::getInLinks() const { return in_links; }

void Tile::addRegisterFileCluster(RegisterFileCluster *register_file_cluster) {
  assert(register_file_cluster && "Cannot add null register file cluster");
  if (this->register_file_cluster != nullptr) {
    llvm::errs() << "Warning: Overwriting existing register file cluster ("
                 << this->register_file_cluster->getId() << ") in Tile "
                 << this->id << "\n";
    // Remove the old register file cluster before adding the new one.
    delete this->register_file_cluster;
  }
  this->register_file_cluster = register_file_cluster;
  register_file_cluster->setTile(this);
}

const RegisterFileCluster *Tile::getRegisterFileCluster() const {
  return register_file_cluster;
}

const std::vector<RegisterFile *> Tile::getRegisterFiles() const {
  std::vector<RegisterFile *> all_register_files;
  if (this->register_file_cluster) {
    for (const auto &[id, file] :
         this->register_file_cluster->getRegisterFiles()) {
      all_register_files.push_back(file);
    }
  }
  return all_register_files;
}

const std::vector<Register *> Tile::getRegisters() const {
  std::vector<Register *> all_registers;
  if (this->register_file_cluster) {
    for (const auto &[reg_file_id, reg_file] :
         this->register_file_cluster->getRegisterFiles()) {
      for (const auto &[reg_id, reg] : reg_file->getRegisters()) {
        all_registers.push_back(reg);
      }
    }
  }
  return all_registers;
}

//===----------------------------------------------------------------------===//
// Link
//===----------------------------------------------------------------------===//

Link::Link(int id) { this->id = id; }

int Link::getId() const { return id; }

Tile *Link::getSrcTile() const { return src_tile; }

Tile *Link::getDstTile() const { return dst_tile; }

void Link::connect(Tile *src, Tile *dst) {
  assert(src && dst && "Cannot connect null tiles");
  src_tile = src;
  dst_tile = dst;
  src->linkDstTile(this, dst);
}

//===----------------------------------------------------------------------===//
// FunctionUnit
//===----------------------------------------------------------------------===//

FunctionUnit::FunctionUnit(int id) { this->id = id; }

int FunctionUnit::getId() const { return id; }

void FunctionUnit::setTile(Tile *tile) { this->tile = tile; }

Tile *FunctionUnit::getTile() const { return this->tile; }

//===----------------------------------------------------------------------===//
// Register
//===----------------------------------------------------------------------===//

Register::Register(int id) { this->id = id; }

int Register::getId() const { return id; }

Tile *Register::getTile() const {
  return this->register_file ? register_file->getTile() : nullptr;
}

void Register::setRegisterFile(RegisterFile *register_file) {
  this->register_file = register_file;
}

RegisterFile *Register::getRegisterFile() const { return this->register_file; }

//===----------------------------------------------------------------------===//
// Register File
//===----------------------------------------------------------------------===//

RegisterFile::RegisterFile(int id) { this->id = id; }

int RegisterFile::getId() const { return id; }

Tile *RegisterFile::getTile() const {
  return this->register_file_cluster ? register_file_cluster->getTile()
                                     : nullptr;
}

void RegisterFile::setRegisterFileCluster(
    RegisterFileCluster *register_file_cluster) {
  this->register_file_cluster = register_file_cluster;
}

void RegisterFile::addRegister(Register *reg) {
  registers[reg->getId()] = reg;
  reg->setRegisterFile(this);
}

const std::map<int, Register *> &RegisterFile::getRegisters() const {
  return this->registers;
}

RegisterFileCluster *RegisterFile::getRegisterFileCluster() const {
  return this->register_file_cluster;
}

//===----------------------------------------------------------------------===//
// Register File Cluster
//===----------------------------------------------------------------------===//

RegisterFileCluster::RegisterFileCluster(int id) { this->id = id; }

int RegisterFileCluster::getId() const { return id; }

void RegisterFileCluster::setTile(Tile *tile) { this->tile = tile; }

Tile *RegisterFileCluster::getTile() const { return this->tile; }

void RegisterFileCluster::addRegisterFile(RegisterFile *register_file) {
  register_files[register_file->getId()] = register_file;
  register_file->setRegisterFileCluster(this);
}

const std::map<int, RegisterFile *> &
RegisterFileCluster::getRegisterFiles() const {
  return this->register_files;
}

//===----------------------------------------------------------------------===//
// Architecture
//===----------------------------------------------------------------------===//

// Initializes tiles in the architecture.
void Architecture::initializeTiles(int width, int height) {
  const int num_tiles = width * height;
  tile_storage.reserve(num_tiles);
  
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int id = y * width + x;
      auto tile = std::make_unique<Tile>(id, x, y);
      id_to_tile[id] = tile.get();
      coord_to_tile[{x, y}] = tile.get();
      tile_storage.push_back(std::move(tile));
    }
  }
}

// Creates register file cluster for a tile.
void Architecture::createRegisterFileCluster(Tile *tile, int num_registers, int &reg_id) {
  const int k_num_regs_per_regfile = 8;  // Keep this fixed for now.
  const int k_num_regfiles_per_cluster = num_registers / k_num_regs_per_regfile;
  
  RegisterFileCluster *register_file_cluster = new RegisterFileCluster(tile->getId());

  // Creates registers as a register file.
  for (int file_idx = 0; file_idx < k_num_regfiles_per_cluster; ++file_idx) {
    RegisterFile *register_file = new RegisterFile(file_idx);
    for (int reg_idx = 0; reg_idx < k_num_regs_per_regfile; ++reg_idx) {
      Register *reg = new Register(reg_id++);
      register_file->addRegister(reg);
    }
    register_file_cluster->addRegisterFile(register_file);
  }

  tile->addRegisterFileCluster(register_file_cluster);
}

// Configures default tile settings.
void Architecture::configureDefaultTileSettings(const TileDefaults& tile_defaults) {
  int reg_id = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      Tile *tile = getTile(x, y);
      
      // Creates register file cluster with default capacity.
      createRegisterFileCluster(tile, tile_defaults.num_registers, reg_id);
      
      // Configures function units based on tile_defaults.operations.
      configureTileFunctionUnits(tile, tile_defaults.operations);
      
      // Sets default ports for the tile.
      tile->setPorts(tile_defaults.default_ports);
    }
  }
}

// Recreates register file cluster with new capacity.
void Architecture::recreateRegisterFileCluster(Tile *tile, int num_registers) {
  const int k_num_regs_per_regfile = 8;  // Keep this fixed for now.
  const int k_num_regfiles_per_cluster = num_registers / k_num_regs_per_regfile;
  
  // Remove existing register file cluster.
  if (tile->getRegisterFileCluster()) {
    delete tile->getRegisterFileCluster();
  }
  
  // Creates new register file cluster with override capacity.
  RegisterFileCluster *new_register_file_cluster = 
      new RegisterFileCluster(tile->getId());
  
  // Creates registers with new capacity.
  int reg_id = tile->getId() * 1000;  // Use tile ID as base to avoid conflicts.
  for (int file_idx = 0; file_idx < k_num_regfiles_per_cluster; ++file_idx) {
    RegisterFile *register_file = new RegisterFile(file_idx);
    for (int reg_idx = 0; reg_idx < k_num_regs_per_regfile; ++reg_idx) {
      Register *reg = new Register(reg_id++);
      register_file->addRegister(reg);
    }
    new_register_file_cluster->addRegisterFile(register_file);
  }
  
  // Add new register file cluster to the tile.
  tile->addRegisterFileCluster(new_register_file_cluster);
}

// Applies tile overrides to modify specific tiles.
void Architecture::applyTileOverrides(const std::vector<TileOverride>& tile_overrides) {
  for (const auto &override : tile_overrides) {
    Tile *tile = nullptr;
    if (override.id >= 0) {
      tile = getTile(override.id);
    } else if (override.x >= 0 && override.y >= 0) {
      tile = getTile(override.x, override.y);
    }
    
    if (tile) {
      // Overrides operations if specified.
      if (!override.operations.empty()) {
        configureTileFunctionUnits(tile, override.operations, true);
      }
      
      // Overrides num_registers if specified.
      if (override.num_registers > 0) {
        recreateRegisterFileCluster(tile, override.num_registers);
      }
      
      // Overrides ports if specified.
      if (!override.ports.empty()) {
        tile->setPorts(override.ports);
      }
      
      // Overrides memory capacity if specified.
      if (override.memory.capacity > 0) {
        tile->setMemoryCapacity(override.memory.capacity);
      }
    }
  }
}

// Creates a single link between two tiles.
void Architecture::createSingleLink(int &link_id, Tile *src_tile, Tile *dst_tile, 
                                   const LinkDefaults& link_defaults) {
  auto link = std::make_unique<Link>(link_id++);
  link->setLatency(link_defaults.latency);
  link->setBandwidth(link_defaults.bandwidth);
  link->connect(src_tile, dst_tile);
  link_storage.push_back(std::move(link));
}

// Creates links between tiles based on topology.
void Architecture::createLinks(const LinkDefaults& link_defaults, BaseTopology base_topology) {
  int link_id = 0;
  
  switch (base_topology) {
    case BaseTopology::MESH:
      createMeshLinks(link_id, link_defaults);
      break;
    case BaseTopology::KING_MESH:
      createKingMeshLinks(link_id, link_defaults);
      break;
    case BaseTopology::RING:
      createRingLinks(link_id, link_defaults);
      break;
    default:
      // Default to mesh if unknown topology
      createMeshLinks(link_id, link_defaults);
      break;
  }
}

// Checks if a tile is on the boundary of the architecture.
bool isOnBoundary(int x, int y, int width, int height) {
  return (x == 0 || x == width - 1 || y == 0 || y == height - 1);
}

// Creates a link if the destination tile exists within bounds.
void Architecture::createLinkIfValid(int &link_id, Tile *src_tile, int dst_x, int dst_y, 
                                   const LinkDefaults& link_defaults) {
  if (dst_x >= 0 && dst_x < width && dst_y >= 0 && dst_y < height) {
    createSingleLink(link_id, src_tile, getTile(dst_x, dst_y), link_defaults);
  }
}

// Creates 4-connected mesh links (N, S, W, E).
void Architecture::createMeshLinks(int &link_id, const LinkDefaults& link_defaults) {
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      Tile *tile = getTile(i, j);

      // Creates links to neighboring tiles with default properties.
      createLinkIfValid(link_id, tile, i - 1, j, link_defaults);     // West
      createLinkIfValid(link_id, tile, i + 1, j, link_defaults);     // East
      createLinkIfValid(link_id, tile, i, j - 1, link_defaults);     // South
      createLinkIfValid(link_id, tile, i, j + 1, link_defaults);     // North
    }
  }
}

// Creates 8-connected king mesh links (N, S, W, E, NE, NW, SE, SW).
void Architecture::createKingMeshLinks(int &link_id, const LinkDefaults& link_defaults) {
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      Tile *tile = getTile(i, j);

      // Creates 4-connected links (N, S, W, E).
      createLinkIfValid(link_id, tile, i - 1, j, link_defaults);     // West
      createLinkIfValid(link_id, tile, i + 1, j, link_defaults);     // East
      createLinkIfValid(link_id, tile, i, j - 1, link_defaults);     // South
      createLinkIfValid(link_id, tile, i, j + 1, link_defaults);     // North
      
      // Creates diagonal links for king mesh (NE, NW, SE, SW).
      createLinkIfValid(link_id, tile, i - 1, j - 1, link_defaults); // Southwest
      createLinkIfValid(link_id, tile, i + 1, j - 1, link_defaults); // Southeast
      createLinkIfValid(link_id, tile, i - 1, j + 1, link_defaults); // Northwest
      createLinkIfValid(link_id, tile, i + 1, j + 1, link_defaults); // Northeast
    }
  }
}

// Creates ring topology links (only outer boundary connections).
void Architecture::createRingLinks(int &link_id, const LinkDefaults& link_defaults) {
  // Connect tiles on the outer boundary only.
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      Tile *tile = getTile(i, j);
      
      // Check if tile is on the boundary.
      if (isOnBoundary(i, j, width, height)) {
        // Create connections only to adjacent boundary tiles.
        createLinkIfValid(link_id, tile, i - 1, j, link_defaults);     // West
        createLinkIfValid(link_id, tile, i + 1, j, link_defaults);     // East
        createLinkIfValid(link_id, tile, i, j - 1, link_defaults);     // South
        createLinkIfValid(link_id, tile, i, j + 1, link_defaults);     // North
      }
    }
  }
}

// Checks if a link already exists between two tiles.
bool Architecture::linkExists(Tile *src_tile, Tile *dst_tile) {
  for (const auto &link : link_storage) {
    if (link && link->getSrcTile() == src_tile && link->getDstTile() == dst_tile) {
      return true;
    }
  }
  return false;
}

// Applies link overrides to create, modify, or remove links.
void Architecture::applyLinkOverrides(const std::vector<LinkOverride>& link_overrides) {
  int next_link_id = link_storage.size(); // Start from the next available ID
  
  for (const auto &override : link_overrides) {
    // Handle existing link modifications/removals by ID
    if (override.id >= 0 && override.id < static_cast<int>(link_storage.size())) {
      Link *link = link_storage[override.id].get();
      if (link) {
        if (override.latency > 0) {
          link->setLatency(override.latency);
        }
        
        if (override.bandwidth > 0) {
          link->setBandwidth(override.bandwidth);
        }
        
        if (!override.existence) {
          removeLink(override.id);
        }
      }
    }
    // Handle link creation/removal by tile IDs
    else if (override.src_tile_id >= 0 && override.dst_tile_id >= 0) {
      Tile *src_tile = getTile(override.src_tile_id);
      Tile *dst_tile = getTile(override.dst_tile_id);
      
      if (src_tile && dst_tile) {
        bool link_already_exists = linkExists(src_tile, dst_tile);
        
        if (override.existence && !link_already_exists) {
          // Create new link
          auto link = std::make_unique<Link>(next_link_id++);
          
          // Set link properties
          if (override.latency > 0) {
            link->setLatency(override.latency);
          }
          if (override.bandwidth > 0) {
            link->setBandwidth(override.bandwidth);
          }
          
          // Connect the tiles
          link->connect(src_tile, dst_tile);
          link_storage.push_back(std::move(link));
        } else if (!override.existence && link_already_exists) {
          // Remove existing link
          for (size_t i = 0; i < link_storage.size(); ++i) {
            if (link_storage[i] && 
                link_storage[i]->getSrcTile() == src_tile && 
                link_storage[i]->getDstTile() == dst_tile) {
              removeLink(static_cast<int>(i));
              break;
            }
          }
        }
      }
    }
  }
}

// Main constructor - handles all cases internally.
Architecture::Architecture(int width, int height, 
                          const TileDefaults& tile_defaults,
                          const std::vector<TileOverride>& tile_overrides,
                          const LinkDefaults& link_defaults,
                          const std::vector<LinkOverride>& link_overrides,
                          BaseTopology base_topology) {
  this->width = width;
  this->height = height;

  // Initializes architecture components using helper methods.
  initializeTiles(width, height);
  configureDefaultTileSettings(tile_defaults);
  applyTileOverrides(tile_overrides);
  createLinks(link_defaults, base_topology);
  applyLinkOverrides(link_overrides);
}


Tile *Architecture::getTile(int id) {
  auto it = id_to_tile.find(id);
  assert(it != id_to_tile.end() && "Tile with given ID not found");
  return it->second;
}

Tile *Architecture::getTile(int x, int y) {
  auto it = coord_to_tile.find({x, y});
  assert(it != coord_to_tile.end() && "Tile with given coordinates not found");
  return it->second;
}

std::vector<Tile *> Architecture::getAllTiles() const {
  std::vector<Tile *> result;
  for (auto &tile : tile_storage)
    result.push_back(tile.get());
  return result;
}

int Architecture::getNumTiles() const {
  return static_cast<int>(id_to_tile.size());
}

std::vector<Link *> Architecture::getAllLinks() const {
  std::vector<Link *> all_links;
  for (const auto &link : link_storage) {
    all_links.push_back(link.get());
  }
  return all_links;
}

void Architecture::removeLink(int link_id) {
  if (link_id < 0 || link_id >= static_cast<int>(link_storage.size())) {
    return;
  }
  
  Link *link = link_storage[link_id].get();
  if (!link) {
    return;
  }
  
  Tile *src_tile = link->getSrcTile();
  Tile *dst_tile = link->getDstTile();
  
  if (src_tile && dst_tile) {
    // Remove the link from both tiles' connection sets.
    src_tile->unlinkDstTile(link, dst_tile);
  }
  
  // Marks the link as removed by setting it to null.
  link_storage[link_id].reset();
}
