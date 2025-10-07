#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Architecture/ArchitectureSpec.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace mlir;
using namespace mlir::neura;

// Helper function to configure all supported operations.
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

// Helper function to create a function unit for a specific operation.
// Maps YAML operation names to OperationKind enum values and creates appropriate function units.
void createFunctionUnitForOperation(Tile *tile, const std::string &operation, int &function_unit_id) {
  auto function_unit = std::make_unique<CustomizableFunctionUnit>(function_unit_id++);
  
  // Configures all supported operations using the unified function.
  configureSupportedOperations(function_unit.get(), operation);
  
  // TODO: Add support for unknown operations with warning instead of silent failure.
  // This would help users identify typos in their YAML configuration.
  
  tile->addFunctionUnit(std::move(function_unit));
}

// Helper function to configure tile function units based on operations.
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
  for (const auto &[id, file] :
       this->register_file_cluster->getRegisterFiles()) {
    all_register_files.push_back(file);
  }
  return all_register_files;
}

const std::vector<Register *> Tile::getRegisters() const {
  std::vector<Register *> all_registers;
  for (const auto &[reg_file_id, reg_file] :
       this->register_file_cluster->getRegisterFiles()) {
    for (const auto &[reg_id, reg] : reg_file->getRegisters()) {
      all_registers.push_back(reg);
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

// Helper method to initialize tiles in the architecture.
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

// Helper method to create register file cluster for a tile.
void Architecture::createRegisterFileCluster(Tile *tile, int num_registers, int &reg_id) {
  const int k_num_regs_per_regfile = 8;  // Keep this fixed for now.
  const int k_num_regfiles_per_cluster = num_registers / k_num_regs_per_regfile;
  
  RegisterFileCluster *register_file_cluster = new RegisterFileCluster(tile->getId());

  // Creates registers as a register file.
  // FIXME: We have to assign different IDs due to the hash function
  // cannot distinguish between different register files.
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

// Helper method to configure default tile settings.
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

// Helper method to recreate register file cluster with new capacity.
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

// Helper method to apply tile overrides.
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
        removeUnsupportedLinks(tile);
      }
      
      // Overrides memory capacity if specified.
      if (override.memory.capacity > 0) {
        tile->setMemoryCapacity(override.memory.capacity);
      }
    }
  }
}

// Helper method to create links between tiles.
void Architecture::createLinks(const LinkDefaults& link_defaults) {
  int link_id = 0;
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      Tile *tile = getTile(i, j);

      // Creates links to neighboring tiles with default properties.
      if (i > 0) {
        auto link_towards_left = std::make_unique<Link>(link_id++);
        link_towards_left->setLatency(link_defaults.latency);
        link_towards_left->setBandwidth(link_defaults.bandwidth);
        link_towards_left->connect(tile, getTile(i - 1, j));
        link_storage.push_back(std::move(link_towards_left));
      }
      if (i < width - 1) {
        auto link_towards_right = std::make_unique<Link>(link_id++);
        link_towards_right->setLatency(link_defaults.latency);
        link_towards_right->setBandwidth(link_defaults.bandwidth);
        link_towards_right->connect(tile, getTile(i + 1, j));
        link_storage.push_back(std::move(link_towards_right));
      }
      if (j > 0) {
        auto link_towards_down = std::make_unique<Link>(link_id++);
        link_towards_down->setLatency(link_defaults.latency);
        link_towards_down->setBandwidth(link_defaults.bandwidth);
        link_towards_down->connect(tile, getTile(i, j - 1));
        link_storage.push_back(std::move(link_towards_down));
      }
      if (j < height - 1) {
        auto link_towards_up = std::make_unique<Link>(link_id++);
        link_towards_up->setLatency(link_defaults.latency);
        link_towards_up->setBandwidth(link_defaults.bandwidth);
        link_towards_up->connect(tile, getTile(i, j + 1));
        link_storage.push_back(std::move(link_towards_up));
      }
    }
  }
}

// Helper method to apply link overrides.
void Architecture::applyLinkOverrides(const std::vector<LinkOverride>& link_overrides) {
  for (const auto &override : link_overrides) {
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
  }
}

// Main constructor - handles all cases internally.
Architecture::Architecture(int width, int height, 
                          const TileDefaults& tile_defaults,
                          const std::vector<TileOverride>& tile_overrides,
                          const LinkDefaults& link_defaults,
                          const std::vector<LinkOverride>& link_overrides) {
  this->width = width;
  this->height = height;

  // Initializes architecture components using helper methods.
  initializeTiles(width, height);
  configureDefaultTileSettings(tile_defaults);
  applyTileOverrides(tile_overrides);
  createLinks(link_defaults);
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

// Helper method to remove links for directions not supported by the tile.
void Architecture::removeUnsupportedLinks(Tile *tile) {
  int x = tile->getX();
  int y = tile->getY();
  
  // Checks each direction and removes links if tile doesn't have the corresponding port.
  
  // Checks West direction.
  if (!tile->hasPort("W") && x > 0) {
    // Finds the link ID from this tile to the west tile.
    for (size_t i = 0; i < link_storage.size(); ++i) {
      if (link_storage[i] && 
          link_storage[i]->getSrcTile() == tile && 
          link_storage[i]->getDstTile() == getTile(x - 1, y)) {
        removeLink(i);
        break;
      }
    }
  }
  
  // Checks East direction.
  if (!tile->hasPort("E") && x < width - 1) {
    // Finds the link ID from this tile to the east tile.
    for (size_t i = 0; i < link_storage.size(); ++i) {
      if (link_storage[i] && 
          link_storage[i]->getSrcTile() == tile && 
          link_storage[i]->getDstTile() == getTile(x + 1, y)) {
        removeLink(i);
        break;
      }
    }
  }
  
  // Checks South direction.
  if (!tile->hasPort("S") && y > 0) {
    // Finds the link ID from this tile to the south tile.
    for (size_t i = 0; i < link_storage.size(); ++i) {
      if (link_storage[i] && 
          link_storage[i]->getSrcTile() == tile && 
          link_storage[i]->getDstTile() == getTile(x, y - 1)) {
        removeLink(i);
        break;
      }
    }
  }
  
  // Checks North direction.
  if (!tile->hasPort("N") && y < height - 1) {
    // Finds the link ID from this tile to the north tile.
    for (size_t i = 0; i < link_storage.size(); ++i) {
      if (link_storage[i] && 
          link_storage[i]->getSrcTile() == tile && 
          link_storage[i]->getDstTile() == getTile(x, y + 1)) {
        removeLink(i);
        break;
      }
    }
  }
}

