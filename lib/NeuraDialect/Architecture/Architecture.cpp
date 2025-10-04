#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Architecture/ArchitectureSpec.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace mlir;
using namespace mlir::neura;

// Helper function to configure arithmetic operations.
void configureArithmeticOperations(CustomizableFunctionUnit *functionUnit, const std::string &operation) {
  if (operation == "add") {
    functionUnit->addSupportedOperation(IAdd);
  } else if (operation == "sub") {
    functionUnit->addSupportedOperation(ISub);
  } else if (operation == "mul") {
    functionUnit->addSupportedOperation(IMul);
  } else if (operation == "div") {
    functionUnit->addSupportedOperation(IDiv);
  } else if (operation == "rem") {
    functionUnit->addSupportedOperation(IRem);
  } else if (operation == "fadd") {
    functionUnit->addSupportedOperation(FAdd);
  } else if (operation == "fsub") {
    functionUnit->addSupportedOperation(FSub);
  } else if (operation == "fmul") {
    functionUnit->addSupportedOperation(FMul);
  } else if (operation == "fdiv") {
    functionUnit->addSupportedOperation(FDiv);
  }
}

// Helper function to configure memory operations.
void configureMemoryOperations(CustomizableFunctionUnit *functionUnit, const std::string &operation) {
  if (operation == "load") {
    functionUnit->addSupportedOperation(ILoad);
  } else if (operation == "store") {
    functionUnit->addSupportedOperation(IStore);
  } else if (operation == "load_indexed") {
    functionUnit->addSupportedOperation(ILoadIndexed);
  } else if (operation == "store_indexed") {
    functionUnit->addSupportedOperation(IStoreIndexed);
  } else if (operation == "alloca") {
    functionUnit->addSupportedOperation(IAlloca);
  }
}

// Helper function to configure logical operations.
void configureLogicalOperations(CustomizableFunctionUnit *functionUnit, const std::string &operation) {
  if (operation == "or") {
    functionUnit->addSupportedOperation(IOr);
  } else if (operation == "not") {
    functionUnit->addSupportedOperation(INot);
  } else if (operation == "icmp") {
    functionUnit->addSupportedOperation(ICmp);
  } else if (operation == "fcmp") {
    functionUnit->addSupportedOperation(FCmp);
  } else if (operation == "sel") {
    functionUnit->addSupportedOperation(ISel);
  }
}

// Helper function to configure type conversion operations.
void configureTypeConversionOperations(CustomizableFunctionUnit *functionUnit, const std::string &operation) {
  if (operation == "cast") {
    functionUnit->addSupportedOperation(ICast);
  } else if (operation == "sext") {
    functionUnit->addSupportedOperation(ISExt);
  } else if (operation == "zext") {
    functionUnit->addSupportedOperation(IZExt);
  } else if (operation == "shl") {
    functionUnit->addSupportedOperation(IShl);
  }
}

// Helper function to configure specialized operations.
void configureSpecializedOperations(CustomizableFunctionUnit *functionUnit, const std::string &operation) {
  if (operation == "vfmul") {
    functionUnit->addSupportedOperation(VFMul);
  } else if (operation == "fadd_fadd") {
    functionUnit->addSupportedOperation(FAddFAdd);
  } else if (operation == "fmul_fadd") {
    functionUnit->addSupportedOperation(FMulFAdd);
  } else if (operation == "return") {
    functionUnit->addSupportedOperation(IReturn);
  } else if (operation == "phi") {
    functionUnit->addSupportedOperation(IPhi);
  } else if (operation == "data_mov") {
    functionUnit->addSupportedOperation(IDataMov);
  } else if (operation == "ctrl_mov") {
    functionUnit->addSupportedOperation(ICtrlMov);
  } else if (operation == "reserve") {
    functionUnit->addSupportedOperation(IReserve);
  } else if (operation == "grant_predicate") {
    functionUnit->addSupportedOperation(IGrantPredicate);
  } else if (operation == "grant_once") {
    functionUnit->addSupportedOperation(IGrantOnce);
  } else if (operation == "grant_always") {
    functionUnit->addSupportedOperation(IGrantAlways);
  } else if (operation == "loop_control") {
    functionUnit->addSupportedOperation(ILoopControl);
  } else if (operation == "constant") {
    functionUnit->addSupportedOperation(IConstant);
  }
}

// Helper function to create a function unit for a specific operation.
// Maps YAML operation names to OperationKind enum values and creates appropriate function units.
void createFunctionUnitForOperation(Tile *tile, const std::string &operation, int &functionUnitId) {
  auto functionUnit = std::make_unique<CustomizableFunctionUnit>(functionUnitId++);
  
  // Configures different types of operations using helper functions.
  configureArithmeticOperations(functionUnit.get(), operation);
  configureMemoryOperations(functionUnit.get(), operation);
  configureLogicalOperations(functionUnit.get(), operation);
  configureTypeConversionOperations(functionUnit.get(), operation);
  configureSpecializedOperations(functionUnit.get(), operation);
  
  // TODO: Add support for unknown operations with warning instead of silent failure.
  // This would help users identify typos in their YAML configuration.
  
  tile->addFunctionUnit(std::move(functionUnit));
}

// Helper function to configure tile function units based on operations.
void configureTileFunctionUnits(Tile *tile, const std::vector<std::string> &operations, bool clearExisting = true) {
  // Configures function units based on YAML operations specification.
  // If clearExisting is true, this replaces any existing function units with the specified ones.
  
  if (clearExisting) {
    tile->clearFunctionUnits();
  }
  
  int functionUnitId = 0;
  for (const auto &operation : operations) {
    createFunctionUnitForOperation(tile, operation, functionUnitId);
  }
}

//===----------------------------------------------------------------------===//
// Tile
//===----------------------------------------------------------------------===//

Tile::Tile(int id, int x, int y) {
  this->id = id;
  this->x = x;
  this->y = y;

  // Function units are now configured via YAML specification in Architecture constructor.
  // The old hardcoded FixedPointAdder has been replaced with configurable function units.
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
  }
  assert(this->register_file_cluster == nullptr &&
         "Register file cluster already exists");
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
  const int kNUM_REGS_PER_REGFILE = 8;  // Keep this fixed for now
  const int kNUM_REGFILES_PER_CLUSTER = num_registers / kNUM_REGS_PER_REGFILE;
  
  RegisterFileCluster *register_file_cluster = new RegisterFileCluster(tile->getId());

  // Createss registers as a register file.
  // FIXME: We have to assign different IDs due to the hash function
  // cannot distinguish between different register files..
  for (int file_idx = 0; file_idx < kNUM_REGFILES_PER_CLUSTER; ++file_idx) {
    RegisterFile *register_file = new RegisterFile(file_idx);
    for (int reg_idx = 0; reg_idx < kNUM_REGS_PER_REGFILE; ++reg_idx) {
      Register *reg = new Register(reg_id++);
      register_file->addRegister(reg);
    }
    register_file_cluster->addRegisterFile(register_file);
  }

  tile->addRegisterFileCluster(register_file_cluster);
}

// Helper method to configure default tile settings.
void Architecture::configureDefaultTileSettings(const TileDefaults& tileDefaults) {
  int reg_id = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      Tile *tile = getTile(x, y);
      
      // Creates register file cluster with default capacity.
      createRegisterFileCluster(tile, tileDefaults.num_registers, reg_id);
      
      // Configures function units based on tileDefaults.operations.
      configureTileFunctionUnits(tile, tileDefaults.operations);
      
      // Sets default ports for the tile.
      tile->setPorts(tileDefaults.default_ports);
    }
  }
}

// Helper method to recreate register file cluster with new capacity.
void Architecture::recreateRegisterFileCluster(Tile *tile, int num_registers) {
  const int kNUM_REGS_PER_REGFILE = 8;  // Keep this fixed for now
  const int kNUM_REGFILES_PER_CLUSTER = num_registers / kNUM_REGS_PER_REGFILE;
  
  // Remove existing register file cluster.
  if (tile->getRegisterFileCluster()) {
    delete tile->getRegisterFileCluster();
  }
  
  // Creates new register file cluster with override capacity.
  RegisterFileCluster *new_register_file_cluster = 
      new RegisterFileCluster(tile->getId());
  
  // Creates registers with new capacity.
  int reg_id = tile->getId() * 1000;  // Use tile ID as base to avoid conflicts
  for (int file_idx = 0; file_idx < kNUM_REGFILES_PER_CLUSTER; ++file_idx) {
    RegisterFile *register_file = new RegisterFile(file_idx);
    for (int reg_idx = 0; reg_idx < kNUM_REGS_PER_REGFILE; ++reg_idx) {
      Register *reg = new Register(reg_id++);
      register_file->addRegister(reg);
    }
    new_register_file_cluster->addRegisterFile(register_file);
  }
  
  // Add new register file cluster to the tile.
  tile->addRegisterFileCluster(new_register_file_cluster);
}

// Helper method to apply tile overrides.
void Architecture::applyTileOverrides(const std::vector<TileOverride>& tileOverrides) {
  for (const auto &override : tileOverrides) {
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

// Helper method to create links between tiles.
void Architecture::createLinks(const LinkDefaults& linkDefaults) {
  int link_id = 0;
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      Tile *tile = getTile(i, j);

      // Createss links to neighboring tiles with default properties.
      if (i > 0) {
        auto link_towards_left = std::make_unique<Link>(link_id++);
        link_towards_left->setLatency(linkDefaults.latency);
        link_towards_left->setBandwidth(linkDefaults.bandwidth);
        link_towards_left->connect(tile, getTile(i - 1, j));
        link_storage.push_back(std::move(link_towards_left));
      }
      if (i < width - 1) {
        auto link_towards_right = std::make_unique<Link>(link_id++);
        link_towards_right->setLatency(linkDefaults.latency);
        link_towards_right->setBandwidth(linkDefaults.bandwidth);
        link_towards_right->connect(tile, getTile(i + 1, j));
        link_storage.push_back(std::move(link_towards_right));
      }
      if (j > 0) {
        auto link_towards_down = std::make_unique<Link>(link_id++);
        link_towards_down->setLatency(linkDefaults.latency);
        link_towards_down->setBandwidth(linkDefaults.bandwidth);
        link_towards_down->connect(tile, getTile(i, j - 1));
        link_storage.push_back(std::move(link_towards_down));
      }
      if (j < height - 1) {
        auto link_towards_up = std::make_unique<Link>(link_id++);
        link_towards_up->setLatency(linkDefaults.latency);
        link_towards_up->setBandwidth(linkDefaults.bandwidth);
        link_towards_up->connect(tile, getTile(i, j + 1));
        link_storage.push_back(std::move(link_towards_up));
      }
    }
  }
}

// Helper method to apply link overrides.
void Architecture::applyLinkOverrides(const std::vector<LinkOverride>& linkOverrides) {
  for (const auto &override : linkOverrides) {
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
                          const TileDefaults& tileDefaults,
                          const std::vector<TileOverride>& tileOverrides,
                          const LinkDefaults& linkDefaults,
                          const std::vector<LinkOverride>& linkOverrides) {
  this->width = width;
  this->height = height;

  // Initializes architecture components using helper methods.
  initializeTiles(width, height);
  configureDefaultTileSettings(tileDefaults);
  applyTileOverrides(tileOverrides);
  createLinks(linkDefaults);
  applyLinkOverrides(linkOverrides);
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

void Architecture::removeLink(int linkId) {
  if (linkId < 0 || linkId >= static_cast<int>(link_storage.size())) {
    return;
  }
  
  Link *link = link_storage[linkId].get();
  if (!link) {
    return;
  }
  
  Tile *srcTile = link->getSrcTile();
  Tile *dstTile = link->getDstTile();
  
  if (srcTile && dstTile) {
    // Remove the link from both tiles' connection sets
    srcTile->unlinkDstTile(link, dstTile);
  }
  
  // Marks the link as removed by setting it to null
  link_storage[linkId].reset();
}

