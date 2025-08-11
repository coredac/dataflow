#include "NeuraDialect/Architecture/Architecture.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

using namespace mlir;
using namespace mlir::neura;

//===----------------------------------------------------------------------===//
// Tile
//===----------------------------------------------------------------------===//

Tile::Tile(int id, int x, int y) {
  this->id = id;
  this->x = x;
  this->y = y;

  // TODO: Add function units based on architecture specs.
  // @Jackcuii, https://github.com/coredac/dataflow/issues/82.
  // addFunctionUnit(std::make_unique<FixedPointAdder>(0));
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

const std::set<Tile *> &Tile::getDstTiles() const { return dst_tiles; }

const std::set<Tile *> &Tile::getSrcTiles() const { return src_tiles; }

const std::set<Link *> &Tile::getOutLinks() const { return out_links; }

const std::set<Link *> &Tile::getInLinks() const { return in_links; }

void Tile::addRegisterFileCluster(RegisterFileCluster* register_file_cluster) {
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

const RegisterFileCluster* Tile::getRegisterFileCluster() const {
  return register_file_cluster;
}

const std::vector<RegisterFile *> Tile::getRegisterFiles() const {
  std::vector<RegisterFile*> all_register_files;
  for (const auto& [id, file] : this->register_file_cluster->getRegisterFiles()) {
    all_register_files.push_back(file);
  }
  return all_register_files;
}

const std::vector<Register *> Tile::getRegisters() const {
  std::vector<Register *> all_registers;
  for (const auto& [reg_file_id, reg_file] : this->register_file_cluster->getRegisterFiles()) {
    for (const auto& [reg_id, reg] : reg_file->getRegisters()) {
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

void FunctionUnit::setTile(Tile* tile) {
  this->tile = tile;
}

Tile *FunctionUnit::getTile() const {
  return this->tile;
}

//===----------------------------------------------------------------------===//
// Register
//===----------------------------------------------------------------------===//

Register::Register(int id) { this->id = id; }

int Register::getId() const { return id; }

Tile *Register::getTile() const {
  return this->register_file ? register_file->getTile() : nullptr;
}

void Register::setRegisterFile(RegisterFile* register_file) {
  this->register_file = register_file;
}

RegisterFile *Register::getRegisterFile() const {
  return this->register_file;
}

//===----------------------------------------------------------------------===//
// Register File
//===----------------------------------------------------------------------===//

RegisterFile::RegisterFile(int id) { this->id = id; }

int RegisterFile::getId() const { return id; }

Tile *RegisterFile::getTile() const {
  return this->register_file_cluster ? register_file_cluster->getTile() : nullptr;
}

void RegisterFile::setRegisterFileCluster(RegisterFileCluster* register_file_cluster) {
  this->register_file_cluster = register_file_cluster;
}

void RegisterFile::addRegister(Register* reg) {
  registers[reg->getId()] = reg;
  reg->setRegisterFile(this);
}

const std::map<int, Register*>& RegisterFile::getRegisters() const {
  return this->registers;
}

RegisterFileCluster* RegisterFile::getRegisterFileCluster() const {
  return this->register_file_cluster;
}

//===----------------------------------------------------------------------===//
// Register File Cluster
//===----------------------------------------------------------------------===//

RegisterFileCluster::RegisterFileCluster(int id) { this->id = id; }

int RegisterFileCluster::getId() const { return id; }

void RegisterFileCluster::setTile(Tile* tile) {
  this->tile = tile;
}

Tile *RegisterFileCluster::getTile() const {
  return this->tile;
}

void RegisterFileCluster::addRegisterFile(RegisterFile* register_file) {
  register_files[register_file->getId()] = register_file;
  register_file->setRegisterFileCluster(this);
}

const std::map<int, RegisterFile*>& RegisterFileCluster::getRegisterFiles() const {
  return this->register_files;
}

//===----------------------------------------------------------------------===//
// Architecture
//===----------------------------------------------------------------------===//

Architecture::Architecture(int width, int height) {
  const int num_tiles = width * height;

  // Initializes tiles.
  tile_storage.reserve(num_tiles);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int id = y * width + x;
      // const int id = x * width + y;
      auto tile = std::make_unique<Tile>(id, x, y);
      id_to_tile[id] = tile.get();
      coord_to_tile[{x, y}] = tile.get();
      tile_storage.push_back(std::move(tile));
    }
  }

  // Initializes register file cluster for each tile.
  int reg_id = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // Gets the tile by coordinates.
      Tile *tile = getTile(x, y);

      // Creates registers as a register file.
      // FIXME: We have to assign different IDs due to the hash function
      // cannot distinguish between different register files..
      Register *register_0 = new Register(reg_id++);
      Register *register_1 = new Register(reg_id++);
      RegisterFile *register_file_0 = new RegisterFile(0);
      register_file_0->addRegister(register_0);
      register_file_0->addRegister(register_1);

      // Creates registers as a register file.
      Register *register_2 = new Register(reg_id++);
      Register *register_3 = new Register(reg_id++);
      RegisterFile *register_file_1 = new RegisterFile(1);
      register_file_1->addRegister(register_2);
      register_file_1->addRegister(register_3);

      // Assembles register files into a cluster.
      RegisterFileCluster *register_file_cluster = new RegisterFileCluster(y * width + x);
      register_file_cluster->addRegisterFile(register_file_0);
      register_file_cluster->addRegisterFile(register_file_1);

      // Adds register file cluster to the tile.
      tile->addRegisterFileCluster(register_file_cluster);
      llvm::errs() << "Tile (" << x << ", " << y
                 << ") added register file cluster with ID: "
                 << register_file_cluster->getId() << "\n";
    }
  }

  // TODO: Model topology based on the architecture specs.
  // https://github.com/coredac/dataflow/issues/52.
  int link_id = 0;
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      // Gets the tile by coordinates.
      Tile *tile = getTile(i, j);

      // Creates links to neighboring tiles.
      if (i > 0) {
        auto link_towards_left = std::make_unique<Link>(link_id++);
        link_towards_left->connect(tile, getTile(i - 1, j));
        link_storage.push_back(std::move(link_towards_left));
      }
      if (i < width - 1) {
        auto link_towards_right = std::make_unique<Link>(link_id++);
        link_towards_right->connect(tile, getTile(i + 1, j));
        link_storage.push_back(std::move(link_towards_right));
      }
      if (j > 0) {
        auto link_towards_down = std::make_unique<Link>(link_id++);
        link_towards_down->connect(tile, getTile(i, j - 1));
        link_storage.push_back(std::move(link_towards_down));
      }
      if (j < height - 1) {
        auto link_towards_up = std::make_unique<Link>(link_id++);
        link_towards_up->connect(tile, getTile(i, j + 1));
        link_storage.push_back(std::move(link_towards_up));
      }
    }
  }
}

Architecture::Architecture(const YAML::Node& config) {
  // Extract width and height from config
  int width = 4;  // default
  int height = 4; // default
  
  if (config["architecture"] && config["architecture"]["width"] && config["architecture"]["height"]) {
    width = config["architecture"]["width"].as<int>();
    height = config["architecture"]["height"].as<int>();
  }
  
  // Call the constructor with width and height.
  *this = Architecture(width, height);

  // Add function units based on the architecture specs.
  int num_tiles = width * height;
  for (int i = 0; i < num_tiles; ++i) {
    Tile *tile = getTile(i);
    int fu_id = 0;
    if (config["tile_overrides"][i]) {
      // Override the default function units.
      for (const auto& operation : config["tile_overrides"][i]["operations"]) {
        if (operation.as<std::string>() == "add") {
          tile->addFunctionUnit(std::make_unique<FixedPointAdder>(fu_id++));
          // Add more function units here if more operations are supported.
        }
      }
    } else if (config["tile_defaults"]) {
      // Add default function units.
      for (const auto& operation : config["tile_defaults"]["operations"]) {
        if (operation.as<std::string>() == "add") {
          tile->addFunctionUnit(std::make_unique<FixedPointAdder>(fu_id++));
        }
      }
    }
  } 
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
