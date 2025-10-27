#ifndef NEURA_ARCHITECTURE_H
#define NEURA_ARCHITECTURE_H

#include <algorithm>
#include <cassert>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "NeuraDialect/Architecture/ArchitectureSpec.h"

namespace mlir {
namespace neura {

// Enumeration for identifying resource type.
enum class ResourceKind {
  Tile,
  Link,
  FunctionUnit,
  Register,
  RegisterFile,
  RegisterFileCluster,
};

// Enumeration for function unit resource type.
enum class FunctionUnitKind {
  FixedPointAdder,
  FixedPointMultiplier,
  CustomizableFunctionUnit,
};

// Enumeration for supported operation types.
enum OperationKind { 
  // Integer arithmetic operations.
  IAdd = 0, IMul = 1, ISub = 2, IDiv = 3, IRem = 4,
  // Floating-point arithmetic operations.
  FAdd = 5, FMul = 6, FSub = 7, FDiv = 8,
  // Memory operations.
  ILoad = 9, IStore = 10, ILoadIndexed = 11, IStoreIndexed = 12, IAlloca = 13,
  // Logical operations.
  IOr = 14, INot = 15, ICmp = 16, FCmp = 17, ISel = 18,
  // Type conversion operations.
  ICast = 19, ISExt = 20, IZExt = 21, IShl = 22,
  // Vector operations.
  VFMul = 23,
  // Fused operations.
  FAddFAdd = 24, FMulFAdd = 25,
  // Control flow operations.
  IReturn = 26, IPhi = 27,
  // Data movement operations.
  IDataMov = 28, ICtrlMov = 29,
  // Predicate operations.
  IReserve = 30, IGrantPredicate = 31, IGrantOnce = 32, IGrantAlways = 33,
  // Loop control operations.
  ILoopControl = 34,
  // Constant operations.
  IConstant = 35,
  // Steering control fused operations.
  ICarryInvariant = 36, IConditionalSelect = 37, IInvariantGroup = 38
};

//===----------------------------------------------------------------------===//
// BasicResource: abstract base class for Tile, Link, etc.
//===----------------------------------------------------------------------===//

class BasicResource {
public:
  virtual ~BasicResource() = default;
  virtual int getId() const = 0;
  virtual std::string getType() const = 0;
  virtual ResourceKind getKind() const = 0;
};

//===----------------------------------------------------------------------===//
// Forward declaration for use in Tile.
class Tile;
class Link;
class FunctionUnit;
class Register;
class RegisterFile;
class RegisterFileCluster;

//===----------------------------------------------------------------------===//
// Function Unit.
//===----------------------------------------------------------------------===//

class FunctionUnit : public BasicResource {
public:
  FunctionUnit(int id);

  int getId() const override;
  std::string getType() const override { return "function_unit"; }
  ResourceKind getKind() const override { return ResourceKind::FunctionUnit; }

  static bool classof(const BasicResource *res) {
    return res && res->getKind() == ResourceKind::FunctionUnit;
  }

  Tile *getTile() const;

  void setTile(Tile *tile);

  std::set<OperationKind> getSupportedOperations() const {
    return supported_operations;
  }

  bool canSupportOperation(OperationKind operation) const {
    for (const auto &op : supported_operations) {
      if (op == operation) {
        return true;
      }
    }
    return false;
  }

protected:
  std::set<OperationKind> supported_operations;

private:
  int id;
  Tile *tile;
};

class CustomizableFunctionUnit : public FunctionUnit {
public:
  CustomizableFunctionUnit(int id) : FunctionUnit(id) {}
  std::string getType() const override { return "customizable_function_unit"; }
  ResourceKind getKind() const override { return ResourceKind::FunctionUnit; }
  void addSupportedOperation(OperationKind operation_kind) {
    supported_operations.insert(operation_kind);
  }
};

//===----------------------------------------------------------------------===//
// Tile.
//===----------------------------------------------------------------------===//

class Tile : public BasicResource {
public:
  Tile(int id, int x, int y);

  int getId() const override;
  std::string getType() const override { return "tile"; }

  ResourceKind getKind() const override { return ResourceKind::Tile; }

  static bool classof(const BasicResource *res) {
    return res && res->getKind() == ResourceKind::Tile;
  }

  int getX() const;
  int getY() const;

  void linkDstTile(Link *link, Tile *tile);
  void unlinkDstTile(Link *link, Tile *tile);
  const std::set<Tile *> &getDstTiles() const;
  const std::set<Tile *> &getSrcTiles() const;
  const std::set<Link *> &getOutLinks() const;
  const std::set<Link *> &getInLinks() const;

  void addFunctionUnit(std::unique_ptr<FunctionUnit> func_unit) {
    assert(func_unit && "Cannot add null function unit");
    func_unit->setTile(this);
    functional_unit_storage.push_back(std::move(func_unit));
    functional_units.insert(functional_unit_storage.back().get());
  }

  void clearFunctionUnits() {
    functional_unit_storage.clear();
    functional_units.clear();
  }

  bool canSupportOperation(OperationKind operation) const {
    // Checks if any function unit in this tile supports the operation.
    // The implementation checks all functional units in the tile.
    for (FunctionUnit *fu : functional_units) {
      if (fu->canSupportOperation(operation)) {
        return true;
      }
    }
    return false;
  }

  void addRegisterFileCluster(RegisterFileCluster *register_file_cluster);

  const RegisterFileCluster *getRegisterFileCluster() const;

  const std::vector<RegisterFile *> getRegisterFiles() const;

  const std::vector<Register *> getRegisters() const;

  // Port management.
  const std::vector<std::string> &getPorts() const { return ports; }
  void setPorts(const std::vector<std::string> &new_ports) { ports = new_ports; }
  bool hasPort(const std::string &port) const {
    return std::find(ports.begin(), ports.end(), port) != ports.end();
  }

  // Memory management.
  int getMemoryCapacity() const { return memory_capacity; }
  void setMemoryCapacity(int capacity) { memory_capacity = capacity; }

private:
  int id;
  int x, y;
  std::set<Tile *> src_tiles;
  std::set<Tile *> dst_tiles;
  std::set<Link *> in_links;
  std::set<Link *> out_links;
  std::vector<std::unique_ptr<FunctionUnit>>
      functional_unit_storage;               // Owns FUs.
  std::set<FunctionUnit *> functional_units; // Non-owning, for fast lookup.
  RegisterFileCluster *register_file_cluster = nullptr;
  
  // Port and memory configuration.
  std::vector<std::string> ports;
  int memory_capacity = -1;  // -1 means not configured.
};

//===----------------------------------------------------------------------===//
// Link.
//===----------------------------------------------------------------------===//

class Link : public BasicResource {
public:
  Link(int id);

  int getId() const override;

  std::string getType() const override { return "link"; }

  ResourceKind getKind() const override { return ResourceKind::Link; }

  static bool classof(const BasicResource *res) {
    return res && res->getKind() == ResourceKind::Link;
  }
  Tile *getSrcTile() const;
  Tile *getDstTile() const;

  void connect(Tile *src, Tile *dst);

  // Link properties.
  int getLatency() const { return latency; }
  int getBandwidth() const { return bandwidth; }
  void setLatency(int l) { latency = l; }
  void setBandwidth(int b) { bandwidth = b; }

private:
  int id;
  Tile *src_tile;
  Tile *dst_tile;
  int latency = 1;        // Latency in cycles.
  int bandwidth = 32;     // Bandwidth in bits per cycle.
};

//===----------------------------------------------------------------------===//
// Register.
//===----------------------------------------------------------------------===//

class Register : public BasicResource {
public:
  Register(int id);

  int getId() const override;

  std::string getType() const override { return "register"; }

  ResourceKind getKind() const override { return ResourceKind::Register; }

  static bool classof(const BasicResource *res) {
    return res && res->getKind() == ResourceKind::Register;
  }

  Tile *getTile() const;

  void setRegisterFile(RegisterFile *register_file);

  RegisterFile *getRegisterFile() const;

private:
  int id;
  RegisterFile *register_file;
};

//===----------------------------------------------------------------------===//
// Register File.
//===----------------------------------------------------------------------===//

class RegisterFile : public BasicResource {
public:
  RegisterFile(int id);

  int getId() const override;

  std::string getType() const override { return "register_file"; }

  ResourceKind getKind() const override { return ResourceKind::RegisterFile; }

  static bool classof(const BasicResource *res) {
    return res && res->getKind() == ResourceKind::RegisterFile;
  }

  Tile *getTile() const;

  void setRegisterFileCluster(RegisterFileCluster *register_file_cluster);

  void addRegister(Register *reg);

  const std::map<int, Register *> &getRegisters() const;
  RegisterFileCluster *getRegisterFileCluster() const;

private:
  int id;
  std::map<int, Register *> registers;
  RegisterFileCluster *register_file_cluster = nullptr;
};

//===----------------------------------------------------------------------===//
// Register File Cluster.
//===----------------------------------------------------------------------===//

class RegisterFileCluster : public BasicResource {
public:
  RegisterFileCluster(int id);
  int getId() const override;

  std::string getType() const override { return "register_file_cluster"; }

  ResourceKind getKind() const override {
    return ResourceKind::RegisterFileCluster;
  }

  static bool classof(const BasicResource *res) {
    return res && res->getKind() == ResourceKind::RegisterFileCluster;
  }

  Tile *getTile() const;
  void setTile(Tile *tile);

  void addRegisterFile(RegisterFile *register_file);
  const std::map<int, RegisterFile *> &getRegisterFiles() const;

private:
  int id;
  Tile *tile;
  std::map<int, RegisterFile *> register_files;
};

//===----------------------------------------------------------------------===//

struct PairHash {
  std::size_t operator()(const std::pair<int, int> &coord) const {
    return std::hash<int>()(coord.first) ^
           (std::hash<int>()(coord.second) << 1);
  }
};

// Forward declaration.
struct TileDefaults;
struct TileOverride;
struct LinkDefaults;
struct LinkOverride;

// Describes the CGRA architecture template.
// Now supports comprehensive configuration via YAML including ports, memory, and function units.
class Architecture {
public:
  // Single constructor - handles all cases internally.
  Architecture(int multi_cgra_rows,
               int multi_cgra_columns,
               BaseTopology multi_cgra_base_topology = BaseTopology::MESH,
               int per_cgra_rows = 4,
               int per_cgra_columns = 4,
               BaseTopology per_cgra_base_topology = BaseTopology::MESH,
               const TileDefaults& tile_defaults = TileDefaults(),
               const std::vector<TileOverride>& tile_overrides = std::vector<TileOverride>(),
               const LinkDefaults& link_defaults = LinkDefaults(),
               const std::vector<LinkOverride>& link_overrides = std::vector<LinkOverride>());

  Tile *getTile(int id);
  Tile *getTile(int x, int y);

  int getMultiCgraRows() const { return multi_cgra_rows_; }
  int getMultiCgraColumns() const { return multi_cgra_columns_; }
  int getPerCgraRows() const { return per_cgra_rows_; }
  int getPerCgraColumns() const { return per_cgra_columns_; }

  Link *getLink(int id);
  Link *getLink(int src_tile_x, int src_tile_y, int dst_tile_x, int dst_tile_y);
  void removeLink(int link_id);
  void removeLink(Tile *src_tile, Tile *dst_tile);
  void removeLink(int src_tile_x, int src_tile_y, int dst_tile_x, int dst_tile_y);

  // Tile management.
  void removeTile(int tile_id);

  int getNumTiles() const;
  std::vector<Tile *> getAllTiles() const;
  std::vector<Link *> getAllLinks() const;

private:
  // Helper methods for constructor initialization.
  void initializeTiles(int rows, int columns);
  void configureDefaultTileSettings(const TileDefaults& tile_defaults);
  void applyTileOverrides(const std::vector<TileOverride>& tile_overrides);
  void createLinks(const LinkDefaults& link_defaults, BaseTopology base_topology);
  void applyLinkOverrides(const std::vector<LinkOverride>& link_overrides);
  void createRegisterFileCluster(Tile *tile, int num_registers, int &reg_id);
  void recreateRegisterFileCluster(Tile *tile, int num_registers);
  bool linkExists(Tile *src_tile, Tile *dst_tile);
  
  // Helper methods for creating different topology links.
  void createSingleLink(int &link_id, Tile *src_tile, Tile *dst_tile, const LinkDefaults& link_defaults);
  void createLinkIfValid(int &link_id, Tile *src_tile, int dst_x, int dst_y, const LinkDefaults& link_defaults);
  void createMeshLinks(int &link_id, const LinkDefaults& link_defaults);
  void createKingMeshLinks(int &link_id, const LinkDefaults& link_defaults);
  void createRingLinks(int &link_id, const LinkDefaults& link_defaults);

  // Architecture components: tiles, links, and their mappings.
  // Ports and memory are now modeled as part of Tile class.
  std::map<int, std::unique_ptr<Tile>> tile_storage_;  // Owns tiles, key is unique tile_id.
  std::map<int, std::unique_ptr<Link>> link_storage_;  // Owns links, key is unique link_id.
  std::unordered_map<int, Tile *> id_to_tile_;  // Maps unique tile_id to Tile pointer.
  std::unordered_map<std::pair<int, int>, Tile *, PairHash> coord_to_tile_;  // Maps (x,y) coordinates to Tile pointer.

  int multi_cgra_rows_;
  int multi_cgra_columns_;
  int per_cgra_rows_;
  int per_cgra_columns_;
};

} // namespace neura
} // namespace mlir

#endif // NEURA_ARCHITECTURE_H
