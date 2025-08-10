#ifndef NEURA_ARCHITECTURE_H
#define NEURA_ARCHITECTURE_H

#include <cassert>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlir {
namespace neura {

// Enum for identifying resource type.
enum class ResourceKind {
  Tile,
  Link,
  FunctionUnit,
  Register,
  RegisterFile,
  RegisterFileCluster,
};

// Enum for function unit resource type.
enum class FunctionUnitKind {
  FixedPointAdder,
  FixedPointMultiplier,
  CustomizableFunctionUnit,
};

// Enum for supported operation types.
enum OperationKind { IAdd = 0, IMul = 1, FAdd = 2, FMul = 3 };

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
// Forward declaration for use in Tile
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

class FixedPointAdder : public FunctionUnit {
public:
  FixedPointAdder(int id) : FunctionUnit(id) {
    supported_operations.insert(OperationKind::IAdd);
  }
  std::string getType() const override { return "fixed_point_adder"; }
  ResourceKind getKind() const override { return ResourceKind::FunctionUnit; }
};

class FixedPointMultiplier : public FunctionUnit {
public:
  FixedPointMultiplier(int id) : FunctionUnit(id) {
    supported_operations.insert(OperationKind::IMul);
  }
  std::string getType() const override { return "fixed_point_multiplier"; }
  ResourceKind getKind() const override { return ResourceKind::FunctionUnit; }
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
// Tile
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

  bool canSupportOperation(OperationKind operation) const {
    for (FunctionUnit *fu : functional_units) {
      if (fu->canSupportOperation(operation)) {
        return true;
      }
    }
    // TODO: Check if the tile can support the operation based on its
    // capabilities.
    // @Jackcuii, https://github.com/coredac/dataflow/issues/82.
    return true;
  }

  void addRegisterFileCluster(RegisterFileCluster *register_file_cluster);

  const RegisterFileCluster *getRegisterFileCluster() const;

  const std::vector<RegisterFile *> getRegisterFiles() const;

  const std::vector<Register *> getRegisters() const;

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
};

//===----------------------------------------------------------------------===//
// Link
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

private:
  int id;
  Tile *src_tile;
  Tile *dst_tile;
};

//===----------------------------------------------------------------------===//
// Register
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
// Register File
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
// Register File Cluster
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

// Describes the CGRA architecture template.
// TODO: Model architecture in detail (e.g., registers, ports).
class Architecture {
public:
  Architecture(int width, int height);

  Tile *getTile(int id);
  Tile *getTile(int x, int y);

  int getWidth() const { return width; }
  int getHeight() const { return height; }

  Link *getLink(int id);

  int getNumTiles() const;
  std::vector<Tile *> getAllTiles() const;
  std::vector<Link *> getAllLinks() const;

private:
  // TODO: Model architecture in detail, e.g., ports, registers, crossbars, etc.
  // https://github.com/coredac/dataflow/issues/52.
  std::vector<std::unique_ptr<Tile>> tile_storage;
  std::vector<std::unique_ptr<Link>> link_storage;
  std::unordered_map<int, Tile *> id_to_tile;
  std::unordered_map<std::pair<int, int>, Tile *, PairHash> coord_to_tile;

  int width;
  int height;
};

} // namespace neura
} // namespace mlir

#endif // NEURA_ARCHITECTURE_H
