#ifndef NEURA_ARCHITECTURE_H
#define NEURA_ARCHITECTURE_H

#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <optional>
#include <memory>

namespace mlir {
namespace neura {

//===----------------------------------------------------------------------===//
// BasicResource: abstract base class for Tile, Link, etc.
//===----------------------------------------------------------------------===//

class BasicResource {
public:
  virtual ~BasicResource() = default;
  virtual int getId() const = 0;
  virtual std::string getType() const = 0;
};

//===----------------------------------------------------------------------===//
// Forward declaration for use in Tile
class Link;

//===----------------------------------------------------------------------===//
// Tile
//===----------------------------------------------------------------------===//

class Tile : public BasicResource {
public:
  Tile(int id, int x, int y);

  int getId() const override;
  std::string getType() const override { return "tile"; }

  int getX() const;
  int getY() const;

  void linkDstTile(Link* link, Tile* tile);
  const std::set<Tile*>& getDstTiles() const;
  const std::set<Tile*>& getSrcTiles() const;
  const std::set<Link*>& getOutLinks() const;
  const std::set<Link*>& getInLinks() const;

private:
  int id;
  int x, y;
  std::set<Tile*> src_tiles;
  std::set<Tile*> dst_tiles;
  std::set<Link*> in_links;
  std::set<Link*> out_links;
};

//===----------------------------------------------------------------------===//
// Link
//===----------------------------------------------------------------------===//

class Link : public BasicResource {
public:
  Link(int id);

  int getId() const override;
  std::string getType() const override { return "link"; }

  Tile* getSrcTile() const;
  Tile* getDstTile() const;

  void connect(Tile* src, Tile* dst);

private:
  int id;
  Tile* src_tile;
  Tile* dst_tile;
};

struct PairHash {
  std::size_t operator()(const std::pair<int, int> &coord) const {
    return std::hash<int>()(coord.first) ^ (std::hash<int>()(coord.second) << 1);
  }
};

/// Describes the entire CGRA architecture.
class Architecture {
public:
  Architecture(int width, int height);

  Tile* getTile(int id);
  Tile* getTile(int x, int y);

  Link* getLink(int id);

  int getNumTiles() const;
  std::vector<Tile*> getAllTiles() const;
  std::vector<Link*> getAllLinks() const;

private:
  std::vector<std::unique_ptr<Tile>> tile_storage;
//   std::vector<Tile*> tiles;
  std::vector<std::unique_ptr<Link>> link_storage;
  std::unordered_map<int, Tile*> id_to_tile;
  std::unordered_map<std::pair<int, int>, Tile*, PairHash> coord_to_tile;
};

} // namespace neura
} // namespace mlir

#endif // NEURA_ARCHITECTURE_H
