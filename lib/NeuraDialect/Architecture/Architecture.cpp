#include "NeuraDialect/Architecture/Architecture.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

using namespace mlir;
using namespace mlir::neura;

Tile::Tile(int id, int x, int y) {
  this->id = id;
  this->x = x;
  this->y = y;
}

int Tile::getId() const {
  return id;
}

int Tile::getX() const {
  return x;
}

int Tile::getY() const {
  return y;
}

void Tile::linkDstTile(Link* link, Tile* tile) {
  assert(tile && "Cannot link to a null tile");
  dst_tiles.insert(tile);
  out_links.insert(link);
  tile->src_tiles.insert(this);
  tile->in_links.insert(link);
}

const std::set<Tile*>& Tile::getDstTiles() const {
  return dst_tiles;
}

const std::set<Tile*>& Tile::getSrcTiles() const {
  return src_tiles;
}

const std::set<Link*>& Tile::getOutLinks() const {
  return out_links;
}

const std::set<Link*>& Tile::getInLinks() const {
  return in_links;
}

Link::Link(int id) {
  this->id = id;
}

int Link::getId() const {
  return id;
}

Tile* Link::getSrcTile() const {
  return src_tile;
}

Tile* Link::getDstTile() const {
  return dst_tile;
}

void Link::connect(Tile* src, Tile* dst) {
  assert(src && dst && "Cannot connect null tiles");
  src_tile = src;
  dst_tile = dst;
  src->linkDstTile(this, dst);
}

Architecture::Architecture(int width, int height) {
  const int num_tiles = width * height;

  tile_storage.reserve(num_tiles);

  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < height; ++j) {
      const int id = i * width + j;
      auto tile = std::make_unique<Tile>(id, i, j);
      id_to_tile[id] = tile.get();
      coord_to_tile[{i, j}] = tile.get();
      tile_storage.push_back(std::move(tile));
    }
  }

  // TODO: Model topology based on the architecture specs.
  // https://github.com/coredac/dataflow/issues/52.
  int link_id = 0;
  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < height; ++j) {
      Tile* tile = getTile(i, j);
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

Tile* Architecture::getTile(int id) {
  auto it = id_to_tile.find(id);
  assert(it != id_to_tile.end() && "Tile with given ID not found");
  return it->second;
}

Tile* Architecture::getTile(int x, int y) {
  auto it = coord_to_tile.find({x, y});
  assert(it != coord_to_tile.end() && "Tile with given coordinates not found");
  return it->second;
}

std::vector<Tile*> Architecture::getAllTiles() const {
  std::vector<Tile*> result;
  for (auto &tile : tile_storage)
    result.push_back(tile.get());
  return result;
}

int Architecture::getNumTiles() const {
  return static_cast<int>(id_to_tile.size());
}

std::vector<Link*> Architecture::getAllLinks() const {
  std::vector<Link*> all_links;
  for (const auto &link : link_storage) {
    all_links.push_back(link.get());
  }
  return all_links;
}
