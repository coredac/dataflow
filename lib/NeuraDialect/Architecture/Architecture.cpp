#include "NeuraDialect/Architecture/Architecture.h"
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

  tileStorage.reserve(num_tiles);
  tiles.reserve(num_tiles);

  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < height; ++j) {
      auto tile = std::make_unique<Tile>(i * width + j, i, j);
      tiles.push_back(tile.get());
      tileStorage.push_back(std::move(tile));
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
      }
      if (i < width - 1) {
        auto link_towards_right = std::make_unique<Link>(link_id++);
        link_towards_right->connect(tile, getTile(i + 1, j));
    }
      if (j > 0) {
        auto link_towards_down = std::make_unique<Link>(link_id++);
        link_towards_down->connect(tile, getTile(i, j - 1));
      }
      if (j < height - 1) {
        auto link_towards_up = std::make_unique<Link>(link_id++);
        link_towards_up->connect(tile, getTile(i, j + 1));
      }
    }
  }
}

Tile* Architecture::getTile(int id) {
  for (const auto &tile : tiles) {
    if (tile->getId() == id) {
      return tile;
    }
  }
  assert(false && "Tile with given ID not found");
  return nullptr;
}

Tile* Architecture::getTile(int x, int y) {
  for (const auto &tile : tiles) {
    if (tile->getX() == x && tile->getY() == y) {
      return tile;
    }
  }
  assert(false && "Tile with given coordinates not found");
  return nullptr;
}
