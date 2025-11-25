#ifndef NEURA_ARCHITECTURE_SPEC_H
#define NEURA_ARCHITECTURE_SPEC_H

#include <string>
#include <vector>

namespace mlir {
namespace neura {

// Enumeration for base topology types.
enum class BaseTopology {
  MESH,      // 4-connected mesh (N, S, W, E).
  KING_MESH, // 8-connected mesh (N, S, W, E, NE, NW, SE, SW).
  RING       // Ring topology (only outer boundary connections).
};

// Structure for holding tile default configuration.
struct TileDefaults {
  int num_registers = 64;  // default value.
  std::vector<std::string> operations = {
    "add",
    "mul",
    "sub",
    "div",
    "rem", 
    "fadd",
    "fmul",
    "fsub",
    "fdiv", 
    "or",
    "not",
    "icmp",
    "fcmp",
    "sel",
    "cast",
    "sext",
    "zext",
    "shl",
    "vfmul",
    "fadd_fadd",
    "fmul_fadd",
    "data_mov",
    "ctrl_mov",
    "reserve",
    "grant_predicate",
    "grant_once",
    "grant_always",
    "loop_control",
    "phi",
    "constant",
    "load",
    "store",
    "return",
    "load_indexed",
    "store_indexed",
    "alloca"
  };  // default operations - includes all supported operations for newbie convenience.
};

// Structure for holding memory configuration.
struct MemoryConfig {
  int capacity = 1024;  // Memory capacity in bytes.
  int banks_per_cgra = 4;  // Number of banks per CGRA.
};

// Structure for holding tile override configuration.
struct TileOverride {
  // CGRA coordinates.
  int cgra_x = -1;
  int cgra_y = -1;
  // Tile coordinates within per-CGRA.
  int tile_x = -1;
  int tile_y = -1;
  std::vector<std::string> operations;
  int num_registers = 4;
  bool existence = true;
};

// Structure for holding link default configuration.
struct LinkDefaults {
  int latency = 1;  // default latency.
  int bandwidth = 32;  // default bandwidth.
};

// Structure for holding link override configuration.
struct LinkOverride {
  int latency = 1;
  int bandwidth = -1;
  int src_cgra_x = -1;
  int src_cgra_y = -1;
  int dst_cgra_x = -1;
  int dst_cgra_y = -1;
  int src_tile_x = -1;
  int src_tile_y = -1;
  int dst_tile_x = -1;
  int dst_tile_y = -1;
  bool existence = true;
};

// Function for getting the architecture specification file path.
// This is set by the command line tool when a YAML file is provided.
std::string getArchitectureSpecFile();

// Function for getting tile defaults configuration.
TileDefaults getTileDefaults();

} // namespace neura
} // namespace mlir

#endif // NEURA_ARCHITECTURE_SPEC_H
