#ifndef NEURA_ARCHITECTURE_SPEC_H
#define NEURA_ARCHITECTURE_SPEC_H

#include <string>
#include <vector>

namespace mlir {
namespace neura {

// Structure to hold tile default configuration
struct TileDefaults {
  int num_registers = 64;  // default value
  std::vector<std::string> default_ports = {"N", "S", "W", "E"};  // default ports
  std::vector<std::string> operations = {"add", "mul", "sub"};  // default operations
};

// Structure to hold memory configuration
struct MemoryConfig {
  int capacity = -1;  // Memory capacity in bytes
};

// Structure to hold tile override configuration
struct TileOverride {
  int id = -1;
  int x = -1, y = -1;
  std::vector<std::string> operations;
  int num_registers = -1;
  std::vector<std::string> ports;
  MemoryConfig memory;
};

// Structure to hold link default configuration
struct LinkDefaults {
  int latency = 1;  // default latency
  int bandwidth = 32;  // default bandwidth
};

// Structure to hold link override configuration
struct LinkOverride {
  int id = -1;
  int latency = -1;
  int bandwidth = -1;
  int src_tile_id = -1;
  int dst_tile_id = -1;
  bool existence = true;
};

// Function to get the architecture specification file path
// This is set by the command line tool when a YAML file is provided
std::string getArchitectureSpecFile();

// Function to get tile defaults configuration
TileDefaults getTileDefaults();

} // namespace neura
} // namespace mlir

#endif // NEURA_ARCHITECTURE_SPEC_H
