#ifndef NEURA_ARCH_PARSER_H
#define NEURA_ARCH_PARSER_H

#include "NeuraDialect/Architecture/Architecture.h"
#include "llvm/Support/YAMLParser.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace neura {
namespace util {
class ArchParser {
public:
  ArchParser(const std::string &architecture_spec_file);
  ~ArchParser() = default;

  mlir::FailureOr<Architecture> getArchitecture();

private:
  std::string architecture_spec_file;
  bool parseArchitectureYaml(
      llvm::yaml::Document &doc, int &multi_cgra_rows, int &multi_cgra_columns,
      mlir::neura::BaseTopology &multi_cgra_base_topology, int &per_cgra_rows,
      int &per_cgra_columns, mlir::neura::BaseTopology &per_cgra_base_topology,
      int &max_ctrl_mem_items, mlir::neura::TileDefaults &tile_defaults,
      std::vector<mlir::neura::TileOverride> &tile_overrides,
      mlir::neura::LinkDefaults &link_defaults,
      std::vector<mlir::neura::LinkOverride> &link_overrides);
};
} // namespace util
} // namespace neura
} // namespace mlir
#endif // NEURA_ARCH_PARSER_H
