#ifndef NEURA_PARSER_UTILS_H
#define NEURA_PARSER_UTILS_H

#include "NeuraDialect/Architecture/ArchitectureSpec.h"
#include "NeuraDialect/Util/NeuraYamlKeys.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace neura {
namespace util {
bool parseYamlScalarInt(const llvm::yaml::Node *node, int &result);
bool parseYamlScalarString(const llvm::yaml::Node *node, std::string &result);
void parseYamlStringSequence(llvm::yaml::Node *node,
                             std::vector<std::string> &result);

bool yamlParseError(const std::string &msg, const std::string &file = "");

void parseTileDefaults(llvm::yaml::MappingNode *tile_defaults_map,
                       mlir::neura::TileDefaults &tile_defaults);
void parseTileOverrideOperations(llvm::yaml::MappingNode *override_map,
                                 mlir::neura::TileOverride &override);
void parseSingleTileOverride(llvm::yaml::MappingNode *override_map,
                             mlir::neura::TileOverride &override);
bool parseTileOverrides(llvm::yaml::SequenceNode *tile_overrides_seq,
                        std::vector<mlir::neura::TileOverride> &tile_overrides);
bool parseLinkDefaults(llvm::yaml::MappingNode *link_defaults_map,
                       mlir::neura::LinkDefaults &link_defaults);
void parseSingleLinkOverride(llvm::yaml::MappingNode *override_map,
                             mlir::neura::LinkOverride &override);
bool parseLinkOverrides(llvm::yaml::SequenceNode *link_overrides_seq,
                        std::vector<mlir::neura::LinkOverride> &link_overrides);
mlir::neura::BaseTopology parseTopologyString(const std::string &topology_str);
} // namespace util
} // namespace neura
} // namespace mlir
#endif // NEURA_PARSER_UTILS_H
