#pragma once

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace neura {

namespace attr {

// Attribute Keys
constexpr llvm::StringLiteral kDataflowMode = "dataflow_mode";
constexpr llvm::StringLiteral kMappingMode = "mapping_mode";
constexpr llvm::StringLiteral kMappingStrategy = "mapping_strategy";
constexpr llvm::StringLiteral kBacktrackConfig = "backtrack_config";
constexpr llvm::StringLiteral kDumpMappingTable = "dump_mapping_table";

// Identification & Results
constexpr llvm::StringLiteral kDfgId = "dfg_id";
constexpr llvm::StringLiteral kMappingInfo = "mapping_info";
constexpr llvm::StringLiteral kXTiles = "x_tiles";
constexpr llvm::StringLiteral kYTiles = "y_tiles";
constexpr llvm::StringLiteral kCompiledII = "compiled_ii";
constexpr llvm::StringLiteral kRecMII = "rec_mii";
constexpr llvm::StringLiteral kResMII = "res_mii";

// Attribute Values & Constants
namespace val {
// Strategy & Mode
constexpr llvm::StringLiteral kSpatialOnly = "spatial-only";
constexpr llvm::StringLiteral kSpatialTemporal = "spatial-temporal";
constexpr llvm::StringLiteral kTemporal = "temporal";
constexpr llvm::StringLiteral kHeuristic = "heuristic";
constexpr llvm::StringLiteral kCustomized = "customized";
constexpr llvm::StringLiteral kSimple = "simple";
constexpr llvm::StringLiteral kGreedy = "greedy";
constexpr llvm::StringLiteral kExhaustive = "exhaustive";

// Identifiers
constexpr llvm::StringLiteral kModeSteering = "steering";
constexpr llvm::StringLiteral kModePredicate = "predicate";

// Operation Logic
constexpr llvm::StringLiteral kOpFused = "fused_op";
constexpr llvm::StringLiteral kNeuraFusedOp = "neura.fused_op";

} // namespace val

} // namespace attr
} // namespace neura
} // namespace mlir